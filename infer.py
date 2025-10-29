# src/infer_final.py
"""
Hands2Voice — FINAL stable inference + non-blocking TTS
Usage:
  (hands2voice_env) python src/infer_final.py

Controls:
  Q - quit
  S - speak full sentence immediately
  C - clear sentence

Behavior:
 - majority voting over a rolling window
 - extra stability / cooldown before committing a letter
 - ignores low-confidence predictions
 - commits word on long pause (no hand) or on SPACE label
 - every committed word is spoken via background thread (pyttsx3)
"""
import os
import time
import threading
import pickle
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import tensorflow as tf

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "sign_keypoint_lstm.h5"),
    os.path.join(BASE_DIR, "models", "asl_keypoints_model.h5"),
    os.path.join(BASE_DIR, "models", "hands2voice_model.h5"),
]
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

CAM_INDEX = 0

FRAME_WINDOW = 9            # number of frames used for majority voting
STABILITY_COUNT = 3         # how many consecutive equal majorities needed to commit
CONF_THRESHOLD = 0.78       # minimum average confidence for the majority label
PAUSE_COMMIT_FRAMES = 20    # frames of "no hand" before committing the current word
CHAR_COOLDOWN = 0.9         # seconds between committing characters
SPEAK_COOLDOWN = 0.8        # seconds between speaking words
AUTO_SPEAK_ON_SPACE = True  # automatically speak word when SPACE is committed

# ---------------- Helpers ----------------
def load_model_and_labels():
    model = None
    model_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            model = tf.keras.models.load_model(p)
            model_path = p
            break
    if model is None:
        raise FileNotFoundError(f"No model found in {MODEL_PATHS}. Train model first.")
    # load label encoder if present
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        labels = [str(x) for x in le.classes_]
        print("[INFO] Loaded label encoder with labels:", labels[:10], "...")
    else:
        # fallback
        labels = [chr(i) for i in range(65, 91)] + ["space", "del", "nothing"]
        print("[WARN] label_encoder.pkl not found — using fallback labels A-Z + space/del/nothing")
    return model, labels, model_path

def tts_speak_background(text):
    """Speak text on a separate thread so we don't block the main loop."""
    if not text or not text.strip():
        return
    def _speak(t):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(t)
            engine.runAndWait()
        except Exception as e:
            print("[TTS ERR]", e)
    t = threading.Thread(target=_speak, args=(text,), daemon=True)
    t.start()

def normalize_label_name(lbl):
    """Standardize label names for comparisons."""
    if lbl is None:
        return None
    s = str(lbl).strip().lower()
    if s in ("space", " "):
        return "space"
    if s in ("del", "delete", "backspace"):
        return "del"
    if s in ("nothing", "none"):
        return "nothing"
    # If label is single character letter e.g. 'A' or 'a' -> return uppercase char
    if len(s) == 1 and s.isalpha():
        return s.upper()
    return s  # other values left as-is

# ---------------- Load model + labels ----------------
print("[INFO] Loading model and labels...")
model, LABELS, model_path = load_model_and_labels()
LABELS = [str(x) for x in LABELS]  # ensure strings
LABEL_SET = set([normalize_label_name(l) for l in LABELS])
print("[INFO] Model loaded from:", model_path)
print("[INFO] Number of labels:", len(LABELS))

# ---------------- Mediapipe init ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---------------- State ----------------
recent_preds = deque(maxlen=FRAME_WINDOW)  # stores tuples (label_idx, confidence)
majority_history = deque(maxlen=STABILITY_COUNT)  # stores most-common label indexes
no_hand_frames = 0
last_char_time = 0.0
last_speak_time = 0.0

current_word_chars = []   # characters for the word being typed
sentence_words = []       # committed words (list)

last_committed_label = None

# convenience: determine model input type
model_input_shape = model.input_shape  # e.g., (None,63) or (None,30,63)
is_lstm_model = len(model_input_shape) == 3

print("[INFO] Model input shape:", model_input_shape, "is_lstm_model:", is_lstm_model)

# ---------------- Utility functions ----------------
def extract_keypoints_from_results(results):
    """Return flattened 63-d vector if hand present, else zeros."""
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        arr = []
        for p in lm.landmark:
            arr.extend([p.x, p.y, p.z])
        return np.array(arr, dtype=np.float32)
    return np.zeros(21 * 3, dtype=np.float32)

def majority_and_avg_conf(buffer):
    """Return (best_label_idx, avg_conf) from buffer of (idx, conf)."""
    if not buffer:
        return None, 0.0
    counts = Counter()
    conf_sum = {}
    for idx, conf in buffer:
        counts[idx] += 1
        conf_sum[idx] = conf_sum.get(idx, 0.0) + conf
    best_idx, best_count = counts.most_common(1)[0]
    avg_conf = conf_sum[best_idx] / best_count
    return best_idx, avg_conf

# ---------------- Camera loop ----------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check CAM_INDEX or permissions.")

print("[INFO] Starting inference. Press Q to quit, S to speak, C to clear.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # process frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    kps = extract_keypoints_from_results(results)
    hand_present = not np.allclose(kps, 0.0)

    if hand_present:
        no_hand_frames = 0

        # build model input to match training (your model used flattened 63-d vectors)
        if not is_lstm_model:
            inp = np.expand_dims(kps, axis=0)  # shape (1,63)
        else:
            # If model expects sequences, repeat last frame to create a seq of length T
            T = model_input_shape[1] or FRAME_WINDOW
            inp = np.tile(kps.reshape(1,1,-1), (1, T, 1))  # (1,T,63)

        preds = model.predict(inp, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if preds.ndim == 3:
            preds = preds.mean(axis=1)
        preds = preds.ravel()
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])

        # only append if confidence passes threshold
        if top_conf >= CONF_THRESHOLD:
            recent_preds.append((top_idx, top_conf))
        # else do not append (noise)

        # compute majority and average confidence
        maj_idx, maj_conf = majority_and_avg_conf(recent_preds)
        if maj_idx is not None and maj_conf >= CONF_THRESHOLD:
            majority_history.append(maj_idx)
            # check if there is a stable majority across recent majorities
            if len(majority_history) == STABILITY_COUNT and len(set(majority_history)) == 1:
                label_raw = LABELS[maj_idx]
                label_norm = normalize_label_name(label_raw)

                now = time.time()
                # cooldown to avoid quick duplicate chars
                if label_norm is not None and (now - last_char_time) > CHAR_COOLDOWN:
                    # handle specials
                    if label_norm == "space":
                        # commit current word if present
                        if current_word_chars:
                            word = "".join(current_word_chars)
                            sentence_words.append(word)
                            print("[COMMIT WORD]", word)
                            # speak word in background
                            if AUTO_SPEAK_ON_SPACE:
                                if time.time() - last_speak_time > SPEAK_COOLDOWN:
                                    tts_speak_background(word)
                                    last_speak_time = time.time()
                            current_word_chars = []
                    elif label_norm == "del":
                        if current_word_chars:
                            current_word_chars.pop()
                    elif label_norm == "nothing":
                        pass
                    else:
                        # it's a letter (single char)
                        ch = label_norm if len(label_norm) == 1 and label_norm.isalpha() else (label_raw[0] if label_raw else "")
                        ch = ch.upper()
                        if ch:
                            # avoid immediate duplicate
                            if not current_word_chars or current_word_chars[-1] != ch:
                                current_word_chars.append(ch)
                    last_char_time = now
                    # reset buffers
                    recent_preds.clear()
                    majority_history.clear()
    else:
        no_hand_frames += 1
        # If long pause with no hand and current word exists -> commit it
        if no_hand_frames >= PAUSE_COMMIT_FRAMES and current_word_chars:
            word = "".join(current_word_chars)
            sentence_words.append(word)
            print("[PAUSE COMMIT]", word)
            if time.time() - last_speak_time > SPEAK_COOLDOWN:
                tts_speak_background(word)
                last_speak_time = time.time()
            current_word_chars = []
            recent_preds.clear()
            majority_history.clear()

    # Draw landmarks if present
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # UI: black info bar at bottom
    cv2.rectangle(frame, (0, h-140), (w, h), (0, 0, 0), -1)
    display_word = "".join(current_word_chars)
    display_sentence = " ".join(sentence_words + ([] if not display_word else [""]))
    cv2.putText(frame, f"Word: {display_word}", (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv2.putText(frame, f"Sentence: {display_sentence}", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # top buttons (visual only)
    cv2.rectangle(frame, (10,10), (130,50), (0,180,0), -1); cv2.putText(frame,"Speak (S)", (15,37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
    cv2.rectangle(frame, (150,10), (270,50), (0,0,180), -1); cv2.putText(frame,"Clear (C)", (155,37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    cv2.rectangle(frame, (290,10), (410,50), (120,120,120), -1); cv2.putText(frame,"Exit (Q)", (295,37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    cv2.imshow("Hands2Voice - FINAL", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence_words = []
        current_word_chars = []
        recent_preds.clear()
        majority_history.clear()
        print("[CLEARED]")
    elif key == ord('s'):
        # speak the whole sentence now
        full = " ".join(sentence_words + ([ "".join(current_word_chars) ] if current_word_chars else []))
        if full.strip():
            tts_speak_background(full)
            last_speak_time = time.time()
            print("[SPEAK]", full)

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("[INFO] Exited cleanly.")
