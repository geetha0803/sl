import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import argparse

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image):
    """Extracts hand keypoints using Mediapipe Hands."""
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return np.zeros(63)  # 21 points × 3 coordinates
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)

def process_dataset(dataset_dir, output_dir):
    """Process all images in the dataset and save keypoints."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    labels = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"Detected classes: {labels}")

    for label in tqdm(labels, desc="Processing classes"):
        class_dir = os.path.join(dataset_dir, label)
        save_path = os.path.join(output_dir, label)
        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            keypoints = extract_keypoints(image)
            np.save(os.path.join(save_path, img_name.replace(".jpg", ".npy")), keypoints)

if __name__ == "__main__":
    # Auto-detect dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asl_path = os.path.join(base_dir, "datasets", "asl_alphabet", "asl_alphabet_train", "asl_alphabet_train")
    sign_mnist_path = os.path.join(base_dir, "datasets", "sign_mnist")

    # Default: use ASL dataset if present
    dataset_to_use = asl_path if os.path.exists(asl_path) else sign_mnist_path
    output_dir = os.path.join(base_dir, "data", "keypoints")

    print(f"Using dataset: {dataset_to_use}")
    process_dataset(dataset_to_use, output_dir)
    print(f"\n✅ Keypoints saved under: {output_dir}")
