import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model

# ======== PATHS ========
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_keypoints_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
KEYPOINT_DIR = os.path.join(BASE_DIR, "data", "keypoints")
X_CACHE_PATH = os.path.join(BASE_DIR, "data", "X_test.npy")
Y_CACHE_PATH = os.path.join(BASE_DIR, "data", "y_test.npy")

# ======== LOAD MODEL & ENCODER ========
print("[INFO] Loading model and label encoder...")
model = load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ======== LOAD OR BUILD TEST DATA ========
def load_test_data(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                sample = np.load(os.path.join(label_path, file))
                if sample.shape == (30, 21, 3):
                    sample = sample.reshape(1, -1)
                elif sample.shape == (63,):
                    sample = sample.reshape(1, -1)
                X.append(sample)
                y.append(label)
    return np.vstack(X), np.array(y)

if os.path.exists(X_CACHE_PATH) and os.path.exists(Y_CACHE_PATH):
    print("[INFO] Loading cached test data...")
    X_test = np.load(X_CACHE_PATH)
    y_test = np.load(Y_CACHE_PATH)
else:
    print("[INFO] Building test data from keypoints folder...")
    X_test, y_test = load_test_data(KEYPOINT_DIR)
    np.save(X_CACHE_PATH, X_test)
    np.save(Y_CACHE_PATH, y_test)
    print("[INFO] Test data cached for future runs.")

y_true = label_encoder.transform(y_test)

# ======== PREDICT ========
print("[INFO] Running predictions...")
y_pred_probs = model.predict(X_test, batch_size=128)
y_pred = np.argmax(y_pred_probs, axis=1)

# ======== METRICS ========
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# ======== CONFUSION MATRIX ========
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix for Gesture Classification")
plt.tight_layout()
plt.show()
