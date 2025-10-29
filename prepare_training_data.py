import os
import numpy as np
from sklearn.utils import shuffle

# ------------------------------
# PATHS AND SETTINGS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "keypoints")

X, y = [], []

print(f"📁 Loading keypoints from: {DATA_PATH}")

# ------------------------------
# LOAD ALL SAMPLES PER CLASS
# ------------------------------
for label in sorted(os.listdir(DATA_PATH)):
    label_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_path):
        continue

    files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
    print(f"📦 Class '{label}': {len(files)} samples")

    for file in files:
        keypoints = np.load(os.path.join(label_path, file))
        X.append(keypoints)
        y.append(label)

# ------------------------------
# SHUFFLE AND SAVE
# ------------------------------
X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=42)

np.save(os.path.join(DATA_PATH, "X_keypoints.npy"), X)
np.save(os.path.join(DATA_PATH, "y_labels.npy"), y)

print(f"\n✅ Saved X_keypoints.npy and y_labels.npy with {len(X)} total samples.")
