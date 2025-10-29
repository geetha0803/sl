# src/train_model.py
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "keypoints")

X = np.load(os.path.join(DATA_PATH, "X_keypoints.npy"))
y = np.load(os.path.join(DATA_PATH, "y_labels.npy"))

print("Total samples loaded:", len(X))
print("Unique classes:", len(np.unique(y)))

# === Limit samples per class (to 300 for fast training) ===
def limit_samples(X, y, limit=87000):
    limited_X, limited_y = [], []
    class_counts = Counter(y)
    temp_counts = {cls: 0 for cls in class_counts}

    for i in range(len(y)):
        label = y[i]
        if temp_counts[label] < limit:
            limited_X.append(X[i])
            limited_y.append(y[i])
            temp_counts[label] += 1
    return np.array(limited_X), np.array(limited_y)

X, y = limit_samples(X, y, limit=800)
print("Samples used for training after limiting:", len(X))

# === Encode labels ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# === Model ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train ===
#model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)


# === Evaluate ===
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.3f}")

# === Save model ===
model.save(os.path.join(BASE_DIR, "models", "asl_keypoints_model.h5"))
print("✅ Model saved successfully!")
