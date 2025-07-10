import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Class ID to Class Name (from CR7-DET dataset)
class_map = {
    '0': 'inclusion',
    '1': 'dent',
    '2': 'oil_spot',
    '3': 'pit',
    '4': 'punching',
    '5': 'linear',
    '6': 'macular'
}

# Load CR7-DET dataset
def load_cr7_dataset(image_dir, label_dir, image_size=(128, 128)):
    X, y = [], []

    for img_name in os.listdir(image_dir):
        if not img_name.endswith(".jpg"):
            continue

        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = first_line.split()[0]
                class_name = class_map.get(class_id)

                if class_name:
                    img_path = os.path.join(image_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        X.append(img)
                        y.append(class_name)

    return np.array(X), np.array(y)

# Load data
print("🔄 Loading dataset...")
X, y = load_cr7_dataset("CR7-DET/image", "CR7-DET/label")

# Preprocess
print("✅ Preprocessing...")
X = X / 255.0
X = X.reshape(-1, 128, 128, 1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

np.save("classes.npy", le.classes_)  # Save class labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Model definition
print("🚀 Training model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("steel_defect_model.h5")
print("✅ Model and classes saved!")
