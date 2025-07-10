import os
import cv2
import numpy as np

# 📌 Class ID to Name Mapping (from CR7-DET paper)
class_map = {
    '0': 'inclusion',
    '1': 'dent',
    '2': 'oil_spot',
    '3': 'pit',
    '4': 'punching',
    '5': 'linear',
    '6': 'macular'
}

def load_cr7_dataset("CR7-DET/image", "CR7-DET/label", image_size=(128, 128)):
    X = []
    y = []

    # Loop over each image in the image directory
    for img_name in os.listdir(CR7-DET/image):
        if not img_name.endswith(".jpg"):
            continue

        # Get corresponding label file
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as label_file:
            label_line = label_file.readline().strip()
            if not label_line:
                continue

            class_id = label_line.split()[0]
            class_name = class_map.get(class_id, None)

            if class_name is None:
                continue

            # Load and preprocess the image
            img_path = os.path.join(image_dir, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, image_size)
            X.append(image)
            y.append(class_name)

    return np.array(X), np.array(y)
