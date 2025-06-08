import os
import random
import pandas as pd
import cv2
import numpy as np
from config import IMG_SIZE, IMG_EXTENSIONS

def build_labeled_dataframe(pos_dir, neg_dir):
    data = []
    for image in os.listdir(pos_dir):
        if not is_image_file(image): continue
        image_url = os.path.join(pos_dir, image)
        data.append({"image": image_url, "label": 1})
    for image in os.listdir(neg_dir):
        if not is_image_file(image): continue
        image_path = os.path.join(neg_dir, image)
        data.append({"image": image_path, "label": 0})
    random.shuffle(data)
    return pd.DataFrame(data)

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def load_images_and_labels(df):
    images, labels = [], []
    for _, row in df.iterrows():
        image = cv2.imread(row['image'])
        if image is None:
            print(f"⚠️ Skipping unreadable image: {row['image']}")
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        labels.append(row['label'])
    images = np.array(images).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    return images, labels

def load_dataset(pos_dir, neg_dir):
    df = build_labeled_dataframe(pos_dir, neg_dir)
    return load_images_and_labels(df)
