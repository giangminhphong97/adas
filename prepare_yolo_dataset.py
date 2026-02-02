"""
Prepare YOLO format dataset from traffic sign folder structure.
Expected structure: dataset/traffic_sign/Train/[0-42]/image.ppm
Creates: dataset_yolo/images/{train,val}/... and dataset_yolo/labels/{train,val}/...
"""
import os
import shutil
import csv
import random
import cv2
import numpy as np
from pathlib import Path

# Config
DATA_ROOT = "dataset/traffic_sign"
YOLO_ROOT = "dataset_yolo"
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Create directories
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(YOLO_ROOT, folder), exist_ok=True)

# Collect all class folders
train_dir = os.path.join(DATA_ROOT, "Train")
class_ids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

total_files = 0
for cls_id in class_ids:
    cls_folder = os.path.join(train_dir, cls_id)
    files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.ppm', '.png', '.jpg'))]
    
    # Randomly split into train/val
    random.shuffle(files)
    train_files = files[:int(len(files) * TRAIN_SPLIT)]
    val_files = files[int(len(files) * TRAIN_SPLIT):]
    
    for split, split_files in [("train", train_files), ("val", val_files)]:
        for fname in split_files:
            src = os.path.join(cls_folder, fname)
            dst_img = os.path.join(YOLO_ROOT, f"images/{split}", f"{cls_id}_{fname}.jpg")
            
            # Convert to jpg and save
            img = cv2.imread(src)
            if img is None:
                print(f"Warning: could not read {src}")
                continue
            cv2.imwrite(dst_img, img)
            
            # Create YOLO label file (one object per image, full image bbox)
            h, w = img.shape[:2]
            label_file = dst_img.replace("images", "labels").replace(".jpg", ".txt")
            # YOLO format: class_id center_x center_y width height (normalized 0-1)
            with open(label_file, 'w') as f:
                f.write(f"{int(cls_id)} 0.5 0.5 1.0 1.0\n")
            
            total_files += 1
            if total_files % 500 == 0:
                print(f"Processed {total_files} images...")

print(f"Total files prepared: {total_files}")

# Create data.yaml for YOLO
class_names = {int(c): f"class_{c}" for c in class_ids}
yaml_content = f"""path: {os.path.abspath(YOLO_ROOT)}
train: images/train
val: images/val
nc: {len(class_ids)}
names: {class_names}
"""

with open(os.path.join(YOLO_ROOT, "data.yaml"), 'w') as f:
    f.write(yaml_content)

print("YOLO dataset created at:", YOLO_ROOT)
print("data.yaml written")
