"""
Create a test video from traffic sign dataset images.
"""
import os
import cv2
import numpy as np

# Collect images from Train directory
train_dir = os.path.join('dataset', 'traffic_sign', 'Train')
images_list = []

for cls_id in sorted(os.listdir(train_dir)):
    cls_path = os.path.join(train_dir, cls_id)
    if not os.path.isdir(cls_path):
        continue
    for fname in sorted(os.listdir(cls_path))[:3]:  # take 3 per class for speed
        images_list.append(os.path.join(cls_path, fname))

print(f"Collected {len(images_list)} images")

# Create video
output_path = 'traffic_sign_test_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 5.0, (640, 480))

for img_path in images_list:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        continue
    # Resize to 640x480
    img = cv2.resize(img, (640, 480))
    # Repeat frame 10 times (so ~2s per image at 5fps)
    for _ in range(10):
        out.write(img)

out.release()
print(f"Video created: {output_path}")
