"""
Train YOLOv8 on traffic sign dataset.
"""
import os
from ultralytics import YOLO

# Make sure dataset is prepared
if not os.path.exists("dataset_yolo/data.yaml"):
    print("Preparing dataset...")
    os.system("python prepare_yolo_dataset.py")

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on traffic signs (CPU mode, smaller batch/epochs)
results = model.train(
    data="dataset_yolo/data.yaml",
    epochs=15,
    imgsz=416,
    batch=16,
    patience=3,
    device="cpu",
    name="traffic_sign_yolo"
)

# Save best model
best_model_path = "runs/detect/traffic_sign_yolo/weights/best.pt"
if os.path.exists(best_model_path):
    import shutil
    shutil.copy(best_model_path, "traffic_sign_yolo.pt")
    print("Saved trained model as traffic_sign_yolo.pt")
else:
    print("Warning: best model not found at expected path")
