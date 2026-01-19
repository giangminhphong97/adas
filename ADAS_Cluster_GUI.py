import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from ADAS_Lane_Detection import pipeline

# ===============================
# Load YOLO model (object detection)
# ===============================
yolo_model = YOLO("yolov8n.pt")  # lightweight & fast

# ===============================
# Load video
# ===============================
cap = cv2.VideoCapture("52430-468806577_small.mp4")
if not cap.isOpened():
    print("Error: Could not open video")
    exit(1)

# ===============================
# GUI Setup
# ===============================
root = tk.Tk()
root.title("ADAS Cluster GUI")
root.geometry("1920x1080")

# Top frame for video and lane detection
top_frame = tk.Frame(root)
top_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Video display label
video_label = tk.Label(top_frame)
video_label.pack()

# Lane status label
lane_status_label = tk.Label(top_frame, text="Initializing...", font=("Arial", 12))
lane_status_label.pack()

# Right frame for traffic sign detection
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Traffic sign title
sign_title = tk.Label(right_frame, text="Traffic Sign Detection", font=("Arial", 14, "bold"))
sign_title.pack()

# Traffic sign image display
sign_image_label = tk.Label(right_frame, width=150, height=150, bg="gray")
sign_image_label.pack(pady=10)

# Traffic sign text display
sign_text_label = tk.Label(right_frame, text="No sign detected", font=("Arial", 12), wraplength=150)
sign_text_label.pack(pady=10)

# Confidence label
confidence_label = tk.Label(right_frame, text="Confidence: 0%", font=("Arial", 10))
confidence_label.pack()

# ===============================
# YOLO Traffic Sign Detection
# ===============================
def detect_traffic_sign(frame):
    """
    YOLO-based traffic sign detection
    Returns: (label, confidence, cropped_image)
    """

    results = yolo_model(frame, imgsz=640, conf=0.25, verbose=False)

    best_conf = 0
    best_crop = None
    best_label = None

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO: traffic light = 9, stop sign = 11
            if cls_id in [9, 11] and conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                best_conf = conf
                best_crop = crop
                best_label = "Traffic Sign"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if best_label:
        return best_label, best_conf, best_crop

    return None, 0, None

# ===============================
# Main update loop
# ===============================
def update_frame():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    if ret:
        # Lane detection
        processed, lane_center = pipeline(frame)

        # Traffic sign detection
        detected_sign, confidence, sign_image = detect_traffic_sign(processed)

        # Display video
        img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Lane status
        if lane_center is not None:
            width = frame.shape[1]
            on_track = abs(lane_center - width / 2) < 100
            lane_status_label.config(
                text="On Track" if on_track else "Off Track",
                fg="green" if on_track else "red"
            )
        else:
            lane_status_label.config(text="Lane Not Detected", fg="orange")

        # Traffic sign GUI
        if detected_sign and sign_image is not None:
            sign_img = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB)
            sign_img = Image.fromarray(sign_img).resize((150, 150))
            sign_imgtk = ImageTk.PhotoImage(image=sign_img)

            sign_image_label.imgtk = sign_imgtk
            sign_image_label.configure(image=sign_imgtk, bg="white")
            sign_text_label.config(text=detected_sign)
            confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
        else:
            sign_image_label.configure(image="", bg="gray")
            sign_text_label.config(text="No sign detected")
            confidence_label.config(text="Confidence: 0%")

        root.after(30, update_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        root.after(30, update_frame)

# ===============================
# Start GUI
# ===============================
update_frame()
root.mainloop()
cap.release()
