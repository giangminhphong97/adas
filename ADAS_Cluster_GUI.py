import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import sys
import traceback
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Safe import of YOLO and lane pipeline
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from ADAS_Lane_Detection import pipeline
except Exception:
    def pipeline(frame):
        # fallback: return original frame and no lane center
        return frame, None

# Try to load traffic-sign classifier (trained on small 30x30 images)
classifier_model = None
classifier_names = {}
try:
    from tensorflow.keras.models import load_model
    if os.path.exists('traffic_sign_model.h5'):
        try:
            classifier_model = load_model('traffic_sign_model.h5')
            print('Loaded traffic_sign_model.h5')
        except Exception as e:
            print('Failed to load traffic_sign_model.h5:', e)
            classifier_model = None
    else:
        classifier_model = None
except Exception:
    classifier_model = None

# Try to load class name mappings from Meta.csv (optional)
meta_path = os.path.join('dataset', 'traffic_sign', 'Meta.csv')
if os.path.exists(meta_path):
    try:
        with open(meta_path, 'r', encoding='utf-8') as mf:
            # CSV format: Path,ClassId,ShapeId,ColorId,SignId
            for line in mf:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1].isdigit():
                    cid = int(parts[1])
                    signid = parts[4] if len(parts) > 4 else ''
                    name = f"{signid}" if signid and signid != 'None' else f"Class {cid}"
                    classifier_names[cid] = name
    except Exception:
        classifier_names = {}
else:
    classifier_names = {i: f"Class {i}" for i in range(43)}

# ===============================
# Load YOLO model (object detection)
# ===============================
yolo_model = None
if YOLO is not None:
    # Try trained traffic-sign model first, fallback to COCO
    model_path = "traffic_sign_yolo.pt" if os.path.exists("traffic_sign_yolo.pt") else "yolov8n.pt"
    try:
        yolo_model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")
    except Exception as e:
        print(f"Warning: Could not load YOLO model '{model_path}': {e}. Continuing without detection.")
        yolo_model = None

# ===============================
# Load video (supports --video or --webcam fallback)
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--video', default='traffic sign test video.mp4', help='Path to video file')
parser.add_argument('--webcam', action='store_true', help='Use webcam instead of video file')
args, _ = parser.parse_known_args()

video_source = 0 if args.webcam else args.video
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    # try fallback to webcam if a file was requested
    if not args.webcam:
        print(f"Warning: Could not open video '{args.video}'. Trying webcam...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source (file or webcam). Exiting.")
        sys.exit(1)

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
    # First try YOLO if available
    results = None
    if yolo_model is not None:
        try:
            results = yolo_model(frame, imgsz=640, conf=0.25, verbose=False)
        except Exception:
            traceback.print_exc()
            results = None

    best_conf = 0.0
    best_crop = None
    best_label = None

    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue

        # boxes may be iterable of Box objects
        for box in boxes:
            try:
                # defensively access properties (handle different ultralytics versions)
                cls_val = None
                conf = 0.0
                xyxy = None

                if hasattr(box, 'cls'):
                    cls_arr = box.cls
                    cls_val = int(cls_arr[0]) if len(cls_arr) > 0 else int(cls_arr)
                elif hasattr(box, 'id'):
                    cls_val = int(box.id)

                if hasattr(box, 'conf'):
                    conf_arr = box.conf
                    conf = float(conf_arr[0]) if len(conf_arr) > 0 else float(conf_arr)

                if hasattr(box, 'xyxy'):
                    xyxy = box.xyxy[0] if len(box.xyxy) > 0 else box.xyxy
                elif hasattr(box, 'xyxyn'):
                    xyxy = box.xyxyn[0]

                if cls_val is None or xyxy is None:
                    continue

                # COCO: traffic light = 9, stop sign = 11
                if cls_val in [9, 11] and conf > best_conf:
                    x1, y1, x2, y2 = map(int, xyxy)
                    # clamp coords
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    best_conf = conf
                    best_crop = crop
                    best_label = "Traffic Sign"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception:
                continue

    if best_label:
        return best_label, best_conf, best_crop

    # If YOLO didn't find any traffic sign, try classifier-based region proposals
    if classifier_model is None:
        return None, 0, None

    # Improved region proposals: combine color mask + sliding window + edge detection
    best_conf_c = 0.0
    best_crop_c = None
    best_label_c = None

    # Strategy 1: Color-based segmentation (red/blue/yellow signs)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = []
    # red (two ranges)
    mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    masks.append(mask_r1 | mask_r2)
    # blue
    masks.append(cv2.inRange(hsv, np.array([90, 60, 50]), np.array([140, 255, 255])))
    # yellow
    masks.append(cv2.inRange(hsv, np.array([15, 60, 50]), np.array([30, 255, 255])))

    for mask in masks:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 or h < 15:
                continue
            pad = int(0.15 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (30, 30)).astype('float32') / 255.0
                img = np.expand_dims(img, 0)
                preds = classifier_model.predict(img, verbose=0)
                conf_c = float(np.max(preds))
                if conf_c > best_conf_c and conf_c > 0.3:
                    best_conf_c = conf_c
                    best_crop_c = crop
                    cls_idx = int(np.argmax(preds))
                    label_name = classifier_names.get(cls_idx, f"Class {cls_idx}")
                    best_label_c = label_name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            except Exception:
                continue

    # Strategy 2: Sliding window on high-gradient regions (DISABLED for performance)
    # Edge detection + sliding window can be very slow on video frames; enable only if needed
    # Uncomment below to enable:
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150)
    # ... (sliding window code)

    if best_label_c:
        return best_label_c, best_conf_c, best_crop_c

    return None, 0, None

# ===============================
# Main update loop
# ===============================
def update_frame():
    ret, frame = cap.read()
    if not ret or frame is None:
        # restart or wait
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        root.after(30, update_frame)
        return

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
