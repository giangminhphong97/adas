import os
import cv2
from ultralytics import YOLO

MODEL_PATH = 'yolov8n.pt'
IMG = os.path.join('dataset','traffic_sign','Test','00004.png')

model = YOLO(MODEL_PATH)
img = cv2.imread(IMG)
results = model(img, imgsz=640, conf=0.1)
for i, r in enumerate(results):
    print(f'Result {i}:')
    boxes = getattr(r, 'boxes', None)
    if boxes is None:
        print('  no boxes')
        continue
    for j, box in enumerate(boxes):
        try:
            cls = box.cls[0] if hasattr(box.cls, '__len__') else box.cls
            conf = box.conf[0] if hasattr(box.conf, '__len__') else box.conf
            xy = box.xyxy[0] if hasattr(box.xyxy, '__len__') else box.xyxy
            print(f'  Box {j}: cls={int(cls)}, conf={float(conf):.3f}, xy={xy}')
        except Exception as e:
            print('  error reading box', e)
