import os
import cv2
import numpy as np
from ultralytics import YOLO

# load classifier if present
classifier = None
try:
    from tensorflow.keras.models import load_model
    if os.path.exists('traffic_sign_model.h5'):
        classifier = load_model('traffic_sign_model.h5')
except Exception as e:
    classifier = None

# load yolo if present
yolo = None
if os.path.exists('yolov8n.pt'):
    try:
        yolo = YOLO('yolov8n.pt')
    except Exception as e:
        print('YOLO load failed:', e)

# helper detection (simplified)
def detect_on_image(img):
    h,w = img.shape[:2]
    # try yolo
    if yolo is not None:
        try:
            res = yolo(img, imgsz=640, conf=0.25, verbose=False)
            for r in res:
                boxes = getattr(r,'boxes',None)
                if boxes is None:
                    continue
                for box in boxes:
                    try:
                        cls = int(box.cls[0]) if hasattr(box.cls,'__len__') else int(box.cls)
                        conf = float(box.conf[0]) if hasattr(box.conf,'__len__') else float(box.conf)
                        if cls in [9,11] and conf>0.25:
                            xy = box.xyxy[0]
                            if hasattr(xy,'cpu'):
                                xy = xy.cpu().numpy()
                            x1,y1,x2,y2 = map(int,xy)
                            x1,y1 = max(0,x1), max(0,y1)
                            x2,y2 = min(w-1,x2), min(h-1,y2)
                            return ('yolo', f'class_{cls}', conf, img[y1:y2,x1:x2])
                    except Exception:
                        continue
        except Exception as e:
            print('YOLO inference error', e)
    # classifier fallback via color segmentation
    if classifier is None:
        return (None, 0, None, None)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0,70,50]); upper_red1=np.array([10,255,255])
    lower_red2 = np.array([170,70,50]); upper_red2=np.array([180,255,255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    lower_blue=np.array([90,60,50]); upper_blue=np.array([140,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = mask_red | mask_blue
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_conf=0; best=None
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        if wc<15 or hc<15:
            continue
        pad=int(0.15*max(wc,hc))
        x1=max(0,x-pad); y1=max(0,y-pad); x2=min(img.shape[1], x+wc+pad); y2=min(img.shape[0], y+hc+pad)
        crop=img[y1:y2,x1:x2]
        try:
            im = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im,(30,30)).astype('float32')/255.0
            im = np.expand_dims(im,0)
            preds = classifier.predict(im)
            p = float(np.max(preds)); cidx=int(np.argmax(preds))
            if p>best_conf:
                best_conf=p; best=('clf', f'class_{cidx}', p, crop)
        except Exception:
            continue
    if best:
        return best
    return (None, 0, None, None)

# select sample images from dataset
folder = os.path.join('dataset','traffic_sign','Test')
if not os.path.isdir(folder):
    print('Test folder not found:', folder); exit(2)

imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])[:20]

for fn in imgs:
    path = os.path.join(folder, fn)
    img = cv2.imread(path)
    if img is None:
        print('failed read', path); continue
    det = detect_on_image(img)
    print(fn, '->', det[0], det[1], round(det[2],3) if det[2] else None)

print('Done')
