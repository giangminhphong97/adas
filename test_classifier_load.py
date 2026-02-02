import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL='traffic_sign_model.h5'
IMG=os.path.join('dataset','traffic_sign','Test','00004.png')

if not os.path.exists(MODEL):
    print('Model not found:', MODEL); raise SystemExit(1)
if not os.path.exists(IMG):
    print('Image not found:', IMG); raise SystemExit(2)

model = load_model(MODEL)
print('Model loaded')
img = cv2.imread(IMG)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(30,30)).astype('float32')/255.0
img = np.expand_dims(img,0)
preds = model.predict(img)
cls = int(np.argmax(preds))
conf = float(np.max(preds))
print('Predicted', cls, 'confidence', conf)

# try reading Meta.csv
meta_path = os.path.join('dataset','traffic_sign','Meta.csv')
name = f'Class {cls}'
if os.path.exists(meta_path):
    with open(meta_path,'r',encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split(',')
            if len(parts)>=2 and parts[1].isdigit() and int(parts[1])==cls:
                signid = parts[4] if len(parts)>4 else ''
                if signid and signid!='None':
                    name=signid
                break
print('Mapped name:', name)
