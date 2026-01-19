import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
np.object = object
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

data = []
labels = []
classes = 43
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path, 'dataset','traffic_sign', 'Train', str(i))
    images = os.listdir(path)
    
    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {a}: {e}")
            continue

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 15
history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data=(X_test, y_test))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy on the first subplot
ax1.plot(history.history['accuracy'], label='training accuracy')
ax1.plot(history.history['val_accuracy'], label='validation accuracy')
ax1.set_title('Accuracy')
ax1.set_xlabel('epochs')
ax1.set_ylabel('accuracy')
ax1.legend()

# Plot loss on the second subplot
ax2.plot(history.history['loss'], label='training loss')
ax2.plot(history.history['val_loss'], label='validation loss')
ax2.set_title('Loss')
ax2.set_xlabel('epochs')
ax2.set_ylabel('loss')
ax2.legend()

model.save('traffic_sign_model.h5')
print("Model saved as traffic_sign_model.h5")

plt.tight_layout()
plt.show()

# Testing accuracy on test dataset
test_data = pd.read_csv(os.path.join(cur_path, 'dataset', 'traffic_sign', 'Test.csv'))
labels = test_data["ClassId"].values
imgs = test_data["Path"].values

test_images = []
valid_labels = []

for idx, img in enumerate(imgs):
    try:
        image = Image.open(os.path.join(cur_path, 'dataset', 'traffic_sign', img))
        image = image.resize((30, 30))
        test_images.append(np.array(image))
        valid_labels.append(labels[idx])
    except Exception as e:
        print(f"Skipping corrupted test image {img}: {e}")
        continue

X_test = np.array(test_images)
valid_labels = np.array(valid_labels)

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
print("Test Accuracy:", accuracy_score(valid_labels, pred_classes))