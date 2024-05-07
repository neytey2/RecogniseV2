from re import T
from hmac import new
import os
import time
import uuid
import albumentations as alb
import enum
import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
facetracker = load_model('facetracker.h5')

# Real Time Detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:1000, 50:1000, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, [120, 120])

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]
    confidence = yhat[0]
    if confidence > 0.5:
        # Extracting coordinates
        x1, y1, x2, y2 = sample_coords
        # Convert to pixel coordinates
        x1 *= frame.shape[1]
        y1 *= frame.shape[0]
        x2 *= frame.shape[1]
        y2 *= frame.shape[0]
        # Calculate center and dimensions
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        width = int(x2 - x1)
        height = int(y2 - y1)
        # Draw rectangle around the face
        cv2.rectangle(frame,
                      (int(center_x - width / 2), int(center_y - height / 2)),
                      (int(center_x + width / 2), int(center_y + height / 2)),
                      (255, 0, 0), 2)
        cv2.rectangle(frame,
                      (int(center_x - width / 2), int(center_y - height / 2 - 30)),
                      (int(center_x - width / 2 + 80), int(center_y - height / 2)),
                      (255, 0, 0), -1)
        cv2.putText(frame, 'Mensch', (int(center_x - width / 2), int(center_y - height / 2 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FaceTracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
