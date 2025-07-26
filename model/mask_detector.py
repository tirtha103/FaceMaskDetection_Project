# mask_detector.py

import cv2
import numpy as np
import winsound  # For beep alert (Windows only)
from tensorflow.keras.models import load_model

# Load model and Haar cascade
model = load_model("model/mask_detector_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (100, 100))
        normalized = resized / 255.0
        reshaped = np.expand_dims(normalized, axis=0)

        prediction = model.predict(reshaped)[0][0]

        if prediction > 0.5:
            label = "No Mask 😷"
            color = (0, 0, 255)  # Red

            # ✅ Play beep sound for alert
            winsound.Beep(1000, 200)  # freq=1000Hz, duration=200ms

            # ✅ Show on-screen alert
            cv2.putText(frame, "ALERT! No Mask Detected!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        else:
            label = "With Mask 😷"
            color = (0, 255, 0)  # Green

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Mask Detection with Alert", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

