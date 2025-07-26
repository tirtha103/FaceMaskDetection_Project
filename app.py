# app.py

# app.py

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and Haar cascade
model = load_model("model/mask_detector_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Video generator
def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized = cv2.resize(face, (100, 100))
            normalized = resized / 255.0
            reshaped = np.expand_dims(normalized, axis=0)

            prediction = model.predict(reshaped)[0][0]
            label = "No Mask 😷" if prediction > 0.5 else "With Mask 😷"
            color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')  # make sure index.html is inside templates/


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)