# 😷 Face Mask Detection with Live Alert System

A real-time face mask detection system using Python, OpenCV, TensorFlow/Keras, and Flask.

---

## 🔍 Overview

This project uses a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask using their webcam. It includes:

- Real-time face detection using Haar Cascades
- Mask / No-Mask classification via a trained CNN
- Live video stream rendered in browser using Flask
- On-screen alerts with color-coded boxes

---

## 🚀 Features

✅ Real-time face detection  
✅ CNN-based mask classification  
✅ Live webcam stream via Flask web app  
✅ Visual alerts (Red box = No Mask, Green box = Mask)  
✅ Easy to run on any computer with a webcam  
✅ Modular and clean code structure  

---

## 🛠️ Technologies Used

- Python 3.10
- OpenCV
- TensorFlow / Keras
- Flask
- NumPy, Pandas, Matplotlib
- Haar Cascades (for face detection)

---

## 🧪 How to Run the Project
```bash
git clone https://github.com/tirtha103/FaceMaskDetection_Project.git
cd FaceMaskDetection_Project
python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt
python train_model.py
python mask_detector.py
python app.py
http://127.0.0.1:5000/

