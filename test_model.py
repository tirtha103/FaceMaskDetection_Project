# test_model.py

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load trained model
model_path = "model/mask_detector_model.h5"
model = load_model(model_path)
print(f"✅ Loaded model from: {model_path}")

# ✅ Correct file name from your folder
test_image_path = "dataset/with_mask/with_mask_1.jpg"

# Load the image
img = cv2.imread(test_image_path)

# Handle file not found
if img is None:
    print(f"❌ Image not found at: {test_image_path}")
    print("👉 Please check the filename and extension (e.g., .jpg or .png)")
    exit()

# Preprocess the image
img_resized = cv2.resize(img, (100, 100))
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

# Predict
prediction = model.predict(img_input)[0][0]

# Display result
label = "No Mask 😷" if prediction > 0.5 else "With Mask 😷"
color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

# Show the image
img_display = cv2.putText(img.copy(), label, (10, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow("Prediction", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

