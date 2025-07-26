# train_model.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------------------
# STEP 1: Load & Preprocess Data
# ---------------------------

with_mask_path = "dataset/with_mask"
without_mask_path = "dataset/without_mask"

data = []
labels = []

img_size = 100  # Resize all images to 100x100

# Load with_mask images
for img in os.listdir(with_mask_path):
    img_path = os.path.join(with_mask_path, img)
    try:
        img_array = cv2.imread(img_path)
        resized = cv2.resize(img_array, (img_size, img_size))
        data.append(resized)
        labels.append(0)
    except Exception as e:
        print("Error reading image:", img_path)

# Load without_mask images
for img in os.listdir(without_mask_path):
    img_path = os.path.join(without_mask_path, img)
    try:
        img_array = cv2.imread(img_path)
        resized = cv2.resize(img_array, (img_size, img_size))
        data.append(resized)
        labels.append(1)
    except Exception as e:
        print("Error reading image:", img_path)

# Convert to NumPy arrays
X = np.array(data) / 255.0  # Normalize
y = np.array(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Preview a sample image
plt.imshow(X_train[0])
plt.title(f"Label: {'No Mask' if y_train[0]==1 else 'With Mask'}")
plt.show()

# ---------------------------
# STEP 2: Build CNN Model
# ---------------------------

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # 1 neuron for binary classification

# ---------------------------
# STEP 3: Compile & Train
# ---------------------------

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)

# ---------------------------
# STEP 4: Evaluate & Save
# ---------------------------

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Create model/ folder if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/mask_detector_model.h5")
print("✅ Model saved to model/mask_detector_model.h5")
