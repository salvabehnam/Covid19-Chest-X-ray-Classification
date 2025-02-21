import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import label_encoder  

#  Load the trained model
model_path = "covid_xray_classifier.keras"
if not os.path.exists(model_path):
    print(" Keras model not found! Falling back to HDF5 format.")
    model_path = "covid_xray_classifier.h5"

#  Load the model
model = load_model(model_path)
print(f" Model loaded successfully from {model_path}")

#  Define path to an X-ray image 
new_image_path = r"D:\Python\Tumor Detection\test_images\example_xray.png"

#  Check if the file exists
if not os.path.exists(new_image_path):
    print(f" Image file not found: {new_image_path}")
    exit()

#  Load and preprocess the image
image = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)  # grayscale
image = cv2.resize(image, (128, 128))  # Resize to 128x128
image = image / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = np.expand_dims(image, axis=-1)  # Add channel dimension (1 for grayscale)

#  predict
prediction = model.predict(image)
predicted_label = label_encoder.classes_[np.argmax(prediction)]

#  Display the image with prediction
plt.imshow(cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
plt.title(f"Predicted Class: {predicted_label}")
plt.axis("off")
plt.show()

print(f"🔹 Model Prediction: {predicted_label}")
