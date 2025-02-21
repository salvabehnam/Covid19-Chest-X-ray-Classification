from preprocess import X_train, X_test, y_train, y_test, label_encoder  #  Import processed data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#  Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")  # Multi-class classification
])

#  Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#  Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#  Evaluate model performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f" Test Accuracy: {test_acc * 100:.2f}%")

#  Save in new Keras format 
model.save("models/covid_xray_classifier.keras")

#  Save in legacy HDF5 format 
model.save("models/covid_xray_classifier.h5")

print(" Model saved as covid_xray_classifier.h5")
