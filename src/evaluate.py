import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from preprocess import X_test, y_test, label_encoder  #  Load test data

#  Load the trained model
model = load_model("covid_xray_classifier.h5")
print(" Model loaded successfully!")

#  Predict test images
y_pred = np.argmax(model.predict(X_test), axis=1)

#  Get unique classes present in y_test
unique_classes = np.unique(y_test)
print(f" Unique classes in y_test: {unique_classes}")

#  Generate class labels that exist in y_test
existing_labels = label_encoder.inverse_transform(unique_classes)

#  Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_classes)

#  Plot
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=existing_labels, 
            yticklabels=existing_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#  Print classification report using only existing labels
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=existing_labels))
