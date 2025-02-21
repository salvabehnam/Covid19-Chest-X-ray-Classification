from load_data import labels, data  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#  Convert text labels to numbers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

from sklearn.utils import shuffle

#  Shuffle dataset before splitting
data, labels = shuffle(data, labels, random_state=42)

#  Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

print(f" Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")
print("Class Labels:", label_encoder.classes_)  # Shows mapping of labels to numbers
