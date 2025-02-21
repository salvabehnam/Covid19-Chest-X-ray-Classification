import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from command import df 

#  Define dataset path
dataset_path = r"D:\Python\Tumor Detection\covid-chestxray-dataset"

#  Define image folder path
image_folder = os.path.join(dataset_path, "images")

if not os.path.exists(image_folder):
    print(" Image folder does not exist! Check dataset extraction.")
else:
    print(" Image folder found!")

#  Filter out non-image files (.nii.gz, .dcm)
valid_extensions = (".jpg", ".jpeg", ".png")
df = df[df["filename"].str.endswith(valid_extensions, na=False)]

#  Create dataset lists
data = []
labels = []

#  Select a random subset of images (500) to avoid memory issues
subset_df = df[df["finding"].notna()].sample(500)  

for index, row in subset_df.iterrows():
    img_name = row["filename"]
    label = row["finding"]
    img_path = os.path.join(image_folder, img_name)

    #  Check if the file exists and is an image
    if os.path.exists(img_path) and img_name.lower().endswith(valid_extensions):
        try:
            #  Read and preprocess image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
            if image is None:
                print(f" Corrupted image: {img_name}, skipping...")
                continue

            image = cv2.resize(image, (128, 128))  # Resize
            image = image / 255.0  # Normalize pixel values

            data.append(image)
            labels.append(label)
        except Exception as e:
            print(f" Error processing {img_name}: {e}")
    else:
        print(f" Skipping missing or unsupported image: {img_name}")

#  Convert to NumPy arrays
if len(data) > 0:
    data = np.array(data).reshape(-1, 128, 128, 1)
    labels = np.array(labels)

    print(f" Dataset loaded successfully! Total Valid Images: {len(data)}")

    #  sample image
    plt.imshow(data[0].reshape(128, 128), cmap="gray")
    plt.title(f"Sample Image: {labels[0]}")
    plt.axis("off")
    plt.show()
else:
    print(" No valid images found!")
