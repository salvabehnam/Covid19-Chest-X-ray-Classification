# 🩺 COVID-19 Chest X-ray Classification using CNN

This project is a **deep learning-based classifier** for detecting **COVID-19 and other pneumonia types** from chest X-ray images. The model is trained using **Convolutional Neural Networks (CNNs)** on a publicly available X-ray dataset.
 
## 📌 Features

- **Dataset Handling**: Loads and preprocesses chest X-ray images.
- **CNN Model Training**: Uses TensorFlow/Keras to train a deep learning model.
- **Evaluation & Metrics**: Computes accuracy, precision, recall, and confusion matrix.
- **Prediction**: Classifies new chest X-ray images based on the trained model.
- **Model Generation**: Users must train the model themselves as it is not included in this repository.

## 📂 Project Structure

    covid_xray_project/
    ├── dataset/
    │   ├── covid-chestxray-dataset/    # Chest X-ray dataset (must be downloaded separately)
    │       ├── images/                 # X-ray images
    │       ├── metadata.csv             # Labels & patient info
    ├── models/                          # Folder for trained models (Generated after training)
    ├── test_images/                     # New X-ray images for testing
    ├── src/
    │   ├── load_data.py                 # Loads and preprocesses the dataset
    │   ├── preprocess.py                # Converts labels & splits data
    │   ├── train.py                     # Trains the CNN model (Generates model file)
    │   ├── evaluate.py                   # Evaluates model performance
    │   ├── predict.py                    # Uses the model to classify new images
    ├── requirements.txt                   # List of required Python packages
    ├── README.md                          # Project documentation

## 🚀 Installation & Usage

1. **Clone the Repository**

       git clone https://github.com/salvabehnam/covid_xray_project.git
       cd covid_xray_project

2. **Install Dependencies**

   Ensure you have **Python 3.x** installed, then install the required packages:

       pip install -r requirements.txt

3. **Download the Chest X-ray Dataset**

   The dataset is **not included in the repository** due to size limitations.  
   You must manually **download the dataset** from the official source:

   🔗 **[Download COVID-19 Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)**

   After downloading:
   - Extract the dataset into the `dataset/` folder so the structure looks like:

         covid_xray_project/
         ├── dataset/
         │   ├── covid-chestxray-dataset/
         │       ├── images/          # X-ray images
         │       ├── metadata.csv      # Labels & patient info

4. **Load and Preprocess the Dataset**

   Run the following command to **load and preprocess the dataset**:

       python src/load_data.py

   ✅ This will:
   - Load **X-ray images** from `dataset/covid-chestxray-dataset/images/`
   - Convert images to **grayscale, resize, and normalize them**.
   - Convert labels to **numerical values**.

5. **Train the CNN Model**

   Train the CNN model using:

       python src/train.py

   ✅ This will:
   - Train the model using **Convolutional Neural Networks**.
   - Save the trained model in the **models/** directory.

6. **Evaluate the Model**

   Check model accuracy and generate a confusion matrix:

       python src/evaluate.py

   ✅ This will:
   - Compute **accuracy, precision, recall, F1-score**.
   - Display a **confusion matrix** to visualize classification performance.

7. **Make Predictions on New Images**

   To classify a **new chest X-ray image**, place it inside `test_images/` and run:

       python src/predict.py

   ✅ This will:
   - Load the trained model.
   - Preprocess the test image.
   - Predict the disease type (**COVID-19, Pneumonia, or Normal**).
   - Display the image with the **predicted label**.

## ✨ Technologies Used

- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep learning framework for training CNN models.
- **OpenCV**: Image preprocessing and manipulation.
- **Matplotlib & Seaborn**: Visualization tools for evaluating the model.
- **scikit-learn**: Data preprocessing and classification metrics.





