import pandas as pd

# Set the path to the dataset folder
dataset_path = "D:\\Python\\Tumor Detection\\covid-chestxray-dataset"  # Update this to your extracted folder

# Load metadata
metadata_file = dataset_path + "\\metadata.csv"
df = pd.read_csv(metadata_file)

# Show first few rows
print(df.head())

