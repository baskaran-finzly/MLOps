"""
Data Preparation Script for Wellness Tourism Package Prediction
This script loads data from Hugging Face Hub, performs cleaning and preprocessing,
splits into train/test sets, and uploads processed data back to HF Hub.
"""

# For data manipulation
import pandas as pd
import sklearn
# For creating folders
import os
# For data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# For converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder
# For Hugging Face Hub authentication to upload files
from huggingface_hub import HfApi


HF_USERNAME = "BaskaranAIExpert"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define constants for the dataset and output paths
DATASET_PATH = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/wellness_tourism_dataset.csv"  # Update filename if different

print("Loading dataset from Hugging Face Hub...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded successfully. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Data Cleaning: Drop unnecessary columns
print("\nDropping unnecessary columns...")
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)
    print("Dropped 'CustomerID' column.")
else:
    print("'CustomerID' column not found. Skipping.")

# Check for missing values
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values found:")
    print(missing_values[missing_values > 0])
    # Fill missing values or drop rows based on your strategy
    df = df.dropna()  # Drop rows with missing values
    print("Dropped rows with missing values.")
else:
    print("No missing values found.")

# Encoding categorical columns
print("\nEncoding categorical columns...")
categorical_columns = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}' column.")
    else:
        print(f"Warning: '{col}' column not found in dataset.")

# Define target column
target_col = 'ProdTaken'

# Verify target column exists
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# Split into X (features) and y (target)
print(f"\nSplitting data into features and target (target: '{target_col}')...")
X = df.drop(columns=[target_col])
y = df[target_col]

# Display class distribution
print(f"\nTarget variable distribution:")
print(y.value_counts())
if len(y.value_counts()) == 2:
    print(f"Class ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")

# Perform train-test split (80-20 split)
print("\nPerforming train-test split (80% train, 20% test)...")
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {Xtrain.shape}")
print(f"Test set shape: {Xtest.shape}")

# Save train and test datasets locally
print("\nSaving train and test datasets locally...")
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Files saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

# Upload processed datasets back to Hugging Face Hub
print("\nUploading processed datasets to Hugging Face Hub...")
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # Just the filename
        repo_id=f"{HF_USERNAME}/wellness-tourism-dataset",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path}")

print("\nData preparation completed successfully!")

