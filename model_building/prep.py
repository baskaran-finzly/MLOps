"""
Data Preparation Script for Wellness Tourism Package Prediction
================================================================

This script performs comprehensive data preprocessing:
1. Loads raw dataset from Hugging Face Hub
2. Cleans data (removes unnecessary columns, handles missing values)
3. Encodes categorical variables using LabelEncoder
4. Splits data into train/test sets (80/20 split)
5. Saves processed datasets locally
6. Uploads processed datasets back to Hugging Face Hub

Author: MLOps Engineer
Date: 2024
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
import os

# Third-party imports for data manipulation
import pandas as pd

# Scikit-learn imports for data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Hugging Face Hub imports for dataset management
from huggingface_hub import HfApi


# ============================================================================
# SECTION 2: CONFIGURATION AND CONSTANTS
# ============================================================================

# Hugging Face Configuration
HF_USERNAME = "BaskaranAIExpert"
DATASET_REPO = "wellness-tourism-dataset"
RAW_DATA_FILE = "wellness_tourism_dataset.csv"

# Data Split Configuration
TEST_SIZE = 0.2          # 20% for testing
RANDOM_STATE = 42        # For reproducibility

# Target Column
TARGET_COLUMN = 'ProdTaken'

# Columns to Drop
COLUMNS_TO_DROP = ['CustomerID', 'Unnamed: 0']

# Categorical Columns to Encode
CATEGORICAL_COLUMNS = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

# Output File Names
OUTPUT_FILES = {
    'Xtrain': 'Xtrain.csv',
    'Xtest': 'Xtest.csv',
    'ytrain': 'ytrain.csv',
    'ytest': 'ytest.csv'
}


# ============================================================================
# SECTION 3: ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment():
    """
    Validates that all required environment variables are set.
    
    Raises:
        ValueError: If HF_TOKEN is not set
    """
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set it before running this script."
        )
    print("✓ Environment validation passed")


# ============================================================================
# SECTION 4: HUGGING FACE HUB INITIALIZATION
# ============================================================================

def initialize_huggingface_client():
    """
    Initializes and returns Hugging Face API client.
    
    Returns:
        HfApi: Initialized Hugging Face API client
        
    Raises:
        ConnectionError: If API initialization fails
    """
    try:
        api = HfApi(token=os.getenv("HF_TOKEN"))
        print("✓ Successfully initialized Hugging Face API client")
        return api
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Hugging Face API: {str(e)}")


# ============================================================================
# SECTION 5: DATA LOADING
# ============================================================================

def load_dataset(hf_username, dataset_repo, raw_data_file):
    """
    Loads the raw dataset from Hugging Face Hub.
    
    Args:
        hf_username (str): Hugging Face username
        dataset_repo (str): Dataset repository name
        raw_data_file (str): Name of the raw data file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If dataset cannot be loaded
    """
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    dataset_path = f"hf://datasets/{hf_username}/{dataset_repo}/{raw_data_file}"
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Dataset loaded successfully. Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Failed to load dataset: {str(e)}")


# ============================================================================
# SECTION 6: DATA CLEANING
# ============================================================================

def drop_unnecessary_columns(df, columns_to_drop):
    """
    Drops unnecessary columns from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_drop (list): List of column names to drop
        
    Returns:
        pd.DataFrame: Dataframe with columns dropped
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    print("Dropping unnecessary columns...")
    
    dropped_cols = [col for col in columns_to_drop if col in df.columns]
    
    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        print(f"✓ Dropped columns: {dropped_cols}")
    else:
        print("  No unnecessary columns found to drop.")
    
    return df


def handle_missing_values(df):
    """
    Checks for and handles missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    print("\nChecking for missing values...")
    
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        print("⚠ Missing values found:")
        print(missing_values[missing_values > 0])
        
        initial_shape = df.shape[0]
        df = df.dropna()  # Drop rows with missing values
        dropped_rows = initial_shape - df.shape[0]
        
        print(f"✓ Dropped {dropped_rows} rows with missing values.")
        print(f"  Remaining rows: {df.shape[0]}")
    else:
        print("✓ No missing values found.")
    
    return df


# ============================================================================
# SECTION 7: FEATURE ENCODING
# ============================================================================

def encode_categorical_features(df, categorical_columns):
    """
    Encodes categorical features using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        tuple: (encoded_dataframe, label_encoders_dict)
    """
    print("\n" + "="*50)
    print("FEATURE ENCODING")
    print("="*50)
    print("Encoding categorical columns...")
    
    label_encoders = {}
    encoded_count = 0
    
    for col in categorical_columns:
        if col in df.columns:
            # Initialize LabelEncoder
            le = LabelEncoder()
            
            # Fit and transform the column
            df[col] = le.fit_transform(df[col].astype(str))
            
            # Store encoder for potential inverse transformation
            label_encoders[col] = le
            encoded_count += 1
            
            print(f"  ✓ Encoded '{col}' ({len(le.classes_)} unique values)")
        else:
            print(f"  ⚠ Warning: '{col}' column not found in dataset.")
    
    print(f"\n✓ Successfully encoded {encoded_count} categorical columns.")
    
    return df, label_encoders


# ============================================================================
# SECTION 8: DATA SPLITTING
# ============================================================================

def split_features_and_target(df, target_column):
    """
    Splits the dataset into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) - Features and target
        
    Raises:
        ValueError: If target column doesn't exist
    """
    print(f"\n" + "="*50)
    print("DATA SPLITTING")
    print("="*50)
    print(f"Splitting data into features and target (target: '{target_column}')...")
    
    # Verify target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")
    
    # Split into X (features) and y (target)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Display class distribution
    print(f"\n  Target variable distribution:")
    class_dist = y.value_counts()
    print(f"    Class 0 (No Purchase): {class_dist[0]} ({class_dist[0]/len(y)*100:.1f}%)")
    print(f"    Class 1 (Purchase):    {class_dist[1]} ({class_dist[1]/len(y)*100:.1f}%)")
    
    if len(class_dist) == 2:
        class_ratio = class_dist[0] / class_dist[1]
        print(f"    Class ratio: {class_ratio:.2f}:1 (imbalanced dataset)")
    
    return X, y


def perform_train_test_split(X, y, test_size, random_state):
    """
    Performs train-test split with stratification.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing (0.0 to 1.0)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (Xtrain, Xtest, ytrain, ytest)
    """
    print(f"\n  Performing train-test split ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution in both sets
    )
    
    print(f"  ✓ Training set shape: {Xtrain.shape}")
    print(f"  ✓ Test set shape: {Xtest.shape}")
    
    return Xtrain, Xtest, ytrain, ytest


# ============================================================================
# SECTION 9: DATA PERSISTENCE
# ============================================================================

def save_processed_datasets(Xtrain, Xtest, ytrain, ytest, output_files):
    """
    Saves processed datasets to local CSV files.
    
    Args:
        Xtrain (pd.DataFrame): Training features
        Xtest (pd.DataFrame): Test features
        ytrain (pd.Series): Training target
        ytest (pd.Series): Test target
        output_files (dict): Dictionary mapping dataset names to filenames
        
    Raises:
        IOError: If file saving fails
    """
    print("\n" + "="*50)
    print("SAVING PROCESSED DATASETS")
    print("="*50)
    print("Saving train and test datasets locally...")
    
    try:
        Xtrain.to_csv(output_files['Xtrain'], index=False)
        Xtest.to_csv(output_files['Xtest'], index=False)
        ytrain.to_csv(output_files['ytrain'], index=False)
        ytest.to_csv(output_files['ytest'], index=False)
        
        print("✓ Files saved successfully:")
        for name, filename in output_files.items():
            print(f"  - {filename}")
    except Exception as e:
        raise IOError(f"Failed to save files: {str(e)}")


def upload_processed_datasets(api, hf_username, dataset_repo, output_files):
    """
    Uploads processed datasets to Hugging Face Hub.
    
    Args:
        api (HfApi): Hugging Face API client
        hf_username (str): Hugging Face username
        dataset_repo (str): Dataset repository name
        output_files (dict): Dictionary mapping dataset names to filenames
        
    Raises:
        RuntimeError: If upload fails
    """
    print("\n" + "="*50)
    print("UPLOADING TO HUGGING FACE HUB")
    print("="*50)
    print("Uploading processed datasets to Hugging Face Hub...")
    
    repo_id = f"{hf_username}/{dataset_repo}"
    files = list(output_files.values())
    uploaded_count = 0
    
    for file_path in files:
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),  # Just the filename
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  ✓ Uploaded {file_path}")
            uploaded_count += 1
        except Exception as e:
            print(f"  ✗ Failed to upload {file_path}: {str(e)}")
    
    if uploaded_count == len(files):
        print(f"\n✓ All {uploaded_count} files uploaded successfully!")
    else:
        print(f"\n⚠ Warning: Only {uploaded_count}/{len(files)} files uploaded successfully.")


# ============================================================================
# SECTION 10: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function that orchestrates the data preparation pipeline.
    """
    print("\n" + "="*70)
    print("WELLNESS TOURISM PACKAGE PREDICTION - DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Step 1: Validate environment
    validate_environment()
    
    # Step 2: Initialize Hugging Face client
    api = initialize_huggingface_client()
    
    # Step 3: Load dataset
    df = load_dataset(HF_USERNAME, DATASET_REPO, RAW_DATA_FILE)
    
    # Step 4: Clean data
    df = drop_unnecessary_columns(df, COLUMNS_TO_DROP)
    df = handle_missing_values(df)
    
    # Step 5: Encode categorical features
    df, label_encoders = encode_categorical_features(df, CATEGORICAL_COLUMNS)
    
    # Step 6: Split features and target
    X, y = split_features_and_target(df, TARGET_COLUMN)
    
    # Step 7: Perform train-test split
    Xtrain, Xtest, ytrain, ytest = perform_train_test_split(
        X, y, TEST_SIZE, RANDOM_STATE
    )
    
    # Step 8: Save processed datasets locally
    save_processed_datasets(Xtrain, Xtest, ytrain, ytest, OUTPUT_FILES)
    
    # Step 9: Upload processed datasets to Hugging Face Hub
    upload_processed_datasets(api, HF_USERNAME, DATASET_REPO, OUTPUT_FILES)
    
    # Final summary
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"✓ Training samples: {len(Xtrain)}")
    print(f"✓ Test samples: {len(Xtest)}")
    print(f"✓ Features: {Xtrain.shape[1]}")
    print("="*50)


# ============================================================================
# SECTION 11: SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()