"""
Data Registration Script for Wellness Tourism Package Prediction
================================================================

This script uploads the raw dataset to Hugging Face Hub as a dataset repository.
It handles repository creation if it doesn't exist and uploads all files from
the local data directory.

Author: MLOps Engineer
Date: 2024
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
import os

# Hugging Face Hub imports for dataset management
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


# ============================================================================
# SECTION 2: CONFIGURATION AND CONSTANTS
# ============================================================================

# Repository Configuration
REPO_ID = "BaskaranAIExpert/wellness-tourism-dataset"
REPO_TYPE = "dataset"

# Data Path Configuration
DATA_FOLDER_PATH = "data"  # Local folder containing dataset files


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
# SECTION 5: REPOSITORY MANAGEMENT
# ============================================================================

def check_or_create_repository(api, repo_id, repo_type):
    """
    Checks if the repository exists, creates it if it doesn't.
    
    Args:
        api (HfApi): Hugging Face API client
        repo_id (str): Repository identifier (username/repo-name)
        repo_type (str): Type of repository ('dataset', 'model', or 'space')
        
    Raises:
        RuntimeError: If repository creation fails
    """
    try:
        # Check if repository exists
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✓ Dataset repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        # Repository doesn't exist, create it
        print(f"⚠ Dataset repository '{repo_id}' not found. Creating new repository...")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=False  # Make repository public
            )
            print(f"✓ Dataset repository '{repo_id}' created successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to create repository: {str(e)}")


# ============================================================================
# SECTION 6: DATASET UPLOAD
# ============================================================================

def upload_dataset(api, folder_path, repo_id, repo_type):
    """
    Uploads the dataset folder to Hugging Face Hub.
    
    Args:
        api (HfApi): Hugging Face API client
        folder_path (str): Path to local folder containing dataset files
        repo_id (str): Repository identifier
        repo_type (str): Type of repository
        
    Raises:
        FileNotFoundError: If data folder doesn't exist
        RuntimeError: If upload fails
    """
    # Validate folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Data folder '{folder_path}' not found. "
            "Please ensure the folder exists and contains your dataset files."
        )
    
    print("\n" + "="*50)
    print("UPLOADING DATASET")
    print("="*50)
    print(f"Uploading dataset from '{folder_path}' to Hugging Face Hub...")
    print(f"Repository: {repo_id}")
    
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"\n✓ Dataset uploaded successfully!")
        print(f"✓ Dataset available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload dataset: {str(e)}")


# ============================================================================
# SECTION 7: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function that orchestrates the dataset registration pipeline.
    """
    print("\n" + "="*70)
    print("WELLNESS TOURISM PACKAGE PREDICTION - DATA REGISTRATION PIPELINE")
    print("="*70)
    
    # Step 1: Validate environment
    validate_environment()
    
    # Step 2: Initialize Hugging Face client
    api = initialize_huggingface_client()
    
    # Step 3: Check or create repository
    check_or_create_repository(api, REPO_ID, REPO_TYPE)
    
    # Step 4: Upload dataset
    upload_dataset(api, DATA_FOLDER_PATH, REPO_ID, REPO_TYPE)
    
    # Final summary
    print("\n" + "="*50)
    print("DATA REGISTRATION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"✓ Repository: {REPO_ID}")
    print(f"✓ Data folder: {DATA_FOLDER_PATH}")
    print("="*50)


# ============================================================================
# SECTION 8: SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()