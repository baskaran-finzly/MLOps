"""
Hosting Script for Wellness Tourism Package Prediction
======================================================

This script uploads all deployment files to Hugging Face Spaces.
It handles the deployment of the Streamlit application, Dockerfile,
and requirements.txt to make the application publicly accessible.

Author: Baskaran Radhakrishnan
Date: 2026
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
import os

# Hugging Face Hub imports for space management
from huggingface_hub import HfApi


# ============================================================================
# SECTION 2: CONFIGURATION AND CONSTANTS
# ============================================================================

# Hugging Face Configuration
HF_USERNAME = "BaskaranAIExpert"
SPACE_NAME = "Wellness-Tourism-Prediction"  # Use hyphens, not underscores!

# Deployment Configuration
DEPLOYMENT_FOLDER = "deployment"  # Local folder containing deployment files
REPO_TYPE = "space"               # Type: 'space' for Hugging Face Spaces


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
# SECTION 5: DEPLOYMENT FOLDER VALIDATION
# ============================================================================

def validate_deployment_folder(folder_path):
    """
    Validates that the deployment folder exists and contains required files.
    
    Args:
        folder_path (str): Path to deployment folder
        
    Raises:
        FileNotFoundError: If deployment folder doesn't exist
        ValueError: If required files are missing
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Deployment folder '{folder_path}' not found. "
            "Please ensure the folder exists and contains deployment files."
        )
    
    # Check for required files
    required_files = ['app.py', 'Dockerfile', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        raise ValueError(
            f"Required files missing in deployment folder: {', '.join(missing_files)}"
        )
    
    print(f"✓ Deployment folder validated: {folder_path}")
    print(f"  Required files found: {', '.join(required_files)}")


# ============================================================================
# SECTION 6: SPACE UPLOAD
# ============================================================================

def upload_to_huggingface_space(api, folder_path, hf_username, space_name, repo_type):
    """
    Uploads deployment files to Hugging Face Space.
    
    Args:
        api (HfApi): Hugging Face API client
        folder_path (str): Path to local deployment folder
        hf_username (str): Hugging Face username
        space_name (str): Name of the Hugging Face Space
        repo_type (str): Type of repository ('space')
        
    Raises:
        RuntimeError: If upload fails
    """
    repo_id = f"{hf_username}/{space_name}"
    
    print("\n" + "="*50)
    print("UPLOADING TO HUGGING FACE SPACE")
    print("="*50)
    print(f"Uploading deployment files from '{folder_path}'...")
    print(f"Space: {repo_id}")
    print(f"Repository Type: {repo_type}")
    
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo="",  # Upload to root of the space
        )
        print(f"\n✓ Deployment files uploaded successfully!")
        print(f"✓ Space URL: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload to Hugging Face Space: {str(e)}")


# ============================================================================
# SECTION 7: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function that orchestrates the hosting deployment pipeline.
    """
    print("\n" + "="*70)
    print("WELLNESS TOURISM PACKAGE PREDICTION - HOSTING DEPLOYMENT PIPELINE")
    print("="*70)
    
    # Step 1: Validate environment
    validate_environment()
    
    # Step 2: Initialize Hugging Face client
    api = initialize_huggingface_client()
    
    # Step 3: Validate deployment folder
    validate_deployment_folder(DEPLOYMENT_FOLDER)
    
    # Step 4: Upload to Hugging Face Space
    upload_to_huggingface_space(
        api, DEPLOYMENT_FOLDER, HF_USERNAME, SPACE_NAME, REPO_TYPE
    )
    
    # Final summary
    print("\n" + "="*50)
    print("DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"✓ Space: {HF_USERNAME}/{SPACE_NAME}")
    print(f"✓ Deployment folder: {DEPLOYMENT_FOLDER}")
    print(f"✓ App URL: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
    print("\n💡 Note: It may take a few minutes for the app to build and deploy.")
    print("="*50)


# ============================================================================
# SECTION 8: SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()