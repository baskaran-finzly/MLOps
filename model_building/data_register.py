"""
Data Registration Script for Wellness Tourism Package Prediction
This script uploads the raw dataset to Hugging Face Hub as a dataset repository.
"""

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


# TODO: Replace with your Hugging Face username
repo_id = "YOUR_USERNAME/wellness-tourism-dataset"  # Change this!
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repository '{repo_id}' created successfully.")

# Step 2: Upload the data folder to Hugging Face Hub
print("Uploading dataset to Hugging Face Hub...")
api.upload_folder(
    folder_path="data",  # Path to the folder containing your dataset
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Data registration completed successfully!")

