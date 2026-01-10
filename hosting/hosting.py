"""
Hosting Script for Wellness Tourism Package Prediction
This script uploads all deployment files to Hugging Face Spaces.
"""

from huggingface_hub import HfApi
import os

# TODO: Replace with your Hugging Face username
HF_USERNAME = "YOUR_USERNAME"  # Change this!

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload deployment folder to Hugging Face Space
print("Uploading deployment files to Hugging Face Space...")
api.upload_folder(
    folder_path="deployment",     # The local folder containing your deployment files
    repo_id=f"{HF_USERNAME}/Wellness-Tourism-Prediction",  # Your HF Space name (use hyphens, not underscores!)
    repo_type="space",             # dataset, model, or space
    path_in_repo="",               # Optional: subfolder path inside the repo
)

print("Deployment files uploaded successfully!")
print(f"Your app should be available at: https://huggingface.co/spaces/{HF_USERNAME}/Wellness-Tourism-Prediction")

