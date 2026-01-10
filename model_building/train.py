"""
Model Training Script for Wellness Tourism Package Prediction
This script loads preprocessed data, trains models with hyperparameter tuning,
evaluates performance, and uploads the best model to Hugging Face Hub.
"""

# For data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# For model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score, confusion_matrix
# For model serialization
import joblib
# For creating folders
import os
# For Hugging Face Hub authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# TODO: Replace with your Hugging Face username
HF_USERNAME = "BaskaranAIExpert"  # Change this!

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load preprocessed data from Hugging Face Hub
print("Loading preprocessed data from Hugging Face Hub...")
Xtrain_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/Xtrain.csv"
Xtest_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/Xtest.csv"
ytrain_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/ytrain.csv"
ytest_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # Convert DataFrame to Series
ytest = pd.read_csv(ytest_path).squeeze()  # Convert DataFrame to Series

# Drop 'Unnamed: 0' column if it exists (from CSV index)
if 'Unnamed: 0' in Xtrain.columns:
    Xtrain = Xtrain.drop(columns=['Unnamed: 0'])
    print("Dropped 'Unnamed: 0' column from Xtrain")
if 'Unnamed: 0' in Xtest.columns:
    Xtest = Xtest.drop(columns=['Unnamed: 0'])
    print("Dropped 'Unnamed: 0' column from Xtest")

print(f"Training set shape: {Xtrain.shape}")
print(f"Test set shape: {Xtest.shape}")

# Define feature types
numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch',
    'NumberOfChildrenVisiting',
    'Passport',
    'OwnCar'
]

categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

# Filter features that exist in the dataset
numeric_features = [f for f in numeric_features if f in Xtrain.columns]
categorical_features = [f for f in categorical_features if f in Xtrain.columns]

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Calculate class weight to handle imbalance
print("\nCalculating class weights for imbalanced data...")
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# Create preprocessing pipeline
print("\nCreating preprocessing pipeline...")
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    remainder='passthrough'
)

# Define XGBoost model with class weight handling
print("Initializing XGBoost model...")
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

# Define hyperparameter grid for tuning
print("Setting up hyperparameter grid...")
param_grid = {
    'xgbclassifier__n_estimators': [100, 150, 200],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__colsample_bytree': [0.6, 0.7, 0.8],
    'xgbclassifier__colsample_bylevel': [0.6, 0.7, 0.8],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
print("\nStarting hyperparameter tuning with GridSearchCV...")
print("This may take several minutes...")
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='recall',  # Optimize for recall to catch more potential buyers
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain, ytrain)

# Get best model
best_model = grid_search.best_estimator_
print("\n" + "="*50)
print("HYPERPARAMETER TUNING RESULTS")
print("="*50)
print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest Cross-Validation Score (Recall): {grid_search.best_score_:.4f}")

# Predict on training set
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Calculate metrics
train_accuracy = accuracy_score(ytrain, y_pred_train)
test_accuracy = accuracy_score(ytest, y_pred_test)
train_recall = recall_score(ytrain, y_pred_train)
test_recall = recall_score(ytest, y_pred_test)
train_precision = precision_score(ytrain, y_pred_train)
test_precision = precision_score(ytest, y_pred_test)
train_f1 = f1_score(ytrain, y_pred_train)
test_f1 = f1_score(ytest, y_pred_test)

print("\nTraining Set Metrics:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print("\nTest Set Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

print("\nTraining Set Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Set Classification Report:")
print(classification_report(ytest, y_pred_test))

print("\nTest Set Confusion Matrix:")
print(confusion_matrix(ytest, y_pred_test))

# Save best model
model_filename = "wellness_tourism_model_v1.joblib"
print(f"\nSaving best model as '{model_filename}'...")
joblib.dump(best_model, model_filename)
print("Model saved successfully.")

# Upload model to Hugging Face Hub
repo_id = f"{HF_USERNAME}/wellness-tourism-model"
repo_type = "model"

print(f"\nUploading model to Hugging Face Hub ({repo_id})...")

# Check if the model repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model repository '{repo_id}' created successfully.")

# Upload the model file
api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Model '{model_filename}' uploaded successfully!")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)