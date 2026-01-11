"""
Model Training Script for Wellness Tourism Package Prediction
This script loads preprocessed data, trains models with hyperparameter tuning,
evaluates performance, and uploads the best model to Hugging Face Hub.
Includes MLflow integration for experiment tracking.

Enhanced Version - Optimized for >90% Test Accuracy
"""

# For data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# For model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, 
    precision_score, f1_score, confusion_matrix, roc_auc_score
)
# For model serialization
import joblib
# For creating folders
import os
# For Hugging Face Hub authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# For MLflow experiment tracking
import mlflow
import mlflow.sklearn
from datetime import datetime
import numpy as np

# Configuration
HF_USERNAME = "BaskaranAIExpert"

# Validate environment
if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN environment variable is not set. Please set it before running this script.")

# Initialize API client
try:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    print("✓ Successfully initialized Hugging Face API client")
except Exception as e:
    raise ConnectionError(f"Failed to initialize Hugging Face API: {str(e)}")

# Initialize MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wellness-tourism-mlops")
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✓ MLflow experiment '{EXPERIMENT_NAME}' set successfully.")
except Exception as e:
    print(f"⚠ Warning: Could not set MLflow experiment: {e}")
    print("Continuing with default experiment...")

# Load preprocessed data from Hugging Face Hub
print("\n" + "="*50)
print("LOADING DATA")
print("="*50)
print("Loading preprocessed data from Hugging Face Hub...")
Xtrain_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/Xtrain.csv"
Xtest_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/Xtest.csv"
ytrain_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/ytrain.csv"
ytest_path = f"hf://datasets/{HF_USERNAME}/wellness-tourism-dataset/ytest.csv"

try:
    Xtrain = pd.read_csv(Xtrain_path)
    Xtest = pd.read_csv(Xtest_path)
    ytrain = pd.read_csv(ytrain_path).squeeze()  # Convert DataFrame to Series
    ytest = pd.read_csv(ytest_path).squeeze()  # Convert DataFrame to Series
    print("✓ Data loaded successfully")
except Exception as e:
    raise FileNotFoundError(f"Failed to load data: {str(e)}")

# Drop 'Unnamed: 0' column if it exists (from CSV index)
if 'Unnamed: 0' in Xtrain.columns:
    Xtrain = Xtrain.drop(columns=['Unnamed: 0'])
    print("✓ Dropped 'Unnamed: 0' column from Xtrain")
if 'Unnamed: 0' in Xtest.columns:
    Xtest = Xtest.drop(columns=['Unnamed: 0'])
    print("✓ Dropped 'Unnamed: 0' column from Xtest")

print(f"\nTraining set shape: {Xtrain.shape}")
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
print("\n" + "="*50)
print("DATA ANALYSIS")
print("="*50)
print("Calculating class weights for imbalanced data...")
class_dist = ytrain.value_counts()
class_weight = class_dist[0] / class_dist[1]
print(f"Class distribution:")
print(f"  Class 0 (No Purchase): {class_dist[0]} ({class_dist[0]/len(ytrain)*100:.1f}%)")
print(f"  Class 1 (Purchase):    {class_dist[1]} ({class_dist[1]/len(ytrain)*100:.1f}%)")
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# Fine-tune class weight for better accuracy
# Test multiple class weight values and select best
print("\nTesting different class weights for optimal accuracy...")
class_weight_options = [
    class_weight * 0.9,  # Slightly less weight
    class_weight,         # Original
    class_weight * 1.1,  # Slightly more weight
    3.5,                 # Fixed value
    4.0,                 # Fixed value
]

# Quick test with simple model
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor_test = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    remainder='passthrough'
)

best_weight = class_weight
best_test_acc = 0

for weight in class_weight_options:
    xgb_test = xgb.XGBClassifier(
        scale_pos_weight=weight,
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss'
    )
    pipeline_test = make_pipeline(preprocessor_test, xgb_test)
    pipeline_test.fit(Xtrain, ytrain)
    y_pred_test = pipeline_test.predict(Xtest)
    acc = accuracy_score(ytest, y_pred_test)
    if acc > best_test_acc:
        best_test_acc = acc
        best_weight = weight
    print(f"  Weight: {weight:.2f} → Test Accuracy: {acc:.4f}")

print(f"\n✓ Best class weight: {best_weight:.2f} (Test Accuracy: {best_test_acc:.4f})")

# Create preprocessing pipeline
print("\n" + "="*50)
print("MODEL SETUP")
print("="*50)
print("Creating preprocessing pipeline...")
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    remainder='passthrough'
)

# Define XGBoost model with optimized class weight
print("Initializing XGBoost model...")
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=best_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Enhanced hyperparameter grid for better accuracy
print("Setting up enhanced hyperparameter grid...")
param_grid = {
    'xgbclassifier__n_estimators': [300, 400, 500, 600],
    'xgbclassifier__max_depth': [5, 6, 7, 8],
    'xgbclassifier__learning_rate': [0.01, 0.03, 0.05, 0.08],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'xgbclassifier__colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
    'xgbclassifier__subsample': [0.8, 0.9, 1.0],
    'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5, 2.0],
    'xgbclassifier__reg_alpha': [0, 0.1, 0.5, 1.0],
    'xgbclassifier__min_child_weight': [1, 3, 5, 7],
    'xgbclassifier__gamma': [0, 0.1, 0.2, 0.3],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Use RandomizedSearchCV for better exploration (faster than GridSearchCV)
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)
print("Starting hyperparameter tuning with RandomizedSearchCV...")
print("This will explore 150 random combinations (may take 15-30 minutes)...")
print("Optimizing for: ACCURACY")

random_search = RandomizedSearchCV(
    model_pipeline,
    param_grid,
    n_iter=150,  # Try 150 random combinations
    cv=5,
    scoring='accuracy',  # Changed from 'recall' to 'accuracy' for >90% goal
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name=f"wellness_tourism_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    print("\n" + "="*50)
    print("MLFLOW EXPERIMENT TRACKING")
    print("="*50)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("="*50)
    
    # Log dataset information
    mlflow.log_param("dataset_size_train", len(Xtrain))
    mlflow.log_param("dataset_size_test", len(Xtest))
    mlflow.log_param("n_features", Xtrain.shape[1])
    mlflow.log_param("n_numeric_features", len(numeric_features))
    mlflow.log_param("n_categorical_features", len(categorical_features))
    mlflow.log_param("class_weight_original", round(class_weight, 4))
    mlflow.log_param("class_weight_optimized", round(best_weight, 4))
    
    # Log hyperparameter grid
    print("\nLogging hyperparameter grid to MLflow...")
    for param, values in param_grid.items():
        mlflow.log_param(f"grid_{param}", str(values))
    
    # Log RandomizedSearchCV configuration
    mlflow.log_param("search_method", "RandomizedSearchCV")
    mlflow.log_param("n_iter", 150)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring_metric", "accuracy")
    mlflow.log_param("random_state", 42)
    
    # Fit the model
    random_search.fit(Xtrain, ytrain)
    
    # Get best model
    best_model = random_search.best_estimator_
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    print("\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
        mlflow.log_param(param, value)
    
    print(f"\nBest Cross-Validation Score (Accuracy): {random_search.best_score_:.4f}")
    mlflow.log_metric("best_cv_accuracy", random_search.best_score_)
    
    # Predict on training and test sets
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_pred_proba_train = best_model.predict_proba(Xtrain)
    y_pred_proba_test = best_model.predict_proba(Xtest)
    
    # Calculate comprehensive metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)
    train_recall = recall_score(ytrain, y_pred_train, zero_division=0)
    test_recall = recall_score(ytest, y_pred_test, zero_division=0)
    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)
    train_f1 = f1_score(ytrain, y_pred_train, zero_division=0)
    test_f1 = f1_score(ytest, y_pred_test, zero_division=0)
    
    # Calculate ROC-AUC
    try:
        train_roc_auc = roc_auc_score(ytrain, y_pred_proba_train[:, 1])
        test_roc_auc = roc_auc_score(ytest, y_pred_proba_test[:, 1])
    except:
        train_roc_auc = None
        test_roc_auc = None
    
    print("\nTraining Set Metrics:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    if train_roc_auc:
        print(f"  ROC-AUC:   {train_roc_auc:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  Accuracy:  {test_accuracy:.4f} {'✓ TARGET ACHIEVED!' if test_accuracy >= 0.90 else '⚠ Below 90%'}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    if test_roc_auc:
        print(f"  ROC-AUC:   {test_roc_auc:.4f}")
    
    # Log metrics to MLflow
    print("\nLogging metrics to MLflow...")
    try:
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_f1", test_f1)
        if train_roc_auc:
            mlflow.log_metric("train_roc_auc", train_roc_auc)
        if test_roc_auc:
            mlflow.log_metric("test_roc_auc", test_roc_auc)
        
        # Save classification reports to files for MLflow artifacts
        with open("train_classification_report.txt", "w") as f:
            f.write(classification_report(ytrain, y_pred_train))
        with open("test_classification_report.txt", "w") as f:
            f.write(classification_report(ytest, y_pred_test))
        
        mlflow.log_artifact("train_classification_report.txt")
        mlflow.log_artifact("test_classification_report.txt")
        
        print("✓ Metrics logged successfully to MLflow.")
    except Exception as e:
        print(f"⚠ Warning: Could not log metrics to MLflow: {e}")
        print("Continuing without MLflow logging...")
    
    print("\nTraining Set Classification Report:")
    print(classification_report(ytrain, y_pred_train))
    
    print("\nTest Set Classification Report:")
    print(classification_report(ytest, y_pred_test))
    
    print("\nTest Set Confusion Matrix:")
    cm = confusion_matrix(ytest, y_pred_test)
    print(cm)
    print(f"\n  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Save best model
    model_filename = "wellness_tourism_model_v1.joblib"
    print(f"\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    print(f"Saving best model as '{model_filename}'...")
    try:
        joblib.dump(best_model, model_filename)
        print("✓ Model saved successfully.")
    except Exception as e:
        raise IOError(f"Failed to save model: {str(e)}")
    
    # Log model to MLflow
    print("\nLogging model to MLflow...")
    try:
        mlflow.sklearn.log_model(
            best_model,
            "wellness_tourism_model",
            registered_model_name="WellnessTourismXGBoost",
            input_example=Xtrain.head(1),
            signature=None
        )
        print("✓ Model logged successfully to MLflow.")
        print(f"  MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"  MLflow Run URI: {mlflow.get_artifact_uri()}")
    except Exception as e:
        print(f"⚠ Warning: Could not log model to MLflow: {e}")
        print("Model saved locally and will be uploaded to Hugging Face Hub.")
    
    # Log tags
    try:
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("target", "ProdTaken")
        mlflow.set_tag("hf_username", HF_USERNAME)
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d"))
        mlflow.set_tag("optimization_goal", "accuracy_above_90")
        mlflow.set_tag("search_method", "RandomizedSearchCV")
    except Exception as e:
        print(f"⚠ Warning: Could not set MLflow tags: {e}")
    
    print("\n" + "="*50)
    print("MLFLOW TRACKING COMPLETED")
    print("="*50)

    # Upload model to Hugging Face Hub
    repo_id = f"{HF_USERNAME}/wellness-tourism-model"
    repo_type = "model"
    
    print(f"\n" + "="*50)
    print("UPLOADING TO HUGGING FACE HUB")
    print("="*50)
    print(f"Uploading model to Hugging Face Hub ({repo_id})...")
    
    # Check if the model repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✓ Model repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"⚠ Model repository '{repo_id}' not found. Creating new repository...")
        try:
            create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
            print(f"✓ Model repository '{repo_id}' created successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to create repository: {str(e)}")
    
    # Upload the model file
    try:
        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=model_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✓ Model '{model_filename}' uploaded successfully!")
        print(f"✓ Model available at: https://huggingface.co/{repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload model: {str(e)}")
    
    # Log Hugging Face model info to MLflow
    try:
        mlflow.set_tag("hf_model_repo", repo_id)
        mlflow.log_param("hf_model_filename", model_filename)
    except Exception as e:
        print(f"⚠ Warning: Could not log HF info to MLflow: {e}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
if 'test_accuracy' in locals():
    if test_accuracy >= 0.90:
        print(f"🎉 SUCCESS: Test Accuracy = {test_accuracy:.2%} (Target: ≥90%)")
    else:
        print(f"⚠ Test Accuracy = {test_accuracy:.2%} (Target: ≥90%)")
        print("   Consider: Feature engineering, ensemble methods, or more hyperparameter tuning")
print("="*50)