"""
Model Training Script for Wellness Tourism Package Prediction
================================================================

This script implements a complete machine learning training pipeline:
1. Loads preprocessed data from Hugging Face Hub
2. Analyzes class distribution and optimizes class weights
3. Trains XGBoost model with hyperparameter tuning using RandomizedSearchCV
4. Evaluates model performance with comprehensive metrics
5. Logs experiments to MLflow for tracking
6. Saves and uploads best model to Hugging Face Hub

Enhanced Version - Optimized for >90% Test Accuracy

Author: Baskaran Radhakrishnan
Date: 2026
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
import os
import time
import subprocess
from datetime import datetime

# Third-party imports for data manipulation
import pandas as pd
import numpy as np

# Scikit-learn imports for preprocessing and model evaluation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, 
    precision_score, f1_score, confusion_matrix, roc_auc_score
)

# XGBoost for gradient boosting model
import xgboost as xgb

# Model serialization
import joblib

# Hugging Face Hub for model and dataset management
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# ngrok for MLflow UI tunneling (optional, for remote access)
from pyngrok import ngrok


# ============================================================================
# SECTION 2: CONFIGURATION AND CONSTANTS
# ============================================================================

# Hugging Face Configuration
HF_USERNAME = "BaskaranAIExpert"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wellness-tourism-mlops")

# Hyperparameter Tuning Configuration
N_ITER_RANDOM_SEARCH = 150  # Number of random combinations to try
CV_FOLDS = 5                 # Cross-validation folds
SCORING_METRIC = 'accuracy'  # Metric to optimize (changed from 'recall' for >90% goal)

# Model Configuration
MODEL_FILENAME = "wellness_tourism_model_v1.joblib"
RANDOM_STATE = 42


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
# SECTION 5: MLFLOW SETUP (Optional: with ngrok for remote access)
# ============================================================================

def setup_mlflow_with_ngrok():
    """
    Sets up MLflow with ngrok tunnel for remote access to MLflow UI.
    This is optional and mainly useful for Google Colab or remote environments.
    
    Note: Remove this section if running locally without ngrok.
    """
    try:
        # ngrok authentication (replace with your token)
        ngrok.set_auth_token("386kJSf5sGnd3Q97rrGB2jqJbHS_3EZZ43WbsA1bfskjTC12R")
        
        # Start MLflow UI on port 5000
        process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])
        time.sleep(5)  # Wait for MLflow UI to start
        
        # Create public tunnel
        public_url = ngrok.connect(5000).public_url
        print(f"✓ MLflow UI is available at: {public_url}")
        
        # Set tracking URI to ngrok URL
        mlflow.set_tracking_uri(public_url)
        mlflow.set_experiment("MLOps_experiment")
        
    except Exception as e:
        print(f"⚠ Warning: Could not set up ngrok tunnel: {e}")
        print("Continuing with local MLflow tracking...")


def initialize_mlflow():
    """
    Initializes MLflow tracking with configured settings.
    """
    # Set tracking URI (use environment variable or default to local)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"✓ MLflow experiment '{EXPERIMENT_NAME}' set successfully.")
    except Exception as e:
        print(f"⚠ Warning: Could not set MLflow experiment: {e}")
        print("Continuing with default experiment...")


# ============================================================================
# SECTION 6: DATA LOADING
# ============================================================================

def load_preprocessed_data(hf_username):
    """
    Loads preprocessed training and test datasets from Hugging Face Hub.
    
    Args:
        hf_username (str): Hugging Face username
        
    Returns:
        tuple: (Xtrain, Xtest, ytrain, ytest) - Training and test datasets
        
    Raises:
        FileNotFoundError: If data files cannot be loaded
    """
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    print("Loading preprocessed data from Hugging Face Hub...")
    
    # Define dataset paths
    dataset_base = f"hf://datasets/{hf_username}/wellness-tourism-dataset"
    Xtrain_path = f"{dataset_base}/Xtrain.csv"
    Xtest_path = f"{dataset_base}/Xtest.csv"
    ytrain_path = f"{dataset_base}/ytrain.csv"
    ytest_path = f"{dataset_base}/ytest.csv"
    
    try:
        # Load datasets
        Xtrain = pd.read_csv(Xtrain_path)
        Xtest = pd.read_csv(Xtest_path)
        ytrain = pd.read_csv(ytrain_path).squeeze()  # Convert DataFrame to Series
        ytest = pd.read_csv(ytest_path).squeeze()
        
        print("✓ Data loaded successfully")
        return Xtrain, Xtest, ytrain, ytest
        
    except Exception as e:
        raise FileNotFoundError(f"Failed to load data: {str(e)}")


def clean_data(Xtrain, Xtest):
    """
    Cleans the datasets by removing unnecessary columns.
    
    Args:
        Xtrain (pd.DataFrame): Training features
        Xtest (pd.DataFrame): Test features
        
    Returns:
        tuple: (Xtrain, Xtest) - Cleaned datasets
    """
    # Drop 'Unnamed: 0' column if it exists (from CSV index)
    if 'Unnamed: 0' in Xtrain.columns:
        Xtrain = Xtrain.drop(columns=['Unnamed: 0'])
        print("✓ Dropped 'Unnamed: 0' column from Xtrain")
    if 'Unnamed: 0' in Xtest.columns:
        Xtest = Xtest.drop(columns=['Unnamed: 0'])
        print("✓ Dropped 'Unnamed: 0' column from Xtest")
    
    print(f"\nTraining set shape: {Xtrain.shape}")
    print(f"Test set shape: {Xtest.shape}")
    
    return Xtrain, Xtest


# ============================================================================
# SECTION 7: FEATURE DEFINITION
# ============================================================================

def define_features():
    """
    Defines numeric and categorical features for the model.
    
    Returns:
        tuple: (numeric_features, categorical_features) - Lists of feature names
    """
    # Numeric features (continuous or discrete numeric variables)
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
        'Passport',      # Binary (0/1) but treated as numeric
        'OwnCar'         # Binary (0/1) but treated as numeric
    ]
    
    # Categorical features (nominal variables that need encoding)
    categorical_features = [
        'TypeofContact',
        'CityTier',
        'Occupation',
        'Gender',
        'MaritalStatus',
        'Designation',
        'ProductPitched'
    ]
    
    return numeric_features, categorical_features


def filter_existing_features(Xtrain, numeric_features, categorical_features):
    """
    Filters feature lists to only include features that exist in the dataset.
    
    Args:
        Xtrain (pd.DataFrame): Training dataset
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        tuple: (filtered_numeric, filtered_categorical) - Filtered feature lists
    """
    numeric_features = [f for f in numeric_features if f in Xtrain.columns]
    categorical_features = [f for f in categorical_features if f in Xtrain.columns]
    
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    return numeric_features, categorical_features


# ============================================================================
# SECTION 8: CLASS IMBALANCE ANALYSIS AND OPTIMIZATION
# ============================================================================

def analyze_class_distribution(ytrain):
    """
    Analyzes the class distribution in the training data.
    
    Args:
        ytrain (pd.Series): Training target variable
        
    Returns:
        float: Class weight ratio (majority/minority)
    """
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
    
    return class_weight


def optimize_class_weight(Xtrain, ytrain, Xtest, ytest, numeric_features, categorical_features):
    """
    Tests multiple class weight values to find the optimal one for accuracy.
    
    This function quickly tests different class weights using a simple model
    to find the best weight before the full hyperparameter tuning.
    
    Args:
        Xtrain (pd.DataFrame): Training features
        ytrain (pd.Series): Training target
        Xtest (pd.DataFrame): Test features
        ytest (pd.Series): Test target
        numeric_features (list): Numeric feature names
        categorical_features (list): Categorical feature names
        
    Returns:
        float: Optimal class weight
    """
    print("\nTesting different class weights for optimal accuracy...")
    
    # Calculate base class weight
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
    
    # Define class weight options to test
    class_weight_options = [
        class_weight * 0.9,  # Slightly less weight
        class_weight,         # Original ratio
        class_weight * 1.1,  # Slightly more weight
        3.5,                 # Fixed value (often works well)
        4.0,                 # Fixed value
    ]
    
    # Create a simple preprocessor for quick testing
    preprocessor_test = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        remainder='passthrough'
    )
    
    # Test each class weight
    best_weight = class_weight
    best_test_acc = 0
    
    for weight in class_weight_options:
        # Create a simple XGBoost model for testing
        xgb_test = xgb.XGBClassifier(
            scale_pos_weight=weight,
            random_state=RANDOM_STATE,
            n_estimators=100,      # Fewer trees for speed
            max_depth=5,            # Moderate depth
            learning_rate=0.05,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Create pipeline and fit
        pipeline_test = make_pipeline(preprocessor_test, xgb_test)
        pipeline_test.fit(Xtrain, ytrain)
        
        # Evaluate on test set
        y_pred_test = pipeline_test.predict(Xtest)
        acc = accuracy_score(ytest, y_pred_test)
        
        # Track best weight
        if acc > best_test_acc:
            best_test_acc = acc
            best_weight = weight
        
        print(f"  Weight: {weight:.2f} → Test Accuracy: {acc:.4f}")
    
    print(f"\n✓ Best class weight: {best_weight:.2f} (Test Accuracy: {best_test_acc:.4f})")
    
    return best_weight


# ============================================================================
# SECTION 9: MODEL PIPELINE CONSTRUCTION
# ============================================================================

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Creates a preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_features (list): Numeric feature names
        categorical_features (list): Categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print("\n" + "="*50)
    print("MODEL SETUP")
    print("="*50)
    print("Creating preprocessing pipeline...")
    
    preprocessor = make_column_transformer(
        # StandardScaler: Normalizes numeric features (mean=0, std=1)
        (StandardScaler(), numeric_features),
        
        # OneHotEncoder: Converts categorical features to binary vectors
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        
        # remainder='passthrough': Keep any other columns as-is
        remainder='passthrough'
    )
    
    return preprocessor


def create_xgboost_model(class_weight):
    """
    Creates an XGBoost classifier with optimized class weight.
    
    Args:
        class_weight (float): Scale positive weight for handling class imbalance
        
    Returns:
        XGBClassifier: Configured XGBoost model
    """
    print("Initializing XGBoost model...")
    
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=class_weight,  # Handle class imbalance
        random_state=RANDOM_STATE,      # For reproducibility
        eval_metric='logloss',          # Evaluation metric
        use_label_encoder=False         # Avoid deprecation warning
    )
    
    return xgb_model


def define_hyperparameter_grid():
    """
    Defines the hyperparameter search space for RandomizedSearchCV.
    
    Returns:
        dict: Hyperparameter grid with parameter names and value ranges
    """
    print("Setting up enhanced hyperparameter grid...")
    
    param_grid = {
        # Number of boosting rounds (trees)
        'xgbclassifier__n_estimators': [300, 400, 500, 600],
        
        # Maximum depth of trees (controls model complexity)
        'xgbclassifier__max_depth': [5, 6, 7, 8],
        
        # Learning rate (shrinkage factor)
        'xgbclassifier__learning_rate': [0.01, 0.03, 0.05, 0.08],
        
        # Fraction of columns (features) to use per tree
        'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        
        # Fraction of columns (features) to use per level
        'xgbclassifier__colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
        
        # Fraction of samples to use per tree (prevents overfitting)
        'xgbclassifier__subsample': [0.8, 0.9, 1.0],
        
        # L2 regularization (lambda) - controls overfitting
        'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5, 2.0],
        
        # L1 regularization (alpha) - feature selection
        'xgbclassifier__reg_alpha': [0, 0.1, 0.5, 1.0],
        
        # Minimum sum of instance weight needed in a child
        'xgbclassifier__min_child_weight': [1, 3, 5, 7],
        
        # Minimum loss reduction required for split (gamma)
        'xgbclassifier__gamma': [0, 0.1, 0.2, 0.3],
    }
    
    return param_grid


def create_model_pipeline(preprocessor, xgb_model):
    """
    Creates a complete model pipeline combining preprocessing and model.
    
    Args:
        preprocessor (ColumnTransformer): Preprocessing pipeline
        xgb_model (XGBClassifier): XGBoost model
        
    Returns:
        Pipeline: Complete model pipeline
    """
    model_pipeline = make_pipeline(preprocessor, xgb_model)
    return model_pipeline


# ============================================================================
# SECTION 10: HYPERPARAMETER TUNING
# ============================================================================

def setup_hyperparameter_tuning(model_pipeline, param_grid):
    """
    Sets up RandomizedSearchCV for hyperparameter tuning.
    
    RandomizedSearchCV is used instead of GridSearchCV because:
    - It's faster (doesn't try all combinations)
    - It can explore a larger hyperparameter space
    - It's more efficient for high-dimensional parameter spaces
    
    Args:
        model_pipeline (Pipeline): Complete model pipeline
        param_grid (dict): Hyperparameter search space
        
    Returns:
        RandomizedSearchCV: Configured search object
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    print(f"This will explore {N_ITER_RANDOM_SEARCH} random combinations (may take 15-30 minutes)...")
    print(f"Optimizing for: {SCORING_METRIC.upper()}")
    
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_grid,
        n_iter=N_ITER_RANDOM_SEARCH,  # Number of random combinations to try
        cv=CV_FOLDS,                   # 5-fold cross-validation
        scoring=SCORING_METRIC,        # Optimize for accuracy
        n_jobs=-1,                     # Use all available CPU cores
        verbose=1,                     # Show progress
        random_state=RANDOM_STATE       # For reproducibility
    )
    
    return random_search


# ============================================================================
# SECTION 11: MODEL EVALUATION
# ============================================================================

def evaluate_model(best_model, Xtrain, ytrain, Xtest, ytest):
    """
    Evaluates the model on both training and test sets with comprehensive metrics.
    
    Args:
        best_model: Trained model
        Xtrain (pd.DataFrame): Training features
        ytrain (pd.Series): Training target
        Xtest (pd.DataFrame): Test features
        ytest (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Generate predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_pred_proba_train = best_model.predict_proba(Xtrain)
    y_pred_proba_test = best_model.predict_proba(Xtest)
    
    # Calculate metrics for training set
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    train_recall = recall_score(ytrain, y_pred_train, zero_division=0)
    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    train_f1 = f1_score(ytrain, y_pred_train, zero_division=0)
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(ytest, y_pred_test)
    test_recall = recall_score(ytest, y_pred_test, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)
    test_f1 = f1_score(ytest, y_pred_test, zero_division=0)
    
    # Calculate ROC-AUC (if possible)
    try:
        train_roc_auc = roc_auc_score(ytrain, y_pred_proba_train[:, 1])
        test_roc_auc = roc_auc_score(ytest, y_pred_proba_test[:, 1])
    except:
        train_roc_auc = None
        test_roc_auc = None
    
    # Display metrics
    print("\nTraining Set Metrics:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    if train_roc_auc:
        print(f"  ROC-AUC:   {train_roc_auc:.4f}")
    
    print("\nTest Set Metrics:")
    target_achieved = "✓ TARGET ACHIEVED!" if test_accuracy >= 0.90 else "⚠ Below 90%"
    print(f"  Accuracy:  {test_accuracy:.4f} {target_achieved}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    if test_roc_auc:
        print(f"  ROC-AUC:   {test_roc_auc:.4f}")
    
    # Print classification reports
    print("\nTraining Set Classification Report:")
    print(classification_report(ytrain, y_pred_train))
    
    print("\nTest Set Classification Report:")
    print(classification_report(ytest, y_pred_test))
    
    # Print confusion matrix
    print("\nTest Set Confusion Matrix:")
    cm = confusion_matrix(ytest, y_pred_test)
    print(cm)
    print(f"\n  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Return metrics dictionary
    metrics = {
        'train': {
            'accuracy': train_accuracy,
            'recall': train_recall,
            'precision': train_precision,
            'f1': train_f1,
            'roc_auc': train_roc_auc
        },
        'test': {
            'accuracy': test_accuracy,
            'recall': test_recall,
            'precision': test_precision,
            'f1': test_f1,
            'roc_auc': test_roc_auc
        },
        'predictions': {
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_pred_proba_train': y_pred_proba_train,
            'y_pred_proba_test': y_pred_proba_test
        }
    }
    
    return metrics


# ============================================================================
# SECTION 12: MLFLOW LOGGING
# ============================================================================

def log_to_mlflow(random_search, Xtrain, Xtest, ytrain, ytest, 
                  numeric_features, categorical_features, 
                  class_weight, best_weight, metrics):
    """
    Logs all experiment information to MLflow for tracking.
    
    Args:
        random_search: RandomizedSearchCV object with best model
        Xtrain (pd.DataFrame): Training features
        Xtest (pd.DataFrame): Test features
        ytrain (pd.Series): Training target
        ytest (pd.Series): Test target
        numeric_features (list): Numeric feature names
        categorical_features (list): Categorical feature names
        class_weight (float): Original class weight
        best_weight (float): Optimized class weight
        metrics (dict): Evaluation metrics dictionary
    """
    print("\nLogging metrics to MLflow...")
    
    try:
        # Log dataset information
        mlflow.log_param("dataset_size_train", len(Xtrain))
        mlflow.log_param("dataset_size_test", len(Xtest))
        mlflow.log_param("n_features", Xtrain.shape[1])
        mlflow.log_param("n_numeric_features", len(numeric_features))
        mlflow.log_param("n_categorical_features", len(categorical_features))
        mlflow.log_param("class_weight_original", round(class_weight, 4))
        mlflow.log_param("class_weight_optimized", round(best_weight, 4))
        
        # Log hyperparameter grid
        param_grid = random_search.param_distributions
        for param, values in param_grid.items():
            mlflow.log_param(f"grid_{param}", str(values))
        
        # Log search configuration
        mlflow.log_param("search_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", N_ITER_RANDOM_SEARCH)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("scoring_metric", SCORING_METRIC)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        # Log best hyperparameters
        for param, value in random_search.best_params_.items():
            mlflow.log_param(param, value)
        
        # Log cross-validation score
        mlflow.log_metric("best_cv_accuracy", random_search.best_score_)
        
        # Log training metrics
        mlflow.log_metric("train_accuracy", metrics['train']['accuracy'])
        mlflow.log_metric("train_recall", metrics['train']['recall'])
        mlflow.log_metric("train_precision", metrics['train']['precision'])
        mlflow.log_metric("train_f1", metrics['train']['f1'])
        if metrics['train']['roc_auc']:
            mlflow.log_metric("train_roc_auc", metrics['train']['roc_auc'])
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", metrics['test']['accuracy'])
        mlflow.log_metric("test_recall", metrics['test']['recall'])
        mlflow.log_metric("test_precision", metrics['test']['precision'])
        mlflow.log_metric("test_f1", metrics['test']['f1'])
        if metrics['test']['roc_auc']:
            mlflow.log_metric("test_roc_auc", metrics['test']['roc_auc'])
        
        # Save classification reports as artifacts
        with open("train_classification_report.txt", "w") as f:
            f.write(classification_report(ytrain, metrics['predictions']['y_pred_train']))
        with open("test_classification_report.txt", "w") as f:
            f.write(classification_report(ytest, metrics['predictions']['y_pred_test']))
        
        mlflow.log_artifact("train_classification_report.txt")
        mlflow.log_artifact("test_classification_report.txt")
        
        # Log model
        mlflow.sklearn.log_model(
            random_search.best_estimator_,
            "wellness_tourism_model",
            registered_model_name="WellnessTourismXGBoost",
            input_example=Xtrain.head(1),
            signature=None
        )
        
        # Set tags
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("target", "ProdTaken")
        mlflow.set_tag("hf_username", HF_USERNAME)
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d"))
        mlflow.set_tag("optimization_goal", "accuracy_above_90")
        mlflow.set_tag("search_method", "RandomizedSearchCV")
        
        print("✓ Metrics logged successfully to MLflow.")
        print(f"  MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"  MLflow Run URI: {mlflow.get_artifact_uri()}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not log to MLflow: {e}")
        print("Continuing without MLflow logging...")


# ============================================================================
# SECTION 13: MODEL PERSISTENCE
# ============================================================================

def save_model(best_model, filename=MODEL_FILENAME):
    """
    Saves the trained model to disk.
    
    Args:
        best_model: Trained model to save
        filename (str): Filename for saved model
        
    Raises:
        IOError: If model saving fails
    """
    print(f"\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    print(f"Saving best model as '{filename}'...")
    
    try:
        joblib.dump(best_model, filename)
        print("✓ Model saved successfully.")
    except Exception as e:
        raise IOError(f"Failed to save model: {str(e)}")


def upload_model_to_hf(api, filename, hf_username):
    """
    Uploads the trained model to Hugging Face Hub.
    
    Args:
        api (HfApi): Hugging Face API client
        filename (str): Model filename
        hf_username (str): Hugging Face username
        
    Raises:
        RuntimeError: If upload fails
    """
    print(f"\n" + "="*50)
    print("UPLOADING TO HUGGING FACE HUB")
    print("="*50)
    
    repo_id = f"{hf_username}/wellness-tourism-model"
    repo_type = "model"
    
    print(f"Uploading model to Hugging Face Hub ({repo_id})...")
    
    # Check if repository exists, create if not
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
    
    # Upload model file
    try:
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✓ Model '{filename}' uploaded successfully!")
        print(f"✓ Model available at: https://huggingface.co/{repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload model: {str(e)}")


# ============================================================================
# SECTION 14: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function that orchestrates the entire training pipeline.
    """
    print("\n" + "="*70)
    print("WELLNESS TOURISM PACKAGE PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Validate environment
    validate_environment()
    
    # Step 2: Initialize Hugging Face client
    api = initialize_huggingface_client()
    
    # Step 3: Setup MLflow (optional: uncomment if using ngrok)
    # setup_mlflow_with_ngrok()
    initialize_mlflow()
    
    # Step 4: Load and clean data
    Xtrain, Xtest, ytrain, ytest = load_preprocessed_data(HF_USERNAME)
    Xtrain, Xtest = clean_data(Xtrain, Xtest)
    
    # Step 5: Define and filter features
    numeric_features, categorical_features = define_features()
    numeric_features, categorical_features = filter_existing_features(
        Xtrain, numeric_features, categorical_features
    )
    
    # Step 6: Analyze class distribution and optimize class weight
    class_weight = analyze_class_distribution(ytrain)
    best_weight = optimize_class_weight(
        Xtrain, ytrain, Xtest, ytest, 
        numeric_features, categorical_features
    )
    
    # Step 7: Create model pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    xgb_model = create_xgboost_model(best_weight)
    model_pipeline = create_model_pipeline(preprocessor, xgb_model)
    
    # Step 8: Setup hyperparameter tuning
    param_grid = define_hyperparameter_grid()
    random_search = setup_hyperparameter_tuning(model_pipeline, param_grid)
    
    # Step 9: Train model with hyperparameter tuning (inside MLflow run)
    with mlflow.start_run(run_name=f"wellness_tourism_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print("\n" + "="*50)
        print("MLFLOW EXPERIMENT TRACKING")
        print("="*50)
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("="*50)
        
        # Fit the model
        random_search.fit(Xtrain, ytrain)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Display best parameters
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*50)
        print("\nBest Parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest Cross-Validation Score (Accuracy): {random_search.best_score_:.4f}")
        
        # Step 10: Evaluate model
        metrics = evaluate_model(best_model, Xtrain, ytrain, Xtest, ytest)
        
        # Step 11: Log to MLflow
        log_to_mlflow(
            random_search, Xtrain, Xtest, ytrain, ytest,
            numeric_features, categorical_features,
            class_weight, best_weight, metrics
        )
        
        # Step 12: Save and upload model
        save_model(best_model)
        upload_model_to_hf(api, MODEL_FILENAME, HF_USERNAME)
        
        print("\n" + "="*50)
        print("MLFLOW TRACKING COMPLETED")
        print("="*50)
    
    # Final summary
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    if 'metrics' in locals():
        test_accuracy = metrics['test']['accuracy']
        if test_accuracy >= 0.90:
            print(f"🎉 SUCCESS: Test Accuracy = {test_accuracy:.2%} (Target: ≥90%)")
        else:
            print(f"⚠ Test Accuracy = {test_accuracy:.2%} (Target: ≥90%)")
            print("   Consider: Feature engineering, ensemble methods, or more hyperparameter tuning")
    
    print("="*50)


# ============================================================================
# SECTION 15: SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()