import os
import sys
import numpy as np
import pandas as pd

"""
defining common constant variable for training pipeline
"""

PIPELINE_NAME: str = "churnprediction"
ARTIFACTS_DIR: str = "Artifacts"
FILE_NAME: str =  "customerchurn.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "ChurnPrediction"
DATA_INGESTION_DATABASE_NAME: str = "SaiShashankDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

"""
Data transformation related constant start with DATA_TRANSFORMATION VAR NAME 
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Transformed file names using SMOTE
X_TRAIN_SMOTE_FILE_NAME: str = "X_train_smote.npy"
Y_TRAIN_SMOTE_FILE_NAME: str = "y_train_smote.npy"
TEST_FILE_NAME: str = "test.npy"  # Combined X_test and y_test
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessor.pkl"

"""
Model Trainer related constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
