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

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "ChurnPrediction"
DATA_INGESTION_DATABASE_NAME: str = "SaiShashankDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

