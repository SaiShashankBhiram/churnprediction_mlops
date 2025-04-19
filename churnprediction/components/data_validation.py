from churnprediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from churnprediction.entity.config_entity import DataValidationConfig
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os, sys, json
from churnprediction.utils.main_utils.utils import read_yaml_file, write_yaml_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory if it doesn't exist
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_validation(self):
        try:
            # Define paths
            input_file_path = "artifacts/data_ingestion/customerchurn.csv"
            output_dir = "artifacts/data_validation"
            encoders_file_path = os.path.join(output_dir, "label_encoders.json")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Load data
            df = pd.read_csv(input_file_path)
            logging.info(f"Loaded data from {input_file_path} with shape {df.shape}")

            # Initialize encoder dictionary
            encoder_dict = {}

            # Encode all categorical columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoder_dict[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
            logging.info("Categorical columns label encoded.")

            # Split into features and target
            if "Churn" not in df.columns:
                raise ChurnPredictionException("Target column 'Churn' not found in dataset.", sys)

            X = df.drop("Churn", axis=1)
            y = df["Churn"]

            self.detect_dataset_drift(base_df=df, current_df=df)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Save splits
            X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

            # Save encoders
            with open(encoders_file_path, "w") as f:
                json.dump(encoder_dict, f, indent=4)

            logging.info("Data validation completed and saved successfully.")

        except Exception as e:
            raise ChurnPredictionException(e, sys)
