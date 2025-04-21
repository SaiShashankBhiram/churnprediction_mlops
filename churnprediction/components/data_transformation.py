from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from churnprediction.utils.main_utils.utils import save_numpy_array_data, save_object
from churnprediction.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from churnprediction.entity.config_entity import DataTransformationConfig
from churnprediction.logging.logger import logging
from churnprediction.exception.exception import ChurnPredictionException
import os, sys


class DataTransformation:
    def __init__(self, 
                 data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("🔁 Starting data transformation")
            
            # ✅ Use the paths directly from the artifact
            X_train_file_path = self.data_validation_artifact.X_train_file_path
            X_test_file_path = self.data_validation_artifact.X_test_file_path
            y_train_file_path = self.data_validation_artifact.y_train_file_path
            y_test_file_path = self.data_validation_artifact.y_test_file_path


            # Load the already split files
            X_train = pd.read_csv(X_train_file_path)
            X_test = pd.read_csv(X_test_file_path)
            y_train = pd.read_csv(y_train_file_path)
            y_test = pd.read_csv(y_test_file_path)

            # Flatten y_train and y_test if needed
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

            # Replace missing values in TotalCharges with 0.0
            if 'TotalCharges' in X_train.columns:
                X_train['TotalCharges'] = X_train['TotalCharges'].fillna(0.0)
                X_test['TotalCharges'] = X_test['TotalCharges'].fillna(0.0)
                logging.info("🧼 Filled missing TotalCharges values with 0.0")

            # Fill other missing values with 0.0 just to be safe
            X_train.fillna(0.0, inplace=True)
            X_test.fillna(0.0, inplace=True)

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            logging.info(f"📊 After SMOTE: X_train_smote shape: {X_train_smote.shape}, y_train_smote shape: {y_train_smote.shape}")

            # Save transformed arrays separately
            train_array_path = os.path.join(self.data_transformation_config.data_transformation_dir, "X_train_smote.npy")
            train_target_path = os.path.join(self.data_transformation_config.data_transformation_dir, "y_train_smote.npy")
            
            save_numpy_array_data(train_array_path, X_train_smote)
            save_numpy_array_data(train_target_path, y_train_smote)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,
                                  array=np.c_[X_test.values, y_test.values])

            # No preprocessor object, but placeholder to maintain pipeline structure
            save_object(self.data_transformation_config.transformed_object_file_path, None)

            # Prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=train_array_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                label_encoder_path = self.data_validation_artifact.label_encoder_path
            )

            logging.info("✅ Data transformation completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys)
