from datetime import datetime
import os
from churnprediction.constant import training_pipeline


print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACTS_DIR)


CURRENT_TIMESTAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACTS_DIR
        self.artifact_dir = os.path.join(self.artifact_name, CURRENT_TIMESTAMP)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_ingestion"
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.FILE_NAME  # "customerchurn.csv"
        )
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir, "data_validation"
        )

        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, "drift_report.yaml"
        )

        self.invalid_data_report_file_path: str = os.path.join(
            self.data_validation_dir, "invalid_data_report.yaml"
        )

        self.X_train_file_path: str = os.path.join(self.data_validation_dir, "X_train.csv")
        self.X_test_file_path: str = os.path.join(self.data_validation_dir, "X_test.csv")
        self.y_train_file_path: str = os.path.join(self.data_validation_dir, "y_train.csv")
        self.y_test_file_path: str = os.path.join(self.data_validation_dir, "y_test.csv")
        self.label_encoder_path: str = os.path.join(self.data_validation_dir, "label_encoders.json")

        self.schema_file_path: str = os.path.join("schemas", "data_schema.yaml")


class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.X_TRAIN_SMOTE_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)
