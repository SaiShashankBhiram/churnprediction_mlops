from datetime import datetime
import os
from churnprediction.constant import training_pipeline


print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACTS_DIR)


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACTS_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline.PIPELINE_NAME, "data_ingestion"
        )
        self.feature_store_file_path: str = os.path.join(
            "Artifacts", "data_ingestion", training_pipeline.FILE_NAME  # "customerchurn.csv"
        )
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.drift_report_file_path: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_validation", "drift_report.yaml"
        )
        self.invalid_data_report_file_path: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_validation", "invalid_data_report.yaml"
        )
        self.schema_file_path: str = os.path.join(
            "schemas", "data_schema.yaml"
        )