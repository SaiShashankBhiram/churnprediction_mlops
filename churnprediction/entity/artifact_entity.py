from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    def __init__(
        self,
        drift_report_file_path: str,
        invalid_data_report_file_path: str,
        X_train_file_path: str,
        X_test_file_path: str,
        y_train_file_path: str,
        y_test_file_path: str,
        schema_validation_status: bool,
        dataset_drift_status: bool
    ):
        self.drift_report_file_path = drift_report_file_path
        self.invalid_data_report_file_path = invalid_data_report_file_path
        self.X_train_file_path = X_train_file_path
        self.X_test_file_path = X_test_file_path
        self.y_train_file_path = y_train_file_path
        self.y_test_file_path = y_test_file_path
        self.schema_validation_status = schema_validation_status
        self.dataset_drift_status = dataset_drift_status

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
