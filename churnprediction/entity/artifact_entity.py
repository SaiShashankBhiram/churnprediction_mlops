from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    drift_report_file_path: str  
    invalid_data_report_file_path: str  
    schema_validation_status: bool  
    dataset_drift_status: bool
    metadata: Optional[dict[str, str]] = None
