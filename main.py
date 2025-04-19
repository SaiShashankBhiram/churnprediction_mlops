from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.components.data_validation import DataValidation
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.entity.config_entity import DataIngestionConfig, DataValidationConfig
from churnprediction.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_validation_config = DataValidationConfig(training_pipeline_config)

        logging.info("Starting data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")
        print(f"✅ Data saved to: {data_ingestion_artifact.feature_store_file_path}")

        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Starting data validation...")
        data_validation.initiate_data_validation()
        logging.info("Data Validation completed successfully.")
        print(f"✅ Data Validation successful and saved to: {data_validation_config.drift_report_file_path} ")

    except Exception as e:
        raise ChurnPredictionException(e, sys)

