from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.components.data_validation import DataValidation
from churnprediction.components.data_transformation import DataTransformation
from churnprediction.components.model_trainer import ModelTrainer
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from churnprediction.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        print(f"🧭 Using artifact directory: {training_pipeline_config.artifact_dir}")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Starting data ingestion...")
        data_validation_config = DataValidationConfig(training_pipeline_config)

        logging.info("Starting data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")
        print(f"✅ Data saved to: {data_ingestion_artifact.feature_store_file_path}")

        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Starting data validation...")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed successfully.")
        print(f"✅ Data Validation successful and saved to: {data_validation_config.drift_report_file_path} ")

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)

        logging.info("🔄 Starting data transformation...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("✅ Data transformation completed.")
        print(f"✅ Data Transformation completed.")
        print(f"   ➕ Transformed Train File: {data_transformation_artifact.transformed_train_file_path}")
        print(f"   🧪 Transformed Test File:  {data_transformation_artifact.transformed_test_file_path}")

        logging.info("🏁 Pipeline execution completed successfully.")
        logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")

    except Exception as e:
        raise ChurnPredictionException(e, sys)



