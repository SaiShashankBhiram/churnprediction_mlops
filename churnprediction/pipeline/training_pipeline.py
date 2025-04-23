import os
import sys

from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging

from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.components.data_validation import DataValidation
from churnprediction.components.data_transformation import DataTransformation
from churnprediction.components.model_trainer import ModelTrainer
from churnprediction.cloud.s3_syncer import S3Sync
from churnprediction.constant.training_pipeline import TRAINING_BUCKET_NAME


from churnprediction.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from churnprediction.entity.artifact_entity import(
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            logging.info("Initiate the data Validation")
            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    ## local artifact is going to s3 bucket    
    def sync_artifact_dir_to_s3(self):
        try:
            artifacts_base_dir = "Artifacts"
            if not os.path.exists(artifacts_base_dir):
                logging.warning(f"No Artifacts directory found at: {artifacts_base_dir}")
                return

            for folder_name in os.listdir(artifacts_base_dir):
                full_path = os.path.join(artifacts_base_dir, folder_name)
                if os.path.isdir(full_path):
                    aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{folder_name}"
                    logging.info(f"Uploading artifact folder {full_path} to {aws_bucket_url}")
                    self.s3_sync.sync_folder_to_s3(folder=full_path, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            logging.error(f"Failed to upload artifacts: {e}")
            raise ChurnPredictionException(e, sys)
        
    ## local final model is going to s3 bucket 
        
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact)
            
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            return model_trainer_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
