from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.exception.exception import ChurnPredictionException

from churnprediction.logging.logger import logging
from churnprediction.entity.config_entity import DataIngestionConfig
from churnprediction.entity.config_entity import TrainingPipelineConfig


if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data Ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)

    except Exception as e:
           raise ChurnPredictionException(e,sys)