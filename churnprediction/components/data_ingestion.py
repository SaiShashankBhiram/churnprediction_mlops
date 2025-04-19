from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging

from churnprediction.entity.config_entity import DataIngestionConfig
from churnprediction.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Read data from MongoDB
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)

            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Save raw data to artifacts/data_ingestion/full_data.csv
        """
        try:
            file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(file_path, index=False, header=True)
            logging.info(f"Data saved to {file_path}")
            return file_path

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            feature_store_file_path = self.export_data_into_feature_store(df)

            return DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path
            )

        except Exception as e:
            raise ChurnPredictionException(e, sys)
