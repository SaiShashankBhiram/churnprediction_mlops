import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging

class ChurnPredictionExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    def cv_to_json_convertor(self, file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)

            if "customerID" in data.columns:
                data.drop("customerID", axis=1, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection=collection
            self.records = records
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)

            return(len(self.records))
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
if __name__=='__main__':
    FILE_PATH = "churnprediction_data\customerchurn.csv"
    DATABASE="SaiShashankDB"
    Collection="ChurnPrediction"
    churnobj = ChurnPredictionExtract()
    records = churnobj.cv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = churnobj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)

        



