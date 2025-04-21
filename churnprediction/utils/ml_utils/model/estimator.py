import os
import sys
import json
import joblib
import pandas as pd

from churnprediction.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging


class ChurnPredictionModel:
    def __init__(self, encoder_path, model_path):
        try:
            # Load label encoders
            with open(encoder_path, 'r') as file:
                self.label_encoders = json.load(file)

            # Load model (could include pipeline or pure model)
            self.model = joblib.load(model_path)

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        try:
            # Apply label encoding manually using stored encoders
            for col, mapping in self.label_encoders.items():
                if col in x.columns:
                    x[col] = x[col].map(mapping)

            return x

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def predict(self, x: pd.DataFrame):
        try:
            x_processed = self.preprocess(x)
            predictions = self.model.predict(x_processed)
            return predictions
        except Exception as e:
            raise ChurnPredictionException(e, sys)
