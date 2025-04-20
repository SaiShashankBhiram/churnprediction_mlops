import yaml
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os,sys
import numpy as np
import dill
import pickle

def read_yaml_file(file_path: str)-> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise ChurnPredictionException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_obj method of mainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            # Cross-validated accuracy
            accuracy_cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
            accuracy_cv = accuracy_cv_scores.mean()

            y_test_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='binary')
            recall = recall_score(y_test, y_test_pred, average='binary')
            f1 = f1_score(y_test, y_test_pred, average='binary')

            report[model_name] = {
                "cv_accuracy": accuracy_cv,
                "test_accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

        return report

    except Exception as e:
        raise ChurnPredictionException(e, sys)
