import os
import sys
from xgboost import XGBClassifier
from urllib.parse import urlparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow

from churnprediction.exception.exception import ChurnPredictionException 
from churnprediction.logging.logger import logging

from churnprediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from churnprediction.entity.config_entity import ModelTrainerConfig

from churnprediction.utils.ml_utils.model.estimator import ChurnPredictionModel
from churnprediction.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from churnprediction.utils.ml_utils.metric.classification_metric import get_classification_score

import dagshub
dagshub.init(repo_owner='SaiShashankBhiram', repo_name='churnprediction_mlops', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    def track_mlflow(self, best_model, classificationmetric):
            with mlflow.start_run():
                f1_score = classificationmetric.f1_score
                precision_score = classificationmetric.precision_score
                recall_score = classificationmetric.recall_score
                accuracy_score = classificationmetric.accuracy_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                mlflow.log_metric("accuracy_score", accuracy_score)
                mlflow.sklearn.log_model(best_model,"model")


    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy'],
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [5, 10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            },
            "XGBoost": {
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7]
            }
        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)
        
        logging.info(f"Model Report: {model_report}")

        best_model_score = max([report.get('accuracy', 0) for report in model_report.values()])
        logging.info(f"Best model score: {best_model_score}")

        best_model_name = max(model_report, key=lambda k: model_report[k].get('accuracy', 0))
        best_model = models[best_model_name]

        logging.info(f"🏆 Best Model Selected: {best_model_name} with accuracy = {best_model_score}")
        print(f"🏆 Best Model Selected: {best_model_name} with accuracy = {best_model_score}")

        # Fit final best model
        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        
        # track the experiments with mlflow
        self.track_mlflow(best_model, classification_train_metric)
        self.track_mlflow(best_model, classification_test_metric)

        #preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        encoder_path = self.data_transformation_artifact.label_encoder_path
        model_path = self.model_trainer_config.trained_model_file_path

        save_object(model_path, obj=best_model)

        churn_prediction_model = ChurnPredictionModel(encoder_path=encoder_path, model_path=model_path)
        save_object(model_path, obj=churn_prediction_model)
        save_object("final_model/model.pkl", best_model)

        #copy label_encoders.json to final_model
        
        import shutil

        label_encoder_path = self.data_transformation_artifact.label_encoder_path
        final_encoder_path = "final_model/label_encoders.json"

        os.makedirs("final_model", exist_ok=True)

        shutil.copy(label_encoder_path, final_encoder_path)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"✅ Model trainer artifact created: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_X_path = os.path.join(self.data_transformation_artifact.transformed_train_file_path)
            train_y_path = train_X_path.replace("X_train_smote.npy", "y_train_smote.npy")
            test_array_path = self.data_transformation_artifact.transformed_test_file_path

            X_train = load_numpy_array_data(train_X_path)
            y_train = load_numpy_array_data(train_y_path)
            test_arr = load_numpy_array_data(test_array_path)

            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys)
