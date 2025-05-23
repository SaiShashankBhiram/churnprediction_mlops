from churnprediction.entity.artifact_entity import ClassificationMetricArtifact
from churnprediction.exception.exception import ChurnPredictionException
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score
import sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_accuracy = accuracy_score(y_true, y_pred)
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score,
                    accuracy_score = model_accuracy)
        return classification_metric
    except Exception as e:
        raise ChurnPredictionException(e,sys)