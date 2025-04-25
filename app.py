import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)

import pymongo
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from churnprediction.utils.main_utils.utils import load_object
from churnprediction.utils.ml_utils.model.estimator import ChurnPredictionModel

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from churnprediction.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from churnprediction.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise ChurnPredictionException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip().replace('',pd.NA)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace = True)

        #print(df)
        encoder_path = "final_model/label_encoders.json"
        model_path = "final_model/model.pkl"

        churnprediction_model = ChurnPredictionModel(
             encoder_path=encoder_path,
             model_path=model_path
        )

        y_pred = churnprediction_model.predict(df)
        df['predicted_column'] = y_pred

        df.to_csv('prediction_output/output.csv', index=False)

        table_html = df.to_html(classes='table table-striped', index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise ChurnPredictionException(e,sys)


if __name__=="__main__":
    app_run(app, host = "0.0.0.0", port=8080)

