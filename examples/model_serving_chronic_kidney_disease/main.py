# -*- coding: utf-8 -*-


import logging
import pickle as pkl
from io import StringIO

import pandas as pd
import uvicorn
from chronoc_kidney_disease_project.ml.data.sqllite_func import Databasecon
from chronoc_kidney_disease_project.ml.model.dataframe_create import Dframe
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
)
app = FastAPI()

model = pkl.load(open("kidney_disease.pkl", "rb"))

# Class to define the request body and the type hints of each attribute
class RequestBody(BaseModel):
    type: str
    content: str


@app.get("/")
def hello_world():
    return {"Hello": "World"}


# Endpoint to receive data for making prediction
@app.post("/predict")
def predict(data: RequestBody):

    if data.type == "csv":
        pass
        csvStringIO = StringIO(data.content)
        logging.info(csvStringIO)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=0)

    if data.type == "json":
        pass

    prediction = model.predict(to_predict)

    data_with_prediction = Dframe().dataframe_create(prediction, to_predict)

    Databasecon().sql_insertion(data_with_prediction)

    prediction_as_list = prediction.tolist()
    return prediction_as_list
