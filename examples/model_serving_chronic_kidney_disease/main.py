# -*- coding: utf-8 -*-
import csv
import logging
import pickle as pkl
from io import StringIO

import byteplay as bp
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
)
app = FastAPI()
rows = []
predictions = []
data_list = []


model = pkl.load(open("kidney_disease.pkl", "rb"))

# Class to define the request body and the type hints of each attribute
class RequestBody(BaseModel):
    type: str
    content: str


# Endpoint to receive data for making prediction
@app.post("/predict")
# Endpoint to receive data for making prediction
@app.post("/predict")
def predict(data: RequestBody):

    if data.type == "csv":
        data_points = data.content
        data_list = data_points.splitlines()
        print(data_list)
        csvStringIO = StringIO(data_points)
        logging.info(csvStringIO)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=None)
        prediction = model.predict(to_predict)

    if data.type == "json":
        pass

    # make predictions on the input dataset
    prediction = model.predict(to_predict)

    predictions = prediction.tolist()
    print(predictions)

    data_tuples = list(zip(data_list, predictions))
    print(data_tuples)

    df = pd.DataFrame(data_tuples, columns=["Input", "Output"])
    print(df)

    return predictions
    # return data.content
    return prediction.tolist()
