# -*- coding: utf-8 -*-
import logging
import pickle as pkl
import sys
from io import StringIO

import pandas as pd
import uvicorn
from fastapi import FastAPI
from numpy import ndarray
from pydantic import BaseModel

sys.path.append("../../")
from pulsar_data_collection.data_capture.data_capture import (
    DatabaseLogin,
    DataCapture,
    DataWithPrediction,
)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
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

    database_login = DatabaseLogin(
        db_host="influx",
        db_port=8086,
        db_user="admin",
        db_password="pass123",
        db_name="testDB",
        protocol="line",
        measurement="something",
    )

    dat_predict = DataWithPrediction(prediction=prediction, data_points=to_predict)

    dat_capture = DataCapture(
        storage_engine="influxdb",
        model_id="RS101",
        model_version="1.0",
        data_id="FluxDB",
        y_name="y_pred",
        pred_name="clf_target",
        operation_type="INSERT",
        login_url=database_login,
    )

    dat_capture.push(dat_predict)
    prediction_as_list = prediction.tolist()
    return prediction_as_list
