# -*- coding: utf-8 -*-
import datetime
import logging
import pickle as pkl
import sys
from io import StringIO

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append("../../")
from pulsar_data_collection.data_capture.data_capture import (
    DataCaptureParameters as DCParam,
)
from pulsar_data_collection.data_capture.data_capture import DataFrame as DFrame
from pulsar_data_collection.data_capture.data_capture import (
    DataFrameCreate as DFrameCreate,
)
from pulsar_data_collection.data_capture.data_capture import (
    DataWithPrediction as DPredict,
)
from pulsar_data_collection.db_connectors.sqlite.db_connection import (
    StorageEngine as SQLiteStorage,
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
async def predict(data: RequestBody):

    if data.type == "csv":
        pass
        csvStringIO = StringIO(data.content)
        logging.info(csvStringIO)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=0)

    if data.type == "json":
        pass
        jsonStringIO = StringIO(data.content)
        logging.info(jsonStringIO)
        to_predict = pd.read_json(jsonStringIO, strict=False)

    prediction = model.predict(to_predict)

    dframe = DFrame(prediction=prediction, to_predict=to_predict)
    # struc_dframe = DFrame(**dframe)
    data_with_prediction = DFrameCreate().dataframe_create(dframe)

    SQLiteStorage().sql_insertion(data_with_prediction)

    prediction_as_list = prediction.tolist()
    return prediction_as_list
