# -*- coding: utf-8 -*-
import csv
import pickle as pkl
from io import StringIO

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = pkl.load(open("kidney_disease.pkl", "rb"))


class RequestBody(BaseModel):
    type: str
    content: str


@app.get("/")
def hello_world():
    return {"Hello": "World"}


@app.post("/predict")
def predict(data: RequestBody):

    if data.type == "csv":
        csvStringIO = StringIO(data.content)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=True)

    if data.type == "json":
        pass

    prediction = model.predict(to_predict)

    return prediction
