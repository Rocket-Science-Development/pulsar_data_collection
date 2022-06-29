# -*- coding: utf-8 -*-
import csv
import pickle as pkl
from io import StringIO

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = pkl.load(open("kidney_disease.pkl", "rb"))

list_of_usernames = list()

# Class to define the request body and the type hints of each attribute
class RequestBody(BaseModel):
    type: str
    content: str


@app.get("/")
def hello_world():
    return {"Hello": "World"}


@app.get("/home/{user_name}")
def write_home(user_name: str, query):
    return {"Name": user_name, "query": query}


@app.put("/username/{user_name}")
def put_data(user_name: str):
    print(user_name)
    list_of_usernames.append(user_name)
    return {"username": user_name}


@app.post("/postData")
def post_data(user_name: str):
    list_of_usernames.append(user_name)
    return {"usernames": list_of_usernames}


@app.delete("/deleteData/{user_name}")
def delete_data(user_name: str):
    return {"usernames": list_of_usernames}


# Endpoint to receive data for making prediction
@app.post("/predict")
def predict(data: RequestBody):

    if data.type == "csv":
        csvStringIO = StringIO(data.content)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=None)

    if data.type == "json":
        pass

    prediction = model.predict(to_predict)

    return prediction
