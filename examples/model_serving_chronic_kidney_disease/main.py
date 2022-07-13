# -*- coding: utf-8 -*-
import csv
import logging
import pickle as pkl
import sqlite3 as db
from io import StringIO

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


@app.get("/")
def hello_world():
    return {"Hello": "World"}


# Endpoint to receive data for making prediction
@app.post("/predict")
def predict(data: RequestBody):

    if data.type == "csv":
        data_points = data.content
        data_list = data_points.splitlines()
        # print(data_list)
        csvStringIO = StringIO(data_points)
        logging.info(csvStringIO)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=None)

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

    sql_insertion(df)

    return predictions
    # return data.content


def sql_insertion(df):
    try:
        conn = db.connect("SQLite_Python.db")
        cursor = conn.cursor()
        print("Database created and Successfully Connected to SQLite")

        # cursor.execute("SELECT * FROM mpm_data_ing;")

        df.to_sql("df", conn, if_exists="replace")

        # print(cursor.fetchall())

        # cursor.execute(
        #     """
        #     CREATE TABLE IF NOT EXISTS mpm_data_ing as
        #     SELECT * FROM df
        #     """
        # )

        df.to_sql(name="mpm_10jul", con=conn, if_exists="append", index=False)

        df = pd.read_sql_query("SELECT * from mpm_10jul", conn)
        print(df)
        # cursor.execute("SELECT * FROM mpm_10jul;")
        # print(cursor.fetchall())

    except db.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if cursor:
            cursor.close()
            conn.close()
            print("The SQLite connection is closed")

    return df
