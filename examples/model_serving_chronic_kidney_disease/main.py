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
    csvStringIO = StringIO(data_points)
    logging.info(csvStringIO)
    to_predict = pd.read_csv(csvStringIO, sep=",")

    if data.type == "json":
        pass

    prediction = model.predict(to_predict)

    prediction_as_list = prediction.tolist()

    columns_names_list = to_predict.columns
    columns_names_list.append("Target")

    data_with_prediction = pd.concat([to_predict, prediction], axis=1, names=features_names_list)

    sql_insertion(df)

    return prediction_as_list


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

        df.to_sql(name="mpm_13jul", con=conn, if_exists="append", index=False)

        df = pd.read_sql_query("SELECT * from mpm_13jul", conn)
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
