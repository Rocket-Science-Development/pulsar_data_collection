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

    pred_df = pd.DataFrame(prediction, columns=["class"])
    data_with_prediction = pd.concat([to_predict, pred_df], axis=1)

    print(data_with_prediction)

    sql_insertion(data_with_prediction)

    prediction_as_list = prediction.tolist()
    return prediction_as_list


def sql_insertion(data_with_prediction):
    try:
        conn = db.connect("SQLite_Python.db")
        cursor = conn.cursor()
        logging.info("Database created and Successfully Connected to SQLite")

        # cursor.execute("SELECT * FROM mpm_data_ing;")

        data_with_prediction.to_sql("data_with_prediction", conn, if_exists="replace")

        # print(cursor.fetchall())

        # cursor.execute(
        #     """
        #     CREATE TABLE IF NOT EXISTS mpm_data_ing as
        #     SELECT * FROM df
        #     """
        # )

        data_with_prediction.to_sql(
            name="mpm_15jul", con=conn, if_exists="append", index=False
        )

        data_with_prediction = pd.read_sql_query("SELECT * from mpm_15jul", conn)
        logging.info(data_with_prediction)
        # cursor.execute("SELECT * FROM mpm_10jul;")
        # print(cursor.fetchall())

    except db.Error as error:
        logging.error("Error while connecting to sqlite", error)
    finally:
        if cursor:
            cursor.close()
            conn.close()
            logging.info("The SQLite connection is closed")

    return data_with_prediction
