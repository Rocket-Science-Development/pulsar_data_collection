# -*- coding: utf-8 -*-
import csv
import datetime
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

    data_with_prediction = dataframe_create(prediction, to_predict)

    sql_insertion(data_with_prediction)

    prediction_as_list = prediction.tolist()
    return prediction_as_list


# Function to convert prediction output to Pandas dataframe to be inserted in DB
def dataframe_create(prediction, to_predict: pd.DataFrame):
    # Creating dataframe with the output prediction
    pred_df = pd.DataFrame(prediction, columns=["class"])
    # Concat the input and output predicton dataframes on y-axis (columns)
    df = pd.concat([to_predict, pred_df], axis=1)
    # Adding current timestamp as a new column to existing Dataframe
    df.loc[:, "Timestamp"] = datetime.datetime.now()

    return df


def sql_insertion(df):
    try:
        conn = db.connect("SQLite_Python.db")
        cursor = conn.cursor()
        logging.info("Database created and Successfully Connected to SQLite")

        # cursor.execute("SELECT * FROM mpm_data_ing;")

        df.to_sql("df", conn, if_exists="replace")

        # print(cursor.fetchall())

        # cursor.execute(
        #     """
        #     CREATE TABLE IF NOT EXISTS mpm_data_ing as
        #     SELECT * FROM df
        #     """
        # )

        df.to_sql(name="mpm_19jul", con=conn, if_exists="append", index=False)

        df = pd.read_sql_query("SELECT * from mpm_19jul", conn)
        logging.info(df)
        # cursor.execute("SELECT * FROM mpm_10jul;")
        # print(cursor.fetchall())

    except db.Error as error:
        logging.error("Error while connecting to sqlite", error)
    finally:
        if cursor:
            cursor.close()
            conn.close()
            logging.info("The SQLite connection is closed")

    return df
