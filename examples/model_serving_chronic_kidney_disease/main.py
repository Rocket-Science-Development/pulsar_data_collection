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

id = []
age = []
bp = []
sg = []
al = []
su = []
rbc = []
pc = []
pcc = []
ba = []
bgr = []
bu = []
sc = []
sod = []
pot = []
hemo = []
pcv = []
wbcc = []
rbcc = []
htn = []
dm = []
cad = []
appet = []
pe = []
ane = []
# classvar = []


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
        for data_line in data_list:
            data_line = data_line.strip()
            print(data_line)
            # if len(data_line) >= 24:
            line_info = data_line.split(",")
            id.append(line_info[0].strip())
            age.append(line_info[1].strip())
            bp.append(line_info[2].strip())
            sg.append(line_info[3].strip())
            al.append(line_info[4].strip())
            su.append(line_info[5].strip())
            rbc.append(line_info[6].strip())
            pc.append(line_info[7].strip())
            pcc.append(line_info[8].strip())
            ba.append(line_info[9].strip())
            bgr.append(line_info[10].strip())
            bu.append(line_info[11].strip())
            sc.append(line_info[12].strip())
            sod.append(line_info[13].strip())
            pot.append(line_info[14].strip())
            hemo.append(line_info[15].strip())
            pcv.append(line_info[16].strip())
            wbcc.append(line_info[17].strip())
            rbcc.append(line_info[18].strip())
            htn.append(line_info[19].strip())
            dm.append(line_info[20].strip())
            cad.append(line_info[21].strip())
            appet.append(line_info[22].strip())
            pe.append(line_info[23].strip())
            ane.append(line_info[24].strip())
            # classvar.append(line_info[25].strip())

        csvStringIO = StringIO(data_points)
        logging.info(csvStringIO)
        to_predict = pd.read_csv(csvStringIO, sep=",", header=None)

    if data.type == "json":
        pass

    # make predictions on the input dataset
    prediction = model.predict(to_predict)

    print(id)

    predictions = prediction.tolist()
    print(predictions)

    # data_tuples = list(zip(data_list, predictions))
    # print(data_tuples)

    data_tuples = list(
        zip(
            id,
            age,
            bp,
            sg,
            al,
            su,
            rbc,
            pc,
            pcc,
            ba,
            bgr,
            bu,
            sc,
            sod,
            pot,
            hemo,
            pcv,
            wbcc,
            rbcc,
            htn,
            dm,
            cad,
            appet,
            pe,
            ane,
            predictions,
        )
    )
    print(data_tuples)

    # df = pd.DataFrame(data_tuples, columns=["Input", "Output"])
    # print(df)

    df = pd.DataFrame(
        data_tuples,
        columns=[
            "id",
            "age",
            "bp",
            "sg",
            "al",
            "su",
            "rbc",
            "pc",
            "pcc",
            "ba",
            "bgr",
            "bu",
            "sc",
            "sod",
            "pot",
            "hemo",
            "pcv",
            "wbcc",
            "rbcc",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane",
            "predictions",
        ],
    )
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
