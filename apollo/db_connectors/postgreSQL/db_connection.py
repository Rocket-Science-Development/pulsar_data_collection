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


def sqlupdate(df):
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

        df.to_sql(name="mpm_10jul", con=conn, if_exists="append")

        cursor.execute("SELECT * FROM mpm_10jul;")
        print(cursor.fetchall())

    except db.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if cursor:
            cursor.close()
            conn.close()
            print("The SQLite connection is closed")

    return
