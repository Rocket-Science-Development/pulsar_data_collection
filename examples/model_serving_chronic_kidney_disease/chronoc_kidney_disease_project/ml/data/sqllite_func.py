# -*- coding: utf-8 -*-
import logging
import sqlite3 as db

import pandas as pd

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
)


class Databasecon:
    def __init__(self):
        pass

    def sql_insertion(self, df):
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