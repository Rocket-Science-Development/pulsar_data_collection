# -*- coding: utf-8 -*-
import sqlalchemy as sqla


class StorageEngine:
    def __init__(self, login_url):
        self.sqlite_engine = sqla.create_engine(login_url, future=True, echo=False)

    def connect(self, dataframe):
        pass


# def sqlupdate(df):
#     try:
#         conn = db.connect("SQLite_Python.db")
#         cursor = conn.cursor()
#         print("Database created and Successfully Connected to SQLite")

#         # cursor.execute("SELECT * FROM mpm_data_ing;")

#         df.to_sql("df", conn, if_exists="replace")

#         # print(cursor.fetchall())

#         # cursor.execute(
#         #     """
#         #     CREATE TABLE IF NOT EXISTS mpm_data_ing as
#         #     SELECT * FROM df
#         #     """
#         # )

#         df.to_sql(name="mpm_10jul", con=conn, if_exists="append")

#         cursor.execute("SELECT * FROM mpm_10jul;")
#         print(cursor.fetchall())

#     except db.Error as error:
#         print("Error while connecting to sqlite", error)
#     finally:
#         if cursor:
#             cursor.close()
#             conn.close()
#             print("The SQLite connection is closed")

#     return
