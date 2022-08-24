# -*- coding: utf-8 -*-
# import influxdb_client as ifdb
from datetime import datetime

import pandas as pd
from influxdb import DataFrameClient
from influxdb import InfluxDBClient as ifdb

from .config import DB_NAME, DB_PASSWORD, DB_PORT, DB_HOST, DB_USER, DB_PROTOCOL, DB_MEASURMENT

# Setup database
# client = ifdb('localhost', 8086, 'admin', 'pass123', 'test2Aug')
# client.create_database('test2Aug')
# client.get_list_database()
# client.switch_database('test2Aug')



class StorageEngine:
    def __init__(self):
        pass

    def sql_insertion(self, df: pd.DataFrame, db_host: str=DB_HOST, db_port: str=DB_PORT,
                      db_user: str=DB_USER, db_password: str=DB_PASSWORD, db_name: str=DB_NAME, protocol: str=DB_PROTOCOL,
                      measurment: str=DB_MEASURMENT):

        """
        Function to push Pandas Dataframe into Influx DB.
        """

        # Set 'TimeStamp' field as index of dataframe
        df.set_index("Timestamp", inplace=True)

        print(df.head())

        client = DataFrameClient(db_host, db_port, db_user, db_password, db_name)
        print(db_host, db_port, db_user, db_password, db_name)
        # client.create_database(db_name)
        # client.get_list_database()
        # client.switch_database(db_name)

        client.write_points(df, "test5Aug", protocol=protocol, time_precision="u")

        return df
