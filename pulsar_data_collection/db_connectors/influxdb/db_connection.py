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

    def sql_insertion(self, df: pd.DataFrame, database_login=None):

        """
        Function to push Pandas Dataframe into Influx DB.
        """

        # Set 'TimeStamp' field as index of dataframe
        df.set_index("Timestamp", inplace=True)

        print(df.head())

        client = DataFrameClient(database_login.db_host, database_login.db_port, database_login.db_user, database_login.db_password, database_login.db_name)

        client.create_database(database_login.db_name)
        client.get_list_database()
        client.switch_database(database_login.db_name)

        client.write_points(df, "test5Aug", protocol=database_login.protocol, time_precision="u")

        return df
