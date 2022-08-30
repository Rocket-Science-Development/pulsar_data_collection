# -*- coding: utf-8 -*-
import pandas as pd
from influxdb import DataFrameClient
from influxdb import InfluxDBClient as ifdb

# Setup database
# client = ifdb('localhost', 8086, 'admin', 'pass123', 'test2Aug')
# client.create_database('test2Aug')
# client.get_list_database()
# client.switch_database('test2Aug')

dbhost = "localhost"
dbport = 8086
dbuser = "admin"
dbpasswd = "pass123"
dbname = "testDB"
protocol = "line"


class StorageEngine:
    def __init__(self):
        pass

    def sql_insertion(self, df: pd.DataFrame):

        """
        Function to push Pandas Dataframe into Influx DB.
        """

        # Set 'TimeStamp' field as index of dataframe
        df.set_index("Timestamp", inplace=True)

        print(df.head())

        client = DataFrameClient(dbhost, dbport, dbuser, dbpasswd, dbname)

        client.create_database(dbname)
        client.get_list_database()
        client.switch_database(dbname)

        client.write_points(df, "test5Aug", protocol=protocol, time_precision="u")

        return df
