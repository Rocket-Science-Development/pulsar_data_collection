# -*- coding: utf-8 -*-
# import influxdb_client as ifdb
import pandas as pd
from influxdb import DataFrameClient
from .config import DB_NAME, DB_METRICS_MEASURMENT, DB_PREDICTION_MEASURMENT, \
    DB_PERIOD_MEASURMENT


class StorageEngine:
    def __init__(self):
        pass

    def sql_insertion(self, measurment_name, df: pd.DataFrame, database_login=None):

        """
        Function to push Pandas Dataframe into Influx DB.
        """

        client = DataFrameClient(
            database_login.db_host,
            database_login.db_port,
            database_login.db_user,
            database_login.db_password,
            DB_NAME
        )
        client.create_database(DB_NAME)
        client.get_list_database()
        client.switch_database(DB_NAME)

        client.write_points(
            df, measurment_name, protocol=database_login.protocol, time_precision="u", tag_columns=("uuid",)
        )

        return df

    def sql_digestion(self, measurment_name, database_login):
        client = DataFrameClient(
            database_login.db_host,
            database_login.db_port,
            database_login.db_user,
            database_login.db_password,
            DB_NAME
        )

        client.switch_database(DB_NAME)

        df = client.query(f"select * from {measurment_name}", chunked=True)

        return df

