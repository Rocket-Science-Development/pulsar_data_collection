# -*- coding: utf-8 -*-
# import influxdb_client as ifdb
import pandas as pd
from influxdb import DataFrameClient


class StorageEngine:
    def __init__(self):
        pass

    def sql_insertion(self, df: pd.DataFrame, database_login=None):

        """
        Function to push Pandas Dataframe into Influx DB.
        """

        # Set 'TimeStamp' field as index of dataframe
        df.set_index("Timestamp", inplace=True)

        client = DataFrameClient(
            database_login.db_host,
            database_login.db_port,
            database_login.db_user,
            database_login.db_password,
            database_login.db_name,
        )

        client.create_database(database_login.db_name)
        client.get_list_database()
        client.switch_database(database_login.db_name)

        client.write_points(
            df, database_login.db_name, protocol=database_login.protocol, time_precision="u", tag_columns=("uuid",)
        )

        return df

    def sql_digestion(self, database_login):
        client = DataFrameClient(
            database_login.db_host,
            database_login.db_port,
            database_login.db_user,
            database_login.db_password,
            database_login.db_name,
        )

        client.switch_database(database_login.db_name)

        df = client.query(f"select * from {database_login.db_name}", chunked=True)

        return df
