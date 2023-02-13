from typing import List

import pandas as pd
from database_actions import DatabaseActions, DatabaseActionsFactory
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync


class InfluxdbActions(DatabaseActions):
    """Infuxdb database connection"""

    def make_connection(self, url: str, token: str, org: str) -> InfluxDBClientAsync:
        """Makes connection to the Influxdb database"""

        return InfluxDBClientAsync(url=url, token=token, org=org)

    async def write_data(
        self,
        async_client: InfluxDBClientAsync,
        bucket: str,
        records: pd.DataFrame,
        data_frame_measurement_name: str,
        data_frame_tag_columns: List[str],
        data_frame_timestamp_column: str,
    ):
        """Sends data to the database"""
        async with async_client as client:
            await client.write_api().write(
                bucket=bucket,
                record=records,
                data_frame_measurement_name=data_frame_measurement_name,
                data_frame_tag_columns=data_frame_tag_columns,
                data_frame_timestamp_column=data_frame_timestamp_column,
            )


class Influxdb(DatabaseActionsFactory):
    """Infuxdb database connection"""

    def get_database_actions(self) -> DatabaseActions:
        """Makes connection to the Influxdb database"""
        return InfluxdbActions
