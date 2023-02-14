from typing import List

import pandas as pd
from database_actions import DatabaseActions, DatabaseActionsFactory
from influxdb_client.client.influxdb_client_async import (
    InfluxDBClient,
    InfluxDBClientAsync,
)


class InfluxdbActions(DatabaseActions):
    """Infuxdb database connection"""

    def make_connection(self, url: str, token: str, org: str, bucket_name: str) -> InfluxDBClientAsync:
        """Makes connection to the Influxdb database"""
        if self._check_if_bucket_exists(url=url, token=token, org=org, bucket_name=bucket_name):
            return InfluxDBClientAsync(url=url, token=token, org=org)

    async def write_data(
        self,
        async_client: InfluxDBClientAsync,
        bucket_name: str,
        records: pd.DataFrame,
        data_frame_measurement_name: str,
        data_frame_tag_columns: List[str],
        data_frame_timestamp_column: str,
    ):
        """Sends data to the database"""
        async with async_client as client:
            await client.write_api().write(
                bucket=bucket_name,
                record=records,
                data_frame_measurement_name=data_frame_measurement_name,
                data_frame_tag_columns=data_frame_tag_columns,
                data_frame_timestamp_column=data_frame_timestamp_column,
            )

    def _check_if_bucket_exists(self, url: str, token: str, org: str, bucket_name: str):
        with InfluxDBClient(url=url, token=token, org=org) as client:
            if client.BucketsAPI().find_bucket_by_name(bucket_name=bucket_name):
                return True


class Influxdb(DatabaseActionsFactory):
    """Infuxdb database connection"""

    def get_database_actions(self) -> DatabaseActions:
        """Makes connection to the Influxdb database"""
        return InfluxdbActions
