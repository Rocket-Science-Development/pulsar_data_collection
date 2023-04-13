from influxdb_client import InfluxDBClient
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

from .database_actions import DatabaseActions, DatabaseActionsFactory


class InfluxdbActions(DatabaseActions):
    """Infuxdb database connection"""

    def make_connection(self, **kwargs) -> InfluxDBClientAsync:
        """Makes connection to the Influxdb database
        Parameters
        ----------
        url: str
        token: str
        org: str
        bucket_name: str
        """
        if self._check_if_bucket_exists(url=kwargs.url, token=kwargs.token, org=kwargs.org, bucket_name=kwargs.bucket_name):
            return InfluxDBClientAsync(url=kwargs.url, token=kwargs.token, org=kwargs.org)

    async def write_data(self, **kwargs):
        """Write data to the database
        Parameters
        ----------
        async_client: InfluxDBClientAsync
        bucket_name: str
        records: pd.DataFrame
        data_frame_measurement_name: str
        data_frame_tag_columns: List[str]
        data_frame_timestamp_column: str
        """
        async with kwargs.async_client as client:
            await client.write_api().write(
                bucket=kwargs.bucket_name,
                record=kwargs.records,
                data_frame_measurement_name=kwargs.data_frame_measurement_name,
                data_frame_tag_columns=kwargs.data_frame_tag_columns,
                data_frame_timestamp_column=kwargs.data_frame_timestamp_column,
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
