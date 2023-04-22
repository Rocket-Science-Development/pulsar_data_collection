from typing import Tuple

from influxdb_client import BucketsApi, InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError

from .database_actions import DatabaseActions, DatabaseActionsFactory

# from influxdb_client.client.write_api import SYNCHRONOUS


class BatchingCallback(object):
    def success(self, conf: Tuple[str], data: str):
        print(f"Written batch: {conf}")

    def error(self, conf: Tuple[str], data: str, exception: InfluxDBError):
        print(f"Cannot write batch: {conf} \n due: {exception}")
        raise exception

    def retry(self, conf: Tuple[str], data: str, exception: InfluxDBError):
        print(f"Retryable error occurs for batch: {conf} \n retry: {exception}")


class InfluxdbActions(DatabaseActions):
    """Infuxdb database connection"""

    def make_connection(self, **kwargs) -> InfluxDBClient:
        """Makes connection to the Influxdb database
        Parameters
        ----------
        url: str
        token: str
        org: str
        bucket_name: str
        """
        if self._check_if_bucket_exists(
            url=kwargs["url"], token=kwargs["token"], org=kwargs["org"], bucket_name=kwargs["bucket_name"]
        ):
            return InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"])

    def write_data(self, **kwargs):
        """Write data to the database
        Parameters
        ----------
        client: InfluxDBClient
        bucket_name: str
        records: pd.DataFrame
        data_frame_measurement_name: str
        data_frame_tag_columns: List[str]
        data_frame_timestamp_column: str
        """
        db_client = kwargs["client"]
        with db_client as client:
            callback = BatchingCallback()
            with client.write_api(
                success_callback=callback.success,
                error_callback=callback.error,
                retry_callback=callback.retry,
            ) as write_api:
                write_api.write(
                    bucket=kwargs["bucket_name"],
                    record=kwargs["records"],
                    data_frame_measurement_name=kwargs["data_frame_measurement_name"],
                    data_frame_tag_columns=kwargs["data_frame_tag_columns"],
                    data_frame_timestamp_column=kwargs["data_frame_timestamp_column"],
                )

    def _check_if_bucket_exists(self, url: str, token: str, org: str, bucket_name: str):
        connection = InfluxDBClient(url=url, token=token, org=org)
        if BucketsApi(connection).find_bucket_by_name(bucket_name=bucket_name):
            return True


class Influxdb(DatabaseActionsFactory):
    """Infuxdb database connection"""

    def get_database_actions(self) -> DatabaseActions:
        """Makes connection to the Influxdb database"""
        return InfluxdbActions()
