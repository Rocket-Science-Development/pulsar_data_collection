from typing import Tuple

import pandas as pd
from influxdb_client import BucketsApi, InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import PointSettings

from .database_actions import DatabaseActions, DatabaseActionsFactory


class BatchingCallback(object):
    def success(self, conf: Tuple[str], data: str):
        print(f"Written batch: {conf}. \n")

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

    def create_param_dict(self, **kwargs):
        """Create parameter dictionary containing all required information to write to an influx Database
        Parameters
        ----------
        client: InfluxDBClient
        bucket_name: str
        model_id: str
        model_version: str
        data_id: str
        additional_tags: Optional[Dict]

        """

        tags = {"model_id": kwargs["model_id"], "model_version": kwargs["model_version"], "data_id": kwargs["data_id"]}
        if kwargs["additional_tags"]:
            tags.update(**kwargs["additional_tags"])
        return {
            "client": kwargs["client"],
            "bucket_name": kwargs["bucket_name"],
            "data_frame_measurement_name": f"{kwargs['model_id']}_{kwargs['model_version']}_input_data",
            "data_frame_timestamp_column": kwargs["timestamp_column_name"],
            "tags": tags,
        }

    def write_data(self, **kwargs):
        """Write data to the database
        Parameters
        ----------
        kwargs:
            client: InfluxDBClient
            bucket_name: str
            data_frame_measurement_name: str
            data_frame_timestamp_column: str
            tags: Dict[str, str]
            prediction: pd.DataFrame
            data_points: pd.DataFrame
            timestamp: datetime

        """
        db_client = kwargs["client"]

        data = kwargs["data_points"]
        if "prediction" in kwargs.keys():
            data = pd.concat(objs=[data, kwargs["prediction"]], copy=False, axis=1)

        data.loc[:, "_time"] = pd.date_range(
            start=kwargs["timestamp"],
            periods=kwargs["data_points"].shape[0],
            freq="L",
            inclusive="left",
        )
        data.set_index("_time")

        with db_client as client:
            callback = BatchingCallback()
            point_settings = PointSettings()
            for key, value in kwargs["tags"].items():
                point_settings.add_default_tag(key, value)
            with client.write_api(
                success_callback=callback.success,
                error_callback=callback.error,
                retry_callback=callback.retry,
                point_settings=point_settings,
            ) as write_api:
                write_api.write(
                    bucket=kwargs["bucket_name"],
                    record=data,
                    data_frame_measurement_name=kwargs["data_frame_measurement_name"],
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
