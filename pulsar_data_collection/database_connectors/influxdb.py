import json

import pandas as pd
from influxdb_client import BucketsApi, InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings
from influxdb_client.rest import ApiException
from urllib3.exceptions import HTTPError

from ..exceptions import StorageConfigurationError, StorageRejectedError, StorageUnavailableError
from .database_actions import DatabaseActions, DatabaseActionsFactory

DEFAULT_TIMEOUT_S = 10.0


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

        Raises
        ------
        StorageConfigurationError
            if the bucket or the organization does not exist, or the token is rejected
        StorageUnavailableError
            if InfluxDB cannot be reached
        """
        timeout_s = kwargs.get("timeout_s", DEFAULT_TIMEOUT_S)
        self._check_if_bucket_exists(
            url=kwargs["url"], token=kwargs["token"], org=kwargs["org"], bucket_name=kwargs["bucket_name"], timeout_s=timeout_s
        )
        return self.create_client(**kwargs)

    @staticmethod
    def create_client(**kwargs) -> InfluxDBClient:
        """Client without connection check; `timeout_s` bounds every request (default 10 s)"""
        timeout_ms = int(kwargs.get("timeout_s", DEFAULT_TIMEOUT_S) * 1000)
        return InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"], timeout=timeout_ms)

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
            "bucket_name": kwargs["login"]["bucket_name"],
            "data_frame_measurement_name": f"{kwargs['model_id']}_{kwargs['model_version']}_input_data",
            "data_frame_timestamp_column": kwargs["timestamp_column_name"],
            "tags": tags,
        }

    @staticmethod
    def prepare_frame(data_points: pd.DataFrame, prediction: pd.DataFrame = None, timestamp=None) -> pd.DataFrame:
        """Rows to write: features and predictions side by side, row i timestamped `timestamp + i` milliseconds

        Always a new frame: the caller's DataFrames are not modified.
        """
        frame = pd.concat(objs=[data_points, prediction], axis=1) if prediction is not None else data_points.copy()
        # start + periods only: `inclusive="left"` drops the last timestamp under pandas 3
        frame["_time"] = pd.date_range(start=timestamp, periods=frame.shape[0], freq="ms")
        return frame

    @staticmethod
    def write_frame(
        client: InfluxDBClient,
        frame: pd.DataFrame,
        bucket_name: str,
        data_frame_measurement_name: str,
        data_frame_timestamp_column: str,
        tags: dict,
    ) -> None:
        """Synchronous write; raises on failure (bounded by the client timeout, no hidden retries)"""
        point_settings = PointSettings(**{key: str(value) for key, value in tags.items()})
        with client.write_api(write_options=SYNCHRONOUS, point_settings=point_settings) as write_api:
            try:
                write_api.write(
                    bucket=bucket_name,
                    record=frame,
                    data_frame_measurement_name=data_frame_measurement_name,
                    data_frame_timestamp_column=data_frame_timestamp_column,
                )
            except ApiException as e:
                if e.status in (400, 422):  # the data itself is refused (field type conflict...): never retryable
                    raise StorageRejectedError(f"InfluxDB rejected the batch ({e.status}): {e.body}") from e
                raise

    def write_data(self, **kwargs):
        """Write one capture synchronously; raises on failure

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
        frame = self.prepare_frame(kwargs["data_points"], kwargs.get("prediction"), kwargs["timestamp"])
        self.write_frame(
            client=kwargs["client"],
            frame=frame,
            bucket_name=kwargs["bucket_name"],
            data_frame_measurement_name=kwargs["data_frame_measurement_name"],
            data_frame_timestamp_column=kwargs["data_frame_timestamp_column"],
            tags=kwargs["tags"],
        )

    def _check_if_bucket_exists(
        self, url: str, token: str, org: str, bucket_name: str, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> bool:
        """Check that the bucket exists in the organization and that the token can access it

        Raises
        ------
        StorageConfigurationError
            with the reason given by InfluxDB (unknown bucket, unknown organization, unauthorized token)
        """
        with InfluxDBClient(url=url, token=token, org=org, timeout=int(timeout_s * 1000)) as connection:
            try:
                buckets = BucketsApi(connection).find_buckets(name=bucket_name, org=org).buckets
            except (HTTPError, OSError) as e:
                raise StorageUnavailableError(f"InfluxDB at {url} cannot be reached: {e}") from e
            except ApiException as e:
                raise StorageConfigurationError(
                    f"InfluxDB at {url} rejected org '{org}' / bucket '{bucket_name}': {e.status} {_api_error_message(e)}"
                ) from e
        if not buckets:
            raise StorageConfigurationError(f"Bucket '{bucket_name}' not found in organization '{org}' at {url}")
        return True


def _api_error_message(error: ApiException) -> str:
    """Message returned by the InfluxDB API, or the HTTP reason"""
    try:
        return json.loads(error.body).get("message") or error.reason
    except (TypeError, ValueError, AttributeError):
        return str(error.reason)


class Influxdb(DatabaseActionsFactory):
    """Infuxdb database connection"""

    def get_database_actions(self) -> DatabaseActions:
        """Makes connection to the Influxdb database"""
        return InfluxdbActions()
