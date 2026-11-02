import json
import re
import threading
from typing import Optional

import numpy as np
import pandas as pd
from influxdb_client import BucketsApi, InfluxDBClient, WritePrecision
from influxdb_client.client.write.dataframe_serializer import DataframeSerializer
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings
from influxdb_client.rest import ApiException
from urllib3.exceptions import HTTPError, MaxRetryError, ReadTimeoutError

from ..exceptions import StorageConfigurationError, StorageRejectedError, StorageUnavailableError, StorageWriteTimeoutError
from .database_actions import DatabaseActions, DatabaseActionsFactory

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_ROW_SPACING_NS = 1_000  # 1 µs between the rows of a capture: up to 1 million rows/s per model without running ahead
_DROPPED = re.compile(r"dropped=(\d+)")


def _dropped(body) -> Optional[int]:
    """Number of refused points reported by an InfluxDB partial write (HTTP 422 "... dropped=N"), None if absent"""
    match = _DROPPED.search(str(body or ""))
    return int(match.group(1)) if match else None


def is_read_timeout(error: BaseException) -> bool:
    """The request was sent and no answer came back in time (the rows may have been stored)"""
    for _ in range(10):
        if error is None:
            return False
        if isinstance(error, (ReadTimeoutError, StorageWriteTimeoutError)):
            return True
        if isinstance(error, MaxRetryError) and isinstance(error.reason, ReadTimeoutError):
            return True
        error = error.__cause__ or error.__context__
    return False


def _raise_write_error(error: Exception, rows: int):
    """Translate a write failure: refused data (never retryable), timeout (outcome unknown) or unavailable storage"""
    if isinstance(error, ApiException) and error.status in (400, 422):
        # the data itself is refused (field type conflict...): never retryable
        raise StorageRejectedError(
            f"InfluxDB rejected the batch ({error.status}): {error.body}",
            dropped=_dropped(error.body) if error.status == 422 else None,
        ) from error
    if is_read_timeout(error):
        raise StorageWriteTimeoutError(
            f"InfluxDB did not answer in time, {rows} rows may or may not be stored: {error}"
        ) from error
    raise error


class _SharedClients:
    """Process-wide InfluxDB clients, one per (url, token, org, timeout), shared by the Pulse instances that ask for it

    One HTTP connection pool instead of one per Pulse, closed when its last user releases it. The bucket check is also
    done once per process and bucket.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: dict = {}  # key -> [client, users]
        self._checked: set = set()

    @staticmethod
    def key(**login) -> tuple:
        return (login["url"], login["token"], login["org"], float(login.get("timeout_s", DEFAULT_TIMEOUT_S)))

    def acquire(self, **login) -> InfluxDBClient:
        key = self.key(**login)
        with self._lock:
            entry = self._clients.get(key)
            if entry is None:
                entry = self._clients[key] = [InfluxdbActions.create_client(**login), 0]
            entry[1] += 1
            return entry[0]

    def release(self, client) -> None:
        with self._lock:
            for key, entry in list(self._clients.items()):
                if entry[0] is client:
                    entry[1] -= 1
                    if entry[1] <= 0:
                        del self._clients[key]
                        client.close()
                    return
        client.close()  # not a shared client

    def checked(self, **login) -> bool:
        with self._lock:
            return (*self.key(**login), login["bucket_name"]) in self._checked

    def mark_checked(self, **login) -> None:
        with self._lock:
            self._checked.add((*self.key(**login), login["bucket_name"]))


shared_clients = _SharedClients()


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

    def make_shared_connection(self, **kwargs) -> InfluxDBClient:
        """Like make_connection, but the client (and its HTTP connection pool) is shared by every caller of the process
        with the same url, token, org and timeout, and the bucket is checked once per process. Release it with
        `release_connection`.
        """
        if not shared_clients.checked(**kwargs):
            self.make_connection(**kwargs).close()
            shared_clients.mark_checked(**kwargs)
        return shared_clients.acquire(**kwargs)

    @staticmethod
    def create_shared_client(**kwargs) -> InfluxDBClient:
        """Shared client without connection check (storage unavailable at startup)"""
        return shared_clients.acquire(**kwargs)

    @staticmethod
    def release_connection(client) -> None:
        """Close a client, or release a shared one (closed with its last user)"""
        shared_clients.release(client)

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
    def prepare_frame(
        data_points: pd.DataFrame, prediction: pd.DataFrame = None, timestamp=None, row_spacing_ns: int = DEFAULT_ROW_SPACING_NS
    ) -> pd.DataFrame:
        """Rows to write: features and predictions side by side, row i timestamped `timestamp + i * row_spacing_ns`

        The default spacing is 1 µs. It used to be 1 ms, so that a model writing more than 1000 rows/s ran ahead of the
        clock (queries stopping at now() missed its rows). Always a new frame: the caller's DataFrames are not modified.
        """
        frame = pd.concat(objs=[data_points, prediction], axis=1) if prediction is not None else data_points.copy()
        offsets = pd.to_timedelta(np.arange(frame.shape[0], dtype="int64") * int(row_spacing_ns), unit="ns")
        frame["_time"] = pd.DatetimeIndex(pd.Timestamp(timestamp) + offsets)
        return frame

    def supports_lines(self) -> bool:
        """Line protocol writes (several batches, of several models, in one request) are used unless a subclass replaced
        write_frame, whose behaviour is then kept"""
        return type(self).write_frame is InfluxdbActions.write_frame

    @staticmethod
    def serialize_frame(
        frame: pd.DataFrame, data_frame_measurement_name: str, data_frame_timestamp_column: str, tags: dict, **_
    ) -> list:
        """Line protocol of prepared rows (nanosecond timestamps); rows without any field are left out"""
        point_settings = PointSettings(**{key: str(value) for key, value in tags.items()})
        return DataframeSerializer(
            frame,
            point_settings,
            precision=WritePrecision.NS,
            data_frame_measurement_name=data_frame_measurement_name,
            data_frame_timestamp_column=data_frame_timestamp_column,
        ).serialize()

    @staticmethod
    def write_lines(client: InfluxDBClient, bucket_name: str, lines: list) -> None:
        """One synchronous request for lines of any measurement; raises StorageRejectedError (with `dropped` on a partial
        write), StorageWriteTimeoutError (outcome unknown) or the transport error"""
        if not lines:
            return
        with client.write_api(write_options=SYNCHRONOUS) as write_api:
            try:
                write_api.write(bucket=bucket_name, record="\n".join(lines), write_precision=WritePrecision.NS)
            except Exception as error:  # noqa: BLE001 - translated
                _raise_write_error(error, len(lines))

    @staticmethod
    def ping(client: InfluxDBClient) -> bool:
        """Cheap availability probe (GET /ping), never raises"""
        try:
            return bool(client.ping())
        except Exception:  # noqa: BLE001
            return False

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
            except Exception as error:  # noqa: BLE001 - translated
                _raise_write_error(error, len(frame))

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
