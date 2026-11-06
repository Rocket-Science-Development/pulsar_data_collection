import gzip
import json
import math
import re
import threading
from typing import Optional

import numpy as np
import pandas as pd
from influxdb_client import BucketsApi, InfluxDBClient, WritePrecision
from influxdb_client.client.write.dataframe_serializer import DataframeSerializer
from influxdb_client.client.write.point import _ESCAPE_KEY, _ESCAPE_MEASUREMENT, _ESCAPE_STRING
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings
from influxdb_client.rest import ApiException
from urllib3.exceptions import HTTPError, MaxRetryError, ProtocolError, ReadTimeoutError

from ..exceptions import (
    StorageConfigurationError,
    StorageConnectionResetError,
    StorageRejectedError,
    StorageUnavailableError,
    StorageWriteTimeoutError,
)
from .database_actions import DatabaseActions, DatabaseActionsFactory

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_ROW_SPACING_NS = 1_000  # 1 µs between the rows of a capture: up to 1 million rows/s per model without running ahead
_DROPPED = re.compile(r"dropped=(\d+)")
_NANOSECONDS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}
_NAT = np.iinfo("int64").min
_CHUNK_ROWS = 200  # rows formatted per call: about 2 ms of GIL
GZIP_LEVEL = 1  # write bodies: 2.3 times smaller line protocol in about 10 ms per 1,000 rows (level 9: 2.6, 65 ms)
_WRITE_PATH = "/api/v2/write"


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


_CUT = (ProtocolError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


def is_connection_cut(error: BaseException) -> bool:
    """The connection was reset or closed by the other end once established, while the request was sent or its answer
    awaited (the rows may have been stored)

    urllib3 reports these as ProtocolError("Connection aborted.", ConnectionResetError / RemoteDisconnected...); a
    refused connection (NewConnectionError) never reached the storage and is not one of them.
    """
    for _ in range(10):
        if error is None:
            return False
        if isinstance(error, (_CUT, StorageConnectionResetError)):
            return True
        if isinstance(error, MaxRetryError) and isinstance(error.reason, _CUT):
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
    if is_connection_cut(error):
        raise StorageConnectionResetError(
            f"Connection to InfluxDB cut before its answer, {rows} rows may or may not be stored: {error}"
        ) from error
    raise error


# ---------------------------------------------------------------------------------------------------------- line protocol
def _literal(text: str) -> str:
    """Text placed as is in a str.format pattern"""
    return text.replace("{", "{{").replace("}", "}}")


def _nanoseconds(column: pd.Series) -> Optional[list]:
    """Timestamps as int nanoseconds since the epoch (naive times are UTC); None if not datetimes or with NaT"""
    if not pd.api.types.is_datetime64_any_dtype(column.dtype):
        return None
    values = column.array
    factor = _NANOSECONDS.get(getattr(values, "unit", None))
    if factor is None:
        return None
    stamps = values.asi8
    if (stamps == _NAT).any():
        return None
    return (stamps * factor).tolist() if factor != 1 else stamps.tolist()


def _kind(dtype) -> str:
    """ "i" integer, "n" float or boolean, "s" anything else (written as text); nullable dtypes like the numpy ones"""
    kind = dtype.type
    return "i" if issubclass(kind, np.integer) else "n" if issubclass(kind, (np.floating, np.bool_)) else "s"


def _numbers(frame: pd.DataFrame, dtypes: list, timestamp: int) -> dict:
    """Values of the columns of a frame whose fields are all numpy numbers or booleans: one conversion for the frame
    (a Series per column costs more than formatting 50 rows); NaN only in float columns"""
    columns = frame.to_numpy(dtype=object).T.tolist()
    out = {}
    for position, (key, dtype, items) in enumerate(zip(frame.columns, dtypes, columns)):
        if position == timestamp:
            continue
        mask = None
        if dtype.kind == "f" and any(map(math.isnan, items)):
            mask = [math.isnan(v) for v in items]
        out[key] = ("i" if dtype.kind in "iu" else "n", items, mask)
    return out


def _any_values(frame: pd.DataFrame, timestamp_column: str) -> dict:
    out = {}
    for key, column in frame.items():
        if key == timestamp_column:
            continue
        array = column.to_numpy(dtype=object)
        mask = pd.isna(array)
        out[key] = (_kind(column.dtype), array.tolist(), mask.tolist() if mask.any() else None)
    return out


def encode_lines(frame: pd.DataFrame, measurement: str, timestamp_column: str, tags: dict) -> Optional[list]:
    """Line protocol of a frame, built column by column; None when the frame needs the generic serializer

    Same lines as influxdb_client's DataframeSerializer (fields and tags sorted by key, missing values left out, rows
    without any field left out, integers suffixed `i`, text quoted and escaped, nanosecond timestamps), but several
    times faster on small frames: the serializer costs milliseconds per frame, whatever its size, and the shared writer
    encodes one frame per model per request. Only the column named `timestamp_column` is the timestamp (the serializer
    also took a column whose name is part of it, such as "time" or "e" for "_time").
    """
    columns = frame.columns.tolist()
    if not all(isinstance(c, str) for c in columns) or len(set(columns)) != len(columns) or timestamp_column not in columns:
        return None
    if len(columns) < 2:
        return None
    stamps = _nanoseconds(frame[timestamp_column])
    if stamps is None:
        return None
    tag_items = sorted((str(k), str(v)) for k, v in tags.items() if k not in columns)  # a column wins over a tag
    if any(v == "" for _, v in tag_items):
        return None
    dtypes = frame.dtypes.tolist()
    timestamp = columns.index(timestamp_column)
    numeric = all(isinstance(t, np.dtype) and t.kind in "fiub" for i, t in enumerate(dtypes) if i != timestamp)
    fields = _numbers(frame, dtypes, timestamp) if numeric else _any_values(frame, timestamp_column)
    head = str(measurement).translate(_ESCAPE_MEASUREMENT)
    head += "".join(f",{k.translate(_ESCAPE_KEY)}={v.translate(_ESCAPE_KEY)}" for k, v in tag_items) + " "
    parts, values, missing = [], [], False
    for key in sorted(fields):
        name = key.translate(_ESCAPE_KEY)
        kind, items, mask = fields[key]
        if kind == "s":
            items = [str(v).translate(_ESCAPE_STRING) for v in items]
            before, after = f',{name}="', '"'
        else:
            before, after = f",{name}=", "i" if kind == "i" else ""
        if mask is None:
            parts.append(_literal(before) + "{}" + _literal(after))
            values.append(items)
        else:  # the whole field, or nothing, per row
            missing = True
            parts.append("{}")
            values.append(["" if m else f"{before}{v}{after}" for v, m in zip(items, mask)])
    values.append(stamps)
    if not missing:
        pattern = _literal(head) + "".join(parts)[1:] + " {}"
        return _formatted(pattern.format, values)
    pattern = "".join(parts) + " {}"
    return [head + line[1:] for line in _formatted(pattern.format, values) if line[0] != " "]


def _formatted(format_row, columns: list) -> list:
    """format_row(*row) for every row, a chunk at a time: one call over thousands of rows would hold the GIL for tens of
    milliseconds, and the threads calling capture_data() would wait as long"""
    rows = len(columns[0])
    if rows <= _CHUNK_ROWS:
        return list(map(format_row, *columns))
    lines = []
    for start in range(0, rows, _CHUNK_ROWS):
        lines.extend(map(format_row, *(column[start : start + _CHUNK_ROWS] for column in columns)))
    return lines


def enable_write_gzip(client: InfluxDBClient, level: int = GZIP_LEVEL) -> InfluxDBClient:
    """gzip the bodies of the write requests of `client` (`Content-Encoding: gzip`, which InfluxDB 2 accepts)

    influxdb-client's own `enable_gzip=True` sets the headers, but compresses at level 9: for line protocol of random
    floats about 7 times the CPU of level 1 (330 ms instead of 48 ms per 5,000 rows of 560 bytes, in the writer
    thread) for bodies 10 % smaller. Here the body of a write is compressed at `level`; query answers are asked
    gzipped as with `enable_gzip` (the HTTP layer decodes them). Returns the client.
    """
    conf = client.conf
    conf.enable_gzip = True  # headers: Content-Encoding gzip on writes, Accept-Encoding gzip on queries

    def update_request_body(path: str, body):
        if path == _WRITE_PATH and body is not None:
            data = body if isinstance(body, bytes) else str(body).encode("utf-8")
            return gzip.compress(data, compresslevel=level)
        return body

    conf.update_request_body = update_request_body
    return client


def _compression(login: dict) -> Optional[str]:
    value = login.get("write_compression")
    if value not in (None, "gzip"):
        raise ValueError(f'write_compression must be None or "gzip", not {value!r}')
    return value


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
        timeout_s = float(login.get("timeout_s", DEFAULT_TIMEOUT_S))
        return (login["url"], login["token"], login["org"], timeout_s, _compression(login))

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

    idempotent_writes = True  # points carry their timestamp: writing them again overwrites them, no duplicate

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
        """Client without connection check; `timeout_s` bounds every request (default 10 s); `write_compression="gzip"`
        compresses the bodies of the writes (see `enable_write_gzip`)"""
        timeout_ms = int(kwargs.get("timeout_s", DEFAULT_TIMEOUT_S) * 1000)
        client = InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"], timeout=timeout_ms)
        if _compression(kwargs) == "gzip":
            enable_write_gzip(client)
        return client

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
        lines = encode_lines(frame, data_frame_measurement_name, data_frame_timestamp_column, tags)
        if lines is not None:
            return lines
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
        write), StorageWriteTimeoutError or StorageConnectionResetError (outcome unknown) or the transport error"""
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
