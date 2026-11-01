import logging
import threading
from typing import Optional

import pandas as pd

from .config import factories
from .exceptions import StorageUnavailableError
from .models import CaptureOptions, DataWithPrediction, PulseParameters
from .spool import Spool
from .writer import BackgroundWriter

logger = logging.getLogger("pulsar_data_collection")


class Pulse:
    """
    This class expose methods in order to collect data from
    an inference container/webapp

    By default capture_data() does not wait for the storage engine: rows are queued and written by a background thread,
    so a slow or unavailable database does not slow down predictions. See CaptureOptions.
    """

    def __init__(self, data: PulseParameters, options: Optional[CaptureOptions] = None):
        """
        Initializing Pulse class instance

        Parameters
        ----------

        data : PulseParameters
        Pydantic Model providing the interface to the class constructor,
        Refer to PulseParameters model attributes for detailed list of inputs

        options : CaptureOptions, optional
        How the data is written (background or sync, queue size, spool directory...)

        Raises
        ------
        StorageConfigurationError
        unknown bucket or organization, rejected token
        StorageUnavailableError
        storage engine unreachable, only in sync mode or with strict_startup=True
        """
        self.options = options or CaptureOptions()
        self.model_id = data.model_id
        self.model_version = data.model_version
        self.reference_data_storage = data.reference_data_storage
        self.target_name = data.target_name
        factory = factories.get(data.storage_engine)
        self.storage_engine = factory.get_database_actions()
        login = {**data.login, "timeout_s": self.options.write_timeout_s}
        try:
            self.db_connection = self.storage_engine.make_connection(**login)
        except StorageUnavailableError as error:
            if self.options.mode == "sync" or self.options.strict_startup:
                raise
            logger.warning("Storage engine unavailable at startup, capturing anyway: %s", error)
            self.db_connection = self.storage_engine.create_client(**login)
        self.additional_tags = data.additional_tags
        self.params = self.storage_engine.create_param_dict(client=self.db_connection, **data.model_dump())
        self._write_params = {key: value for key, value in self.params.items() if key != "client"}

        self._writer = None
        self._closed = False
        self._last_time = None
        self._time_lock = threading.Lock()
        if self.options.mode == "background":
            self._writer = BackgroundWriter(
                write=self._write,
                meta=self._write_params,
                max_queue_rows=self.options.max_queue_rows,
                max_batch_rows=self.options.max_batch_rows,
                flush_interval_s=self.options.flush_interval_s,
                max_retry_delay_s=self.options.max_retry_delay_s,
                spool=Spool(self.options.spool_dir) if self.options.spool_dir else None,
            )

    def _write(self, frame, write_params: dict):
        self.storage_engine.write_frame(client=self.db_connection, frame=frame, **write_params)

    def capture_data(self, data: DataWithPrediction):
        """
        Capturing data from inference code

        Background mode: queues the rows and returns immediately; never raises because of the storage engine.
        Sync mode: writes immediately and raises on failure.
        """
        if self._closed:
            raise RuntimeError("Pulse is closed")
        data_points, predictions = data.data_points, data.predictions
        if self.options.pii is not None:
            # masked here, in the caller's thread: raw values never reach the queue nor the spool directory
            data_points = self.options.pii.apply(data_points)
            predictions = self.options.pii.apply(predictions, count=False)
        frame = self._unique_times(self.storage_engine.prepare_frame(data_points, predictions, data.timestamp))
        if self._writer is None:
            self._write(frame, self._write_params)
        else:
            self._writer.submit(frame)

    def _unique_times(self, frame):
        """Shift a batch so that its timestamps follow the previous batch of this Pulse

        InfluxDB keeps one point per (measurement, tags, timestamp): two captures with the same timestamp would overwrite
        each other. System clocks are coarse (15.6 ms on Windows) and concurrent requests share timestamps, so each Pulse
        keeps its timestamps strictly increasing; a batch starting at or before the previous one is moved just after it.
        """
        if frame.empty or "_time" not in frame:
            return frame
        with self._time_lock:
            first = frame["_time"].iloc[0]
            if self._last_time is not None and first <= self._last_time:
                frame["_time"] = frame["_time"] + (self._last_time - first + pd.Timedelta(1, "ns"))
            self._last_time = frame["_time"].iloc[-1]
        return frame

    def flush(self, timeout_s: float = 30.0) -> bool:
        """Wait until the queued rows are written, spooled or dropped (background mode); True if done in time"""
        return self._writer.flush(timeout_s) if self._writer and not self._closed else True

    def stats(self) -> dict:
        """Completeness counters: captured, written, spooled, replayed and dropped rows, write errors, queue size

        Still available after close(), with the final values.
        """
        if self._writer is None:
            return {"mode": "sync"}
        return {"mode": "background", **self._writer.stats()}

    def close(self, timeout_s: float = 30.0):
        """Write (or spool) the pending rows, then close the connection to the storage engine"""
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.close(timeout_s)
        self.db_connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
