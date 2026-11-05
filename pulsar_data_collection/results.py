"""Write analysis results and capture statistics to InfluxDB, so that dashboards can show them.

The analyses of pulsar-metrics return `MetricResults` in memory (drift, fairness, explanations, probes). `ResultsSink`
stores them in one measurement, `pulsar_results`, one point per result:

- tags: `model_id`, `model_version`, `metric_type`, `metric_name`, `feature_name`, `group`, `reference_group`,
  `test_type`, `model_digest` when the sink has a model fingerprint, plus the tags given by the caller (e.g.
  `campaign`, `scenario`);
- fields: `metric_value`, `ci_low`, `ci_high`, `threshold_low`, `threshold_high`, `n`, `e_value`, `status`,
  `suppressed`, `method`, `error`, plus the extra numeric fields of the caller (e.g. `p_adjusted`, `confirmed`).

`status` is the three-way verdict as a number: 1 = drift or disparity established, 0 = none / within the zone,
-1 = inconclusive or not computable. Pulse's own counters go to `pulsar_capture_stats` (`StatsReporter`), and the
optional process health (RSS, threads, open descriptors, CPU, writer lag, write latency) to `pulsar_process_health`.
"""

import logging
import math
import os
import socket
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from .fingerprint import as_fingerprint

logger = logging.getLogger("pulsar_data_collection")

STATUS = {True: 1, False: 0, None: -1}
TAG_FIELDS = ("metric_type", "metric_name", "feature_name", "group", "reference_group", "test_type")
NUMBER_FIELDS = ("metric_value", "ci_low", "ci_high", "n", "e_value")
# optional MetricResults fields (None when not computed): written as floats when present
OPTIONAL_NUMBER_FIELDS = ("n_labelled", "label_coverage", "missing_share_current", "missing_share_reference")
_HANDLED = ("threshold", "drift_status", "suppressed", "method", "error", "status", "reliable", "result_status")
_HANDLED += OPTIONAL_NUMBER_FIELDS


def _as_dict(result: Any) -> dict:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return dict(result)


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def result_point(result: Any, timestamp: datetime, measurement: str = "pulsar_results", tags: Optional[dict] = None) -> Point:
    """One InfluxDB point for one result (MetricResults or dict); None values are left out"""
    r = _as_dict(result)
    point = Point(measurement).time(timestamp, WritePrecision.NS)
    for key, value in (tags or {}).items():
        if value is not None:
            point.tag(key, str(value))
    for key in TAG_FIELDS:
        if r.get(key) not in (None, ""):
            point.tag(key, str(r[key]))
    for key in NUMBER_FIELDS:
        number = _number(r.get(key))
        if number is not None:
            point.field(key, number)
    threshold = r.get("threshold")
    if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
        for key, value in zip(("threshold_low", "threshold_high"), threshold):
            if _number(value) is not None:
                point.field(key, _number(value))
    elif _number(threshold) is not None:
        point.field("threshold", _number(threshold))
    point.field("status", STATUS.get(r.get("drift_status"), -1))
    if r.get("suppressed") is not None:
        point.field("suppressed", bool(r["suppressed"]))
    for key in ("method", "error"):
        if r.get(key):
            point.field(key, str(r[key])[:1000])
    # MetricResults.status (e.g. "constant_reference") is a string reason, kept apart from the numeric verdict `status`
    if r.get("status") not in (None, ""):
        point.field("result_status", str(r["status"])[:200])
    if r.get("reliable") is not None:
        point.field("reliable", bool(r["reliable"]))
    for key in OPTIONAL_NUMBER_FIELDS:  # always floats, so that the field type never changes across writes
        number = _number(r.get(key))
        if number is not None:
            point.field(key, number)
    for key, value in r.items():  # extra numeric fields added by the caller (p_adjusted, confirmed...)
        if key in TAG_FIELDS or key in NUMBER_FIELDS or key in _HANDLED:
            continue
        if isinstance(value, bool):
            point.field(key, value)
        elif _number(value) is not None and not isinstance(value, str):
            point.field(key, _number(value))
    return point


def stats_point(measurement: str, stats: dict, timestamp: datetime, tags: Optional[dict] = None) -> Point:
    """One point of counters: numbers and booleans become fields, the rest is left out"""
    point = Point(measurement).time(timestamp, WritePrecision.NS)
    for key, value in (tags or {}).items():
        if value is not None:
            point.tag(key, str(value))
    for key, value in stats.items():
        if isinstance(value, bool):
            point.field(key, value)
        elif _number(value) is not None and not isinstance(value, str):
            point.field(key, _number(value))
    return point


_PROCESS = None


def process_health(pulses: Iterable = ()) -> dict:
    """Health of the current process: threads, RSS, open descriptors (handles on Windows), CPU, plus the writer lag,
    the queued rows and the write latency of the given Pulse instances

    psutil (optional) gives RSS, descriptors and CPU; without it only the threads and the writer gauges are reported.
    """
    global _PROCESS
    health = {"threads": threading.active_count()}
    try:
        import psutil

        if _PROCESS is None or _PROCESS.pid != os.getpid():
            _PROCESS = psutil.Process()
            _PROCESS.cpu_percent(None)  # the first call only starts the measure
        with _PROCESS.oneshot():
            health["rss_bytes"] = _PROCESS.memory_info().rss
            health["cpu_percent"] = _PROCESS.cpu_percent(None)
            health["os_threads"] = _PROCESS.num_threads()
            if hasattr(_PROCESS, "num_fds"):
                health["open_fds"] = _PROCESS.num_fds()
            elif hasattr(_PROCESS, "num_handles"):
                health["open_handles"] = _PROCESS.num_handles()
    except Exception as error:  # noqa: BLE001 - psutil missing or failing: health reporting is best effort
        logger.debug("Process health without psutil gauges: %s", error)
    lags, latencies, queued, count = [], [], 0, 0
    for pulse in pulses:
        try:
            stats = pulse.stats()
        except Exception as error:  # noqa: BLE001
            logger.debug("Pulse statistics unavailable: %s", error)
            stats = None
        if stats is None:
            continue
        count += 1
        queued += stats.get("queued_rows") or 0
        if stats.get("writer_lag_s") is not None:
            lags.append(stats["writer_lag_s"])
        if stats.get("max_write_ms") is not None:
            latencies.append(stats["max_write_ms"])
    health["pulses"] = count
    health["queued_rows"] = queued
    if lags:
        health["writer_lag_s"] = max(lags)
    if latencies:
        health["max_write_ms"] = max(latencies)
    return health


class ResultsSink:
    """Synchronous writer of analysis results and capture statistics (small volumes: one point per result)

    Parameters
    ----------
    url, token, org, bucket : str
        InfluxDB 2 connection
    measurement : str, optional
        Measurement of the results (default "pulsar_results")
    stats_measurement : str, optional
        Measurement of Pulse's counters (default "pulsar_capture_stats")
    model_fingerprint : ModelFingerprint, dict or "sha256:<hex>", optional
        Digest of the measured model (pulsar_data_collection.fingerprint): every result point is tagged
        `model_digest` = "sha256:<hex>" (a `model_digest` tag given to write_results must agree)
    """

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        measurement: str = "pulsar_results",
        stats_measurement: str = "pulsar_capture_stats",
        timeout_s: float = 10.0,
        model_fingerprint=None,
    ):
        self.bucket = bucket
        self.model_fingerprint = as_fingerprint(model_fingerprint)
        self.org = org
        self.measurement = measurement
        self.stats_measurement = stats_measurement
        self._client = InfluxDBClient(url=url, token=token, org=org, timeout=int(timeout_s * 1000))
        self._write = self._client.write_api(write_options=SYNCHRONOUS)

    def write_results(self, results: Iterable[Any], timestamp: Optional[datetime] = None, tags: Optional[dict] = None) -> int:
        """Write results (MetricResults or dicts) at `timestamp` (default: now), with the given tags; returns the count"""
        timestamp = timestamp or datetime.now(timezone.utc)
        if self.model_fingerprint is not None:
            digest = str(self.model_fingerprint)
            if (tags or {}).get("model_digest", digest) != digest:
                raise ValueError(f"tag model_digest={tags['model_digest']!r} differs from the sink's model fingerprint {digest}")
            tags = {**(tags or {}), "model_digest": digest}
        points = [result_point(r, timestamp, self.measurement, tags) for r in results]
        if points:
            self._write.write(bucket=self.bucket, org=self.org, record=points)
        return len(points)

    def write_stats(self, stats: dict, timestamp: Optional[datetime] = None, tags: Optional[dict] = None) -> None:
        """Write Pulse.stats() counters (numbers and booleans) at `timestamp`"""
        point = stats_point(self.stats_measurement, stats, timestamp or datetime.now(timezone.utc), tags)
        self._write.write(bucket=self.bucket, org=self.org, record=point)

    def write_stats_many(
        self, items: Iterable[tuple], timestamp: Optional[datetime] = None, measurement: Optional[str] = None
    ) -> int:
        """Write several (stats, tags) points in one request; returns the count"""
        timestamp = timestamp or datetime.now(timezone.utc)
        points = [stats_point(measurement or self.stats_measurement, stats, timestamp, tags) for stats, tags in items]
        if points:
            self._write.write(bucket=self.bucket, org=self.org, record=points)
        return len(points)

    def close(self) -> None:
        self._write.close()
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class StatsReporter:
    """Publish the counters of one or several Pulse instances every `interval_s` seconds (one background thread and one
    request per interval for all of them), and once more when stopped

    `StatsReporter(pulse, sink)` keeps the former use; `StatsReporter(None, sink)` then `add(pulse, tags)` makes a
    single reporter for the whole process. `process_health=True` also writes one point of process health (see
    `process_health`) to `health_measurement`, tagged with the host and the pid.
    A failed write is skipped: reporting must never disturb the capture.
    """

    def __init__(
        self,
        pulse=None,
        sink: ResultsSink = None,
        interval_s: float = 10.0,
        tags: Optional[dict] = None,
        process_health: bool = False,
        health_measurement: str = "pulsar_process_health",
    ):
        if sink is None:
            raise ValueError("StatsReporter needs a sink")
        self.pulse = pulse
        self.sink = sink
        self.interval_s = interval_s
        self.common_tags = dict(tags or {})
        self.process_health = process_health
        self.health_measurement = health_measurement
        self.errors = 0
        self._lock = threading.Lock()
        self._pulses: list = []  # (pulse, tags)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pulsar-stats", daemon=True)
        if pulse is not None:
            self.add(pulse)
        self.tags = self._pulses[0][1] if self._pulses else dict(self.common_tags)

    def add(self, pulse, tags: Optional[dict] = None) -> "StatsReporter":
        """Report the counters of `pulse` too, tagged with its model, the common tags and `tags`"""
        merged = {"model_id": pulse.model_id, "model_version": pulse.model_version, **self.common_tags, **(tags or {})}
        with self._lock:
            self._pulses.append((pulse, merged))
        return self

    def remove(self, pulse) -> None:
        with self._lock:
            self._pulses = [(p, t) for p, t in self._pulses if p is not pulse]

    def start(self) -> "StatsReporter":
        self._thread.start()
        return self

    def report(self) -> None:
        with self._lock:
            pulses = list(self._pulses)
        try:
            items = [(pulse.stats(), tags) for pulse, tags in pulses]
            if len(items) > 1 and hasattr(self.sink, "write_stats_many"):
                self.sink.write_stats_many(items)
            else:
                for stats, tags in items:
                    self.sink.write_stats(stats, tags=tags)
        except Exception:  # noqa: BLE001 - reporting never disturbs the capture
            self.errors += 1
        if self.process_health:
            try:
                health = process_health(pulse for pulse, _ in pulses)
                tags = {"host": socket.gethostname(), "pid": os.getpid(), **self.common_tags}
                self.sink.write_stats_many([(health, tags)], measurement=self.health_measurement)
            except Exception:  # noqa: BLE001
                self.errors += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.report()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_s + 5)
        self.report()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
