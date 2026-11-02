"""Write analysis results and capture statistics to InfluxDB, so that dashboards can show them.

The analyses of pulsar-metrics return `MetricResults` in memory (drift, fairness, explanations, probes). `ResultsSink`
stores them in one measurement, `pulsar_results`, one point per result:

- tags: `model_id`, `model_version`, `metric_type`, `metric_name`, `feature_name`, `group`, `reference_group`,
  `test_type`, plus the tags given by the caller (e.g. `campaign`, `scenario`);
- fields: `metric_value`, `ci_low`, `ci_high`, `threshold_low`, `threshold_high`, `n`, `e_value`, `status`,
  `suppressed`, `method`, `error`, plus the extra numeric fields of the caller (e.g. `p_adjusted`, `confirmed`).

`status` is the three-way verdict as a number: 1 = drift or disparity established, 0 = none / within the zone,
-1 = inconclusive or not computable. Pulse's own counters go to `pulsar_capture_stats` (`StatsReporter`).
"""

import math
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

STATUS = {True: 1, False: 0, None: -1}
TAG_FIELDS = ("metric_type", "metric_name", "feature_name", "group", "reference_group", "test_type")
NUMBER_FIELDS = ("metric_value", "ci_low", "ci_high", "n", "e_value")


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
    for key, value in r.items():  # extra numeric fields added by the caller (p_adjusted, confirmed...)
        if key in TAG_FIELDS or key in NUMBER_FIELDS or key in ("threshold", "drift_status", "suppressed", "method", "error"):
            continue
        if isinstance(value, bool):
            point.field(key, value)
        elif _number(value) is not None and not isinstance(value, str):
            point.field(key, _number(value))
    return point


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
    ):
        self.bucket = bucket
        self.org = org
        self.measurement = measurement
        self.stats_measurement = stats_measurement
        self._client = InfluxDBClient(url=url, token=token, org=org, timeout=int(timeout_s * 1000))
        self._write = self._client.write_api(write_options=SYNCHRONOUS)

    def write_results(self, results: Iterable[Any], timestamp: Optional[datetime] = None, tags: Optional[dict] = None) -> int:
        """Write results (MetricResults or dicts) at `timestamp` (default: now), with the given tags; returns the count"""
        timestamp = timestamp or datetime.now(timezone.utc)
        points = [result_point(r, timestamp, self.measurement, tags) for r in results]
        if points:
            self._write.write(bucket=self.bucket, org=self.org, record=points)
        return len(points)

    def write_stats(self, stats: dict, timestamp: Optional[datetime] = None, tags: Optional[dict] = None) -> None:
        """Write Pulse.stats() counters (numbers and booleans) at `timestamp`"""
        point = Point(self.stats_measurement).time(timestamp or datetime.now(timezone.utc), WritePrecision.NS)
        for key, value in (tags or {}).items():
            point.tag(key, str(value))
        for key, value in stats.items():
            if isinstance(value, bool):
                point.field(key, value)
            elif _number(value) is not None and not isinstance(value, str):
                point.field(key, _number(value))
        self._write.write(bucket=self.bucket, org=self.org, record=point)

    def close(self) -> None:
        self._write.close()
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class StatsReporter:
    """Publish the counters of a Pulse every `interval_s` seconds (background thread), and once more when stopped

    A failed write is skipped: reporting must never disturb the capture.
    """

    def __init__(self, pulse, sink: ResultsSink, interval_s: float = 10.0, tags: Optional[dict] = None):
        self.pulse = pulse
        self.sink = sink
        self.interval_s = interval_s
        self.tags = {"model_id": pulse.model_id, "model_version": pulse.model_version, **(tags or {})}
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pulsar-stats", daemon=True)

    def start(self) -> "StatsReporter":
        self._thread.start()
        return self

    def report(self) -> None:
        try:
            self.sink.write_stats(self.pulse.stats(), tags=self.tags)
        except Exception:  # noqa: BLE001 - reporting never disturbs the capture
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
