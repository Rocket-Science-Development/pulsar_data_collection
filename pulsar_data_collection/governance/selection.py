"""Which data a deletion or an export is about: tags written by Pulse and ResultsSink, and a time period.

Pulse tags every row with `model_id`, `model_version`, `data_id` and the keys of `additional_tags` (by convention
`system_id` for the system the model belongs to); its measurement is `{model_id}_{model_version}_input_data`.
ResultsSink writes `pulsar_results` with the tags given by the caller (`model_id`, `model_version`...). The same
selector gives the InfluxDB delete predicate, the Flux filter, and the matching of spooled batches and evidence records.

The period is inclusive at both ends, like the InfluxDB delete API.
"""

import re
from datetime import datetime
from typing import Dict, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# widest period the InfluxDB delete API accepts (int64 nanoseconds since the epoch)
MIN_TIME = pd.Timestamp("1677-09-21T00:12:44Z")
MAX_TIME = pd.Timestamp("2262-04-11T23:47:16Z")
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FORBIDDEN = ('"', "\\", "\n", "\r", "\t")
EVIDENCE_KEYS = ("system_id", "model_id", "model_version")  # top-level keys of a "pulsar.evidence/1" record

TimeLike = Union[str, datetime, pd.Timestamp, None]


def _check_value(value: str, what: str) -> str:
    value = str(value)
    if not value or any(c in value for c in _FORBIDDEN):
        raise ValueError(f"{what}: empty value or quote, backslash or control character (not expressible in a delete predicate)")
    return value


class DataSelector(BaseModel):
    """Series selected by tag values (all must match); at least one criterion, or `all_series=True` for a period only

    Attributes
    ----------
    system_id, model_id, model_version, data_id: str, optional
    Tags written by Pulse (system_id through additional_tags)
    measurement: str, optional
    Restrict to one measurement (e.g. "credit_model_1.0_input_data" or "pulsar_results")
    tags: dict
    Other tags (e.g. {"environment": "prod"})
    all_series: bool
    Select every series of the period (no tag criterion): must be explicit
    """

    model_config = ConfigDict(extra="forbid")

    system_id: Optional[str] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    data_id: Optional[str] = None
    measurement: Optional[str] = None
    tags: Dict[str, str] = {}
    all_series: bool = False

    @field_validator("system_id", "model_id", "model_version", "data_id", "measurement")
    @classmethod
    def check_value(cls, value, info):
        return None if value is None else _check_value(value, info.field_name)

    @field_validator("tags")
    @classmethod
    def check_tags(cls, tags):
        for key, value in tags.items():
            if not _KEY.match(key):
                raise ValueError(f"tag key {key!r}: letters, digits and underscores only, starting with a letter")
            if key in ("system_id", "model_id", "model_version", "data_id"):
                raise ValueError(f"use the {key} field rather than tags[{key!r}]")
            _check_value(value, f"tag {key}")
        return {key: str(value) for key, value in tags.items()}

    @model_validator(mode="after")
    def check_not_empty(self):
        if not self.criteria() and not self.all_series:
            raise ValueError("no criterion: give a system_id, model_id, ... or all_series=True to select a whole period")
        if self.criteria() and self.all_series:
            raise ValueError("all_series=True selects every series: do not combine it with criteria")
        return self

    def tag_criteria(self) -> Dict[str, str]:
        """Tag key -> value, in a stable order"""
        named = {k: getattr(self, k) for k in ("system_id", "model_id", "model_version", "data_id")}
        return {**{k: v for k, v in named.items() if v is not None}, **dict(sorted(self.tags.items()))}

    def criteria(self) -> Dict[str, str]:
        """Every criterion, the measurement included (key `_measurement`)"""
        out = {"_measurement": self.measurement} if self.measurement else {}
        return {**out, **self.tag_criteria()}

    def predicate(self) -> str:
        """InfluxDB delete predicate (`key="value" AND ...`); empty for all_series"""
        return " AND ".join(f'{key}="{value}"' for key, value in self.criteria().items())

    def flux_filter(self) -> str:
        """Flux filter function selecting the same series as the predicate"""
        terms = [f'r["{key}"] == "{value}"' for key, value in self.criteria().items()]
        return "(r) => " + (" and ".join(terms) if terms else "true")

    def matches(self, tags: dict, measurement: Optional[str] = None) -> bool:
        """Whether a spooled batch (its tags and measurement) is selected"""
        if self.measurement is not None and measurement != self.measurement:
            return False
        return all(str(tags.get(key)) == value for key, value in self.tag_criteria().items())

    def evidence_applicable(self) -> bool:
        """Evidence records carry system_id, model_id and model_version only: other criteria cannot select them"""
        return not self.all_series and not self.measurement and not self.data_id and not self.tags

    def matches_evidence(self, record: dict) -> bool:
        return all(str(record.get(key)) == value for key, value in self.tag_criteria().items())

    def describe(self) -> dict:
        return {"all_series": True} if self.all_series else self.criteria()


def to_timestamp(value: TimeLike, default: pd.Timestamp) -> pd.Timestamp:
    """UTC timestamp (naive times are UTC) clipped to what InfluxDB accepts; `default` for None"""
    if value is None:
        return default
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return min(max(stamp, MIN_TIME), MAX_TIME)


class Period:
    """Inclusive time period [start, stop]; None ends mean the widest period InfluxDB accepts"""

    def __init__(self, start: TimeLike = None, stop: TimeLike = None):
        self.start = to_timestamp(start, MIN_TIME)
        self.stop = to_timestamp(stop, MAX_TIME)
        self.bounded = start is not None or stop is not None
        if self.start > self.stop:
            raise ValueError("start must not be after stop")

    @staticmethod
    def rfc3339(stamp: pd.Timestamp) -> str:
        """RFC 3339 with nanoseconds, in UTC"""
        nanos = stamp.value % 1_000_000_000
        text = stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")
        return f"{text}.{nanos:09d}Z" if nanos else f"{text}Z"

    def flux_range(self) -> str:
        """Flux range() arguments; its stop is exclusive, hence one nanosecond after the inclusive stop"""
        return f"start: {self.rfc3339(self.start)}, stop: {self.rfc3339(self.stop + pd.Timedelta(1, 'ns'))}"

    def contains_ns(self, values) -> "pd.Series":
        """Mask of int nanoseconds (UTC) inside the period"""
        return (values >= self.start.value) & (values <= self.stop.value)

    def overlaps(self, start: TimeLike, end: TimeLike) -> bool:
        """Whether [start, end] intersects the period (an unknown end is open)"""
        low = to_timestamp(start, MIN_TIME)
        high = to_timestamp(end, MAX_TIME)
        return low <= self.stop and high >= self.start

    def describe(self) -> dict:
        return {"start": self.rfc3339(self.start), "stop": self.rfc3339(self.stop), "bounded": self.bounded}
