"""Data control policy: where the data lives and how long it is kept, declared by the client in a file or a dict.

```yaml
influxdb:
  url: http://influxdb.ministry.internal:8086
  org: ministry
  token_env: PULSAR_INFLUX_TOKEN     # the token stays in the environment, never in this file
buckets:                             # role -> bucket; retention: "90d", "12h", "2w", "1y", seconds or "infinite"
  raw: {name: captures, retention: 90d, content: raw captured rows (features, predictions)}
  results: {name: results, retention: 2y, content: analysis results and capture statistics}
spool_dirs: [/var/lib/pulsar/spool]
outbox_dirs: [/var/lib/pulsar/outbox]
journal: /var/lib/pulsar/governance/journal.jsonl
export_dir: /var/lib/pulsar/exports
central_service: https://governance.example/api/evidence
```
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MIN_RETENTION_S = 3600  # InfluxDB refuses a retention period shorter than one hour
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400, "y": 365 * 86400}
_DURATION = re.compile(r"^\s*(\d+)\s*([smhdwy])\s*$")
_INFINITE = ("infinite", "forever", "inf", "none")
_SECRET_KEYS = ("token", "password", "secret", "authorization")


def parse_duration(value: Union[str, int, None]) -> Optional[int]:
    """Seconds of a duration: "90d", "12h", "30m", "2w", "1y" (365 days), an int of seconds; 0 for "infinite"

    None stays None (not managed). Raises ValueError for anything else.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid duration: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"invalid duration: {value!r}")
        return value
    text = str(value).strip().lower()
    if text in _INFINITE or text == "0":
        return 0
    if text.isdigit():
        return int(text)
    match = _DURATION.match(text)
    if not match:
        raise ValueError(f"invalid duration {value!r}: use e.g. 90d, 12h, 2w, 1y, seconds or 'infinite'")
    return int(match.group(1)) * _UNITS[match.group(2)]


def format_duration(seconds: Optional[int]) -> Optional[str]:
    """Readable form of a number of seconds: 7776000 -> "90d", 0 -> "infinite" """
    if seconds is None:
        return None
    if seconds == 0:
        return "infinite"
    for unit in ("y", "w", "d", "h", "m"):
        if seconds % _UNITS[unit] == 0:
            return f"{seconds // _UNITS[unit]}{unit}"
    return f"{seconds}s"


def default_shard_group_s(retention_s: int) -> int:
    """Shard group duration InfluxDB picks for a retention period: 1 h below 2 days, 1 day below 6 months, 7 days
    beyond (and for an infinite retention)"""
    if retention_s == 0 or retention_s >= 180 * 86400:
        return 7 * 86400
    if retention_s >= 2 * 86400:
        return 86400
    return 3600


class BucketPolicy(BaseModel):
    """One bucket and its declared retention

    Attributes
    ----------
    name: str
    Bucket name in the client's InfluxDB
    retention: str or int, optional
    Declared retention ("90d", "12h", "2w", "1y", seconds, or "infinite"); None: not managed by the policy
    shard_group_duration: str or int, optional
    Shard group duration to set with the retention; default: the one InfluxDB would pick (see default_shard_group_s)
    content: str
    What the bucket holds, for describe()
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    retention: Optional[Union[str, int]] = None
    shard_group_duration: Optional[Union[str, int]] = None
    content: str = ""

    @field_validator("retention", "shard_group_duration")
    @classmethod
    def check_duration(cls, value):
        seconds = parse_duration(value)
        if seconds is not None and 0 < seconds < MIN_RETENTION_S:
            raise ValueError(f"{value!r} is shorter than one hour, the minimum accepted by InfluxDB")
        return value

    @property
    def retention_s(self) -> Optional[int]:
        return parse_duration(self.retention)

    @property
    def shard_group_s(self) -> Optional[int]:
        if self.retention_s is None:
            return None
        explicit = parse_duration(self.shard_group_duration)
        return explicit if explicit else default_shard_group_s(self.retention_s)


class InfluxdbLocation(BaseModel):
    """The client's InfluxDB; the token is read from the environment variable `token_env`"""

    model_config = ConfigDict(extra="forbid")

    url: str
    org: str
    token_env: str = "PULSAR_INFLUX_TOKEN"
    timeout_s: float = Field(30.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def refuse_secrets(cls, data):
        if isinstance(data, dict) and any(key.lower() in _SECRET_KEYS for key in data):
            raise ValueError("no secret in the policy: put the token in the environment variable named by token_env")
        return data


class GovernancePolicy(BaseModel):
    """Data control policy of one deployment (see the module docstring for a file example)

    Attributes
    ----------
    influxdb: InfluxdbLocation
    buckets: dict
    Role ("raw", "results"...) -> BucketPolicy
    spool_dirs, outbox_dirs: list of paths
    Spool directories of the Pulse instances (CaptureOptions.spool_dir) and directories of the EvidenceOutbox stores
    journal: path
    Append-only audit journal of the retention changes, deletions and exports (JSON lines, hash-chained)
    journal_key_env: str, optional
    Environment variable holding a key: the journal entries are then chained with HMAC-SHA256 instead of SHA-256
    export_dir: path, optional
    Default directory of the exports
    central_service: str, optional
    URL of the service receiving the evidence records (what leaves the client), for describe()
    """

    model_config = ConfigDict(extra="forbid")

    influxdb: InfluxdbLocation
    buckets: Dict[str, BucketPolicy] = Field(min_length=1)
    spool_dirs: List[Path] = []
    outbox_dirs: List[Path] = []
    journal: Path
    journal_key_env: Optional[str] = None
    export_dir: Optional[Path] = None
    central_service: Optional[str] = None

    def bucket(self, role: str) -> BucketPolicy:
        if role not in self.buckets:
            raise KeyError(f"unknown bucket role {role!r}; declared: {', '.join(self.buckets)}")
        return self.buckets[role]


def load_policy(source: Union[GovernancePolicy, dict, str, os.PathLike]) -> GovernancePolicy:
    """Policy from a GovernancePolicy, a dict, or a JSON or YAML file (YAML needs PyYAML)

    Relative paths of a file are relative to the file's directory.
    """
    if isinstance(source, GovernancePolicy):
        return source
    if isinstance(source, dict):
        return GovernancePolicy.model_validate(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("a YAML policy needs PyYAML (pip install pyyaml), or use a JSON file") from error
        data: Any = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the policy must be a mapping")
    base = path.parent
    for key in ("journal", "export_dir"):
        if data.get(key) and not Path(data[key]).is_absolute():
            data[key] = str(base / data[key])
    for key in ("spool_dirs", "outbox_dirs"):
        data[key] = [str(p) if Path(p).is_absolute() else str(base / p) for p in data.get(key) or []]
    return GovernancePolicy.model_validate(data)
