"""Local spool: batches that could not be written are saved to disk and replayed when the storage is back."""

import json
import os
import uuid
from pathlib import Path
from time import time_ns
from typing import Optional

import pandas as pd


def frame_to_json(frame: pd.DataFrame, meta: dict) -> str:
    """Serialize a batch and its write parameters; dates are stored as int64 nanoseconds (exact round trip)"""
    dtypes = {str(c): str(t) for c, t in frame.dtypes.items()}
    out = frame.reset_index(drop=True).copy()
    for column, dtype in dtypes.items():
        if dtype.startswith("datetime64"):
            out[column] = out[column].astype("int64")
    payload = {"meta": meta, "dtypes": dtypes, "columns": list(dtypes), "data": json.loads(out.to_json(orient="values"))}
    return json.dumps(payload)


def frame_from_json(text: str) -> tuple[pd.DataFrame, dict]:
    """Inverse of frame_to_json"""
    payload = json.loads(text)
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    for column, dtype in payload["dtypes"].items():
        if dtype.startswith("datetime64"):
            values = pd.to_datetime(frame[column].astype("int64"), unit="ns", utc=True)
            timezone = dtype[dtype.index(",") + 1 : -1].strip() if "," in dtype else None
            frame[column] = values.dt.tz_convert(timezone) if timezone else values.dt.tz_localize(None)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame, payload["meta"]


class Spool:
    """Directory of pending batches, replayed oldest first"""

    def __init__(self, directory: os.PathLike):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, frame: pd.DataFrame, meta: dict) -> Path:
        path = self.directory / f"{time_ns()}_{uuid.uuid4().hex[:8]}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(frame_to_json(frame, meta), encoding="utf-8")
        temporary.replace(path)  # atomic: a crash never leaves a half-written batch
        return path

    def files(self) -> list[Path]:
        return sorted(self.directory.glob("*.json"))

    def load(self, path: Path) -> tuple[pd.DataFrame, dict]:
        return frame_from_json(path.read_text(encoding="utf-8"))

    def oldest(self) -> Optional[Path]:
        files = self.files()
        return files[0] if files else None
