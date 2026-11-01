"""Masking of personal information in tabular data before it is written (Pulse.capture_data)."""

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .detectors import PiiDetector


class PiiPolicy(BaseModel):
    """What Pulse does with personal information found in the text columns it captures

    Attributes
    ----------
    action: "mask" (default) replaces each finding by its label (e.g. "[SIN]"); "flag" leaves the text unchanged and
    only counts
    columns: text columns to scan; None (default) scans every column holding strings
    types: personal information types to look for; None (default) = all
    min_confidence: findings below are ignored
    count_column: numeric column added with the number of findings per row (None: not added)
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["mask", "flag"] = "mask"
    columns: Optional[list[str]] = None
    types: Optional[list[str]] = None
    min_confidence: float = 0.5
    count_column: Optional[str] = "pii_count"
    _detector: Optional[PiiDetector] = PrivateAttr(default=None)

    @property
    def detector(self) -> PiiDetector:
        if self._detector is None:
            self._detector = PiiDetector(types=self.types, min_confidence=self.min_confidence)
        return self._detector

    def apply(self, frame: pd.DataFrame, count: bool = True) -> pd.DataFrame:
        """A copy of `frame` with personal information masked (or only counted) in its text columns"""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return frame
        columns = (
            self.columns
            if self.columns is not None
            else [c for c in frame.columns if frame[c].map(lambda v: isinstance(v, str)).any()]
        )
        out = frame.copy()
        counts = pd.Series(0, index=frame.index)
        for column in columns:
            if column not in out:
                continue
            masked, found = [], []
            for value in out[column]:
                if isinstance(value, str):
                    findings = self.detector.scan(value)
                    masked.append(self.detector.mask(value, findings) if self.action == "mask" and findings else value)
                    found.append(len(findings))
                else:
                    masked.append(value)
                    found.append(0)
            if self.action == "mask":
                out[column] = pd.Series(masked, index=out.index, dtype=out[column].dtype)  # keeps missing values as they were
            counts += pd.Series(found, index=out.index)
        if count and self.count_column:
            out[self.count_column] = counts.astype("int64")
        return out
