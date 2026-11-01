import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import factories
from .privacy import PiiPolicy


class PulseParameters(BaseModel):
    """
    Model of input Pulse class

    Attributes
    ----------
    model_id: str
    Model identifier
    model_version: str
    Model version
    data_id: str
    Data identifier
    reference_data_storage: Any  # Union[str, Dict[str, str]]
    Storage path for reference data used in drift recognition, usually training dataset
    target_name: str
    Name of target feature
    storage_engine: str
    Storage engine used to store collected logs
    login: Dict[str, Union[str, int, bool]]
    Dictionary containing the element required to perform successful login to storage engine
    features_metadata: Optional[Dict[str, type]] = None  # key: feature_name, value : feature type
    Dictionary containing the schema of features used for the model prediction
    other_labels: Optional[Dict[str, Union[str, int, bool]]] = None
    Dictionary of additional labels used provide more metadata regarding the context surrounding the model
    timestamp_column_name: str = "timestamp"
    Name of the column containing the timestamp

    """

    # Unknown fields are rejected: a misspelt field (e.g. other_labels instead of additional_tags) must not be lost
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_version: str
    data_id: str
    reference_data_storage: Any  # Union[str, Dict[str, str]]
    target_name: str
    storage_engine: str
    login: Dict[str, Union[str, int, bool]]
    features_metadata: Optional[Dict[str, type]] = None  # key: feature_name, value : feature type
    additional_tags: Optional[Dict[str, Union[str, int, bool]]] = None
    timestamp_column_name: str = "timestamp"

    @field_validator("storage_engine")
    @classmethod
    def check_storage_engine(cls, value):
        if value not in factories:
            raise ValueError(f"Storage Engine for {value} not supported")
        return value


class DataWithPrediction(BaseModel):
    """
    Model of input for capture_data method of Pulse parameter class

    Attributes
    ----------

    prediction_id: Optional[str] = None
    Identifier of prediction
    timestamp: datetime.datetime = datetime.datetime.now()
    timestamp of when the prediction have been performed
    predictions: Any  # Union[Dict, pd.DataFrame]
    Object containing the array of predictions
    data_points: pd.DataFrame
    Object containing the data points on which the predicitions were performed
    features_names: List[str]
    list of features that were used to perform the prediction

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    prediction_id: Optional[str] = None
    timestamp: datetime.datetime
    predictions: Any
    data_points: pd.DataFrame
    features_names: List[str]


class CaptureOptions(BaseModel):
    """
    How Pulse writes the captured data

    Attributes
    ----------
    mode: "background" (default) or "sync"
    background: capture_data() only queues the rows (non-blocking); a thread writes them in batches.
    sync: capture_data() writes immediately and raises on failure (notebooks, batch jobs, tests).
    max_queue_rows: int
    Maximum rows waiting in memory; beyond, the oldest batches are dropped and counted (background mode)
    max_batch_rows: int
    Rows per write; a write starts as soon as this many rows are waiting
    flush_interval_s: float
    Maximum time a row waits before being written
    write_timeout_s: float
    Timeout of every request to the storage engine
    max_retry_delay_s: float
    Upper bound of the exponential delay before retrying an unavailable storage engine
    spool_dir: Optional[Path]
    Directory where batches are saved while the storage engine is unavailable, then replayed. Without it, these rows
    are dropped and counted.
    strict_startup: bool
    Raise at creation when the storage engine cannot be reached. By default (background mode) Pulse starts anyway and
    spools or drops until the storage is back. Configuration errors (unknown bucket or organization, rejected token)
    always raise.
    pii: Optional[PiiPolicy]
    Personal information in the text columns: masked (or only counted) before the rows are queued, spooled or
    written. None (default): text is written as captured.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["background", "sync"] = "background"
    max_queue_rows: int = Field(100_000, gt=0)
    max_batch_rows: int = Field(5_000, gt=0)
    flush_interval_s: float = Field(1.0, gt=0)
    write_timeout_s: float = Field(5.0, gt=0)
    max_retry_delay_s: float = Field(60.0, gt=0)
    spool_dir: Optional[Path] = None
    strict_startup: bool = False
    pii: Optional[PiiPolicy] = None
