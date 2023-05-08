import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytz
from config import factories
from pydantic import BaseModel, validator


class PulseParameters(BaseModel):

    model_id: str
    model_version: str
    data_id: str
    reference_data_storage: Any  # Union[str, Dict[str, str]]
    target_name: str
    storage_engine: str
    login: Dict[str, Union[str, int, bool]]
    features_metadata: Optional[Dict[str, type]]  # key: feature_name, value : feature type
    other_labels: Optional[Dict[str, Union[str, int, bool]]] = None
    timestamp_column_name: str = "timestamp"

    @validator("storage_engine")
    def check_storage_engine(cls, value):
        if value not in factories:
            raise ValueError(f"Storage Engine for {value} not supported")
        return value


class DataWithPrediction(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    prediction_id: str
    timestamp: datetime.datetime = datetime.datetime.now()
    predictions: Any  # Union[Dict, pd.DataFrame]
    data_points: pd.DataFrame
    features_names: List[str]
    timezone: str = str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)

    @validator("timezone")
    def check_timezone(cls, value):
        if value not in pytz.timezone:
            raise ValueError(f"chose timezone value in {pytz.timezone}")
        return value
