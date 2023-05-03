from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytz
from config import factories
from pydantic import BaseModel, Field, validator


class PulseParameters(BaseModel):

    model_id: str
    model_version: str
    data_id: str
    reference_data_storage: str
    y_name: str
    storage_engine: str
    login: Dict[Any, Union[str, int, bool]]
    features_metadata: Optional[Dict[str, type]]  # key: feature_name, value : feature type
    other_labels: Optional[Dict[Any, Union[str, int, Any]]]
    timezone: str = "UTC"

    @validator("model_id", "model_version", "data_id", "reference_data")
    def check_model_version(cls, value):
        if not value:
            raise ValueError("model_id, model_version, data_id, reference_data missing.")
        return value

    @validator("storage_engine")
    def check_storage_engine(cls, value):
        if value not in factories.keys:
            raise ValueError(f"Storage Engine for {value} not supported")
        return value

    @validator("timezone")
    def check_timezone(cls, value):
        if value not in pytz.timezone:
            raise ValueError("timezone does not exist")
        return value


class DataWithPrediction(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    prediction_id: str
    timestamp: datetime = datetime.now()
    prediction: np.ndarray = Field(..., allow_mutation=False)
    data_points: pd.DataFrame = Field(..., allow_mutation=False)
    features_names: List[str]
