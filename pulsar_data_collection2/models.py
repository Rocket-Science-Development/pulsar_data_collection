from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from config import factories
from pydantic import BaseModel, Field, validator


class DataWithPrediction(BaseModel):
    timestamp: datetime = datetime.now()
    prediction: np.ndarray = Field(...)
    data_points: pd.DataFrame = Field(...)
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


class PulseParameters(BaseModel):
    model_id: str
    model_version: str
    data_id: str
    reference_data: str
    y_name: Optional[str]
    storage_engine: str
    login: Dict[Any, Union[str, int]]
    other_labels: Dict[Any, Union[str, int, Any]]

    @validator("model_id", "model_version", "data_id", "reference_data")
    def check_model_version(cls, value):
        if not value:
            raise ValueError("model_id, model_version, data_id, reference_data missing.")
        return value

    @validator("storage_engine")
    def check_storage_engine(cls, value):
        if value not in factories:
            raise ValueError(f"Storage Engine for {value} not supported")
        return value
