from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from config import factories
from pydantic import BaseModel, Field, validator


class DatabaseLogin(BaseModel):
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    protocol: str


class DataWithPrediction(BaseModel):
    timestamp: datetime = datetime.now()
    prediction: np.ndarray = Field(...)
    data_points: pd.DataFrame = Field(...)
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


# TODO : add a pointer to reference data file location
class PulseParameters(BaseModel):
    model_id: str
    model_version: str
    data_id: str
    reference_data: str
    y_name: Optional[str]  # TODO y_name and pred_name, same thing?
    pred_name: Optional[str]
    storage_engine: str
    login_info: Optional[DatabaseLogin]  # TODO : change variable name to something more significant
    other_labels: Optional[List[str]] = None  # TODO : single value or key-value?

    @validator("model_id", "model_version", "data_id", "reference_data")
    def check_model_version(cls, value):
        if not value:
            raise ValueError("model_id, model_version, data_id, reference_data missing.")
        return value

    @validator("storage_engine")
    def check_storage_engine(cls, value):
        if value not in factories:
            raise ValueError("Storage Engine not supported")
        return value
