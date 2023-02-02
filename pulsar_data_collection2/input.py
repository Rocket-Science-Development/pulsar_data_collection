import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from exceptions import CustomExceptionWithMessage as e
from pydantic import BaseModel, Field, validator


class DatabaseLogin(BaseModel):
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    protocol: str


class DataWithPrediction(BaseModel):
    timestamp: datetime = datetime.utcnow()
    prediction: np.ndarray = Field(...)
    data_points: pd.DataFrame = Field(...)
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


class PulseParameters(BaseModel):
    operation_type: str = Field(...)
    storage_engine: str
    login_url: Optional[DatabaseLogin]
    model_id: str
    model_version: str
    data_id: str
    y_name: Optional[str]
    pred_name: Optional[str]
    other_labels: Optional[List[str]] = None

    @validator("model_id", "model_version", "data_id")
    def check_model_version(cls, value):
        if not value:
            raise ValueError(
                "Required parameters missing. model_id, model_version, and data_id are required for Insert operation."
            )
        return value

    @validator("storage_engine")
    def valid_import(cls, value):
        # Only influxdb is working storage engine rn
        if not value == "influxdb":
            raise e(
                value=value,
                message=f"{value} Storage Engine doesn't exist",
            )
        return value

    # @validator("operation_type")
    # def validate_op_type(cls, value):
    #     if value not in DATABASE_OPERATION_TYPES:
    #         raise e(
    #             value=value,
    #             message=f"{value} is an Invalid Operation type",
    #         )
    #     return value

    @validator("login_url")
    def validate_login_url(cls, value):
        if not value:
            return DatabaseLogin()
        return value
