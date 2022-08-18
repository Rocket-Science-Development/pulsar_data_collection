import importlib
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

# from ..db_connectors.influxdb.db_connection import StorageEngine
from .exceptions import CustomExceptionWithMessage as e

# TODO : aadd validators for both models


class DatabaseLogin(BaseModel):
    dbhost: str = "localhost"
    dbport: int
    dbuser: str
    dbpassword: str
    dbname: str
    protocol: str
    measurement: str


class DataWithPrediction(BaseModel):
    timestamp: datetime = datetime.now()
    prediction: np.ndarray = Field(...)
    data_points: pd.DataFrame = Field(...)
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


class DataCaptureParameters(BaseModel):
    storage_engine: str
    login_url: Optional[List[DatabaseLogin]]
    model_id: str = Field(...)
    model_version: str = Field(...)
    data_id: str = Field(...)
    y_name: Optional[str]
    pred_name: Optional[str]
    operation_type: str = Field(...)
    other_labels: Optional[List[str]] = None

    # Adding a sample validator for checking the value of model_id.
    @validator("model_id")
    def check_model_id(cls, value):
        if value not in ("RS101", "RS102"):
            raise ValueError("Model ID can only be RS101 or RS102.")
        return value

    @validator("storage_engine")
    def valid_import(cls, value):
        if value == "":
            raise e(
                value=value,
                message=f"{value} Storage Engine doesn't exist",
            )
        return value

    @validator("operation_type")
    def validate_op_type(cls, value):
        if value not in ("INS", "DEL", "MOD"):
            raise e(
                value=value,
                message=f"{value} is an Invalid Operation type",
            )
        return value


class DataCapture(DataCaptureParameters):
    def __init__(self, *args, **kwargs):

        return super().__init__(*args, **kwargs)

    def collect(self, data=DataWithPrediction):

        """
        Function to convert prediction output to Pandas dataframe to be inserted in DB
        """

        pred = pd.DataFrame(data.prediction, columns=["target"])
        data_with_prediction = pd.concat([data.data_points, pred], axis=1)

        data_with_prediction.loc[:, "Timestamp"] = data.timestamp
        data_with_prediction.loc[:, "model_id"] = self.model_id
        data_with_prediction.loc[:, "model_version"] = self.model_version
        data_with_prediction.loc[:, "data_id"] = self.data_id

        if self.y_name:
            data_with_prediction.loc[:, "y_name"] = self.y_name
        if self.pred_name:
            data_with_prediction.loc[:, "pred_name"] = self.pred_name

        # for label in data.other_labels:
        #     data_with_prediction.loc[:, f"{label}"] = label

        if self.operation_type in ("INS"):
            DataFactory.sql_ingestion(self.storage_engine, data_with_prediction)

        return


class DataFactory:
    @staticmethod
    def sql_ingestion(storage_engine: str, dataframe: pd.DataFrame):
        """Function to import DB connection based on storage engine and call sql_insertion"""

        if storage_engine in ("influxdb"):
            from ..db_connectors.influxdb.db_connection import StorageEngine
        else:
            raise e(
                value=storage_engine,
                message=f"{storage_engine} is an Invalid Storage Engine",
            )

        StorageEngine().sql_insertion(df=dataframe)

    @staticmethod
    def imp_module(storage_engine: str):
        """Function to import different DB connections based on StorageEngine passed in input."""

        module = importlib.import_module(storage_engine)
        mod_nm = module.__name__

        if mod_nm in ("influxdb"):
            name = "pulsar_data_collection.db_connectors.influxdb.db_connection"
            if (spec := importlib.util.find_spec(name)) is not None:

                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                cls = getattr(module, "StorageEngine")

            else:
                print(f"can't find the {name!r} module")
        else:
            raise e(
                value=mod_nm,
                message=f"{mod_nm} is an Invalid Storage Engine",
            )

        return cls
