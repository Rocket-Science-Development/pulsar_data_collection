import importlib
import logging
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from ..db_connectors.influxdb.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_PROTOCOL

from .exceptions import CustomExceptionWithMessage as e


logger = logging.getLogger()

DATABASE_OPERATION_TYPE_INSERT = "INSERT"
DATABASE_OPERATION_TYPE_DELETE = "DELETE"
DATABASE_OPERATION_TYPE_UPDATE = "UPDATE"
DATABASE_OPERATION_TYPE_RETRIEVE = "RETRIEVE"

DATABASE_OPERATION_TYPES = (
    DATABASE_OPERATION_TYPE_INSERT,
    DATABASE_OPERATION_TYPE_DELETE,
    DATABASE_OPERATION_TYPE_UPDATE,
    DATABASE_OPERATION_TYPE_RETRIEVE
)


class DatabaseLogin(BaseModel):
    db_host: str = DB_HOST
    db_port: int = DB_PORT
    db_user: str = DB_USER
    db_password: str = DB_PASSWORD
    db_name: str = DB_NAME
    protocol: str = DB_PROTOCOL
    measurement: Optional[str]


class DataWithPrediction(BaseModel):
    timestamp: datetime = datetime.now()
    prediction: np.ndarray = Field(...)
    data_points: pd.DataFrame = Field(...)
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


class DataCaptureParameters(BaseModel):
    operation_type: str = Field(...)
    storage_engine: str
    login_url: Optional[DatabaseLogin]
    model_id: str = Field(...)
    model_version: str = Field(...)
    data_id: str = Field(...)
    y_name: Optional[str]
    pred_name: Optional[str]
    other_labels: Optional[List[str]] = None

    # Adding a sample validator for checking the value of model_id.
    @validator("model_id")
    def check_model_id(cls, value, values):
        if values.get("operation_type") == DATABASE_OPERATION_TYPE_INSERT:
            if value not in ("RS101", "RS102"):
                raise ValueError("Model ID can only be RS101 or RS102.")
            return value
        return ""

    @validator("model_version", "data_id")
    def check_model_version(cls, value, values):
        if values.get("operation_type") == DATABASE_OPERATION_TYPE_INSERT:
            if not value:
                raise ValueError("Some required parameters were missed. Fields model_id, model_version, and data_id are required for Insert operation.")
            return value
        return ""

    @validator("storage_engine")
    def valid_import(cls, value):
        # Only influxdb is working storage engine rn
        if not value == "influxdb":
            raise e(
                value=value,
                message=f"{value} Storage Engine doesn't exist",
            )
        return value

    @validator("operation_type")
    def validate_op_type(cls, value):
        if value not in DATABASE_OPERATION_TYPES:
            raise e(
                value=value,
                message=f"{value} is an Invalid Operation type",
            )
        return value

    @validator("login_url")
    def validate_login_url(cls, value):
        if not value:
            return DatabaseLogin()
        return value


class DataCapture(DataCaptureParameters):
    def __init__(self, *args, **kwargs):

        return super().__init__(*args, **kwargs)

    def push(self, data=DataWithPrediction):

        """
        Function to convert prediction output to Pandas dataframe to be inserted in DB
        """
        if not self.operation_type in (DATABASE_OPERATION_TYPE_INSERT, ):
            raise Exception(f"Method is only allowed for operation type {DATABASE_OPERATION_TYPE_INSERT}")

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

        DataFactory.sql_ingestion(self.storage_engine, data_with_prediction, self.login_url)
        logger.info("Data was successfully ingested into the db")
        return

    def collect(self):
        """ Function for retrieving Pandas dataframe from the DB
        """
        if not self.operation_type in (DATABASE_OPERATION_TYPE_RETRIEVE, ):
            raise Exception(f"Method is only allowed for operation type {DATABASE_OPERATION_TYPE_RETRIEVE}")

        return DataFactory.sql_digestion(self.storage_engine, self.login_url)


class DataFactory:
    @classmethod
    def get_storage_engine(cls, storage_engine: str):
        """ Returns storage current storage engine"""
        if storage_engine in ("influxdb",):
            from ..db_connectors.influxdb.db_connection import StorageEngine
            return StorageEngine
        else:
            raise e(
                value=storage_engine,
                message=f"{storage_engine} is an Invalid Storage Engine",
            )

    @classmethod
    def sql_ingestion(cls, storage_engine: str, dataframe: pd.DataFrame, database_login: DatabaseLogin):
        """Function to import DB connection based on storage engine and call sql_insertion
        """

        sengine = cls.get_storage_engine(storage_engine)

        sengine().sql_insertion(df=dataframe, database_login=database_login)


    @classmethod
    def sql_digestion(cls, storage_engine: str, database_login: DatabaseLogin=None):
        """ Function to export DB connection based on storage engine and call sql_digestion
        """

        sengine = cls.get_storage_engine(storage_engine)
        return sengine().sql_digestion(database_login=database_login)


    @classmethod
    def imp_module(cls, storage_engine: str):
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
