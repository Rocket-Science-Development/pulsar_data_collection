import importlib
import logging
import sys
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from ..db_connectors.influxdb.config import (  # DB_NAME,
    DB_EVAL_TIMESTAMP_MEASURMENT,
    DB_HOST,
    DB_METRICS_MEASURMENT,
    DB_PASSWORD,
    DB_PORT,
    DB_PREDICTION_MEASURMENT,
    DB_PROTOCOL,
    DB_USER,
)
from .exceptions import CustomExceptionWithMessage as e

logger = logging.getLogger()

DATABASE_OPERATION_TYPE_INSERT_PREDICTION = "INSERT_PREDICTION"
DATABASE_OPERATION_TYPE_METRICS = "METRICS"

DATABASE_OPERATION_TYPES = (DATABASE_OPERATION_TYPE_INSERT_PREDICTION, DATABASE_OPERATION_TYPE_METRICS)


class DatabaseLogin(BaseModel):
    db_host: str = DB_HOST
    db_port: int = DB_PORT
    db_user: str = DB_USER
    db_password: str = DB_PASSWORD
    protocol: str = DB_PROTOCOL


class DataWithPrediction(BaseModel):
    timestamp: datetime = datetime.utcnow()
    prediction: np.ndarray = Field(...)
    data_points: pd.DataFrame = Field(...)
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


class DataCaptureParameters(BaseModel):
    operation_type: str = Field(...)
    storage_engine: str
    login_url: Optional[DatabaseLogin]
    model_id: str = ""
    model_version: str = ""
    data_id: str = ""
    y_name: Optional[str]
    pred_name: Optional[str]
    other_labels: Optional[List[str]] = None

    @validator("model_id", "model_version", "data_id")
    def check_model_version(cls, value, values):
        if values.get("operation_type") == DATABASE_OPERATION_TYPE_INSERT_PREDICTION:
            if not value:
                raise ValueError(
                    "Required parameters missing. model_id, model_version, and data_id are required for Insert operation."
                )
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
        Function to convert prediction output to Pandas dataframe
        to be inserted in DB
        """
        if self.operation_type not in (DATABASE_OPERATION_TYPE_INSERT_PREDICTION,):
            raise Exception(f"Method is only allowed for operation type {DATABASE_OPERATION_TYPE_INSERT_PREDICTION}")

        data_with_prediction = data.data_points.copy()
        data_with_prediction["y_pred"] = data.prediction

        data_with_prediction.loc[:, "Timestamp"] = data.timestamp
        data_with_prediction.loc[:, "model_id"] = self.model_id
        data_with_prediction.loc[:, "model_version"] = self.model_version
        data_with_prediction.loc[:, "data_id"] = self.data_id
        if self.pred_name:
            data_with_prediction.loc[:, self.pred_name] = self.pred_name

        data_with_prediction["uuid"] = [uuid.uuid4() for _ in range(len(data_with_prediction.index))]
        # Set 'TimeStamp' field as index of dataframe
        data_with_prediction.set_index("Timestamp", inplace=True)
        DataFactory.sql_ingestion(DB_PREDICTION_MEASURMENT, self.storage_engine, data_with_prediction, self.login_url)
        logger.info("Data was successfully ingested into the db")
        return

    def collect(self, filters: dict = None):
        """Function for retrieving Pandas dataframe from the DB"""
        return DataFactory.sql_digestion(DB_PREDICTION_MEASURMENT, self.storage_engine, self.login_url, filters)

    def collect_eval_timestamp(self):
        """Retrieves last period what was inseted to the database"""
        results = DataFactory.sql_digestion(DB_EVAL_TIMESTAMP_MEASURMENT, self.storage_engine, self.login_url)

        if results and list(results.get("eval_timestamp")):
            df = pd.DataFrame(results.get("eval_timestamp"))
            df["eval_timestamp"] = pd.to_datetime(df["eval_timestamp"])
            return df.iloc[df["eval_timestamp"].argmax()]["eval_timestamp"]
        return None

    def push_eval_timestamp(self, eval_df):
        """Inserts period to the database"""
        DataFactory.sql_ingestion(DB_EVAL_TIMESTAMP_MEASURMENT, self.storage_engine, eval_df, self.login_url)

    def push_metrics(self, metrics_df):
        """Insterts metrics dataframe to the database"""
        DataFactory.sql_ingestion(DB_METRICS_MEASURMENT, self.storage_engine, metrics_df, self.login_url)


class DataFactory:
    @classmethod
    def get_storage_engine(cls, storage_engine: str):
        """Returns storage current storage engine"""
        if storage_engine in ("influxdb",):
            from ..db_connectors.influxdb.db_connection import StorageEngine

            return StorageEngine
        else:
            raise e(
                value=storage_engine,
                message=f"{storage_engine} is an Invalid Storage Engine",
            )

    @classmethod
    def sql_ingestion(cls, measurment_name, storage_engine: str, dataframe: pd.DataFrame, database_login: DatabaseLogin):
        """Function to import DB connection based on storage engine and call sql_insertion"""
        sengine = cls.get_storage_engine(storage_engine)

        sengine().sql_insertion(
            measurment_name,
            df=dataframe,
            database_login=database_login,
        )

    @classmethod
    def sql_digestion(cls, measurment_name, storage_engine: str, database_login: DatabaseLogin = None, filters: dict = None):
        """Function to export DB connection based on storage engine and call sql_digestion"""

        sengine = cls.get_storage_engine(storage_engine)
        return sengine().sql_digestion(measurment_name, database_login=database_login, filters=filters)

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
