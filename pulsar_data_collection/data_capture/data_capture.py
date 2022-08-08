import importlib
import sys
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

sys.path.append("../")
import pulsar_data_collection.data_capture.exceptions as e
from pulsar_data_collection.db_connectors.influxdb.db_connection import StorageEngine

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
    prediction: np.ndarray
    data_points: pd.DataFrame
    features_names: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True


class DataCaptureParameters(BaseModel):
    storage_engine: str = "influxdb"
    login_url: Optional[DatabaseLogin]
    org_id: str
    project_id: str
    environment_id: str
    other_labels: Optional[List[str]] = None


class DataCapture(DataCaptureParameters):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.storage_engine:
            print(importlib.import_module("..self.storage_engine", "pulsar_data_collection.db_connectors"))

        else:
            raise e.StorageEngineDoesntExist(
                value=DataCaptureParameters.storage_engine,
                message=f"{DataCaptureParameters.storage_engine} Storage Engine doesn't exist",
            )

        return super().__init__(*args, **kwargs)

    def collect(self, data=DataWithPrediction):

        """
        Function to convert prediction output to Pandas dataframe to be inserted in DB
        """

        pred = pd.DataFrame(data.prediction, columns=["target"])
        data_with_prediction = pd.concat([data.data_points, pred], axis=1)

        data_with_prediction.loc[:, "Timestamp"] = data.timestamp
        data_with_prediction.loc[:, "org_id"] = self.org_id
        data_with_prediction.loc[:, "project_id"] = self.project_id
        data_with_prediction.loc[:, "environment_id"] = self.environment_id

        # for label in data.other_labels:
        #     data_with_prediction.loc[:, f"{label}"] = label

        StorageEngine().sql_insertion(data_with_prediction)

        return
