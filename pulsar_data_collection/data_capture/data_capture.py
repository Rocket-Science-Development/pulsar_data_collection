from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..db_connectors.influxdb.db_connection import StorageEngine
from .exceptions import StorageEngineDoesntExist as e

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
    model_id: str
    model_version: str
    data_id: str
    y_name: Optional[str]
    pred_name: Optional[str]
    other_labels: Optional[List[str]] = None


class DataCapture(DataCaptureParameters):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.storage_engine:
            # print(importlib.import_module("self.storage_engine", "pulsar_data_collection.db_connectors"))
            pass

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
        data_with_prediction.loc[:, "model_id"] = self.model_id
        data_with_prediction.loc[:, "model_version"] = self.model_version
        data_with_prediction.loc[:, "data_id"] = self.data_id

        if self.y_name:
            data_with_prediction.loc[:, "y_name"] = self.y_name
        if self.pred_name:
            data_with_prediction.loc[:, "pred_name"] = self.pred_name

        # for label in data.other_labels:
        #     data_with_prediction.loc[:, f"{label}"] = label

        StorageEngine().sql_insertion(data_with_prediction)

        return
