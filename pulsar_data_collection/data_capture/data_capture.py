import importlib
from datetime import datetime
from typing import List, Optional

import exceptions as e
import numpy as np
import pandas as pd
from pydantic import BaseModel


# TODO : aadd validators for both models
class DataCaptureParameters(BaseModel):
    storage_engine: str
    login_url: str
    org_id: str
    project_id: str
    environment_id: str
    other_labels: Optional[List[str]] = None


class DataWithPrediction(BaseModel):
    timestamp: datetime
    prediction: np.ndarray
    data_points: pd.DataFrame
    features_names: List[str]


class DataCapture(DataCaptureParameters):
    def __init__(self):

        if DataCaptureParameters.storage_engine:
            self.storage_engine = importlib.import_module(
                DataCaptureParameters.storage_engine, "db_connectors"
            ).StorageEngine(login_url=DataCaptureParameters.login_url)
        else:
            raise e.StorageEngineDoesntExist(
                value=DataCaptureParameters.storage_engine,
                message=f"{DataCaptureParameters.storage_engine} Storage Engine doesn't exist",
            )

    def collect(self, data=DataWithPrediction):
        pred = pd.DataFrame(data.prediction, columns=["target"])
        data_with_prediction = pd.concat([data.data_points, pred], axis=1)

        data_with_prediction.loc[:, "Timestamp"] = data.timestamp
        data_with_prediction.loc[:, "org_id"] = DataCaptureParameters.org_id
        data_with_prediction.loc[:, "project_id"] = DataCaptureParameters.project_id
        data_with_prediction.loc[:, "environment_id"] = DataCaptureParameters.environment_id

        for label in data.other_labels:
            data_with_prediction.loc[:, f"{label}"] = label
