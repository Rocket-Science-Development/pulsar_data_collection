import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from config import factories
from pydantic import BaseModel, validator


class PulseParameters(BaseModel):
    """
    Model of input Pulse class

    Attributes
    ----------
    model_id: str
    Model identifier
    model_version: str
    Model version
    data_id: str
    Data identifier
    reference_data_storage: Any  # Union[str, Dict[str, str]]
    Storage path for reference data used in drift recognition, usually training dataset
    target_name: str
    Name of target feature
    storage_engine: str
    Storage engine used to store collected logs
    login: Dict[str, Union[str, int, bool]]
    Dictionary containing the element required to perform successful login to storage engine
    features_metadata: Optional[Dict[str, type]]  # key: feature_name, value : feature type
    Dictionary containing the schema of features used for the model prediction
    other_labels: Optional[Dict[str, Union[str, int, bool]]] = None
    Dictionary of additional labels used provide more metadata regarding the context surrounding the model
    timestamp_column_name: str = "timestamp"
    Name of the column containing the timestamp

    """

    model_id: str
    model_version: str
    data_id: str
    reference_data_storage: Any  # Union[str, Dict[str, str]]
    target_name: str
    storage_engine: str
    login: Dict[str, Union[str, int, bool]]
    features_metadata: Optional[Dict[str, type]]  # key: feature_name, value : feature type
    additional_tags: Optional[Dict[str, Union[str, int, bool]]] = None
    timestamp_column_name: str = "timestamp"

    @validator("storage_engine")
    def check_storage_engine(cls, value):
        if value not in factories:
            raise ValueError(f"Storage Engine for {value} not supported")
        return value


class DataWithPrediction(BaseModel):
    """
    Model of input for capture_data method of Pulse parameter class

    Attributes
    ----------

    prediction_id: Optional[str]
    Identifier of prediction
    timestamp: datetime.datetime = datetime.datetime.now()
    timestamp of when the prediction have been performed
    predictions: Any  # Union[Dict, pd.DataFrame]
    Object containing the array of predictions
    data_points: pd.DataFrame
    Object containing the data points on which the predicitions were performed
    features_names: List[str]
    list of features that were used to perform the prediction

    """

    class Config:
        arbitrary_types_allowed = True

    prediction_id: Optional[str]
    timestamp: datetime.datetime
    predictions: Any
    data_points: pd.DataFrame
    features_names: List[str]
