import pandas as pd
from config import factories
from models import DataWithPrediction, PulseParameters


class Pulse(PulseParameters):
    """
    This class expose methods in order to collect data from
    an inference container/webapp
    """

    def __init__(self, data=PulseParameters):
        """
        Initializing Pulse class instance
        """
        self.model_id = data.model_id
        self.model_version = data.model_version
        self.reference_data_storage = data.reference_data_storage
        self.y_name = data.y_name
        self.storage_engine = factories[data.storage_engine]
        self.db_connection = self.storage_engine.make_connection(**data.login)
        self.login = data.login
        self.other_labels = data.other_labels
        self.timezone = data.timezone

    def capture_data(self, data=DataWithPrediction):
        """
        Capturing data from inference code
        """
        data_with_prediction = data.data_points.copy()
        data_with_prediction.loc[:, "timestamp"] = pd.date_range(
            start=data.timestamp, periods=data.data_points.shape[0], freq="L", inclusive="left", tz=self.timezone
        )
        data_with_prediction.set_index("timestamp")
        data_with_prediction.loc[f"{self.y_name}_prediction"] = data.prediction

        params = {
            "client": self.db_connection,
            "bucket_name": self.login["bucket_name"],
            "record": data_with_prediction,
            "data_frame_measurement_name": f"{self.model_id}",
            "data_frame_timestamp_column": "timestamp",
            "data_frame_tag_columns": [],
            "default_tags": {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "data_id": self.data_id,
                **self.other_labels,
            },
        }
        # TODO : plan to dynamically set the method's input in
        # relation to the storage engine assigned
        self.database.write_data(**params)
