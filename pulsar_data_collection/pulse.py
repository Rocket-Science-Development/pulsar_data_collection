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
        database = factories.get(data.storage_engine)
        self.db_connection = database.make_connection(**data.login)
        self.login = data.login
        self.other_labels = data.other_labels

    def capture_data(self, data=DataWithPrediction):
        """
        Capturing data from inference code
        """
        data_with_prediction = data.data_points.copy()
        data_with_prediction[f"{self.y_name}_prediction"] = data.prediction
        data_with_prediction.loc[:, "timestamp"] = data.timestamp
        data_with_prediction.loc[:, "model_id"] = self.model_id
        data_with_prediction.loc[:, "model_version"] = self.model_version
        data_with_prediction.loc[:, "data_id"] = self.data_id
        data_with_prediction.set_index("Timestamp", inplace=True)

        self.database.write_data(
            async_client=self.db_connection,
            bucket_name=self.login["bucket_name"],
        )
