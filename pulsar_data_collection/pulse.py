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
        super().__init__()
        self.model_id = data.model_id
        self.model_version = data.model_version
        self.reference_data_storage = data.reference_data_storage
        self.target_name = data.target_name
        self.storage_engine = factories[data.storage_engine]
        self.db_connection = self.storage_engine.make_connection(**data.login)
        self.login = data.login
        self.other_labels = data.other_labels
        self.params = self.storage_engine.param_dict(client=self.db_connection, **data)

    def capture_data(self, data=DataWithPrediction):
        """
        Capturing data from inference code
        """
        self.params["record"] = data.data_points
        self.params["prediction"] = data.prediction
        self.params["timestamp"] = data.timestamp
        self.params["timezone"] = data.timezone
        self.database.write_data(**self.params)
