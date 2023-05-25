from config import factories
from models import DataWithPrediction, PulseParameters


class Pulse:
    """
    This class expose methods in order to collect data from
    an inference container/webapp
    """

    def __init__(self, data: PulseParameters):
        """
        Initializing Pulse class instance

        Parameters
        ----------

        data : PulseParameters
        Pydantic Model providing the interface to the class constructor,
        Refer to PulseParameters model attributes for detailed list of inputs



        """
        self.model_id = data.model_id
        self.model_version = data.model_version
        self.reference_data_storage = data.reference_data_storage
        self.target_name = data.target_name
        factory = factories.get(data.storage_engine)
        self.storage_engine = factory.get_database_actions()
        self.db_connection = self.storage_engine.make_connection(**data.login)
        self.additional_tags = data.additional_tags
        self.params = self.storage_engine.create_param_dict(client=self.db_connection, **data.dict())

    def capture_data(self, data: DataWithPrediction):
        """
        Capturing data from inference code
        """

        self.params.update(
            {
                "data_points": data.data_points,
                "prediction": data.predictions,
                "timestamp": data.timestamp,
            }
        )
        self.storage_engine.write_data(**self.params)
