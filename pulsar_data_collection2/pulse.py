from config import factories

from pulsar_data_collection2.models import PulseParameters


class Pulse(PulseParameters):
    """
    This class expose methods in order to collect data from
    an inference container/webapp
    """

    def __init__(self, data=PulseParameters):
        self.model_id = data.model_id
        self.model_version = data.model_version
        self.reference_data = data.reference_data
        self.y_name = data.y_name
        database = factories.get(data.storage_engine)
        self.db_connection = database.make_connection(**data.login)
        self.other_labels = data.other_labels

    def capture_data(self):
        pass
