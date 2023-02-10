from config import factories

from pulsar_data_collection2.data_models import PulseParameters


class Pulse(PulseParameters):
    """
    This class expose methods in order to collect data from
    an inference container/webapp
    """

    def __init__(self, data=PulseParameters):
        self.database = factories.get(data.storage_engine)

    def capture_data(self):
        pass
