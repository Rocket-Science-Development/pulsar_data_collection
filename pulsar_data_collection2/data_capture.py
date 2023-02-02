import database as db
from input import PulseParameters

factories = {"influxdb": db.database.InfluxdbActionsExporter}


class Pulse(PulseParameters):
    """
    This class expose methods in order to collect data from
    an inference container/webapp
    """

    def __init__(self, data=PulseParameters):

        if data.storage_engine not in factories:
            raise ValueError("Storage Engine not supported")
        else:
            self.database = factories[data.storage_engine]

    def capture_data(self):
        pass
