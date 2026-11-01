from abc import ABC, abstractmethod

"""
Factory that represents actions related to a database,
i.e. make a connection, and send data
"""


class DatabaseActions(ABC):
    """
    Basic representation of database connections instance
    """

    @abstractmethod
    def make_connection(self):
        """Sets up the connection object"""

    @abstractmethod
    def create_client(self, **login):
        """Client object without connection check"""

    @abstractmethod
    def prepare_frame(self, data_points, prediction=None, timestamp=None):
        """Rows to write for one capture"""

    @abstractmethod
    def write_frame(self, client, frame, **write_params):
        """Writes prepared rows synchronously; raises on failure"""

    @abstractmethod
    def write_data(self):
        """Writes Data to the database"""

    @abstractmethod
    def create_param_dict(self):
        """Create the parameter dictionary to write to a database implementation"""


class DatabaseActionsFactory(ABC):
    """
    Factory that represents database connection
    """

    @abstractmethod
    def get_database_actions(self) -> DatabaseActions:
        """Returns a new video exporter belonging to this factory."""
