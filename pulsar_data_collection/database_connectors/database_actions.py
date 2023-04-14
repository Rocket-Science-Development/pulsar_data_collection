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
    def write_data(self):
        """Writes Data to the database"""


class DatabaseActionsFactory(ABC):
    """
    Factory that represents database connection
    """

    @abstractmethod
    def get_database_actions(self) -> DatabaseActions:
        """Returns a new video exporter belonging to this factory."""
