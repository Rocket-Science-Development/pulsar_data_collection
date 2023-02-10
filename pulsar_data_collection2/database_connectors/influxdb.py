from database_actions import DatabaseActions, DatabaseActionsFactory
from influxdb import InfluxDBClient


class InfluxdbActions(DatabaseActions):
    """Infuxdb database connection"""

    def make_connection(
        self, host: str, port: int, password: str, database_name: str, ssl: bool = False, verify_ssl: bool = False
    ) -> InfluxDBClient:
        """Makes connection to the Influxdb database"""

        return InfluxDBClient(host=host, port=port, password=password, ssl=ssl, verify_ssl=verify_ssl)

    def write_data(self, client: InfluxDBClient):
        """Sends data to the database"""
        pass


class Influxdb(DatabaseActionsFactory):
    """Infuxdb database connection"""

    def get_database_actions(self) -> DatabaseActions:
        """Makes connection to the Influxdb database"""
        return InfluxdbActions
