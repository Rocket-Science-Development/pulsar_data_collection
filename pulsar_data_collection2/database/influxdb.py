from database_actions import DatabaseActions, DatabaseActionsFactory


class InfluxdbActions(DatabaseActions):
    """Infuxdb database connection"""

    def make_connection(self):
        """Makes connection to the Influxdb database"""
        pass

    def write_data(self):
        """Sends data to the database"""


class InfluxdbActionsFactory(DatabaseActionsFactory):
    """Infuxdb database connection"""

    def get_database_actions(self) -> DatabaseActions:
        """Makes connection to the Influxdb database"""
        return InfluxdbActions
