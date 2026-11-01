from .database_connectors import influxdb

factories = {"influxdb": influxdb.Influxdb()}
