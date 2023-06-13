import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 8086)
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pass123")
DB_NAME = os.getenv("DB_NAME", "PULSAR_DATA_COLLECTION")
DB_PROTOCOL = os.getenv("DB_PROTOCOL", "line")
DB_PREDICTION_MEASURMENT = os.getenv("DB_PREDICTION_MEASURMENT", "prediction")
DB_EVAL_TIMESTAMP_MEASURMENT = os.getenv("DB_EVAL_TIMESTAMP_MEASURMENT", "eval_timestamp")
DB_METRICS_MEASURMENT = os.getenv("DB_METRICS_MEASURMENT", "metrics")
