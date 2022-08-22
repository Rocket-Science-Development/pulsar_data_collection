import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 8086)
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pass123")
DB_NAME = os.getenv("DB_NAME", "testDB")
DB_PROTOCOL = os.getenv("DB_PROTOCOL", "line")
DB_MEASURMENT = os.getenv("DB_MEASURMENT", "")