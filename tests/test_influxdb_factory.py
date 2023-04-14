import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

from pulsar_data_collection.config import factories


def is_responsive(**kwargs):
    try:
        with InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"]) as conn:
            response = conn.ping()
            if response:
                return True
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def storage_engine():
    return factories["influxdb"]


@pytest.fixture(scope="session")
def db_login(docker_ip):
    load_dotenv(dotenv_path=Path("db/influxdb/"))
    db_login = {
        "url": os.getenv("INFLUXDB_URL", f"http://{docker_ip}:8086/"),
        "token": os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN", "mytoken"),
        "org": os.getenv("DOCKER_INFLUXDB_INIT_ORG", "pulsarml"),
        "bucket_name": os.getenv("DOCKER_INFLUXDB_INIT_BUCKET", "demo"),
    }
    return db_login


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return Path("tests/db/influxdb/docker-compose.yaml")


@pytest.fixture(scope="session")
def run_influxdb(db_login, docker_services):
    docker_services.wait_until_responsive(timeout=2, pause=0, check=lambda: is_responsive(**db_login))


@pytest.mark.usefixtures("db_login", "run_influxdb", "storage_engine")
class TestInfluxDB:
    def test_make_connection_is_not_none(self, db_login, storage_engine):
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        assert db_connection is not None

    def test_make_connection_isinstance_InfluxDBClient(self, db_login, storage_engine):
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        assert isinstance(db_connection, InfluxDBClient)
