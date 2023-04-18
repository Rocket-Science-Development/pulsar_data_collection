from pathlib import Path

import pytest
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
    db_login = {
        "url": f"http://{docker_ip}:8086/",
        "token": "mytoken",
        "org": "pulsarml",
        "bucket_name": "demo",
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

    def test_make_connection_args_check(self, db_login, storage_engine, docker_ip):
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        assert db_connection.url == f"http://{docker_ip}:8086/"
        assert db_connection.token == "mytoken"
        assert db_connection.org == "pulsarml"
