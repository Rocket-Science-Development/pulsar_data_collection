from datetime import datetime
from pathlib import Path

import pytest
from influxdb_client import InfluxDBClient

from pulsar_data_collection.models import PulseParameters
from pulsar_data_collection.pulse import Pulse

now = datetime.now().isoformat()


def is_responsive(**kwargs):
    try:
        with InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"]) as conn:
            response = conn.ping()
            if response:
                return True
    except ConnectionError:
        return False


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
def pulse_parameters(db_login):
    return PulseParameters(
        model_id="test_id",
        model_version="version_test",
        data_id="test_data_id",
        reference_data_storage="whatever",
        y_name="class",
        storage_engine="influxdb",
        login=db_login,
    )


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return Path("db/influxdb/docker-compose.yaml")


# TODO: change docker based fixture to a mock in order to make tests faster
@pytest.fixture(scope="session")
def run_influxdb(db_login, docker_services):
    docker_services.wait_until_responsive(timeout=2, pause=0, check=lambda: is_responsive(**db_login))


class TestModels:
    def test_PulseParameters(self, docker_ip):

        Pulse(
            PulseParameters(
                model_id="test_id",
                model_version="version_test",
                data_id="test_data_id",
                reference_data_storage={"bucket": "bucket-address", "filename": "filename.csv"},
                target_name="class",
                storage_engine="influxdb",
                login={
                    "url": f"http://{docker_ip}:8086/",
                    "token": "mytoken",
                    "org": "pulsarml",
                    "bucket_name": "demo",
                },
            )
        )

        assert False

    def test_DataWithPrediction(self):
        pass
