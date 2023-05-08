import pickle as pkl
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
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


# @pytest.fixture(scope="session")
# def docker_compose_file(pytestconfig):
#     return Path("db/influxdb/docker-compose.yaml")


# # TODO: change docker based fixture to a mock in order to make tests faster
# @pytest.fixture(scope="session")
# def run_influxdb(db_login, docker_services):
#     docker_services.wait_until_responsive(timeout=2, pause=0, check=lambda: is_responsive(**db_login))


class TestModels:
    def test_PulseParameters(self, docker_ip):

        params = PulseParameters(
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
        pulse = Pulse(data=params)

        assert pulse.model_id == "test_id"
        assert pulse.model_version == "version_test"
        assert isinstance(pulse.reference_data_storage, Dict)
        assert pulse.reference_data_storage == {"bucket": "bucket-address", "filename": "filename.csv"}
        assert pulse.target_name == "class"
        assert isinstance(pulse.db_connection, InfluxDBClient)
        assert isinstance(pulse.login, Dict)
        assert pulse.login == {
            "url": f"http://{docker_ip}:8086/",
            "token": "mytoken",
            "org": "pulsarml",
            "bucket_name": "demo",
        }

    def test_Pulse_capture_data_method(self, docker_ip):

        # TODO : move these to become
        model_path = Path("model/kidney_disease.pkl")
        inference_dataset = pd.read_csv("data/split/test_data_no_class.csv", header=0)
        reference_data = Path("../data/raw/csv_result-chronic_kidney_disease.csv")
        params = PulseParameters(
            model_id="test_id",
            model_version="version_test",
            data_id="test_data_id",
            reference_data_storage=reference_data,
            target_name="class",
            storage_engine="influxdb",
            login={
                "url": f"http://{docker_ip}:8086/",
                "token": "mytoken",
                "org": "pulsarml",
                "bucket_name": "demo",
            },
        )
        pulse = Pulse(data=params)
        capture_dict = {}
        pickle_model = pkl.load(open(model_path, "rb"))
        prediction = pickle_model.predict(inference_dataset)

        capture_dict["data_points"] = inference_dataset
        capture_dict["predictions"] = prediction
        capture_dict["timestamp"] = datetime.now()

        pulse.capture_data(data=capture_dict)

        assert False
