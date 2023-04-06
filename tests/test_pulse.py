# import pickle as pkl
# from io import StringIO

import os

import pytest

from pulsar_data_collection import models, pulse

# from unittest.mock import Mock, call, patch


# @pytest.fixture
# def mock_influxdb() -> Mock:
#     mock_influxdb_instance = Mock(set=Mock(return_value=True))
#     return mock_influxdb_instance

db_login = {
    "url": os.getenv("INFLUXDB_URL", "http://localhost:8086"),
    "token": os.getenv("INFLUX_TOKEN", "mytoken"),
    "org": os.getenv("INFLUXDB_ORG", "pulsarml"),
    "bucket_name": os.getenv("INFLUXDB_BUCKET_NAME", "demo"),
}


@pytest.fixture
def valid_pulse_paramaters() -> models.PulseParameters:
    return pulse.PulseParameters(
        model_id="1",
        model_version="v1",
        data_id="1",
        reference_data_storage="./data/clean/clean_data.csv",
        y_name="y_pred",
        storage_engine="influxdb",
        login=db_login,
    )


# class TestPulse:
#     def test_storage_engine_exists(valid_params: models.PulseParameters):
#         ...
