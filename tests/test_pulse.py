# # import pickle as pkl
# # from io import StringIO

# import os
# import pickle as pkl
# from pathlib import Path

# import pytest
# from dotenv import load_dotenv

# from pulsar_data_collection import models, pulse


# ## env file path
# def load_database_config():
#     dotenv_path = Path("db/influxdb/")
#     load_dotenv(dotenv_path=dotenv_path)
#     db_login = {
#         "url": os.getenv("INFLUXDB_URL", "http://localhost:8086"),
#         "token": os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN", "mytoken"),
#         "org": os.getenv("DOCKER_INFLUXDB_INIT_ORG", "pulsarml"),
#         "bucket_name": os.getenv("DOCKER_INFLUXDB_INIT_BUCKET", "demo"),
#     }
#     return db_login


# @pytest.fixture
# def load_variables():
#     model_path = Path("model/kidney_disease.pkl")
#     valid_pulse_params = pulse.PulseParameters(
#         model_id="1",
#         model_version="v1",
#         data_id="1",
#         reference_data_storage="./data/clean/clean_data.csv",
#         y_name="y_pred",
#         storage_engine="influxdb",
#         login=load_database_config(),
#     )


# @pytest.fixture(autouse=True)
# def mock_influxdb(mocker):
#     mock_influxdb_instance = mocker.patch.object()
#     return mock_influxdb_instance


# @pytest.mark.usefixtures("load_variables", "load_database_config", "mock_influxdb")
# class TestInfluxdb:
#     def test_ingestion(self):
#         model = pkl.load(open(model_path))
