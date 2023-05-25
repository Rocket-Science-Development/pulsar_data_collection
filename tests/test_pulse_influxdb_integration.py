import pickle as pkl
from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest
from influxdb_client import InfluxDBClient

from pulsar_data_collection.config import factories
from pulsar_data_collection.models import DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse


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


# TODO: change docker based fixture to a mock in order to make tests faster
@pytest.fixture(scope="session")
def run_influxdb(db_login, docker_services):
    docker_services.wait_until_responsive(timeout=2, pause=0, check=lambda: is_responsive(**db_login))


@pytest.mark.usefixtures("db_login", "storage_engine", "run_influxdb")
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
        assert pulse.reference_data_storage == {"bucket": "bucket-address", "filename": "filename.csv"}
        assert pulse.target_name == "class"
        assert isinstance(pulse.db_connection, InfluxDBClient)

    def test_Pulse_capture_data_method_simple_prediction(self, docker_ip):
        # TODO : move these to become
        model_path = Path("tests/model/kidney_disease.pkl")
        inference_dataset = pd.read_csv("tests/data/split/test_data_no_class.csv", header=0)
        reference_data = "/data/raw/csv_result-chronic_kidney_disease.csv"
        params = PulseParameters(
            model_id="simple_prediction",
            model_version="simple_prediction0.1",
            data_id="simple_prediction_data",
            reference_data_storage=reference_data,
            target_name="class",
            storage_engine="influxdb",
            timestamp_column_name="_time",
            login={
                "url": f"http://{docker_ip}:8086/",
                "token": "mytoken",
                "org": "pulsarml",
                "bucket_name": "demo",
            },
            additional_tags={"timezone": "EST", "reference_dataset": reference_data},
        )
        pulse = Pulse(data=params)

        pickle_model = pkl.load(open(model_path, "rb"))
        prediction_simple = pickle_model.predict(inference_dataset)

        time = dt.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        capture_params = DataWithPrediction(
            data_points=inference_dataset,
            predictions=pd.DataFrame(prediction_simple, columns=["prediction"]),
            timestamp=time,
            features_names=inference_dataset.columns.tolist(),
        )

        pulse.capture_data(data=capture_params)

        query = f"""
        from(bucket: "demo")
        |> range(start: {time})
        |> filter(fn: (r) => r["_measurement"] == "{params.model_id}_{params.model_version}_input_data"
            and r["model_id"] == "{params.model_id}"
            and r["model_version"] == "{params.model_version}")
        |> group(columns: ["_measurement","_field", "timezone","model_id","model_version","data_id"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        storage_engine = factories[params.storage_engine]
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**params.login)
        query_api = db_connection.query_api()
        data_frame = query_api.query_data_frame(query=query)

        assert "prediction" in data_frame.columns
        assert prediction_simple in data_frame["prediction"].values

    def test_Pulse_capture_data_method_proba_prediction(self, docker_ip):
        # TODO : move these to become fixtures
        model_path = Path("tests/model/kidney_disease.pkl")
        inference_dataset = pd.read_csv("tests/data/split/test_data_no_class.csv", header=0)
        reference_data = "/data/raw/csv_result-chronic_kidney_disease.csv"
        params = PulseParameters(
            model_id="test_id",
            model_version="version_test",
            data_id="test_data_id",
            reference_data_storage=reference_data,
            target_name="class",
            storage_engine="influxdb",
            timestamp_column_name="_time",
            login={
                "url": f"http://{docker_ip}:8086/",
                "token": "mytoken",
                "org": "pulsarml",
                "bucket_name": "demo",
            },
            additional_tags={"timezone": "EST", "reference_dataset": reference_data},
        )

        pulse = Pulse(data=params)

        pickle_model = pkl.load(open(model_path, "rb"))
        prediction_proba = pd.DataFrame(pickle_model.predict_proba(inference_dataset), columns=["proba_class_0", "proba_class_1"])
        prediction_simple = pd.DataFrame(pickle_model.predict(inference_dataset), columns=["prediction"])
        print(f"proba prediction is {type(prediction_proba)} of size {prediction_proba.shape}")

        time = dt.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        capture_params = DataWithPrediction(
            data_points=inference_dataset,
            predictions=pd.concat(objs=[prediction_simple, prediction_proba], copy=False, axis=1),
            timestamp=time,
            features_names=inference_dataset.columns.tolist(),
        )

        pulse.capture_data(data=capture_params)

        query = f"""
        from(bucket: "demo")
        |> range(start: {time})
        |> filter(fn: (r) => r["_measurement"] == "{params.model_id}_{params.model_version}_input_data"
            and r["model_id"] == "{params.model_id}"
            and r["model_version"] == "{params.model_version}")
        |> group(columns: ["_measurement","_field", "timezone","model_id","model_version","data_id"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """

        storage_engine = factories[params.storage_engine]
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**params.login)
        query_api = db_connection.query_api()
        data_frame = query_api.query_data_frame(query=query)

        print(data_frame.columns)

        assert "prediction" in data_frame.columns
        assert "proba_class_0" in data_frame.columns
        assert "proba_class_1" in data_frame.columns
        assert prediction_simple["prediction"] in data_frame["prediction"].values
        assert prediction_proba["proba_class_0"] in data_frame["proba_class_0"].values
        assert prediction_proba["proba_class_1"] in data_frame["proba_class_1"].values
