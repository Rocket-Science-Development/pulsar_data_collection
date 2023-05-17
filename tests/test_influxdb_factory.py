from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from influxdb_client import InfluxDBClient

from pulsar_data_collection.config import factories

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
class TestInfluxDBIntegration:
    def test_make_connection_is_not_none(self, db_login, storage_engine):
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        assert db_connection is not None

    def test_make_connection_isinstance_InfluxDBClient(self, db_login, storage_engine):
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        assert isinstance(db_connection, InfluxDBClient)

    def test_make_connection_args_check(self, db_login, storage_engine):
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        assert db_connection.url == "http://127.0.0.1:8086/"
        assert db_connection.token == "mytoken"
        assert db_connection.org == "pulsarml"

    def test_write_data_writes_data(self, storage_engine, db_login, capsys):
        # get dataframe to write
        test_data = pd.read_csv("tests/data/split/test_data_no_class.csv", header=0).copy()

        # Create connections
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)

        other_labels = {"environment": "test", "tag": "whatever"}
        default_tags = {"model_id": "test_id", "model_version": "test_version", "data_id": "test_data_id", **other_labels}

        params = {
            "client": db_connection,
            "bucket_name": "demo",
            "data_points": test_data,
            "data_frame_measurement_name": "test_write_data",
            "data_frame_timestamp_column": "_time",
            "data_frame_tag_columns": [],
            "default_tags": default_tags,
            "timestamp": now,
            "timezone": "EST",
        }

        influxdb.write_data(**params)
        captured = capsys.readouterr()
        assert "Written batch" in captured.out

    def test_write_data_fails(self, storage_engine, db_login, capsys):
        # get dataframe to write
        test_data = pd.read_csv("tests/data/split/test_data_no_class.csv", header=0).copy()

        # Create connections
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)

        other_labels = {"environment": "test", "tag": "whatever"}
        default_tags = {"model_id": "test_id", "model_version": "test_version", "data_id": "test_data_id", **other_labels}

        params = {
            "client": db_connection,
            "bucket_name": "WrongBucketName",
            "data_points": test_data,
            "data_frame_measurement_name": "test_write_data",
            "data_frame_timestamp_column": "_time",
            "data_frame_tag_columns": [],
            "default_tags": default_tags,
            "timestamp": now,
            "timezone": "EST",
        }

        influxdb.write_data(**params)
        captured = capsys.readouterr()
        assert "Cannot write batch:" in captured.out

    def test_write_data_returns_not_none(self, storage_engine, db_login):
        # get dataframe to write
        test_data = pd.read_csv("tests/data/split/test_data_no_class.csv", header=0).copy()

        # Create connections
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)
        other_labels = {"environment": "test", "tag": "whatever"}
        default_tags = {"model_id": "test_id", "model_version": "test_version", "data_id": "test_data_id", **other_labels}

        params = {
            "client": db_connection,
            "bucket_name": "demo",
            "data_points": test_data,
            "data_frame_measurement_name": "test_write_data",
            "data_frame_timestamp_column": "_time",
            "data_frame_tag_columns": [],
            "default_tags": default_tags,
            "timestamp": now,
            "timezone": "EST",
        }
        query2 = """
        from(bucket: "demo")
        |> range(start: 1)
        |> filter(fn: (r) => r["_measurement"] == "test_write_data")
        |> group(columns: ["_field"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        influxdb.write_data(**params)
        db_connection = influxdb.make_connection(**db_login)
        query_api = db_connection.query_api()
        data_frame = query_api.query_data_frame(query=query2)

        assert data_frame is not None
        assert isinstance(data_frame, pd.DataFrame)
