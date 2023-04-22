from datetime import datetime

import pandas as pd
import pytest
from influxdb_client import InfluxDBClient

from pulsar_data_collection.config import factories

now = datetime.now().isoformat()


# def is_responsive(**kwargs):
#     try:
#         with InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"]) as conn:
#             response = conn.ping()
#             if response:
#                 return True
#     except ConnectionError:
#         return False


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


# @pytest.fixture(scope="session")
# def docker_compose_file(pytestconfig):
#     return Path("db/influxdb/docker-compose.yaml")


# # TODO: change docker based fixture to a mock in order to make tests faster
# @pytest.fixture(scope="session")
# def run_influxdb(db_login, docker_services):
#     docker_services.wait_until_responsive(timeout=2, pause=0,
# check=lambda: is_responsive(**db_login))


@pytest.mark.usefixtures("db_login", "storage_engine")
class TestInfluxDBMakeConnection:
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
        test_data = pd.read_csv("data/split/test_data_no_class.csv", header=0).copy()
        test_data.loc[:, "_time"] = now
        test_data.loc[:, "model_id"] = "test_id"
        test_data.loc[:, "model_version"] = "test_version"
        test_data.loc[:, "data_id"] = "test_data_id"
        test_data.set_index("_time")

        # Create connections
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)

        params = {
            "client": db_connection,
            "bucket_name": "demo",
            "records": test_data,
            "data_frame_measurement_name": "test_write_data",
            "data_frame_timestamp_column": "_time",
            "data_frame_tag_columns": [],
        }

        influxdb.write_data(**params)
        captured = capsys.readouterr()
        assert "Written batch" in captured.out

    def test_write_data_fails(self, storage_engine, db_login, capsys):

        # get dataframe to write
        test_data = pd.read_csv("data/split/test_data_no_class.csv", header=0).copy()
        test_data.loc[:, "_time"] = now
        test_data.loc[:, "model_id"] = "test_id"
        test_data.loc[:, "model_version"] = "test_version"
        test_data.loc[:, "data_id"] = "test_data_id"
        test_data.set_index("_time")

        # Create connections
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)

        params = {
            "client": db_connection,
            "bucket_name": "WrongBucketName",
            "records": test_data,
            "data_frame_measurement_name": "test_write_data",
            "data_frame_timestamp_column": "_time",
            "data_frame_tag_columns": [],
        }

        influxdb.write_data(**params)
        captured = capsys.readouterr()
        assert "Cannot write batch:" in captured.out

    def test_write_data_returns_data(self, storage_engine, db_login):

        # get dataframe to write
        test_data = pd.read_csv("data/split/test_data_no_class.csv", header=0).copy()
        test_data.loc[:, "_time"] = now
        test_data.loc[:, "model_id"] = "test_id"
        test_data.loc[:, "model_version"] = "test_version"
        test_data.loc[:, "data_id"] = "test_data_id"
        test_data.set_index("_time")

        # print(test_data.shape)
        # Create connections
        influxdb = storage_engine.get_database_actions()
        db_connection = influxdb.make_connection(**db_login)

        params = {
            "client": db_connection,
            "bucket_name": "demo",
            "records": test_data,
            "data_frame_measurement_name": "test_write_data",
            "data_frame_timestamp_column": "_time",
            "data_frame_tag_columns": [],
        }

        influxdb.write_data(**params)
        # query_api = db_connection.query_api()

        # query2 = """
        # from(bucket: "demo")
        # |> range(start: -1m)
        # |> filter(fn: (r) => r["_measurement"] == "test_write_data")
        # |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        # |> keep(columns: ["id","model_id", "model_version", "data_id","age","bp","sg","al","su",
        # "rbc","pc","pcc","ba", "bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc",
        # "htn","dm","cad","appet","pe","ane"])
        # """

        # data_frame = query_api.query_data_frame(query=query2)

        # print(data_frame.to_string())
        assert False
