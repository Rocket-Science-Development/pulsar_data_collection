import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pulsar_data_collection.config import factories


# env file path
@pytest.fixture(scope="session")
def load_database_config():
    dotenv_path = Path("db/influxdb/")
    load_dotenv(dotenv_path=dotenv_path)
    db_login = {
        "url": os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        "token": os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN", "mytoken"),
        "org": os.getenv("DOCKER_INFLUXDB_INIT_ORG", "pulsarml"),
        "bucket_name": os.getenv("DOCKER_INFLUXDB_INIT_BUCKET", "demo"),
    }
    return db_login


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return Path(f"{pytestconfig.rootdir}/db/influxdb/docker-compose.yml")


@pytest.mark.usefixtures("load_database_config")
class TestInfluxDB:
    def test_make_connection_not_none(self):
        db_login = load_database_config()
        storage_engine = factories.get("influxdb")
        db_connection = storage_engine.make_connection(**db_login)
        assert db_connection is not None
