import pytest
import pandas as pd
import pickle as pkl
from io import StringIO
from pulsar_data_collection.data_capture import DataCapture, DataWithPrediction, DatabaseLogin



class TestDataIngestionInfluxDB:
    """
        This test class trying to ingest data to the influxdb
    """

    def test_data_ingestion(self):

        model = pkl.load(open("examples/kidney_disease_influxdb/kidney_disease.pkl", "rb"))

        to_predict = pd.read_csv("examples/kidney_disease_influxdb/test_data_no_class.csv", sep=",", header=0)

        prediction = model.predict(to_predict)

        database_login = DatabaseLogin(db_host="localhost", db_port=8086, db_user="admin", db_password="pass123",
                                       db_name="testDB", protocol="line", measurement="something")

        dat_predict = DataWithPrediction(prediction=prediction, data_points=to_predict)

        dat_capture = DataCapture(
            storage_engine="influxdb",
            model_id="RS101",
            model_version="1.0",
            data_id="FluxDB",
            y_name="ABC",
            pred_name="ABC",
            operation_type="INSERT",
            login_url=database_login
        )

        dat_capture.push(dat_predict)


    def test_data_digestion(self):
        database_login = DatabaseLogin(db_host="localhost", db_port=8086, db_user="admin", db_password="pass123",
                                       db_name="testDB", protocol="line", measurement="something")

        dat_capture = DataCapture(
            storage_engine="influxdb",
            operation_type="RETRIEVE",
            login_url=database_login
        )




