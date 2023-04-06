# import pickle as pkl

# import pandas as pd

# from pulsar_data_collection.data_capture import (
#     DatabaseLogin,
#     DataCapture,
#     DataWithPrediction,
# )


# class TestDataIngestionInfluxDB:
#     """
#     This test class trying to ingest data to the influxdb
#     """

#     def test_data_ingestion(self):

#         model = pkl.load(open("examples/kidney_disease_influxdb/kidney_disease.pkl", "rb"))

#         to_predict = pd.read_csv("examples/kidney_disease_influxdb/test_data_no_class.csv", sep=",", header=0)
#         to_predict1 = to_predict[:15]
#         to_predict2 = to_predict[15:]

#         prediction = model.predict(to_predict)
#         prediction1 = model.predict(to_predict1)
#         prediction2 = model.predict(to_predict2)

#         database_login = DatabaseLogin(db_host="localhost", db_port=8086, db_user="admin",
#                                       db_password="pass123", protocol="line")

#         dat_predict = DataWithPrediction(prediction=prediction, data_points=to_predict)
#         dat_predict1 = DataWithPrediction(prediction=prediction1, data_points=to_predict1)
#         dat_predict2 = DataWithPrediction(prediction=prediction2, data_points=to_predict2)

#         dat_capture = DataCapture(
#             storage_engine="influxdb",
#             model_id="RS101",
#             model_version="1.0",
#             data_id="FluxDB",
#             y_name="ABC",
#             pred_name="target",
#             operation_type="INSERT_PREDICTION",
#             login_url=database_login,
#         )
#         dat_capture.push(dat_predict)
#         dat_capture.push(dat_predict1)
#         dat_capture.push(dat_predict2)

#     def test_data_digestion(self):
#         database_login = DatabaseLogin(db_host="localhost", db_port=8086, db_user="admin",
#                                       db_password="pass123", protocol="line")

#         dat_capture = DataCapture(storage_engine="influxdb", operation_type="METRICS", login_url=database_login)
#         dat_capture.collect()

#     def test_period_digestion(self):
#         database_login = DatabaseLogin(db_host="localhost", db_port=8086, db_user="admin",
#            db_password="pass123", protocol="line")
#         dat_capture = DataCapture(storage_engine="influxdb", operation_type="METRICS", login_url=database_login)
#         dat_capture.collect_eval_timestamp()
