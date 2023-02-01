# pulsar_data_collection

Pulsar data collection SDK is an open-source Python library for
pushing/processing/collecting features, predictions and metadata. Works with different
data storages, at this point InfluxDB is implemented.

## Getting started

Install Pulsar Data Collection with pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade pulsar-data-collection
```

### Components

There are two core components in data collection SDK: storage engine and data capture.
Right now storage engine implemented only for InfluxDb, it helps to make ingestion and digestion operations
to the database.

#### Data Capture

`DataCapture` class helps to ingest dataset to database with needed parameters and needed format for future
digestion and metrics calculation without any significant changes of data.

It requires `storage_engine` (available only influxdb right now), `operation_type` (`DATABASE_OPERATION_TYPE_INSERT_PREDICTION`,
`DATABASE_OPERATION_TYPE_METRICS`),  `login_url` (object of DatabaseLogin class) as input parameters.

Operation type `DATABASE_OPERATION_TYPE_INSERT_PREDICTION` uses for any ingestion operations to the database.
It requires additional parameters: `model_id`, `model_version`, `data_id`", `y_name`, `pred_name` what describes
an input dataset.
For operation type `DATABASE_OPERATION_TYPE_METRICS` what commonly uses for retrieving dataset ready for metrics
calculation these parameters aren't required.

The last and probably one the most important class to work with is `DataWithPrediction`.
It requires two parameters as input: `prediction`, `data_points`. Where `prediction` is prediction value of the model,
and `data_points` is features dataset. `Push` method of the `DataCapture` takes object of `DataWithPrediction` as
required parameter, and after that makes ingestion operation to database with data transforming, like adding timestamp,
changing name of prediction column in dataset, combining features with prediction into single dataset, creating
influxdb unique cache, etc.

List of methods of `DataCapture` class:

- push(data: DataWithPrediction)
- ingests data to the db after preprocessing it;
- collect(filters: dict) - retrieves data from db;
- collect_eval_timestamp - retrieves the newest timestamp in the database;
- push_eval_timestamp(eval_df: df) - ingesting new one timestamp into db;
- push_metrics(metrics_df: df) - ingesting metrics dataframe to the database after calculations

### Example usage

Initialize Database credentials:

```python
from pulsar_data_collection.data_capture import DatabaseLogin
database_login = DatabaseLogin(db_host=<db_host>), db_port=<db_port>, db_user=<db_user>, db_password=<db_password>, protocol=<db_protocol>)
```

Initialize DataCapture class, depends on operation type use appropriate constant.
For inserting data into the database:

```python
from pulsar_data_collection.data_capture import DataCapture, DATABASE_OPERATION_TYPE_INSERT_PREDICTION

dat_predict = DataWithPrediction(prediction=prediction, data_points=to_predict)

dat_capture = DataCapture(
    storage_engine="influxdb",
    model_id=<model_id>,
    model_version=<model_verstion>,
    data_id=<data_id>,
    y_name=<y_name>,
    pred_name=<pred_name>,
    operation_type=<operation_type>,
    login_url=<database_login>,
)

dat_capture.push(dat_predict)
```

For collecting data from the database:

```python
from pulsar_data_collection.data_capture import DataCapture, DATABASE_OPERATION_TYPE_METRICS

dat_capture = DataCapture(
    storage_engine="influxdb",
    operation_type=DATABASE_OPERATION_TYPE_METRICS,
    login_url=database_login
)

dat_capture.collect()
```

Collection the newest prediction data what wasn't precessed

```python
# receiving the last period of data

last_eval_timestamp = dat_capture.collect_eval_timestamp()

# if last period exists, collecting only data what wasn't collected previously
if last_eval_timestamp:
    last_eval_timestamp_str = last_eval_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    db_df = pd.DataFrame(dat_capture.collect({"time": f">= '{last_eval_timestamp_str}'"}).get("prediction"))
else:
    db_df = pd.DataFrame(dat_capture.collect().get("prediction"))
```

Example of pushing calculated metrics:

```python
dat_capture.push_metrics(df_result_drift)
```

Example of pushing the timestamp when metrics were calculated:

```python
dat_capture.push_eval_timestamp(eval_timestamp_df)
```
TODO: add use cases of input dataframes: metrics, prediction, datapoint

## About [PulsarML](https://pulsar.ml/)

PulsarML is a project helping with monitoring your models and gain powerful insights into its performance.

We released two Open Source packages :

- [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) :  lightweight python SDK enabling data collection of features, predictions and metadata from an ML model serving code/micro-service
- [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) : library for evaluating and monitoring data and concept drift with an extensive set of metrics. It also offers the possibility to use custom metrics defined by the user.

We also created [pulsar demo](https://github.com/Rocket-Science-Development/pulsar_demo) to display an example use-case showing how to leverage both packages to implement model monitoring and performance management.

Want to interact with the community? join our [slack channel](https://pulsarml.slack.com)

Powered by [Rocket Science Development](https://rocketscience.one/)

TODO: add use cases of input dataframes: metrics, prediction, datapoint
