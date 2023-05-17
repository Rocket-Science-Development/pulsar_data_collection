# pulsar_data_collection

[![PyPI Latest Release](https://img.shields.io/pypi/v/pulsar-data-collection.svg)](https://pypi.org/project/pulsar-data-collection/)
[![Package Status](https://img.shields.io/pypi/status/pulsar-data-collection.svg)](https://pypi.org/project/pulsar-data-collection/)
[![License](https://img.shields.io/pypi/l/pulsar-data-collection.svg)](https://github.com/Rocket-Science-Development/pulsar_data_collection/blob/main/LICENSE)
[![Coverage](https://codecov.io/github/pandas-dev/pandas/coverage.svg?branch=main)](https://codecov.io/gh/pandas-dev/pandas)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What is it?

**pulsar-data-collection** is a package for collecting and pushing data (features, predictions, and metadata) from an inference/prediction Jupyter notebook or Python micro-service serving a ML model (Flask, FastAPI) to a Database. It aims to provide an easy and flexible way to log data point used to perform predictions, prediction, and other relevant metadate informing the context in which the model is perform its predection. Once this data is stored, it will be used to compute metrics in order to provide ML models monitoring using our [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) package. Further demonstration on how these packages are leveraged can found in [pulsar-demo](https://github.com/Rocket-Science-Development/pulsar_demo).

We currently support writing to InfluxDB v2.6.1, support for other technologies is coming

## Main Features

Here are just a few of the things that pandas does well:

- Collect data related to a model's lifecycle in production, i.e, data points, predictions, and metadata
- Can be used inside a Python inference micro-service or a Jupyter Notebook used to perform predictions
- Easy and flexible interface that has the ability to integrate with provided support of storage solution as well as the ability to easily integrate with custom storage solutions.
- Lightweight package to limit the impact on the inference/prediction service performance

## Where to get it

The source code is currently hosted on GitHub at: [https://github.com/Rocket-Science-Development/pulsar_data_collection](https://github.com/Rocket-Science-Development/pulsar_data_collection)

Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/pulsar-data-collection/)
Install Pulsar Data Collection with pip:

```sh
pip install pulsar-data-collection
```

### Example usage

```python
import pickle as pkl
from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

import pandas as pd

from pulsar_data_collection.models import DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse

# init paths
model_path = Path("path_to_model")
inference_dataset = pd.read_csv("path_of_data_for_prediction", header=0)
reference_data = "path_or_location_reference_data_used_to_train_the_model"

params = PulseParameters(
    model_id="model_id",
    model_version="model_version",
    data_id="reference_data_id",
    reference_data_storage=reference_data,
    target_name="target_feature_name",
    storage_engine="influxdb",
    timestamp_column_name="_time",
    login={
        "url": "url_influxdb",
        "token": "mytoken",
        "org": "pulsarml",
        "bucket_name": "demo",
    },
    other_labels={"timezone": "EST", "reference_dataset": reference_data},
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
```

## About [PulsarML](https://pulsar.ml/)

PulsarML is a project helping with monitoring your models and gain powerful insights into its performance.

We released two Open Source packages :

- [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) :  lightweight python package enabling data collection of features, predictions and metadata from an ML model serving code/micro-service

- [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) : library for evaluating and monitoring data and concept drift with an extensive set of metrics. It also offers the possibility to use custom metrics defined by the user.

We also created [pulsar demo](https://github.com/Rocket-Science-Development/pulsar_demo) to display an example use-case showing how to leverage both packages to implement model monitoring and performance management.

Want to interact with the community? join our [slack channel](https://pulsarml.slack.com)

Powered by [Rocket Science Development](https://rocketscience.one/)


