from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from pulsar_data_collection.models import DataWithPrediction, PulseParameters

PARAMS = {
    "model_id": "m",
    "model_version": "1",
    "data_id": "d",
    "reference_data_storage": "ref.csv",
    "target_name": "y",
    "storage_engine": "influxdb",
    "login": {"url": "http://localhost:8086", "token": "t", "org": "o", "bucket_name": "b"},
}


def test_pulse_parameters_accept_known_fields():
    params = PulseParameters(**PARAMS, additional_tags={"environment": "test"})
    assert params.additional_tags == {"environment": "test"}


def test_pulse_parameters_reject_unknown_fields():
    # other_labels was used in the former README: it was silently ignored and the tags were lost
    with pytest.raises(ValidationError, match="other_labels"):
        PulseParameters(**PARAMS, other_labels={"environment": "test"})


def test_pulse_parameters_reject_unknown_storage_engine():
    with pytest.raises(ValidationError, match="not supported"):
        PulseParameters(**{**PARAMS, "storage_engine": "unknown"})


def test_data_with_prediction_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="prediction"):
        DataWithPrediction(
            data_points=pd.DataFrame({"a": [1.0]}),
            predictions=pd.DataFrame({"y": [0]}),
            prediction=pd.DataFrame({"y": [0]}),
            timestamp=datetime.now(timezone.utc),
            features_names=["a"],
        )
