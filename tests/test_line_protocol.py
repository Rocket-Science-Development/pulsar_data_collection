"""Line protocol encoder of the write path: same lines as influxdb_client's DataframeSerializer, much cheaper per frame."""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from influxdb_client import WritePrecision
from influxdb_client.client.write.dataframe_serializer import DataframeSerializer
from influxdb_client.client.write_api import PointSettings

from pulsar_data_collection.database_connectors.influxdb import InfluxdbActions, encode_lines

START = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
TAGS = {"model_id": "m 1", "model_version": "1", "data_id": "d,x=y"}


def reference(frame, measurement="m_1_input_data", tags=TAGS) -> list:
    """What influxdb_client produces"""
    return DataframeSerializer(
        frame,
        PointSettings(**{k: str(v) for k, v in tags.items()}),
        precision=WritePrecision.NS,
        data_frame_measurement_name=measurement,
        data_frame_timestamp_column="_time",
    ).serialize()


def prepared(columns: dict) -> pd.DataFrame:
    return InfluxdbActions.prepare_frame(pd.DataFrame(columns), None, START)


def frames() -> dict:
    rng = np.random.default_rng(3)
    n = 40
    floats = rng.normal(size=n) * 10.0 ** rng.integers(-8, 20, n)
    with_nan = rng.normal(size=n)
    with_nan[::3] = np.nan
    first_nan = rng.normal(size=n)
    first_nan[:5] = np.nan
    text = np.array(["plain", 'with "quotes"', "back\\slash", "comma, space=equal", "{braces}"] * 8, dtype=object)
    text_nan = text.copy()
    text_nan[::4] = None
    return {
        "floats and int prediction": prepared(
            {**{f"f{i}": rng.normal(size=n) for i in range(20)}, "y_pred": rng.integers(0, 2, n)}
        ),
        "extreme floats": prepared({"x": floats, "y": [np.inf, -0.0, 1e-320, 5e-324, 1.7976931348623157e308] * 8}),
        "float32 and unsigned": prepared({"a": rng.normal(size=n).astype("float32"), "b": np.arange(n, dtype="uint16")}),
        "booleans": prepared({"ok": rng.integers(0, 2, n).astype(bool), "x": rng.normal(size=n)}),
        "text": prepared({"label": text, "x": rng.normal(size=n)}),
        "missing values": prepared({"a": with_nan, "b": text_nan, "c": rng.normal(size=n)}),
        "missing first field": prepared({"a": first_nan, "b": with_nan}),
        "rows without any field": prepared({"a": with_nan, "b": with_nan.copy()}),
        "nullable dtypes": prepared(
            {
                "n_int": pd.array([1, None, 3, 4] * 10, dtype="Int64"),
                "n_float": pd.array([1.5, 2.5, None, 4.0] * 10, dtype="Float64"),
                "n_bool": pd.array([True, None, False, True] * 10, dtype="boolean"),
                "n_str": pd.array(["a", None, "c", "d"] * 10, dtype="string"),
                "n_cat": pd.Categorical(["u", "v", None, "u"] * 10),
            }
        ),
        "escaped keys": prepared({"a b": rng.normal(size=n), "c,d": rng.normal(size=n), "e=f": rng.normal(size=n), "{g}": 1.0}),
        "column overrides a tag": prepared({"model_id": rng.normal(size=n), "x": rng.normal(size=n)}),
        "naive and nanosecond times": pd.DataFrame(
            {"x": rng.normal(size=n), "_time": pd.date_range("2026-11-01", periods=n, freq="7ns").as_unit("ns")}
        ),
        "other time zone": pd.DataFrame(
            {"x": rng.normal(size=n), "_time": pd.date_range("2026-11-01", periods=n, freq="1s", tz="America/Toronto")}
        ),
        "formatted in several chunks": prepared({"a": np.tile(with_nan, 25), "b": np.arange(25 * n), "c": np.tile(text_nan, 25)}),
    }


@pytest.mark.parametrize("case", list(frames()))
def test_lines_are_those_of_the_client_serializer(case):
    frame = frames()[case]
    assert encode_lines(frame, "m_1_input_data", "_time", TAGS) == reference(frame)
    assert InfluxdbActions.serialize_frame(frame, "m_1_input_data", "_time", TAGS) == reference(frame)


def test_frames_the_encoder_does_not_handle_go_to_the_client_serializer():
    no_time = pd.DataFrame({"x": [1.0, 2.0], "_time": ["2026-11-01", "2026-11-02"]})  # text timestamps
    assert encode_lines(no_time, "m", "_time", TAGS) is None
    assert InfluxdbActions.serialize_frame(no_time, "m", "_time", TAGS) == reference(no_time, "m")
    numbered = prepared({0: [1.0, 2.0]})
    assert encode_lines(numbered, "m", "_time", TAGS) is None


def test_a_field_named_like_part_of_the_timestamp_column_is_written():
    # The client serializer took any column whose name is part of "_time" ("e", "time"...) as the timestamp: the whole
    # batch could not be encoded and was refused
    frame = prepared({"time": [1.0, 2.0], "e": [3.0, 4.0], "x": [5.0, 6.0]})
    lines = InfluxdbActions.serialize_frame(frame, "m", "_time", {"model_id": "a"})
    stamp = int(pd.Timestamp(START).value)
    assert lines == [f"m,model_id=a e=3.0,time=1.0,x=5.0 {stamp}", f"m,model_id=a e=4.0,time=2.0,x=6.0 {stamp + 1000}"]


def test_small_frames_are_encoded_several_times_faster():
    # the shared writer encodes one frame per model in each request: 50 rows per frame with 100 models
    rng = np.random.default_rng(0)
    frame = InfluxdbActions.prepare_frame(
        pd.DataFrame(rng.normal(size=(50, 20)), columns=[f"f{i}" for i in range(20)]),
        pd.DataFrame({"y_pred": rng.integers(0, 2, 50)}),
        START,
    )

    def best(encode) -> float:
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(20):
                encode()
            timings.append(time.perf_counter() - started)
        return min(timings)

    fast = best(lambda: InfluxdbActions.serialize_frame(frame, "m_1_input_data", "_time", TAGS))
    slow = best(lambda: reference(frame))
    assert slow / fast > 3  # about 5 to 8 measured
