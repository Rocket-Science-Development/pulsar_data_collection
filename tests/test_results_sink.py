import time
from datetime import datetime, timezone

from pulsar_data_collection.results import STATUS, StatsReporter, result_point

T = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def line(result, **kwargs):
    return result_point(result, T, **kwargs).to_line_protocol()


def test_fairness_result_becomes_one_point_with_tags_and_fields():
    result = {
        "metric_type": "fairness",
        "metric_name": "disparate_impact_ratio",
        "feature_name": "language",
        "group": "language=other",
        "reference_group": "language=fr",
        "metric_value": 0.62,
        "ci_low": 0.55,
        "ci_high": 0.70,
        "threshold": [0.8, 1.25],
        "n": 1200,
        "drift_status": True,
        "test_type": "fixed",
        "method": "stratified bootstrap",
    }
    text = line(result, tags={"model_id": "elig", "model_version": "1", "campaign": None})
    assert text.startswith("pulsar_results,")
    for part in (r"group=language\=other", "metric_name=disparate_impact_ratio", "model_id=elig", "test_type=fixed"):
        assert part in text
    assert "campaign" not in text  # None tags are left out
    for part in ("metric_value=0.62", "ci_low=0.55", "threshold_low=0.8", "threshold_high=1.25", "n=1200", "status=1i"):
        assert part in text
    assert 'method="stratified bootstrap"' in text


def test_status_codes_and_missing_values():
    assert STATUS == {True: 1, False: 0, None: -1}
    inconclusive = line({"metric_type": "fairness", "metric_name": "x", "metric_value": None, "drift_status": None})
    assert "status=-1i" in inconclusive and "metric_value" not in inconclusive
    suppressed = line({"metric_type": "fairness", "metric_name": "x", "suppressed": True, "drift_status": None})
    assert "suppressed=true" in suppressed
    nan = line({"metric_type": "drift", "metric_name": "ks", "metric_value": float("nan"), "drift_status": False})
    assert "metric_value" not in nan and "status=0i" in nan


def test_extra_numeric_fields_are_kept():
    text = line(
        {
            "metric_type": "drift",
            "metric_name": "ks_2samp",
            "metric_value": 0.01,
            "drift_status": True,
            "p_adjusted": 0.03,
            "confirmed": True,
            "note": "text is ignored",
        }
    )
    assert "p_adjusted=0.03" in text and "confirmed=true" in text and "note" not in text


def test_metric_results_objects_are_accepted():
    from pydantic import BaseModel

    class Result(BaseModel):  # stands for pulsar_metrics.MetricResults
        metric_type: str
        metric_name: str
        metric_value: float
        drift_status: bool

    assert "metric_value=0.5" in line(Result(metric_type="drift", metric_name="psi", metric_value=0.5, drift_status=False))


class FakePulse:
    model_id, model_version = "m", "1"

    def stats(self):
        return {
            "mode": "background",
            "captured_rows": 10,
            "written_rows": 8,
            "storage_available": True,
            "last_error": "TimeoutError",
        }


class FakeSink:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def write_stats(self, stats, tags=None):
        if self.fail:
            raise ConnectionError("down")
        self.calls.append((stats, tags))


def test_stats_reporter_publishes_periodically_and_at_stop():
    sink = FakeSink()
    with StatsReporter(FakePulse(), sink, interval_s=0.05, tags={"environment": "test"}):
        time.sleep(0.2)
    assert len(sink.calls) >= 3
    stats, tags = sink.calls[-1]
    assert stats["written_rows"] == 8 and tags == {"model_id": "m", "model_version": "1", "environment": "test"}


def test_stats_reporter_never_raises():
    reporter = StatsReporter(FakePulse(), FakeSink(fail=True), interval_s=0.05).start()
    time.sleep(0.12)
    reporter.stop()
    assert reporter.errors >= 2
