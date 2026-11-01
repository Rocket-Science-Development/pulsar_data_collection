"""End to end: OpenTelemetry GenAI spans -> PulsarSpanExporter -> PulseSink -> InfluxDB 2.6 (Docker)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from influxdb_client import InfluxDBClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from pulsar_data_collection.genai import PulsarSpanExporter, PulseSink, agent_span, llm_span, record_llm_response, tool_span
from pulsar_data_collection.models import CaptureOptions, PulseParameters


def is_responsive(**kwargs):
    try:
        with InfluxDBClient(url=kwargs["url"], token=kwargs["token"], org=kwargs["org"]) as conn:
            return bool(conn.ping())
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def db_login(docker_ip):
    return {"url": f"http://{docker_ip}:8086/", "token": "mytoken", "org": "pulsarml", "bucket_name": "demo"}


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return Path("tests/db/influxdb/docker-compose.yaml")


@pytest.fixture(scope="session")
def run_influxdb(db_login, docker_services):
    docker_services.wait_until_responsive(timeout=30, pause=0.5, check=lambda: is_responsive(**db_login))


@pytest.mark.usefixtures("run_influxdb")
def test_agent_traces_are_written_by_record_type(db_login):
    run_id = datetime.now(timezone.utc).strftime("genai%H%M%S%f")
    params = PulseParameters(
        model_id="permit_assistant",
        model_version="1.0",
        data_id="production",
        reference_data_storage="none",
        target_name="task_success",
        storage_engine="influxdb",
        timestamp_column_name="_time",
        login=db_login,
        additional_tags={"run_id": run_id},
    )
    sink = PulseSink(params, CaptureOptions(flush_interval_s=0.2))
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink, allowed_tools={"search_programs"})))
    tracer = provider.get_tracer("test")

    for _ in range(3):
        with agent_span(tracer, "permit_assistant"):
            with llm_span(tracer, model="llama3.2:3b", provider="ollama") as span:
                record_llm_response(span, input_tokens=80, output_tokens=12, finish_reason="tool_calls")
            with tool_span(tracer, "search_programs", arguments={"q": "x"}):
                pass
            with llm_span(tracer, model="llama3.2:3b", provider="ollama") as span:
                record_llm_response(span, input_tokens=120, output_tokens=40, finish_reason="stop")
    provider.shutdown()  # assembles, writes and closes the Pulse instances
    assert all(stats["written_rows"] == stats["captured_rows"] for stats in sink.stats().values())

    start = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = {}
    with InfluxDBClient(url=db_login["url"], token="mytoken", org="pulsarml") as client:
        for record_type in ("agent_run", "tool_call", "llm_call"):
            query = f"""
            from(bucket: "demo")
            |> range(start: {start}, stop: 1m)
            |> filter(fn: (r) => r["_measurement"] == "permit_assistant_1.0_input_data")
            |> filter(fn: (r) => r["run_id"] == "{run_id}" and r["record_type"] == "{record_type}")
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            """
            frame = client.query_api().query_data_frame(query=query)
            counts[record_type] = len(frame)
            if record_type == "agent_run":
                assert set(frame["steps"]) == {2} and set(frame["tool_sequence"]) == {"search_programs"}
                assert set(frame["schema"]) == {"pulsar-genai/1"}
    assert counts == {"agent_run": 3, "tool_call": 3, "llm_call": 6}
