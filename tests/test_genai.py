"""GenAI collection: OpenTelemetry spans -> agent runs, tool calls and LLM calls (no Docker needed)."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from pulsar_data_collection.genai import ListSink, PulsarSpanExporter, agent_span, llm_span, record_llm_response, tool_span
from pulsar_data_collection.genai import semconv as sc
from pulsar_data_collection.genai.trajectory import SpanView, TrajectoryAssembler

ALLOWED = {"search_programs", "check_eligibility", "create_file"}


@pytest.fixture
def pipeline():
    sink = ListSink()
    exporter = PulsarSpanExporter(sink, allowed_tools=ALLOWED)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), sink, exporter
    provider.shutdown()


def chat(tracer, model="llama3.2:3b", tokens=(100, 20), finish="tool_calls"):
    with llm_span(tracer, model=model, provider="ollama", temperature=0.0) as span:
        record_llm_response(span, response_model=model, input_tokens=tokens[0], output_tokens=tokens[1], finish_reason=finish)


def run_agent(tracer, failing_tool=False, loop=False, out_of_scope=False, sub_agent=False):
    with agent_span(tracer, "permit_assistant", prompt_version="v3") as root:
        chat(tracer)
        with tool_span(tracer, "search_programs", arguments={"q": "renovation"}):
            pass
        chat(tracer)
        if failing_tool:
            with pytest.raises(TimeoutError):
                with tool_span(tracer, "check_eligibility", arguments={"id": 1}):
                    raise TimeoutError("eligibility service down")
        if loop:
            for _ in range(3):
                with tool_span(tracer, "search_programs", arguments={"q": "renovation"}):
                    pass
        if out_of_scope:
            with tool_span(tracer, "delete_file", arguments={"id": 7}):
                pass
        if sub_agent:
            with agent_span(tracer, "document_checker"):
                chat(tracer, tokens=(50, 10), finish="stop")
                with tool_span(tracer, "check_eligibility", arguments={"id": 2}):
                    pass
        chat(tracer, finish="stop")
        root.set_attribute(sc.TERMINATION, "completed")
        root.set_attribute(sc.TASK_SUCCESS, not failing_tool)


def test_simple_run_produces_one_record_of_each_kind(pipeline):
    tracer, sink, _ = pipeline
    run_agent(tracer)
    (run,) = sink.of("agent_run")
    assert run["agent_name"] == "permit_assistant"
    assert run["steps"] == 3 and run["tool_calls"] == 1 and run["distinct_tools"] == 1
    assert run["input_tokens"] == 300 and run["output_tokens"] == 60
    assert run["termination"] == "completed" and run["task_success"] is True
    assert run["tool_sequence"] == "search_programs" and run["first_tool"] == "search_programs"
    assert run["request_model"] == "llama3.2:3b" and run["prompt_version"] == "v3"
    assert run["loop_detected"] is False and run["out_of_scope_calls"] == 0 and run["delegation_depth"] == 0
    assert len(sink.of("llm_call")) == 3 and len(sink.of("tool_call")) == 1
    llm = sink.of("llm_call")[0]
    assert llm["request_model"] == "llama3.2:3b" and llm["temperature"] == 0.0 and llm["finish_reason"] == "tool_calls"


def test_failures_loops_scope_and_delegation(pipeline):
    tracer, sink, _ = pipeline
    run_agent(tracer, failing_tool=True, loop=True, out_of_scope=True, sub_agent=True)
    runs = sink.of("agent_run")
    assert len(runs) == 1  # the sub-agent is part of the run, not a separate run
    run = runs[0]
    assert run["tool_calls"] == 7  # search + failing check + 3 loop + delete + sub-agent check
    assert run["tool_error_rate"] == pytest.approx(1 / 7)
    assert run["repeated_calls"] == 2 and run["loop_detected"] is True  # 3 identical calls in a row = 2 repeats
    assert run["out_of_scope_calls"] == 1
    assert run["delegation_depth"] == 1
    assert run["task_success"] is False
    tools = {(t["tool_name"], t["error"], t["out_of_scope"]) for t in sink.of("tool_call")}
    assert ("check_eligibility", True, False) in tools and ("delete_file", False, True) in tools


def test_no_prompt_or_argument_content_is_written(pipeline):
    tracer, sink, _ = pipeline
    run_agent(tracer)
    for _, record, _ in sink.records:
        assert "renovation" not in str(record)


def test_aliases_from_other_sdks_are_normalized():
    assembler = TrajectoryAssembler()
    span = SpanView(
        trace_id=1,
        span_id=2,
        parent_id=None,
        name="llm",
        start_ns=0,
        end_ns=10**9,
        status_error=False,
        attributes={sc.OPERATION: "chat", "llm.model_name": "granite3.3:2b", "llm.token_count.prompt": 12},
    )
    ((record_type, record, _),) = assembler.add(span)
    assert record_type == "llm_call" and record["request_model"] == "granite3.3:2b" and record["input_tokens"] == 12


def test_trace_without_local_root_is_assembled_after_timeout():
    assembler = TrajectoryAssembler(trace_timeout_s=5)
    # the agent span's parent (e.g. an HTTP server span) lives in another process and never arrives here
    tool = SpanView(1, 3, 2, "t", 0, 10, False, {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: "search_programs"})
    agent = SpanView(1, 2, 99, "a", 0, 20, False, {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "x"})
    assert [r[0] for r in assembler.add(tool, now_ns=0)] == ["tool_call"]
    assert assembler.add(agent, now_ns=1) == []
    assert assembler.flush_idle(now_ns=2 * 10**9) == []  # not idle yet
    ((record_type, run, _),) = assembler.flush_idle(now_ns=10 * 10**9)
    assert record_type == "agent_run" and run["tool_calls"] == 1


def test_shutdown_assembles_pending_traces():
    sink = ListSink()
    exporter = PulsarSpanExporter(sink)
    exporter.assembler.add(SpanView(5, 6, 77, "a", 0, 10, False, {sc.OPERATION: sc.OP_INVOKE_AGENT}))
    exporter.shutdown()
    assert len(sink.of("agent_run")) == 1
