"""Agent run assembly and features: long and stuck runs, per-tool failures, loops, tool catalogue, rule violations.

Each test reproduces a limit found by the night lab (M24, M25, M27 to M30, C3 of notes/laboratoire/rapport_nuit.md).
"""

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from pulsar_data_collection.genai import ListSink, PulsarSpanExporter, agent_span, llm_span, record_llm_response, tool_span
from pulsar_data_collection.genai import semconv as sc
from pulsar_data_collection.genai.trajectory import SpanView, TrajectoryAssembler

S = 10**9
ALLOWED = {"search_programs", "check_eligibility", "compute_amount"}


def tool(span_id, name, start_s, end_s, parent=1, arguments="{}", error=None, **attributes):
    attrs = {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: name, sc.TOOL_CALL_ARGUMENTS: arguments, **attributes}
    if error:
        attrs[sc.ERROR_TYPE] = error
    return SpanView(7, span_id, parent, f"execute_tool {name}", int(start_s * S), int(end_s * S), bool(error), attrs)


def chat(span_id, start_s, end_s, parent=1):
    attrs = {sc.OPERATION: sc.OP_CHAT, sc.REQUEST_MODEL: "llama3.2:3b", sc.INPUT_TOKENS: 10, sc.OUTPUT_TOKENS: 5}
    return SpanView(7, span_id, parent, "chat", int(start_s * S), int(end_s * S), False, attrs)


def root(end_s, start_s=0.0, **attributes):
    attrs = {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "permit_assistant", **attributes}
    return SpanView(7, 1, None, "invoke_agent", int(start_s * S), int(end_s * S), False, attrs)


def run_of(assembler, calls, **root_attributes):
    """agent_run record of a run made of the given tool spans"""
    for span in calls:
        assembler.add(span)
    records = assembler.add(root(100, **root_attributes))
    (run,) = [r for t, r, _ in records if t == "agent_run"]
    return run


@pytest.fixture
def pipeline():
    sink = ListSink()
    exporter = PulsarSpanExporter(sink, allowed_tools=ALLOWED, max_run_s=60)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, provider.get_tracer("test"), sink, exporter
    provider.shutdown()


# --- M24: a run longer than trace_timeout_s kept none of its calls


def test_run_longer_than_trace_timeout_keeps_its_llm_and_tool_calls():
    # replay of the lab: tool span ends at 5 s, another export at 35.5 s, root at 45 s
    assembler = TrajectoryAssembler(trace_timeout_s=30)
    assembler.add(chat(2, 0, 4), now_ns=4 * S)
    assembler.add(tool(3, "check_eligibility", 4, 5), now_ns=5 * S)
    assert assembler.flush_idle(now_ns=int(35.5 * S)) == []
    records = assembler.add(root(45), now_ns=45 * S)
    (run,) = [r for t, r, _ in records if t == "agent_run"]
    assert run["tool_calls"] == 1 and run["steps"] == 1 and run["tool_sequence"] == "check_eligibility"
    assert run["incomplete"] is False
    assert assembler.stats()["partial_traces_dropped"] == 0


def test_trace_whose_root_never_arrives_is_given_up_and_counted():
    assembler = TrajectoryAssembler(orphan_timeout_s=120, max_run_s=None)
    assembler.add(tool(3, "check_eligibility", 0, 5), now_ns=5 * S)
    assert assembler.flush_idle(now_ns=60 * S) == []  # still waiting for the root
    assert assembler.flush_idle(now_ns=200 * S) == []  # no agent span received: nothing to assemble
    stats = assembler.stats()
    assert stats["partial_traces_dropped"] == 1 and stats["partial_spans_dropped"] == 1 and stats["pending_traces"] == 0


def test_overflow_of_pending_traces_is_counted_as_partial():
    assembler = TrajectoryAssembler(max_pending_traces=1, max_run_s=None)
    first = tool(3, "search_programs", 0, 1)
    second = SpanView(8, 30, 1, "t", 0, S, False, {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: "search_programs"})
    assembler.add(first)
    assembler.add(second)
    assert assembler.stats()["partial_traces_dropped"] == 1 and assembler.stats()["pending_traces"] == 1


def test_exporter_exposes_assembly_stats(pipeline):
    _, tracer, _, exporter = pipeline
    with agent_span(tracer, "permit_assistant"):
        with tool_span(tracer, "search_programs"):
            pass
    stats = exporter.stats()
    assert stats["partial_traces_dropped"] == 0 and stats["pending_traces"] == 0 and stats["open_runs"] == 0


# --- M25: a run that never ends was invisible


def test_run_that_never_ends_is_reported_incomplete_once(pipeline):
    provider, tracer, sink, exporter = pipeline
    provider.add_span_processor(exporter.run_tracker(check_interval_s=None))
    span = tracer.start_span("invoke_agent permit_assistant", attributes={sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "a"})
    start = span.start_time
    assert exporter.check_open_runs(now_ns=start + 30 * S) == 0
    assert exporter.check_open_runs(now_ns=start + 61 * S) == 1
    assert exporter.check_open_runs(now_ns=start + 120 * S) == 0  # reported once
    (run,) = sink.of("agent_run")
    assert run["termination"] == "incomplete" and run["incomplete"] is True and run["agent_name"] == "a"
    assert run["latency_s"] == pytest.approx(61)
    assert exporter.stats()["incomplete_runs"] == 1 and exporter.stats()["open_runs"] == 1
    span.end()  # the run finally ends: its complete record follows
    final = sink.of("agent_run")[-1]
    assert final["incomplete"] is False and final["termination"] == "completed"
    assert exporter.stats()["open_runs"] == 0


def test_open_run_is_reported_at_shutdown():
    sink = ListSink()
    exporter = PulsarSpanExporter(sink)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(exporter.run_tracker(check_interval_s=None))
    tracer = provider.get_tracer("test")
    tracer.start_span("invoke_agent x", attributes={sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "x"})
    provider.shutdown()
    (run,) = sink.of("agent_run")
    assert run["termination"] == "incomplete" and run["incomplete"] is True


def test_sub_agent_start_is_not_a_separate_open_run(pipeline):
    provider, tracer, sink, exporter = pipeline
    provider.add_span_processor(exporter.run_tracker(check_interval_s=None))
    with agent_span(tracer, "permit_assistant"):
        with agent_span(tracer, "document_checker"):
            assert exporter.stats()["open_runs"] == 1
    assert exporter.stats()["open_runs"] == 0 and len(sink.of("agent_run")) == 1


def test_llm_calls_without_agent_never_make_a_fake_run():
    # review: an LLM-only trace under a long-lived parent got an agent_run named "unknown"
    assembler = TrajectoryAssembler(max_run_s=600)
    assembler.add(chat(2, 0, 1, parent=10), now_ns=S)
    assert assembler.flush_idle(now_ns=700 * S) == []
    assert assembler.stats()["incomplete_runs"] == 0


def test_completed_run_under_a_session_root_is_written_once():
    # review: a finished agent run under a chat session span was written incomplete, then complete
    assembler = TrajectoryAssembler(max_run_s=600)
    assembler.add(chat(21, 1, 2, parent=20), now_ns=2 * S)
    agent = SpanView(7, 20, 1, "invoke_agent", 0, 3 * S, False, {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "bot"})
    assembler.add(agent, now_ns=3 * S)
    assert assembler.flush_idle(now_ns=700 * S) == []
    session = SpanView(7, 1, None, "session", 0, 800 * S, False, {})
    runs = [r for t, r, _ in assembler.add(session, now_ns=800 * S) if t == "agent_run"]
    assert [(r["agent_name"], r["incomplete"], r["steps"]) for r in runs] == [("bot", False, 1)]


def test_hung_run_is_reported_once_even_after_its_trace_is_given_up():
    from pulsar_data_collection.genai.trajectory import OpenRun

    assembler = TrajectoryAssembler(max_run_s=600, orphan_timeout_s=3600)
    attrs = {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "bot"}
    assembler.run_started(OpenRun(7, 1, None, "invoke_agent bot", 0, attrs))
    assembler.add(chat(2, 1, 2), now_ns=2 * S)
    assert len(assembler.flush_idle(now_ns=700 * S)) == 1
    assert assembler.flush_idle(now_ns=3700 * S) == []  # trace given up: the run is not reported again
    assert assembler.flush_idle(now_ns=3800 * S) == []
    stats = assembler.stats()
    assert stats["incomplete_runs"] == 1 and stats["open_runs"] == 0 and not assembler._reported


def test_hung_run_without_any_ended_span_is_forgotten_after_orphan_timeout():
    from pulsar_data_collection.genai.trajectory import OpenRun

    assembler = TrajectoryAssembler(max_run_s=600, orphan_timeout_s=3600)
    assembler.run_started(OpenRun(7, 1, None, "a", 0, {sc.OPERATION: sc.OP_INVOKE_AGENT}))
    assert len(assembler.flush_idle(now_ns=700 * S)) == 1
    assert assembler.flush_idle(now_ns=3700 * S) == []
    assert assembler.stats()["open_runs"] == 0 and not assembler._reported and not assembler._open_tree


def test_sub_agent_below_an_intermediate_span_is_part_of_the_run(pipeline):
    provider, tracer, sink, exporter = pipeline
    provider.add_span_processor(exporter.run_tracker(check_interval_s=None))
    with agent_span(tracer, "permit_assistant"):
        with tool_span(tracer, "delegate"):  # a tool that starts a sub-agent
            with agent_span(tracer, "document_checker"):
                assert exporter.stats()["open_runs"] == 1
    assert exporter.stats()["open_runs"] == 0 and len(sink.of("agent_run")) == 1
    assert not exporter.assembler._open_tree


def test_run_still_open_is_inferred_from_its_ended_spans_without_tracker():
    assembler = TrajectoryAssembler(max_run_s=60)
    assembler.add(tool(3, "check_eligibility", 0, 5, **{sc.AGENT_NAME: "permit_assistant"}), now_ns=5 * S)
    assert assembler.flush_idle(now_ns=30 * S) == []
    ((record_type, run, _),) = assembler.flush_idle(now_ns=61 * S)
    assert record_type == "agent_run" and run["incomplete"] is True and run["tool_calls"] == 1
    assert run["agent_name"] == "permit_assistant"
    assert assembler.flush_idle(now_ns=90 * S) == []


def test_tracker_thread_reports_open_runs_by_itself():
    import time

    sink = ListSink()
    exporter = PulsarSpanExporter(sink, max_run_s=0.05)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(exporter.run_tracker(check_interval_s=0.02))
    span = provider.get_tracer("t").start_span("invoke_agent x", attributes={sc.OPERATION: sc.OP_INVOKE_AGENT})
    deadline = time.monotonic() + 5
    while not sink.of("agent_run") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sink.of("agent_run")[0]["incomplete"] is True
    span.end()
    provider.shutdown()


# --- M27: timeouts on one tool were never flagged


def test_per_tool_error_and_timeout_counts(pipeline):
    _, tracer, sink, _ = pipeline
    with agent_span(tracer, "permit_assistant"):
        with tool_span(tracer, "search_programs"):
            pass
        for i in range(4):
            try:
                with tool_span(tracer, "check_eligibility", arguments={"id": i}):
                    if i % 2:
                        raise TimeoutError("eligibility service timed out")
            except TimeoutError:
                pass
    (run,) = sink.of("agent_run")
    assert run["tool_check_eligibility_calls"] == 4
    assert run["tool_check_eligibility_errors"] == 2 and run["tool_check_eligibility_timeouts"] == 2
    assert run["tool_search_programs_calls"] == 1 and run["tool_search_programs_errors"] == 0
    assert "tool_compute_amount_calls" not in run  # only the tools called get counters (tool_fields="called")
    assert run["tool_timeouts"] == 2 and run["tool_errors"] == 2 and run["max_tool_error_rate"] == pytest.approx(0.5)
    assert run["tool_status_counts"] == "ok=3,timeout=2"
    timeouts = [t for t in sink.of("tool_call") if t["timeout"]]
    assert len(timeouts) == 2 and all(t["status"] == "timeout" and t["error"] for t in timeouts)


def test_error_that_is_not_a_timeout():
    run = run_of(TrajectoryAssembler(), [tool(3, "check_eligibility", 0, 1, error="ValueError")])
    assert run["tool_check_eligibility_errors"] == 1 and run["tool_check_eligibility_timeouts"] == 0
    assert run["tool_status_counts"] == "error=1"


# --- M28: a tool answering "retry" with unusable data was invisible


def test_retry_status_and_invalid_output_are_soft_failures(pipeline):
    from pulsar_data_collection.genai import record_tool_result

    _, tracer, sink, _ = pipeline
    with agent_span(tracer, "permit_assistant"):
        with tool_span(tracer, "check_eligibility") as span:
            assert record_tool_result(span, {"status": "retry", "data": "#@!"}) == "retry"
        with tool_span(tracer, "check_eligibility") as span:
            record_tool_result(span, {"eligible": "maybe"}, validator=lambda out: isinstance(out["eligible"], bool))
        with tool_span(tracer, "check_eligibility") as span:
            record_tool_result(span, {"eligible": True}, validator=lambda out: isinstance(out["eligible"], bool))
    (run,) = sink.of("agent_run")
    assert run["tool_error_rate"] == 0.0  # not failures in the strict sense...
    assert run["tool_soft_failures"] == 2 and run["tool_check_eligibility_soft_failures"] == 2  # ...but counted
    assert run["soft_failure_rate"] == pytest.approx(2 / 3)
    assert run["tool_status_counts"] == "invalid=1,ok=1,retry=1"
    assert [t["soft_failure"] for t in sink.of("tool_call")] == [True, True, False]
    for _, record, _ in sink.records:
        assert "#@!" not in str(record)  # the output itself is never written


def test_validator_that_raises_means_invalid_output():
    from pulsar_data_collection.genai import record_tool_result

    span = SimpleNamespace(attributes={})
    span.set_attribute = lambda key, value: span.attributes.__setitem__(key, value)
    record_tool_result(span, None, validator=lambda out: out["x"])
    assert span.attributes == {sc.TOOL_OUTPUT_VALID: False}


# --- M29: loop_detected only saw consecutive calls identical byte for byte


def test_ping_pong_is_a_loop():
    calls = [tool(10 + i, "search_programs" if i % 2 == 0 else "check_eligibility", i, i + 0.5) for i in range(6)]
    run = run_of(TrajectoryAssembler(), calls)
    assert run["repeated_calls"] == 0  # historical field unchanged
    assert run["repeated_calls_window"] == 4 and run["loop_detected"] is True
    assert run["tool_cycle_repeats"] == 3


def test_same_call_with_arguments_of_another_type_is_a_loop():
    args = ['{"income": 36000}', '{"income": 36000.0}', '{"income": "36000"}', '{"Income": " 36000 "}']
    run = run_of(TrajectoryAssembler(), [tool(10 + i, "check_eligibility", i, i + 0.5, arguments=a) for i, a in enumerate(args)])
    assert run["repeated_calls"] == 0 and run["repeated_calls_window"] == 3 and run["loop_detected"] is True


def test_same_tool_with_other_arguments_is_counted_but_not_a_loop():
    calls = [tool(10 + i, "check_eligibility", i, i + 0.5, arguments=f'{{"id": {i}}}') for i in range(4)]
    run = run_of(TrajectoryAssembler(), calls)
    assert run["loop_detected"] is False and run["max_same_tool_calls"] == 4


def test_one_repeated_call_in_a_normal_run_is_not_a_loop():
    calls = [
        tool(10, "search_programs", 0, 1, arguments='{"q": "loge"}'),
        tool(11, "check_eligibility", 1, 2),
        tool(12, "search_programs", 2, 3, arguments='{"q": "loge"}'),
    ]
    run = run_of(TrajectoryAssembler(), calls)
    assert run["repeated_calls_window"] == 1 and run["loop_detected"] is False


def test_loop_window_and_threshold_are_configurable():
    calls = [tool(10 + i, "search_programs" if i % 2 == 0 else "check_eligibility", i, i + 0.5) for i in range(6)]
    assert run_of(TrajectoryAssembler(loop_window=1), calls)["loop_detected"] is False
    assert run_of(TrajectoryAssembler(loop_min_repeats=5), calls)["loop_detected"] is False


# --- M30: a removed tool was detected late


def test_removed_tool_shows_in_the_catalogue_from_the_first_run(pipeline):
    _, tracer, sink, _ = pipeline
    with agent_span(tracer, "permit_assistant", available_tools=["search_programs", "compute_amount"]):
        with tool_span(tracer, "search_programs"):
            pass
        try:
            with tool_span(tracer, "check_eligibility"):  # the model still calls the removed tool
                raise LookupError("unknown tool check_eligibility")
        except LookupError:
            pass
    (run,) = sink.of("agent_run")
    assert run["tool_catalog_size"] == 2 and len(run["tool_catalog_hash"]) == 12
    assert run["catalog_missing_tools"] == 1 and run["missing_tools"] == "check_eligibility"
    assert run["unavailable_tool_calls"] == 1


def test_catalogue_hash_changes_with_the_catalogue():
    first = run_of(TrajectoryAssembler(), [], **{sc.AVAILABLE_TOOLS: ["a", "b"]})
    same = run_of(TrajectoryAssembler(), [], **{sc.AVAILABLE_TOOLS: "b,a"})
    other = run_of(TrajectoryAssembler(), [], **{sc.AVAILABLE_TOOLS: ["a"]})
    assert first["tool_catalog_hash"] == same["tool_catalog_hash"] != other["tool_catalog_hash"]
    assert "catalog_missing_tools" not in first  # no allowed_tools to compare with


def test_unavailable_status_without_catalogue():
    run = run_of(TrajectoryAssembler(), [tool(3, "check_eligibility", 0, 1, **{sc.TOOL_STATUS: "unavailable"})])
    assert run["unavailable_tool_calls"] == 1 and "tool_catalog_hash" not in run


# --- C3: agent_run had no violation field


def test_no_violation_fields_without_rules():
    run = run_of(TrajectoryAssembler(), [tool(3, "check_eligibility", 0, 1)])
    assert "violated_rules" not in run and "rules_version" not in run


def test_violated_rules_from_a_callable():
    def rules(trace):
        return ["amount-after-eligibility"] if "compute_amount" in trace and "check_eligibility" not in trace else []

    assembler = TrajectoryAssembler(compliance=rules, rules_version="2026-11-01")
    run = run_of(assembler, [tool(3, "compute_amount", 0, 1)])
    assert run["violated_rules"] == "amount-after-eligibility" and run["violations"] == 1
    assert run["rules_version"] == "2026-11-01" and run["critical_violations"] == 0
    ok = run_of(TrajectoryAssembler(compliance=rules), [tool(3, "check_eligibility", 0, 1), tool(4, "compute_amount", 1, 2)])
    assert ok["violated_rules"] == "none" and ok["violations"] == 0 and ok["rules_version"] == "unversioned"


class FakeMonitor:
    """Same interface as pulsar_metrics.genai.ComplianceMonitor (violations, rules with id/severity, version)"""

    version = "rules-v3"
    rules = [SimpleNamespace(id="auth-before-file", severity="critical"), SimpleNamespace(id="polite", severity="minor")]

    def violations(self, trace):
        out = []
        if "create_file" in trace and ("authenticate" not in trace or trace.index("authenticate") > trace.index("create_file")):
            out.append("auth-before-file")
        if len(trace) > 2:
            out.append("polite")
        return out


def test_violated_rules_from_a_compliance_monitor(pipeline):
    sink = ListSink()
    exporter = PulsarSpanExporter(sink, compliance=FakeMonitor())
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    with agent_span(tracer, "permit_assistant"):
        for name in ("search_programs", "create_file", "authenticate"):
            with tool_span(tracer, name):
                pass
    provider.shutdown()
    (run,) = sink.of("agent_run")
    assert run["violated_rules"] == "auth-before-file,polite" and run["violations"] == 2
    assert run["critical_violations"] == 1 and run["rules_version"] == "rules-v3"


def test_faulty_rules_never_stop_the_collection():
    def broken(trace):
        raise RuntimeError("bad rule file")

    assembler = TrajectoryAssembler(compliance=broken)
    run = run_of(assembler, [tool(3, "check_eligibility", 0, 1)])
    assert "violated_rules" not in run and run["tool_calls"] == 1
    assert assembler.stats()["compliance_errors"] == 1


def test_with_the_real_compliance_monitor():
    compliance = pytest.importorskip("pulsar_metrics.genai.compliance")
    monitor = compliance.ComplianceMonitor(
        [{"id": "auth-before-file", "template": "precedence", "events": ["authenticate", "create_file"], "severity": "critical"}],
        version="v1",
    )
    run = run_of(TrajectoryAssembler(compliance=monitor), [tool(3, "create_file", 0, 1)])
    assert run["violated_rules"] == "auth-before-file" and run["critical_violations"] == 1 and run["rules_version"] == "v1"


def test_llm_helpers_still_feed_the_run(pipeline):
    _, tracer, sink, _ = pipeline
    with agent_span(tracer, "permit_assistant"):
        with llm_span(tracer, model="m", provider="ollama") as call:
            record_llm_response(call, input_tokens=3, output_tokens=1)
    (run,) = sink.of("agent_run")
    assert run["steps"] == 1 and run["tool_errors"] == 0 and run["tool_status_counts"] == "none"


# --- LA-01 (lab P2): the agent_run schema grew with the tool catalogue (2308 per-tool fields for 580 tools)

COUNTER_SUFFIXES = ("calls", "errors", "timeouts", "soft_failures")
RUN_TOTALS = {"tool_calls", "tool_errors", "tool_timeouts", "tool_soft_failures", "tool_args_type_errors"}


def per_tool_keys(run):
    return {k for k in run if k.startswith("tool_") and k.endswith(COUNTER_SUFFIXES)} - RUN_TOTALS


def counters(*tools):
    return {f"tool_{t}_{c}" for t in tools for c in COUNTER_SUFFIXES}


def test_per_tool_counters_do_not_grow_with_the_catalogue():
    catalogue = [f"tool_{i:03d}" for i in range(580)]
    calls = [tool(3, "tool_007", 0, 1), tool(4, "tool_042", 1, 2, error="ValueError")]
    assembler = TrajectoryAssembler(allowed_tools=set(catalogue[:50]))
    run = run_of(assembler, calls, **{sc.AVAILABLE_TOOLS: catalogue})
    assert per_tool_keys(run) == counters("tool_007", "tool_042")
    assert run["tool_tool_042_errors"] == 1 and run["max_tool_error_rate"] == 1.0 and run["tool_catalog_size"] == 580
    former = TrajectoryAssembler(allowed_tools=set(catalogue[:50]), tool_fields="all")
    assert len(per_tool_keys(run_of(former, calls, **{sc.AVAILABLE_TOOLS: catalogue}))) == 4 * 580  # kept as an option


def test_per_tool_counter_modes():
    calls = [tool(3, "check_eligibility", 0, 1), tool(4, "check_eligibility", 1, 2), tool(5, "search_programs", 2, 3)]
    allowed = run_of(TrajectoryAssembler(allowed_tools=ALLOWED, tool_fields="allowed"), calls)
    assert per_tool_keys(allowed) == counters(*ALLOWED) and allowed["tool_compute_amount_calls"] == 0
    top = run_of(TrajectoryAssembler(tool_fields="top_k", tool_fields_k=1), calls)
    assert per_tool_keys(top) == counters("check_eligibility")
    none = run_of(TrajectoryAssembler(tool_fields="none"), calls)
    assert per_tool_keys(none) == set() and none["tool_calls"] == 3
    with pytest.raises(ValueError):
        TrajectoryAssembler(tool_fields="allowed")  # needs allowed_tools
    with pytest.raises(ValueError):
        TrajectoryAssembler(tool_fields="every")


# --- LA-01 (lab P2): without labels, a 3b -> 8b model change was invisible; argument quality moved (854 vs 150)

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "compute_amount",
            "description": "not recorded",
            "parameters": {
                "type": "object",
                "properties": {"income": {"type": "integer"}, "region": {"type": "string", "enum": ["qc", "on"]}},
                "required": ["income"],
            },
        },
    },
    {"name": "search_programs", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}}},
]


def test_arguments_are_checked_against_the_tool_schemas(pipeline):
    _, tracer, sink, _ = pipeline
    with agent_span(tracer, "permit_assistant", available_tools=SCHEMAS):
        for arguments in (
            {"income": 36000, "region": "qc"},  # valid
            {"income": "36000"},  # number sent as a string
            {"region": "qc"},  # required argument missing
            {"income": 36000, "region": "bc"},  # outside the enum
            "{not json",
        ):
            with tool_span(tracer, "compute_amount", arguments=arguments):
                pass
        with tool_span(tracer, "search_programs", arguments={"topic": 3}):  # wrong type
            pass
        with tool_span(tracer, "check_eligibility", arguments={"id": 1}):  # no schema: not checked
            pass
    (run,) = sink.of("agent_run")
    assert run["tool_args_checked"] == 6 and run["tool_args_invalid"] == 5
    assert run["tool_args_invalid_rate"] == pytest.approx(5 / 6)
    assert run["tool_args_string_numbers"] == 1 and run["tool_args_type_errors"] == 2
    assert run["tool_args_missing_required"] == 1 and run["tool_args_unparsable"] == 1
    assert run["tool_catalog_size"] == 2  # the names are still recorded from the definitions
    assert all("36000" not in str(v) for r in sink.of("tool_call") + [run] for v in r.values())  # values never written


def test_tool_schemas_from_openllmetry_attributes_and_absent_without_definitions():
    assembler = TrajectoryAssembler()
    llm = chat(2, 0, 1)
    llm.attributes["llm.request.functions.0.name"] = "compute_amount"
    llm.attributes["llm.request.functions.0.parameters"] = '{"type": "object", "required": ["income"]}'
    assembler.add(llm)
    run = run_of(assembler, [tool(3, "compute_amount", 1, 2, arguments="{}")])
    assert run["tool_args_checked"] == 1 and run["tool_args_missing_required"] == 1
    plain = run_of(TrajectoryAssembler(), [tool(3, "compute_amount", 0, 1)])
    assert not any(k.startswith("tool_args_") for k in plain)


def test_first_step_without_tool_call():
    no_tool = TrajectoryAssembler()
    no_tool.add(chat(2, 0, 1))
    assert run_of(no_tool, [])["first_step_no_tool"] is True
    with_tool = TrajectoryAssembler()
    for span in (chat(2, 0, 1), tool(3, "check_eligibility", 1, 2), chat(4, 2, 3)):
        with_tool.add(span)
    assert run_of(with_tool, [])["first_step_no_tool"] is False
    late_tool = TrajectoryAssembler()
    for span in (chat(2, 0, 1), chat(3, 1, 2), tool(4, "check_eligibility", 2, 3)):
        late_tool.add(span)
    assert run_of(late_tool, [])["first_step_no_tool"] is True


# --- LA-02 (lab P2): the confirmation policy could not be checked on tool names alone, nor rules on arguments


class RichMonitor:
    """Monitor declaring accepts_rich_trace: keeps what it receives"""

    accepts_rich_trace = True
    version = "rich-v1"
    rules = [SimpleNamespace(id="confirm-before-cancel", severity="critical")]

    def __init__(self):
        self.traces = []

    def violations(self, trace):
        self.traces.append(trace)
        names = [e["name"] for e in trace]
        if "cancel_order" in names and "user_confirmation" not in names[: names.index("cancel_order")]:
            return ["confirm-before-cancel"]
        return []


def test_rich_trace_has_events_arguments_and_status_in_time_order():
    from pulsar_data_collection.genai import record_event

    monitor = RichMonitor()
    sink = ListSink()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink, compliance=monitor)))
    tracer = provider.get_tracer("t")
    with agent_span(tracer, "retail"):
        with tool_span(tracer, "get_order", arguments={"order_id": "W123"}):
            pass
        record_event("user_confirmation", answer_kind="yes")  # on the agent span, between the two calls
        try:
            with tool_span(tracer, "cancel_order", arguments={"order_id": "W123", "reason": "no longer needed"}):
                raise TimeoutError("timed out")
        except TimeoutError:
            pass
        with tool_span(tracer, "notify", arguments="not json") as span:
            span.add_event("unrelated")  # not a pulsar.event.*: ignored
    provider.shutdown()
    (trace,) = monitor.traces
    assert trace == [
        {"name": "get_order", "kind": "tool", "status": "ok", "args": {"order_id": "W123"}},
        {"name": "user_confirmation", "kind": "event"},
        {"name": "cancel_order", "kind": "tool", "status": "timeout", "args": {"order_id": "W123", "reason": "no longer needed"}},
        {"name": "notify", "kind": "tool", "status": "ok"},
    ]
    (run,) = sink.of("agent_run")
    assert run["violated_rules"] == "none" and run["rules_version"] == "rich-v1"
    assert run["tool_sequence"] == "get_order>cancel_order>notify"  # events are not tools
    assert all("W123" not in str(v) and "no longer" not in str(v) for _, r, _ in sink.records for v in r.values())


def test_rich_trace_flags_a_write_without_confirmation():
    monitor = RichMonitor()
    run = run_of(TrajectoryAssembler(compliance=monitor), [tool(3, "cancel_order", 0, 1, arguments='{"order_id": "W1"}')])
    assert run["violated_rules"] == "confirm-before-cancel" and run["critical_violations"] == 1
    assert monitor.traces[0] == [{"name": "cancel_order", "kind": "tool", "status": "ok", "args": {"order_id": "W1"}}]


def test_monitor_without_rich_trace_still_receives_tool_names():
    seen = []

    def rules(trace):
        seen.append(trace)
        return []

    assembler = TrajectoryAssembler(compliance=rules)
    agent = root(100)
    agent.events = [("user_confirmation", int(0.5 * S))]
    assembler.add(tool(3, "get_order", 0, 1))
    assembler.add(tool(4, "cancel_order", 1, 2))
    assembler.add(agent)
    assert seen == [["get_order", "cancel_order"]]


def test_rules_the_trace_cannot_decide_are_stored():
    class Monitor(FakeMonitor):
        def not_evaluable(self, trace):
            return ["confirm-before-write"] if "create_file" in trace else []

    run = run_of(TrajectoryAssembler(compliance=Monitor()), [tool(3, "create_file", 0, 1)])
    assert run["not_evaluable_rules"] == "confirm-before-write"
    assert "not_evaluable_rules" not in run_of(TrajectoryAssembler(compliance=Monitor()), [tool(3, "search", 0, 1)])
    assert "not_evaluable_rules" not in run_of(TrajectoryAssembler(compliance=FakeMonitor()), [tool(3, "create_file", 0, 1)])


def test_tool_spans_of_other_instrumentations_keep_their_order_on_a_frozen_clock(monkeypatch):
    """Review: on a coarse clock, a tool span without pulsar.sequence sorted before every event of the same tick"""
    import opentelemetry.sdk.trace as sdk_trace

    from pulsar_data_collection.genai import record_event

    monkeypatch.setattr(sdk_trace, "time_ns", lambda: 10**9)
    monitor = RichMonitor()
    provider = TracerProvider()
    exporter = PulsarSpanExporter(ListSink(), compliance=monitor)
    provider.add_span_processor(exporter.run_tracker(check_interval_s=None))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    with agent_span(tracer, "retail"):
        record_event("e0")
        with tool_span(tracer, "a"):
            pass
        with tracer.start_as_current_span("execute_tool c", attributes={sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: "c"}):
            pass
        record_event("e1")
    provider.shutdown()
    # the frozen clock makes the run look overdue (an early incomplete record): the finished run is the last trace
    assert [e["name"] for e in monitor.traces[-1]] == ["e0", "a", "c", "e1"]


def test_a_monitor_returning_trace_elements_never_writes_arguments():
    """Review: violated_rules took str() of whatever the monitor returned, arguments included"""

    class Leaky(RichMonitor):
        def violations(self, trace):
            return list(trace)

    assembler = TrajectoryAssembler(compliance=Leaky())
    run = run_of(assembler, [tool(3, "cancel_order", 0, 1, arguments='{"order_id": "W1"}')])
    assert "W1" not in str(run) and assembler.stats()["compliance_errors"] == 1
