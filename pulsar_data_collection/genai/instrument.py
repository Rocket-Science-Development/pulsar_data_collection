"""Small helpers to instrument an LLM application or an agent with GenAI spans (gen_ai.* conventions).

Applications already instrumented by an OpenTelemetry GenAI instrumentation (OpenLLMetry, vendor SDKs) do not need
them: PulsarSpanExporter reads any span carrying gen_ai.operation.name.
"""

import itertools
import json
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import semconv as sc

# order of the tool spans and events of this process (pulsar.sequence): timestamps can be equal on a coarse clock
_sequence = itertools.count(1)


@contextmanager
def agent_span(
    tracer: trace.Tracer,
    agent_name: str,
    prompt_version: Optional[str] = None,
    available_tools: Optional[Iterable] = None,
    **attributes,
):
    """Span of one agent run; set `pulsar.agent.termination` / `pulsar.agent.task_success` on it when known

    `available_tools`: the tools offered to the model for this run (the catalogue sent with the request), so that a
    removed tool shows from the first run (`tool_catalog_hash`, `catalog_missing_tools`, `unavailable_tool_calls` in
    agent_run). Either tool names, or the tool definitions themselves ({"name", "parameters"} or OpenAI
    {"type": "function", "function": {...}}): their parameter schemas are then recorded (`gen_ai.tool.definitions`,
    names and schemas only) and the arguments of every call are checked against them (`tool_args_*` in agent_run).
    """
    attrs = {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: agent_name, **attributes}
    if prompt_version:
        attrs[sc.PROMPT_VERSION] = prompt_version
    if available_tools is not None:
        names, definitions = set(), []
        for item in available_tools:
            if isinstance(item, dict):
                function = item.get("function") if isinstance(item.get("function"), dict) else item
                if function.get("name"):
                    names.add(str(function["name"]))
                    parameters = function.get("parameters", function.get("input_schema"))
                    if isinstance(parameters, dict):
                        definitions.append({"name": str(function["name"]), "parameters": parameters})
            else:
                names.add(str(item))
        attrs[sc.AVAILABLE_TOOLS] = sorted(names)
        if definitions:
            attrs[sc.TOOL_DEFINITIONS] = json.dumps(sorted(definitions, key=lambda d: d["name"]), sort_keys=True)
    with tracer.start_as_current_span(f"invoke_agent {agent_name}", kind=trace.SpanKind.INTERNAL, attributes=attrs) as span:
        yield span


@contextmanager
def llm_span(tracer: trace.Tracer, model: str, provider: str, temperature: Optional[float] = None, **attributes):
    """Span of one model call; record usage with `record_llm_response`"""
    attrs = {sc.OPERATION: sc.OP_CHAT, sc.REQUEST_MODEL: model, sc.PROVIDER: provider, **attributes}
    if temperature is not None:
        attrs[sc.REQUEST_TEMPERATURE] = temperature
    with tracer.start_as_current_span(f"chat {model}", kind=trace.SpanKind.CLIENT, attributes=attrs) as span:
        yield span


def record_llm_response(span, response_model=None, input_tokens=None, output_tokens=None, finish_reason=None):
    if response_model:
        span.set_attribute(sc.RESPONSE_MODEL, response_model)
    if input_tokens is not None:
        span.set_attribute(sc.INPUT_TOKENS, int(input_tokens))
    if output_tokens is not None:
        span.set_attribute(sc.OUTPUT_TOKENS, int(output_tokens))
    if finish_reason:
        span.set_attribute(sc.FINISH_REASONS, [finish_reason])


@contextmanager
def tool_span(tracer: trace.Tracer, tool_name: str, arguments=None, call_id: Optional[str] = None):
    """Span of one tool execution; an exception marks the call as failed (error.type) and is re-raised"""
    attrs = {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: tool_name, sc.SEQUENCE: next(_sequence)}
    if call_id:
        attrs[sc.TOOL_CALL_ID] = call_id
    if arguments is not None:
        attrs[sc.TOOL_CALL_ARGUMENTS] = arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True)
    with tracer.start_as_current_span(f"execute_tool {tool_name}", kind=trace.SpanKind.INTERNAL, attributes=attrs) as span:
        try:
            yield span
        except Exception as error:
            span.set_attribute(sc.ERROR_TYPE, type(error).__name__)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise


def record_tool_result(
    span,
    output: Any = None,
    status: Optional[str] = None,
    valid: Optional[bool] = None,
    validator: Optional[Callable[[Any], bool]] = None,
) -> Optional[str]:
    """Record the outcome of a tool call that did not raise, without writing the output itself

    - `status`: e.g. "ok", "retry", "invalid", "timeout", "unavailable"; by default the `status` key of a dict output,
      when it is a string (a tool answering {"status": "retry", ...} is then counted as a soft failure);
    - `valid` / `validator(output)`: result of the application's schema check (`pulsar.tool.output_valid`); a validator
      that raises counts as invalid.

    Returns the status recorded, if any. Soft failures ("retry", "invalid", "empty", "partial", "garbage", or an
    invalid output) are counted per tool in agent_run (`tool_<tool>_soft_failures`, `tool_soft_failures`).
    """
    if status is None and isinstance(output, dict) and isinstance(output.get("status"), str):
        status = output["status"]
    if valid is None and validator is not None:
        try:
            valid = bool(validator(output))
        except Exception:  # noqa: BLE001 - a validator that cannot read the output means the output is unusable
            valid = False
    if status:
        span.set_attribute(sc.TOOL_STATUS, str(status).strip().lower())
    if valid is not None:
        span.set_attribute(sc.TOOL_OUTPUT_VALID, bool(valid))
    return str(status).strip().lower() if status else None


def record_event(name: str, span=None, **attributes) -> None:
    """Record a conversation event (e.g. "user_confirmation") on the current span, or on `span`

    The event is an OpenTelemetry span event named `pulsar.event.<name>`. PulsarSpanExporter puts it in the
    compliance trace of the run, in time order with the tool calls, for the rules that need it (e.g. "a write only
    after the user confirmed"): a monitor declaring `accepts_rich_trace = True` receives {"name": name,
    "kind": "event"}. Only the name and the time are read; `attributes` go to the span event (other exporters) and are
    never written by Pulsar.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("an event needs a name")
    target = span if span is not None else trace.get_current_span()
    target.add_event(sc.EVENT_PREFIX + name, attributes={**attributes, sc.SEQUENCE: next(_sequence)})
