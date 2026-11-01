"""Small helpers to instrument an LLM application or an agent with GenAI spans (gen_ai.* conventions).

Applications already instrumented by an OpenTelemetry GenAI instrumentation (OpenLLMetry, vendor SDKs) do not need
them: PulsarSpanExporter reads any span carrying gen_ai.operation.name.
"""

import json
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import semconv as sc


@contextmanager
def agent_span(tracer: trace.Tracer, agent_name: str, prompt_version: Optional[str] = None, **attributes):
    """Span of one agent run; set `pulsar.agent.termination` / `pulsar.agent.task_success` on it when known"""
    attrs = {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: agent_name, **attributes}
    if prompt_version:
        attrs[sc.PROMPT_VERSION] = prompt_version
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
    attrs = {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: tool_name}
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
