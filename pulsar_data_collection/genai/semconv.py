"""OpenTelemetry GenAI semantic conventions used by Pulsar, pinned and normalized.

The GenAI conventions (gen_ai.*) are still at the "Development" stage and SDKs disagree on attribute names
(e.g. llm.model_name vs gen_ai.request.model). Pulsar reads a pinned set of attributes and maps known aliases to it,
so that a renaming upstream changes this file only.
"""

# Version of the normalized record schema produced by pulsar_data_collection.genai
SCHEMA_VERSION = "pulsar-genai/1"

OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
FINISH_REASONS = "gen_ai.response.finish_reasons"
AGENT_NAME = "gen_ai.agent.name"
AGENT_ID = "gen_ai.agent.id"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
CONVERSATION_ID = "gen_ai.conversation.id"
ERROR_TYPE = "error.type"

# Pulsar extensions (not in the conventions)
TERMINATION = "pulsar.agent.termination"  # e.g. "completed", "max_steps", "error", "refused"
TASK_SUCCESS = "pulsar.agent.task_success"  # bool, when the application knows the outcome
PROMPT_VERSION = "pulsar.prompt.version"
# Tool catalogue offered to the model for this run (list of tool names), set on the agent span: a removed or added tool
# is then visible from the first run (tool_catalog_hash, catalog_missing_tools, unavailable_tool_calls)
AVAILABLE_TOOLS = "pulsar.agent.available_tools"
# Outcome of a tool call, set on the tool span (see record_tool_result): "ok", "error", "timeout", "retry", "invalid",
# "unavailable"... When absent, it is derived from the span status ("timeout" when error.type names a timeout).
TOOL_STATUS = "pulsar.tool.status"
TOOL_OUTPUT_VALID = "pulsar.tool.output_valid"  # bool: the output passed the application's schema check
# Counterfactual probes (pulsar_data_collection.genai.probes): set on the agent span of each probe request
CF_SET_ID = "pulsar.cf.set_id"  # requests of one set differ only by the probed attribute
CF_VARIANT = "pulsar.cf.variant"  # group of the value used in this request, e.g. "fr"
CF_ATTRIBUTE = "pulsar.cf.attribute"  # probed attribute, e.g. "first_name"
CF_CONTROL = "pulsar.cf.control"  # "none", "placebo" (irrelevant attribute) or "order_swapped"
# Personal information (pulsar_data_collection.privacy.PiiSession): counts and types only, never the values
from ..privacy.attributes import PII_COUNT, PII_ROLES, PII_TYPES  # noqa: E402,F401

# Values of gen_ai.operation.name
OP_CHAT = "chat"
OP_TEXT_COMPLETION = "text_completion"
OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"
LLM_OPERATIONS = {OP_CHAT, OP_TEXT_COMPLETION, "generate_content"}

# Values of pulsar.tool.status
TOOL_OK = "ok"
TOOL_ERROR = "error"
TOOL_TIMEOUT = "timeout"
TOOL_UNAVAILABLE = "unavailable"  # the model called a tool that is not (or no longer) offered
# Statuses of a call that did not raise but returned nothing usable: counted as soft failures
SOFT_FAILURE_STATUSES = frozenset({"retry", "invalid", "empty", "partial", "garbage"})
# Statuses meaning that the tool does not exist for the agent
UNAVAILABLE_STATUSES = frozenset({TOOL_UNAVAILABLE, "unknown_tool", "not_found"})

# Termination of an agent_run written before its root span ended (run still open after max_run_s, or trace whose
# root never arrived)
TERMINATION_INCOMPLETE = "incomplete"

# Attribute names used by other SDKs or older versions of the conventions -> pinned name
ALIASES = {
    "gen_ai.system": PROVIDER,
    "llm.model_name": REQUEST_MODEL,
    "llm.request.model": REQUEST_MODEL,
    "gen_ai.usage.prompt_tokens": INPUT_TOKENS,
    "gen_ai.usage.completion_tokens": OUTPUT_TOKENS,
    "llm.token_count.prompt": INPUT_TOKENS,
    "llm.token_count.completion": OUTPUT_TOKENS,
    "llm.usage.prompt_tokens": INPUT_TOKENS,
    "llm.usage.completion_tokens": OUTPUT_TOKENS,
    "tool.name": TOOL_NAME,
    "gen_ai.tool.call.arguments.json": TOOL_CALL_ARGUMENTS,
}


def normalize(attributes) -> dict:
    """Copy of the span attributes where known aliases are renamed to the pinned names (pinned names win)"""
    attrs = dict(attributes or {})
    for alias, name in ALIASES.items():
        if alias in attrs and name not in attrs:
            attrs[name] = attrs[alias]
    return attrs
