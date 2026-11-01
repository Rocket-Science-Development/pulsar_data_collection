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

# Values of gen_ai.operation.name
OP_CHAT = "chat"
OP_TEXT_COMPLETION = "text_completion"
OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"
LLM_OPERATIONS = {OP_CHAT, OP_TEXT_COMPLETION, "generate_content"}

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
