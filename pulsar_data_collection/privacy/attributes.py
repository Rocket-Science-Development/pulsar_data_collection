"""Span attributes written by PiiSession (also exposed in genai.semconv); no dependency, so that tabular users do not
need the genai extra."""

PII_COUNT = "pulsar.pii.{role}.count"  # role: input, context, output, leak
PII_TYPES = "pulsar.pii.{role}.types"  # comma-separated types, e.g. "email,sin"
PII_ROLES = ("input", "context", "output", "leak")
