"""Model fingerprint: which exact model was captured and measured, from the capture configuration to the evidence.

`model_version` is a free label. A fingerprint is a digest of the model itself, with the schema of the
`model_fingerprint` field of the "pulsar.evidence/1" records of pulsar-metrics (>= 0.16):

    {"algorithm": "sha256" | "sha512", "digest": "<lowercase hex>", "source": "artifact" | "endpoint" | "declared",
     "ref": "<optional label>"}

This module does not depend on pulsar-metrics. It gives:

- `ModelFingerprint`, the validated value, and `fingerprint_path` for a saved model file or directory (the same
  digests as `pulsar_metrics.fingerprint.fingerprint_path`: the sha256 of a file's bytes, or of the sorted
  `sha256sum`-style manifest of a directory, every file included);
- `ollama_fingerprint` / `openai_fingerprint` / `endpoint_fingerprint`: the digest an LLM endpoint reports for the
  model it serves. Ollama reports one per model tag (`GET /api/tags`, the sha256 of the model manifest, which names the
  weights, template and parameters layers). OpenAI-compatible APIs usually report none: the function then returns
  None and the fingerprint can only be `declared`;
- `EndpointFingerprint`: observes the served digest (cached for `refresh_s`), compares it with the declared one and
  counts the mismatches and the changes. A mismatch means that the endpoint does not serve the model that was
  declared (and possibly assessed): it is a governance signal, recorded rather than raised.

Where the fingerprint goes: `PulseParameters(model_fingerprint=...)` tags every captured row `model_digest`;
`ResultsSink(model_fingerprint=...)` tags every result point; `EvidenceOutbox(expected_fingerprint=...)` counts the
records whose fingerprint differs or is missing; `PulsarSpanExporter(endpoint_fingerprint=...)` adds the observed
`endpoint_digest` and `model_digest_mismatch` to the agent_run and llm_call records; `CanaryProbe.run(...,
fingerprint=...)` observes the digest before and after the canary. The evidence records themselves are built by
pulsar-metrics (`model_fingerprint=` of its converters): their `evidence_id` hashes the fingerprint, so it cannot be
added to a record afterwards.
"""

import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("pulsar_data_collection")

Algorithm = Literal["sha256", "sha512"]
Source = Literal["artifact", "endpoint", "declared"]
HEX_LENGTHS = {"sha256": 64, "sha512": 128}
CHUNK_BYTES = 1024 * 1024
MAX_REF_LENGTH = 512
DIGEST_KEYS = ("digest", "sha256", "model_digest")  # keys of an OpenAI-compatible /models/{id} answer read as a digest


class ModelFingerprint(BaseModel):
    """Digest of a model (schema of the `model_fingerprint` field of pulsar-metrics evidence records)"""

    model_config = ConfigDict(extra="ignore", frozen=True)

    algorithm: Algorithm = "sha256"
    digest: str = Field(description="Lowercase hexadecimal digest (64 characters for sha256, 128 for sha512)")
    source: Source = Field(description="artifact: saved file or directory; endpoint: reported by the server; declared")
    ref: Optional[str] = Field(None, max_length=MAX_REF_LENGTH, description="Label: file name, served model tag...")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        """Accept "sha256:<hex>" (the prefix must match the algorithm) and uppercase hexadecimal"""
        if isinstance(data, dict) and isinstance(data.get("digest"), str):
            data = dict(data)
            digest = data["digest"].strip()
            prefix, sep, rest = digest.partition(":")
            if sep:
                algorithm = data.get("algorithm", "sha256")
                if prefix.lower() != algorithm:
                    raise ValueError(f"digest prefix {prefix!r} does not match algorithm {algorithm!r}")
                digest = rest
            data["digest"] = digest.lower()
        return data

    @model_validator(mode="after")
    def _check_digest(self):
        length = HEX_LENGTHS[self.algorithm]
        if not re.fullmatch(f"[0-9a-f]{{{length}}}", self.digest):
            raise ValueError(f"digest must be {length} hexadecimal characters for {self.algorithm}")
        return self

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest}"

    def matches(self, other) -> bool:
        """Same algorithm and digest (source and ref are labels and are not compared)"""
        other = as_fingerprint(other)
        return other is not None and (self.algorithm, self.digest) == (other.algorithm, other.digest)


def as_fingerprint(value) -> Optional[ModelFingerprint]:
    """ModelFingerprint of None, a dict, a ModelFingerprint (of this package or of pulsar-metrics), or a digest string
    "sha256:<hex>" (then source "declared"); ValueError when it is not a valid fingerprint"""
    if value is None or isinstance(value, ModelFingerprint):
        return value
    if isinstance(value, str):
        value = {"digest": value, "algorithm": value.partition(":")[0].lower() if ":" in value else "sha256"}
        value["source"] = "declared"
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        return ModelFingerprint.model_validate(value)
    except Exception as error:  # pydantic ValidationError is a ValueError, other types are not
        raise ValueError(f"not a model fingerprint: {error}") from None


# ------------------------------------------------------------------------------------------------ artifacts
def _hash_file(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_manifest(path: Union[str, Path], algorithm: Algorithm = "sha256") -> str:
    """`<hex>  <relative path>` per file, sorted by relative path (POSIX, UTF-8): the text hashed for a directory.
    Links to files are followed; links to directories, broken links and special files raise instead of being skipped"""
    root = Path(path)
    if algorithm not in HEX_LENGTHS:
        raise ValueError(f"algorithm must be one of {sorted(HEX_LENGTHS)}")
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    found = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError(f"symbolic link to a directory in the model directory: {os.path.join(directory, name)}")
        for name in filenames:
            file = Path(directory) / name
            try:
                mode = os.stat(file).st_mode
            except FileNotFoundError:
                raise ValueError(f"broken symbolic link in the model directory: {file}") from None
            if not stat.S_ISREG(mode):
                raise ValueError(f"not a regular file in the model directory: {file}")
            found.append((file.relative_to(root).as_posix(), file))
    if not found:
        raise ValueError(f"no file in the model directory {root}: nothing to fingerprint")
    found.sort(key=lambda item: item[0].encode("utf-8"))
    return "".join(f"{_hash_file(file, algorithm)}  {rel}\n" for rel, file in found)


def fingerprint_path(path: Union[str, Path], algorithm: Algorithm = "sha256", ref: Optional[str] = None) -> ModelFingerprint:
    """Fingerprint (source "artifact") of the saved model file or directory the server loads; ref defaults to its name

    Hash the saved artifact, not the object in memory: pickle bytes change with the library versions and the protocol.
    """
    if algorithm not in HEX_LENGTHS:
        raise ValueError(f"algorithm must be one of {sorted(HEX_LENGTHS)}")
    path = Path(path)
    if path.is_dir():
        digest = hashlib.new(algorithm, directory_manifest(path, algorithm).encode("utf-8")).hexdigest()
    elif path.is_file():
        digest = _hash_file(path, algorithm)
    else:
        raise FileNotFoundError(f"no model file or directory at {path}")
    return ModelFingerprint(algorithm=algorithm, digest=digest, source="artifact", ref=ref if ref is not None else path.name)


# ------------------------------------------------------------------------------------------------ endpoints
def _get_json(url: str, timeout_s: float, api_key: Optional[str] = None) -> dict:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"only http(s) endpoints are supported, got {url!r}")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    # the scheme is checked above: no file:// or custom scheme
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _root(base_url: str) -> str:
    """Server root of an Ollama URL given with or without its OpenAI-compatible /v1 suffix"""
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _tag(model: str) -> str:
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def ollama_fingerprint(model: str, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 5.0) -> ModelFingerprint:
    """Digest Ollama reports for `model` (`GET /api/tags`): source "endpoint", ref "ollama:<tag>"

    "llama3.2" is read as "llama3.2:latest". LookupError when the model is not served; OSError (URLError, timeout)
    when the server cannot be reached. Use 127.0.0.1 rather than localhost on Windows (IPv6 first, slow fallback).
    """
    tags = _get_json(_root(base_url) + "/api/tags", timeout_s)
    wanted = _tag(model)
    for entry in tags.get("models") or []:
        names = {entry.get("name"), entry.get("model")}
        if wanted in names and entry.get("digest"):
            return ModelFingerprint(digest=entry["digest"], source="endpoint", ref=f"ollama:{wanted}")
    raise LookupError(f"model {wanted!r} is not served by the Ollama server")


def openai_fingerprint(
    model: str, base_url: str, api_key: Optional[str] = None, timeout_s: float = 5.0
) -> Optional[ModelFingerprint]:
    """Digest reported by an OpenAI-compatible `GET {base_url}/models/{model}`, when the server gives one (a `digest`
    or `sha256` key holding a sha256); None otherwise (the OpenAI API, vLLM and Ollama's /v1 report none)"""
    data = _get_json(base_url.rstrip("/") + "/models/" + urllib.parse.quote(model, safe=":"), timeout_s, api_key)
    for key in DIGEST_KEYS:
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, str):
            try:
                return ModelFingerprint(digest=value, source="endpoint", ref=f"openai:{model}")
            except ValueError:
                continue
    return None


def endpoint_fingerprint(
    model: str,
    base_url: str,
    kind: Literal["auto", "ollama", "openai"] = "auto",
    api_key: Optional[str] = None,
    timeout_s: float = 5.0,
) -> Optional[ModelFingerprint]:
    """Digest of the served model: Ollama's (`kind="ollama"`), the OpenAI-compatible one (`"openai"`), or the first
    available (`"auto"`: Ollama, then OpenAI-compatible). None when the endpoint reports no digest; errors of the
    explicit kind propagate"""
    if kind == "ollama":
        return ollama_fingerprint(model, base_url, timeout_s)
    if kind == "openai":
        return openai_fingerprint(model, base_url, api_key, timeout_s)
    try:
        return ollama_fingerprint(model, base_url, timeout_s)
    except (OSError, LookupError, ValueError) as error:  # not an Ollama server, or not this model
        logger.debug("no Ollama digest for %s: %s", model, error)
    try:
        return openai_fingerprint(model, base_url, api_key, timeout_s)
    except (OSError, ValueError) as error:
        logger.debug("no OpenAI-compatible digest for %s: %s", model, error)
    return None


class EndpointFingerprint:
    """Digest of the model served by an endpoint, observed at run time and compared with the declared one

    Parameters
    ----------
    model, base_url : str
        Served model (e.g. "llama3.2:3b") and endpoint (e.g. "http://127.0.0.1:11434" or ".../v1")
    declared : ModelFingerprint, dict or "sha256:<hex>", optional
        The fingerprint of the model that should be served (assessed, registered). Without it nothing is compared
    kind : "auto", "ollama" or "openai"
        See endpoint_fingerprint
    refresh_s : float
        An observation is reused for this long (default 300 s): records do not each cost a request
    fetch : callable, optional
        `fetch() -> ModelFingerprint | dict | None` replacing the HTTP call (tests, other servers)

    `observe()` returns the observed fingerprint, or None when the endpoint reports none or cannot be reached (counted
    in `failures`; the previous observation is not reused, so that an outage does not hide a change). `mismatch` is
    True when the observed digest differs from the declared one, None when either is missing. `stats()` counts the
    observations, failures, mismatches and changes (the observed digest differs from the previous observation).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        declared=None,
        kind: Literal["auto", "ollama", "openai"] = "auto",
        api_key: Optional[str] = None,
        timeout_s: float = 5.0,
        refresh_s: float = 300.0,
        fetch: Optional[Callable[[], object]] = None,
    ):
        if refresh_s < 0:
            raise ValueError("refresh_s must be >= 0")
        self.model = model
        self.base_url = base_url
        self.declared = as_fingerprint(declared)
        self.kind = kind
        self.timeout_s = timeout_s
        self.refresh_s = refresh_s
        self._api_key = api_key
        self._fetch = fetch
        self._lock = threading.Lock()
        self._observed: Optional[ModelFingerprint] = None
        self._previous: Optional[ModelFingerprint] = None
        self._at: Optional[float] = None  # monotonic time of the last observation attempt
        self._stats = {"observations": 0, "failures": 0, "mismatches": 0, "changes": 0}

    def __repr__(self) -> str:
        return f"EndpointFingerprint(model={self.model!r}, base_url={self.base_url!r}, declared={str(self.declared)!r})"

    def _get(self) -> Optional[ModelFingerprint]:
        if self._fetch is not None:
            return as_fingerprint(self._fetch())
        return endpoint_fingerprint(self.model, self.base_url, self.kind, self._api_key, self.timeout_s)

    def observe(self, force: bool = False) -> Optional[ModelFingerprint]:
        """Observed fingerprint (a request at most every refresh_s unless force=True)"""
        with self._lock:
            now = time.monotonic()
            if not force and self._at is not None and now - self._at < self.refresh_s:
                return self._observed
            self._at = now
            try:
                observed = self._get()
            except Exception as error:  # noqa: BLE001 - an unreachable endpoint is counted, never raised to the caller
                logger.warning("Model digest of %s not observed: %s", self.model, error)
                observed = None
            self._observed = observed
            if observed is None:
                self._stats["failures"] += 1
                return None
            self._stats["observations"] += 1
            if self._previous is not None and not observed.matches(self._previous):
                self._stats["changes"] += 1
                logger.warning("Model digest of %s changed: %s -> %s", self.model, self._previous, observed)
            self._previous = observed
            if self.declared is not None and not observed.matches(self.declared):
                self._stats["mismatches"] += 1
                logger.warning("Model digest of %s is %s, declared %s", self.model, observed, self.declared)
            return observed

    @property
    def mismatch(self) -> Optional[bool]:
        """Last observation differs from the declared fingerprint; None when either is missing"""
        observed = self._observed
        if observed is None or self.declared is None:
            return None
        return not observed.matches(self.declared)

    def fields(self) -> dict:
        """Record fields after an observation: endpoint_digest ("sha256:<hex>") and model_digest_mismatch, None when
        unknown"""
        observed = self.observe()
        return {
            "endpoint_digest": None if observed is None else str(observed),
            "model_digest_mismatch": self.mismatch,
        }

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
