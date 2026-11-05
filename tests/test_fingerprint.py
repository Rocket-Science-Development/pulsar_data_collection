"""Model fingerprint: validation, artifacts, served digests (Ollama), mismatch signal, and where the digest is carried"""

import hashlib
import json
import urllib.request

import pytest
from pydantic import ValidationError

from pulsar_data_collection import fingerprint as fpm
from pulsar_data_collection.evidence import EvidenceOutbox
from pulsar_data_collection.fingerprint import (
    EndpointFingerprint,
    ModelFingerprint,
    as_fingerprint,
    directory_manifest,
    endpoint_fingerprint,
    fingerprint_path,
    ollama_fingerprint,
    openai_fingerprint,
)
from pulsar_data_collection.models import PulseParameters
from pulsar_data_collection.results import ResultsSink

HEX_A, HEX_B = "ab" * 32, "cd" * 32
OLLAMA = "http://127.0.0.1:11434"  # 127.0.0.1, not localhost (IPv6 first on Windows)
# digest of model_dir() computed by pulsar_metrics.fingerprint.fingerprint_path (0.16): both packages must agree
MODEL_DIR_DIGEST = "18d881ea17494c6bf6b94335a9d49633ea008e0465f9c4c5ba57dff0890ceb7f"
PARAMS = dict(
    model_id="m",
    model_version="1",
    data_id="d",
    reference_data_storage="ref.csv",
    target_name="y",
    storage_engine="influxdb",
    login={"url": "http://127.0.0.1:8086", "token": "t", "org": "o", "bucket_name": "b"},
)


def model_dir(root):
    files = {"config.json": b'{"a": 1}', "weights/part-1.bin": b"\x00" * 1000, ".hidden": b"h", "Z.txt": b"z"}
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    return root


# ------------------------------------------------------------------ value and artifacts
def test_fingerprint_value_is_validated_and_normalized():
    fp = as_fingerprint("SHA256:" + HEX_A.upper())
    assert fp == ModelFingerprint(digest=HEX_A, source="declared") and str(fp) == f"sha256:{HEX_A}"
    assert as_fingerprint(None) is None and as_fingerprint(fp) is fp
    assert as_fingerprint({"digest": HEX_A, "source": "endpoint", "ref": "x"}).matches(fp)  # labels not compared
    for bad in ("sha256:1234", "md5:" + HEX_A, {"digest": HEX_A, "source": "weights"}, {"digest": HEX_A}, 42):
        with pytest.raises(ValueError):
            as_fingerprint(bad)


def test_artifact_digests_match_pulsar_metrics(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 3))
    assert fingerprint_path(path).digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert fingerprint_path(path).ref == "model.joblib"
    root = model_dir(tmp_path / "model")
    assert fingerprint_path(root).digest == MODEL_DIR_DIGEST
    assert [line.split("  ")[1] for line in directory_manifest(root).splitlines()] == [
        ".hidden",
        "Z.txt",
        "config.json",
        "weights/part-1.bin",
    ]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no file"):
        fingerprint_path(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        fingerprint_path(tmp_path / "missing")


# ------------------------------------------------------------------ endpoints (no network)
def test_openai_compatible_digest_when_reported(monkeypatch):
    answers = {"/models/llama3.2:3b": {"id": "llama3.2:3b", "object": "model"}, "/models/served": {"digest": HEX_A}}

    def fake(url, timeout_s, api_key=None):
        for suffix, answer in answers.items():
            if url.endswith(suffix):
                return answer
        raise OSError("unreachable")

    monkeypatch.setattr(fpm, "_get_json", fake)
    assert openai_fingerprint("llama3.2:3b", "http://127.0.0.1:8000/v1") is None  # no digest reported
    fp = openai_fingerprint("served", "http://127.0.0.1:8000/v1")
    assert fp.digest == HEX_A and fp.source == "endpoint" and fp.ref == "openai:served"
    # auto: not an Ollama server (no /api/tags), falls back to the OpenAI-compatible answer
    assert endpoint_fingerprint("served", "http://127.0.0.1:8000/v1").digest == HEX_A
    assert endpoint_fingerprint("llama3.2:3b", "http://127.0.0.1:8000/v1") is None


def test_only_http_endpoints():
    with pytest.raises(ValueError, match="http"):
        ollama_fingerprint("m", "file:///etc")


def test_endpoint_fingerprint_counts_mismatches_changes_and_failures():
    served = [HEX_A, HEX_A, HEX_B, None, "boom"]

    def fetch():
        value = served.pop(0)
        if value == "boom":
            raise OSError("endpoint down")
        return None if value is None else {"digest": value, "source": "endpoint"}

    probe = EndpointFingerprint("m", declared=f"sha256:{HEX_A}", refresh_s=0, fetch=fetch)
    assert probe.fields() == {"endpoint_digest": f"sha256:{HEX_A}", "model_digest_mismatch": False}
    assert probe.observe().digest == HEX_A
    assert probe.fields() == {"endpoint_digest": f"sha256:{HEX_B}", "model_digest_mismatch": True}
    assert probe.observe() is None and probe.mismatch is None  # no digest reported: unknown, not "same"
    assert probe.observe() is None  # unreachable: counted, not raised
    assert probe.stats() == {"observations": 3, "failures": 2, "mismatches": 1, "changes": 1}

    cached = EndpointFingerprint("m", refresh_s=3600, fetch=lambda: {"digest": HEX_A, "source": "endpoint"})
    for _ in range(5):
        cached.fields()
    assert cached.stats()["observations"] == 1 and cached.mismatch is None  # nothing declared: nothing compared
    with pytest.raises(ValueError):
        EndpointFingerprint("m", refresh_s=-1)


# ------------------------------------------------------------------ where the digest goes
def test_pulse_parameters_tag_every_row_with_the_digest():
    params = PulseParameters(**PARAMS, additional_tags={"env": "test"}, model_fingerprint=f"sha256:{HEX_A}")
    assert params.additional_tags == {"env": "test", "model_digest": f"sha256:{HEX_A}"}
    assert params.model_fingerprint.source == "declared"
    assert PulseParameters(**PARAMS).additional_tags is None  # unchanged without fingerprint
    same = PulseParameters(**PARAMS, additional_tags={"model_digest": f"sha256:{HEX_A}"}, model_fingerprint=HEX_A)
    assert same.additional_tags == {"model_digest": f"sha256:{HEX_A}"}
    with pytest.raises(ValidationError, match="differs"):
        PulseParameters(**PARAMS, additional_tags={"model_digest": "other"}, model_fingerprint=HEX_A)
    with pytest.raises(ValidationError):
        PulseParameters(**PARAMS, model_fingerprint="sha256:nothex")


def test_results_sink_tags_points_with_the_digest():
    sink = ResultsSink("http://127.0.0.1:8086", "t", "o", "b", model_fingerprint={"digest": HEX_A, "source": "artifact"})
    written = []
    sink._write = type("W", (), {"write": lambda self, **kw: written.extend(kw["record"]), "close": lambda self: None})()
    result = {"metric_type": "drift", "metric_name": "ks_2samp", "feature_name": "x", "metric_value": 0.2}
    assert sink.write_results([result], tags={"model_id": "m"}) == 1
    assert f"model_digest=sha256:{HEX_A}" in written[0].to_line_protocol()
    with pytest.raises(ValueError, match="differs"):
        sink.write_results([result], tags={"model_digest": "sha256:" + HEX_B})
    sink.close()


def evidence(i, fingerprint=None):
    record = {"schema_version": "pulsar.evidence/1", "evidence_id": f"ev-{i}"}
    if fingerprint is not None:
        record["model_fingerprint"] = fingerprint
    return record


def test_outbox_counts_records_about_another_model(tmp_path, caplog):
    box = EvidenceOutbox("https://collector.example/evidence", tmp_path, start=False, expected_fingerprint=HEX_A)
    try:
        good = {"algorithm": "sha256", "digest": HEX_A, "source": "artifact", "ref": "credit.joblib"}
        other = {"algorithm": "sha256", "digest": HEX_B, "source": "endpoint", "ref": "ollama:x"}
        assert box.put_many([evidence(1, good), evidence(2, other), evidence(3)]) == 3
        stats = box.stats()
        assert stats["fingerprint_mismatches"] == 1 and stats["unfingerprinted"] == 1 and stats["pending"] == 3
        assert "expected sha256:" + HEX_A in caplog.text
        with pytest.raises(ValueError, match="model_fingerprint"):
            box.put(evidence(4, {"digest": "nope", "source": "declared"}))
        assert box.stats()["pending"] == 3
    finally:
        box.close(flush=False)
    plain = EvidenceOutbox("https://collector.example/evidence", tmp_path / "plain", start=False)
    try:
        assert plain.put(evidence(5, {"digest": HEX_B, "source": "declared"}))
        assert plain.stats()["fingerprint_mismatches"] == 0 and plain.stats()["unfingerprinted"] == 0
    finally:
        plain.close(flush=False)


def test_agent_run_and_llm_call_records_get_the_endpoint_digest():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from pulsar_data_collection.genai import ListSink, PulsarSpanExporter, agent_span, llm_span, tool_span

    served = {"digest": HEX_B, "source": "endpoint", "ref": "ollama:llama3.2:3b"}
    probe = EndpointFingerprint("llama3.2:3b", declared=HEX_A, fetch=lambda: served)
    sink = ListSink()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink, endpoint_fingerprint=probe)))
    tracer = provider.get_tracer("test")
    with agent_span(tracer, "assistant"):
        with llm_span(tracer, model="llama3.2:3b", provider="ollama"):
            pass
        with tool_span(tracer, "search", arguments={"q": "x"}):
            pass
    provider.shutdown()
    (run,) = sink.of("agent_run")
    (call,) = sink.of("llm_call")
    for record in (run, call):
        assert record["endpoint_digest"] == f"sha256:{HEX_B}" and record["model_digest_mismatch"] is True
    assert "endpoint_digest" not in sink.of("tool_call")[0]
    assert probe.stats()["observations"] == 1  # cached between records


def test_canary_observes_the_digest_before_and_after(tmp_path):
    from pulsar_data_collection.genai import CanaryProbe

    served = [HEX_A, HEX_B]
    probe = EndpointFingerprint("m", declared=HEX_A, fetch=lambda: {"digest": served.pop(0), "source": "endpoint"})
    canary = CanaryProbe(["p1", "p2"], samples=2, warn=False)
    answers = canary.run(lambda prompt: "answer", fingerprint=probe)
    assert canary.fingerprint == {
        "declared": f"sha256:{HEX_A}",
        "before": f"sha256:{HEX_A}",
        "after": f"sha256:{HEX_B}",
        "mismatch": True,
        "changed": True,
    }
    path = CanaryProbe.save(answers, tmp_path / "canary.json", fingerprint=canary.fingerprint, model="m")
    metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
    assert metadata["model_fingerprint"]["changed"] is True and metadata["model"] == "m"
    canary.run(lambda prompt: "answer")
    assert canary.fingerprint is None


# ------------------------------------------------------------------ integration: local Ollama
def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/version", timeout=2):  # nosec B310 - fixed loopback URL
            return True
    except OSError:
        return False


@pytest.mark.skipif(not ollama_up(), reason="no Ollama server at 127.0.0.1:11434")
def test_ollama_reports_a_digest_per_served_model():
    tags = fpm._get_json(OLLAMA + "/api/tags", 5)
    names = [m["name"] for m in tags.get("models", [])]
    if not names:
        pytest.skip("Ollama serves no model")
    model = "llama3.2:3b" if "llama3.2:3b" in names else names[0]
    fp = ollama_fingerprint(model, OLLAMA)
    assert fp.source == "endpoint" and fp.ref == f"ollama:{model}" and len(fp.digest) == 64
    assert endpoint_fingerprint(model, OLLAMA + "/v1").matches(fp)  # the OpenAI-compatible base URL works too
    assert openai_fingerprint(model, OLLAMA + "/v1") is None  # Ollama's /v1/models reports no digest
    with pytest.raises(LookupError):
        ollama_fingerprint("no-such-model:0b", OLLAMA)
    probe = EndpointFingerprint(model, OLLAMA, declared=fp)
    assert probe.fields() == {"endpoint_digest": str(fp), "model_digest_mismatch": False}
    if len(names) > 1:  # another served model is a mismatch
        other = next(n for n in names if n != model)
        assert EndpointFingerprint(other, OLLAMA, declared=fp).fields()["model_digest_mismatch"] is True
