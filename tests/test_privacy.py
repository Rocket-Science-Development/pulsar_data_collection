import json
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

import pulsar_data_collection.pulse as pulse_module
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction
from pulsar_data_collection.privacy import PiiDetector, PiiPolicy, PiiSession
from pulsar_data_collection.pulse import Pulse

from .pii_corpus import AMBIGUOUS, evaluate
from .test_background_capture import PARAMS, FakeActions, FakeFactory

SIN = "130 692 544"  # fictitious, valid checksum
CARD = "4111 1111 1111 1111"  # test card number


@pytest.fixture
def detector():
    return PiiDetector()


def types(detector, text):
    return [f.type for f in detector.scan(text)]


def test_validated_identifiers(detector):
    assert types(detector, f"Mon NAS est {SIN}") == ["sin"]
    assert types(detector, "Mon NAS est 130 692 545") == []  # checksum fails
    assert types(detector, "NAS 046 454 286") == []  # 0 is not issued
    assert types(detector, f"Visa {CARD}") == ["credit_card"]
    assert types(detector, "Carte soleil TREM 8551 1512") == ["ramq"]  # month 55 = May, woman
    assert types(detector, "Carte soleil TREM 8513 1512") == []  # month 13 does not exist
    assert types(detector, "permis T1234-567890-12") == ["drivers_licence_qc"]


def test_context_raises_or_lowers_confidence(detector):
    bare = detector.scan("130692544")[0]
    assert bare.type == "sin" and bare.confidence == 0.6
    assert detector.scan(f"SIN: {SIN}")[0].confidence == 0.99
    assert types(detector, "Votre dossier 130692544 est prêt") == []  # business number
    assert types(detector, "Commande 4501234567 expédiée") == []
    assert types(detector, "Budget de 514 555 0199 $") == []  # an amount
    assert types(detector, "passeport AB123456") == ["passport"]
    assert types(detector, "Référence AB123456") == []  # passport-like codes need the context word
    assert types(detector, "Rendez-vous le 2026-11-13") == []  # a date is personal only as a date of birth
    assert types(detector, "Je suis née le 15 mars 1985") == ["date_of_birth"]


def test_overlaps_keep_the_most_confident(detector):
    findings = detector.scan(f"Payez avec la carte {CARD}, merci")
    assert [f.type for f in findings] == ["credit_card"]


def test_mask_replaces_values_with_labels(detector):
    text = f"NAS {SIN}, courriel lea@exemple.ca, tél. 514-555-0199"
    masked = detector.mask(text)
    assert masked == "NAS [SIN], courriel [EMAIL], tél. [PHONE]"
    assert detector.mask("rien à signaler") == "rien à signaler"


def test_unknown_types_rejected():
    with pytest.raises(ValueError):
        PiiDetector(types=["sin", "blood_type"])
    assert types(PiiDetector(types=["email"]), f"NAS {SIN}, lea@exemple.ca") == ["email"]


def test_synthetic_corpus_precision_and_recall():
    for kind, s in evaluate().items():
        assert s["recall"] >= 0.95, (kind, s)
        assert s["precision"] >= 0.95, (kind, s)
    assert types(PiiDetector(), AMBIGUOUS[0]) == ["credit_card"]  # masked on purpose


def test_session_tells_provided_values_from_leaks():
    with PiiSession() as session:
        session.scan("input", "Je suis Léa, lea@exemple.ca, 514-555-0199")
        echo = session.scan("output", "Je vous écris à lea@exemple.ca")
        assert len(echo.findings) == 1 and echo.leaks == []
        session.scan("context", f"Dossier client : NAS {SIN}")  # retrieved by a tool
        leak = session.scan("output", f"Votre voisin a le NAS {SIN}")
        assert [f.type for f in leak.leaks] == ["sin"]
    assert session._provided == set()  # values dropped at the end of the session
    with pytest.raises(ValueError):
        PiiSession().scan("prompt", "text")


def test_agent_records_carry_counts_and_types_never_values():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from pulsar_data_collection.genai import ListSink, PulsarSpanExporter, agent_span, llm_span, tool_span

    sink = ListSink()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink)))
    tracer = provider.get_tracer("test")
    request = "Bonjour, lea@exemple.ca ici. Mon NAS est 130 692 544."
    with PiiSession() as session, agent_span(tracer, "assistant") as run:
        session.record(run, "input", request)
        with tool_span(tracer, "lookup_client", arguments={"id": 1}) as tool:
            session.record(tool, "context", "Client voisin : 418 555 0142")
        with llm_span(tracer, model="m", provider="local") as call:
            session.record(call, "output", "Votre voisin est joignable au 418 555 0142.")
    run_record = sink.of("agent_run")[0]
    assert run_record["pii_input_count"] == 2 and run_record["pii_input_types"] == "email,sin"
    assert run_record["pii_context_count"] == 1
    assert run_record["pii_output_count"] == 1 and run_record["pii_leak_count"] == 1
    assert run_record["pii_leak_types"] == "phone"
    llm_record = sink.of("llm_call")[0]
    assert llm_record["pii_leak_count"] == 1 and "pii_input_count" not in llm_record
    serialized = json.dumps(sink.records, default=str)
    for value in ("lea@exemple.ca", "130 692 544", "418 555 0142"):
        assert value not in serialized


def test_records_unchanged_without_a_session():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from pulsar_data_collection.genai import ListSink, PulsarSpanExporter, agent_span

    sink = ListSink()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink)))
    with agent_span(provider.get_tracer("test"), "assistant"):
        pass
    assert not any(key.startswith("pii_") for key in sink.of("agent_run")[0])


def test_policy_masks_text_columns_and_counts():
    frame = pd.DataFrame({"note": [f"NAS {SIN}", "rien", None], "age": [30, 41, 52]})
    masked = PiiPolicy().apply(frame)
    assert masked["note"].tolist()[:2] == ["NAS [SIN]", "rien"] and pd.isna(masked["note"].iloc[2])
    assert masked["pii_count"].tolist() == [1, 0, 0] and masked["age"].tolist() == [30, 41, 52]
    assert frame["note"].iloc[0] == f"NAS {SIN}"  # the caller's frame is not modified
    flagged = PiiPolicy(action="flag", count_column="rp").apply(frame)
    assert flagged["note"].iloc[0] == f"NAS {SIN}" and flagged["rp"].iloc[0] == 1
    only = PiiPolicy(columns=["other"]).apply(frame)
    assert only["pii_count"].sum() == 0


@pytest.fixture
def fake(monkeypatch):
    actions = FakeActions()
    monkeypatch.setattr(pulse_module, "factories", {"influxdb": FakeFactory(actions)})
    return actions


def comment_capture(text: str) -> DataWithPrediction:
    return DataWithPrediction(
        data_points=pd.DataFrame({"comment": [text], "income": [32000.0]}),
        predictions=pd.DataFrame({"y_pred": [1]}),
        timestamp=datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc),
        features_names=["comment", "income"],
    )


def test_pulse_masks_before_writing(fake):
    pulse = Pulse(PARAMS, CaptureOptions(mode="sync", pii=PiiPolicy()))
    pulse.capture_data(comment_capture(f"Mon NAS est {SIN}, joignez-moi au 514-555-0199"))
    written = fake.written()
    assert written["comment"].iloc[0] == "Mon NAS est [SIN], joignez-moi au [PHONE]"
    assert written["pii_count"].iloc[0] == 2


def test_raw_values_never_reach_the_spool(fake, tmp_path):
    fake.down = True
    pulse = Pulse(PARAMS, CaptureOptions(flush_interval_s=0.05, max_retry_delay_s=0.2, spool_dir=tmp_path, pii=PiiPolicy()))
    pulse.capture_data(comment_capture(f"NAS {SIN}"))
    deadline = time.monotonic() + 5
    while not list(tmp_path.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(0.05)
    spooled = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json"))
    assert spooled and "[SIN]" in spooled and SIN not in spooled and SIN.replace(" ", "") not in spooled
    pulse.close(timeout_s=0.5)
