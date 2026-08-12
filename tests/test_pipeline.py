"""Test suite.

The leak tests are the important ones: they take every secret the detector
found in the source data and assert that the literal string appears in none of
the generated artefacts.

    python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from msgiq.classifier import normalise                      # noqa: E402
from msgiq.extraction import extract                        # noqa: E402
from msgiq.loader import Message, load_messages, parse_timestamp, strip_lead_in  # noqa: E402
from msgiq.rules import apply_rules                         # noqa: E402
from msgiq.sensitive import DETECTORS, scan_message         # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
HAS_DATA = (DATA / "messages.csv").exists()
HAS_OUT = (OUT / "combined.json").exists()


# ------------------------------------------------------------------- loader

def test_strip_lead_in_removes_stacked_prefixes():
    assert strip_lead_in("Quick update: FYI: The report is ready.") == "The report is ready."
    assert strip_lead_in("Can you help? The lift is broken.") == "The lift is broken."


def test_strip_lead_in_leaves_plain_text_alone():
    assert strip_lead_in("The report is ready.") == "The report is ready."


def test_bad_timestamp_returns_none_not_a_guess():
    assert parse_timestamp("not a date") is None
    assert parse_timestamp("") is None
    assert parse_timestamp("2026-09-01 08:00:00") is not None


# ---------------------------------------------------------------- detection

SECRET_CASES = [
    ("Your OTP is 123-456. It expires in 10 minutes.", "one_time_password", "critical"),
    ("Use password BlueRiver#12-34 to sign in.",       "password",          "critical"),
    ("My card number is 4111 1111 1111 1111-92.",      "payment_card",      "critical"),
    ("Please note my bank account number 1234-5678.",  "bank_account",      "critical"),
    ("The temporary access token is tok_demo_A1K2Q-34.", "auth_token",      "critical"),
    ("My account recovery code is RC-1234-KL-56-78.",  "account_recovery_code", "critical"),
    ("My identification number is ID-1234-XY-56.",     "personal_identifier", "high"),
    ("My recent test result says vitamin D deficiency-97.", "health_information", "high"),
    ("My home address is 42 Lake View Road, Chennai-45.", "private_address", "medium"),
    ("You can contact me on 98765 43210-12.",          "contact_number",    "medium"),
]


@pytest.mark.parametrize("text,expected_type,expected_risk", SECRET_CASES)
def test_detector_finds_type_and_risk(text, expected_type, expected_risk):
    r = scan_message("T", text)
    assert r.is_sensitive
    assert r.sensitivity_type == expected_type
    assert r.risk == expected_risk


@pytest.mark.parametrize("text,_t,_r", SECRET_CASES)
def test_masked_text_never_contains_the_secret(text, _t, _r):
    r = scan_message("T", text)
    for f in r.findings:
        if f.span is None:
            continue
        secret = text[f.span[0]:f.span[1]]
        assert secret not in r.masked_text, f"leaked {f.sensitivity_type}"
    assert "*" in r.masked_text


def test_mask_width_does_not_reveal_secret_length():
    short = scan_message("T", "Your OTP is 1234.")
    long = scan_message("T", "Your OTP is 1234567890123.")
    a = re.search(r"\*+", short.masked_text).group(0)
    b = re.search(r"\*+", long.masked_text).group(0)
    assert len(a) == len(b), "mask length leaks the length of the secret"


BENIGN = [
    "The laptop battery is fully charged.",
    "Please join the AI workshop on 2026-09-18, 13:00 at Google Meet.",
    "Flash sale on laptops starts at 6 PM. Use code SAVE23.",
    "Remember that i drink coffee without sugar.",
    "Can you review the privacy checklist before 2026-09-09?",
    "The cafeteria closes at 8 PM.",
]


@pytest.mark.parametrize("text", BENIGN)
def test_no_false_positive_on_benign_messages(text):
    assert not scan_message("T", text).is_sensitive


def test_credential_mention_is_flagged_but_has_nothing_to_mask():
    r = scan_message("T", "I will send the login details separately.")
    assert r.is_sensitive
    assert r.sensitivity_type == "credential_reference"
    assert r.risk == "low"
    assert r.masked_text == "I will send the login details separately."


def test_multiple_secrets_in_one_message_are_all_masked():
    text = "Your OTP is 998877 and my card number is 4111 1111 1111 1111."
    r = scan_message("T", text)
    assert len(r.findings) >= 2
    assert "998877" not in r.masked_text
    assert "4111" not in r.masked_text


def test_every_detector_declares_a_value_group():
    for d in DETECTORS:
        assert "value" in d.regex.groupindex, f"{d.sensitivity_type} has no value group"


# ----------------------------------------------------------------- rules/ML

def test_rules_assign_expected_categories():
    cases = [
        ("Flash sale on laptops starts at 6 PM. Use code SAVE23.", "promotional"),
        ("Please join the AI workshop on 2026-09-18, 13:00 at Zoom.", "meeting_or_event"),
        ("I need you to review the model results by 2026-09-03.", "action_required"),
        ("For my profile, i am vegetarian.", "personal_information"),
        ("The laptop battery is fully charged.", "general_information"),
    ]
    for text, expected in cases:
        assert apply_rules(text).category == expected, text


def test_normalise_replaces_literals_with_type_tokens():
    n = normalise("Submit by 2026-09-14 at 10:30 with 5 copies")
    assert "<date>" in n and "<time>" in n and "<num>" in n
    assert "2026-09-14" not in n


# ---------------------------------------------------------------- extraction

def _msg(text: str, ts: str = "2026-09-01 08:00:00", sender: str = "Meera") -> Message:
    return Message("MSG_TEST", parse_timestamp(ts), sender, text, ts, 0)


def test_explicit_date_is_copied_verbatim():
    item = extract(_msg("Please submit the weekly report by 2026-09-05."), "action_required")
    assert item.date == "2026-09-05"
    assert item.date_provenance == "explicit"


def test_vague_timing_is_left_null_not_guessed():
    item = extract(_msg("Let us meet sometime next week."), "meeting_or_event")
    assert item.date is None
    assert item.date_provenance == "unresolved"
    assert "date" in item.unresolved_fields


def test_relative_day_is_derived_by_arithmetic_and_labelled():
    item = extract(_msg("The report may be needed tomorrow.", "2026-09-05 09:00:00"),
                   "action_required")
    assert item.date == "2026-09-06"
    assert item.date_provenance == "derived_from_timestamp"


def test_person_only_extracted_when_named_in_the_text():
    named = extract(_msg("Please call Maya when you are free."), "action_required")
    assert named.person == "Maya"
    unnamed = extract(_msg("Please submit the weekly report by 2026-09-05."), "action_required")
    assert unnamed.person is None
    assert "person" in unnamed.unresolved_fields


def test_sender_is_never_assumed_to_be_the_person_involved():
    item = extract(_msg("Please submit the report by 2026-09-05.", sender="Rohan"),
                   "action_required")
    assert item.person is None
    assert item.sender == "Rohan"


def test_event_fields_are_populated():
    item = extract(_msg("Please join the internship orientation on 2026-09-18, "
                        "13:00 at Conference Room 2."), "meeting_or_event")
    assert item.type == "event"
    assert item.date == "2026-09-18" and item.time == "13:00"
    assert "Conference Room" in item.location


def test_hedged_message_is_low_priority():
    item = extract(_msg("The review could be Friday afternoon."), "meeting_or_event")
    assert item.priority == "low"


def test_non_actionable_message_yields_no_item():
    assert extract(_msg("The laptop battery is fully charged."), "general_information") is None


# ------------------------------------------------- integration + leak audit

@pytest.mark.skipif(not HAS_DATA, reason="dataset not present (not committed)")
def test_dataset_is_chronological_after_load():
    msgs, report = load_messages(DATA / "messages.csv")
    stamps = [m.timestamp for m in msgs if m.timestamp]
    assert stamps == sorted(stamps)
    assert report.unparsed_timestamps == 0
    assert not report.duplicate_ids


@pytest.mark.skipif(not (HAS_DATA and HAS_OUT), reason="run run_pipeline.py first")
def test_no_secret_value_appears_in_any_generated_file():
    """The headline safety guarantee."""
    msgs, _ = load_messages(DATA / "messages.csv")
    secrets: set[str] = set()
    for m in msgs:
        core = strip_lead_in(m.text)
        for f in scan_message(m.message_id, core).findings:
            if f.span is None:
                continue
            value = core[f.span[0]:f.span[1]].strip()
            if len(value) >= 4:
                secrets.add(value)

    assert secrets, "expected the corpus to contain secrets to test against"

    leaks: list[str] = []
    for path in sorted(OUT.glob("*.json")):
        blob = path.read_text(encoding="utf-8")
        for secret in secrets:
            if secret in blob:
                leaks.append(f"{path.name}: {secret[:4]}...")
    assert not leaks, f"sensitive values leaked into outputs: {leaks[:10]}"


@pytest.mark.skipif(not HAS_OUT, reason="run run_pipeline.py first")
def test_every_message_has_exactly_one_classification():
    combined = json.loads((OUT / "combined.json").read_text(encoding="utf-8"))
    ids = [c["message_id"] for c in combined]
    assert len(ids) == len(set(ids))
    for c in combined:
        assert c["classification"]["category"]
        assert 0.0 <= c["classification"]["confidence"] <= 1.0
        assert c["classification"]["reason"]


@pytest.mark.skipif(not HAS_OUT, reason="run run_pipeline.py first")
def test_all_six_categories_are_represented():
    summary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
    counts = summary["part1_classification"]["counts"]
    expected = {"action_required", "meeting_or_event", "personal_information",
                "general_information", "promotional", "sensitive_information"}
    assert expected <= set(counts), f"missing categories: {expected - set(counts)}"


@pytest.mark.skipif(not HAS_OUT, reason="run run_pipeline.py first")
def test_all_mandatory_demo_ids_resolve():
    demo = json.loads((OUT / "mandatory_demo_report.json").read_text(encoding="utf-8"))
    assert len(demo) == 15
    for row in demo:
        assert row.get("status") != "NOT_FOUND_IN_DATASET", row["message_id"]
        assert row["classification"]["category"]


@pytest.mark.skipif(not HAS_OUT, reason="run run_pipeline.py first")
def test_extracted_items_never_invent_a_field():
    items = json.loads((OUT / "tasks_events.json").read_text(encoding="utf-8"))
    for i in items:
        for fname in ("date", "time", "person", "location"):
            key = "deadline" if (fname == "date" and i["type"] == "task") else fname
            if i.get(fname) is None:
                assert key in i["unresolved_fields"] or fname == "location", i["item_id"]
        if i["date"] is not None and i["date_provenance"] == "explicit":
            assert i["date"] in i["description"], i["item_id"]


@pytest.mark.skipif(not HAS_OUT, reason="run run_pipeline.py first")
def test_sensitive_export_carries_masked_text_only():
    rows = json.loads((OUT / "sensitive.json").read_text(encoding="utf-8"))
    assert rows
    for r in rows:
        assert r["risk"] in {"low", "medium", "high", "critical"}
        assert r["recommended_action"]
        if r["values_redacted"] > 0:
            assert "*" in r["masked_text"], r["message_id"]
