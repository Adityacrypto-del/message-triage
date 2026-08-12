"""End-to-end orchestration.

Order matters and is fixed:

1. Load and sort chronologically.
2. Scan for sensitive content **first**, so that no later stage ever handles a
   message without knowing it contains a secret.
3. Train the classifier on rule-derived weak labels.
4. Classify, extract, and emit outputs - all in chronological order, with IDs
   assigned in that same order so the run is reproducible.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .classifier import Classification, HybridClassifier
from .config import CATEGORIES, REVIEW_FLAG_THRESHOLD, SENSITIVE_INFORMATION
from .extraction import Item, extract
from .loader import LoadReport, Message, load_mandatory_ids, load_messages, strip_lead_in
from .sensitive import ScanResult, scan_message


class PipelineResult:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.load_report: LoadReport | None = None
        self.scans: dict[str, ScanResult] = {}
        self.classifications: dict[str, Classification] = {}
        self.items: list[Item] = []
        self.training_report = None
        self.mandatory_ids: list[str] = []

    # ---------------------------------------------------------------- helpers
    def safe_text(self, message_id: str) -> str:
        """The ONLY approved way to display a message anywhere.

        Returns the masked form when the message contains a secret. Nothing in
        the UI, the exports, or the logs reads ``Message.text`` directly.
        """
        scan = self.scans.get(message_id)
        if scan is not None and scan.is_sensitive:
            return scan.masked_text
        msg = next((m for m in self.messages if m.message_id == message_id), None)
        return msg.text if msg else ""

    @property
    def category_counts(self) -> Counter:
        return Counter(c.category for c in self.classifications.values())

    @property
    def flagged(self) -> list[Classification]:
        return [c for c in self.classifications.values()
                if c.needs_review or c.confidence < REVIEW_FLAG_THRESHOLD]


def run(data_dir: str | Path = "data", verbose: bool = True) -> PipelineResult:
    data_dir = Path(data_dir)
    res = PipelineResult()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1 ----------------------------------------------------------------- load
    res.messages, res.load_report = load_messages(data_dir / "messages.csv")
    res.mandatory_ids = load_mandatory_ids(data_dir / "mandatory_demo_ids.csv")
    log(f"[1/5] Loaded {res.load_report.total_rows} messages "
        f"({res.load_report.parsed_timestamps} timestamps parsed, "
        f"already chronological: {res.load_report.was_already_chronological})")

    # 2 ------------------------------------------------- sensitive scan first
    for m in res.messages:
        res.scans[m.message_id] = scan_message(m.message_id, strip_lead_in(m.text))
    n_sens = sum(1 for s in res.scans.values() if s.is_sensitive)
    log(f"[2/5] Sensitive scan complete: {n_sens} messages contain sensitive "
        f"content (values masked at detection time)")

    # 3 ---------------------------------------------------------------- train
    clf = HybridClassifier()
    res.training_report = clf.fit(res.messages, res.scans)
    tr = res.training_report
    log(f"[3/5] Model trained on {tr.n_training} weakly-labelled examples | "
        f"5-fold CV accuracy {tr.cv_accuracy:.3f}, macro-F1 {tr.cv_macro_f1:.3f} | "
        f"rule coverage {tr.coverage:.1%}, rule/model agreement {tr.agreement:.1%}")

    # 4 ------------------------------------------------------------- classify
    for m in res.messages:                       # chronological order
        res.classifications[m.message_id] = clf.classify(m, res.scans.get(m.message_id))
    log(f"[4/5] Classified {len(res.classifications)} messages: "
        f"{dict(res.category_counts)}")

    # 5 -------------------------------------------------------------- extract
    seq = 0
    for m in res.messages:                       # chronological order -> stable IDs
        cat = res.classifications[m.message_id].category
        if cat == SENSITIVE_INFORMATION:
            # Never build a calendar entry out of a message containing a secret.
            continue
        item = extract(m, cat)
        if item is None:
            continue
        seq += 1
        item.item_id = f"{'TASK' if item.type == 'task' else 'EVENT'}_{seq:04d}"
        res.items.append(item)
    n_tasks = sum(1 for i in res.items if i.type == "task")
    log(f"[5/5] Extracted {len(res.items)} items "
        f"({n_tasks} tasks, {len(res.items) - n_tasks} events)")

    return res


# --------------------------------------------------------------------- export

def export(res: PipelineResult, out_dir: str | Path = "outputs") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def dump(name: str, payload) -> None:
        p = out / name
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written[name] = p

    # Part 1 ---------------------------------------------------------------
    dump("classifications.json", [
        res.classifications[m.message_id].to_dict() for m in res.messages
    ])

    # Part 2 ---------------------------------------------------------------
    dump("tasks_events.json", [i.to_dict() for i in res.items])

    # Part 3 - masked text only; raw values never reach disk -----------------
    sensitive_rows = []
    for m in res.messages:
        s = res.scans[m.message_id]
        if not s.is_sensitive:
            continue
        sensitive_rows.append({
            "message_id": m.message_id,
            "timestamp": m.raw_timestamp,
            "sender": m.sender,
            "sensitivity_type": s.sensitivity_type,
            "all_types_detected": s.types,
            "risk": s.risk,
            "masked_text": s.masked_text,
            "recommended_action": s.recommended_action,
            "additional_actions": s.additional_actions,
            "reason": s.reason,
            "detector_confidence": round(s.confidence, 4),
            "values_redacted": sum(1 for f in s.findings if f.span is not None),
        })
    dump("sensitive.json", sensitive_rows)

    # Combined per-message view (masked) ------------------------------------
    combined = []
    for m in res.messages:
        c = res.classifications[m.message_id]
        s = res.scans[m.message_id]
        item = next((i for i in res.items if i.source_message_id == m.message_id), None)
        combined.append({
            "message_id": m.message_id,
            "timestamp": m.raw_timestamp,
            "sender": m.sender,
            "text": s.masked_text,           # masked, always
            "is_masked": s.is_sensitive and any(f.span for f in s.findings),
            "classification": c.to_dict(),
            "sensitivity": ({
                "type": s.sensitivity_type, "risk": s.risk,
                "recommended_action": s.recommended_action,
            } if s.is_sensitive else None),
            "extracted_item": item.to_dict() if item else None,
        })
    dump("combined.json", combined)

    # Mandatory demo IDs ----------------------------------------------------
    by_id = {m.message_id: m for m in res.messages}
    demo = []
    for mid in res.mandatory_ids:
        if mid not in by_id:
            demo.append({"message_id": mid, "status": "NOT_FOUND_IN_DATASET"})
            continue
        entry = next(x for x in combined if x["message_id"] == mid)
        demo.append(entry)
    dump("mandatory_demo_report.json", demo)

    # Run summary -----------------------------------------------------------
    tr = res.training_report
    risk_counts = Counter(s.risk for s in res.scans.values() if s.is_sensitive)
    dump("run_summary.json", {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "messages": res.load_report.total_rows,
            "timestamps_parsed": res.load_report.parsed_timestamps,
            "already_chronological": res.load_report.was_already_chronological,
            "duplicate_ids": res.load_report.duplicate_ids,
            "first_timestamp": str(res.load_report.first_timestamp),
            "last_timestamp": str(res.load_report.last_timestamp),
        },
        "part1_classification": {
            "counts": dict(res.category_counts),
            "decision_sources": dict(Counter(
                c.decision_source for c in res.classifications.values())),
            "mean_confidence": round(sum(
                c.confidence for c in res.classifications.values())
                / max(1, len(res.classifications)), 4),
            "flagged_for_review": len(res.flagged),
            "model": {
                "training_examples": tr.n_training,
                "cv_accuracy": round(tr.cv_accuracy, 4),
                "cv_macro_f1": round(tr.cv_macro_f1, 4),
                "rule_coverage": round(tr.coverage, 4),
                "rule_model_agreement": round(tr.agreement, 4),
                "excluded_low_confidence": tr.excluded_low_confidence,
                "labels": tr.labels,
                "confusion_matrix": tr.confusion,
                "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                              for k, v in tr.per_class.items()},
            },
        },
        "part2_extraction": {
            "total_items": len(res.items),
            "tasks": sum(1 for i in res.items if i.type == "task"),
            "events": sum(1 for i in res.items if i.type == "event"),
            "with_unresolved_fields": sum(1 for i in res.items if i.unresolved_fields),
            "date_provenance": dict(Counter(i.date_provenance for i in res.items)),
            "priority": dict(Counter(i.priority for i in res.items)),
        },
        "part3_sensitive": {
            "messages_flagged": sum(1 for s in res.scans.values() if s.is_sensitive),
            "by_type": dict(Counter(
                s.sensitivity_type for s in res.scans.values() if s.is_sensitive)),
            "by_risk": dict(risk_counts),
            "by_recommended_action": dict(Counter(
                s.recommended_action for s in res.scans.values() if s.is_sensitive)),
        },
    })
    return written
