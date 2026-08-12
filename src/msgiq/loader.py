"""Dataset loading with chronological-order enforcement.

The brief requires messages to be processed in chronological order, so ordering
is validated here rather than assumed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S")


@dataclass(frozen=True)
class Message:
    message_id: str
    timestamp: datetime | None
    sender: str
    text: str
    raw_timestamp: str
    index: int  # position after chronological sort

    @property
    def has_timestamp(self) -> bool:
        return self.timestamp is not None


def parse_timestamp(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


# Conversational lead-ins that carry no classification signal. Stripping them
# stops the model from learning "Can you help?" as an action-required cue when
# it is really just filler glued onto every category.
LEAD_IN_RE = re.compile(
    r"^\s*(?:"
    r"for today:|fyi:|one more thing:|hi,|important:|can you help\?|"
    r"just checking[—\-–]|please note:|quick update:|quick note:|heads up:"
    r")\s*",
    re.IGNORECASE,
)


def strip_lead_in(text: str) -> str:
    """Remove up to two stacked conversational prefixes."""
    out = text.strip()
    for _ in range(2):
        stripped = LEAD_IN_RE.sub("", out)
        if stripped == out:
            break
        out = stripped
    return out.strip()


@dataclass
class LoadReport:
    total_rows: int
    parsed_timestamps: int
    unparsed_timestamps: int
    was_already_chronological: bool
    duplicate_ids: list[str]
    first_timestamp: datetime | None
    last_timestamp: datetime | None


def load_messages(path: str | Path) -> tuple[list[Message], LoadReport]:
    """Load the message CSV and return messages sorted chronologically."""
    path = Path(path)
    # utf-8-sig strips the BOM present in the supplied files.
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    required = {"message_id", "timestamp", "sender", "message"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    seen: dict[str, int] = {}
    duplicates: list[str] = []
    staged: list[tuple[datetime | None, int, dict]] = []
    for i, row in enumerate(rows):
        mid = (row["message_id"] or "").strip()
        if mid in seen:
            duplicates.append(mid)
        seen[mid] = i
        staged.append((parse_timestamp(row["timestamp"]), i, row))

    already_sorted = all(
        staged[i][0] is not None
        and staged[i + 1][0] is not None
        and staged[i][0] <= staged[i + 1][0]
        for i in range(len(staged) - 1)
    )

    # Rows with an unparseable timestamp keep their original file position
    # instead of being dropped or shuffled to an invented date.
    staged.sort(key=lambda t: (t[0] is None, t[0] or datetime.min, t[1]))

    messages = [
        Message(
            message_id=(row["message_id"] or "").strip(),
            timestamp=ts,
            sender=(row["sender"] or "").strip(),
            text=(row["message"] or "").strip(),
            raw_timestamp=(row["timestamp"] or "").strip(),
            index=pos,
        )
        for pos, (ts, _, row) in enumerate(staged)
    ]

    parsed = [m.timestamp for m in messages if m.timestamp is not None]
    report = LoadReport(
        total_rows=len(messages),
        parsed_timestamps=len(parsed),
        unparsed_timestamps=len(messages) - len(parsed),
        was_already_chronological=already_sorted,
        duplicate_ids=duplicates,
        first_timestamp=min(parsed) if parsed else None,
        last_timestamp=max(parsed) if parsed else None,
    )
    return messages, report


def load_mandatory_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        col = reader.fieldnames[0] if reader.fieldnames else "message_id"
        return [(r[col] or "").strip() for r in reader if (r[col] or "").strip()]
