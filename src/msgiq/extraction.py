"""Part 2 - task and event extraction.

Governing rule from the brief: *do not guess missing information*. Every field
is therefore one of

* a value copied or deterministically derived from the message, or
* ``null`` with an entry in ``unresolved_fields`` explaining what was missing.

Date policy (three tiers, recorded in ``date_provenance``):

``explicit``
    An ISO date appears in the text. Copied verbatim.
``derived_from_timestamp``
    A relative word with exactly one possible meaning given the message's own
    timestamp - "tomorrow", "tonight", "today". Arithmetic, not inference.
``unresolved``
    Anything vague - "next week", "Friday afternoon", "soon". Left ``null``.
    We do not pick a plausible date, because a plausible date is a fabricated
    date.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta

from .config import PRIORITY_HIGH, PRIORITY_LOW, PRIORITY_MEDIUM
from .loader import Message, strip_lead_in

ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
CLOCK = re.compile(r"\b(\d{1,2}:\d{2})\b")
MERIDIEM = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.IGNORECASE)

# People who appear as senders in this corpus; used only to recognise names
# already present in the text. No name is ever invented.
KNOWN_PEOPLE = {
    "meera", "ishaan", "kabir", "aarav", "ananya", "neha",
    "vikram", "tara", "rohan", "maya",
}
PERSON_RE = re.compile(
    r"\b(" + "|".join(sorted(KNOWN_PEOPLE, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

VAGUE_TIME = re.compile(
    r"\b(?:next week|sometime|soon|later|shortly|this week|"
    r"friday afternoon|in a (?:few|couple)|afternoon|evening|morning)\b",
    re.IGNORECASE,
)
RELATIVE_DAY = re.compile(r"\b(today|tomorrow|tonight|yesterday)\b", re.IGNORECASE)

HEDGE = re.compile(r"\b(?:may|might|could|possibly|perhaps|if possible|"
                   r"when you are free|whenever)\b", re.IGNORECASE)
URGENT = re.compile(r"\b(?:urgent|asap|immediately|right away|critical|"
                    r"don'?t forget|must)\b", re.IGNORECASE)

LOCATION_RE = re.compile(
    r"(?:\bat\s+|\bin\s+|\bLocation:\s*)"
    r"(?P<loc>(?:the\s+)?(?:[A-Z][\w\-]*(?:\s+[A-Z]?[\w\-]+){0,3}|"
    r"[a-z]+(?:\s+[a-z]+){0,3}))\s*\.?$",
)

# ------------------------------------------------------------------- patterns
# Each entry: (regex, kind, title-group). Ordered specific -> general.
EVENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Calendar update:\s*(?P<title>[^,]+),", re.I), "event"),
    (re.compile(r"^Reminder:\s*(?P<title>.+?)\s+happens\s+on\b", re.I), "event"),
    (re.compile(r"^Please join\s+(?:the\s+)?(?P<title>.+?)\s+on\b", re.I), "event"),
    (re.compile(r"^The\s+(?P<title>.+?)\s+is scheduled for\b", re.I), "event"),
    (re.compile(r"^Are you available for\s+(?:the\s+)?(?P<title>.+?)\s+at\b", re.I), "event"),
    (re.compile(r"^(?P<title>Let us meet|Let's meet)\b.*", re.I), "event"),
    (re.compile(r"^The\s+(?P<title>\w+)\s+could be\b", re.I), "event"),
]

TASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:Can|Could|Would) you\s+(?P<title>.+?)\s*(?:before|by)\b", re.I), "task"),
    (re.compile(r"^(?:Can|Could|Would) you\s+(?P<title>.+?)\?", re.I), "task"),
    (re.compile(r"^Please\s+(?P<title>.+?)\s*(?:before|by)\b", re.I), "task"),
    (re.compile(r"^Please\s+(?P<title>.+?)\s*(?:when|\.)", re.I), "task"),
    (re.compile(r"^Don'?t forget to\s+(?P<title>.+?)\s*(?:;|,|\.)", re.I), "task"),
    (re.compile(r"^I need you to\s+(?P<title>.+?)\s*(?:before|by)\b", re.I), "task"),
    (re.compile(r"^(?P<title>.+?)\s+is due on\b", re.I), "task"),
    (re.compile(r"^If possible,\s*(?P<title>.+?)\s*(?:before|\.)", re.I), "task"),
    (re.compile(r"^(?P<title>.+?)\s+may be needed\b", re.I), "task"),
]


@dataclass
class Item:
    item_id: str
    type: str
    title: str
    description: str
    date: str | None
    time: str | None
    person: str | None
    location: str | None
    priority: str
    source_message_id: str
    sender: str
    date_provenance: str
    unresolved_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "date": self.date,
            "deadline": self.date if self.type == "task" else None,
            "time": self.time,
            "person": self.person,
            "location": self.location,
            "priority": self.priority,
            "source_message_id": self.source_message_id,
            "sender": self.sender,
            "confidence": round(self.confidence, 4),
            "date_provenance": self.date_provenance,
            "unresolved_fields": self.unresolved_fields,
            "notes": self.notes,
        }


def _clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" .,;:—-")
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.I)
    return t[:1].upper() + t[1:] if t else t


def _extract_time(text: str) -> str | None:
    m = CLOCK.search(text)
    if m:
        h, mm = m.group(1).split(":")
        return f"{int(h):02d}:{mm}"
    m = MERIDIEM.search(text)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2).lower() == "pm":
            h += 12
        return f"{h:02d}:00"
    return None


def _extract_date(text: str, msg: Message) -> tuple[str | None, str, list[str]]:
    """Return (iso_date_or_none, provenance, notes)."""
    m = ISO_DATE.search(text)
    if m:
        return m.group(1), "explicit", []

    rel = RELATIVE_DAY.search(text)
    if rel and msg.timestamp is not None:
        word = rel.group(1).lower()
        offset = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}[word]
        d = (msg.timestamp + timedelta(days=offset)).date().isoformat()
        return d, "derived_from_timestamp", [
            f"'{word}' resolved to {d} by adding {offset} day(s) to the message "
            f"timestamp; this is arithmetic, not an assumption."
        ]

    if VAGUE_TIME.search(text):
        phrase = VAGUE_TIME.search(text).group(0)
        return None, "unresolved", [
            f"Timing phrase '{phrase}' is not specific enough to resolve to a "
            f"calendar date; left null rather than guessed."
        ]
    return None, "unresolved", ["No date expression present in the message."]


def _extract_person(text: str, sender: str) -> tuple[str | None, list[str]]:
    """Only names actually written in the message body count."""
    hits = [h.capitalize() for h in PERSON_RE.findall(text)]
    hits = [h for h in dict.fromkeys(hits) if h.lower() != sender.lower()]
    if hits:
        return hits[0], []
    return None, ["No person named in the message text; the sender is recorded "
                  "separately but is not assumed to be the person involved."]


def _extract_location(text: str) -> str | None:
    body = text.rstrip()
    m = re.search(r"Location:\s*(?P<loc>.+?)\s*\.?$", body, re.I)
    if m:
        return _clean_title(m.group("loc"))
    m = re.search(r",\s*(?P<loc>(?:the\s+)?[\w][\w\s\-]{2,30})\.$", body)
    if m and not ISO_DATE.search(m.group("loc")):
        return _clean_title(m.group("loc"))
    m = re.search(r"\b(?:in|at)\s+(?P<loc>(?:the\s+)?[A-Za-z][\w\s\-]{2,30})\.$", body)
    if m:
        loc = m.group("loc")
        if not CLOCK.search(loc) and not ISO_DATE.search(loc):
            return _clean_title(loc)
    return None


def _priority(text: str, date: str | None, kind: str, hedged: bool) -> tuple[str, str]:
    if URGENT.search(text) and date:
        return PRIORITY_HIGH, "Urgency wording combined with a fixed date."
    if hedged:
        return PRIORITY_LOW, "Hedged wording ('may', 'could', 'if possible') " \
                             "signals this is optional or tentative."
    if date and kind == "task":
        return PRIORITY_HIGH, "A firm, dated deadline is stated."
    if date:
        return PRIORITY_MEDIUM, "A scheduled date is given but no urgency wording."
    return PRIORITY_LOW, "No date or urgency wording, so it cannot be ranked higher."


def extract(msg: Message, category: str) -> Item | None:
    """Extract one task/event from a message, or None if it contains neither."""
    core = strip_lead_in(msg.text)

    kind, title = None, None
    for rx, k in EVENT_PATTERNS + TASK_PATTERNS:
        m = rx.match(core)
        if m:
            kind, title = k, _clean_title(m.group("title"))
            break

    if kind is None:
        return None
    if not title or len(title) < 3:
        return None

    hedged = bool(HEDGE.search(core))
    date, provenance, notes = _extract_date(core, msg)
    time = _extract_time(core)
    person, person_notes = _extract_person(core, msg.sender)
    location = _extract_location(core)
    priority, prio_reason = _priority(core, date, kind, hedged)

    unresolved: list[str] = []
    if date is None:
        unresolved.append("deadline" if kind == "task" else "date")
    if time is None:
        unresolved.append("time")
    if person is None:
        unresolved.append("person")
    if location is None and kind == "event":
        unresolved.append("location")

    notes = notes + (person_notes if person is None else []) + [prio_reason]
    if hedged:
        notes.append("Message is hedged, so the item is recorded as tentative.")

    # Confidence: a full slate of resolved fields is worth more than a bare hit.
    resolved = sum(x is not None for x in (date, time, person, location))
    confidence = min(0.97, 0.55 + 0.10 * resolved - (0.10 if hedged else 0.0))

    return Item(
        item_id="",  # assigned by the pipeline in chronological order
        type=kind,
        title=title,
        description=core,
        date=date,
        time=time,
        person=person,
        location=location,
        priority=priority,
        source_message_id=msg.message_id,
        sender=msg.sender,
        date_provenance=provenance,
        unresolved_fields=unresolved,
        notes=notes,
        confidence=confidence,
    )
