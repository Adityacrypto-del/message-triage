"""Rule layer for Part 1.

Each rule scores a category and carries a human-readable reason. Rules do not
decide the final label on their own - they (a) supply weak labels used to train
the model in :mod:`msgiq.classifier` and (b) act as a high-precision override
for the categories where a mistake is expensive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import (
    ACTION_REQUIRED,
    CATEGORY_PRECEDENCE,
    GENERAL_INFORMATION,
    MEETING_OR_EVENT,
    PERSONAL_INFORMATION,
    PROMOTIONAL,
)


def _c(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


@dataclass(frozen=True)
class Rule:
    category: str
    regex: re.Pattern
    weight: float
    reason: str


RULES: list[Rule] = [
    # ------------------------------------------------------------ promotional
    Rule(PROMOTIONAL, _c(r"\buse code\s+[A-Z0-9]{3,}\b"), 0.95,
         "Contains a promotional discount code."),
    Rule(PROMOTIONAL, _c(r"\b(?:flash sale|special .{0,12}discount|limited-time offer|"
                         r"cashback|coupon|buy one .{0,15}get one|reward points|"
                         r"free delivery|upgrade your subscription|premium plan|"
                         r"\d{1,3}%\s*off|save \d{1,3}%)\b"), 0.9,
         "Uses marketing-offer language."),
    Rule(PROMOTIONAL, _c(r"\byou may like our\b|\bexclusive benefits\b|\bsubscribe now\b"), 0.8,
         "Pitches a product or plan to the reader."),

    # -------------------------------------------------------- meeting / event
    Rule(MEETING_OR_EVENT, _c(r"\b(?:is scheduled for|calendar update:|"
                              r"are you available for)\b"), 0.95,
         "Announces a scheduled calendar entry."),
    Rule(MEETING_OR_EVENT, _c(r"\bplease join\b|\bjoin (?:us|the)\b"), 0.93,
         "Invites the reader to attend something."),
    Rule(MEETING_OR_EVENT, _c(r"\breminder:\s*\w[\w\s\-]{2,40}\s+happens\s+on\b"), 0.93,
         "Reminder about an occasion with a date."),
    Rule(MEETING_OR_EVENT, _c(r"\b(?:meeting|stand-?up|sync|catch-?up|briefing|"
                              r"orientation|seminar|workshop|interview|review|demo|"
                              r"appointment|session|dinner|sprint planning)\b"
                              r"[^.]{0,60}\b(?:at|on)\b[^.]{0,20}"
                              r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2})"), 0.85,
         "Names an occasion together with an explicit date or time."),
    Rule(MEETING_OR_EVENT, _c(r"\blet us meet\b|\blet's meet\b|\bshall we meet\b"), 0.7,
         "Proposes a meeting, though the timing may be unspecified."),
    Rule(MEETING_OR_EVENT, _c(r"\b(?:review|meeting|call|sync|discussion|catch-?up|demo)\b"
                              r"[^.]{0,15}\bcould be\b"), 0.68,
         "Floats a tentative slot for an occasion without confirming it."),

    # -------------------------------------------------------- action required
    Rule(ACTION_REQUIRED, _c(r"\b(?:i need you to|please (?:submit|reply|send|complete|"
                             r"confirm|call|review|update|upload|finish|prepare|share))\b"), 0.92,
         "Contains a direct request addressed to the reader."),
    Rule(ACTION_REQUIRED, _c(r"\bdon'?t forget to\b|\bremember to\b|\bmake sure (?:to|you)\b"), 0.92,
         "Phrased as a reminder to perform a task."),
    Rule(ACTION_REQUIRED, _c(r"\b(?:is due on|deadline is|due by|by the end of)\b"), 0.9,
         "States an explicit deadline for a deliverable."),
    Rule(ACTION_REQUIRED, _c(r"\b(?:can|could|would) you\b[^?]*\?"), 0.85,
         "Asks the reader to carry out an action."),
    Rule(ACTION_REQUIRED, _c(r"\b(?:before|by)\s+\d{4}-\d{2}-\d{2}\b"), 0.8,
         "Ties a request to a dated cut-off."),
    Rule(ACTION_REQUIRED, _c(r"\bif possible,?\s+\w+\b|\bcould you send it soon\b"), 0.62,
         "Soft request with no firm deadline."),

    # ---------------------------------------------------- personal information
    Rule(PERSONAL_INFORMATION, _c(r"\b(?:for my profile|personal note:|just so you know|"
                                  r"remember that)\b[^.]{0,10}\bi\b"), 0.92,
         "States a personal preference or profile detail about the sender."),
    Rule(PERSONAL_INFORMATION, _c(r"\bmy (?:favourite|favorite|emergency contact|"
                                  r"t-?shirt size|dietary|preference)\b"), 0.9,
         "Declares a personal attribute of the sender."),
    Rule(PERSONAL_INFORMATION, _c(r"\bi (?:am|prefer|use|drink|live|usually|might prefer)\b"), 0.82,
         "First-person statement of habit or preference."),

    # ---------------------------------------------------- general information
    Rule(GENERAL_INFORMATION, _c(r"^\s*the\s+\w+[^?]*\.$"), 0.55,
         "Neutral third-person statement with no request or invitation."),
    Rule(GENERAL_INFORMATION, _c(r"\b(?:is (?:now )?available|has been updated|"
                                 r"was reorganized|under maintenance|closes at|opens at|"
                                 r"leaves every|extended|changed its|forecast says|"
                                 r"public holiday|is on the portal|fully charged|"
                                 r"moved temporarily)\b"), 0.8,
         "Announcement or status update requiring no action."),
]


@dataclass
class RuleVerdict:
    category: str | None
    confidence: float
    reason: str
    scores: dict[str, float]
    matched: list[str]


def apply_rules(text: str) -> RuleVerdict:
    """Score every category; return the winner under the precedence order."""
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    matched: list[str] = []

    for rule in RULES:
        if rule.regex.search(text):
            matched.append(f"{rule.category}:{rule.reason}")
            # Take the strongest single rule per category rather than summing,
            # so a category cannot win just by matching many weak patterns.
            if rule.weight > scores.get(rule.category, 0.0):
                scores[rule.category] = rule.weight
                reasons[rule.category] = rule.reason

    if not scores:
        return RuleVerdict(None, 0.0, "No rule matched.", {}, [])

    best = max(scores.values())
    tied = [c for c, s in scores.items() if s == best]
    winner = min(tied, key=CATEGORY_PRECEDENCE.index)

    # A close runner-up in a different category means the message is genuinely
    # ambiguous, so shave the confidence rather than reporting a false certainty.
    runner_up = max((s for c, s in scores.items() if c != winner), default=0.0)
    margin = best - runner_up
    confidence = best if margin >= 0.15 else round(best - (0.15 - margin), 4)

    return RuleVerdict(
        category=winner,
        confidence=round(max(0.3, min(0.99, confidence)), 4),
        reason=reasons[winner],
        scores=scores,
        matched=matched,
    )
