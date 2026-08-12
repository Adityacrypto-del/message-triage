"""Part 3 - sensitive information detection and masking.

Design notes
------------
1. Detection is *contextual first, format second*. A pattern like
   ``account recovery code is <X>`` is matched on the surrounding phrase, so
   the detector does not depend on the exact shape of the secret. Pure-format
   detectors (card numbers, emails) act as a safety net for values that arrive
   with no framing text.
2. Every detector captures the secret in a named group called ``value``. The
   masker only ever redacts that span, so the surrounding message stays
   readable and a human can still judge what the message was about.
3. Masking happens once, at detection time. Nothing downstream - logs, output
   files, the Streamlit UI - is given a route to the raw value. See
   :func:`scan_message`, which returns the raw span length only, never the span.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import RISK_ORDER, SENSITIVITY_POLICY

MASK_CHAR = "*"
MASK_MIN, MASK_MAX = 4, 12


@dataclass(frozen=True)
class Detector:
    sensitivity_type: str
    regex: re.Pattern
    reason: str
    # Confidence that a hit is genuinely sensitive (drives classifier weight).
    precision: float = 0.95
    # Format-only detectors are suppressed if a contextual one already fired.
    format_only: bool = False
    # Identifier-like secrets must contain a digit. Without this guard, phrases
    # such as "our new student plan" match the identifier rule and "plan" is
    # redacted as if it were an ID number.
    require_digit: bool = False


def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# ------------------------------------------------------------------ detectors
# Ordered most specific -> most general. `value` is the span that gets masked.

DETECTORS: list[Detector] = [
    # --- credentials -------------------------------------------------------
    Detector(
        "one_time_password",
        _c(r"\b(?:otp|one[\s\-]?time\s+(?:password|code|pin)|verification\s+code)\b"
           r"(?:\s+(?:is|:))?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9\-]{2,})"),
        "Message states a one-time password / verification code.",
        precision=0.99,
    ),
    Detector(
        "password",
        _c(r"\b(?:password|passcode|pass\s?phrase)\b\s*(?:is|:|=)?\s*"
           r"(?P<value>[^\s,.;]{4,})"),
        "Message states an account password in plain text.",
        precision=0.99,
    ),
    Detector(
        "account_recovery_code",
        _c(r"\b(?:account\s+)?recovery\s+(?:code|key)\b\s*(?:is|:)?\s*"
           r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-_]{2,})"),
        "Message states an account recovery code.",
        precision=0.99,
    ),
    Detector(
        "auth_token",
        _c(r"\b(?:access|auth(?:entication|orisation|orization)?|api|bearer|session)"
           r"[\s\-]?(?:token|key|secret)\b\s*(?:is|:|=)?\s*"
           r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-_.]{4,})"),
        "Message states an authentication or access token.",
        precision=0.99,
    ),
    Detector(
        "auth_token",
        _c(r"\b(?P<value>(?:tok|sk|pk|ghp|xox[baprs])[_\-][A-Za-z0-9\-_.]{6,})\b"),
        "Token-shaped string detected by format.",
        precision=0.9,
        format_only=True,
    ),
    Detector(
        "personal_identifier",
        # The qualifier must be followed by an explicit noun ("number", "no",
        # "id", "card"). Matching the qualifier alone made "student plan" look
        # like a student ID.
        _c(r"\b(?:identification|identity|aadhaar|aadhar|pan|passport|"
           r"national|employee|student|voter|licence|license)\s*"
           r"(?:number|no\.?|id|card)\b"
           r"\s*(?:is|:)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9\-/]{3,})"),
        "Message states a personal identification number.",
        precision=0.95,
        require_digit=True,
    ),

    # --- financial ---------------------------------------------------------
    Detector(
        "payment_card",
        _c(r"\b(?:card|credit\s*card|debit\s*card)\s*(?:number|no\.?)\b"
           r"\s*(?:is|:)?\s*(?P<value>[0-9][0-9\s\-]{8,})"),
        "Message states a payment card number.",
        precision=0.99,
    ),
    Detector(
        "payment_card",
        _c(r"(?P<value>\b(?:\d{4}[\s\-]){3}\d{4}\b)"),
        "16-digit card-shaped number detected by format.",
        precision=0.9,
        format_only=True,
    ),
    Detector(
        "bank_account",
        _c(r"\b(?:bank\s+)?account\s+(?:number|no\.?)\b\s*(?:is|:)?\s*"
           r"(?P<value>[0-9][0-9\s\-]{3,})"),
        "Message states a bank account number.",
        precision=0.99,
    ),
    Detector(
        "bank_account",
        _c(r"\b(?:ifsc|iban|swift|routing|sort\s*code|upi\s*id)\b\s*(?:is|:)?\s*"
           r"(?P<value>[A-Za-z0-9@][A-Za-z0-9@\-]{3,})"),
        "Message states bank routing / payment handle details.",
        precision=0.97,
    ),
    Detector(
        "one_time_password",
        _c(r"\b(?:pin)\b\s*(?:is|:|=)?\s*(?P<value>\d{4,8})\b"),
        "Message states a numeric PIN.",
        precision=0.97,
    ),

    # --- special-category personal data ------------------------------------
    Detector(
        "health_information",
        _c(r"\b(?:test\s+result|diagnos\w+|prescription|blood\s+(?:group|test)|"
           r"medical\s+report)\b[^.]{0,20}?(?:says|is|:)\s*(?P<value>[^.]+)"),
        "Message discloses a medical or health finding.",
        precision=0.9,
    ),

    # --- contactability ----------------------------------------------------
    Detector(
        "private_address",
        _c(r"\b(?:home|residential|postal|house|my)\s+address\b\s*(?:is|:)?\s*"
           r"(?P<value>[^.]+)"),
        "Message discloses a private residential address.",
        precision=0.95,
    ),
    Detector(
        "contact_number",
        _c(r"\b(?:contact|call|reach|whatsapp|text)\s+me\s+(?:on|at)\b\s*"
           r"(?P<value>[+0-9][0-9\s\-]{6,})"),
        "Message discloses a personal contact number.",
        precision=0.97,
    ),
    Detector(
        "contact_number",
        _c(r"\b(?:phone|mobile|cell|contact)\s*(?:number|no\.?)\b\s*(?:is|:)?\s*"
           r"(?P<value>[+0-9][0-9\s\-]{6,})"),
        "Message discloses a personal contact number.",
        precision=0.97,
    ),
    Detector(
        "email_address",
        _c(r"(?P<value>\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b)"),
        "Message contains an email address.",
        precision=0.9,
        format_only=True,
    ),
]

# Credential *mentions* that carry no secret value. Kept separate because the
# right answer is "flag, but do not panic" - there is nothing to mask.
REFERENCE_ONLY = _c(
    r"\b(?:login|log-?in|sign-?in|account)\s+(?:details|credentials|info(?:rmation)?)\b"
    r"|\bcredentials\b|\bwill send (?:the|my) (?:password|otp|code)\b"
)


@dataclass
class Finding:
    sensitivity_type: str
    risk: str
    recommended_action: str
    additional_actions: list[str]
    reason: str
    precision: float
    span: tuple[int, int] | None
    value_length: int = 0
    value_preview: str = ""  # masked preview only - never the raw value


@dataclass
class ScanResult:
    message_id: str
    is_sensitive: bool
    findings: list[Finding] = field(default_factory=list)
    masked_text: str = ""
    risk: str | None = None
    sensitivity_type: str | None = None
    recommended_action: str | None = None
    additional_actions: list[str] = field(default_factory=list)
    reason: str = ""
    # Highest detector precision that fired; used as classifier confidence.
    confidence: float = 0.0

    @property
    def types(self) -> list[str]:
        seen: list[str] = []
        for f in self.findings:
            if f.sensitivity_type not in seen:
                seen.append(f.sensitivity_type)
        return seen


def _mask_for(value: str) -> str:
    """Fixed-width mask.

    Deliberately does *not* scale with the true length: a mask that mirrors
    length leaks how long the secret is, which narrows a brute-force search.
    """
    n = min(MASK_MAX, max(MASK_MIN, 6))
    return MASK_CHAR * n


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def scan_message(message_id: str, text: str) -> ScanResult:
    """Detect, rank, and mask sensitive content in a single message."""
    raw_hits: list[tuple[Detector, tuple[int, int], str]] = []

    for det in DETECTORS:
        for m in det.regex.finditer(text):
            try:
                value = m.group("value")
            except IndexError:
                continue
            if value is None:
                continue
            value = value.strip().rstrip(".,;:")
            if len(value) < 2:
                continue
            if det.require_digit and not any(ch.isdigit() for ch in value):
                continue
            start = m.start("value")
            raw_hits.append((det, (start, start + len(value)), value))

    # Contextual detectors beat format-only detectors on the same span, and
    # earlier (more specific) detectors beat later ones.
    raw_hits.sort(key=lambda h: (h[0].format_only, DETECTORS.index(h[0])))
    kept: list[tuple[Detector, tuple[int, int], str]] = []
    for hit in raw_hits:
        if any(_overlaps(hit[1], k[1]) for k in kept):
            continue
        kept.append(hit)

    if not kept:
        # No secret value - but is a credential being referred to?
        if REFERENCE_ONLY.search(text):
            risk, action, extra = SENSITIVITY_POLICY["credential_reference"]
            f = Finding(
                sensitivity_type="credential_reference",
                risk=risk,
                recommended_action=action,
                additional_actions=list(extra),
                reason=("Message refers to credentials being shared but contains "
                        "no secret value, so there is nothing to redact."),
                precision=0.6,
                span=None,
            )
            return ScanResult(
                message_id=message_id, is_sensitive=True, findings=[f],
                masked_text=text, risk=risk, sensitivity_type=f.sensitivity_type,
                recommended_action=action, additional_actions=list(extra),
                reason=f.reason, confidence=0.6,
            )
        return ScanResult(message_id=message_id, is_sensitive=False, masked_text=text)

    findings: list[Finding] = []
    for det, span, value in kept:
        risk, action, extra = SENSITIVITY_POLICY[det.sensitivity_type]
        findings.append(Finding(
            sensitivity_type=det.sensitivity_type,
            risk=risk,
            recommended_action=action,
            additional_actions=list(extra),
            reason=det.reason,
            precision=det.precision,
            span=span,
            value_length=len(value),
            value_preview=_mask_for(value),
        ))

    # Apply masks back-to-front so earlier spans keep their offsets.
    masked = text
    for f in sorted(findings, key=lambda x: x.span[0], reverse=True):
        s, e = f.span
        masked = masked[:s] + _mask_for(masked[s:e]) + masked[e:]

    top = max(findings, key=lambda f: (RISK_ORDER[f.risk], f.precision))
    # Aggregate: the strictest action across all findings wins.
    all_actions: list[str] = []
    for f in findings:
        for a in [f.recommended_action, *f.additional_actions]:
            if a not in all_actions:
                all_actions.append(a)

    types = ", ".join(dict.fromkeys(f.sensitivity_type for f in findings))
    reason = top.reason if len(findings) == 1 else (
        f"{len(findings)} sensitive values detected ({types}); "
        f"strictest handling applied."
    )

    return ScanResult(
        message_id=message_id,
        is_sensitive=True,
        findings=findings,
        masked_text=masked,
        risk=top.risk,
        sensitivity_type=top.sensitivity_type,
        recommended_action=top.recommended_action,
        additional_actions=[a for a in all_actions if a != top.recommended_action],
        reason=reason,
        confidence=round(min(0.99, top.precision), 4),
    )


def redact_free_text(text: str) -> str:
    """Belt-and-braces masker for any string about to be displayed or logged."""
    return scan_message("_", text).masked_text
