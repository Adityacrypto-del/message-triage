"""Central configuration: categories, risk taxonomy, thresholds.

Everything tunable lives here so the video demo can point at one file.
"""
from __future__ import annotations

# ---------------------------------------------------------------- categories

ACTION_REQUIRED = "action_required"
MEETING_OR_EVENT = "meeting_or_event"
PERSONAL_INFORMATION = "personal_information"
GENERAL_INFORMATION = "general_information"
PROMOTIONAL = "promotional"
SENSITIVE_INFORMATION = "sensitive_information"

CATEGORIES = [
    ACTION_REQUIRED,
    MEETING_OR_EVENT,
    PERSONAL_INFORMATION,
    GENERAL_INFORMATION,
    PROMOTIONAL,
    SENSITIVE_INFORMATION,
]

# Resolution order when several rule families fire on the same message.
# Sensitive is first on purpose: a safety label must never lose a tie-break.
CATEGORY_PRECEDENCE = [
    SENSITIVE_INFORMATION,
    PROMOTIONAL,
    MEETING_OR_EVENT,
    ACTION_REQUIRED,
    PERSONAL_INFORMATION,
    GENERAL_INFORMATION,
]

# ------------------------------------------------------------- sensitivity

RISK_CRITICAL = "critical"
RISK_HIGH = "high"
RISK_MEDIUM = "medium"
RISK_LOW = "low"

RISK_ORDER = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2, RISK_CRITICAL: 3}

# The four recommendation strings named in the assignment brief.
ACTION_SAFE_LOCAL = "safe_to_process_locally"
ACTION_CONFIRM = "ask_for_confirmation"
ACTION_DO_NOT_STORE = "do_not_store"
ACTION_NO_EXTERNAL = "do_not_send_to_external_service"

# sensitivity_type -> (risk level, primary recommended action, extra actions)
SENSITIVITY_POLICY: dict[str, tuple[str, str, list[str]]] = {
    "password":            (RISK_CRITICAL, ACTION_DO_NOT_STORE, [ACTION_NO_EXTERNAL]),
    "one_time_password":   (RISK_CRITICAL, ACTION_DO_NOT_STORE, [ACTION_NO_EXTERNAL]),
    "auth_token":          (RISK_CRITICAL, ACTION_DO_NOT_STORE, [ACTION_NO_EXTERNAL]),
    "account_recovery_code": (RISK_CRITICAL, ACTION_DO_NOT_STORE, [ACTION_NO_EXTERNAL]),
    "payment_card":        (RISK_CRITICAL, ACTION_DO_NOT_STORE, [ACTION_NO_EXTERNAL]),
    "bank_account":        (RISK_CRITICAL, ACTION_DO_NOT_STORE, [ACTION_NO_EXTERNAL]),
    "personal_identifier": (RISK_HIGH,     ACTION_NO_EXTERNAL,  [ACTION_CONFIRM]),
    "health_information":  (RISK_HIGH,     ACTION_NO_EXTERNAL,  [ACTION_CONFIRM]),
    "email_address":       (RISK_MEDIUM,   ACTION_CONFIRM,      [ACTION_NO_EXTERNAL]),
    "private_address":     (RISK_MEDIUM,   ACTION_CONFIRM,      [ACTION_NO_EXTERNAL]),
    "contact_number":      (RISK_MEDIUM,   ACTION_CONFIRM,      [ACTION_NO_EXTERNAL]),
    "credential_reference": (RISK_LOW,     ACTION_SAFE_LOCAL,   []),
}

# ------------------------------------------------------------- priorities

PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

# ------------------------------------------------------------- thresholds

# Rule confidence at or above this is treated as trustworthy enough that the
# rule label wins even when the ML model disagrees.
RULE_TRUST_THRESHOLD = 0.80

# Rule confidence below this means "rule barely fired" -> defer to the model.
RULE_WEAK_THRESHOLD = 0.55

# Weak-supervision training set only uses examples the rules are confident on,
# so the model is not taught our own guesses.
TRAIN_CONFIDENCE_FLOOR = 0.75

# Blend weight when rule and model agree (higher = trust the rule more).
AGREEMENT_RULE_WEIGHT = 0.6

# Any final confidence below this gets flagged for human review.
REVIEW_FLAG_THRESHOLD = 0.60

RANDOM_SEED = 42
