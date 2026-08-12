"""Part 1 - hybrid classifier: rules + locally-trained TF-IDF / logistic model.

Why hybrid
----------
The dataset ships with no labels, so there is nothing to train on directly.
The rule layer supplies *weak labels*; a TF-IDF + logistic-regression model is
then trained on the confident subset of those labels. That buys three things a
pure rule list cannot give:

* a real probability per category, so ``confidence`` means something;
* generalisation to phrasings the regexes never anticipated;
* a second opinion - when rule and model disagree the message is genuinely
  ambiguous, and we can say so instead of pretending certainty.

Safety carve-out: sensitive detections from :mod:`msgiq.sensitive` are never
overturned by the model. A false negative there is a data leak; a false
positive is a minor annoyance.

The model is trained from scratch on the local machine at pipeline runtime. No
message text leaves the process; there is no network call anywhere in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import (
    AGREEMENT_RULE_WEIGHT,
    CATEGORIES,
    GENERAL_INFORMATION,
    RANDOM_SEED,
    REVIEW_FLAG_THRESHOLD,
    RULE_TRUST_THRESHOLD,
    RULE_WEAK_THRESHOLD,
    SENSITIVE_INFORMATION,
    TRAIN_CONFIDENCE_FLOOR,
)
from .loader import Message, strip_lead_in
from .rules import apply_rules
from .sensitive import ScanResult


@dataclass
class Classification:
    message_id: str
    category: str
    confidence: float
    reason: str
    decision_source: str
    rule_category: str | None = None
    rule_confidence: float = 0.0
    model_category: str | None = None
    model_confidence: float = 0.0
    needs_review: bool = False
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "decision_source": self.decision_source,
            "needs_review": self.needs_review,
            "evidence": {
                "rule_category": self.rule_category,
                "rule_confidence": round(self.rule_confidence, 4),
                "model_category": self.model_category,
                "model_confidence": round(self.model_confidence, 4),
                "runner_up": [
                    {"category": c, "probability": round(p, 4)}
                    for c, p in self.alternatives
                ],
            },
        }


def build_model() -> Pipeline:
    """Word n-grams catch phrasing; char n-grams survive typos and new nouns."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=1, lowercase=True, strip_accents="unicode",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=2, lowercase=True,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=4.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])


def normalise(text: str) -> str:
    """Feature-space normalisation.

    Concrete dates, times and numbers are replaced with type tokens. Without
    this the model memorises `2026-09-14` as a feature and learns nothing about
    the *shape* of a deadline.
    """
    import re
    t = strip_lead_in(text)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " <date> ", t)
    t = re.sub(r"\b\d{1,2}:\d{2}\b", " <time> ", t)
    t = re.sub(r"\b\d{1,2}\s*(?:am|pm)\b", " <time> ", t, flags=re.I)
    t = re.sub(r"\b[A-Z]{3,}\d+\b", " <code> ", t)
    t = re.sub(r"\d+", " <num> ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


@dataclass
class TrainingReport:
    n_total: int
    n_training: int
    cv_accuracy: float
    cv_macro_f1: float
    per_class: dict
    confusion: list[list[int]]
    labels: list[str]
    coverage: float           # share of messages a rule fired on
    agreement: float          # rule/model agreement on the full set
    excluded_low_confidence: int


class HybridClassifier:
    """Rules for precision, a trained model for coverage and calibration."""

    def __init__(self) -> None:
        self.model = build_model()
        self.report: TrainingReport | None = None
        self._classes: list[str] = []

    # -------------------------------------------------------------- training
    def fit(self, messages: list[Message], scans: dict[str, ScanResult]) -> TrainingReport:
        texts, weak_labels, keep = [], [], []
        rule_hits = 0

        for m in messages:
            core = strip_lead_in(m.text)
            scan = scans.get(m.message_id)
            if scan is not None and scan.is_sensitive and scan.risk != "low":
                label, conf = SENSITIVE_INFORMATION, scan.confidence
                rule_hits += 1
            else:
                v = apply_rules(core)
                if v.category is None:
                    texts.append(normalise(m.text)); weak_labels.append(None); keep.append(False)
                    continue
                label, conf = v.category, v.confidence
                rule_hits += 1

            texts.append(normalise(m.text))
            weak_labels.append(label)
            keep.append(conf >= TRAIN_CONFIDENCE_FLOOR)

        X = [t for t, k in zip(texts, keep) if k]
        y = [l for l, k in zip(weak_labels, keep) if k]

        # Any class with fewer than 5 examples cannot be cross-validated 5-fold.
        counts = {c: y.count(c) for c in set(y)}
        rare = {c for c, n in counts.items() if n < 5}
        if rare:
            X = [x for x, l in zip(X, y) if l not in rare]
            y = [l for l in y if l not in rare]

        folds = min(5, min(y.count(c) for c in set(y)))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
        y_pred = cross_val_predict(build_model(), X, y, cv=cv, n_jobs=1)

        rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
        labels = sorted(set(y))
        cm = confusion_matrix(y, y_pred, labels=labels).tolist()

        self.model.fit(X, y)
        self._classes = list(self.model.named_steps["clf"].classes_)

        # Rule/model agreement measured on every message, training subset or not.
        probs = self.model.predict_proba([normalise(m.text) for m in messages])
        model_labels = [self._classes[i] for i in probs.argmax(axis=1)]
        comparable = [(w, p) for w, p in zip(weak_labels, model_labels) if w is not None]
        agreement = (sum(1 for w, p in comparable if w == p) / len(comparable)) if comparable else 0.0

        self.report = TrainingReport(
            n_total=len(messages),
            n_training=len(X),
            cv_accuracy=float(rep["accuracy"]),
            cv_macro_f1=float(rep["macro avg"]["f1-score"]),
            per_class={k: v for k, v in rep.items() if k in labels},
            confusion=cm,
            labels=labels,
            coverage=rule_hits / len(messages),
            agreement=agreement,
            excluded_low_confidence=sum(1 for l, k in zip(weak_labels, keep) if l and not k),
        )
        return self.report

    # ------------------------------------------------------------ prediction
    def _model_predict(self, text: str) -> tuple[str, float, list[tuple[str, float]]]:
        p = self.model.predict_proba([normalise(text)])[0]
        order = np.argsort(p)[::-1]
        top = self._classes[order[0]]
        alts = [(self._classes[i], float(p[i])) for i in order[1:3]]
        return top, float(p[order[0]]), alts

    def classify(self, message: Message, scan: ScanResult | None) -> Classification:
        core = strip_lead_in(message.text)
        model_cat, model_conf, alts = self._model_predict(message.text)

        # 1. Safety override - a confirmed secret is sensitive, full stop.
        if scan is not None and scan.is_sensitive and scan.risk != "low":
            return Classification(
                message_id=message.message_id,
                category=SENSITIVE_INFORMATION,
                confidence=round(min(0.99, scan.confidence), 4),
                reason=(f"Detector override: {scan.reason} "
                        f"Sensitive content outranks any other category."),
                decision_source="sensitive_detector_override",
                rule_category=SENSITIVE_INFORMATION,
                rule_confidence=scan.confidence,
                model_category=model_cat,
                model_confidence=model_conf,
                alternatives=alts,
            )

        # 1b. A credential *mention* with no secret in it. Worth surfacing - it
        #     predicts that a secret is about to arrive - but the confidence
        #     stays low and the message is flagged, because reasonable people
        #     would file this under general information instead.
        if scan is not None and scan.is_sensitive and scan.risk == "low":
            return Classification(
                message_id=message.message_id,
                category=SENSITIVE_INFORMATION,
                confidence=0.55,
                reason=(f"{scan.reason} Categorised as sensitive so the thread is "
                        f"watched, but confidence is deliberately low and the "
                        f"message is flagged: nothing here actually needs masking."),
                decision_source="sensitive_reference_low_confidence",
                rule_category=SENSITIVE_INFORMATION,
                rule_confidence=scan.confidence,
                model_category=model_cat,
                model_confidence=model_conf,
                needs_review=True,
                alternatives=alts,
            )

        v = apply_rules(core)

        # 2. No rule fired - the model is the only opinion available.
        if v.category is None:
            conf = round(model_conf * 0.9, 4)  # discount: unsupported by rules
            return Classification(
                message_id=message.message_id,
                category=model_cat,
                confidence=conf,
                reason=(f"No rule pattern matched; the trained model assigns "
                        f"'{model_cat}' with probability {model_conf:.2f} based on "
                        f"wording similar to other {model_cat} messages."),
                decision_source="model_only",
                model_category=model_cat,
                model_confidence=model_conf,
                needs_review=conf < REVIEW_FLAG_THRESHOLD,
                alternatives=alts,
            )

        # 3. Rule and model agree - blend for a calibrated score.
        if v.category == model_cat:
            conf = AGREEMENT_RULE_WEIGHT * v.confidence + (1 - AGREEMENT_RULE_WEIGHT) * model_conf
            conf = round(min(0.99, conf + 0.05), 4)  # small agreement bonus
            return Classification(
                message_id=message.message_id,
                category=v.category,
                confidence=conf,
                reason=f"{v.reason} The trained model independently agrees "
                       f"(p={model_conf:.2f}).",
                decision_source="rule+model_agree",
                rule_category=v.category, rule_confidence=v.confidence,
                model_category=model_cat, model_confidence=model_conf,
                alternatives=alts,
            )

        # 4. They disagree. Trust a strong rule; otherwise take the model and
        #    flag the message, because this is exactly where errors live.
        if v.confidence >= RULE_TRUST_THRESHOLD:
            conf = round(max(0.5, v.confidence - 0.15), 4)
            return Classification(
                message_id=message.message_id,
                category=v.category,
                confidence=conf,
                reason=(f"{v.reason} Confidence reduced because the trained model "
                        f"preferred '{model_cat}' (p={model_conf:.2f})."),
                decision_source="rule_override_model",
                rule_category=v.category, rule_confidence=v.confidence,
                model_category=model_cat, model_confidence=model_conf,
                needs_review=conf < REVIEW_FLAG_THRESHOLD,
                alternatives=alts,
            )

        # Both signals are weak. Take whichever is less unsure, keep the score
        # low, and flag it. Falling back to a blanket 'general_information'
        # would throw away the one targeted pattern that did fire.
        if v.confidence >= model_conf:
            chosen, conf, src = v.category, round(v.confidence * 0.8, 4), "weak_rule_over_model"
            detail = (f"a hand-written rule suggested '{v.category}' "
                      f"({v.confidence:.2f}) and was followed by a narrow margin")
        else:
            chosen, conf, src = model_cat, round(model_conf * 0.85, 4), "model_over_weak_rule"
            detail = (f"the model preferred '{model_cat}' ({model_conf:.2f}) over a "
                      f"weak rule suggesting '{v.category}' ({v.confidence:.2f})")

        return Classification(
            message_id=message.message_id,
            category=chosen,
            confidence=max(0.3, conf),
            reason=(f"Ambiguous message: {detail}. Confidence is low and the "
                    f"message is flagged for human review."),
            decision_source=src,
            rule_category=v.category, rule_confidence=v.confidence,
            model_category=model_cat, model_confidence=model_conf,
            needs_review=True,
            alternatives=alts,
        )
