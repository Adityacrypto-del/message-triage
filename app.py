"""Streamlit demo UI.

Safety note for anyone reading this file: the UI never touches raw message
text. Every display path goes through the masked copy produced by the detector,
so a screen recording of this app cannot leak a secret even by accident.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from msgiq.classifier import HybridClassifier  # noqa: E402
from msgiq.config import CATEGORIES, REVIEW_FLAG_THRESHOLD  # noqa: E402
from msgiq.extraction import extract  # noqa: E402
from msgiq.loader import Message, parse_timestamp, strip_lead_in  # noqa: E402
from msgiq.pipeline import run  # noqa: E402
from msgiq.sensitive import scan_message  # noqa: E402

st.set_page_config(page_title="Message Triage", page_icon="■", layout="wide")

RISK_COLOUR = {"critical": "#b3261e", "high": "#e8710a",
               "medium": "#b58900", "low": "#4a7c59"}
DATA_DIR = ROOT / "data"


@st.cache_resource(show_spinner="Running pipeline (load → scan → train → classify → extract)...")
def bootstrap():
    result = run(DATA_DIR, verbose=False)
    clf = HybridClassifier()
    clf.fit(result.messages, result.scans)
    return result, clf


def rows_frame(res) -> pd.DataFrame:
    data = []
    for m in res.messages:
        c = res.classifications[m.message_id]
        s = res.scans[m.message_id]
        data.append({
            "message_id": m.message_id,
            "timestamp": m.raw_timestamp,
            "sender": m.sender,
            "message (masked)": s.masked_text,     # never m.text
            "category": c.category,
            "confidence": round(c.confidence, 3),
            "source": c.decision_source,
            "risk": s.risk or "",
            "flagged": "yes" if c.needs_review else "",
            "reason": c.reason,
        })
    return pd.DataFrame(data)


if not (DATA_DIR / "messages.csv").exists():
    st.error("`data/messages.csv` not found.")
    st.info("The dataset is deliberately excluded from this repository as the "
            "brief requires. Place `messages.csv` and `mandatory_demo_ids.csv` "
            "in a `data/` folder to run the demo.")
    st.stop()

res, clf = bootstrap()
df = rows_frame(res)
tr = res.training_report

st.title("Message Triage")
st.caption("Classification, task/event extraction, and sensitive-data redaction. "
           "Runs entirely offline — no message text leaves this process.")

tabs = st.tabs([
    "Overview", "Classification", "Mandatory IDs", "Tasks & Events",
    "Sensitive", "Uncertain", "Try a message",
])

# ------------------------------------------------------------------ overview
with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("Messages", f"{len(res.messages):,}")
    b.metric("Tasks + events", f"{len(res.items):,}")
    c.metric("Sensitive messages", f"{sum(1 for s in res.scans.values() if s.is_sensitive):,}")
    d.metric("Flagged for review", f"{len(res.flagged):,}")

    st.subheader("Pipeline")
    st.code(
        "1. load        chronological sort + timestamp validation\n"
        "2. scan        sensitive detection; values masked at this point\n"
        "3. train       rules produce weak labels -> TF-IDF + logistic regression\n"
        "4. classify    rules and model combined; disagreements flagged\n"
        "5. extract     tasks/events; unresolved fields left null",
        language="text")

    left, right = st.columns(2)
    with left:
        st.subheader("Category distribution")
        st.bar_chart(df["category"].value_counts())
    with right:
        st.subheader("How each decision was made")
        st.dataframe(
            df["source"].value_counts().rename_axis("decision_source")
              .reset_index(name="messages"),
            hide_index=True, width="stretch")

    st.subheader("Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training examples", f"{tr.n_training:,}")
    m2.metric("5-fold CV accuracy", f"{tr.cv_accuracy:.3f}")
    m3.metric("Rule coverage", f"{tr.coverage:.1%}")
    m4.metric("Rule/model agreement", f"{tr.agreement:.1%}")
    st.info("CV accuracy is measured against **rule-derived weak labels**, not a "
            "human-annotated gold set. It shows the model reproduces the rule "
            "logic from wording alone; it is not proof of real-world accuracy. "
            "The honest headline number is the 97.7% rule/model agreement.")

    st.subheader("Dataset structure")
    st.dataframe(pd.DataFrame([{
        "column": k, "example": v} for k, v in {
            "message_id": res.messages[0].message_id,
            "timestamp": res.messages[0].raw_timestamp,
            "sender": res.messages[0].sender,
            "message": "(shown masked throughout this app)",
        }.items()]), hide_index=True, width="stretch")

# ------------------------------------------------------------ classification
with tabs[1]:
    st.subheader("All 900 classifications")
    f1, f2, f3 = st.columns([2, 2, 3])
    cats = f1.multiselect("Category", CATEGORIES, default=CATEGORIES)
    lo = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    q = f3.text_input("Search text or message ID")

    view = df[df["category"].isin(cats) & (df["confidence"] >= lo)]
    if q:
        mask = (view["message (masked)"].str.contains(q, case=False, na=False)
                | view["message_id"].str.contains(q, case=False, na=False))
        view = view[mask]
    st.caption(f"{len(view)} of {len(df)} messages")
    st.dataframe(view, hide_index=True, width="stretch", height=420)

    st.subheader("Explain one decision")
    pick = st.selectbox("Message", view["message_id"].tolist()[:400] or ["—"])
    if pick in res.classifications:
        c = res.classifications[pick]
        st.write(f"**{res.scans[pick].masked_text}**")
        k1, k2, k3 = st.columns(3)
        k1.metric("Category", c.category)
        k2.metric("Confidence", f"{c.confidence:.2f}")
        k3.metric("Decision source", c.decision_source)
        st.write("**Reason**"); st.write(c.reason)
        st.json(c.to_dict()["evidence"])

# ------------------------------------------------------------ mandatory IDs
with tabs[2]:
    st.subheader("15 mandatory demonstration IDs")
    demo = df[df["message_id"].isin(res.mandatory_ids)].copy()
    demo["order"] = demo["message_id"].map(
        {m: i for i, m in enumerate(res.mandatory_ids)})
    demo = demo.sort_values("order").drop(columns="order")
    st.dataframe(demo, hide_index=True, width="stretch", height=560)
    missing = set(res.mandatory_ids) - set(df["message_id"])
    if missing:
        st.error(f"Missing from the dataset: {sorted(missing)}")
    else:
        st.success(f"All {len(res.mandatory_ids)} IDs resolved.")
    for mid in res.mandatory_ids:
        if mid not in res.classifications:
            continue
        c = res.classifications[mid]
        item = next((i for i in res.items if i.source_message_id == mid), None)
        with st.expander(f"{mid} — {c.category} ({c.confidence:.2f})"):
            st.write(res.scans[mid].masked_text)
            st.caption(c.reason)
            s = res.scans[mid]
            if s.is_sensitive:
                st.markdown(
                    f"Sensitivity **{s.sensitivity_type}** · risk "
                    f"**:red[{s.risk}]** · action `{s.recommended_action}`")
            if item:
                st.json(item.to_dict())

# --------------------------------------------------------- tasks and events
with tabs[3]:
    st.subheader("Extracted tasks and events")
    kind = st.radio("Type", ["all", "task", "event"], horizontal=True)
    only_unresolved = st.checkbox("Only items with unresolved fields")

    idf = pd.DataFrame([{
        "item_id": i.item_id, "type": i.type, "title": i.title,
        "date": i.date or "—", "time": i.time or "—",
        "person": i.person or "—", "location": i.location or "—",
        "priority": i.priority, "confidence": round(i.confidence, 2),
        "date_provenance": i.date_provenance,
        "unresolved": ", ".join(i.unresolved_fields) or "—",
        "source": i.source_message_id,
    } for i in res.items])

    v = idf if kind == "all" else idf[idf["type"] == kind]
    if only_unresolved:
        v = v[v["unresolved"] != "—"]
    st.caption(f"{len(v)} items · dashes mark fields left null on purpose")
    st.dataframe(v, hide_index=True, width="stretch", height=420)

    st.subheader("Item detail")
    sel = st.selectbox("Item", v["item_id"].tolist()[:400] or ["—"])
    hit = next((i for i in res.items if i.item_id == sel), None)
    if hit:
        st.json(hit.to_dict())
        if hit.unresolved_fields:
            st.warning("Unresolved: " + ", ".join(hit.unresolved_fields)
                       + " — left null rather than guessed.")
        for n in hit.notes:
            st.caption("· " + n)

# ----------------------------------------------------------------- sensitive
with tabs[4]:
    st.subheader("Sensitive information")
    st.caption("Values are masked at detection time. The raw value is never "
               "loaded into this page, so it cannot appear in a recording.")

    sens = [(m, res.scans[m.message_id]) for m in res.messages
            if res.scans[m.message_id].is_sensitive]
    c1, c2 = st.columns(2)
    with c1:
        st.write("**By risk level**")
        st.dataframe(pd.Series([s.risk for _, s in sens]).value_counts()
                     .rename_axis("risk").reset_index(name="messages"),
                     hide_index=True, width="stretch")
    with c2:
        st.write("**By recommended action**")
        st.dataframe(pd.Series([s.recommended_action for _, s in sens]).value_counts()
                     .rename_axis("action").reset_index(name="messages"),
                     hide_index=True, width="stretch")

    sdf = pd.DataFrame([{
        "message_id": m.message_id, "sender": m.sender,
        "type": s.sensitivity_type, "risk": s.risk,
        "masked_text": s.masked_text,
        "recommended_action": s.recommended_action,
        "values_redacted": sum(1 for f in s.findings if f.span),
    } for m, s in sens])

    pickr = st.multiselect("Risk", ["critical", "high", "medium", "low"],
                           default=["critical", "high", "medium", "low"])
    st.dataframe(sdf[sdf["risk"].isin(pickr)], hide_index=True,
                 width="stretch", height=420)

    st.subheader("One per sensitivity type")
    for t in sdf["type"].unique():
        m, s = next((m, s) for m, s in sens if s.sensitivity_type == t)
        with st.expander(f"{t} — {s.risk}"):
            st.code(s.masked_text, language="text")
            st.markdown(f"**Risk** :red[{s.risk}] · **Action** "
                        f"`{s.recommended_action}`")
            if s.additional_actions:
                st.caption("Also: " + ", ".join(s.additional_actions))
            st.caption(s.reason)

# ----------------------------------------------------------------- uncertain
with tabs[5]:
    st.subheader("Where the system is unsure")
    st.caption("Messages the pipeline flagged itself. These are the honest "
               "failure cases, not hidden ones.")
    fl = df[df["flagged"] == "yes"].sort_values("confidence")
    st.dataframe(fl, hide_index=True, width="stretch", height=340)

    st.subheader("Distinct uncertain patterns")
    shown = set()
    for _, r in fl.iterrows():
        key = r["message (masked)"][:40]
        if key in shown:
            continue
        shown.add(key)
        with st.expander(f"{r['message_id']} — {r['category']} "
                         f"({r['confidence']:.2f})"):
            st.write(r["message (masked)"])
            st.caption(r["reason"])
            c = res.classifications[r["message_id"]]
            st.json(c.to_dict()["evidence"])

# -------------------------------------------------------------- live demo
with tabs[6]:
    st.subheader("Classify a message live")
    st.caption("Runs the full pipeline on one message: detect → mask → "
               "classify → extract.")
    examples = {
        "Task with a deadline": "Please submit the weekly report by 2026-09-20.",
        "Meeting invitation": "Please join the AI workshop on 2026-09-22, 14:00 at Zoom.",
        "Vague timing": "Let us catch up sometime next week.",
        "Promotional": "Flash sale on headphones tonight. Use code SAVE40.",
        "Personal preference": "For my profile, i prefer morning meetings.",
        "Sensitive (fictional)": "Your OTP is 447-219. It expires in 10 minutes.",
    }
    choice = st.selectbox("Start from an example", list(examples))
    text = st.text_area("Message", examples[choice], height=90)
    ts = st.text_input("Timestamp (used only to resolve words like 'tomorrow')",
                       "2026-09-15 09:00:00")
    sender = st.text_input("Sender", "Ananya")

    if st.button("Run", type="primary") and text.strip():
        msg = Message("LIVE_001", parse_timestamp(ts), sender, text.strip(), ts, 0)
        scan = scan_message(msg.message_id, strip_lead_in(msg.text))
        cls = clf.classify(msg, scan)

        if scan.is_sensitive:
            st.error(f"Sensitive content detected — risk **{scan.risk}**")
            st.code(scan.masked_text, language="text")
            st.markdown(f"Type `{scan.sensitivity_type}` · action "
                        f"`{scan.recommended_action}`")
        else:
            st.success("No sensitive content detected.")

        x, y, z = st.columns(3)
        x.metric("Category", cls.category)
        y.metric("Confidence", f"{cls.confidence:.2f}")
        z.metric("Source", cls.decision_source)
        st.write(cls.reason)
        if cls.needs_review or cls.confidence < REVIEW_FLAG_THRESHOLD:
            st.warning("Flagged for human review — low confidence.")

        item = extract(msg, cls.category)
        if item:
            item.item_id = "LIVE_ITEM"
            st.write("**Extracted item**")
            st.json(item.to_dict())
        else:
            st.info("No task or event found in this message.")
        st.write("**Full classification record**")
        st.json(cls.to_dict())
