# Loom script — 9 minutes

Every item on the brief's demo checklist is covered below, with the timestamp
where it happens. Nothing is skipped; if you run short, cut from section 8, not
from 5 or 6.

**Before you hit record**

- Run `python run_pipeline.py --data data --out outputs` once so the app's cache
  is warm — otherwise the first page load stalls for ~20 seconds on camera.
- Start `streamlit run app.py`, open it, and click through every tab once.
- Browser zoom to **125–140%**. Small text is the most common reason a demo gets
  marked unreadable.
- Close Slack, mail, notifications. Mute anything that pings.
- Have the deployed Streamlit Cloud URL open in a second tab.
- Editor open on `src/msgiq/sensitive.py`, scrolled to `scan_message`.

**Say the numbers out loud.** 900 messages, 410 items, 110 sensitive, 97.7%
agreement, 40 flagged. Graders listening at 1.5× catch numbers, not adjectives.

---

## 1 · Opening — 0:00–0:40

> "Hi, I'm Aditya. This is my submission for the AI/ML intern assignment — a
> message triage system that classifies 900 messages into six categories, pulls
> tasks and events out of them, and detects and masks sensitive values.
>
> One thing up front: this runs entirely offline. There's no API call anywhere
> in the repo. No message from the dataset was ever sent to an external service.
> The classifier is trained locally, from scratch, every run — takes about two
> seconds."

*On screen: the Overview tab.*

---

## 2 · Approach and system flow — 0:40–1:40
**Checklist: brief overview of approach and system flow**

Point at the pipeline box on the Overview tab.

> "Five stages, and the order matters. Load and sort chronologically. Then the
> sensitive scan — that runs **second**, before anything else touches the text,
> so no later stage ever handles a message without knowing it holds a secret.
> Then train, classify, extract.
>
> The classifier is hybrid, and here's why. The dataset ships with no labels,
> so there's nothing to train on directly. A pure rule list would score well —
> this corpus is templated — but it gives no probability, so any confidence
> score would be a number I invented. A pure model needs labels that don't
> exist.
>
> So the rules generate weak labels, and a TF-IDF logistic regression learns
> from them. Where the two disagree, that's genuine ambiguity, and I report it
> instead of hiding it."

Point to the metrics row.

> "Rules fire on 97.8% of messages. Rule and model agree on 97.7%."

**Pre-empt the CV number before they ask:**

> "You'll see cross-validated accuracy of 1.000 in the summary. I want to be
> straight about that — it does **not** mean the system is perfect. The labels
> being predicted are the rule layer's own output, so it only shows the model
> can reproduce the rule logic from wording. The corpus is about 215 templates,
> so that's easy. The number I'd actually defend is the 97.7% agreement. A real
> accuracy figure needs hand-labelled data, which I flag in the README as the
> first thing I'd do next."

---

## 3 · Dataset structure — 1:40–2:10
**Checklist: dataset structure without exposing sensitive-looking values**

Scroll to the dataset table on Overview, then open the Classification tab.

> "Four columns — message ID, timestamp, sender, message. 900 rows, already
> chronological, all timestamps parsed, no duplicate IDs.
>
> Notice the message column here is labelled 'masked'. The UI has no code path
> to the raw text at all — every display reads the masked copy from the
> detector. That's deliberate: it means this recording can't leak a value even
> if I click the wrong thing."

---

## 4 · All six categories — 2:10–2:50
**Checklist: classification results from all six categories**

On the Classification tab, filter to one category at a time, or show the
distribution chart on Overview.

> "All six categories, all populated. Action required 230, meeting or event
> 170, general information 170, sensitive 110, personal 110, promotional 110.
> That's 900."

Filter to `personal_information`, then `promotional`.

> "The distinction I want to point out is personal versus sensitive. 'I'm
> vegetarian' is a personal preference — that's `personal_information`. 'My card
> number is…' is a secret — that's `sensitive_information`, and it gets masked."

---

## 5 · The 15 mandatory IDs — 2:50–4:00
**Checklist: results for all 15 mandatory message IDs**

Open the **Mandatory IDs** tab. Let the full table sit on screen for a beat.

> "These are the 15 IDs supplied with the dataset. All 15 resolved — the app
> confirms that below the table. Reading down:"

| ID | Say |
|---|---|
| MSG_0002, 0007 | "Two action-required, both with dated deadlines, 0.95 and 0.99." |
| MSG_0001, 0003 | "Two events — a calendar entry and a reminder, both with date, time, and location." |
| MSG_0009, 0016 | "Two personal — emergency contact, coffee preference." |
| MSG_0004, 0006 | "Two general information — no action, no date." |
| MSG_0014, 0015 | "Two promotional — both carry a discount code." |
| MSG_0005, 0013 | "Two sensitive — address and card number. Both masked." |
| MSG_0012 | "Credential reference — I'll come back to this one." |
| MSG_0024 | "A hedged preference — 'I might prefer evening meetings'." |
| MSG_0037 | "The uncertain one. Coming up in section 9." |

> "So the 15 cover all six categories, tasks, events, sensitive, and both of my
> edge cases — which is clearly how they were chosen."

Expand two or three rows to show the reason strings and JSON.

---

## 6 · Tasks and events — 4:00–5:15
**Checklist: 3+ tasks, 3+ events, one with missing information**

Open **Tasks & Events**. Filter to `task`.

> "410 items — 240 tasks, 170 events. Three tasks:"

- `Review the privacy checklist` — deadline 2026-09-09, priority high
- `Submit the weekly report` — deadline 2026-09-05, priority high
- `Pay the electricity bill` — deadline 2026-09-09, priority high

> "High priority because each has a firm, dated deadline. Note `person` is null
> on all three, and that's correct — nobody is named in the message text. The
> sender is stored separately in its own field. Copying the sender into `person`
> would be a guess, and the brief says don't guess."

Filter to `event`.

> "Three events:"

- `Internship orientation` — 2026-09-18, 13:00, Conference Room 2
- `Mentor catch-up` — 2026-09-16, 11:00, city clinic
- `Product demo` — 2026-09-07, 10:00, Zoom

> "Date, time, and location all pulled from the text."

**Missing information — tick the "unresolved fields" checkbox.**

> "Here's the one I want to dwell on."

Open `Let us meet sometime next week`.

> "Date null, time null, person null, location null — and every one listed under
> `unresolved_fields` with a note saying why. 'Next week' has seven possible
> answers, so it stays null.
>
> But look at `date_provenance`, because there are three tiers." *(scroll the
> provenance column)* "`explicit` — 350 items, an ISO date copied straight from
> the text. `unresolved` — 50, like this one. And `derived_from_timestamp` — 10.
>
> That middle tier is a judgement call I'll defend. When a message says
> 'tomorrow', I resolve it against that message's **own** timestamp. That's
> arithmetic — there's exactly one answer — and I label it so you can see it was
> derived, not stated. 'Next week' I refuse, because a plausible date is still a
> fabricated date."

---

## 7 · Sensitive detection — 5:15–6:45
**Checklist: detection, masking, risk level, recommended action**

Open **Sensitive**.

> "110 messages, 11 sensitivity types. By risk: 60 critical, 20 high, 20 medium,
> 10 low. Every action here is one of the four strings from the brief."

Scroll the table.

> "Passwords, OTPs, tokens, recovery codes, card and bank numbers — all critical,
> all `do_not_store`. ID numbers and health data — high,
> `do_not_send_to_external_service`. Addresses and phone numbers — medium,
> `ask_for_confirmation`.
>
> Health data sits at high rather than medium on purpose. A password you can
> rotate. A medical finding you can't un-disclose."

Expand two or three of the per-type examples.

> "Every value replaced with exactly six asterisks — and the width is fixed
> regardless of the real length. That's not cosmetic. A mask that mirrors the
> length tells you how long the secret is, which narrows a brute-force search.
> There's a test asserting a four-character OTP and a thirteen-character OTP
> produce identical masks."

**MSG_0012 — the edge case.**

> "'I will send the login details separately.' Credentials referenced, but
> there's no actual secret in it — masking would do nothing. Ignoring it loses a
> real signal, though: a secret is about to arrive.
>
> So it's `credential_reference`, risk low, action `safe_to_process_locally`,
> classified sensitive at 0.55 and flagged for review. A reasonable person could
> file this as general information instead — and the 0.55 is the system saying
> exactly that."

---

## 8 · Code walkthrough — 6:45–7:45
**Checklist: one important code section explained in your own words**

Switch to the editor, `src/msgiq/sensitive.py`, `scan_message`.

> "This is the function I'd point at if you only read one. Three ideas.
>
> **First** — every detector captures the secret in a named group called
> `value`, and the masker only ever redacts that span. So the sentence stays
> readable and a human can still judge what the message was about, but the
> secret is gone.
>
> **Second** — detectors are contextual before they're format-based. They match
> the phrase *around* the secret — 'account recovery code is X' — not the shape
> of X. A recovery code has no fixed shape, but the sentence introducing one
> does. Card-number and token *format* detectors exist as a safety net, and get
> suppressed when a contextual detector already covered the span."

Scroll to the overlap-resolution loop.

> "**Third** — this loop. Hits are sorted contextual-first, most-specific-first,
> and any hit overlapping one already kept is dropped. Without it, 'my card
> number is 4111 1111 1111 1111' matches both the contextual rule and the
> 16-digit format rule, and you'd mask the same span twice and corrupt the
> offsets."

Point at `require_digit`.

> "This flag is a bug I actually hit. My identifier detector matched 'student'
> in 'you may like our new student plan' and redacted the word 'plan' as if it
> were an ID number. My leak test caught it. Two fixes: the qualifier now has to
> be followed by an actual noun like 'number' or 'card', and identifier-type
> values must contain a digit. Ten false positives gone — and it's why that
> category shows 110 promotional now instead of 100."

---

## 9 · An uncertain result — 7:45–8:20
**Checklist: one incorrect or uncertain result and why**

Open the **Uncertain** tab.

> "40 messages — 4.4% — the system flagged itself. Four distinct patterns."

Open `The review could be Friday afternoon` (MSG_0037).

> "Classified `meeting_or_event` at 0.53. Here's what happened: a weak rule said
> meeting, at 0.68, because of 'review… could be'. The model said action
> required, because 'review' is overwhelmingly a task verb in this corpus —
> 'review the checklist', 'review the model results'. Both signals were weak, the
> rule was marginally less unsure, so the rule won and the message got flagged.
>
> I think the rule is right here. But the honest answer is that 'the review' as
> a *noun* is rare in this data, so the model never learned it. That's a real
> limitation, not a fluke."

Also point at `Maya asked whether the demo was ready` — 0.40, `general_information`, no rule fired.

> "This one is reported speech. It's arguably action-required, arguably general.
> No rule fired, so the model decided alone and the confidence got discounted to
> 0.40. I'd rather it say 0.40 and flag itself than say 0.95 and be wrong."

---

## 10 · Live run — 8:20–8:45

Open **Try a message**. Pick the sensitive example, click Run.

> "The full pipeline on one message — detect, mask, classify, extract. OTP
> caught, masked, critical, do-not-store."

Now type a fresh task with a deadline, click Run.

> "And a task it's never seen: category, confidence, the reason, and the
> extracted item with the deadline resolved and person left null."

---

## 11 · Limitations and improvements — 8:45–9:00
**Checklist: limitations and possible improvements**

> "Limitations, honestly. No gold labels — every accuracy number is against
> weak labels, not human annotation. The corpus is 215 templates, so the rules
> are partly fitted to it and would degrade on real inbox text. Detection is
> English regex, so 'my pin is one two three four' gets missed. And there's no
> cross-message context — 'could you send it soon' has no referent and I don't
> look at the previous message to find one.
>
> Next steps: hand-label 150 messages for a real accuracy number — and the 40
> flagged ones are exactly the ones worth labelling first. Swap TF-IDF for local
> sentence embeddings and compare. Add a Luhn check on card numbers. Thread
> messages so pronouns resolve.
>
> Everything's in the repo — the dataset isn't, per the brief. 50 tests,
> including one that takes every secret in the source data and asserts none of
> them appears anywhere in the generated outputs. Thanks for watching."

---

## Final checklist

| Required | Section | Time |
|---|---|---|
| Approach and system flow | 2 | 0:40 |
| Dataset structure, no sensitive values | 3 | 1:40 |
| All six categories | 4 | 2:10 |
| All 15 mandatory IDs | 5 | 2:50 |
| 3+ extracted tasks | 6 | 4:00 |
| 3+ extracted meetings/events | 6 | 4:35 |
| Missing/unclear information example | 6 | 4:50 |
| Sensitive detection, masking, risk, action | 7 | 5:15 |
| 3+ classification decisions explained | 4, 5, 9 | throughout |
| One incorrect/uncertain result and why | 9 | 7:45 |
| One code section in your own words | 8 | 6:45 |
| Limitations and improvements | 11 | 8:45 |
| System actually running (not slides) | all | — |

**Do not** open `data/messages.csv` in a spreadsheet on camera. Every sensitive
value in the corpus is visible there in the clear. Show the dataset structure
through the app, which masks it, or through the schema table in
`data/README.md`.
