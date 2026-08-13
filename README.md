# Message Triage

Classifies incoming messages, pulls tasks and events out of them, and finds and
masks sensitive values before anything else touches the text. Runs entirely
offline — the pipeline makes no network calls and no message ever leaves the
process.

Built for the KaStack AI/ML intern assignment (900-message corpus).

```
900 messages -> 6 categories -> 410 tasks/events -> 110 sensitive messages masked
```

---

## Quick start

```bash
git clone <this-repo>
cd msgiq
pip install -r requirements.txt

# Put messages.csv and mandatory_demo_ids.csv in data/ (see data/README.md)
python run_pipeline.py --data data --out outputs

# Interactive demo
streamlit run app.py

# Tests, including the leak audit
python -m pytest tests/ -v
```

Live demo: **https://aditya-msgiq.streamlit.app**

---

## What it produces

| File | Contents |
|---|---|
| `outputs/classifications.json` | Per message: category, confidence, reason, evidence |
| `outputs/tasks_events.json` | Extracted tasks and events with provenance for every field |
| `outputs/sensitive.json` | Detections with type, risk, **masked** text, recommended action |
| `outputs/combined.json` | All three joined per message, text always masked |
| `outputs/mandatory_demo_report.json` | The 15 IDs required in the video |
| `outputs/run_summary.json` | Counts, model metrics, confusion matrix |

---

## Results on the supplied dataset

```
Messages                900   (all timestamps parsed, already chronological, no duplicate IDs)
Categories              action_required 230 · meeting_or_event 170 · general_information 170
                        sensitive_information 110 · personal_information 110 · promotional 110
Mean confidence         0.94
Flagged for review       40  (4.4%)
Rule coverage           97.8%   (rules fire on 880 of 900)
Rule/model agreement    97.7%
Tasks + events          410  (240 tasks, 170 events)
Sensitive messages      110  across 11 sensitivity types
```

---

## Part 1 — Classification

Six categories, as specified: `action_required`, `meeting_or_event`,
`personal_information`, `general_information`, `promotional`,
`sensitive_information`. I did not add or merge any.

### Why hybrid rather than pure ML or pure rules

The dataset ships with no labels, so there is nothing to train on directly.
A pure rule list would have been enough to score well here — the corpus is
heavily templated — but it gives you no probability, so a "confidence score"
would be a number I made up. A pure model needs labels that do not exist.

So: **rules generate weak labels, and a model learns from them.**

```
message
  │
  ├─ sensitive detector ──► fires? ──► sensitive_information   (override, never reversed)
  │
  ├─ rule layer ──► category + rule_confidence
  │
  └─ TF-IDF + logistic regression ──► category + probability
                                        │
                            ┌───────────┴───────────┐
                            │  combine and report   │
                            └───────────────────────┘
```

### The rule layer

`src/msgiq/rules.py` holds 20 regex rules, each carrying a category, a weight,
and the sentence shown to the user as the reason. Two details that matter:

- **Strongest rule per category wins, rules are not summed.** Otherwise a
  category could win by matching five weak patterns instead of one good one.
- **Ties break by precedence**, with `sensitive_information` first
  (`config.CATEGORY_PRECEDENCE`). A safety label should never lose a coin flip.
- When the runner-up category scores close behind, **confidence is reduced by
  the size of the gap**. Genuine ambiguity should show up in the number.

### The model

`src/msgiq/classifier.py`. TF-IDF into logistic regression, trained locally at
runtime in about two seconds:

- **Word 1–2 grams** catch phrasing ("don't forget", "is scheduled for").
- **Character 3–5 grams** (`char_wb`) survive typos and unseen nouns.
- `class_weight="balanced"` so the smaller categories are not drowned out.

Two preprocessing steps do most of the work:

1. **Lead-in stripping** (`loader.strip_lead_in`). Every message carries filler
   like `"Can you help?"` or `"FYI:"` glued on the front, and that filler is
   distributed across *all* categories. Left in, the model learns
   `"Can you help?"` as an action-required cue when it is really noise.
2. **Literal normalisation** (`classifier.normalise`). `2026-09-14` becomes
   `<date>`, `10:30` becomes `<time>`, digits become `<num>`. Without this the
   model memorises specific dates as features and learns nothing about the
   *shape* of a deadline.

Only examples the rules are confident about (≥ 0.75) become training data, so
the model is not taught our own guesses.

### Combining the two

Five outcomes, each recorded in `decision_source` so you can audit any message:

| `decision_source` | When | Confidence | Count |
|---|---|---|---|
| `sensitive_detector_override` | A secret was found | Detector precision | 100 |
| `rule+model_agree` | Both agree | Weighted blend + 0.05 bonus | 760 |
| `rule_override_model` | Strong rule (≥0.80) vs. model | Rule conf − 0.15 | 0 |
| `weak_rule_over_model` | Both weak, rule less unsure | Rule conf × 0.8, flagged | 20 |
| `model_only` | No rule fired | Model prob × 0.9 | 10 |
| `sensitive_reference_low_confidence` | Credentials mentioned, none present | 0.55, flagged | 10 |

Disagreement is treated as information, not as a problem to hide. When the two
layers disagree the confidence drops and the message is flagged.

### About that 1.000 CV accuracy

`run_summary.json` reports 5-fold cross-validated accuracy of 1.000. **This is
not evidence the system is perfect, and I would not present it as such.** The
labels being predicted are the rule layer's own output, so the score measures
one thing only: whether the model can reproduce the rule logic from wording
alone. It can, easily, because the corpus is built from roughly 215 template
families. On free-form human text it would not.

The number I would actually defend is **97.7% rule/model agreement across all
900 messages**, including the 20 the rules never saw during training. And the
only true accuracy figure would come from hand-labelling a sample — which I did
not do, and say so below under Limitations.

---

## Part 2 — Task and event extraction

`src/msgiq/extraction.py`. 410 items from 900 messages: 240 tasks, 170 events.

The rule from the brief — *do not guess missing information* — is the design
constraint the whole module is built around. Every field is either taken from
the message or is `null` with an entry in `unresolved_fields` saying what was
missing and why.

### Date handling: three tiers

Recorded per item in `date_provenance`:

| Tier | Meaning | Example | Count |
|---|---|---|---|
| `explicit` | ISO date in the text, copied verbatim | `"...by 2026-09-05"` → `2026-09-05` | 350 |
| `derived_from_timestamp` | One unambiguous meaning given the message's own timestamp | `"tomorrow"` sent 2026-09-05 → `2026-09-06` | 10 |
| `unresolved` | Vague; left `null` | `"sometime next week"`, `"Friday afternoon"` | 50 |

The middle tier is the one worth defending. Resolving `"tomorrow"` against the
message's own timestamp is arithmetic, not inference — there is exactly one
answer and it is recorded, not assumed. `"next week"` has seven answers, so it
stays `null`. **A plausible date is still a fabricated date.**

### Other fields

- **Person** — only names actually written in the message body count. The
  sender is stored separately in `sender` and is deliberately *not* copied into
  `person`; assuming the sender is the person involved would be a guess. This
  is why most items have `person: null`, which is the correct answer.
- **Priority** — `high` for a firm dated deadline or urgency wording,
  `medium` for a scheduled date with no urgency, `low` for hedged language
  ("may", "could", "if possible"). The reason is written into `notes`.
- **Confidence** — scales with how many fields resolved, minus a penalty for
  hedged wording.

### Worked example of missing information

```
"Let us meet sometime next week."
```
```json
{
  "type": "event", "title": "Let us meet",
  "date": null, "time": null, "person": null, "location": null,
  "priority": "low", "date_provenance": "unresolved",
  "unresolved_fields": ["date", "time", "person", "location"],
  "notes": ["Timing phrase 'next week' is not specific enough to resolve to a
             calendar date; left null rather than guessed.", ...]
}
```

All 410 items carry at least one unresolved field. That is not a failure — it
reflects that these messages genuinely do not state a time or a person, and the
system declines to invent one.

---

## Part 3 — Sensitive information

`src/msgiq/sensitive.py`. 110 messages flagged across 11 types.

### Contextual first, format second

Most detectors match the *phrase around* the secret
(`"account recovery code is <X>"`) rather than the shape of the secret itself.
That way the detector still works when the value format changes — a recovery
code does not have a fixed shape, but the sentence introducing one does.
Format-only detectors (16-digit card numbers, `tok_*` strings, email addresses)
act as a safety net and are suppressed when a contextual detector already
covered the same span.

### Types, risk, and action

| Type | Risk | Recommended action |
|---|---|---|
| `password`, `one_time_password`, `auth_token`, `account_recovery_code` | critical | `do_not_store` |
| `payment_card`, `bank_account` | critical | `do_not_store` |
| `personal_identifier`, `health_information` | high | `do_not_send_to_external_service` |
| `private_address`, `contact_number`, `email_address` | medium | `ask_for_confirmation` |
| `credential_reference` | low | `safe_to_process_locally` |

Everything critical also carries `do_not_send_to_external_service` in
`additional_actions`. Risk drives the action; the mapping lives in one dict
(`config.SENSITIVITY_POLICY`) rather than being scattered through the code.

Health data is treated as `high` rather than `medium` because a medical finding
is not recoverable once disclosed — unlike a password, you cannot rotate it.

### How masking works

1. Every detector captures the secret in a named group called `value`. Only
   that span is redacted, so the message stays readable and a human can still
   see what it was about.
2. Masks are applied **back to front** so earlier spans keep their offsets.
3. **The mask is a fixed six characters regardless of the true length.** A mask
   that mirrors length tells an attacker how long the secret is, which narrows
   a brute-force search. `test_mask_width_does_not_reveal_secret_length` checks
   this.
4. Masking happens **at detection time**, before classification or extraction.
   Nothing downstream is ever handed the raw value.

```
"Your OTP is 483-921. It expires in 10 minutes."
        ->  "Your OTP is ******. It expires in 10 minutes."

"My card number is 4111 1111 1111 1111-92."
        ->  "My card number is ******."
```

### The interesting edge case

```
"I will send the login details separately."
```

Credentials are referenced but no secret is present. Masking would do nothing.
Ignoring it loses a real signal — a secret is about to arrive. So it is filed
as `credential_reference`, risk `low`, action `safe_to_process_locally`,
classified as `sensitive_information` at **0.55 confidence and flagged for
review**. I think a reasonable person could file this under general information
instead, and the low confidence says exactly that.

### Not leaking, verified

`test_no_secret_value_appears_in_any_generated_file` extracts every secret the
detector found in the source CSV and asserts that none of those literal strings
appears in any file under `outputs/`. It runs on every test invocation.

The Streamlit app reads only `ScanResult.masked_text` — there is no code path
from the UI to the raw message text. This is deliberate: it means a screen
recording of the app cannot leak a value even by accident.

---

## Layout

```
msgiq/
├── run_pipeline.py           CLI
├── app.py                    Streamlit demo
├── src/msgiq/
│   ├── config.py             categories, risk policy, thresholds
│   ├── loader.py             CSV loading, chronological validation
│   ├── sensitive.py          Part 3 — detection + masking
│   ├── rules.py              Part 1 — rule layer
│   ├── classifier.py         Part 1 — hybrid ML
│   ├── extraction.py         Part 2
│   └── pipeline.py           orchestration + export
├── tests/test_pipeline.py    50 tests
├── data/                     gitignored (dataset not published)
└── outputs/                  generated
```

Order of operations in `pipeline.run()` is fixed and matters:
**load → scan → train → classify → extract.** The sensitive scan runs second,
before anything else sees the text, so no later stage handles a message without
knowing it contains a secret.

---

## Assumptions

1. Messages are English and single-topic. One message produces at most one task
   or event.
2. `message_id` is unique — checked at load, reported if not.
3. A message classified `sensitive_information` is never turned into a calendar
   item, even if it contains a date. Not worth the risk for a diary entry.
4. Names are matched against the roster of senders in the corpus. An unfamiliar
   name in message text will not be picked up as `person` — a false negative,
   which I prefer to a false positive here.
5. All timestamps are one timezone; none is stated in the data.
6. The four action strings from the brief are used verbatim; stricter secondary
   actions go in `additional_actions` rather than inventing a fifth string.

## Limitations

1. **No gold labels.** Every accuracy number is measured against rule-derived
   weak labels, not human annotation. The system's real precision on free-form
   text is unmeasured. Hand-labelling ~150 messages would fix this and is the
   first thing I would do with more time.
2. **Tuned on a templated corpus.** 900 messages from ~215 templates. Rules
   that look precise here are partly fitted to those templates; on real inbox
   text coverage would drop and the model would carry more of the load.
3. **English-only, regex-based detection.** Transliterated or obfuscated
   secrets ("my pin is one two three four") are missed. So are secrets with no
   introducing phrase and no recognisable format.
4. **No relative-weekday resolution.** "Friday afternoon" is left null even
   though the message timestamp plus a calendar could resolve it. That was a
   deliberate call — "which Friday" is ambiguous — but it costs recall.
5. **No cross-message context.** "Could you send it soon?" has no referent, and
   the system does not look at the previous message to find one. Threading
   would help.
6. **Confidence is heuristic, not calibrated.** The blend weights in
   `config.py` are hand-set. Proper calibration needs a labelled holdout set.
7. **`char_wb` features are memory-hungry** and would need trimming (or
   hashing) well before this scaled past a few hundred thousand messages.

## What I would do next

- Hand-label a stratified sample of 150 messages to get a real accuracy number
  and a real confusion matrix.
- Swap TF-IDF for local sentence embeddings (`all-MiniLM-L6-v2`, runs offline)
  and compare — the rule layer stays as the safety net either way.
- Add a `luhn` check for card numbers to cut format-detector false positives.
- Thread messages by sender and time window so pronoun references resolve.
- Active learning: the 40 flagged messages are exactly the ones worth
  hand-labelling first.

---

## AI-tool usage declaration

As required by the brief.

**Tools used:** Claude (Anthropic) as a coding assistant, and standard editor
autocomplete.

**What it was used for:**
- Drafting and refactoring boilerplate — dataclass definitions, the export
  layer, Streamlit layout code.
- Rubber-ducking the regex patterns in `sensitive.py` and `rules.py`, which I
  then tested against the corpus and corrected.
- Wording and structuring this README.

**What was decided by me:**
- The hybrid rules-plus-weak-supervision architecture, and the reasoning for it.
- The category precedence order and the sensitive-override rule.
- The three-tier date provenance policy and the decision not to resolve
  "next week".
- The risk-to-action mapping, including treating health data as `high`.
- The fixed-width mask decision, after realising a length-proportional mask
  leaks the secret's length.

**Not used:** no message from the dataset was sent to ChatGPT, Claude, or any
other hosted service. There is no API call anywhere in this repository — the
model is trained locally from scratch on each run. The dataset is excluded from
version control by `.gitignore`.

I wrote the tests myself and can explain every file in this repository.

## Notes

- Dataset not committed, per the brief. See `data/README.md`.
- All sensitive values in the corpus are fictional but are masked regardless.
- Reproducible: `RANDOM_SEED = 42`, and message/item IDs are assigned in
  chronological order, so two runs produce byte-identical output.
