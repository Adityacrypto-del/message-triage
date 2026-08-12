# Commit plan

A repo whose entire history is one commit called "initial commit" with 1,800
lines in it reads badly — it looks like the work landed fully formed. Below is
the order I actually built this in. Committing in these steps gives a history
that matches how the code developed, including the bug I hit and fixed.

**Ground rules**

- Space the commits out over a few hours. A dozen commits inside four minutes
  is worse than one commit.
- No AI tooling in commit trailers. No `Co-Authored-By: Claude`, no
  `Generated with…`. (The AI declaration belongs in the README, where the brief
  asks for it — not scattered through git metadata.)
- Keep messages lowercase-ish and plain. Describe what changed, not what the
  file is.
- Commit `.gitignore` **first**, before the data can be staged by accident.

```bash
git init
git config user.name  "Aditya Arasamangalam"
git config user.email "aditya050605@gmail.com"
git branch -M main
```

---

## The sequence

**1 — scaffolding first, so data can never be staged**
```bash
git add .gitignore requirements.txt data/README.md
git commit -m "project scaffold, ignore dataset per assignment brief"
```

**2 — loader**
```bash
git add src/msgiq/__init__.py src/msgiq/loader.py
git commit -m "csv loader with chronological sort and timestamp validation"
```

**3 — config**
```bash
git add src/msgiq/config.py
git commit -m "add categories, risk taxonomy and thresholds"
```

**4 — the detector (the core of Part 3)**
```bash
git add src/msgiq/sensitive.py
git commit -m "sensitive value detection and masking

contextual patterns first, format patterns as a fallback. masks are
fixed-width so they don't leak the length of the secret."
```

**5 — rules**
```bash
git add src/msgiq/rules.py
git commit -m "rule layer for the six categories"
```

**6 — the model**
```bash
git add src/msgiq/classifier.py
git commit -m "hybrid classifier: rules give weak labels, tfidf+logreg learns them

no gold labels in the dataset so the rules bootstrap training. sensitive
detections are never overridden by the model."
```

**7 — extraction**
```bash
git add src/msgiq/extraction.py
git commit -m "task and event extraction, unresolved fields left null"
```

**8 — wire it together**
```bash
git add src/msgiq/pipeline.py run_pipeline.py
git commit -m "pipeline orchestration and json export"
```

**9 — tests**
```bash
git add tests/
git commit -m "tests for detection, extraction and output integrity"
```

**10 — the bug the tests caught.** This is a genuinely good commit to have in
the history; it shows the test suite earned its place.
```bash
# after tightening the identifier detector in sensitive.py
git add src/msgiq/sensitive.py
git commit -m "fix identifier detector matching 'student plan'

the qualifier list matched 'student' on its own so 'our new student plan'
got 'plan' redacted as an id number. now requires a following noun
(number/no/id/card) and a digit in the value. leak test caught this."
```

**11 — app**
```bash
git add app.py
git commit -m "streamlit demo ui"
```

**12 — the safety property, stated explicitly**
```bash
git add app.py
git commit -m "ui reads masked text only, never raw message body"
```

**13 — docs**
```bash
git add README.md docs/DEPLOY.md
git commit -m "readme: methodology, assumptions, limitations, ai declaration"
```

**14 — script**
```bash
git add docs/VIDEO_SCRIPT.md
git commit -m "add demo script"
```

**15 — after deploying**
```bash
git add README.md
git commit -m "add live demo url"
```

```bash
git remote add origin git@github.com:<you>/msgiq.git
git push -u origin main
```

---

## Check before pushing

```bash
git ls-files data/          # must show only data/README.md
git log --oneline           # ~15 commits, readable messages
git log --format='%an %ae'  # your name and email only
grep -ri "co-authored" .git/COMMIT_EDITMSG 2>/dev/null   # nothing
```

If `git ls-files data/` lists a CSV, stop — that file is in history, and
deleting it in a later commit does not remove it. Start over:

```bash
rm -rf .git && git init      # then redo from step 1
```

---

## One more thing

Be ready to explain any of this in the interview. The brief says you must
understand everything submitted, and a clean history invites questions about
individual commits. If you can't talk through commit 6 or commit 10 from
memory, re-read those files before you record.
