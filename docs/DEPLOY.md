# Deploying the demo

Cloud hosting is mandatory for this submission. Streamlit Community Cloud is
the fastest route — free, connects straight to GitHub, no Docker.

## The one real problem

`.gitignore` excludes `data/`, because the brief forbids publishing the
dataset. But the hosted app needs the data to run. Three ways out, in the order
I'd try them.

### Option A — private repo (recommended)

Streamlit Community Cloud can deploy from a **private** GitHub repo, and the
resulting app URL is still publicly reachable. So:

1. Create the repo as **private**.
2. In that private repo only, allow the data through — either delete the
   `data/*` lines from `.gitignore`, or force-add:
   ```bash
   git add -f data/messages.csv data/mandatory_demo_ids.csv
   ```
3. Deploy from it (steps below).
4. If you also want a public repo to share as your submission link, push the
   same code to a second public repo **without** the data.

This satisfies both constraints: the dataset is never public, and the demo runs.

### Option B — upload at runtime

Add a file-uploader fallback so the app asks for the CSV when `data/` is empty.
The repo stays public and clean, but a reviewer has to upload a file before the
demo does anything — awkward on a shared link.

### Option C — synthetic sample

Commit a small `data/sample_messages.csv` you write yourself, in the same
schema and style, and have the app fall back to it. Nothing from the supplied
dataset is published. The demo works for anyone, but shows your data, not
theirs.

**Option A is what I'd do.** Mention the choice in your video — reviewers are
watching for whether you noticed the conflict at all.

---

## Streamlit Community Cloud

1. Push the repo to GitHub (private, per Option A).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** → pick the repo → branch `main` → main file `app.py`.
4. Advanced settings → Python 3.11.
5. Deploy. First build takes 2–4 minutes while scikit-learn installs.
6. Copy the `*.streamlit.app` URL into the README and your reply email.

### If the build fails

- **`ModuleNotFoundError`** — `requirements.txt` must be at the repo root.
- **App boots but shows the "dataset not found" screen** — the data didn't get
  committed. Check with `git ls-files data/`.
- **Memory limit** — Community Cloud gives ~1 GB. The pipeline peaks well under
  that on 900 messages. If you scale the corpus up, drop the `char` half of the
  `FeatureUnion` in `classifier.py` first; it's the memory hog.
- **Cold-start timeout** — `bootstrap()` is wrapped in `@st.cache_resource`, so
  only the first load pays the ~20 s pipeline cost. **Open the app yourself
  right before recording** so the cache is warm.

---

## Hugging Face Spaces

1. New Space → SDK **Streamlit** → set it Private if you're committing data.
2. Push:
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space>
   git push hf main
   ```
3. Rename `app.py` to `app.py` at the root (already is) — Spaces looks there.

## Render / Railway

`requirements.txt` plus a start command is enough — no Dockerfile needed:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Set the health-check path to `/_stcore/health`.

If you'd rather containerise:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Before you send the submission

- [ ] Deployed URL opens in a private browsing window (catches auth mistakes)
- [ ] All seven tabs load without an error
- [ ] Sensitive tab shows masked values only
- [ ] Mandatory IDs tab confirms all 15 resolved
- [ ] "Try a message" runs end to end
- [ ] Public repo, if you made one, has no dataset: `git ls-files data/`
- [ ] README's demo URL placeholder is filled in
- [ ] `python -m pytest tests/ -q` passes on a clean clone
