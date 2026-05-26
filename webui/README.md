# mmcomposer web UI

A Streamlit-based front-end for `mmcomposer`.  Lets the user configure a
matmul kernel via hyperparameter dropdowns, see the resulting JSON-intent
file, and view what the agent loop's output looks like (step-by-step
optimization progress + final kernel + perf report).

> **Status (pitch demo):** the front-end is functional, the agent loop is
> not yet implemented.  The "Generate kernel" button runs a mocked walk
> using the real performance numbers from the b1 → b41_w8 development
> journey on B200 BF16.  This is intended to demonstrate the *flow* of
> the tool — what the user-facing experience will look like once the
> agent backend lands.

## Run locally

From the repo root:

```bash
pip install -r webui/requirements.txt
streamlit run webui/app.py
```

The app opens at `http://localhost:8501`.  Hot-reloads on file save.

## Deploy to Streamlit Community Cloud (free public URL)

1. Push the repo to GitHub (already done — `tongzhou8086/mmcomposer`).
2. Sign in to <https://streamlit.io/cloud> with the same GitHub account.
3. Click **New app**.
4. Pick the repo / branch / main file: `webui/app.py`.
5. Streamlit Cloud will install `webui/requirements.txt` and deploy.
   Subsequent `git push`es auto-redeploy, same as Read the Docs.

The app URL will look like `https://<slug>.streamlit.app`.

## What's mocked

The "Generate kernel" button shows the agent walking through the
optimization ladder, but doesn't actually invoke an LLM or compile any
code yet.  The displayed perf numbers and kernel snippet are real
artifacts from our hand-written `b41_w8` kernel — they represent what
the agent will produce once the backend is wired up.

Real items to wire up later:
- LLM API integration (user-supplied API key).
- Tutorial-document loader (reads `docs/` and feeds it to the agent as
  prompt context).
- Compile + correctness + bench loop per optimization step.
- Real kernel preview + downloadable artifact.
