# EvidenceRank Sandbox

A small Streamlit app that runs the ranker on a candidate sample (≤100) and returns a ranked
CSV — the mandatory "working hosted environment" from submission_spec.md §10.5.

The ranker itself ([`../execution/rank.py`](../execution/rank.py)) is **pure Python standard
library**. Streamlit is only the demo UI and is **not** part of the ranking reproduction path.

## Run locally

```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

Upload a `candidates.jsonl` (≤100 lines, one JSON object per line matching
`candidate_schema.json`), or use the bundled `sample_candidates.jsonl` (50 candidates). The app
shows the ranked table with reasoning and a CSV download.

## Deploy (free) — pick one

**Streamlit Community Cloud** (simplest):
1. Push this repo to GitHub.
2. At share.streamlit.io, create an app pointing at `sandbox/app.py`.
3. Set the requirements file to `sandbox/requirements.txt`. Done — paste the URL into
   `submission_metadata.yaml` → `sandbox_link`.

**HuggingFace Spaces** (Streamlit SDK):
1. Create a Space, SDK = Streamlit.
2. Add `app.py` (this file's sibling) and `requirements.txt`; include the `execution/` folder
   so `rank.py` is importable (or copy `rank.py` next to `app.py` and adjust the import).
3. Paste the Space URL into `submission_metadata.yaml`.

**Colab fallback:** open a notebook that `pip install streamlit`, clones the repo, and runs
the ranker on a sample — link the notebook. A `docker run` recipe in the root README is also
acceptable per §10.5.
