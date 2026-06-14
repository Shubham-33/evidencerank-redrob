"""EvidenceRank sandbox — run the ranker on a small candidate sample in the browser.

Deploy free on Streamlit Community Cloud / HuggingFace Spaces (Streamlit SDK). The ranker
itself (../execution/rank.py) is stdlib-only; Streamlit is only the demo UI and is NOT part
of the ranking reproduction path.

    streamlit run sandbox/app.py
"""
import io
import json
import os
import sys

import streamlit as st

# Make the stdlib ranker importable regardless of where Streamlit is launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "execution"))
import rank  # noqa: E402  (path set above)

st.set_page_config(page_title="EvidenceRank — Redrob", layout="wide")
st.title("EvidenceRank — candidate ranker")
st.caption(
    "Stdlib-only, CPU-only, no network. Upload a small JSONL sample (≤100 candidates, one "
    "JSON object per line, matching candidate_schema.json) or use the bundled sample."
)

topk = st.sidebar.slider("Top-K to return", 5, 100, 25)
uploaded = st.file_uploader("candidates.jsonl (≤100 lines)", type=["jsonl", "json", "txt"])

SAMPLE = os.path.join(HERE, "sample_candidates.jsonl")


def load_candidates(stream):
    cands = []
    for line in stream:
        line = line.strip()
        if line:
            cands.append(json.loads(line))
    return cands


if uploaded is not None:
    text = io.StringIO(uploaded.getvalue().decode("utf-8"))
    candidates = load_candidates(text)
elif os.path.exists(SAMPLE):
    with open(SAMPLE, encoding="utf-8") as f:
        candidates = load_candidates(f)
    st.info(f"No upload — using bundled sample ({len(candidates)} candidates).")
else:
    st.warning("Upload a JSONL sample to begin.")
    candidates = []

if candidates:
    if len(candidates) > 100:
        st.warning(f"{len(candidates)} candidates provided; the sandbox checks small samples "
                   "(≤100). Ranking all of them anyway.")
    scored = []
    for c in candidates:
        final, comp, honey = rank.score_candidate(c)
        scored.append((round(final, 4), c, comp, honey))
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

    rows = []
    for i, (final, c, comp, honey) in enumerate(scored[:topk], 1):
        triples = rank.evaluate_requirements(c, comp, honey)
        rows.append({
            "rank": i,
            "candidate_id": c["candidate_id"],
            "score": final,
            "title": c["profile"].get("current_title", ""),
            "reasoning": rank.build_reasoning(c, triples, final, honey),
        })
    st.subheader(f"Top {len(rows)} of {len(candidates)}")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    csv_lines = ["candidate_id,rank,score,reasoning"]
    for r in rows:
        reason = r["reasoning"].replace('"', '""')
        csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{reason}"')
    st.download_button("Download ranking CSV", "\n".join(csv_lines) + "\n",
                       file_name="submission_sample.csv", mime="text/csv")
