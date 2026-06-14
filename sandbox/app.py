"""EvidenceRank sandbox — run the ranker on a candidate sample in the browser.

Accepts either JSONL (one JSON object per line) OR a JSON array (e.g. sample_candidates.json).
Streams + heaps so memory stays bounded (~top-K) even for large uploads. The ranker itself
(../execution/rank.py) is pure Python standard library; Streamlit is only the demo UI.

    streamlit run sandbox/app.py
"""
import heapq
import json
import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "execution"))
import rank  # noqa: E402  (path set above)

st.set_page_config(page_title="EvidenceRank — Redrob", layout="wide")
st.title("EvidenceRank — candidate ranker")
st.caption(
    "Stdlib-only, CPU-only, no network. Upload candidates as **JSONL** (one JSON object per "
    "line) **or a JSON array** (e.g. sample_candidates.json), matching candidate_schema.json — "
    "or use the bundled sample. The spec only needs small samples (≤100); larger files work "
    "but may be slow / memory-heavy on the free tier."
)

topk = st.sidebar.slider("Top-K to return", 5, 100, 25)
uploaded = st.file_uploader("candidates file (.jsonl or .json)", type=["jsonl", "json", "txt"])
SAMPLE = os.path.join(HERE, "sample_candidates.jsonl")


def _text(chunk):
    return chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk


def iter_candidates(fileobj):
    """Yield candidate dicts from a JSON array or JSONL, robustly and memory-friendly."""
    fileobj.seek(0)
    head = _text(fileobj.read(64)).lstrip()
    fileobj.seek(0)
    if head.startswith("["):                      # whole-file JSON array
        data = json.loads(_text(fileobj.read()))
        if not isinstance(data, list):
            raise ValueError("JSON file must be an array of candidates or one object per line.")
        yield from data
    else:                                          # JSONL — one object per line
        for ln, raw in enumerate(fileobj, 1):
            line = _text(raw).strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Line {ln} isn't valid JSON ({e}). Expected JSONL (one JSON object per "
                    "line) or a JSON array. If your file is a pretty-printed JSON array, that's "
                    "supported too — make sure it starts with '['."
                )


def rank_stream(cand_iter, topk):
    """Stream candidates through the ranker, keeping only a bounded top-K heap."""
    heap = []
    cap = max(topk * 3, topk)
    n = seq = 0
    for c in cand_iter:
        n += 1
        seq += 1
        final, comp, honey = rank.score_candidate(c)
        # seq (unique int) before the dict guarantees tuples never compare dicts.
        item = (round(final, 4), c.get("candidate_id", f"row{n}"), seq, c, comp, honey)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    ranked = sorted(heap, key=lambda x: (-x[0], x[1]))[:topk]
    return n, ranked


fileobj = uploaded if uploaded is not None else (
    open(SAMPLE, "rb") if os.path.exists(SAMPLE) else None)

if fileobj is None:
    st.warning("Upload a candidates file to begin.")
    st.stop()
if uploaded is None:
    st.info("No upload — using the bundled 50-candidate sample.")

try:
    n, ranked = rank_stream(iter_candidates(fileobj), topk)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001 — surface any parse/scoring issue to the user, not a traceback
    st.error(f"Couldn't process the file: {e}")
    st.stop()

if n == 0:
    st.warning("No candidates found in the file.")
    st.stop()

rows = []
for i, (final, cid, _seq, c, comp, honey) in enumerate(ranked, 1):
    triples = rank.evaluate_requirements(c, comp, honey)
    rows.append({
        "rank": i,
        "candidate_id": cid,
        "score": final,
        "title": c.get("profile", {}).get("current_title", ""),
        "reasoning": rank.build_reasoning(c, triples, final, honey),
    })

st.subheader(f"Top {len(rows)} of {n} candidates")
st.dataframe(rows, use_container_width=True, hide_index=True)

csv_lines = ["candidate_id,rank,score,reasoning"]
for r in rows:
    reason = r["reasoning"].replace('"', '""')
    csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{reason}"')
st.download_button("Download ranking CSV", "\n".join(csv_lines) + "\n",
                   file_name="submission_sample.csv", mime="text/csv")
