"""EvidenceRank sandbox — run the ranker on a candidate sample in the browser.

Accepts JSONL (one JSON object per line) OR a JSON array. Streams + heaps so memory stays
bounded (~top-K) even for large uploads. The ranker (../execution/rank.py) is pure stdlib;
Streamlit + the CSS below are only the demo UI (design inspired by modern recruitment
dashboards — Urbanist type, lavender/plum palette, KPI + candidate cards).

    streamlit run sandbox/app.py
"""
import heapq
import html
import json
import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "execution"))
import rank  # noqa: E402  (path set above)

st.set_page_config(page_title="EvidenceRank", page_icon="🎯", layout="wide")

# ---- design system (Urbanist + the reference palette) ----------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Urbanist',-apple-system,sans-serif; }
.stApp { background:#F4F3F8; }
#MainMenu, header[data-testid="stHeader"], footer { visibility:hidden; }
.block-container { padding-top:1.4rem; padding-bottom:2rem; max-width:1180px; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #ECEAF4; }

.er-hero { background:linear-gradient(120deg,#352B38 0%,#5A4A78 100%); border-radius:22px;
  padding:22px 26px; color:#fff; margin-bottom:18px; box-shadow:0 10px 30px rgba(53,43,56,.18); }
.er-hero h1 { font-size:28px; font-weight:800; margin:0; letter-spacing:-.5px; }
.er-hero p { margin:4px 0 0; color:#DAD8F9; font-weight:500; font-size:14px; }
.er-pill { display:inline-block; background:rgba(255,255,255,.16); color:#fff; font-weight:600;
  font-size:12px; padding:4px 12px; border-radius:99px; margin-top:10px; }

.kpis { display:flex; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
.kpi { flex:1; min-width:170px; border-radius:20px; padding:18px 20px; box-shadow:0 4px 16px rgba(53,43,56,.05); }
.kpi .lbl { color:#352B38; font-weight:600; font-size:13px; opacity:.7; }
.kpi .num { font-size:34px; font-weight:800; color:#352B38; line-height:1.1; margin-top:6px; }
.kpi .sub { color:#7E808C; font-weight:600; font-size:12px; margin-top:2px; }
.kpi.lav { background:#DAD8F9; } .kpi.blu { background:#D6E6FB; }
.kpi.pnk { background:#F7DCEC; } .kpi.grn { background:#D7F0E2; }

.cand { background:#fff; border-radius:18px; padding:15px 18px; margin-bottom:12px;
  box-shadow:0 4px 18px rgba(53,43,56,.06); display:flex; align-items:center; gap:16px; }
.rk { width:30px; height:30px; border-radius:10px; background:#352B38; color:#fff; font-weight:800;
  font-size:13px; display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
.av { width:46px; height:46px; border-radius:14px; background:#DAD8F9; color:#4B3B7A; font-weight:800;
  font-size:17px; display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
.mid { flex:1 1 auto; min-width:0; }
.mid .nm { font-weight:700; font-size:15px; color:#352B38; }
.mid .tt { color:#7E808C; font-weight:600; font-size:13px; margin-bottom:7px; }
.tags { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
.tag { background:#F4F3F8; color:#352B38; border-radius:99px; padding:3px 10px; font-size:11.5px; font-weight:600; }
.tag.ok { background:#D7F0E2; color:#1B7A48; } .tag.warn { background:#FBE6CF; color:#9A5B11; }
.rs { color:#7E808C; font-size:12.5px; font-weight:500; line-height:1.45; }
.scorebox { flex:0 0 auto; width:120px; text-align:right; }
.scorebox .pct { font-size:22px; font-weight:800; }
.scorebox .cap { font-size:11px; color:#7E808C; font-weight:600; }
.bar { height:7px; border-radius:99px; background:#EEECF6; margin-top:6px; overflow:hidden; }
.bar > i { display:block; height:100%; border-radius:99px; }
.sec { color:#352B38; font-weight:800; font-size:17px; margin:6px 0 12px; }
.stDownloadButton button { border-radius:12px; font-weight:700; background:#6C5CE7; color:#fff; border:none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x if x is not None else ""))


def _text(chunk):
    return chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk


def iter_candidates(fileobj):
    fileobj.seek(0)
    head = _text(fileobj.read(64)).lstrip()
    fileobj.seek(0)
    if head.startswith("["):
        data = json.loads(_text(fileobj.read()))
        if not isinstance(data, list):
            raise ValueError("JSON file must be an array of candidates or one object per line.")
        yield from data
    else:
        for ln, raw in enumerate(fileobj, 1):
            line = _text(raw).strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {ln} isn't valid JSON ({e}). Expected JSONL or a JSON array.")


def rank_stream(cand_iter, topk):
    heap, cap, n, seq, strong = [], max(topk * 3, topk), 0, 0, 0
    for c in cand_iter:
        n += 1
        seq += 1
        final, comp, honey = rank.score_candidate(c)
        if final >= 0.70:
            strong += 1
        item = (round(final, 4), c.get("candidate_id", f"row{n}"), seq, c, comp, honey)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    ranked = sorted(heap, key=lambda x: (-x[0], x[1]))[:topk]
    return n, ranked, strong


def score_color(final):
    if final >= 0.85:
        return "#16A34A"
    if final >= 0.70:
        return "#6C5CE7"
    if final >= 0.50:
        return "#D97706"
    return "#7E808C"


def card_html(i, c, final, reasoning):
    prof = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}
    name = (prof.get("anonymized_name") or "").strip()
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "•"
    title = prof.get("current_title", "—")
    cid = c.get("candidate_id", "")
    yoe = prof.get("years_of_experience", 0) or 0
    loc = prof.get("location", "") or "—"
    skills = rank.matched_core_skills(c, limit=3)
    pct = round(min(final, 1.0) * 100)
    col = score_color(final)

    tags = [f'<span class="tag">📍 {esc(loc)}</span>', f'<span class="tag">{yoe:.1f} yrs</span>']
    for s in skills:
        tags.append(f'<span class="tag">{esc(s)}</span>')
    rr = sig.get("recruiter_response_rate")
    if sig.get("open_to_work_flag") and (rr is None or rr >= 0.5):
        tags.append('<span class="tag ok">● open · responsive</span>')
    elif rr is not None and rr < 0.3:
        tags.append('<span class="tag warn">● low response</span>')

    return f"""<div class="cand">
      <div class="rk">{i}</div>
      <div class="av">{esc(initials)}</div>
      <div class="mid">
        <div class="nm">{esc(name or title)}</div>
        <div class="tt">{esc(title)} · {esc(cid)}</div>
        <div class="tags">{''.join(tags)}</div>
        <div class="rs">{esc(reasoning)}</div>
      </div>
      <div class="scorebox">
        <div class="pct" style="color:{col}">{pct}%</div>
        <div class="cap">match</div>
        <div class="bar"><i style="width:{pct}%;background:{col}"></i></div>
      </div>
    </div>"""


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 EvidenceRank")
    st.caption("Senior AI Engineer · Redrob")
    topk = st.slider("Top-K to show", 5, 100, 25)
    min_pct = st.slider("Min match %", 0, 100, 0, help="Hide candidates below this match score")
    uploaded = st.file_uploader("Candidates (.jsonl / .json)", type=["jsonl", "json", "txt"])
    st.caption("JSONL or a JSON array. Up to 500 MB. The bundled 50-candidate sample loads by default.")

# ---- hero ------------------------------------------------------------------
st.markdown(
    '<div class="er-hero"><h1>Candidate shortlist</h1>'
    '<p>Ranked by grounded evidence of fit — not keyword overlap.</p>'
    '<span class="er-pill">CPU-only · stdlib · explainable</span></div>',
    unsafe_allow_html=True)

SAMPLE = os.path.join(HERE, "sample_candidates.jsonl")
fileobj = uploaded if uploaded is not None else (open(SAMPLE, "rb") if os.path.exists(SAMPLE) else None)
if fileobj is None:
    st.warning("Upload a candidates file to begin.")
    st.stop()

try:
    n, ranked, strong = rank_stream(iter_candidates(fileobj), topk)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't process the file: {e}")
    st.stop()

if n == 0:
    st.warning("No candidates found in the file.")
    st.stop()

# apply the min-match filter to the shown shortlist
shown = [(i + 1, c, final) for i, (final, cid, _s, c, comp, honey) in enumerate(ranked)
         if min(final, 1.0) * 100 >= min_pct]
reasonings = {}
for _, c, _f in shown:
    f2, comp, honey = rank.score_candidate(c)
    reasonings[c.get("candidate_id", id(c))] = rank.build_reasoning(
        c, rank.evaluate_requirements(c, comp, honey), f2, honey)

avg_match = round(sum(min(f, 1.0) * 100 for _, _, f in shown) / len(shown)) if shown else 0

# ---- KPI row ---------------------------------------------------------------
st.markdown(
    f'<div class="kpis">'
    f'<div class="kpi lav"><div class="lbl">Candidates scored</div><div class="num">{n:,}</div><div class="sub">in this run</div></div>'
    f'<div class="kpi grn"><div class="lbl">Strong matches</div><div class="num">{strong:,}</div><div class="sub">≥ 70% fit</div></div>'
    f'<div class="kpi blu"><div class="lbl">Showing</div><div class="num">{len(shown)}</div><div class="sub">top shortlist</div></div>'
    f'<div class="kpi pnk"><div class="lbl">Avg match</div><div class="num">{avg_match}%</div><div class="sub">of shown</div></div>'
    f'</div>', unsafe_allow_html=True)

# ---- candidate cards -------------------------------------------------------
st.markdown('<div class="sec">Ranked candidates</div>', unsafe_allow_html=True)
if not shown:
    st.info("No candidates above the selected match threshold — lower the *Min match %* filter.")
else:
    cards = "".join(card_html(i, c, final, reasonings[c.get("candidate_id", id(c))])
                    for i, c, final in shown)
    st.markdown(cards, unsafe_allow_html=True)

# ---- export ----------------------------------------------------------------
csv_lines = ["candidate_id,rank,score,reasoning"]
for i, c, final in shown:
    reason = reasonings[c.get("candidate_id", id(c))].replace('"', '""')
    csv_lines.append(f'{c.get("candidate_id","")},{i},{final},"{reason}"')
st.download_button("⬇  Download ranking CSV", "\n".join(csv_lines) + "\n",
                   file_name="submission_sample.csv", mime="text/csv")
