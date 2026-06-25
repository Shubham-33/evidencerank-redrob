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
sys.path.insert(0, HERE)            # rank.py & parse_jd.py are siblings in this Space
import rank  # noqa: E402  (path set above)

st.set_page_config(page_title="EvidenceRank", page_icon="🎯", layout="wide",
                   initial_sidebar_state="collapsed")

# ---- design system (Urbanist + the reference palette) ----------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Urbanist',-apple-system,sans-serif; }
.stApp { background:#F4F3F8; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1180px; }

/* control bar */
.ctrl-h { color:#352B38; font-weight:800; font-size:15px; margin:2px 0 8px; }
[data-testid="stFileUploader"] section { border-radius:14px; border:1.5px dashed #C9C6E6; background:#FCFCFE; }

.er-hero { background:linear-gradient(120deg,#352B38 0%,#5A4A78 100%); border-radius:22px;
  padding:22px 26px; color:#fff; margin-bottom:18px; box-shadow:0 10px 30px rgba(53,43,56,.18); }
.er-hero h1 { font-size:36px; font-weight:800; margin:0; letter-spacing:-.5px; }
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


# ---- hero ------------------------------------------------------------------
st.markdown(
    '<div class="er-hero"><h1>🎯 EvidenceRank</h1>'
    '<p>Candidate shortlist for the Senior AI Engineer role — ranked by grounded evidence '
    'of fit, not keyword overlap.</p>'
    '<span class="er-pill">CPU-only · stdlib · explainable</span></div>',
    unsafe_allow_html=True)

# ---- controls (in the main area so they're always visible) -----------------
st.markdown('<div class="ctrl-h">⚙️ Upload &amp; options</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Candidates file (.jsonl or .json) — JSONL or a JSON array, up to 500 MB. "
    "Leave empty to use the bundled 50-candidate sample.",
    type=["jsonl", "json", "txt"])
col_a, col_b = st.columns(2)
topk = col_a.slider("Top-K to show", 5, 100, 25)
min_pct = col_b.slider("Min match %", 0, 100, 0, help="Hide candidates below this match score")

# ---- Job Description: feed any JD; NVIDIA parses it; the engine retargets ----
import parse_jd  # noqa: E402  (execution/ is on sys.path)
JD_FILE = os.path.join(HERE, "job_description.txt")
default_jd = open(JD_FILE, encoding="utf-8").read() if os.path.exists(JD_FILE) else ""


def _stored_key():
    """Read a key from Streamlit secrets / env, without crashing when none is configured."""
    try:
        k = st.secrets.get("NVIDIA_API_KEY")  # raises if no secrets.toml exists
        if k:
            return k
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("NVIDIA_API_KEY", "")


with st.expander("📋  Job Description — feed a JD to retarget the ranker (optional)"):
    jd_text = st.text_area("Paste / edit the job description", value=default_jd, height=180)
    key_input = st.text_input(
        "NVIDIA API key (optional — paste 'nvapi-…' to parse the JD with the LLM)",
        type="password", placeholder="nvapi-…",
        help="Used only this session to parse the JD — not stored, not committed. "
             "Free key at build.nvidia.com. Leave blank to use deterministic parsing.")
    nvidia_key = key_input.strip() or _stored_key()
    st.caption("🟢 LLM parsing enabled (key provided)." if nvidia_key
               else "⚪ No key — JDs parse with the deterministic stdlib fallback (still works).")
    cjd1, cjd2 = st.columns(2)
    if cjd1.button("🧠  Parse JD & retarget", use_container_width=True):
        with st.spinner("Parsing the JD…"):
            st.session_state.jd_cfg = parse_jd.parse_jd(jd_text, api_key=nvidia_key or None)
        st.toast(f"JD parsed ({st.session_state.jd_cfg.get('_source')})")
    if cjd2.button("↩  Reset to built-in JD", use_container_width=True):
        st.session_state.pop("jd_cfg", None)
    cfg = st.session_state.get("jd_cfg")
    if cfg:
        st.success(f"Ranking against parsed JD · source: `{cfg.get('_source')}`")
        st.markdown(
            f"**Role:** {cfg.get('role','—')} &nbsp;·&nbsp; **Experience:** "
            f"{cfg.get('experience_min')}–{cfg.get('experience_max')} yrs &nbsp;·&nbsp; "
            f"**Locations:** {', '.join(cfg.get('preferred_locations') or ['any'])}")
        st.markdown("**Core skills extracted:** " + ", ".join(cfg.get("core_skills") or []))

# Retarget the engine from the parsed JD (or restore built-in defaults) every run.
if st.session_state.get("jd_cfg"):
    rank.apply_jd_config(st.session_state.jd_cfg)
else:
    rank.reset_jd_config()

# Submit button — ranks the current source. The button click records WHICH source was
# ranked, so a freshly uploaded file shows a "click to rank" prompt until you press it,
# while the bundled sample demos immediately on first load.
src_name = uploaded.name if uploaded is not None else "bundled sample"
if st.button("🚀  Rank candidates", type="primary", use_container_width=True):
    st.session_state.ranked_src = src_name
    st.toast(f"Ranked: {src_name}")
if st.session_state.get("ranked_src") is None and uploaded is None:
    st.session_state.ranked_src = "bundled sample"   # auto-demo on first visit
if st.session_state.get("ranked_src") != src_name:
    st.info(f"📄 **{src_name}** is ready — click **🚀 Rank candidates** to rank it.")
    st.stop()

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

# ---- export CSV (built first so the download sits ABOVE the list) ----------
csv_lines = ["candidate_id,rank,score,reasoning"]
for i, c, final in shown:
    reason = reasonings[c.get("candidate_id", id(c))].replace('"', '""')
    csv_lines.append(f'{c.get("candidate_id","")},{i},{final},"{reason}"')

# ---- header row: section title (left) + download button (right) ------------
head_l, head_r = st.columns([3, 1])
head_l.markdown('<div class="sec">Ranked candidates</div>', unsafe_allow_html=True)
head_r.download_button("⬇  Download CSV", "\n".join(csv_lines) + "\n",
                       file_name="submission_sample.csv", mime="text/csv",
                       use_container_width=True, disabled=not shown)

# ---- candidate cards -------------------------------------------------------
if not shown:
    st.info("No candidates above the selected match threshold — lower the *Min match %* filter.")
else:
    cards = "".join(card_html(i, c, final, reasonings[c.get("candidate_id", id(c))])
                    for i, c, final in shown)
    st.markdown(cards, unsafe_allow_html=True)
