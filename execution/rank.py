#!/usr/bin/env python3
"""
Redrob hackathon ranker — Senior AI Engineer JD.

Explainable, rule-based, stdlib-only (no numpy/sklearn) so it reproduces inside
any CPU-only sandbox. Single streaming pass over candidates.jsonl, top-K heap,
deterministic tie-breaks.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

final = base x trap_penalties x behavioral_modifier ; honeypots forced near-zero.
See directives/rank_candidates.md for the design rationale.
"""

import argparse
import csv
import heapq
import json
from collections import namedtuple
from datetime import date

# Reference "today" — career end_date is null for current roles; used for recency
# and honeypot duration checks. Matches the dataset's simulated present.
TODAY = date(2026, 6, 14)

# --- term banks ------------------------------------------------------------

# Core skills the JD says you "absolutely need" — retrieval/ranking/eval/IR.
CORE_TERMS = {
    "embedding", "embeddings", "sentence-transformer", "sentence transformers",
    "sbert", "bge", " e5 ", "dense retrieval", "retrieval", "semantic search",
    "neural search", "hybrid search", "hybrid retrieval", "vector search",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss", "opensearch", "elasticsearch",
    "ranking", "re-rank", "rerank", "learning to rank", "learning-to-rank",
    "ltr", "recommendation", "recommender", "recsys", "search relevance",
    "ndcg", "mrr", " map ", "information retrieval",
    "nlp", "natural language processing",
}
# Nice-to-have / supporting signal.
SUPPORT_TERMS = {
    "llm", "large language model", "fine-tune", "fine-tuning", "finetune",
    "lora", "qlora", "peft", "rag", "retrieval augmented", "retrieval-augmented",
    "xgboost", "gradient boosting", "transformer", "bert", "pytorch", "tensorflow",
    "a/b test", "ab test", "offline evaluation", "evaluation framework",
    "python", "distributed", "inference optimization", "open source", "open-source",
}
# Title tokens that mark a real engineering/ML/data builder.
ENG_TITLE_TOKENS = {
    "engineer", "developer", "scientist", "sde", "ml", "machine learning",
    "ai ", "nlp", "research engineer", "applied scientist", "architect",
    "data ", "search", "ranking", "backend", "software", "programmer",
}
# Title tokens that, absent any engineering token, mark a keyword-stuffer trap.
NONTECH_TITLE_TOKENS = {
    "marketing", "content writer", "copywriter", "hr ", "human resresource",
    "human resource", "recruiter", "talent acquisition", "sales", "account manager",
    "business development", "customer success", "operations manager", "finance",
    "accountant", "designer", "ux ", "ui ", "teacher", "professor", "lecturer",
    "consultant", "analyst relations", "project manager", "program manager",
}
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "ltimindtree", "mindtree", "mphasis",
    "deloitte", "ibm global services", "dxc",
}
INDIA_PREFERRED = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "bengaluru", "bangalore", "ncr", "navi mumbai",
}
RESEARCH_MARKERS = {"phd", "postdoc", "research scientist", "research fellow",
                    "research assistant", "academic", "university", "institute of"}
PRODUCTION_MARKERS = {"production", "deployed", "real users", "scale", "shipped",
                      "launched", "users", "latency", "throughput", "serving"}
CV_SPEECH_TOKENS = {"computer vision", "image", "object detection", "speech",
                    "asr", "robotics", "lidar", "segmentation"}
# Distinctive phrasings that describe retrieval/ranking/recsys work WITHOUT the jargon —
# to catch "plain-language Tier 5s". Chosen to be specific enough to avoid the dataset's
# backend/devops/marketing boilerplate (validated to fire rarely on non-core descriptions).
PLAIN_LANGUAGE_PHRASES = {
    "which products to show", "which items to show", "what to show", "what users see",
    "what to recommend", "in what order", "order results", "scoring that ranks",
    "ranks candidate", "ranked by relevance", "rank items", "surface relevant",
    "surface the most relevant", "most relevant ones surface", "predicted interest",
    "predicted relevance", "personaliz", "candidate items", "candidate generation",
    "suggest relevant", "recommend products", "recommend items", "relevance ranking",
    "which results", "order them",
}


def has_any(text, terms):
    return any(t in text for t in terms)


def count_terms(text, terms):
    return sum(1 for t in terms if t in text)


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def parse_date(s):
    if not s:
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def months_between(a, b):
    if not a or not b:
        return 0
    return (b.year - a.year) * 12 + (b.month - a.month)


# --- honeypot detection ----------------------------------------------------

def honeypot_flags(c):
    """Return list of hard internal inconsistencies. Any flag => force out of top 100."""
    flags = []
    sig = c.get("redrob_signals", {})
    prof = c.get("profile", {})
    hist = c.get("career_history", []) or []

    # 1. A job whose stated duration exceeds its actual start->end window.
    for job in hist:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or TODAY
        if start:
            window = months_between(start, end)
            dur = job.get("duration_months", 0) or 0
            if dur - window > 6:  # >6 months of impossible tenure
                flags.append("job_duration_exceeds_window")
                break

    # 2. Stated years of experience far exceeds earliest career start.
    starts = [parse_date(j.get("start_date")) for j in hist]
    starts = [s for s in starts if s]
    if starts:
        earliest = min(starts)
        max_possible_years = months_between(earliest, TODAY) / 12.0
        yoe = prof.get("years_of_experience", 0) or 0
        if yoe - max_possible_years > 2.0:
            flags.append("yoe_exceeds_career_span")

    # 3. Many "expert/advanced" skills with 0 months of actual use.
    zero_dur_expert = sum(
        1 for s in (c.get("skills") or [])
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months", 0) or 0) == 0
    )
    if zero_dur_expert >= 5:
        flags.append("expert_skills_zero_duration")

    # NOTE: an inverted expected_salary range (min > max) was tried as a 4th check
    # but fires on ~19% of the pool — it's synthetic noise here, not an impossibility
    # signal — so it is deliberately excluded.

    return flags


# --- scoring components (each returns 0..1 unless noted) --------------------

# Optional JD-driven relevant-title keywords (set via apply_jd_config). None => built-in.
_JD_TITLES = None


def score_title(prof, all_text):
    title = (prof.get("current_title") or "").lower()
    if _JD_TITLES is not None:
        # JD-driven role match: how well the candidate's title matches the JD's role terms.
        hits = sum(1 for t in _JD_TITLES if t in title)
        if hits >= 2:
            return 1.0
        if hits == 1:
            return 0.65
        return 0.15   # title doesn't match the JD's role at all
    is_eng = has_any(title, ENG_TITLE_TOKENS)
    is_nontech = has_any(title, NONTECH_TITLE_TOKENS)
    if is_eng and not is_nontech:
        # Bonus for explicitly ML/AI/search titles.
        if has_any(title, {"machine learning", "ml ", "ai ", "nlp", "scientist",
                           "search", "ranking", "applied scientist", "research engineer"}):
            return 1.0
        return 0.7  # generic software/data engineer — still a builder
    if is_eng and is_nontech:
        return 0.5  # e.g. "Engineering Manager" — ambiguous
    return 0.12  # non-technical title => likely keyword-stuffer trap


def score_skills(c):
    skills = c.get("skills") or []
    sig = c.get("redrob_signals", {})
    assess = sig.get("skill_assessment_scores", {}) or {}
    total = 0.0
    for s in skills:
        name = (s.get("name") or "").lower()
        if not (any(t.strip() in name for t in CORE_TERMS) or
                any(t.strip() in name for t in SUPPORT_TERMS)):
            continue
        weight = 1.0 if any(t.strip() in name for t in CORE_TERMS) else 0.4
        # Trust multiplier: real use (duration) + endorsements + assessment score
        # discount claimed-but-unbacked skills (the stuffer signature).
        dur = s.get("duration_months", 0) or 0
        end = s.get("endorsements", 0) or 0
        trust = 0.25
        if dur >= 12:
            trust += 0.35
        elif dur >= 6:
            trust += 0.2
        if end >= 5:
            trust += 0.2
        a = assess.get(s.get("name"), None)
        if a is not None and a >= 70:
            trust += 0.2
        total += weight * clamp(trust)
    # ~4 strong core skills saturates.
    return clamp(total / 4.0)


def job_recency_years(job):
    """Years since a job ended; 0 for current/ongoing roles."""
    if job.get("is_current") or not job.get("end_date"):
        return 0.0
    end = parse_date(job.get("end_date"))
    if not end:
        return 0.0
    return max(0.0, (TODAY - end).days / 365.0)


def recency_factor(years):
    """Gentle, FLOORED temporal decay on career evidence.

    The JD rewards recent hands-on work ("production code in the last 18 months") but
    *also* values deep, pre-LLM retrieval/ranking experience ("understood retrieval before
    it became fashionable"). So this is a slow, foundational decay with a high floor — not
    an aggressive exp decay, which would wrongly bury exactly the veterans the JD wants.
    Skills themselves carry no dates in the data, so decay is applied to dated career
    evidence, not the skills array.
    """
    if years <= 2:
        return 1.0    # current / very recent → hot
    if years <= 4:
        return 0.92
    if years <= 7:
        return 0.82
    return 0.72       # old, but foundational IR depth still counts strongly


def score_career(hist, summary):
    """Evidence of building ranking/search/retrieval/recsys at product companies,
    recency-weighted so active builders edge out those who've drifted from the domain."""
    best = 0.0
    services_jobs = 0
    for job in hist:
        desc = (job.get("description") or "").lower()
        industry = (job.get("industry") or "").lower()
        core = count_terms(desc, CORE_TERMS)
        support = count_terms(desc, SUPPORT_TERMS)
        prod = 1 if has_any(desc, PRODUCTION_MARKERS) else 0
        # Plain-language evidence: describes ranking/recsys work without the keywords.
        # Weak and capped — a fallback that lifts genuine plain-language builders without
        # overwhelming keyword-backed evidence or rewarding boilerplate.
        plain = count_terms(desc, PLAIN_LANGUAGE_PHRASES)
        if "services" in industry or "consulting" in industry:
            services_jobs += 1
        job_score = clamp(0.30 * core + 0.10 * support + 0.12 * min(plain, 2) + 0.20 * prod)
        # Product (non-services) industry building real systems is the gold signal.
        if "services" not in industry and "consulting" not in industry and core:
            job_score = clamp(job_score + 0.2)
        job_score *= recency_factor(job_recency_years(job))
        best = max(best, job_score)
    # Summary is the candidate's present self-description (undated) → treat as current.
    s_core = count_terms(summary, CORE_TERMS)
    best = max(best, clamp(0.15 * s_core))
    return best


# Optional experience-band override from a parsed JD (set via apply_jd_config). None => the
# built-in Senior AI Engineer behaviour, so the official submission path is byte-identical.
_EXP_BAND = None


def score_experience(yoe):
    if _EXP_BAND is None:
        # Default: peak 6-8 yrs, full credit 5-9, soft decay outside.
        if 6 <= yoe <= 8:
            return 1.0
        if 5 <= yoe <= 9:
            return 0.85
        if 4 <= yoe < 5 or 9 < yoe <= 11:
            return 0.6
        if 3 <= yoe < 4 or 11 < yoe <= 13:
            return 0.35
        return 0.15
    # JD-driven band: full credit inside [lo,hi] (peak in the middle), soft decay outside.
    lo, hi = _EXP_BAND
    mid, half = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
    if lo <= yoe <= hi:
        return 0.85 + 0.15 * (1 - abs(yoe - mid) / half)
    dist = (lo - yoe) if yoe < lo else (yoe - hi)
    return 0.6 if dist <= 2 else 0.35 if dist <= 4 else 0.15


def score_location(prof, sig):
    loc = (prof.get("location") or "").lower()
    country = (prof.get("country") or "").lower()
    if has_any(loc, INDIA_PREFERRED):
        return 1.0
    if "india" in country:
        return 0.8 if sig.get("willing_to_relocate") else 0.6
    # Outside India — JD says case-by-case, no visa sponsorship.
    return 0.5 if sig.get("willing_to_relocate") else 0.2


# --- optional JD-config override (for arbitrary JDs; see parse_jd.py) --------
# The official submission NEVER calls apply_jd_config, so its scoring is unchanged. The
# sandbox/CLI call it to retarget the same engine at a different JD's requirements.
_JD_DEFAULTS = None


def apply_jd_config(cfg):
    """Retarget the JD-specific knobs (core/support skills, preferred locations, experience
    band) from a parsed-JD config produced by parse_jd.parse_jd()."""
    global CORE_TERMS, SUPPORT_TERMS, INDIA_PREFERRED, _EXP_BAND, _JD_TITLES, _JD_DEFAULTS
    if _JD_DEFAULTS is None:
        _JD_DEFAULTS = (set(CORE_TERMS), set(SUPPORT_TERMS), set(INDIA_PREFERRED))
    if cfg.get("core_skills"):
        CORE_TERMS = {s.lower() for s in cfg["core_skills"]}
    if cfg.get("support_skills"):
        SUPPORT_TERMS = {s.lower() for s in cfg["support_skills"]}
    if cfg.get("preferred_locations"):
        INDIA_PREFERRED = {s.lower() for s in cfg["preferred_locations"]}
    if cfg.get("relevant_titles"):
        _JD_TITLES = {s.lower() for s in cfg["relevant_titles"]}
    lo, hi = cfg.get("experience_min"), cfg.get("experience_max")
    if lo and hi and hi > lo:
        _EXP_BAND = (float(lo), float(hi))


def reset_jd_config():
    """Restore the built-in Senior AI Engineer defaults."""
    global CORE_TERMS, SUPPORT_TERMS, INDIA_PREFERRED, _EXP_BAND, _JD_TITLES
    if _JD_DEFAULTS is not None:
        CORE_TERMS, SUPPORT_TERMS, INDIA_PREFERRED = (set(x) for x in _JD_DEFAULTS)
    _EXP_BAND = None
    _JD_TITLES = None


def score_eval_signal(c, all_text):
    """Education tier + explicit evaluation-framework experience."""
    s = 0.0
    if has_any(all_text, {"ndcg", "mrr", " map ", "a/b test", "ab test",
                          "offline evaluation", "evaluation framework"}):
        s += 0.6
    edu = c.get("education") or []
    if any(e.get("tier") in ("tier_1", "tier_2") for e in edu):
        s += 0.4
    return clamp(s)


def _trap_factors(c, prof, hist, all_text):
    """The JD's explicit anti-patterns as (key, factor, evidence) triples. Single source
    of truth for both the numeric penalty and the reasoning text."""
    out = []
    title = (prof.get("current_title") or "").lower()

    # Keyword-stuffer: non-tech current title but AI skills listed. Only for the built-in
    # JD — when a custom JD is applied, JD-driven score_title already handles role fit.
    if _JD_TITLES is None and has_any(title, NONTECH_TITLE_TOKENS) and not has_any(title, ENG_TITLE_TOKENS):
        out.append(("keyword_stuffer", 0.15, "non-technical current title with AI skills listed"))

    # Consulting/services-only career.
    if hist:
        consult = sum(1 for j in hist
                      if any(f in (j.get("company") or "").lower() for f in CONSULTING_FIRMS))
        if consult == len(hist):
            out.append(("consulting_only", 0.55, "entire career at consulting/services firms"))

    # Research/academia-only with no production evidence.
    if has_any(all_text, RESEARCH_MARKERS) and not has_any(all_text, PRODUCTION_MARKERS):
        out.append(("research_only", 0.5, "research/academia background, no production evidence"))

    # CV/speech/robotics-only without NLP/IR exposure.
    if has_any(all_text, CV_SPEECH_TOKENS) and not has_any(all_text, {"nlp", "retrieval", "search", "ranking", "information retrieval"}):
        out.append(("cv_speech_only", 0.6, "CV/speech/robotics focus without NLP/IR"))

    # Title-chasing job-hopper: many short stints.
    completed = [j for j in hist if not j.get("is_current")]
    if len(completed) >= 3:
        avg = sum(j.get("duration_months", 0) or 0 for j in completed) / len(completed)
        if avg < 18:
            out.append(("job_hopper", 0.8, f"short average tenure (~{avg:.0f} mo across {len(completed)} roles)"))

    return out


def trap_penalties(c, prof, hist, all_text):
    """Multiplicative penalty for the JD's anti-patterns. Returns 0..1. (Kept as a thin
    float-returning wrapper so the eval harness can monkey-patch it cleanly.)"""
    p = 1.0
    for _key, factor, _ev in _trap_factors(c, prof, hist, all_text):
        p *= factor
    return p


def behavioral_modifier(sig):
    """Is the candidate actually hireable right now? Returns ~0.25..1.1.

    FLOORED design: full credit across the healthy range, steep only at the extremes the
    JD actually names as "not available" — "hasn't logged in for 6 months" (~180 days) and a
    "5% recruiter response rate". An earlier linear version over-discounted normal mid-range
    candidates (a 60-day-idle or 0.5-response-rate profile is fine, not unavailable), which
    compressed scores and demoted genuine top-fit people. The eval harness backed the change
    (net +0.024 composite over 12 seeds, asymmetric upside) while the planted `stale` case
    stays buried and top-100 hygiene is unchanged. See directives/evaluate_ranker.md.
    """
    m = 1.0
    last = parse_date(sig.get("last_active_date"))
    if last:
        days = (TODAY - last).days
        # Full credit up to ~6 months (the JD's own threshold), then decline.
        m *= 1.0 if days <= 180 else 0.85 if days <= 365 else 0.6 if days <= 540 else 0.4
    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        # Healthy responders (>=0.5) full credit; steep only toward the JD's 5% example.
        m *= min(1.0, 0.70 + 0.60 * clamp(rr))
    if sig.get("open_to_work_flag"):
        m *= 1.03
    ic = sig.get("interview_completion_rate")
    if ic is not None:
        m *= 1.0 if ic >= 0.5 else 0.85 + 0.30 * clamp(ic)
    pc = sig.get("profile_completeness_score")
    if pc is not None:
        m *= 1.0 if pc >= 60 else 0.90 + pc / 600.0
    return clamp(m, 0.25, 1.12)


# --- requirements model (the JD, made declarative) -------------------------
# Each candidate's fit is a set of (requirement, evidence, satisfaction) triples.
# CORE requirements carry weights that sum to 1.0 and form the base score. NEGATIVE
# requirements are multiplicative trap penalties; MODIFIER is the behavioral multiplier;
# GATE is the honeypot consistency check. This list is the single source of truth — the
# scoring weights and the requirements.json artifact both derive from it, and the reasoning
# text is assembled from the resulting triples (so every claim is grounded in a matcher).

REQUIREMENTS = [
    {"key": "title", "label": "Builder identity", "kind": "core", "weight": 0.22,
     "jd_quote": "This role writes code. ... Strong Python. Yes really, we care about code quality."},
    {"key": "skills", "label": "Retrieval & vector-search skills", "kind": "core", "weight": 0.20,
     "jd_quote": "Production experience with embeddings-based retrieval systems ... vector databases or hybrid search infrastructure."},
    {"key": "career", "label": "Shipped ranking/search systems", "kind": "core", "weight": 0.26,
     "jd_quote": "Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale."},
    {"key": "experience", "label": "Experience band 5-9 yrs", "kind": "core", "weight": 0.16,
     "jd_quote": "5-9 years ... roughly 6-8 years total experience."},
    {"key": "location", "label": "Location / relocation fit", "kind": "core", "weight": 0.10,
     "jd_quote": "Located in or willing to relocate to Noida or Pune."},
    {"key": "eval", "label": "Evaluation frameworks & pedigree", "kind": "core", "weight": 0.06,
     "jd_quote": "designing evaluation frameworks for ranking systems — NDCG, MRR, MAP."},
    {"key": "keyword_stuffer", "label": "Keyword-stuffer trap", "kind": "negative", "factor": 0.15,
     "jd_quote": "not 'find candidates whose skills section contains the most AI keywords'."},
    {"key": "consulting_only", "label": "Consulting-only career", "kind": "negative", "factor": 0.55,
     "jd_quote": "People who have only worked at consulting firms ... we're not a fit."},
    {"key": "research_only", "label": "Research/academia-only", "kind": "negative", "factor": 0.5,
     "jd_quote": "pure research environments ... without any production deployment — we will not move forward."},
    {"key": "cv_speech_only", "label": "CV/speech/robotics-only", "kind": "negative", "factor": 0.6,
     "jd_quote": "primary expertise is computer vision, speech, or robotics without ... NLP/IR exposure."},
    {"key": "job_hopper", "label": "Title-chasing job-hopper", "kind": "negative", "factor": 0.8,
     "jd_quote": "optimizing for ... titles by switching companies every 1.5 years ... not a fit."},
    {"key": "availability", "label": "Behavioral availability", "kind": "modifier",
     "jd_quote": "hasn't logged in for 6 months ... not actually available. Down-weight them appropriately."},
    {"key": "honeypot", "label": "Profile-consistency gate", "kind": "gate",
     "jd_quote": "honeypot candidates with subtly impossible profiles ... forced to relevance tier 0."},
]

# Base-score weights derive from the CORE requirements (single source of truth).
W = {r["key"]: r["weight"] for r in REQUIREMENTS if r["kind"] == "core"}
assert abs(sum(W.values()) - 1.0) < 1e-9, "core requirement weights must sum to 1.0"

Triple = namedtuple("Triple", "key label kind satisfaction score evidence")


def candidate_text(c):
    prof = c.get("profile", {})
    hist = c.get("career_history", []) or []
    return " ".join([
        (prof.get("summary") or "").lower(), (prof.get("headline") or "").lower(),
        (prof.get("current_title") or "").lower(),
        " ".join((j.get("description") or "").lower() for j in hist),
        " ".join((s.get("name") or "").lower() for s in (c.get("skills") or [])),
    ])


def score_candidate(c):
    """Hot path: numeric only (no evidence strings), so 100K candidates score fast.
    Returns (final, comp, honey). Triples/reasoning are built lazily for the top-K."""
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    hist = c.get("career_history", []) or []
    summary = (prof.get("summary") or "").lower()
    all_text = candidate_text(c)

    comp = {
        "title": score_title(prof, all_text),
        "skills": score_skills(c),
        "career": score_career(hist, summary),
        "experience": score_experience(prof.get("years_of_experience", 0) or 0),
        "location": score_location(prof, sig),
        "eval": score_eval_signal(c, all_text),
    }
    base = sum(W[k] * comp[k] for k in W)
    base *= trap_penalties(c, prof, hist, all_text)
    final = base * behavioral_modifier(sig)

    honey = honeypot_flags(c)
    if honey:
        final *= 0.001  # push honeypots out of contention

    return final, comp, honey


# --- reasoning: requirement -> evidence -> satisfaction triples -------------

def matched_core_skills(c, limit=3):
    out = []
    for s in c.get("skills") or []:
        name = s.get("name") or ""
        if any(t.strip() in name.lower() for t in CORE_TERMS):
            out.append(name)
        if len(out) >= limit:
            break
    return out


def _satisfaction(score):
    if score >= 0.7:
        return "strong"
    if score >= 0.4:
        return "partial"
    if score > 0.0:
        return "inferred"
    return "absent"


def _career_evidence(hist):
    """Name the strongest CORE-bearing role as grounded evidence (no hallucination)."""
    best, best_terms = None, []
    for j in hist:
        d = (j.get("description") or "").lower()
        terms = []
        for t in CORE_TERMS:
            ts = t.strip()
            if ts in d and ts not in terms:
                terms.append(ts)
        if len(terms) > len(best_terms):
            best, best_terms = j, terms
    if not best:
        return ""
    named = "/".join(best_terms[:2])
    return f"{named} work at {best.get('company', '')}".strip()


def _evidence_for(key, c, comp):
    prof = c.get("profile", {})
    if key == "title":
        return f"current title '{prof.get('current_title', '')}'"
    if key == "skills":
        sk = matched_core_skills(c)
        return ("core skills " + ", ".join(sk)) if sk else "no endorsement-backed core skills"
    if key == "career":
        return _career_evidence(c.get("career_history") or []) or "no clear shipped-system evidence"
    if key == "experience":
        return f"{prof.get('years_of_experience', 0) or 0:.1f} yrs (target 5-9)"
    if key == "location":
        return prof.get("location", "?") or "?"
    if key == "eval":
        txt = candidate_text(c)
        ev = [t.strip() for t in ("ndcg", "mrr", " map ", "a/b test", "evaluation framework") if t in txt]
        if any(e.get("tier") in ("tier_1", "tier_2") for e in (c.get("education") or [])):
            ev.append("top-tier school")
        return ("eval: " + ", ".join(ev[:3])) if ev else "no explicit eval-framework signal"
    return ""


def evaluate_requirements(c, comp=None, honey=None):
    """Build the explicit (requirement, evidence, satisfaction) triples for one candidate.
    Reuses the hot-path numbers in `comp`; only run for the final top-K."""
    if comp is None:
        _, comp, honey = score_candidate(c)
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    hist = c.get("career_history", []) or []
    all_text = candidate_text(c)

    triples = []
    for r in REQUIREMENTS:
        if r["kind"] != "core":
            continue
        sc = comp[r["key"]]
        triples.append(Triple(r["key"], r["label"], "core",
                              _satisfaction(sc), sc, _evidence_for(r["key"], c, comp)))
    for key, factor, ev in _trap_factors(c, prof, hist, all_text):
        triples.append(Triple(key, key, "negative", "fired", factor, ev))
    if honey:
        triples.append(Triple("honeypot", "Profile-consistency gate", "gate",
                              "failed", 0.001, "; ".join(honey)))
    return triples


def build_reasoning(c, triples, final, honey):
    """Assemble a grounded, varied 1-2 sentence justification from the triples."""
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    core = {t.key: t for t in triples if t.kind == "core"}
    title = prof.get("current_title", "professional")
    yoe = prof.get("years_of_experience", 0) or 0

    parts = [f"{title}, {yoe:.1f} yrs"]
    sk = core.get("skills")
    if sk and sk.satisfaction in ("strong", "partial"):
        parts.append(sk.evidence)
    car = core.get("career")
    if car and car.satisfaction in ("strong", "partial"):
        parts.append(car.evidence)
    ev = core.get("eval")
    if ev and ev.satisfaction == "strong":
        parts.append(ev.evidence)
    parts.append(prof.get("location", "?") or "?")
    lead = "; ".join(parts) + "."

    # Honest concerns, keyed to the actual weak/negative signals.
    concerns = []
    rr = sig.get("recruiter_response_rate")
    if rr is not None and rr < 0.4:
        concerns.append(f"low recruiter response rate ({rr:.2f})")
    last = parse_date(sig.get("last_active_date"))
    if last and (TODAY - last).days > 120:
        concerns.append(f"last active {(TODAY - last).days}d ago")
    np_days = sig.get("notice_period_days")
    if np_days is not None and np_days > 60:
        concerns.append(f"{np_days}d notice")
    loc_t = core.get("location")
    if loc_t and loc_t.score < 0.5:
        concerns.append("location/visa risk")
    for t in triples:
        if t.kind == "negative":
            concerns.append(t.evidence)
    if honey:
        concerns.append("profile-consistency flags")

    if concerns:
        lead += " Concern: " + ", ".join(concerns[:2]) + "."
    return lead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--dump-requirements", metavar="PATH",
                    help="write the declarative REQUIREMENTS schema to PATH and exit")
    args = ap.parse_args()

    if args.dump_requirements:
        with open(args.dump_requirements, "w", encoding="utf-8") as f:
            json.dump(REQUIREMENTS, f, indent=2)
        print(f"Wrote {len(REQUIREMENTS)} requirements to {args.dump_requirements}")
        return
    if not args.candidates:
        ap.error("--candidates is required (unless using --dump-requirements)")

    heap = []  # min-heap of (score, candidate_id, candidate)
    n = 0
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            n += 1
            final, comp, honey = score_candidate(c)
            cid = c["candidate_id"]
            item = (final, cid, c, comp, honey)
            if len(heap) < args.topk * 3:
                heapq.heappush(heap, item)
            elif final > heap[0][0]:
                heapq.heapreplace(heap, item)

    # Deterministic final order. Sort on the SAME rounded score that gets written
    # (validator enforces non-increasing rounded score, ties by candidate_id asc),
    # then candidate_id ascending.
    DECIMALS = 4
    scored = [(round(final, DECIMALS), cid, c, comp, honey)
              for (final, cid, c, comp, honey) in heap]
    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))[:args.topk]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (final, cid, c, comp, honey) in enumerate(ranked, 1):
            triples = evaluate_requirements(c, comp, honey)  # lazy: only the top-K
            reasoning = build_reasoning(c, triples, final, honey)
            w.writerow([cid, rank, f"{final:.4f}", reasoning])

    print(f"Scored {n} candidates -> top {len(ranked)} written to {args.out}")
    if ranked:
        print(f"score range: {ranked[0][0]:.4f} (rank 1) .. {ranked[-1][0]:.4f} (rank {len(ranked)})")


if __name__ == "__main__":
    main()
