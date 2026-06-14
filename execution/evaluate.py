#!/usr/bin/env python3
"""
Offline evaluation harness for the EvidenceRank ranker.

There is no live leaderboard, so we manufacture signal three ways:
  1. PLANTED ORACLE CASES — synthetic perfect-fit / stuffer / honeypot / stale-but-perfect
     candidates with known correct ordering. Regression test on every change.
  2. PROXY METRICS — sample real candidates, label each with an INDEPENDENT rules-based
     oracle (deliberately not the ranker's formula), and compute NDCG@10/@50, MAP, P@10
     of the ranker's ordering against those labels. Optimize the metric that's scored.
  3. ABLATION TABLE — toggle each ranker component off (monkey-patch) and re-measure, so
     only components that measurably lift proxy NDCG survive. Also the Stage-5 defense.

    python evaluate.py --candidates ./candidates.jsonl [--sample 400] [--seed 7]

Labels are a PROXY, not ground truth — use deltas between variants, not absolute values.
See directives/evaluate_ranker.md.
"""

import argparse
import math
import random
import sys

import rank  # the ranker under test (same directory)

TODAY = rank.TODAY


# ===========================================================================
# Independent oracle labeler  (tier 0..5)
# Self-contained: must NOT call any rank.* component that ablations patch, or
# the labels would shift between variants. Reflects the "obvious recruiter read"
# of the JD using raw profile fields, with different thresholds than the ranker.
# ===========================================================================

ORACLE_CORE = ["embedding", "retrieval", "vector", "faiss", "pinecone", "qdrant",
               "milvus", "weaviate", "opensearch", "elasticsearch", "ranking",
               "recommend", "recsys", "semantic search", "information retrieval",
               "nlp", "sentence transformer", "learning to rank"]
ORACLE_BUILDER = ["engineer", "scientist", "developer", "ml", "machine learning",
                  "applied", "research engineer", "nlp", "search", "architect", "sde"]
ORACLE_NONTECH = ["marketing", "content writer", "copywriter", "hr ", "human resource",
                  "recruiter", "talent acquisition", "sales", "account manager",
                  "business development", "operations manager", "accountant", "teacher"]
ORACLE_INDIA = ["noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                "bengaluru", "bangalore", "ncr", "india"]
ORACLE_CONSULTING = ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
                     "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "mphasis"]
# ML-specific current-role tokens — stricter than ORACLE_BUILDER (which includes generic
# "engineer"/"backend"). Used only to decide whether someone has DRIFTED out of the domain.
ORACLE_ML_TITLE = ["ml", "machine learning", "ai ", "applied scientist", "data scientist",
                   "research engineer", "nlp", "search engineer", "recommendation"]


def _parse_date(s):
    return rank.parse_date(s)  # pure helper, never patched


def oracle_is_honeypot(c):
    """Independent internal-consistency check (does not call rank.honeypot_flags)."""
    prof = c.get("profile", {})
    hist = c.get("career_history", []) or []
    for job in hist:
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date")) or TODAY
        if start:
            window = (end.year - start.year) * 12 + (end.month - start.month)
            if (job.get("duration_months", 0) or 0) - window > 6:
                return True
    starts = [_parse_date(j.get("start_date")) for j in hist]
    starts = [s for s in starts if s]
    if starts:
        span_yrs = ((TODAY - min(starts)).days) / 365.0
        if (prof.get("years_of_experience", 0) or 0) - span_yrs > 2.0:
            return True
    zero_expert = sum(1 for s in (c.get("skills") or [])
                      if s.get("proficiency") in ("advanced", "expert")
                      and (s.get("duration_months", 0) or 0) == 0)
    return zero_expert >= 5


def _current_role_has_core(c):
    """Is the candidate's CURRENT role actually in the retrieval/ML domain? (recency-of-domain)"""
    for j in (c.get("career_history") or []):
        if j.get("is_current") or not j.get("end_date"):
            d = (j.get("description") or "").lower()
            if any(t in d for t in ORACLE_CORE):
                return True
    return False


def oracle_tier(c):
    """Recruiter-obvious relevance tier 0..5, derived from the JD's concepts but with
    DIFFERENT mechanics than the ranker (boolean current-role check vs. the ranker's
    continuous decay curve; endorsement gate at 3 vs. the ranker's +0.2-at-5). Shared
    concepts → agreement is corroboration; distinct mechanics → not tautological."""
    if oracle_is_honeypot(c):
        return 0

    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    title = (prof.get("current_title") or "").lower()
    loc = (prof.get("location") or "").lower()
    country = (prof.get("country") or "").lower()

    builder = any(t in title for t in ORACLE_BUILDER)
    nontech = any(t in title for t in ORACLE_NONTECH)
    if nontech and not builder:
        return 0  # keyword-stuffer trap: non-tech title regardless of skill list

    # Endorsement-AWARE skill evidence: a recruiter trusts a backed skill more than a
    # claimed-but-thin one. Duration-used AND endorsed = strong; used-but-unendorsed = weak.
    strong_core = weak_core = 0
    for s in (c.get("skills") or []):
        name = (s.get("name") or "").lower()
        if any(t in name for t in ORACLE_CORE) and (s.get("duration_months", 0) or 0) >= 6:
            if (s.get("endorsements", 0) or 0) >= 3:
                strong_core += 1
            else:
                weak_core += 1

    career_core = 0
    consulting_jobs = 0
    for j in (c.get("career_history") or []):
        d = (j.get("description") or "").lower()
        career_core += sum(1 for t in ORACLE_CORE if t in d)
        if any(f in (j.get("company") or "").lower() for f in ORACLE_CONSULTING):
            consulting_jobs += 1

    yoe = prof.get("years_of_experience", 0) or 0
    in_band = 5 <= yoe <= 9
    near_band = 4 <= yoe <= 11
    in_india = any(t in loc for t in ORACLE_INDIA) or "india" in country

    # Recency-of-domain (mechanic distinct from the ranker's decay curve): is the candidate
    # CURRENTLY in the field, or have they drifted out of it?
    currently_building = _current_role_has_core(c)
    ml_current = any(t in title for t in ORACLE_ML_TITLE)
    has_core_history = career_core >= 1 or strong_core >= 1 or weak_core >= 1
    drifted = has_core_history and not currently_building and not ml_current

    # Availability gate (JD: stale/unresponsive => not actually hireable).
    rr = sig.get("recruiter_response_rate")
    last = _parse_date(sig.get("last_active_date"))
    stale = last is not None and (TODAY - last).days > 180
    unresponsive = rr is not None and rr < 0.25

    # Checklist -> tier.
    pts = 0
    pts += 2 if strong_core >= 2 else (1 if strong_core == 1 else (1 if weak_core >= 2 else 0))
    pts += 1 if career_core >= 2 else 0
    pts += 1 if currently_building else 0           # recency bonus (in the field NOW)
    pts += 2 if in_band else (1 if near_band else 0)
    pts += 1 if in_india else 0
    pts += 1 if builder else 0
    if consulting_jobs and (c.get("career_history") and consulting_jobs == len(c["career_history"])):
        pts -= 2                                     # consulting-only career
    if drifted:
        pts -= 2                                     # recency penalty (drifted out of domain)
    if stale or unresponsive:
        pts = min(pts, 3)                            # can't actually be hired

    if pts >= 8:
        return 5
    if pts >= 6:
        return 4
    if pts >= 4:
        return 3
    if pts >= 3:
        return 2
    if pts >= 1:
        return 1
    return 0


# ===========================================================================
# Metrics
# ===========================================================================

def _gain(tier):
    return (2 ** tier) - 1


def ndcg_at_k(ranked_tiers, k):
    dcg = sum(_gain(t) / math.log2(i + 2) for i, t in enumerate(ranked_tiers[:k]))
    ideal = sorted(ranked_tiers, reverse=True)
    idcg = sum(_gain(t) / math.log2(i + 2) for i, t in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked_tiers, rel_threshold=3):
    rels = [1 if t >= rel_threshold else 0 for t in ranked_tiers]
    total_rel = sum(rels)
    if total_rel == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, r in enumerate(rels, 1):
        if r:
            hits += 1
            ap += hits / i
    return ap / total_rel


def precision_at_k(ranked_tiers, k, rel_threshold=3):
    top = ranked_tiers[:k]
    return sum(1 for t in top if t >= rel_threshold) / max(1, len(top))


def composite(ndcg10, ndcg50, ap, p10):
    # Mirrors the challenge's official weighting.
    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * ap + 0.05 * p10


# ===========================================================================
# Planted oracle cases (synthetic, known-correct ordering)
# ===========================================================================

def _skill(name, prof="expert", dur=36, end=20):
    return {"name": name, "proficiency": prof, "endorsements": end, "duration_months": dur}


def _job(company, title, start, end, dur, current, industry, size, desc):
    return {"company": company, "title": title, "start_date": start, "end_date": end,
            "duration_months": dur, "is_current": current, "industry": industry,
            "company_size": size, "description": desc}


def _signals(**over):
    base = dict(profile_completeness_score=90, signup_date="2020-01-01",
                last_active_date="2026-06-10", open_to_work_flag=True,
                profile_views_received_30d=40, applications_submitted_30d=3,
                recruiter_response_rate=0.85, avg_response_time_hours=5,
                skill_assessment_scores={}, connection_count=300,
                endorsements_received=120, notice_period_days=30,
                expected_salary_range_inr_lpa={"min": 40, "max": 60},
                preferred_work_mode="hybrid", willing_to_relocate=True,
                github_activity_score=70, search_appearance_30d=50,
                saved_by_recruiters_30d=8, interview_completion_rate=0.9,
                offer_acceptance_rate=0.5, verified_email=True, verified_phone=True,
                linkedin_connected=True)
    base.update(over)
    return base


def planted_cases():
    cases = {}

    # PERFECT FIT — builder title, real retrieval skills, product career, India, available.
    cases["perfect"] = {
        "candidate_id": "CAND_9000001",
        "profile": {"anonymized_name": "A", "headline": "Senior ML Engineer | Search & Ranking",
                    "summary": "7 years building embeddings-based retrieval and ranking systems "
                               "deployed to real users at scale; hybrid search, evaluation with NDCG/MRR.",
                    "location": "Pune, Maharashtra", "country": "India",
                    "years_of_experience": 7.0, "current_title": "Senior Machine Learning Engineer",
                    "current_company": "ProductCo", "current_company_size": "201-500",
                    "current_industry": "Software"},
        "career_history": [
            _job("ProductCo", "Senior ML Engineer", "2021-01-01", None, 65, True, "Software",
                 "201-500", "Built hybrid retrieval and learning-to-rank recommendation system "
                 "with FAISS and embeddings, deployed to production users; evaluated with NDCG and A/B tests."),
            _job("SearchStartup", "ML Engineer", "2018-01-01", "2020-12-01", 35, False, "Software",
                 "51-200", "Owned semantic search relevance and vector database (Qdrant) infrastructure."),
        ],
        "education": [{"institution": "IIT", "degree": "BTech", "field_of_study": "CS",
                       "start_year": 2012, "end_year": 2016, "grade": None, "tier": "tier_1"}],
        "skills": [_skill("Embeddings"), _skill("FAISS"), _skill("Semantic Search"),
                   _skill("Learning to Rank"), _skill("Python"), _skill("NDCG Evaluation")],
        "redrob_signals": _signals(),
    }

    # KEYWORD STUFFER — non-tech title, AI skills claimed with zero real use.
    cases["stuffer"] = {
        "candidate_id": "CAND_9000002",
        "profile": {"anonymized_name": "B", "headline": "Marketing Manager",
                    "summary": "Marketing manager interested in AI.",
                    "location": "Pune, Maharashtra", "country": "India",
                    "years_of_experience": 7.0, "current_title": "Marketing Manager",
                    "current_company": "AdCo", "current_company_size": "201-500",
                    "current_industry": "Marketing"},
        "career_history": [
            _job("AdCo", "Marketing Manager", "2019-01-01", None, 89, True, "Marketing",
                 "201-500", "Ran marketing campaigns and brand strategy."),
        ],
        "education": [],
        "skills": [_skill("Embeddings", dur=0, end=0), _skill("FAISS", dur=0, end=0),
                   _skill("Pinecone", dur=0, end=0), _skill("Semantic Search", dur=0, end=0),
                   _skill("Vector Database", dur=0, end=0), _skill("NLP", dur=0, end=0)],
        "redrob_signals": _signals(),
    }

    # HONEYPOT — internally impossible (duration far exceeds the role window).
    cases["honeypot"] = {
        "candidate_id": "CAND_9000003",
        "profile": {"anonymized_name": "C", "headline": "ML Engineer",
                    "summary": "ML engineer.", "location": "Pune, Maharashtra", "country": "India",
                    "years_of_experience": 7.0, "current_title": "Machine Learning Engineer",
                    "current_company": "NewCo", "current_company_size": "11-50",
                    "current_industry": "Software"},
        "career_history": [
            _job("NewCo", "ML Engineer", "2024-01-01", None, 96, True, "Software", "11-50",
                 "Built retrieval and ranking with embeddings and FAISS."),  # 96 mo since 2024 = impossible
        ],
        "education": [],
        "skills": [_skill("Embeddings"), _skill("FAISS"), _skill("Semantic Search"), _skill("Python")],
        "redrob_signals": _signals(),
    }

    # STALE-BUT-PERFECT — identical to perfect but inactive & unresponsive (not hireable).
    stale = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cases["perfect"].items()}
    stale["candidate_id"] = "CAND_9000004"
    stale["redrob_signals"] = _signals(last_active_date="2024-06-01",
                                       recruiter_response_rate=0.05, open_to_work_flag=False)
    cases["stale"] = stale

    # STALE-DOMAIN — strong retrieval work but it ENDED ~6 yrs ago; current role unrelated.
    # Should rank BELOW perfect (recency decay) yet WELL ABOVE the stuffer (foundational
    # IR depth persists — the JD values pre-LLM retrieval experience).
    cases["stale_domain"] = {
        "candidate_id": "CAND_9000005",
        "profile": {"anonymized_name": "E", "headline": "Backend Engineer",
                    "summary": "Backend engineer working on payments infrastructure.",
                    "location": "Pune, Maharashtra", "country": "India",
                    "years_of_experience": 8.0, "current_title": "Backend Engineer",
                    "current_company": "PayCo", "current_company_size": "201-500",
                    "current_industry": "Software"},
        "career_history": [
            _job("PayCo", "Backend Engineer", "2020-02-01", None, 76, True, "Software",
                 "201-500", "Built payments APIs and transaction processing services."),
            _job("SearchCo", "ML Engineer", "2016-01-01", "2020-01-01", 48, False, "Software",
                 "51-200", "Built embeddings-based retrieval and learning-to-rank "
                 "recommendation system with FAISS, deployed to production; evaluated with NDCG."),
        ],
        "education": [],
        "skills": [_skill("Embeddings", dur=48), _skill("FAISS", dur=48),
                   _skill("Semantic Search", dur=48), _skill("Python")],
        "redrob_signals": _signals(),
    }

    return cases


def run_planted(verbose=True):
    cases = planted_cases()
    scored = {k: rank.score_candidate(v)[0] for k, v in cases.items()}
    checks = [
        ("perfect outranks stuffer", scored["perfect"] > scored["stuffer"]),
        ("perfect outranks honeypot", scored["perfect"] > scored["honeypot"]),
        ("perfect outranks stale-but-perfect", scored["perfect"] > scored["stale"]),
        ("honeypot pushed near-zero (<0.05)", scored["honeypot"] < 0.05),
        ("stuffer pushed low (<0.3)", scored["stuffer"] < 0.30),
        ("stale meaningfully below perfect (<0.7x)", scored["stale"] < 0.7 * scored["perfect"]),
        ("stale-domain below perfect (recency decay)", scored["stale_domain"] < scored["perfect"]),
        ("stale-domain well above stuffer (foundational depth persists)",
         scored["stale_domain"] > 2 * scored["stuffer"] + 0.2),
    ]
    passed = sum(1 for _, ok in checks if ok)
    if verbose:
        print("PLANTED ORACLE CASES")
        for k in ["perfect", "stale_domain", "stale", "stuffer", "honeypot"]:
            print(f"  {k:12s} score = {scored[k]:.4f}")
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  => {passed}/{len(checks)} checks passed\n")
    return passed == len(checks)


# ===========================================================================
# Proxy metrics over a real-candidate sample + ablations
# ===========================================================================

def reservoir_sample(path, n, seed):
    rng = random.Random(seed)
    sample = []
    import json
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if len(sample) < n:
                sample.append(c)
            else:
                j = rng.randint(0, i)
                if j < n:
                    sample[j] = c
    return sample


def evaluate_sample(sample, tiers):
    """Score the sample with the (possibly patched) ranker, return the 4 metrics."""
    scored = [(rank.score_candidate(c)[0], tiers[c["candidate_id"]], c["candidate_id"])
              for c in sample]
    scored.sort(key=lambda x: (-round(x[0], 4), x[2]))  # ranker's own ordering rule
    ranked_tiers = [t for _, t, _ in scored]
    ndcg10 = ndcg_at_k(ranked_tiers, 10)
    ndcg50 = ndcg_at_k(ranked_tiers, 50)
    ap = average_precision(ranked_tiers)
    p10 = precision_at_k(ranked_tiers, 10)
    return ndcg10, ndcg50, ap, p10, composite(ndcg10, ndcg50, ap, p10)


# Ablations: (name, patch) where patch swaps in a neutralized component and
# returns the originals to restore. Labels (tiers) are fixed across all variants.
def _flat_skills(c):
    # Isolate trust-weighting ONLY: keep the CORE(1.0)+SUPPORT(0.4) structure but drop the
    # endorsement/duration/assessment trust multiplier (set it to 1.0). An earlier version
    # counted CORE terms only, which CONFOUNDED "remove trust" with "remove SUPPORT credit"
    # and made trust-weighting look harmful. See .tmp/trust_probe.py / the directive.
    total = 0.0
    for s in (c.get("skills") or []):
        name = (s.get("name") or "").lower()
        core = any(t.strip() in name for t in rank.CORE_TERMS)
        supp = any(t.strip() in name for t in rank.SUPPORT_TERMS)
        if not (core or supp):
            continue
        total += 1.0 if core else 0.4
    return rank.clamp(total / 4.0)


ABLATIONS = {
    "full (baseline)": lambda: {},
    "− honeypot gate": lambda: {"honeypot_flags": lambda c: []},
    "− behavioral modifier": lambda: {"behavioral_modifier": lambda sig: 1.0},
    "− trap penalties": lambda: {"trap_penalties": lambda *a: 1.0},
    "− title/builder gate": lambda: {"score_title": lambda prof, txt: 0.7},
    "− skill trust-weighting": lambda: {"score_skills": _flat_skills},
    "− temporal decay": lambda: {"recency_factor": lambda years: 1.0},
}


def run_ablations(sample, tiers):
    print("ABLATION TABLE  (proxy metrics vs. independent oracle labels)")
    print(f"  {'variant':26s} {'NDCG@10':>8s} {'NDCG@50':>8s} {'MAP':>6s} {'P@10':>6s} {'COMP':>7s}  Δcomp")
    base_comp = None
    rows = []
    for name, make_patch in ABLATIONS.items():
        patch = make_patch()
        originals = {k: getattr(rank, k) for k in patch}
        for k, fn in patch.items():
            setattr(rank, k, fn)
        try:
            n10, n50, ap, p10, comp = evaluate_sample(sample, tiers)
        finally:
            for k, fn in originals.items():
                setattr(rank, k, fn)
        if base_comp is None:
            base_comp = comp
        delta = comp - base_comp
        rows.append((name, n10, n50, ap, p10, comp, delta))
        print(f"  {name:26s} {n10:8.4f} {n50:8.4f} {ap:6.3f} {p10:6.3f} {comp:7.4f}  {delta:+.4f}")
    print("\n  Reading: a NEGATIVE Δcomp when a component is removed means that component")
    print("  HELPS (its removal hurts the proxy score). Components with ~0 Δ aren't earning")
    print("  their complexity on this sample.\n")
    return rows


def load_gold(path="execution/gold_labels.json"):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)["labels"]


def fetch_candidates(candidates_path, ids):
    import json
    want = set(ids)
    found = {}
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["candidate_id"] in want:
                found[c["candidate_id"]] = c
                if len(found) == len(want):
                    break
    return found


def evaluate_gold(candidates_path, gold=None, verbose=True):
    """Score the hand-labeled gold candidates with the ranker and report metrics against
    HUMAN tiers — a yardstick far less correlated with the ranker than the rules-oracle."""
    gold = gold or load_gold()
    cands = fetch_candidates(candidates_path, gold.keys())
    tiers = {cid: gold[cid] for cid in cands}
    metrics = evaluate_sample(list(cands.values()), tiers)
    if verbose:
        dist = {}
        for t in tiers.values():
            dist[t] = dist.get(t, 0) + 1
        print("GOLD SET (hand-labeled, n=%d)" % len(cands))
        print("  tier distribution:", {k: dist.get(k, 0) for k in range(6)})
        print(f"  NDCG@10={metrics[0]:.4f} NDCG@50={metrics[1]:.4f} MAP={metrics[2]:.3f} "
              f"P@10={metrics[3]:.3f}  COMPOSITE={metrics[4]:.4f}")
        # show the ranker's ordering of the gold set with tiers, to eyeball failures
        scored = sorted(((rank.score_candidate(c)[0], cid, tiers[cid]) for cid, c in cands.items()),
                        key=lambda x: -x[0])
        line = " ".join(f"{t}" for _, _, t in scored)
        print(f"  ranker order by tier (best->worst): {line}")
        print("  (ideal would be all 3s, then 2s, then 1s, then 0s)\n")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gold", action="store_true", help="also evaluate against the hand-labeled gold set")
    args = ap.parse_args()

    ok = run_planted()

    if args.gold:
        evaluate_gold(args.candidates)

    print(f"Sampling {args.sample} candidates (seed={args.seed}) and labeling with the "
          f"independent oracle…")
    sample = reservoir_sample(args.candidates, args.sample, args.seed)
    tiers = {c["candidate_id"]: oracle_tier(c) for c in sample}
    dist = {}
    for t in tiers.values():
        dist[t] = dist.get(t, 0) + 1
    print("  oracle tier distribution:", {k: dist.get(k, 0) for k in range(6)})
    rel = sum(1 for t in tiers.values() if t >= 3)
    print(f"  relevant (tier>=3): {rel}/{len(sample)}\n")

    run_ablations(sample, tiers)

    if not ok:
        print("PLANTED CASES FAILED — fix the ranker before trusting proxy metrics.")
        sys.exit(1)


if __name__ == "__main__":
    main()
