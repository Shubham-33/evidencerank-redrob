# EvidenceRank — Redrob Intelligent Candidate Discovery & Ranking Challenge

A CPU-only, **dependency-free** (Python standard library only) ranker that scores candidates
on *grounded evidence of fit* — not keyword overlap. It ranks the top 100 of the 100,000-
candidate pool against the Senior AI Engineer JD in **~12 seconds** on a laptop CPU, with **0
honeypots and 0 keyword-stuffers** in the top 100.

## Approach in one paragraph

Each candidate's fit is a set of `(requirement, evidence, satisfaction)` triples derived from a
declarative model of the JD ([`execution/requirements.json`](execution/requirements.json)).
The score is `final = base × trap_penalties × behavioral_modifier`, with honeypots forced
near-zero by internal-consistency checks. `base` is a weighted sum of six grounded components
(builder identity, retrieval/vector-search skills, shipped-systems career evidence, experience
band, location, evaluation/pedigree); skills are trust-weighted by endorsements × duration ×
assessment so claimed-but-unbacked keywords don't count; behavioral signals (response rate,
recency, open-to-work…) are a bidirectional multiplier so a perfect-on-paper but unavailable
candidate is down-weighted, exactly as the JD instructs. Every design choice was kept or
rejected by an **offline evaluation harness** (planted oracle cases + proxy NDCG/MAP + a
hand-labeled gold set), never by intuition. See [`PLAN.md`](PLAN.md) for the full rationale.

## Reproduce the submission (single command)

No third-party packages required — Python 3.9+ standard library only.

```bash
# from the repo root, with candidates.jsonl available (gunzip the provided .gz if needed)
python execution/rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This streams the 100K-candidate file, scores every candidate, and writes the spec-compliant
top-100 CSV (`candidate_id,rank,score,reasoning`). Runtime: ~12s, well under the 5-minute
budget; peak memory well under 16 GB; no GPU; no network access at any point.

## Validate before submitting

```bash
python "validate_submission.py" ./submission.csv   # the organizer-provided validator
# -> "Submission is valid."
```

## Evaluate locally (no leaderboard, so we built our own)

```bash
# planted regression cases + proxy NDCG/MAP/P@10 + per-component ablation table
python execution/evaluate.py --candidates ./candidates.jsonl --sample 600 --seed 7

# score against the hand-labeled gold set (60 candidates, human relevance tiers)
python execution/evaluate.py --candidates ./candidates.jsonl --gold
```

## Repository layout

| Path | What it is |
|---|---|
| [`execution/rank.py`](execution/rank.py) | The ranker. Stdlib-only, streaming, deterministic. **This is the reproduce target.** |
| [`execution/requirements.json`](execution/requirements.json) | Declarative JD model (each requirement quotes the JD). `rank.py --dump-requirements` regenerates it. |
| [`execution/evaluate.py`](execution/evaluate.py) | Offline eval harness: planted cases, proxy metrics, ablations, gold-set scoring. |
| [`execution/parse_jd.py`](execution/parse_jd.py) | **Offline** JD parser (NVIDIA free API, stdlib) → structured requirements, so the engine can retarget at *any* JD. Pre-compute only; ranking stays LLM-free. |
| [`execution/gold_labels.json`](execution/gold_labels.json) | 60 hand-labeled relevance tiers — a less-correlated yardstick. |
| [`PLAN.md`](PLAN.md) | Full architecture + every eval-gated decision (including what we tested and rejected). |
| [`directives/`](directives/) | Living design notes / SOPs with the learnings behind each component. |
| [`sandbox/`](sandbox/) | Streamlit app to run the ranker on a small sample (the hosted sandbox). |
| `requirements.txt` | Empty by design — the ranker has no third-party dependencies. |

## Compute-constraint compliance (spec §3)

| Constraint | This submission |
|---|---|
| ≤ 5 min wall-clock | ~12 s for 100K on a laptop CPU |
| ≤ 16 GB RAM | Streams line-by-line; peak well under |
| CPU only, no GPU | Pure Python, no GPU code |
| No network during ranking | No imports or calls that touch the network |
| ≤ 5 GB disk | No intermediate state written |

## Honeypots & traps

Honeypots are caught by internal-consistency checks (job duration exceeding its window,
stated experience exceeding career span, many "expert" skills with 0 months used) and pushed
out of contention — **0 in the top 100**. Keyword-stuffers (non-technical titles with stuffed
AI skill lists) are gated by a builder-title requirement plus skill trust-weighting — **0 in
the top 100**.

## AI tools

Claude (Anthropic) was used for architecture discussion, code review, and as a pair-programmer.
No candidate data was sent to any hosted LLM, and **no LLM is called during ranking** (the
ranking step is offline by construction). See [`submission_metadata.yaml`](submission_metadata.yaml).
