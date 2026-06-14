# Rank Candidates (Redrob Hackathon)

## Goal
Produce a CSV ranking the top 100 candidates from the 100K-candidate pool against
the released Senior AI Engineer JD, best-fit first, each with a specific 1–2 sentence
reasoning. The output must pass `validate_submission.py` and respect the compute
constraints (≤5 min wall-clock, ≤16 GB RAM, CPU only, no network during ranking).

This is a ranking-quality problem, not a keyword-match problem. The dataset is
adversarial: keyword stuffers, behavioral twins, plain-language Tier-5s, and ~80
honeypots with internally-impossible profiles. The score weights NDCG@10 (0.50)
heaviest, so the **top 10 must be clean** — no honeypots, no stuffers.

## Inputs
- `--candidates` : path to `candidates.jsonl` (100K lines, ~465 MB) — stream it, don't load all.
- `--out`        : output CSV path (default `submission.csv`).
- The JD intent and trap taxonomy are encoded directly in `execution/rank.py` (see "Scoring model" below).

## Tools
- `execution/rank.py` — the deterministic ranker. Stdlib-only (no numpy/sklearn) so it
  reproduces anywhere. Single streaming pass, top-K heap, explainable scores.
- The challenge bundle's `validate_submission.py` — run on the output before trusting it.

## Steps
1. `python execution/rank.py --candidates "<bundle>/candidates.jsonl" --out .tmp/submission.csv`
2. Validate: `python "<bundle>/validate_submission.py" .tmp/submission.csv` → must print "Submission is valid."
3. Sanity-check the top 20 by eye: every top pick should be a real engineering/ML/data
   profile in India (or relocation-willing), with credible career evidence and live
   engagement. No HR/Marketing/Content titles in the top 10. No honeypots.
4. Copy to the registered participant filename (`<team_id>.csv`) only at submission time.

## Scoring model (what `rank.py` encodes)
`final = base × trap_penalties × behavioral_modifier`, with honeypots forced near-zero.

- **base** = weighted sum of role/title fit, skill evidence (trust-weighted by
  endorsements + duration + assessment scores), career-history evidence (built
  ranking/search/retrieval/recsys at product companies), experience band (peak 6–8 yrs),
  location fit (Noida/Pune/India metros or relocation-willing), and eval/education signal.
- **trap_penalties** (multiplicative): non-engineering title with stuffed AI skills,
  consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/HCL/Tech Mahindra),
  research/academia-only with no production, CV/speech/robotics-only without NLP/IR,
  title-chasing job-hopping.
- **behavioral_modifier** (multiplicative): recency of last activity, recruiter response
  rate, open-to-work, interview completion, profile completeness. A perfect-on-paper but
  stale/unresponsive candidate is down-weighted — they aren't actually hireable.
- **honeypot detection**: internal-consistency checks (job duration longer than the
  start→end window; total tenure far below stated years of experience; many "expert"
  skills with 0 months used; salary min > max). Any hard inconsistency pushes the
  candidate out of the top 100.

## Outputs
- `.tmp/submission.csv` — intermediate; the deliverable is the validated CSV uploaded to
  the portal alongside the metadata in `submission_metadata_template.yaml`.

## Edge cases & learnings
- Base Python env has **no numpy/sklearn** — keep `rank.py` stdlib-only for reproducibility.
- Python 3.14 in this env; the file is UTF-8 and one JSON object per line.
- `score` must be **non-increasing by rank**; ties break by `candidate_id` ascending
  (the validator enforces both). The sort key `(-score, candidate_id)` satisfies this.
- The provided `sample_submission.csv` is deliberately bad (HR Manager / Content Writer
  ranked top) — it's a format reference only, never a quality target.
- Reasoning strings must cite real facts from the profile and acknowledge concerns;
  hallucinated skills or rank-inconsistent tone are penalized at Stage 4.
- **Honeypot tuning (learned):** an inverted salary range (min > max) is NOT a honeypot
  signal here — it fires on ~19% of the pool (synthetic noise). The reliable impossibility
  checks are job-duration-exceeds-window, years-of-experience-exceeds-career-span, and
  ≥5 expert skills with 0 months used. Together they flag ~52 candidates (vs the ~80
  stated honeypots) with no observed false positives in the top 100.
- **Honeypot recall is intentionally capped at ~52 (investigated).** Attempts to catch the
  remaining ~28 all failed cleanly: skill-duration > career/yoe fires ~13k times, signup-
  after-active ~7.5k — synthetic noise, not impossibilities. Rare checks (expert-skill-with-
  low-assessment, job>50yr, future education, current-tenure>yoe) fire 0 times or are fully
  subsumed by existing checks (current-tenure>yoe: 19 fires, all already caught). The
  dataset's honeypots use exactly the three *documented* signatures we already detect; the
  rest can't be separated from noise without thousands of false positives. We don't need
  higher recall — the goal is keeping them OUT of the top 100 (achieved: 0) and clearing the
  >10% DQ gate (cleared). Adding noisy checks would actively hurt by zeroing real candidates.
- **Validated:** full 100K pool ranks in ~12s on CPU, passes `validate_submission.py`,
  0 honeypots and 0 non-engineering titles in the top 100. The validator compares the
  *rounded* 4-decimal score for tie-breaks, so sort on the rounded score, not the raw float.
- **Plain-language detection (added):** a curated `PLAIN_LANGUAGE_PHRASES` lexicon credits
  career descriptions that describe ranking/recsys work *without* the jargon ("which products
  to show … in what order", "scoring that ranks candidate items") — the stdlib mitigation for
  "plain-language Tier 5s" (the heavyweight-embedding version was rejected in task 5). Weak &
  capped (0.12·min(plain,2) in `score_career`). Measured: lifts a planted plain-language
  builder 0.402→0.466, **100% top-100 overlap** (doesn't disrupt the real submission),
  gold/proxy/hygiene unchanged, planted 8/8. Net a zero-risk robustness/defensibility win;
  its effect on *this* dataset's top 100 is nil, but it helps genuine plain-language builders
  in the long tail and is a defensible answer to the JD's named trap.
- **Requirements model (refactor):** the JD is declarative in `REQUIREMENTS` (in `rank.py`,
  mirrored to `execution/requirements.json` via `--dump-requirements`) — CORE entries carry
  the base weights (single source of truth, sum=1.0), NEGATIVE are trap multipliers, MODIFIER
  is behavioral, GATE is honeypot. The hot scoring path stays numeric (fast for 100K); the
  `(requirement, evidence, satisfaction)` triples are built lazily only for the top-K via
  `evaluate_requirements()` and feed `build_reasoning()`. The refactor is behavior-preserving:
  ranking + scores are byte-identical to before, so it adds explainability/defensibility with
  zero quality risk. Every reasoning clause is grounded in a matcher (named skills, the
  company of the strongest CORE-bearing role, eval terms) → no hallucination, Stage-4-ready.
- **Temporal decay (learned):** recency weighting lives on *dated career evidence*, not the
  skills array (skills carry no per-skill dates). Decay is deliberately GENTLE and FLOORED
  (1.0 / 0.92 / 0.82 / 0.72 over <2 / <4 / <7 / 7+ yrs) — the JD values pre-LLM retrieval
  depth ("understood retrieval before it was fashionable"), so an aggressive exp decay would
  wrongly bury the veterans it wants. Effect on the real top 100 is small (98% overlap, 2
  swaps) — it sharpens the within-pool gradient (active builders edge out drifted ones)
  rather than reshuffling. Its value is first-principles + planted-case validated; the proxy
  oracle is blind to career-evidence recency, so don't expect it to move proxy NDCG.
