# Evaluate Ranker (offline, no leaderboard)

## Goal
Measure ranking quality without the hidden ground truth, so every change to
`execution/rank.py` is keep-or-drop by evidence, not vibes. Produces three things:
planted-case regression results, proxy NDCG@10/@50 + MAP + P@10 on a real sample, and
an ablation table showing which components actually earn their weight.

## Inputs
- `--candidates` : path to `candidates.jsonl`.
- `--sample` : how many real candidates to sample for proxy metrics (default 400; use 600+ for stability).
- `--seed` : sampling seed (default 7). **Always check ≥3 seeds before trusting a delta.**

## Tools
- `execution/evaluate.py` — imports `rank` and scores against an **independent** oracle labeler.
  Ablations monkey-patch `rank` components (honeypot gate, behavioral modifier, trap
  penalties, title gate, skill trust-weighting), restoring originals after each run.

## Steps
1. `python execution/evaluate.py --candidates "<bundle>/candidates.jsonl" --sample 600 --seed 7`
2. Confirm **planted cases pass 6/6** (perfect ≫ stale ≫ stuffer ≈ honeypot≈0). If not, the
   ranker is broken — fix before reading proxy metrics.
3. Read the ablation Δcomp column: **negative Δ when a component is removed = it helps.**
4. Re-run across seeds {1,13,42}; keep components whose help is consistent, scrutinize any
   whose removal consistently *raises* the proxy score.

## Outputs
- Console report (planted checks, oracle tier distribution, ablation table). No file artifact.

## How the proxy works (and its limits)
- `oracle_tier(c)` assigns tier 0–5 from raw profile fields a recruiter sees, using its OWN
  honeypot/consistency check and thresholds — deliberately simpler and separate from the
  ranker so labels don't move during ablations.
- **Caveat:** the oracle and the ranker both encode JD logic, so they're correlated; treat
  metrics as *relative* signal between variants, not absolute truth. A component whose
  removal *raises* the proxy may simply be modeling something the (simpler) oracle ignores —
  investigate, don't auto-delete. The honeypot gate shows ~0 Δ on random samples (few
  honeypots in contention) yet is a DQ safeguard — its value is proven by the planted test,
  so keep it unconditionally.

## Edge cases & learnings (initial run, seeds 1/13/42, sample 600)
- **Behavioral modifier — strong, robust keep.** Removing it costs ~0.007–0.11 composite
  every seed. Matches the JD's explicit "down-weight stale/unresponsive" instruction.
- **Trap penalties — keep.** Mostly negative Δ (helps), occasionally ~0; never meaningfully hurts.
- **Title/builder gate — keep.** Small consistent negative Δ; cheap insurance against stuffers.
- **Skill trust-weighting — investigated and KEPT (was a false alarm).** The original
  ablation looked like it slightly hurt (Δ +0.015/+0.022), but that was an artifact of two
  things: (1) the ablation was **confounded** — the old `_flat_skills` counted CORE terms
  only, so it removed SUPPORT-skill credit *and* trust at once; fixed to isolate trust, Δ
  drops to **+0.001/+0.008 = noise** (seed range ~0.12). (2) The oracle scores skill
  *duration* but **ignores endorsements/assessment**, so the proxy is structurally blind to
  trust-weighting's sharpest signal. A direct, oracle-independent probe (`.tmp/trust_probe.py`,
  4000-candidate sample) settled it: on CORE skills, **builders avg 24.5 mo duration & 12.7
  endorsements; stuffers avg 11.0 mo & 2.0 endorsements** — a 2× / 6× gap. Trust-weighting
  is doing real anti-stuffer work; the proxy just can't see most of it. It's redundant with
  the title gate for the *top 100* (all variants yield 0 stuffers / 0 honeypots there) but
  earns its place on the within-pool gradient (NDCG@50/MAP) and as defense-in-depth. Kept
  `dur+endorsement+assessment` as-is — dropping endorsements would discard the strongest
  separator on the say-so of a metric blind to it.
- **Methodology lesson:** when an ablation flags a component, check the ablation isn't
  *confounded* (removing two things at once) and that the proxy oracle can actually *see*
  what the component models, before acting. Both bit us here.

## Hand-labeled gold set (less-correlated yardstick)
`execution/gold_labels.json` holds human relevance tiers (0-5) for 60 builder-titled
candidates, assigned by reading full profiles against the JD — independent of both the
ranker formula and the rules-oracle. Run with `python execution/evaluate.py --candidates
... --gold`. The ranker scores **MAP=1.0** here (every relevant candidate above every
irrelevant one) with the 3 genuine ML candidates at the very top, above ~10 keyword-stuffer
traps (Business Analysts with CNN/MLflow, a Mobile Dev with "Recommendation Systems adv/60e",
backend SWEs with Milvus/FAISS) — strong independent confirmation. Caveat: it's *saturated*
(few tier-3s, no 4-5s in the sample), so use it to confirm a change doesn't BREAK the top
ordering, not to chase large gains. Notable: in 60 builder-titled candidates the genuine ML
people all used the keywords — no plain-language Tier-5 appeared, suggesting that gap is rarer
than feared.

## Weight tuning (constrained, dual-metric gated)
A coordinate sensitivity sweep (perturb each core weight ±0.04/0.08, renormalize) measured
BOTH gold and proxy-avg. Only two perturbations moved both up: **eval down** and **experience
up**. Shipped the conservative combined move **experience 0.12→0.16, eval 0.10→0.06** (gold
0.9209→0.9287, proxy +0.008, planted 8/8, top-100 hygiene clean). Rationale beyond the
numbers: the eval component is half *education-prestige*, which the JD is explicitly wary of,
so down-weighting it aligns with the JD's anti-pedigree stance. Did NOT chase the max
(eval→0.02 / experience→0.20) to avoid overfitting a saturated gold + correlated proxy.
`location→0.02` lifted gold +0.032 but hurt proxy −0.024 (mixed) → not touched.

## Oracle enrichment (update — proxy no longer blind to recency & endorsements)
The oracle now models two JD concepts it previously ignored, using mechanics deliberately
DIFFERENT from the ranker's so agreement is corroboration, not tautology:
- **Endorsement-aware skill evidence:** core skill counts as *strong* only if duration ≥6 mo
  AND endorsements ≥3 (ranker uses a +0.2-at-≥5 bump — different threshold/shape).
- **Recency-of-domain (drift):** a boolean "is the CURRENT role in the retrieval/ML domain?"
  with a −2 penalty if the candidate has core history but has drifted out (ranker uses a
  continuous, floored decay curve — different mechanic).

Outcomes across seeds {7,1,13,42}, sample 600:
- **Trust-weighting — vindicated.** Now that the proxy sees endorsements, removing it
  consistently HURTS (Δ −0.003…−0.013). Confirms the earlier "harm" was proxy blindness.
- **Temporal decay — confirmed minor.** Still ~noise (Δ ≈ ±0.004) even with a recency-aware
  oracle: drifted candidates are rare in the top ranks and the decay is gentle by design.
  Keep it (planted `stale_domain` + first principles), but don't expect proxy movement.
- **Behavioral modifier — investigated & FIXED (reformulated to floored).** The mixed
  ablation (Δ +0.040 / −0.004 / +0.014 / −0.048) was real: the old *linear* curve
  over-discounted the healthy middle (it knocked 20–40% off candidates with 0.5–0.65
  response rates or 30–88-day inactivity — all normal). A direct probe confirmed the
  modifier *tracked* fit (tier≥4 mean 0.705 > tier 2-3 0.559 > tier≤1 0.521) but penalized
  too hard in absolute terms. Replaced with a **FLOORED** curve: full credit across the
  healthy range, steep only at the extremes the JD actually names — "6 months inactive"
  (~180 d) and "5% response rate". Result: net **+0.024** composite over 12 seeds (wins
  7/12, asymmetric upside), and the ablation now shows it **consistently helps** every seed
  (Δ −0.008…−0.020). Guardrails held: planted `stale` stays buried (0.29), top-100 still
  0 stuffers / 0 honeypots. Caveat: it churned the real top 100 by ~44% (entirely new top
  3) — expected, since it stopped suppressing healthy candidates; the new top 10 were
  eyeballed and are all active (≤88 d), responsive (≥0.50), open-to-work, in-band India ML
  engineers. Recency is load-bearing for burying stale profiles — do NOT drop it (removing
  it un-buries the planted `stale` case from 0.26 to 0.58).
- **Distribution sanity:** tier≥3 now ≈6–8% (was ~14%), matching the JD's "narrow profile,
  10 great matches not 1000 maybes." Planted cases grade recruiter-correctly: perfect=5,
  stale_domain=3, stale=2, stuffer=0, honeypot=0.
- **Residual-correlation caveat (still applies):** shared JD concepts mean some correlation
  is inherent; distinct mechanics keep it from being circular, and the planted cases remain
  the implementation-independent backstop.
