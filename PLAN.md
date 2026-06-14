# Winning Plan — Redrob Intelligent Candidate Discovery & Ranking Challenge

**Codename:** `EvidenceRank` — a CPU-only, no-network, fully reproducible ranker that scores
candidates on *grounded evidence of fit*, not keyword overlap. It fuses the strongest ideas
from both ideation journeys, drops everything that's banned or off-task, and is built to win
on the metric that's actually scored.

---

## 0. What "winning" actually means here (the only scoreboard that matters)

| Lever | Weight / Rule | Design consequence |
|---|---|---|
| **NDCG@10** | 0.50 | The **top 10 must be flawless** — no honeypots, no stuffers, perfect-fit first. Half the score lives here. |
| **NDCG@50** | 0.30 | Top 50 ordering matters; broaden clean coverage. |
| **MAP** | 0.15 | Precision across all relevance tiers — avoid junk anywhere in the 100. |
| **P@10** | 0.05 | ≥ tier-3 candidates in top 10. |
| **Honeypot rate > 10% in top 100** | **DQ** | Internal-consistency checks are non-negotiable, especially in the top 50. |
| **Reproduce in sandbox** (5 min / 16 GB / CPU / no net) | **DQ if it can't** | Stdlib-first; any model is local + optional + time-budgeted. |
| **Stage-4 reasoning checks** | advancement | Grounded, specific, varied, concern-acknowledging `reasoning` per row. |
| **Stage-5 interview** | finalist | Every component must be explainable and defensible by a human. |

**Strategic read:** this is won by *ranking quality on the top 50 + zero own-goals (honeypots, format, reproduction) + a defensible story*. It is **not** won by architectural novelty. Both ideation plans over-indexed on sophistication; we index on the scoreboard.

---

## 1. The unified architecture (best of both, minus the banned parts)

Pipeline = **Understand → Represent → Match (evidence triples) → Score (gate + boost) → Rerank → Explain → Validate**. Three tiers by *when* they run:

```
TIER 0 — OFFLINE PRE-COMPUTE (no time limit; committed as artifacts)
  [JD] ──Claude(offline)──> requirements.json   (each requirement quotes the JD span)
  [100K candidates] ──local MiniLM(offline, optional)──> summary_embeddings.bin
                     └── precomputed feature cache (optional)

TIER 1 — RANKING STEP (≤5 min, 16GB, CPU, NETWORK OFF)   ← this is what gets reproduced
  stream candidates.jsonl
     │
     ├─ HONEYPOT GATE         internal-consistency checks → force near-zero
     ├─ HARD GATES            builder-title gate, location/visa, must-have coverage
     ├─ EVIDENCE MATCH        per requirement → (strong/partial/inferred/absent) + evidence span
     │                        skills trust-weighted by endorsements×duration×assessment
     ├─ TEMPORAL DECAY        transient skills decay e^(−λt) by recency; foundational don't
     ├─ TRAJECTORY            IC→lead ascent, product-vs-services, tenure/job-hop pattern
     ├─ BEHAVIORAL MODIFIER   bidirectional: response rate, last-active, open-to-work … (×)
     └─ TRAP PENALTIES        stuffer / consulting-only / research-only / CV-only (×)
     │
     ▼  fit score + confidence
  [top ~200] ──local cross-encoder rerank (optional, time-budgeted)──> reorder
     │
     ▼
  GROUNDED REASONING (deterministic, fact-checked vs profile strings)
     │
     ▼
  spec-compliant CSV  (candidate_id,rank,score,reasoning)  → validate_submission.py

TIER 2 — EVAL HARNESS (offline)   planted cases + ablations + rerun-consistency
```

**Why this shape wins:** every expensive idea from the plans (LLM JD parsing, embeddings, cross-encoder) is pushed into Tier 0 pre-compute or made an *optional, measured* Tier-1 add-on. The score-bearing core is deterministic structured reasoning that runs in ~12s and reproduces anywhere.

---

## 2. Ideas we adopt — and exactly where each comes from

**From Plan 2 (the principled one):**
- **Understand→Represent→Match→Rank→Explain** reframe.
- **`(requirement, evidence, satisfaction-level)` triple** as the atomic scoring unit — *the* anti-keyword-stuffer mechanism. (evidence over buzzwords)
- **Gated must-haves + additive nice-to-haves + transferable credit** (mirrors the JD's own structure).
- **Grounding: "no evidence, no credit"**; inferred credit is labeled, capped below exact-match.
- **Fit vs. confidence** as two numbers (confidence is the score tie-breaker + reasoning hedge).
- **Bidirectional level matching** (over-qualified is also a miss).
- **Eval harness**: planted positives/negatives + ablation table.

**From Plan 1 (TalentInsight):**
- **Hybrid lexical + semantic recall** — but lexical = structured field checks, semantic = *local* MiniLM (never a hosted API).
- **Temporal skill decay `e^(−λt)`**, split into foundational (no decay) vs transient (fast decay).
- **Cross-encoder rerank on a bounded shortlist** — local `ms-marco-MiniLM`, optional, only if the eval harness shows it lifts NDCG and it fits the budget.
- **Structured JD → scoring matrix** (Tier-0 artifact).

**From our already-working `rank.py` (validated, 12s, passes validator, 0 honeypots in top 100):**
- Stdlib-only honeypot consistency checks, builder-title gate, **bidirectional** behavioral multiplier, trap penalties, grounded per-candidate reasoning, correct 4-column output with rounded-score tie-break.

---

## 3. Two deliberate reversals (where the plans were wrong for THIS task)

1. **Behavioral signals penalize, not just boost.** Plan 2's "boost-only, absence-neutral" fairness stance contradicts the JD verbatim: *"hasn't logged in for 6 months … not actually available. Down-weight them appropriately."* The 23 `redrob_signals` are a **bidirectional multiplier**. (General-hiring fairness ≠ this JD's explicit instruction.)
2. **No hosted LLM anywhere in the ranking path, and the candidate "parse" is mostly free.** The data is already structured JSON — Plan 2's per-candidate LLM parse is redundant *and* the LLM-as-judge step is banned at ranking time. We keep the *logic* (pointwise → pairwise refine on borderline cases) but implement it deterministically / with a local model.

---

## 4. The requirement schema (Tier-0 artifact, derived from the JD)

`requirements.json` — each entry has `type` (gate | core | nice), `weight`, `jd_quote`, and the evidence matchers:

- **GATE — builder identity:** current/recent title in {engineer, ML/AI/NLP/search/data scientist, applied scientist, research engineer}. Non-tech title + stuffed AI skills → stuffer trap.
- **GATE — geography/visa:** India metros (Noida/Pune/Hyderabad/Mumbai/Delhi-NCR/Bengaluru) or relocation-willing; outside-India + no relocation heavily down-weighted (JD: no visa sponsorship).
- **CORE — embeddings retrieval** (sentence-transformers/BGE/E5/embeddings/dense retrieval).
- **CORE — vector DB / hybrid search** (FAISS/Pinecone/Qdrant/Milvus/Weaviate/OpenSearch/Elasticsearch).
- **CORE — ranking/recsys/search systems shipped at product companies** (career-description evidence).
- **CORE — evaluation frameworks** (NDCG/MRR/MAP/A-B testing).
- **CORE — strong Python.**
- **CORE — experience band** 5–9 yrs (peak 6–8), bidirectional.
- **NICE — LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR-tech, distributed/inference-opt, open-source.**
- **NEGATIVE — penalties:** consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/HCL/Tech Mahindra), research/academia-only w/o production, CV/speech/robotics-only w/o NLP/IR, ≤18-mo-avg job-hopping.

Each candidate × requirement → `strong / partial / inferred / absent` with the evidence string, which feeds both the score and the reasoning.

---

## 5. Scoring math (deterministic, explainable, defensible)

```
honeypot(c)          → if any hard inconsistency: final = ε   (out of top 100)
coverage(c)          = Σ weightᵣ · sat(c,r)          over CORE requirements
                       sat: strong=1.0, partial=0.6, inferred=0.35·cap, absent=0
trust(skill)         = f(endorsements, duration_months, assessment_score)   # discounts buzzwords
decay(skill)         = transient ? exp(−λ·years_idle) : 1.0
base                 = w·[builder, coverage, career_evidence, experience, location, eval]
gate_multiplier      = Π soft-gate factors (stuffer, geography…)
trap_multiplier      = Π trap penalties
behavioral_mult      = Π bidirectional signal factors  ∈ [0.25, 1.12]
final                = base · gate_multiplier · trap_multiplier · behavioral_mult
confidence           = g(profile_completeness, evidence_density)   # tie-break + reasoning hedge
```

Sort by `(−round(final,4), candidate_id)` → ranks 1–100. (Validator compares the rounded score for ties.)

---

## 6. Reasoning generation (Stage-4 winner)

Deterministic, **fact-checked against profile strings** (no hallucination), assembled from the
matched triples so every sentence is grounded and **varies** per candidate:
`<title>, <yrs>; evidence: <named core skills + 1 career proof>; <location>. [Concern: <real weak signal>].`
Tone tracks rank (top ranks confident, lower ranks hedged) — Stage-4 explicitly checks rank-consistency and concern-acknowledgement. We already pass this; we'll enrich it from the triple data.

---

## 7. Evaluation harness — the highest-leverage missing piece (build first)

No leaderboard, so we manufacture signal:
1. **Planted oracle cases** — synthesize/seed a known perfect-fit, a known keyword-stuffer, a known honeypot, a stale-but-perfect-on-paper candidate; assert the perfect-fit lands top-5, the others land outside top-50. Regression test on every change.
2. **Hand-labeled gold set** — label ~150 candidates into relevance tiers (0–5) by reading profiles; compute **proxy NDCG@10/@50, MAP, P@10** offline so we optimize the real metric, not vibes.
3. **Ablation table** — toggle each component (honeypot gate, trust-weighting, temporal decay, behavioral multiplier, trap penalties, optional embeddings/cross-encoder); keep only components that *measurably* lift proxy NDCG. This is also the Stage-5 evidence that each part earns its complexity.
4. **Rerun-consistency** — deterministic ⇒ byte-identical output across runs (trust signal).

---

## 8. Submission & defensibility (Stages 3–5)

- **Repo:** clean git history showing *real iteration* (baseline → +honeypot gate → +trust-weighting → +decay → +eval-driven tuning), `README.md` with the single reproduce command, `requirements.txt` (stdlib core = near-empty; optional extras pinned), pre-compute scripts for any artifact.
- **Reproduce command:** `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` — verified <5 min on 16 GB CPU, network off.
- **Sandbox:** HuggingFace Space / Streamlit that runs the ranker on a ≤100-candidate sample end-to-end.
- **Metadata:** honest AI-tools declaration; `methodology_summary` = the §1–§5 story in ≤200 words.
- **Interview readiness:** every number traces to a rule in the JD/spec and a line of code. No black boxes, nothing we can't rebuild on a whiteboard.

---

## 9. Status & build order

| # | Task | State | Score impact |
|---|---|---|---|
| ✅ | Stdlib ranker: gates, honeypot checks, behavioral multiplier, traps, grounded reasoning, valid 4-col CSV, 12s/100K | **DONE** ([execution/rank.py](execution/rank.py)) | Foundation |
| ✅ | **Eval harness** (planted cases + independent-oracle proxy NDCG/MAP/P@10 + ablation table) | **DONE** ([execution/evaluate.py](execution/evaluate.py)) | **Highest** — now measures everything else |
| ✅ | Refactor scoring into explicit **requirement→evidence→satisfaction triples** + `requirements.json` | **DONE** ([execution/rank.py](execution/rank.py), [execution/requirements.json](execution/requirements.json)) | Reasoning quality + defensibility |
| ✅ | **Temporal decay** — gentle floored recency weighting on dated career evidence (foundational, not aggressive) | **DONE** ([execution/rank.py](execution/rank.py)) | NDCG (within-pool gradient) |
| 4 | Reasoning enrichment from triples (more specific, still fact-checked) | next | Stage-4 |
| ⊘ | **Optional** semantic component (TF-IDF / local MiniLM) | **EVALUATED → NOT shipped** (kept stdlib) | eval-gate said no |
| ✅ | Repo hygiene, sandbox, metadata, methodology write-up | **DONE** (README, requirements.txt, sandbox/, submission_metadata.yaml) | Stages 3–5 |

**Governing principle:** ship the deterministic core that already clears every hard gate, then let the **eval harness** decide which sophistication is worth adding. Sophistication that doesn't move proxy NDCG doesn't ship.

**Worked example of the principle (task 5, semantic component):** tested TF-IDF cosine similarity to the JD as an added component. It *hurt* proxy composite at every weight (−0.0005 / −0.0066 / −0.0082 at w=0.05/0.10/0.15) — lexically redundant with the structured matchers. The one genuine motivation, "plain-language Tier 5s" (real builders who avoid the keywords), is NOT solved by TF-IDF either: a planted plain-language builder scored cosine 0.112, barely above the stuffer's 0.065. Only a true MiniLM embedding could catch them — but that benefit is unmeasurable with the keyword-based proxy, adds a fragile torch dependency (Python 3.14 wheel risk) that threatens the stdlib reproduction advantage, and the upside is bounded (the plain-language case still scores 0.402, mid-pack, not buried). **Decision: keep the ranker stdlib-only.** Strong Stage-4/5 narrative — semantic search was tested and rejected on evidence, not omitted by ignorance. (Documented limitation: pure plain-language Tier 5s with zero domain vocabulary are under-ranked; an offline-precomputed local embedding is the future-work fix if the metric ever justifies it.)

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Honeypot in top 100 → DQ | Consistency gate + eval-harness planted honeypot regression test; audit top 50 by eye. |
| Can't reproduce in sandbox → DQ | Stdlib core; any model local+optional; time/memory-budget every run. |
| Format auto-reject | Exactly 4 columns; run `validate_submission.py` in CI before every submission. |
| Over-fitting to our own gold labels | Keep gold set blind-labeled before tuning; prefer robust monotone features over knife-edge weights. |
| Reasoning flagged as templated/hallucinated | Generate from grounded triples; fact-check vs profile; vary structure by which signals fired. |
| "AI-only" suspicion at Stage 4/5 | Real git iteration + ablation evidence + whiteboard-defensible components. |
```
