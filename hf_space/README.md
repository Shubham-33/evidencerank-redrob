---
title: EvidenceRank
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Evidence-based candidate ranker for the Redrob hiring challenge
---

# EvidenceRank — candidate ranker

Self-contained HuggingFace Space for the Redrob *Intelligent Candidate Discovery & Ranking*
challenge. Upload a candidate sample (JSONL or JSON array), or use the bundled 50-candidate
sample, and the ranker scores them on **grounded evidence of fit** — not keyword overlap.
Optionally paste a Job Description (parsed by NVIDIA's free API) to retarget the engine.

The ranker (`rank.py`) is pure Python standard library — CPU-only, no network during ranking.
Full project: https://github.com/Shubham-33/evidencerank-redrob
