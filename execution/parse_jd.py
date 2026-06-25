#!/usr/bin/env python3
"""Offline JD parser — turn any job description into a structured requirements config the
ranker can consume. This is a PRE-COMPUTE step (run once, offline); the ranking step in
rank.py stays LLM-free and CPU-only per the challenge spec.

Backend: NVIDIA's free OpenAI-compatible API (https://integrate.api.nvidia.com/v1). Set
NVIDIA_API_KEY in the environment (or pass --api-key). Falls back to a deterministic stdlib
extraction if no key / the call fails, so it always returns *something* usable.

    export NVIDIA_API_KEY=nvapi-...
    python parse_jd.py --jd job_description.txt --out requirements_jd.json
    python parse_jd.py --jd job_description.txt --mock      # no network, deterministic

Output schema (consumed by rank.apply_jd_config):
    role, seniority, experience_min, experience_max, core_skills[], support_skills[],
    preferred_locations[], country, anti_patterns[], summary
Only stdlib is used (urllib for HTTP) — no third-party packages.
"""
import argparse
import json
import os
import re
import ssl
import sys
import urllib.request

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

SCHEMA_KEYS = ["role", "seniority", "experience_min", "experience_max", "core_skills",
               "support_skills", "relevant_titles", "preferred_locations", "country",
               "anti_patterns", "summary"]

PROMPT = """You extract structured hiring requirements from a job description so a ranking
system can score candidates. Return ONLY a single JSON object (no prose, no markdown fences)
with EXACTLY these keys:

- role: short role title (string)
- seniority: one of "junior", "mid", "senior", "lead"
- experience_min: minimum years of experience (number)
- experience_max: maximum years of experience (number)
- core_skills: array of lowercase MUST-HAVE skill/technology phrases the role truly needs
  (infer them even if implied — e.g. a search role implies "retrieval", "ranking")
- support_skills: array of lowercase nice-to-have skills
- relevant_titles: array of lowercase job-title keywords a matching candidate's CURRENT
  title would contain (e.g. a Frontend role -> ["frontend","engineer","ui","web"]; an ML
  role -> ["machine learning","ml","ai","engineer"]). Include the domain words AND the
  generic role word (engineer/developer/scientist/designer/manager).
- preferred_locations: array of lowercase cities/regions the role prefers (empty if remote/any)
- country: lowercase country, or "any"
- anti_patterns: array, any of ["consulting_only","research_only","cv_speech_only","job_hopper","non_technical"]
  that this JD would explicitly down-weight (empty if none)
- summary: one sentence describing what the role actually needs

Job description:
\"\"\"
{jd}
\"\"\"
"""


def call_nvidia(jd_text, api_key, model=DEFAULT_MODEL, timeout=60):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(jd=jd_text[:12000])}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }).encode("utf-8")
    req = urllib.request.Request(NVIDIA_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_json(content):
    """Pull the JSON object out of a model response (strips ``` fences / stray prose)."""
    s = content.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no JSON object in model response")
    return json.loads(s[a:b + 1])


# --- deterministic fallback (no LLM) — broad, multi-domain so any JD parses ---
_SKILL_VOCAB = [
    # AI / ML / IR
    "embeddings", "sentence transformers", "bge", "e5", "retrieval", "dense retrieval",
    "semantic search", "hybrid search", "vector database", "vector search", "faiss",
    "pinecone", "qdrant", "milvus", "weaviate", "opensearch", "elasticsearch", "ranking",
    "learning to rank", "recommendation", "recommender", "recsys", "information retrieval",
    "nlp", "llm", "fine-tuning", "lora", "qlora", "peft", "rag", "xgboost", "ndcg", "mrr",
    "a/b test", "pytorch", "tensorflow", "computer vision", "deep learning", "transformers",
    # frontend
    "react", "angular", "vue", "vue.js", "next.js", "typescript", "javascript", "css",
    "html", "tailwind", "redux", "webpack", "figma", "design systems", "accessibility",
    "ui", "ux", "frontend",
    # backend / platform
    "java", "spring", "spring boot", "node.js", "go", "golang", "rust", "c++", ".net",
    "graphql", "rest", "rest apis", "microservices", "grpc", "kafka", "redis", "postgresql",
    "mysql", "mongodb", "backend", "api",
    # data / devops / cloud / mobile / qa / security
    "python", "sql", "spark", "airflow", "dbt", "snowflake", "databricks", "etl",
    "data pipelines", "hadoop", "kubernetes", "docker", "terraform", "aws", "azure", "gcp",
    "ci/cd", "devops", "android", "kotlin", "ios", "swift", "flutter", "selenium",
    "cypress", "security", "penetration testing",
]
_TITLE_VOCAB = ["frontend", "front end", "backend", "back end", "full stack", "fullstack",
                "machine learning", "ml", "ai", "artificial intelligence", "data science",
                "data scientist", "data engineer", "data analyst", "devops", "sre", "mobile",
                "android", "ios", "qa", "test", "security", "cloud", "platform",
                "infrastructure", "ui", "ux", "designer", "product manager", "nlp", "search",
                "ranking", "recommendation", "web", "software", "engineer", "developer",
                "scientist", "architect", "analyst", "manager"]
_CITIES = ["noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
           "bengaluru", "bangalore", "chennai", "kolkata", "ahmedabad", "remote"]


def deterministic_parse(jd_text):
    t = jd_text.lower()
    yrs = re.findall(r"(\d+)\s*[-–to]+\s*(\d+)\s*year", t)
    emin, emax = (int(yrs[0][0]), int(yrs[0][1])) if yrs else (3, 12)
    core = [s for s in _SKILL_VOCAB if s in t]
    titles = [s for s in _TITLE_VOCAB if s in t]
    locs = [c for c in _CITIES if c in t]
    anti = []
    if "consult" in t or "services" in t:
        anti.append("consulting_only")
    if "research" in t and "production" in t:
        anti.append("research_only")
    role = (jd_text.splitlines()[0][:80].strip() if jd_text.strip() else "(from JD)")
    return {
        "role": role, "seniority": "senior" if "senior" in t else "mid",
        "experience_min": emin, "experience_max": emax,
        "core_skills": core[:18] or ["python"], "support_skills": [],
        "relevant_titles": titles[:12], "preferred_locations": locs,
        "country": "india" if "india" in t else "any", "anti_patterns": anti,
        "summary": "Requirements extracted deterministically from the JD text.",
        "_source": "deterministic",
    }


def normalize(cfg):
    """Coerce to the expected schema/types so downstream code is safe."""
    out = {k: cfg.get(k) for k in SCHEMA_KEYS}
    out["core_skills"] = [str(s).lower() for s in (out.get("core_skills") or [])]
    out["support_skills"] = [str(s).lower() for s in (out.get("support_skills") or [])]
    out["relevant_titles"] = [str(s).lower() for s in (out.get("relevant_titles") or [])]
    out["preferred_locations"] = [str(s).lower() for s in (out.get("preferred_locations") or [])]
    out["anti_patterns"] = [str(s).lower() for s in (out.get("anti_patterns") or [])]
    try:
        out["experience_min"] = float(out.get("experience_min") or 0)
        out["experience_max"] = float(out.get("experience_max") or 0)
    except (TypeError, ValueError):
        out["experience_min"], out["experience_max"] = 0.0, 0.0
    out["country"] = str(out.get("country") or "any").lower()
    out["_source"] = cfg.get("_source", "llm")
    return out


def parse_jd(jd_text, api_key=None, model=DEFAULT_MODEL):
    """Return a normalized JD config. Uses NVIDIA if a key is available, else deterministic."""
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if api_key:
        try:
            cfg = extract_json(call_nvidia(jd_text, api_key, model))
            cfg["_source"] = "nvidia:" + model
            return normalize(cfg)
        except Exception as e:  # noqa: BLE001 — fall back gracefully on any API/parse error
            sys.stderr.write(f"[parse_jd] NVIDIA call failed ({e}); using deterministic fallback\n")
    cfg = deterministic_parse(jd_text)
    cfg["_source"] = "deterministic"
    return normalize(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True, help="path to a job-description text file")
    ap.add_argument("--out", default="requirements_jd.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--mock", action="store_true", help="skip the API; deterministic only")
    args = ap.parse_args()

    jd_text = open(args.jd, encoding="utf-8").read()
    cfg = (deterministic_parse(jd_text) if args.mock
           else parse_jd(jd_text, api_key=args.api_key, model=args.model))
    cfg = normalize(cfg)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Parsed JD ({cfg['_source']}) -> {args.out}")
    print(f"  role={cfg['role']!r} exp={cfg['experience_min']}-{cfg['experience_max']} "
          f"core_skills={len(cfg['core_skills'])} locations={cfg['preferred_locations']}")


if __name__ == "__main__":
    main()
