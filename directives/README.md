# Directives

SOPs in Markdown — Layer 1 of the architecture. Each directive tells the
orchestrator (the agent) how to accomplish one task using the deterministic
scripts in `execution/`.

## Writing a directive

Name the file for the task: `scrape_website.md`, `build_dossier.md`, etc.
Cover these sections:

```markdown
# <Task name>

## Goal
What this accomplishes, in one or two sentences.

## Inputs
What the agent needs before starting (args, files, sheet IDs, env vars).

## Tools
Which `execution/` script(s) to run, and how to call them.

## Steps
The ordered flow. Reference scripts, don't do the work inline.

## Outputs
Where the deliverable lands (Google Sheet/Slides URL, file path).

## Edge cases & learnings
API limits, retries, timing, gotchas. Update this as you learn.
```

Directives are living documents — improve them as you discover constraints,
but don't overwrite or delete one without asking the user first.
