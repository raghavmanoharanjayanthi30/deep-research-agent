# Deep Research MVP — Evidence-Grounded OSINT Agent (LangGraph + HITL + Eval)

A small but interview-ready **due-diligence / OSINT research agent** that:
1) resolves identity collisions with a **human-in-the-loop (HITL) disambiguation** step,
2) iteratively searches + extracts claims,
3) integrates claims into structured topic fields via **LLM-generated patches**,
4) applies patches deterministically with a **generic PatchApplier**,
5) generates a **final summary report with citations**, and
6) evaluates performance via an **offline evaluation harness** (structural metrics + LLM-as-judge).

This repo is scoped intentionally as an MVP: strong architecture + evaluation story, not “all topics solved.”

---

## Problem

Public web evidence is noisy:
- **Name collisions** (e.g., multiple “Bruno Fernandes”)
- Fragmented, conflicting sources
- High variance in source credibility
- Risk topics are especially prone to hallucinations or weak sources

The goal is a system that can:
- **converge** on structured findings with citations,
- **avoid wrong-identity evidence**, and
- **measure quality** via evaluation (not vibes).

---

## Architecture Overview

The agent is a LangGraph loop with clear node boundaries. Each node has a single job, enabling debugging, evaluation, and future scaling.

### Core Nodes (LangGraph)
- **Plan Node**
  - Purpose: decide *what to search next*.
  - Inputs: missing fields + open leads + identity anchors (if disambiguated) + query history.
  - Output: a strict JSON search plan (queries + intent + target field).
  - Key feature: avoids repeating queries using `_query_history_raw`.

- **Search Node**
  - Purpose: execute the plan using a search API (e.g., Tavily).
  - Adds evidence stubs (URL, snippet, credibility score, etc.) and dedupes.

- **Fetch Node**
  - Purpose: fetch richer content for extraction (HTML/PDF text when available).
  - This increases extraction quality vs snippets-only.

- **Extract Node**
  - Purpose: convert evidence text into **claims**.
  - Runs with a cheaper/faster model (e.g., `gpt-4o-mini`) to minimize cost.
  - Output: normalized claim objects tied to `evidence_ids`.

- **Integrate Node**
  - Purpose: “reasoning step” that maps claims → **PATCHES**.
  - Runs with a stronger model (e.g., `gpt-4o`) because it must:
    - fill missing fields,
    - avoid duplicate leads,
    - output strict JSON matching a schema.
  - Performance optimization: only passes **new_claim_ids + compact claims** and a compact topic summary, rather than the full state.

- **Apply Node (PatchApplier)**
  - Purpose: deterministic state update.
  - Applies `topic_updates`, `lead_updates`, `relationship_updates`, and `risk_updates`.
  - Records `_field_citations[topic][field] -> evidence_ids` so reporting can attach citations.
  - Recomputes topic status and confidence.

### Leads + Recursive Research
The integration step can spawn **open leads** (“verify X”, “deepen Y”), and the planner can prioritize those leads in subsequent iterations. This makes the system recursive instead of a single pass.

---

## Human-in-the-Loop Disambiguation (HITL)

Before the graph loop (demo mode), the system can run a disambiguation flow:
1) Search candidate identities for the raw target name
2) Show candidate list (name + short descriptor + URLs + confidence)
3) User selects a candidate or provides a hint to rerun search
4) The chosen candidate becomes the target with **identity anchors**:
   - canonical name
   - seed URLs / known orgs / hints

These anchors are passed into the **Plan Node** so the agent actively avoids wrong-identity evidence.

Why HITL?
- It prevents the highest-cost failure mode: researching the wrong person.
- It’s practical and interview-relevant: most real-world systems include human checks early.

---

## Offline Evaluation Pipeline (No HITL)

Eval runs must be non-interactive. For that:
- each eval persona can include a `disambiguation_hint`
- the eval runner uses that hint to auto-resolve identity (or selects top-confidence candidate if skipped)

This keeps evaluation reproducible and scalable.

---

## Evidence-Gated Reporting with Citations

The final summary report includes:
- per-topic populated fields,
- **sources per field** (URLs derived from `evidence_ids`),
- topic status and missing fields,
- top open leads (if any)

This enforces “show your work” and makes the output auditable.

---

## Risk Gating / Source Credibility Constraints

Risk findings can be misleading if sourced from weak pages.
So risk topics support a **minimum evidence quality gate**:
- require `min_sources` and `min_credibility` per topic
- allow override if at least `strong_sources_override` sources exceed `strong_source_credibility`

This doesn’t “decide truth.” It prevents the system from marking risk topics “complete” based on low-quality evidence.

---

## Multi-Model Orchestration

Typical configuration:
- `llm_extract`: cheaper model for extraction (fast, high volume)
- `llm_reason`: stronger model for planning + integration (needs precision and schema adherence)

This reduces cost while preserving correctness where it matters.

---

## Evaluation Metrics

The evaluation harness scores cases using:

### 1) Structural Metrics (deterministic)
- Field coverage per topic
- Topics completed / total
- Distinct sources used
- Evidence quality gate pass/fail (via credibility thresholds)
- Runtime per case

### 2) Expected Field Hits (baseline keyword matcher)
- For each topic field, does the output contain expected tokens?

### 3) LLM-as-Judge
- Given the report + expected facts, the judge scores:
  - which facts were covered,
  - confidence per fact,
  - overall quality score.

This catches semantic matches that keyword heuristics miss and gives a more realistic quality signal.

---

## How to Run

### Setup
```bash
pip install -r requirements.txt
export TAVILY_API_KEY="..."
export OPENAI_API_KEY="..."
