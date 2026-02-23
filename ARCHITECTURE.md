# Architecture — Evidence-Grounded OSINT Agent

This document describes the internal architecture of the Deep Research MVP, including:
- Graph structure
- Node responsibilities
- State model
- Multi-model orchestration
- Evidence gating
- Evaluation integration
- Design tradeoffs

The goal is clarity, debuggability, and interview-readiness.

---

# 1. High-Level Overview

The system is a **stateful LangGraph loop** that iteratively:

1. Plans what to search next
2. Searches the web
3. Fetches content
4. Extracts claims
5. Integrates claims into structured topics
6. Applies patches deterministically
7. Repeats until budget or completion

Core principle:

> The LLM reasons. The system applies deterministically.

---

# 2. Graph Structure

The agent is implemented using LangGraph with the following node sequence:

       ┌───────────┐
       │   PLAN    │
       └─────┬─────┘
             │
             ▼
       ┌───────────┐
       │  SEARCH   │
       └─────┬─────┘
             │
             ▼
       ┌───────────┐
       │   FETCH   │
       └─────┬─────┘
             │
             ▼
       ┌───────────┐
       │  EXTRACT  │
       └─────┬─────┘
             │
             ▼
       ┌───────────┐
       │ INTEGRATE │
       └─────┬─────┘
             │
             ▼
       ┌───────────┐
       │   APPLY   │
       └─────┬─────┘
             │
     should_continue?
        │         │
        ▼         ▼
      PLAN       END


The loop continues until:
- All topics are complete, OR
- Iteration budget is exhausted.

---

# 3. State Model

The entire system operates on a single mutable `state` dictionary.

### Core Keys
state = {
"target": {...},
"topics": {...},
"evidence": [...],
"claims": [...],
"leads": [...],
"relationships": [...],
"risk_flags": [...],
"control": {...},
"_field_citations": {...},
"_query_history_raw": [...]
}


---

# 4. Topic Schema

Each topic is defined in `TOPIC_SPECS`.

Example:
{
"name": "legal_regulatory",
"priority": 3,
"required_fields": [...],
"min_sources": 2,
"min_credibility": 0.65,
"strong_source_credibility": 0.85,
"strong_sources_override": 1
}


Each topic tracks:
{
"status": "unstarted" | "partial" | "complete" | "blocked",
"required_fields": [...],
"missing_fields": [...],
"populated_fields": {
field_name: [ list of entries ]
},
"confidence": float
}


All fields are **list-type** fields for consistency and extensibility.

---

# 5. Node Responsibilities

## PLAN

Purpose:
- Decide what to search next.

Inputs:
- Missing topic fields
- Open leads
- Identity anchors
- Query history

Outputs:
- JSON search plan:

{
"queries": [
{
"query": "...",
"intent": {"kind": "topic"},
"desired_field": "roles",
"priority": 0.8
}
]
}


Planner constraints:
- Avoid repeated queries
- Must target missing fields
- Consecutive refinement

---

## SEARCH

Purpose:
- Execute web searches via API (e.g., Tavily)

Adds:
- Evidence stubs:
{
"id": "...",
"url": "...",
"title": "...",
"snippet": "...",
"credibility": float
}


Deduplication:
- Prevent duplicate URLs
- Preserve source diversity

---

## FETCH

Purpose:
- Retrieve richer content from URLs
- Improve extraction quality beyond snippet-level

Optional optimization:
- Caching
- Domain filtering

---

## EXTRACT

Purpose:
- Convert raw evidence into structured claims

Model:
- Lower-cost LLM (e.g., `gpt-4o-mini`)

Output format:
{
"id": "...",
"text": "...",
"evidence_ids": [...],
"confidence": float
}


Claims are evidence-grounded by design.

---

## INTEGRATE

Purpose:
- Map claims → PATCHES

Model:
- Higher-quality reasoning model (e.g., `gpt-4o`)

Input is optimized:
- Only new claims
- Compact evidence snippets
- Missing field summary

Output (strict JSON):
{
"topic_updates": [...],
"lead_updates": [...],
"relationship_updates": [...],
"risk_updates": [...],
"stop_signals": [...]
}


Integration prioritizes:
- Filling missing fields
- Avoiding duplicate leads
- Evidence-backed updates only

---

## APPLY (PatchApplier)

Purpose:
- Deterministic state mutation

Responsibilities:
- Validate topic + field
- Validate evidence existence
- Enforce dedupe via `dedupe_key`
- Track `_field_citations`
- Recompute topic status + confidence
- Enforce risk credibility gates

Key design principle:

> The LLM suggests. The PatchApplier enforces.

This prevents:
- Schema drift
- Invalid fields
- Duplicate entries
- Hallucinated evidence IDs

---

# 6. Human-in-the-Loop Disambiguation

Interactive mode:

1. Search candidate identities
2. Present candidates with summaries + URLs
3. User selects correct entity
4. Store identity anchors:
   - canonical_name
   - seed URLs
   - associated orgs

These anchors constrain future planning to prevent wrong-person research.

Offline evaluation:
- Uses `disambiguation_hint` instead.

---

# 7. Risk Gating & Evidence Quality

Risk topics are gated by:

- `min_sources`
- `min_credibility`
- optional `strong_source_credibility` override

Example logic:

If:
- fewer than `min_sources` credible evidence items,
Then:
- topic cannot be marked complete.

This prevents:
- Weak blog posts closing risk topics
- Opinion pieces masquerading as fact

---

# 8. Evaluation Architecture

Evaluation is separate from the research graph.

Metrics include:

### Structural Metrics
- Field coverage per topic
- Topics completed / total
- Distinct sources
- Evidence credibility compliance
- Runtime

### Baseline Keyword Matching
- Simple must/should mention scoring

### LLM-as-Judge
- Semantic evaluation of fact coverage
- Returns per-fact coverage + confidence
- Produces overall quality score

This hybrid approach balances:
- Deterministic checks
- Semantic understanding

---

# 9. Multi-Model Orchestration

Typical configuration:

| Task        | Model        | Reason |
|------------|-------------|--------|
| Extraction | gpt-4o-mini | High volume, cheaper |
| Planning   | gpt-4o      | Needs precision |
| Integration| gpt-4o      | Strict JSON + reasoning |
| Judge      | gpt-4o      | Semantic scoring |

Benefits:
- Lower cost
- Faster runtime
- Precision where needed

---

# 10. Design Tradeoffs

## Chosen Tradeoffs

- Deterministic applier over LLM state mutation
- List-type fields for all topics
- Strict JSON patch schema
- Explicit credibility gates
- HITL for identity instead of fully automatic

## Not Implemented (Yet)

- Cross-source contradiction detection
- Domain reputation scoring
- Vector store memory
- Cost/token accounting
- Async fetch concurrency

---

# 11. System Guarantees

The system guarantees:

- No topic field is written without evidence_ids
- No evidence ID can be referenced if not stored
- No duplicate entries with same dedupe_key
- Risk topics cannot close without sufficient credible sources

---
