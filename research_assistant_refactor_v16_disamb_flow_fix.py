
"""LLM-native Deep Research Agent (refactor)

Key design:
- Node1 PLAN: LLM reason model returns STRICT JSON plan with query objects.
- Node2 SEARCH: tool/API executes queries; stores evidence objects.
- Node3 EXTRACT: LLM extract model turns each evidence into entities + atomic claims w/ citations.
- Node4 INTEGRATE: LLM reason model emits STRICT JSON PATCHES to update topics/graph/leads/risks.
- Node5 APPLY: deterministic PatchApplier validates schema + evidence_ids + dedupes + writes state.
- Node6 SYNTHESIZE: LLM reason model composes final report w/ citations.

This file is intentionally opinionated + generic: NO topic-specific extraction or hard-coded rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json, re, uuid

# -----------------------------
# TOPIC SPECS (structure only)
# -----------------------------
# TOPIC_SPECS: List[Dict[str, Any]] = [
#     {"name": "identity", "priority": 1, "required_fields": ["name_variants", "associated_orgs", "general_location"]},
#     {"name": "professional_timeline", "priority": 2, "required_fields": ["roles"]},
#     # {"name": "corporate_affiliations", "priority": 8, "required_fields": ["org_roles", "registry_hits"]},
#     # {"name": "legal_regulatory", "priority": 4, "required_fields": ["legal_involvements"]},
#     # {"name": "network_connections", "priority": 5, "required_fields": ["key_relationships"]},
#     # {"name": "financial_signals", "priority": 6, "required_fields": ["transactions_or_funding"]},
#     # {"name": "reputation_media", "priority": 7, "required_fields": ["press_hits", "controversies"]},
#     #{"name": "risk_inconsistencies", "priority": 3, "required_fields": ["risk_flags", "inconsistencies"]},
#     {
#         "name": "risk_inconsistencies",
#         "priority": 3,
#         "required_fields": ["risk_flags", "inconsistencies"],
#         # evidence quality gate (generic)
#         "min_sources": 2,
#         "min_credibility": 0.60,
#         "strong_source_credibility": 0.85,
#         "strong_sources_override": 1,
#     }
# ]
TOPIC_SPECS: List[Dict[str, Any]] = [

    # 1️⃣ Identity Resolution
    {
        "name": "identity",
        "priority": 1,
        "required_fields": [
            "name_variants",
            "date_of_birth",
            "nationalities",
            "general_location",
            "associated_orgs",
        ],
        "min_sources": 1,
        "min_credibility": 0.70,
    },

    # 2️⃣ Professional History
    {
        "name": "professional_timeline",
        "priority": 2,
        "required_fields": [
            "roles",
            "employers",
            "board_positions",
            "education",
            "role_dates"
        ],
        "min_sources": 2,
        "min_credibility": 0.60,
    },

    # 3️⃣ Legal & Regulatory Records
    {
        "name": "legal_regulatory",
        "priority": 3,
        "required_fields": [
            "criminal_records",
            "civil_cases",
            "regulatory_actions",
            "sanctions_list_mentions",
            "court_involvement"
        ],
        "min_sources": 2,
        "min_credibility": 0.65,
        "strong_source_credibility": 0.85,
        "strong_sources_override": 1,
    },

    # 4️⃣ Corporate Registry Data
    # {
    #     "name": "corporate_affiliations",
    #     "priority": 8,
    #     "required_fields": [
    #         "directorships",
    #         "beneficial_ownerships",
    #         "company_status",
    #         "incorporation_jurisdictions",
    #         "registry_entries"
    #     ],
    #     "min_sources": 2,
    #     "min_credibility": 0.60,
    # },

    # # 5️⃣ Financial Public Records
    # {
    #     "name": "financial_records",
    #     "priority": 5,
    #     "required_fields": [
    #         "bankruptcy_filings",
    #         "insolvency_records",
    #         "tax_liens",
    #         "judgments",
    #         "asset_seizures"
    #     ],
    #     "min_sources": 2,
    #     "min_credibility": 0.65,
    #     "strong_source_credibility": 0.85,
    #     "strong_sources_override": 1,
    # },

    # # 6️⃣ Media & Reputation
    # {
    #     "name": "reputation_media",
    #     "priority": 6,
    #     "required_fields": [
    #         "press_mentions",
    #         "negative_media",
    #         "allegations",
    #         "investigations",
    #         "public_statements"
    #     ],
    #     "min_sources": 2,
    #     "min_credibility": 0.55,
    #     "strong_source_credibility": 0.85,
    #     "strong_sources_override": 1,
    # },

    # # 7️⃣ Network & Associations
    # {
    #     "name": "network_connections",
    #     "priority": 7,
    #     "required_fields": [
    #         "key_associates",
    #         "political_connections",
    #         "high_risk_associations",
    #         "shared_directorships",
    #         "cross_border_links"
    #     ],
    #     "min_sources": 2,
    #     "min_credibility": 0.60,
    # },

    # 8️⃣ Risk-Relevant Inconsistencies (Searchable Conflicts Only)
    {
        "name": "risk_inconsistencies",
        "priority": 4,
        "required_fields": [
            "identity_conflicts",
            "employment_conflicts",
            "ownership_conflicts",
            "timeline_gaps",
            "misrepresentation_indicators"
        ],
        # stricter gate since this drives risk engine
        "min_sources": 2,
        "min_credibility": 0.60,
        "strong_source_credibility": 0.85,
        "strong_sources_override": 1,
    },
]



TOPIC_SPECS_BY_NAME = {t["name"]: t for t in TOPIC_SPECS}

# -----------------------------
# STATE INITIALIZATION
# -----------------------------
def initialize_topics(topic_specs: List[Dict[str, Any]] = TOPIC_SPECS) -> Dict[str, Any]:
    topics: Dict[str, Any] = {}
    for spec in topic_specs:
        req = list(spec["required_fields"])
        topics[spec["name"]] = {
            "name": spec["name"],
            "priority": spec["priority"],
            "status": "unstarted",   # unstarted|partial|complete|blocked
            "attempts": 0,
            "required_fields": req,
            "populated_fields": {f: [] for f in req},
            "missing_fields": req[:],
            "confidence": 0.0,
        }
    return topics

def init_state(raw_target: str) -> Dict[str, Any]:
    return {
        "target": {
            "raw_input": raw_target,
            "canonical_name": None,
            "short_description": None,
            "seed_urls": [],
            "hints": {},
            "disambiguation_candidates": [],
            "chosen_candidate_id": None,
        },
        "topics": initialize_topics(),
        "leads": [],          # [{id,title,type,priority,depth,seed_evidence_ids,status,reason}]
        "evidence": [],       # evidence objects (see Search node)
        "claims": [],         # atomic claims (see Extract node)
        "entities": [],       # typed entities
        "relationships": [],  # edges
        "risk_flags": [],     # risk flags
        "audit_log": [],
        "budgets": {"max_searches": 60, "searches_used": 0, "max_depth": 4},
        "control": {"new_evidence_ids": [], "new_claim_ids": [], "max_results_per_query": 5, "fetch_timeout": 8, "max_extract_per_iter": 3, "extract_raw_text_chars": 12000,
            "extract_batch_size": 10,},
        "_query_history": [],
        "_query_history_raw": [],
    }

# -----------------------------
# LOGGING
# -----------------------------
def log_step(state: Dict[str, Any], step: str, payload: Dict[str, Any]) -> None:
    entry = {"ts": datetime.utcnow().isoformat(), "step": step, "payload": payload}
    state.setdefault("audit_log", []).append(entry)

def normalize_query(q: str) -> str:
    """Normalize query text for dedupe (generic)."""
    return " ".join((q or "").strip().lower().split())


# -----------------------------
# STRICT JSON helpers (reuse)
# -----------------------------
def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    fence = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*)\s*```$", text, flags=re.DOTALL)
    return fence.group(1).strip() if fence else text

def _extract_first_json(text: str) -> Optional[str]:
    """Extract the first top-level JSON object/array substring."""
    text = _strip_code_fences(text)
    # find first { or [
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    if not starts:
        return None
    start = starts[0]
    stack = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                open_ch = stack.pop()
                if not stack:
                    return text[start:i+1]
    return None

def parse_strict_json(text: str) -> Any:
    """Parse JSON from model output. Accepts either pure JSON or JSON embedded in extra text."""
    text = text or ""
    raw = _strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        sub = _extract_first_json(raw)
        if sub:
            return json.loads(sub)
        raise

# -----------------------------
# Node 0: Work selection (generic)
# -----------------------------
def select_work(state: Dict[str, Any]) -> Dict[str, Any]:
    topics = state["topics"]
    leads = state.get("leads", [])
    max_depth = state["budgets"]["max_depth"]

    candidate_topics = [t for t in topics.values() if t["status"] not in ("complete", "blocked")]
    candidate_topics.sort(key=lambda t: t["priority"])
    selected_topics = [candidate_topics[0]["name"]] if candidate_topics else []

    def lead_valid(ld: Dict[str, Any]) -> bool:
        return (
            ld.get("status", "open") in ("open", "deprioritized")
            and ld.get("depth", 0) <= max_depth
            and ld.get("attempts", 0) < 3
            and ld.get("priority", 0.0) >= 0.2
        )

    candidate_leads = [ld for ld in leads if lead_valid(ld)]
    candidate_leads.sort(key=lambda ld: ld.get("priority", 0.0), reverse=True)
    selected_lead = candidate_leads[0] if candidate_leads else None

    if selected_lead is None and len(candidate_topics) > 1:
        selected_topics.append(candidate_topics[1]["name"])

    log_step(state, "select_work", {
        "topics_to_work": selected_topics,
        "lead_to_work_id": selected_lead.get("id") if selected_lead else None,
        "searches_used": state["budgets"]["searches_used"],
    })
    return {"topics_to_work": selected_topics, "lead_to_work": selected_lead}

# -----------------------------
# Node 1 — PLAN (LLM Reason)
# -----------------------------
PLAN_SCHEMA_HINT = {
    "queries": [
        {
            "query": "string",
            "intent": {"kind": "topic|lead", "topic": "optional topic name", "lead_id": "optional lead id"},
            "desired_field": "string (field in topic)",
            "priority": 0.0,
        }
    ],
    "disambiguation_question": "optional string"
}


def repair_json_with_llm(llm, bad_text: str, schema_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Ask an LLM to rewrite the given text into valid strict JSON.
    Returns parsed JSON (dict/list). Raises if still invalid.
    """
    hint = f"\n\nSchema hint:\n{schema_hint}\n" if schema_hint else ""
    prompt = (
        "You are a JSON repair function.\n"
        "Rewrite the following into VALID strict JSON. Output JSON only. "
        "Do not include markdown fences or commentary."
        f"{hint}\n\nTEXT:\n{bad_text}"
    )
    resp = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    txt = getattr(resp, "content", resp)
    return parse_strict_json(txt)



def build_state_summary(state: Dict[str, Any], topics_to_work: List[str], lead_to_work: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    topics = state["topics"]
    topic_summaries = []
    for tname in topics_to_work:
        t = topics.get(tname, {})
        topic_summaries.append({
            "name": tname,
            "status": t.get("status"),
            "missing_fields": t.get("missing_fields", []),
            "populated_fields_preview": {
                k: (v[:3] if isinstance(v, list) else v)
                for k, v in (t.get("populated_fields") or {}).items()
            },
        })

    lead_summary = None
    if lead_to_work:
        lead_summary = {k: lead_to_work.get(k) for k in ["id","title","type","priority","depth","seed_evidence_ids","entity_refs","reason"]}

    return {
        "target": state["target"],
        "topics_to_work": topic_summaries,
        "lead_to_work": lead_summary,
        "budgets_left": {
            "searches_left": max(0, state["budgets"]["max_searches"] - state["budgets"]["searches_used"]),
            "max_depth": state["budgets"]["max_depth"],
        },
    }

def plan_node(llm_reason, state: Dict[str, Any], topics_to_work: List[str], lead_to_work: Optional[Dict[str, Any]], k: int = 6) -> Dict[str, Any]:
    """Return a plan dict matching PLAN_SCHEMA_HINT."""
    summary = build_state_summary(state, topics_to_work, lead_to_work)
    target_name = state["target"].get("canonical_name") or state["target"]["raw_input"]

    canonical = state.get("target", {}).get("canonical_name")

    anchors = state.get("target", {}).get("identity_anchors", [])

    seed_urls = state.get("target", {}).get("seed_urls", [])

    chosen_id = state.get("target", {}).get("chosen_candidate_id")


    identity_context = ""

    if chosen_id or canonical or anchors or seed_urls:

        identity_context = (

            "\nChosen identity constraint (MUST MATCH):\n"

            f"- canonical_name: {canonical}\n"

            f"- anchors: {anchors}\n"

            f"- seed_urls: {seed_urls}\n"

            "\nHard rules:\n"

            "- Generate queries ONLY about the chosen identity.\n"

            "- If a name collision is likely (e.g., LinkedIn business profile), craft queries that explicitly disambiguate (sport/club/national team, DOB, location).\n"

            "- Prefer authoritative sources consistent with the chosen identity.\n"

            "- Do NOT pursue candidates that conflict with anchors.\n"

        )


    prompt = f"""
{identity_context}
SYSTEM:
You are the PLANNING module of a due-diligence OSINT agent.
You produce the NEXT search plan ONLY.
Return STRICT JSON only (no markdown, no prose). Follow this exact shape:
{json.dumps(PLAN_SCHEMA_HINT, indent=2)}

RULES:
- Provide up to {k} queries total, prioritizing high-signal sources.
- Queries MUST be consecutive: each should build on what we already know + what is missing.
- DO NOT repeat any query that has already been executed (see Query history below). Each query must be meaningfully different.
- Every query MUST target a specific missing field for the selected topic/lead. Set desired_field to that missing field name.
- Each query object must include:
  - query (string)
  - intent.kind = "topic" or "lead"
  - desired_field = the topic field you aim to fill (or "verification" if lead)
  - priority in [0,1]
- Avoid doxxing / private data (home address, phone, personal emails).
- Prefer official registries/filings, reputable news, company sites, professional profiles, court/agency dockets.
- If identity is ambiguous, include disambiguation_question and ALSO include at least 2 disambiguation queries.

USER:
Target: {target_name}
State summary (what's missing / what we have):
{json.dumps(summary, indent=2)}

Recent query history (do not repeat):
{json.dumps(state.get('_query_history_raw', [])[-25:], indent=2)}
""".strip()

    resp = llm_reason.invoke(prompt) if hasattr(llm_reason, "invoke") else llm_reason(prompt)
    text = getattr(resp, "content", resp)
    plan = parse_strict_json(text)

    # minimal validation + normalization
    if not isinstance(plan, dict) or "queries" not in plan:
        raise ValueError("Plan node must return a JSON object with 'queries'.")

    queries = plan.get("queries") or []
    if not isinstance(queries, list):
        raise ValueError("'queries' must be a list.")

    norm_queries = []
    for q in queries[:k]:
        if not isinstance(q, dict) or "query" not in q or "intent" not in q:
            continue
        query = str(q["query"]).strip()
        if not query:
            continue
        desired_field = str(q.get("desired_field") or "").strip()
        pr = q.get("priority", 0.5)
        try:
            pr = float(pr)
        except Exception:
            pr = 0.5
        pr = min(1.0, max(0.0, pr))
        intent = q["intent"] if isinstance(q["intent"], dict) else {"kind": str(q["intent"])}
        norm_queries.append({"query": query, "intent": intent, "desired_field": desired_field or "unknown", "priority": pr})

    plan["queries"] = norm_queries
    log_step(state, "plan", {"plan": plan})
    return plan

# -----------------------------
# Node 2 — SEARCH (tool/API)
# -----------------------------
def search_node(search_fn, state: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    """Executes plan queries via injected search_fn and appends evidence objects.

search_fn signature:
    search_fn(query: str) -> List[Dict]  where each result has: title,url,snippet,published_at,provider,credibility(optional)
"""
    new_eids: List[str] = []
    now = datetime.utcnow().isoformat()
    query_hist = state.setdefault("_query_history", [])
    query_hist_raw = state.setdefault("_query_history_raw", [])
    seen = set(query_hist)
    planned_seen = set()

    for qobj in sorted(plan.get("queries", []), key=lambda x: x.get("priority", 0.5), reverse=True):
        if state["budgets"]["searches_used"] >= state["budgets"]["max_searches"]:
            break

        q = qobj["query"]
        q_norm = normalize_query(q)

        # Skip duplicate queries (within-plan and across-run)
        if (q_norm in planned_seen) or (q_norm in seen):
            continue

        planned_seen.add(q_norm)
        seen.add(q_norm)
        query_hist.append(q_norm)
        query_hist_raw.append(q)

        results = search_fn(q) or []
        state["budgets"]["searches_used"] += 1

        max_r = int(state.get("control", {}).get("max_results_per_query", 5) or 5)
        for r in results[:max_r]:
            ev_id = f"ev_{uuid.uuid4().hex[:10]}"
            ev = {
                "id": ev_id,
                "query": q,
                "intent": qobj.get("intent"),
                "desired_field": qobj.get("desired_field"),
                "url": r.get("url"),
                "title": r.get("title"),
                "snippet": r.get("snippet"),
                "published_at": r.get("published_at"),
                "provider": r.get("provider"),
                "retrieved_at": now,
                "credibility": float(r.get("credibility", 0.5)) if r.get("credibility") is not None else 0.5,
                "raw_text": None,
                "content_type": None,
                "fetch_status": None,
                "fetch_error": None,
                "fetch_attempts": 0,
                "next_fetch_after": None,
                "extraction_version": None,

            }
            state["evidence"].append(ev)
            new_eids.append(ev_id)

    state["control"]["new_evidence_ids"] = new_eids
    log_step(state, "search", {"new_evidence_ids": new_eids, "searches_used": state["budgets"]["searches_used"]})
    return new_eids

# -----------------------------
# Node 2.5 — FETCH (generic retrieval)
# -----------------------------

MAX_RAW_TEXT_CHARS = 30000
MIN_GOOD_TEXT_CHARS = 1200
MAX_FETCH_ATTEMPTS = 3
FETCH_COOLDOWN_SECONDS = 600  # 10 minutes
EXTRACTION_VERSION = "trafilatura_v1+pymupdf_v1"

def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()

def _parse_iso(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None

def should_refetch(ev: Dict[str, Any], *, force: bool = False) -> bool:
    """Generic refetch policy: no topic logic.
    Refetch if:
      - force=True
      - never fetched
      - previously failed/blocked but attempts remain and cooldown passed
      - raw_text too short (bad extraction)
      - extraction_version changed (upgrade)
    """
    if force:
        return True

    status = ev.get("fetch_status")
    raw = ev.get("raw_text") or ""

    if not status:
        return True

    if status != "success":
        return True

    if len(raw) < MIN_GOOD_TEXT_CHARS:
        return True

    if ev.get("extraction_version") != EXTRACTION_VERSION:
        return True

    return False

def can_try_fetch(ev: Dict[str, Any]) -> bool:
    attempts = int(ev.get("fetch_attempts", 0) or 0)
    if attempts >= MAX_FETCH_ATTEMPTS:
        return False

    nfa = ev.get("next_fetch_after")
    if not nfa:
        return True

    dt = _parse_iso(nfa)
    if not dt:
        return True

    return datetime.utcnow() >= dt

def fetch_node(
    state: Dict[str, Any],
    evidence_ids: Optional[List[str]] = None,
    timeout: int = 8,
    force: bool = False,
) -> List[str]:
    """Fetches and extracts main text for evidence URLs.

    - HTML: downloads and extracts readable content (trafilatura if installed; fallback to raw HTML text).
    - PDF: downloads and extracts text (PyMuPDF if installed).

    Adds/updates fields on evidence:
      raw_text, content_type, fetched_at, fetch_status, fetch_error,
      fetch_attempts, next_fetch_after, extraction_version

    Returns list of evidence_ids successfully fetched this call.
    """
    fetched_ids: List[str] = []
    ev_index = {e.get("id"): e for e in state.get("evidence", []) if e.get("id")}

    if evidence_ids is None:
        evidence_iter = state.get("evidence", [])
    else:
        evidence_iter = [ev_index[eid] for eid in evidence_ids if eid in ev_index]

    # Lazy imports so the rest of the system works even if optional deps aren't installed.
    try:
        import requests  # type: ignore
    except Exception as e:
        log_step(state, "fetch", {"error": f"requests_not_installed: {e}"})
        return fetched_ids

    try:
        import trafilatura  # type: ignore
    except Exception:
        trafilatura = None

    try:
        import fitz  # type: ignore  # PyMuPDF
    except Exception:
        fitz = None

    for ev in evidence_iter:
        url = ev.get("url")
        if not url:
            continue

        if not should_refetch(ev, force=force):
            continue

        if not can_try_fetch(ev):
            continue

        ev["fetch_attempts"] = int(ev.get("fetch_attempts", 0) or 0) + 1
        ev["extraction_version"] = EXTRACTION_VERSION
        ev["fetched_at"] = datetime.utcnow().isoformat()

        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "DeepResearchAgent/1.0"},
            )
            resp.raise_for_status()

            ctype = (resp.headers.get("Content-Type") or "").lower()
            is_pdf = ("application/pdf" in ctype) or url.lower().endswith(".pdf")

            raw_text = ""
            if is_pdf:
                ev["content_type"] = "application/pdf"
                if fitz is None:
                    raise RuntimeError("pymupdf_not_installed")
                doc = fitz.open(stream=resp.content, filetype="pdf")

                parts: List[str] = []
                total = 0
                for page in doc:
                    t = page.get_text() or ""
                    if not t:
                        continue
                    parts.append(t)
                    total += len(t)
                    if total >= MAX_RAW_TEXT_CHARS:
                        break
                raw_text = "".join(parts)
            else:
                ev["content_type"] = "text/html"
                html = resp.text
                if trafilatura is not None:
                    raw_text = trafilatura.extract(html) or ""
                else:
                    raw_text = html

            raw_text = (raw_text or "")[:MAX_RAW_TEXT_CHARS].strip()
            ev["raw_text"] = raw_text

            if raw_text:
                ev["fetch_status"] = "success"
                ev.pop("fetch_error", None)
                ev.pop("next_fetch_after", None)
                fetched_ids.append(ev["id"])
            else:
                ev["fetch_status"] = "failed"
                ev["fetch_error"] = "empty_extraction"
                ev["next_fetch_after"] = (
                    datetime.utcnow() + timedelta(seconds=FETCH_COOLDOWN_SECONDS)
                ).isoformat()

        except Exception as e:
            ev["fetch_status"] = "failed"
            ev["fetch_error"] = str(e)
            ev["next_fetch_after"] = (
                datetime.utcnow() + timedelta(seconds=FETCH_COOLDOWN_SECONDS)
            ).isoformat()

        # if exhausted attempts, freeze permanently to avoid infinite loop
        if int(ev.get("fetch_attempts", 0) or 0) >= MAX_FETCH_ATTEMPTS and ev.get("fetch_status") != "success":
            ev["fetch_status"] = "blocked"

    log_step(state, "fetch", {"fetched_ids": fetched_ids})
    return fetched_ids

# -----------------------------
# Node 3 — EXTRACT (LLM Extract)
# -----------------------------
EXTRACT_SCHEMA_HINT = {
    "entities": [{"id": "ent_x", "type": "Person|Org|Location|URL|Other", "name": "string", "aliases": [], "attributes": {}}],
    "claims": [
        {
            "id": "clm_x",
            "subject": "string",
            "predicate": "string",
            "object": "string",
            "qualifiers": {"start_date": None, "end_date": None, "location": None},
            "confidence": 0.0,
            "evidence_ids": ["ev_..."]
        }
    ],
    "citations": ["ev_..."]
}

def extract_node(llm_extract, state: Dict[str, Any], evidence_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Batched extraction:
    - Groups evidence_ids into batches (state.control.extract_batch_size)
    - Calls the extractor LLM once per batch
    - Normalizes entities/claims into state using the same schema as v6
    """

    id_to_ev = {e["id"]: e for e in state.get("evidence", [])}
    new_claim_ids: List[str] = []
    new_entity_ids: List[str] = []

    if not evidence_ids:
        state["control"]["new_claim_ids"] = []
        log_step(state, "extract", {"new_claim_ids": [], "new_entity_ids": []})
        return [], []

    control = state.get("control", {})
    cap = int(control.get("extract_raw_text_chars", 12000) or 12000)
    batch_size = int(control.get("extract_batch_size", 3) or 3)
    batch_size = max(1, min(batch_size, 10))  # safety cap

    # Build batches
    for i in range(0, len(evidence_ids), batch_size):
        batch_eids = evidence_ids[i:i + batch_size]

        # Prepare evidence blocks (truncate raw_text for prompt only)
        evidence_blocks = []
        for eid in batch_eids:
            ev = id_to_ev.get(eid)
            if not ev:
                continue
            ev_for_prompt = dict(ev)
            if isinstance(ev_for_prompt.get("raw_text"), str):
                ev_for_prompt["raw_text"] = ev_for_prompt["raw_text"][:cap]
            evidence_blocks.append({"evidence_id": eid, "evidence": ev_for_prompt})

        if not evidence_blocks:
            continue

        prompt = f"""
SYSTEM:
You are the EXTRACTION module. You will extract typed entities and ATOMIC, GROUNDED claims from MULTIPLE evidence items.
Return STRICT JSON only.

OUTPUT SHAPE (STRICT):
{{
  "results": [
    {{
      "evidence_id": "ev_...",
      "entities": {json.dumps(EXTRACT_SCHEMA_HINT["entities"], indent=2)},
      "claims": {json.dumps(EXTRACT_SCHEMA_HINT["claims"], indent=2)}
    }}
  ]
}}

RULES:
- Process each result independently and set result.evidence_id to the corresponding evidence_id.
- Claims must be atomic (one fact per claim) and directly supported by the evidence.
- Use evidence.raw_text when present; otherwise snippet/title/url.
- Do not invent. If unclear, omit or lower confidence.
- Each claim.evidence_ids MUST be ["<that evidence_id>"] only.
- confidence is a float in [0,1].
- Output JSON ONLY (no markdown, no commentary).

USER:
Target: {state["target"].get("canonical_name") or state["target"]["raw_input"]}
Evidence batch:
{json.dumps(evidence_blocks, indent=2)}
""".strip()

        resp = llm_extract.invoke(prompt) if hasattr(llm_extract, "invoke") else llm_extract(prompt)
        text = getattr(resp, "content", resp)

        try:
            out = parse_strict_json(text)
        except Exception:
            repair_prompt = (
                "SYSTEM:\nYou are a JSON repair tool. Convert the following into VALID STRICT JSON. "
                "Output JSON ONLY (no markdown, no commentary).\n\n"
                "INPUT:\n" + str(text)
            )
            r2 = llm_extract.invoke(repair_prompt) if hasattr(llm_extract, "invoke") else llm_extract(repair_prompt)
            repaired_text = getattr(r2, "content", r2)
            out = parse_strict_json(repaired_text)

        if not isinstance(out, dict):
            continue

        results = out.get("results") or []
        if not isinstance(results, list):
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            eid = item.get("evidence_id")
            if not eid or eid not in id_to_ev:
                continue

            # entities
            for ent in item.get("entities") or []:
                if not isinstance(ent, dict):
                    continue
                ent.setdefault("id", f"ent_{uuid.uuid4().hex[:10]}")
                ent_id = ent["id"]
                state["entities"].append(ent)
                new_entity_ids.append(ent_id)

            # claims
            for clm in item.get("claims") or []:
                if not isinstance(clm, dict):
                    # Ignore non-dict claims (string, etc.) to preserve schema integrity
                    continue
                clm.setdefault("id", f"clm_{uuid.uuid4().hex[:10]}")
                clm.setdefault("evidence_ids", [eid])
                # enforce evidence_ids ONLY this eid
                clm["evidence_ids"] = [x for x in clm.get("evidence_ids", []) if x == eid] or [eid]
                # clamp confidence
                try:
                    c = float(clm.get("confidence", 0.5))
                except Exception:
                    c = 0.5
                clm["confidence"] = min(1.0, max(0.0, c))
                state["claims"].append(clm)
                new_claim_ids.append(clm["id"])

    state["control"]["new_claim_ids"] = new_claim_ids
    log_step(state, "extract", {"new_claim_ids": new_claim_ids, "new_entity_ids": new_entity_ids})
    return new_claim_ids, new_entity_ids

# -----------------------------
# Node 4 — INTEGRATE (LLM Reason -> PATCHES)
# -----------------------------
PATCH_SCHEMA_HINT = {
    "topic_updates": [
        {"topic": "identity", "field": "name_variants", "mode": "append|upsert", "value": {}, "evidence_ids": ["ev_..."], "confidence": 0.0, "dedupe_key": "string"}
    ],
    "relationship_updates": [
        {"src": "string", "rel": "string", "dst": "string", "evidence_ids": ["ev_..."], "confidence": 0.0, "start_date": None, "end_date": None, "dedupe_key": "string"}
    ],
    "lead_updates": [
        {"mode": "spawn|update", "lead": {"title": "string", "type": "string", "priority": 0.0, "depth": 0, "seed_evidence_ids": ["ev_..."], "entity_refs": [], "reason": "string"}}
    ],
    "risk_updates": [
        {"risk": {"type": "string", "severity": "low|med|high", "description": "string", "evidence_ids": ["ev_..."], "confidence": 0.0}}
    ],
    "stop_signals": [
        {"scope": "topic|lead|global", "id": "topic_name or lead_id or global", "status": "complete|blocked|continue", "reason": "string"}
    ]
}

def integrate_node(llm_reason, state: Dict[str, Any], new_claim_ids: List[str]) -> Dict[str, Any]:
    """
    Node 4 — INTEGRATE (LLM Reason -> PATCHES)

    Performance-focused: only passes (a) new claim ids + compact claim objects and
    (b) a compact topic summary (missing fields/status) + allowed fields by topic.
    """
    id_to_claim = {c.get("id"): c for c in state.get("claims", []) if isinstance(c, dict) and c.get("id")}
    new_claims_full = [id_to_claim[cid] for cid in new_claim_ids if cid in id_to_claim]

    PATCH_EXAMPLES = """
        Example A — fill professional_timeline.roles:
        {
        "topic_updates": [
            {
            "topic": "professional_timeline",
            "field": "roles",
            "mode": "append",
            "value": ["2020–present: Manchester United — Midfielder"],
            "evidence_ids": ["ev_123"],
            "confidence": 0.7,
            "dedupe_key": "pt_roles_manutd_2020_midfielder"
            }
        ],
        "relationship_updates": [],
        "lead_updates": [],
        "risk_updates": [],
        "stop_signals": []
        }
        """.strip()

    # Compact claims to reduce prompt size (avoid sending large / noisy keys)
    compact_claims = []
    for c in new_claims_full:
        compact_claims.append({
            "id": c.get("id"),
            "type": c.get("type") or c.get("claim_type"),
            "text": c.get("text") or c.get("claim") or c.get("statement"),
            "confidence": c.get("confidence", 0.5),
            "evidence_ids": c.get("evidence_ids", []),
            "entity_refs": c.get("entity_refs", []),
            "when": c.get("when") or c.get("date") or c.get("timestamp"),
        })

    # compress evidence snippets for grounding context
    id_to_ev = {e.get("id"): e for e in state.get("evidence", []) if isinstance(e, dict) and e.get("id")}
    ev_snips: Dict[str, Any] = {}
    for clm in compact_claims:
        for eid in clm.get("evidence_ids", []) or []:
            ev = id_to_ev.get(eid)
            if ev and eid not in ev_snips:
                ev_snips[eid] = {
                    "title": ev.get("title"),
                    "url": ev.get("url"),
                    "snippet": (ev.get("snippet") or "")[:350],
                }

    # Compact topic summary (status + missing fields)
    topic_status = {
        k: {
            "status": v.get("status"),
            "missing_fields": v.get("missing_fields"),
            "attempts": v.get("attempts"),
        }
        for k, v in (state.get("topics") or {}).items()
        if isinstance(v, dict)
    }

    # Allowed fields by topic (instead of full TOPIC_SPECS)
    allowed_fields = {name: (spec.get("required_fields") or []) for name, spec in TOPIC_SPECS_BY_NAME.items()}

    prompt = f"""
SYSTEM:
You are the INTEGRATION module. Update the agent state by producing PATCHES.
Return STRICT JSON only, matching this exact shape:
{json.dumps(PATCH_SCHEMA_HINT)}

CONSTRAINTS (STRICT):

1. ALL required_fields are LIST-TYPE fields.
   - You MUST treat every topic field as a list.
   - topic_update.value MUST ALWAYS be a LIST.
   - Even if adding only one item, wrap it in a list.

2. Each topic_update must include:
   - topic
   - field
   - mode ("append" unless replacing)
   - value (LIST)
   - evidence_ids (non-empty list)
   - confidence (0-1 float)
   - dedupe_key (stable string)

3. If a topic has missing_fields and any New claims support that field:
   - You MUST emit at least one topic_update for that field.
   - Do NOT only spawn leads.
   - Filling missing_fields has priority over spawning leads.

4. For professional_timeline.roles:
   - Prefer simple string summaries if structure is unclear.
   - Example value:
     ["2020–present: Manchester United — Midfielder"]
   - approximate ranges (e.g., “2010–2012”, “Unknown–2016”) and “as of YEAR” are acceptable.

5. For risk_inconsistencies fields:
   - risk_flags: short, factual risk indicators.
   - inconsistencies: factual discrepancies across sources.
   - If no strong risks found, you may add a low-confidence neutral flag
     IF supported by reviewed evidence.

6. Do NOT output empty topic_updates when missing_fields exist and claims support them.

Return STRICT JSON only.

USER:
Target: {state["target"].get("canonical_name") or state["target"]["raw_input"]}

Existing lead titles:
{json.dumps([l.get("title") for l in state.get("leads", []) if isinstance(l, dict)], ensure_ascii=False)}

Recent executed queries:
{json.dumps((state.get("_query_history_raw") or [])[-20:], ensure_ascii=False)}

Allowed fields by topic:
{json.dumps(allowed_fields, ensure_ascii=False)}

Current topic statuses:
{json.dumps(topic_status, ensure_ascii=False)}

New claim ids:
{json.dumps(new_claim_ids, ensure_ascii=False)}

New claims (compact):
{json.dumps(compact_claims, ensure_ascii=False)}

Evidence snippets:
{json.dumps(ev_snips, ensure_ascii=False)}

Topics currently being worked:
{json.dumps(state.get("_topics_to_work"), ensure_ascii=False)}

PATCH EXAMPLES:
{PATCH_EXAMPLES}

You MUST prioritize filling missing_fields for topics currently being worked.
If you are uncertain, still emit a best-effort low-confidence topic_update 
with the most likely value supported by evidence snippets.

After filling missing fields, spawn at least 1 lead for deeper verification if budget remains, 
but only if supported by evidence and not as a way to avoid filling missing fields 
and do these for each topic if possible. Each lead must have a clear reason and evidence support.
""".strip()

    print("TOPICS_TO_WORK:", state.get("_topics_to_work"))
    print("MISSING_FIELDS:", {k:v.get("missing_fields") for k,v in (state.get("topics") or {}).items()})
    print("SAMPLE_COMPACT_CLAIMS:", json.dumps(compact_claims[:3], indent=2)[:2000])
    print("SAMPLE_EVID_SNIPS:", json.dumps(list(ev_snips.items())[:2], indent=2)[:2000])

    resp = llm_reason.invoke(prompt) if hasattr(llm_reason, "invoke") else llm_reason(prompt)
    text = getattr(resp, "content", resp)
    try:
        patches = parse_strict_json(text)
    except Exception:
        # Repair malformed JSON with the reasoning model
        patches = repair_json_with_llm(llm_reason, text, schema_hint=PATCH_SCHEMA_HINT)


    if not isinstance(patches, dict):
        raise ValueError("Integrate node must return a JSON object.")

    log_step(state, "integrate", {"patches": patches})
    return patches

# -----------------------------
# Node 5 — APPLY PATCHES (deterministic + generic)
# -----------------------------
class PatchApplier:
    """Generic patch applier. No topic-specific logic."""

    def __init__(self, topic_specs_by_name: Dict[str, Dict[str, Any]] = TOPIC_SPECS_BY_NAME):
        self.topic_specs_by_name = topic_specs_by_name

    def _evidence_exists(self, state: Dict[str, Any], evidence_ids: List[str]) -> bool:
        have = {e["id"] for e in state.get("evidence", [])}
        return all(eid in have for eid in (evidence_ids or []))

    def _topic_field_valid(self, topic: str, field: str) -> bool:
        spec = self.topic_specs_by_name.get(topic)
        return bool(spec) and field in (spec.get("required_fields") or [])
    
    def _passes_evidence_gate(self, state: Dict[str, Any], topic_name: str) -> bool:
        spec = self.topic_specs_by_name.get(topic_name) or {}
        min_sources = spec.get("min_sources")
        min_cred = spec.get("min_credibility")
        if min_sources is None or min_cred is None:
            return True  # no gate
        
        strong_cred = float(spec.get("strong_source_credibility", 0.0) or 0.0)
        strong_override = int(spec.get("strong_sources_override", 1) or 1)

        # collect cited evidence ids for required fields
        req = (spec.get("required_fields") or [])
        fc = state.get("_field_citations", {}) or {}
        eids: List[str] = []
        for f in req:
            eids.extend((fc.get(topic_name, {}) or {}).get(f, []) or [])
        # de-dupe preserve order
        eids = list(dict.fromkeys([x for x in eids if isinstance(x, str)]))

        ev_by_id = {e.get("id"): e for e in (state.get("evidence") or []) if isinstance(e, dict) and e.get("id")}

        good = []
        strong = 0
        for eid in eids:
            ev = ev_by_id.get(eid) or {}
            try:
                c = float(ev.get("credibility", 0.0))
            except Exception:
                c = 0.0
            if c >= float(min_cred):
                good.append(eid)
            if strong_cred and c >= strong_cred:
                strong += 1
        
        # pass if enough good sources OR enough strong sources
        if strong_cred and strong >= strong_override:
            return True
        return len(set(good)) >= int(min_sources)

    def apply(self, state: Dict[str, Any], patches: Dict[str, Any]) -> Dict[str, Any]:
        # -------- topic updates --------
        for upd in patches.get("topic_updates") or []:
            if not isinstance(upd, dict):
                continue
            topic = upd.get("topic"); field = upd.get("field")
            if not topic or not field or not self._topic_field_valid(topic, field):
                continue
            evidence_ids = upd.get("evidence_ids") or []
            if not self._evidence_exists(state, evidence_ids):
                continue
            try:
                conf = float(upd.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = min(1.0, max(0.0, conf))

            # NEW CHANGE
            # --- record citations per topic/field (JSON-safe: lists only) ---
            fc = state.setdefault("_field_citations", {})  # topic -> field -> [evidence_ids]
            fc_topic = fc.setdefault(topic, {})
            fc_list = fc_topic.setdefault(field, [])
            MAX_EIDS_PER_FIELD = 10
            for eid in evidence_ids:
                if eid not in fc_list:
                    fc_list.append(eid)
            if len(fc_list) > MAX_EIDS_PER_FIELD:
                del fc_list[:-MAX_EIDS_PER_FIELD]
            ## END OF NEW CHANGE


            t = state["topics"][topic]
            arr = t["populated_fields"].setdefault(field, [])
            dedupe = t.setdefault("_dedupe", {}).setdefault(field, [])  # list for JSON-safe state
            dk = upd.get("dedupe_key")
            if dk is None:
                dk = json.dumps(upd.get("value", {}), sort_keys=True)[:200]
            if dk in dedupe:
                continue
            dedupe.append(dk)

            mode = (upd.get("mode") or "append").lower()
            if mode == "upsert" and isinstance(upd.get("value"), dict):
                # upsert by dedupe_key: replace existing item with same key if present
                replaced = False
                for i, item in enumerate(arr):
                    try:
                        item_dk = item.get("_dedupe_key")
                    except Exception:
                        item_dk = None
                    if item_dk == dk:
                        arr[i] = upd["value"]
                        replaced = True
                        break
                if not replaced:
                    #arr.append(upd["value"])
                    val = upd.get("value")
                    if mode == "append" and isinstance(val, list):
                        arr.extend(val)
                    else:
                        arr.append(val)
            else:
                #arr.append(upd.get("value"))
                val = upd.get("value")
                if mode == "append" and isinstance(val, list):
                    arr.extend(val)
                else:
                    arr.append(val)

            # annotate (generic) if dict
            if isinstance(arr[-1], dict):
                arr[-1].setdefault("evidence_ids", evidence_ids)
                arr[-1].setdefault("confidence", conf)
                arr[-1]["_dedupe_key"] = dk

            # missing/status/confidence
            if field in t.get("missing_fields", []):
                t["missing_fields"].remove(field)
            if t["status"] == "unstarted":
                t["status"] = "partial"
            # topic confidence recomputed deterministically after applying patches

        
        # -------- recompute topic meta (generic, deterministic) --------
        for topic_name, t in state.get("topics", {}).items():
            req = t.get("required_fields") or []
            if not req:
                continue
            missing = set(t.get("missing_fields") or [])
            coverage = (len(req) - len(missing)) / max(1, len(req))

            # max confidence across populated required fields (if present)
            max_item_conf = 0.0
            for field in req:
                for item in (t.get("populated_fields", {}).get(field) or []):
                    if isinstance(item, dict):
                        try:
                            max_item_conf = max(max_item_conf, float(item.get("confidence", 0.0)))
                        except Exception:
                            pass

            # Conservative confidence: cannot exceed coverage-based cap
            coverage_cap = 0.2 + 0.8 * coverage
            t["confidence"] = float(min(max_item_conf if max_item_conf > 0 else coverage_cap, coverage_cap))

            if not (t.get("missing_fields") or []):
                t["status"] = "complete"
            elif t.get("status") == "unstarted" and coverage > 0:
                t["status"] = "partial"

        # -------- relationship updates --------
        rel_dedupe = state.setdefault("_rel_dedupe", [])  # list for JSON-safe state
        for upd in patches.get("relationship_updates") or []:
            if not isinstance(upd, dict):
                continue
            evidence_ids = upd.get("evidence_ids") or []
            if not self._evidence_exists(state, evidence_ids):
                continue
            try:
                conf = float(upd.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = min(1.0, max(0.0, conf))

            dk = upd.get("dedupe_key")
            if dk is None:
                dk = f"{upd.get('src')}|{upd.get('rel')}|{upd.get('dst')}|{upd.get('start_date')}|{upd.get('end_date')}"
            if dk in rel_dedupe:
                # merge evidence into existing edge
                for e in state.get("relationships", []):
                    edk = e.get("_dedupe_key")
                    if edk == dk:
                        have = set(e.get("evidence_ids", []))
                        for eid in evidence_ids:
                            if eid not in have:
                                e.setdefault("evidence_ids", []).append(eid)
                        e["confidence"] = max(float(e.get("confidence", 0.0)), conf)
                        break
                continue

            rel_dedupe.append(dk)
            edge = {
                "id": f"rel_{uuid.uuid4().hex[:10]}",
                "src": upd.get("src"),
                "rel": upd.get("rel"),
                "dst": upd.get("dst"),
                "start_date": upd.get("start_date"),
                "end_date": upd.get("end_date"),
                "evidence_ids": evidence_ids,
                "confidence": conf,
                "_dedupe_key": dk,
            }
            state.setdefault("relationships", []).append(edge)

        # -------- lead updates --------
        lead_by_id = {ld["id"]: ld for ld in state.get("leads", []) if isinstance(ld, dict) and ld.get("id")}
        for upd in patches.get("lead_updates") or []:
            if not isinstance(upd, dict):
                continue
            mode = (upd.get("mode") or "spawn").lower()
            lead = upd.get("lead") or {}
            if not isinstance(lead, dict):
                continue
            if mode == "spawn":
                # validate evidence_ids
                seed_eids = lead.get("seed_evidence_ids") or []
                if seed_eids and not self._evidence_exists(state, seed_eids):
                    continue
                new_id = f"lead_{uuid.uuid4().hex[:10]}"
                lead_obj = {
                    "id": new_id,
                    "title": lead.get("title"),
                    "type": lead.get("type"),
                    "priority": float(lead.get("priority", 0.5)),
                    "depth": int(lead.get("depth", 0)),
                    "seed_evidence_ids": seed_eids,
                    "entity_refs": lead.get("entity_refs", []),
                    "reason": lead.get("reason"),
                    "status": lead.get("status", "open"),
                    "attempts": 0,
                    "created_at": datetime.utcnow().isoformat(),
                }
                state.setdefault("leads", []).append(lead_obj)
            elif mode == "update":
                lid = lead.get("id")
                if lid and lid in lead_by_id:
                    lead_by_id[lid].update({k: v for k, v in lead.items() if k != "id"})

        # -------- risk updates --------
        risk_dedupe = state.setdefault("_risk_dedupe", [])  # list for JSON-safe state
        for upd in patches.get("risk_updates") or []:
            risk = (upd.get("risk") or {}) if isinstance(upd, dict) else {}
            if not isinstance(risk, dict):
                continue
            eids = risk.get("evidence_ids") or []
            if eids and not self._evidence_exists(state, eids):
                continue
            try:
                conf = float(risk.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = min(1.0, max(0.0, conf))
            dk = json.dumps({"type": risk.get("type"), "desc": risk.get("description")}, sort_keys=True)[:250]
            if dk in risk_dedupe:
                continue
            risk_dedupe.append(dk)
            risk_obj = {
                "id": f"risk_{uuid.uuid4().hex[:10]}",
                "type": risk.get("type"),
                "severity": risk.get("severity"),
                "description": risk.get("description"),
                "evidence_ids": eids,
                "confidence": conf,
                "created_at": datetime.utcnow().isoformat(),
                "_dedupe_key": dk,
            }
            state.setdefault("risk_flags", []).append(risk_obj)

        # -------- stop signals --------
        for sig in patches.get("stop_signals") or []:
            if not isinstance(sig, dict):
                continue
            scope = sig.get("scope")
            status = sig.get("status")
            if scope == "topic":
                tname = sig.get("id")
                if tname in state.get("topics", {}) and status in ("complete", "blocked", "continue"):
                    state["topics"][tname]["status"] = "complete" if status == "complete" else ("blocked" if status == "blocked" else state["topics"][tname]["status"])
                    state["topics"][tname].setdefault("stop_reason", sig.get("reason"))
            elif scope == "lead":
                lid = sig.get("id")
                for ld in state.get("leads", []):
                    if ld.get("id") == lid and status in ("complete", "blocked", "continue"):
                        ld["status"] = "closed" if status == "complete" else ("blocked" if status == "blocked" else ld.get("status","open"))
                        ld.setdefault("stop_reason", sig.get("reason"))

        # -------- topic completion heuristic (generic) --------
        # Not a deterministic fact decision: just checks required_fields populated.
        for tname, t in state.get("topics", {}).items():
            missing = [f for f in t.get("required_fields", []) if not (t.get("populated_fields", {}).get(f) or [])]
            t["missing_fields"] = missing
            # if not missing and t.get("status") != "blocked":
            #     t["status"] = "complete"
            if not missing and t.get("status") != "blocked":
                if self._passes_evidence_gate(state, tname):
                    t["status"] = "complete"
                else:
                    t["status"] = "partial"
                    t["stop_reason"] = "Evidence gate not met (min_sources/min_credibility)."

        log_step(state, "apply_patches", {"counts": {
            "topic_updates": len(patches.get("topic_updates") or []),
            "relationship_updates": len(patches.get("relationship_updates") or []),
            "lead_updates": len(patches.get("lead_updates") or []),
            "risk_updates": len(patches.get("risk_updates") or []),
            "stop_signals": len(patches.get("stop_signals") or []),
        }})
        return state

# -----------------------------
# Node 6 — SYNTHESIZE (LLM Reason)
# -----------------------------
def synthesize_report_node(llm_reason, state: Dict[str, Any]) -> str:
    prompt = f"""
SYSTEM:
You are the REPORT module. Write a due-diligence style report grounded in evidence.
You MUST cite evidence by including evidence URLs inline (not evidence ids only).
Be explicit about confidence and gaps. Avoid speculation.

USER:
Target:
{json.dumps(state["target"], indent=2)}

Topics (populated_fields):
{json.dumps({k: v.get("populated_fields") for k,v in state["topics"].items()}, indent=2)}

Relationships:
{json.dumps(state.get("relationships", []), indent=2)}

Risk flags:
{json.dumps(state.get("risk_flags", []), indent=2)}

Evidence (id-> url/title):
{json.dumps({e["id"]: {"url": e.get("url"), "title": e.get("title")} for e in state.get("evidence", [])}, indent=2)}
""".strip()
    resp = llm_reason.invoke(prompt) if hasattr(llm_reason, "invoke") else llm_reason(prompt)
    return getattr(resp, "content", resp)

# -----------------------------
# Main loop (single iteration)
# -----------------------------
def run_iteration(state: Dict[str, Any], llm_reason, llm_extract, search_fn, applier: PatchApplier, k_queries: int = 6) -> Dict[str, Any]:
    work = select_work(state)
    plan = plan_node(llm_reason, state, work["topics_to_work"], work["lead_to_work"], k=k_queries)
    new_eids = search_node(search_fn, state, plan)
    # Node 2.5: fetch full page/PDF text for richer extraction
    fetch_node(state, evidence_ids=new_eids, timeout=int(state.get("control", {}).get("fetch_timeout", 8) or 8))
    max_extract = int(state.get("control", {}).get("max_extract_per_iter", 3) or 3)
    eids_for_extract = new_eids[:max_extract]
    new_claim_ids, _ = extract_node(llm_extract, state, eids_for_extract)
    patches = integrate_node(llm_reason, state, new_claim_ids)
    applier.apply(state, patches)
    return state



# =========================
# Disambiguation (HITL)
# =========================

def disambiguate_node(llm_reason, search_fn, state: Dict[str, Any], max_candidates: int = 6) -> Dict[str, Any]:
    """
    LLM-native identity disambiguation.
    Populates state["target"]["candidates"] and related fields.
    """
    target = state["target"].get("raw_input")
    if not target:
        return state

    # Run multiple broad queries to surface different identities (generic disambiguation strategy)
    hint = state.get("target", {}).get("disambiguation_hint")
    queries = [
        f"{target} wikipedia",
        f"{target} (disambiguation) wikipedia",
        f"{target} official profile",
    ]
    if hint:
        queries.insert(0, f"{target} {hint}")

    results = []
    seen_urls = set()
    for q in queries:
        try:
            res = search_fn(q, max_results=8)
        except TypeError:
            res = search_fn(q)
        for r in (res or [])[:8]:
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(r)
    items = []
    for r in results[:8]:
        items.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or r.get("snippet") or "")[:300],
        })

    state.setdefault("target", {})
    state["target"]["disambiguation_search_results"] = items

    prompt = f"""
You are resolving identity ambiguity for a research agent.

Target input: {target}

Given the search results below, identify DISTINCT candidate identities.

Return STRICT JSON ONLY in this format:

{{
  "candidates": [
    {{
      "candidate_id": "cand_1",
      "display_name": "...",
      "descriptor": "...",
      "seed_urls": ["..."],
      "anchors": ["..."],
      "confidence": 0.0
    }}
  ],
  "disambiguation_question": "optional"
}}

Rules:
- Produce 2-{max_candidates} candidates if possible.
- Candidates must be clearly distinguishable.
- Use authoritative seed_urls when available.
- confidence in [0,1].
- Output JSON only.

Search results:
{json.dumps(items, ensure_ascii=False, indent=2)}
"""

    resp = llm_reason.invoke(prompt) if hasattr(llm_reason, "invoke") else llm_reason(prompt)
    text = getattr(resp, "content", resp)

    try:
        out = parse_strict_json(text)
    except Exception:
        out = repair_json_with_llm(llm_reason, text)

    candidates = out.get("candidates", [])[:max_candidates]

    state.setdefault("target", {})
    state["target"]["candidates"] = candidates
    state["target"]["disambiguation_question"] = out.get("disambiguation_question")
    state["target"].setdefault("chosen_candidate_id", None)
    state["target"].setdefault("canonical_name", None)
    state["target"].setdefault("identity_anchors", [])
    state["target"].setdefault("seed_urls", [])

    return state


def choose_candidate_cli(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    CLI-based human selection of disambiguation candidate.
    """
    cands = state.get("target", {}).get("candidates", [])
    if not cands:
        print("No candidates found.")
        return state




def disambiguate_and_choose_cli(
    llm_reason,
    search_fn,
    state: Dict[str, Any],
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """
    Notebook-friendly HITL disambiguation loop.

    Flow:
    - init_state(name) -> state
    - Round i:
        - run disambiguate_node (optionally with a user-provided hint)
        - print candidate list + top search results used
        - user can:
            (1) select a candidate number -> locks identity and returns
            (2) provide a hint -> reruns next round
            (3) skip this round -> continue (no hint change)
            (4) press Enter to stop early
    - If user never selects (skips every round), auto-select highest-confidence candidate
      from the *last* round that produced candidates.

    This avoids anchoring on incorrect candidates by re-running searches with new hints,
    while still guaranteeing a chosen identity if candidates exist.
    """
    state.setdefault("target", {})
    state["target"].setdefault("disambiguation_hint", None)
    state["target"].setdefault("candidates", [])
    state["target"].setdefault("disambiguation_search_results", [])
    state["target"].setdefault("chosen_candidate_id", None)
    state["target"].setdefault("canonical_name", None)
    state["target"].setdefault("identity_anchors", [])
    state["target"].setdefault("seed_urls", [])

    last_candidates: List[Dict[str, Any]] = []

    for round_idx in range(1, max_rounds + 1):
        state = disambiguate_node(llm_reason, search_fn, state)

        candidates = state.get("target", {}).get("candidates", []) or []
        search_items = state.get("target", {}).get("disambiguation_search_results", []) or []

        print(f"\n--- Disambiguation round {round_idx}/{max_rounds} ---")

        if candidates:
            last_candidates = candidates
            print("\nCandidates:\n")
            for idx, c in enumerate(candidates, 1):
                dn = c.get("display_name", "")
                desc = c.get("descriptor", "")
                conf = c.get("confidence", None)
                print(f"{idx}) {dn}")
                if desc:
                    print(f"   - {desc}")
                if conf is not None:
                    print(f"   - confidence: {conf}")
                for u in (c.get("seed_urls") or [])[:2]:
                    print(f"     • {u}")
                for a in (c.get("anchors") or [])[:2]:
                    print(f"     • {a}")
                print()
        else:
            print("No candidates produced by the LLM.")
            if search_items:
                print("Top search results used:\n")
                for j, it in enumerate(search_items[:8], 1):
                    print(f"  {j}) {it.get('title','')}")
                    print(f"     {it.get('url','')}")
                    sn = (it.get('snippet') or '').strip()
                    if sn:
                        print(f"     {sn[:160]}")
                print()

        q = state["target"].get("disambiguation_question")
        if q:
            print("Question:", q)

        # Prompt user action
        while True:
            action = input(
                f"Select [1-{len(candidates)}], (h) hint, (s) skip, or Enter to stop: "
            ).strip().strip("'\"").lower()

            if action == "":
                # Stop early; auto-select below if needed
                round_idx = max_rounds  # force exit outer loop
                break

            if action in ("s", "skip"):
                # Continue to next round without changing hint
                break

            if action in ("h", "hint"):
                hint = input("Enter hint (e.g., 'Manchester United football', 'Portugal midfielder'): ").strip()
                if hint:
                    state["target"]["disambiguation_hint"] = hint
                break

            if action.isdigit():
                k = int(action)
                if 1 <= k <= len(candidates):
                    chosen = candidates[k - 1]
                    state["target"]["chosen_candidate_id"] = chosen.get("candidate_id")
                    state["target"]["canonical_name"] = chosen.get("display_name")
                    state["target"]["seed_urls"] = chosen.get("seed_urls", [])
                    state["target"]["identity_anchors"] = chosen.get("anchors", [])
                    print(f"\nChosen: {state['target']['canonical_name']}")
                    return state
                else:
                    print("Invalid selection.")
                    continue

            # If they typed something else, treat it as a hint (quick UX)
            # This matches your expectation that user can just type a hint.
            if action:
                state["target"]["disambiguation_hint"] = action
                break

        # If we forced stop early by setting round_idx=max_rounds above, outer loop will finish.

    # If user never selected, pick highest-confidence from last candidates list
    if not state["target"].get("chosen_candidate_id") and last_candidates:
        def _conf(c):
            try:
                return float(c.get("confidence", 0.0))
            except Exception:
                return 0.0

        best = sorted(last_candidates, key=_conf, reverse=True)[0]
        state["target"]["chosen_candidate_id"] = best.get("candidate_id")
        state["target"]["canonical_name"] = best.get("display_name")
        state["target"]["seed_urls"] = best.get("seed_urls", [])
        state["target"]["identity_anchors"] = best.get("anchors", [])
        print(f"\nAuto-selected (highest confidence): {state['target']['canonical_name']}")
    else:
        if not state["target"].get("chosen_candidate_id"):
            print("\nNo candidate selected and none available to auto-select.")

    return state



    print("\nDisambiguation candidates:\n")
    for idx, c in enumerate(cands, 1):
        print(f"{idx}) {c.get('display_name','')}")
        print(f"   - {c.get('descriptor','')}")
        print(f"   - confidence: {c.get('confidence')}")
        for u in (c.get("seed_urls") or [])[:2]:
            print(f"     • {u}")
        for a in (c.get("anchors") or [])[:2]:
            print(f"     • {a}")
        print()

    q = state["target"].get("disambiguation_question")
    if q:
        print("Question:", q)

    while True:
        choice = input(f"Choose candidate [1-{len(cands)}] (or 's' to skip): ").strip().strip("\'\"").lower()
        if choice == "s":
            print("Skipping disambiguation.")
            return state
        if choice.isdigit():
            k = int(choice)
            if 1 <= k <= len(cands):
                chosen = cands[k - 1]
                state["target"]["chosen_candidate_id"] = chosen.get("candidate_id")
                state["target"]["canonical_name"] = chosen.get("display_name")
                state["target"]["seed_urls"] = chosen.get("seed_urls", [])
                state["target"]["identity_anchors"] = chosen.get("anchors", [])
                print(f"\nChosen: {state['target']['canonical_name']}")
                return state
        print("Invalid input. Try again.")


def run_loop(
    state: Dict[str, Any],
    llm_reason,
    llm_extract,
    search_fn,
    applier: PatchApplier,
    *,
    max_iterations: int = 6,
    k_queries: int = 6,
    stall_limit: int = 2,
) -> Dict[str, Any]:
    """Runs multiple iterations with generic stopping conditions.

    Stops when:
      - max_iterations reached
      - search budget exhausted
      - state stalls (no new evidence/claims/relationships) for stall_limit iterations
      - stop_signals indicates done (if present)
    """
    stall = 0
    for _ in range(max_iterations):
        if state["budgets"]["searches_used"] >= state["budgets"]["max_searches"]:
            break

        before = (len(state.get("evidence", [])), len(state.get("claims", [])), len(state.get("relationships", [])))
        run_iteration(state, llm_reason, llm_extract, search_fn, applier, k_queries=k_queries)
        after = (len(state.get("evidence", [])), len(state.get("claims", [])), len(state.get("relationships", [])))

        if after == before:
            stall += 1
        else:
            stall = 0

        # Optional: integrate_node can set stop_signals in state via patches
                # Stop if all topics are either complete or blocked
        topics = state.get('topics', {}) or {}
        if topics and all(t.get('status') in ('complete', 'blocked') for t in topics.values()):
            break

        if stall >= stall_limit:
            break

    return state