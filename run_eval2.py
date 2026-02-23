import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Import your agent building blocks
from research_assistant_refactor_v16_disamb_flow_fix import (
    init_state,
    plan_node,
    search_node,
    fetch_node,
    extract_node,
    integrate_node,
    PatchApplier,
)

def pick_open_lead(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the highest-priority open lead (generic, no topic logic)."""
    leads = state.get("leads", []) or []
    open_leads = [l for l in leads if l.get("status", "open") == "open"]
    if not open_leads:
        return None
    # Sort: highest priority first; tiebreaker: shallower depth first, then newest
    open_leads.sort(
        key=lambda x: (
            float(x.get("priority", 0.0)),
            -float(x.get("depth", 0.0)),
            x.get("created_at", "")
        ),
        reverse=True
    )
    return open_leads[0]

# -------------------------
# Tavily search function
# -------------------------
def tavily_search_fn(query: str, max_results: int = 3):
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY")

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
    )

    out = []
    for r in resp.get("results", []):
        url = r.get("url")
        if not url:
            continue
        out.append(
            {
                "title": r.get("title", ""),
                "url": url,
                "snippet": (r.get("content") or "")[:1200],
                "published_at": r.get("published_date") or None,
                "provider": "tavily",
                "credibility": float(r.get("score", 0.5)),
            }
        )
    return out


# -------------------------
# Non-interactive disambiguation (generic)
# -------------------------
def disambiguate_auto(
    llm_reason,
    search_fn,
    state: Dict[str, Any],
    hint: Optional[str] = None,
    max_rounds: int = 2,
    k_results: int = 6,
) -> Dict[str, Any]:
    """
    Generic offline identity resolution:
    - search for disambiguation candidates
    - ask LLM to pick best candidate given (target + hint)
    - store canonical_name + chosen_candidate_id + anchors (seed_urls)
    """
    target = (state.get("target") or {}).get("raw_input") or ""

    # We only need this if the agent relies on canonical_name for good queries
    # If you already have canonical_name, keep it.
    if (state.get("target") or {}).get("canonical_name"):
        return state

    for _round in range(max_rounds):
        q = f"{target} disambiguation"
        if hint:
            q = f"{target} {hint} disambiguation"

        results = search_fn(q, max_results=k_results)

        # Build lightweight candidate list (URLs/titles/snips) for LLM
        # Group by URL domain heuristics and title uniqueness
        candidates = []
        for r in results:
            candidates.append(
                {
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "snippet": (r.get("snippet") or "")[:350],
                    "credibility": float(r.get("credibility", 0.5)),
                }
            )

        prompt = f"""
SYSTEM:
You are an identity disambiguation module.
Given a target name and optional hint, choose the best matching identity candidate from search results.
Return STRICT JSON only:
{{
  "canonical_name": string,
  "confidence": number,
  "seed_urls": [string, ...],
  "rationale": string
}}

RULES:
- Prefer candidates that clearly match the hint if provided.
- Prefer high-credibility sources.
- Prefer Wikipedia/official/primary profiles when available.
- If unsure, choose the best-supported option and lower confidence.

USER:
target: {target}
hint: {hint or ""}

candidates:
{json.dumps(candidates, ensure_ascii=False)}
""".strip()

        resp = llm_reason.invoke(prompt) if hasattr(llm_reason, "invoke") else llm_reason(prompt)
        text = getattr(resp, "content", resp)

        try:
            out = json.loads(text)
        except Exception:
            # very small repair: extract first {...}
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                out = json.loads(text[start : end + 1])
            else:
                out = {}

        canonical = (out or {}).get("canonical_name")
        if canonical:
            state["target"]["canonical_name"] = canonical
            # pseudo-id so downstream prints don't break
            state["target"]["chosen_candidate_id"] = state["target"].get("chosen_candidate_id") or "auto"
            seed_urls = (out or {}).get("seed_urls") or []
            if seed_urls:
                state["target"]["seed_urls"] = list(dict.fromkeys(seed_urls))
            state["target"].setdefault("hints", {})
            if hint:
                state["target"]["hints"]["disambiguation_hint"] = hint
            return state

        # If LLM couldn't pick, try again with no hint or stop
        hint = None

    # fallback: keep raw_input as canonical name
    state["target"]["canonical_name"] = state["target"]["raw_input"]
    state["target"]["chosen_candidate_id"] = state["target"].get("chosen_candidate_id") or "auto_fallback"
    return state


# -------------------------
# LangGraph loop (same structure as graph4)
# -------------------------
# def node_plan(state: Dict[str, Any]) -> Dict[str, Any]:
#     topics = state.get("topics", {}) or {}
#     incomplete = []
#     for name, t in topics.items():
#         if (t or {}).get("status") != "complete":
#             incomplete.append((name, t))
#     incomplete.sort(key=lambda nt: int((nt[1] or {}).get("priority", 999)))
#     topics_to_work = [name for name, _ in incomplete]

#     # store for integrate prompt prioritization
#     state["_topics_to_work"] = topics_to_work

#     if not topics_to_work and not (state.get("leads") or []):
#         state["_last_plan"] = {"queries": []}
#         return state

#     plan = plan_node(
#         state["llm_reason"],
#         state,
#         topics_to_work,
#         lead_to_work=None,
#         k=int(state["control"].get("k_queries", 6)),
#     )
#     state["_last_plan"] = plan
#     return state

def node_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    topics = state.get("topics", {}) or {}

    incomplete = []
    for name, t in topics.items():
        if t.get("status") != "complete":
            incomplete.append((name, t))

    # sort by declared topic priority if available
    incomplete.sort(key=lambda nt: int((nt[1] or {}).get("priority", 999)))
    topics_to_work = [name for name, _ in incomplete]

    # Risk-first scheduling (simple risk pass)
    if "risk_inconsistencies" in topics_to_work:
        topics_to_work.remove("risk_inconsistencies")
        topics_to_work.insert(0, "risk_inconsistencies")

    # expose to LLM for better planning/integration targeting
    state["_topics_to_work"] = topics_to_work

    lead_to_work = pick_open_lead(state)

    if not topics_to_work and not lead_to_work:
        state["_last_plan"] = {"queries": []}
        return state

    print("topics_to_work:", topics_to_work)
    if lead_to_work:
        print("lead_to_work:", lead_to_work.get("title"), "| p=", lead_to_work.get("priority"), "depth=", lead_to_work.get("depth"))

    plan = plan_node(
        state["llm_reason"],
        state,
        topics_to_work,
        lead_to_work,
        k=int(state["control"].get("k_queries", 6)),
    )
    state["_last_plan"] = plan

    try:
        qs = plan.get("queries", [])
        print("planned queries:", len(qs))
        for q in qs[:6]:
            print("  -", q.get("q") or q.get("query"), "|", q.get("intent"), "|", q.get("desired_field"))
    except Exception:
        pass

    return state


def node_search(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("_last_plan") or {"queries": []}
    new_eids = search_node(state["search_fn"], state, plan)
    state["_new_eids"] = new_eids
    return state


def node_fetch(state: Dict[str, Any]) -> Dict[str, Any]:
    fetch_node(state)
    return state


def node_extract(state: Dict[str, Any]) -> Dict[str, Any]:
    eids = state.get("_new_eids", [])
    new_claim_ids, _ = extract_node(state["llm_extract"], state, eids)
    state["_new_claim_ids"] = new_claim_ids
    return state


def node_integrate(state: Dict[str, Any]) -> Dict[str, Any]:
    patches = integrate_node(state["llm_reason"], state, state.get("_new_claim_ids", []))
    state["_patches"] = patches
    return state


def node_apply(state: Dict[str, Any]) -> Dict[str, Any]:
    state["applier"].apply(state, state.get("_patches", {}))
    state["control"]["_iters_done"] = int(state["control"].get("_iters_done", 0)) + 1
    return state


def should_continue(state: Dict[str, Any]) -> str:
    max_it = int(state["control"].get("max_iterations", 3))
    it = int(state["control"].get("_iters_done", 0))
    if it >= max_it:
        return "end"

    topics = state.get("topics", {}) or {}
    if topics and all((t.get("status") == "complete") for t in topics.values()):
        return "end"

    return "loop"


def build_graph():
    g = StateGraph(dict)
    g.add_node("plan", node_plan)
    g.add_node("search", node_search)
    g.add_node("fetch", node_fetch)
    g.add_node("extract", node_extract)
    g.add_node("integrate", node_integrate)
    g.add_node("apply", node_apply)

    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "fetch")
    g.add_edge("fetch", "extract")
    g.add_edge("extract", "integrate")
    g.add_edge("integrate", "apply")
    g.add_conditional_edges("apply", should_continue, {"loop": "plan", "end": END})
    return g.compile()


# -------------------------
# Report flattening + baseline scoring
# -------------------------
def flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return " ".join(flatten_text(x) for x in obj)
    if isinstance(obj, dict):
        return " ".join(flatten_text(v) for v in obj.values())
    return ""


def keyword_fact_hit(report_text: str, must: List[str], should: List[str]) -> Tuple[bool, float]:
    text = report_text.lower()
    must_ok = all(m.lower() in text for m in (must or []))
    if not must_ok:
        return False, 0.0
    if not should:
        return True, 1.0
    should_hits = sum(1 for s in should if s.lower() in text)
    return True, should_hits / max(1, len(should))


def compute_structural_metrics(state: Dict[str, Any], topic_specs_by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    topics = state.get("topics", {}) or {}
    evidence = state.get("evidence", []) or []
    ev_by_id = {e.get("id"): e for e in evidence if isinstance(e, dict) and e.get("id")}
    citations = state.get("_field_citations", {}) or {}

    out = {
        "topics": {},
        "overall": {
            "topics_complete": 0,
            "topics_total": 0,
            "field_coverage_avg": 0.0,
            "distinct_source_urls": len({e.get("url") for e in evidence if e.get("url")}),
        },
    }

    coverages = []
    complete = 0
    total = 0

    for tname, t in topics.items():
        total += 1
        spec = topic_specs_by_name.get(tname, {})
        req = spec.get("required_fields") or (t.get("required_fields") or [])
        pf = t.get("populated_fields") or {}
        missing = t.get("missing_fields") or []

        # field coverage
        have_count = 0
        for f in req:
            val = pf.get(f)
            if isinstance(val, list) and len(val) > 0:
                have_count += 1
        cov = have_count / max(1, len(req))
        coverages.append(cov)

        # source quality vs spec thresholds (based on field citations)
        t_cit = (citations.get(tname) or {})
        all_eids = []
        for f in req:
            all_eids.extend(t_cit.get(f) or [])
        all_eids = list(dict.fromkeys([x for x in all_eids if isinstance(x, str)]))

        min_cred = float(spec.get("min_credibility", 0.0) or 0.0)
        good = []
        for eid in all_eids:
            ev = ev_by_id.get(eid) or {}
            try:
                if float(ev.get("credibility", 0.0)) >= min_cred:
                    good.append(eid)
            except Exception:
                pass
        good = list(dict.fromkeys(good))

        out["topics"][tname] = {
            "status": t.get("status"),
            "missing_fields": missing,
            "required_fields": req,
            "field_coverage": cov,
            "evidence_ids_total": len(all_eids),
            "good_evidence_ids": len(good),
            "min_credibility": min_cred,
        }

        if t.get("status") == "complete":
            complete += 1

    out["overall"]["topics_complete"] = complete
    out["overall"]["topics_total"] = total
    out["overall"]["field_coverage_avg"] = sum(coverages) / max(1, len(coverages))
    return out


# -------------------------
# LLM Judge
# -------------------------
def llm_judge_case(llm_judge, report_text: str, expected_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = f"""
SYSTEM:
You are an evaluator. You will judge whether the agent report supports each expected fact.
Return STRICT JSON ONLY:
{{
  "facts": [
    {{
      "id": string,
      "covered": boolean,
      "confidence": number,
      "evidence_in_report": string
    }}
  ],
  "overall_score": number
}}

RULES:
- "covered" means the report contains enough info to reasonably support the fact.
- Prefer explicit mentions. If only implied, lower confidence.
- evidence_in_report should be a short quote-like excerpt (<= 25 words) copied from the report.

REPORT:
{report_text}

EXPECTED FACTS:
{json.dumps(expected_facts, ensure_ascii=False)}
""".strip()

    resp = llm_judge.invoke(prompt) if hasattr(llm_judge, "invoke") else llm_judge(prompt)
    text = getattr(resp, "content", resp)

    # parse
    try:
        out = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        out = json.loads(text[start : end + 1]) if start != -1 and end != -1 and end > start else {}

    # sanity
    facts = out.get("facts") if isinstance(out, dict) else None
    if not isinstance(facts, list):
        out = {"facts": [], "overall_score": 0.0}
    return out


# -------------------------
# Format report (simple)
# -------------------------
def format_report_for_eval(state: Dict[str, Any]) -> str:
    """
    Keep this simple + stable for eval. Uses populated_fields only.
    If you already have a better format_report in graph4, you can import it instead.
    """
    topics = state.get("topics", {}) or {}
    tgt = state.get("target", {}) or {}
    lines = []
    lines.append(f"TARGET: {tgt.get('canonical_name') or tgt.get('raw_input')}")
    for tname, t in topics.items():
        lines.append(f"\nTOPIC: {tname}")
        lines.append(f"status: {t.get('status')}")
        pf = t.get("populated_fields") or {}
        for k, v in pf.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# -------------------------
# Main eval runner
# -------------------------
def run_case(
    case: Dict[str, Any],
    llm_reason,
    llm_extract,
    llm_judge,
    allowed_topics: Optional[set] = None,
) -> Dict[str, Any]:
    state = init_state(case["target"])

    # Limit topics for eval to what you care about
    if allowed_topics:
        state["topics"] = {k: v for k, v in state.get("topics", {}).items() if k in allowed_topics}

    # Attach deps
    state["llm_reason"] = llm_reason
    state["llm_extract"] = llm_extract
    state["search_fn"] = tavily_search_fn
    state["applier"] = PatchApplier()

    # Control knobs
    state["control"]["max_iterations"] = int(case.get("max_iterations", 2))
    state["control"]["k_queries"] = int(case.get("k_queries", 6))

    # Offline disambiguation
    hint = case.get("disambiguation_hint")
    state = disambiguate_auto(llm_reason, tavily_search_fn, state, hint=hint, max_rounds=2)

    # Run graph
    app = build_graph()
    t0 = time.time()
    final_state = app.invoke(state)
    runtime_s = time.time() - t0

    report_text = format_report_for_eval(final_state)

    # Structural metrics
    # If your TOPIC_SPECS_BY_NAME is defined in the refactor file, import it and use it here.
    # For now, derive a simple spec from state topics as fallback.
    topic_specs_by_name = {}
    for tname, t in (final_state.get("topics") or {}).items():
        topic_specs_by_name[tname] = {
            "required_fields": (t.get("required_fields") or []),
            "min_credibility": 0.0
        }

    structural = compute_structural_metrics(final_state, topic_specs_by_name)

    # Baseline expected-field checks (keyword-ish)
    expected = case.get("expected") or {}
    expected_hits = {}
    for topic, fields in expected.items():
        pf = (((final_state.get("topics") or {}).get(topic) or {}).get("populated_fields") or {})
        blob = flatten_text(pf).lower()
        field_hits = {}
        for field, toks in (fields or {}).items():
            if not toks:
                field_hits[field] = None
            else:
                field_hits[field] = any(str(tok).lower() in blob for tok in toks)
        expected_hits[topic] = field_hits

    # Fact checks: baseline + LLM judge
    expected_facts = case.get("expected_facts") or []
    baseline_fact = []
    for f in expected_facts:
        ok, score = keyword_fact_hit(report_text, f.get("must_mention") or [], f.get("should_mention") or [])
        baseline_fact.append({"id": f.get("id"), "covered": ok, "partial_score": score})

    judge = llm_judge_case(llm_judge, report_text, expected_facts) if expected_facts else {"facts": [], "overall_score": 0.0}

    return {
        "case_id": case.get("id"),
        "target": case.get("target"),
        "canonical_name": (final_state.get("target") or {}).get("canonical_name"),
        "runtime_s": runtime_s,
        "structural": structural,
        "expected_field_hits": expected_hits,
        "baseline_fact_hits": baseline_fact,
        "llm_judge": judge,
        "report_text": report_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", type=str, default="evalset.json")
    ap.add_argument("--out", type=str, default="eval_results.json")
    ap.add_argument("--model_reason", type=str, default="gpt-4o")
    ap.add_argument("--model_extract", type=str, default="gpt-4o-mini")
    ap.add_argument("--model_judge", type=str, default="gpt-4o")
    args = ap.parse_args()

    llm_reason = ChatOpenAI(model=args.model_reason, temperature=0)
    llm_extract = ChatOpenAI(model=args.model_extract, temperature=0)
    llm_judge = ChatOpenAI(model=args.model_judge, temperature=0)

    with open(args.evalset, "r") as f:
        evalset = json.load(f)

    cases = evalset.get("cases", [])
    allowed_topics = {"identity", "professional_timeline", "risk_inconsistencies", "legal_regulatory"}

    results = []
    for c in cases[:1]: # running only the first case for quick testing; remove [:1] to run all cases
        print(f"\n=== Running {c.get('id')} :: {c.get('target')} ===")
        try:
            res = run_case(c, llm_reason, llm_extract, llm_judge, allowed_topics=allowed_topics)
        except Exception as e:
            res = {"case_id": c.get("id"), "target": c.get("target"), "error": str(e)}
        results.append(res)
        print(res)

    out = {"evalset_version": evalset.get("version"), "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote: {args.out}")
    # Print quick summary
    for r in results:
        if r.get("error"):
            print(f"- {r['case_id']}: ERROR {r['error']}")
            continue
        overall = (r.get("llm_judge") or {}).get("overall_score", 0.0)
        runtime = r.get("runtime_s", 0.0)
        cov = ((r.get("structural") or {}).get("overall") or {}).get("field_coverage_avg", 0.0)
        print(f"- {r['case_id']}: judge={overall} coverage={cov:.2f} runtime={runtime:.1f}s")


if __name__ == "__main__":
    main()