import argparse
import os
from typing import Any, Dict, List

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from research_assistant_refactor_v16_disamb_flow_fix import (
    init_state,
    plan_node,
    search_node,
    fetch_node,
    extract_node,
    integrate_node,
    PatchApplier,
    select_work,
    disambiguate_and_choose_cli,
)


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
                "snippet": (r.get("content") or "")[:800],
                "published_at": r.get("published_date") or None,
                "provider": "tavily",
                "credibility": float(r.get("score", 0.5)),
            }
        )
    return out


from typing import Any, Dict, List, Set

# def format_report(state: Dict[str, Any]) -> str:
#     field_citations = state.get("_field_citations", {}) or {}

#     def urls_for_topic_field(topic: str, field: str, limit: int = 5) -> List[str]:
#         eids = (field_citations.get(topic, {}) or {}).get(field, []) or []
#         return urls_from_evidence_ids(eids)[:limit]
#     topics = state.get("topics", {}) or {}
#     ev_by_id = {e.get("id"): e for e in state.get("evidence", [])}

#     def urls_from_evidence_ids(eids: List[str]) -> List[str]:
#         urls: List[str] = []
#         for eid in eids or []:
#             ev = ev_by_id.get(eid) or {}
#             u = ev.get("url")
#             if u:
#                 urls.append(u)
#         # de-dupe preserve order
#         seen: Set[str] = set()
#         out: List[str] = []
#         for u in urls:
#             if u not in seen:
#                 seen.add(u)
#                 out.append(u)
#         return out

#     def collect_evidence_ids(obj: Any, out: Set[str]) -> None:
#         """Recursively collect evidence_ids from nested dict/list structures."""
#         if obj is None:
#             return
#         if isinstance(obj, dict):
#             eids = obj.get("evidence_ids")
#             if isinstance(eids, list):
#                 for x in eids:
#                     if isinstance(x, str):
#                         out.add(x)
#             for v in obj.values():
#                 collect_evidence_ids(v, out)
#         elif isinstance(obj, list):
#             for x in obj:
#                 collect_evidence_ids(x, out)

#     def pretty_value(v: Any, max_len: int = 220) -> str:
#         s = repr(v)
#         if len(s) > max_len:
#             return s[: max_len - 3] + "..."
#         return s

#     lines: List[str] = []
#     lines.append("\n====================")
#     lines.append("FINAL SUMMARY REPORT")
#     lines.append("====================\n")

#     tgt = state.get("target", {}) or {}
#     lines.append("TARGET")
#     lines.append("------")
#     lines.append(f"- raw_input: {tgt.get('raw_input')}")
#     if tgt.get("canonical_name"):
#         lines.append(f"- canonical_name: {tgt.get('canonical_name')}")
#     if tgt.get("chosen_candidate_id"):
#         lines.append(f"- chosen_candidate_id: {tgt.get('chosen_candidate_id')}")
#     lines.append("")

#     for topic_name, topic in topics.items():
#         status = topic.get("status", "unknown")
#         missing = topic.get("missing_fields") or []
#         pf = topic.get("populated_fields") or {}

#         lines.append(topic_name.upper())
#         lines.append("-" * len(topic_name))

#         lines.append(f"- status: {status}")
#         if missing:
#             lines.append(f"- missing_fields: {missing}")

#         if not pf:
#             lines.append("- populated_fields: (none)")
#         else:
#             lines.append("- populated_fields:")
#             for k, v in pf.items():
#                 lines.append(f"  - {k}: {pretty_value(v)}")
#                 # NEW CHANGE
#                 src_urls = urls_for_topic_field(topic_name, k, limit=3)
#                 if src_urls:
#                     lines.append("    - sources:")
#                     for u in src_urls:
#                         lines.append(f"      - {u}")

#         eids: Set[str] = set()
#         collect_evidence_ids(pf, eids)
#         urls = urls_from_evidence_ids(list(eids))
#         if urls:
#             lines.append("  - sources:")
#             for u in urls[:8]:
#                 lines.append(f"    - {u}")

#         lines.append("")

#     lines.append("GAPS")
#     lines.append("----")
#     any_gaps = False
#     for topic_name, topic in topics.items():
#         missing = topic.get("missing_fields") or []
#         if missing:
#             any_gaps = True
#             lines.append(f"- {topic_name}: missing {missing}")
#     if not any_gaps:
#         lines.append("- (none)")
    
#     print(state.get("_field_citations", {}).keys())

#     return "\n".join(lines)
from typing import Any, Dict, List, Set

def format_report(state: Dict[str, Any]) -> str:
    topics = state.get("topics", {}) or {}
    ev_by_id = {e.get("id"): e for e in state.get("evidence", [])}

    def urls_from_evidence_ids(eids: List[str]) -> List[str]:
        urls: List[str] = []
        for eid in eids or []:
            ev = ev_by_id.get(eid) or {}
            u = ev.get("url")
            if u:
                urls.append(u)
        # de-dupe preserve order
        seen: Set[str] = set()
        out: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    field_citations = state.get("_field_citations", {}) or {}

    def urls_for_topic_field(topic: str, field: str, limit: int = 5) -> List[str]:
        eids = (field_citations.get(topic, {}) or {}).get(field, []) or []
        return urls_from_evidence_ids(eids)[:limit]

    def collect_evidence_ids(obj: Any, out: Set[str]) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            eids = obj.get("evidence_ids")
            if isinstance(eids, list):
                for x in eids:
                    if isinstance(x, str):
                        out.add(x)
            for v in obj.values():
                collect_evidence_ids(v, out)
        elif isinstance(obj, list):
            for x in obj:
                collect_evidence_ids(x, out)

    def pretty_value(v: Any, max_len: int = 220) -> str:
        s = repr(v)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    lines: List[str] = []
    lines.append("\n====================")
    lines.append("FINAL SUMMARY REPORT")
    lines.append("====================\n")

    tgt = state.get("target", {}) or {}
    lines.append("TARGET")
    lines.append("------")
    lines.append(f"- raw_input: {tgt.get('raw_input')}")
    if tgt.get("canonical_name"):
        lines.append(f"- canonical_name: {tgt.get('canonical_name')}")
    if tgt.get("chosen_candidate_id"):
        lines.append(f"- chosen_candidate_id: {tgt.get('chosen_candidate_id')}")
    lines.append("")

    for topic_name, topic in topics.items():
        status = topic.get("status", "unknown")
        missing = topic.get("missing_fields") or []
        pf = topic.get("populated_fields") or {}

        lines.append(topic_name.upper())
        lines.append("-" * len(topic_name))

        lines.append(f"- status: {status}")
        if missing:
            lines.append(f"- missing_fields: {missing}")

        if not pf:
            lines.append("- populated_fields: (none)")
        else:
            lines.append("- populated_fields:")
            for k, v in pf.items():
                lines.append(f"  - {k}: {pretty_value(v)}")
                src_urls = urls_for_topic_field(topic_name, k, limit=3)
                if src_urls:
                    lines.append("    - sources:")
                    for u in src_urls:
                        lines.append(f"      - {u}")

        # Optional: old generic source collection (may be empty if values are strings)
        eids: Set[str] = set()
        collect_evidence_ids(pf, eids)
        urls = urls_from_evidence_ids(list(eids))
        if urls:
            lines.append("  - sources:")
            for u in urls[:8]:
                lines.append(f"    - {u}")

        lines.append("")

    lines.append("GAPS")
    lines.append("----")
    any_gaps = False
    for topic_name, topic in topics.items():
        missing = topic.get("missing_fields") or []
        if missing:
            any_gaps = True
            lines.append(f"- {topic_name}: missing {missing}")
    if not any_gaps:
        lines.append("- (none)")

    return "\n".join(lines)


from typing import Optional

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

def has_incomplete_topics(state: Dict[str, Any]) -> bool:
    topics = state.get("topics", {}) or {}
    for _, t in topics.items():
        if t.get("status") != "complete":
            if t.get("missing_fields") or t.get("required_fields"):
                return True
    return False

def has_open_leads(state: Dict[str, Any]) -> bool:
    for l in (state.get("leads", []) or []):
        if l.get("status", "open") == "open":
            return True
    return False


# -------------------------
# LangGraph node wrappers
# -------------------------
# def apply_risk_pass_scheduler(state: Dict[str, Any], topics_to_work: List[str]) -> List[str]:
#     """
#     Force risk_inconsistencies to get at least one "pass" early so it doesn't stay unstarted.
#     Generic: doesn't assume any target.
#     """
#     topics = state.get("topics", {}) or {}
#     risk = topics.get("risk_inconsistencies") or {}

#     if "risk_inconsistencies" not in topics:
#         return topics_to_work

#     # how many early iterations we force risk
#     risk_pass_iters = int(state.get("control", {}).get("risk_pass_iters", 2))
#     it = int(state.get("control", {}).get("_iters_done", 0))

#     # If risk is already started/complete/blocked, don't force it.
#     risk_status = risk.get("status", "unstarted")
#     risk_missing = risk.get("missing_fields") or []

#     should_force = (
#         it < risk_pass_iters
#         and risk_status in ("unstarted",)  # keep strict; or include "partial" if you want
#         and (risk_missing or (risk.get("required_fields") or []))
#     )

#     if should_force and "risk_inconsistencies" not in topics_to_work:
#         # Put risk first (so planner allocates queries to it)
#         return ["risk_inconsistencies"] + topics_to_work

#     return topics_to_work

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
    print("new_eids:", len(new_eids))
    state["_new_eids"] = new_eids
    return state


def node_fetch(state: Dict[str, Any]) -> Dict[str, Any]:
    fetch_node(state)
    return state


def node_extract(state: Dict[str, Any]) -> Dict[str, Any]:
    eids = state.get("_new_eids", [])
    new_claim_ids, _ = extract_node(state["llm_extract"], state, eids)
    print("new_claims:", len(new_claim_ids))
    state["_new_claim_ids"] = new_claim_ids
    return state


def node_integrate(state: Dict[str, Any]) -> Dict[str, Any]:
    patches = integrate_node(state["llm_reason"], state, state.get("_new_claim_ids", []))
    state["_patches"] = patches

    # --- AUDIT: did we produce roles updates? ---
    tus = (patches or {}).get("topic_updates", []) or []
    roles_updates = [u for u in tus if u.get("topic") == "professional_timeline" and u.get("field") == "roles"]
    print(f"patch topic_updates={len(tus)} | roles_updates={len(roles_updates)}")
    if roles_updates:
        u0 = roles_updates[0]
        print("roles_update_sample keys:", list(u0.keys()))
        print("roles_update_sample evidence_ids:", u0.get("evidence_ids"))
        v = u0.get("value")
        print("roles_update_sample value_type:", type(v).__name__, "value_preview:", repr(v)[:200])
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

    # Continue if there's work left: incomplete topics OR open leads
    if has_incomplete_topics(state) or has_open_leads(state):
        return "loop"

    return "end"


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
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--interactive", action="store_true", help="Run HITL disambiguation before the graph loop")
    ap.add_argument("--max_iterations", type=int, default=3)
    ap.add_argument("--k_queries", type=int, default=6)
    ap.add_argument("--max_results_per_query", type=int, default=3)
    ap.add_argument("--max_extract_per_iter", type=int, default=4)
    ap.add_argument("--extract_raw_text_chars", type=int, default=4000)
    ap.add_argument("--fetch_timeout", type=int, default=10)
    args = ap.parse_args()

    llm_reason = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_extract = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    state = init_state(args.target)
    print("topics in state:", list(state["topics"].keys()))

    # MVP scope
    #allowed = {"identity", "professional_timeline"}
    allowed = {"identity", "professional_timeline", "legal_regulatory","risk_inconsistencies"}
    state["topics"] = {k: v for k, v in state["topics"].items() if k in allowed}

    # Attach deps
    state["llm_reason"] = llm_reason
    state["llm_extract"] = llm_extract
    state["search_fn"] = tavily_search_fn
    state["applier"] = PatchApplier()

    # Control knobs
    state["control"]["max_iterations"] = args.max_iterations
    state["control"]["k_queries"] = args.k_queries
    state["control"]["max_results_per_query"] = args.max_results_per_query
    state["control"]["max_extract_per_iter"] = args.max_extract_per_iter
    state["control"]["extract_raw_text_chars"] = args.extract_raw_text_chars
    state["control"]["fetch_timeout"] = args.fetch_timeout
    state["control"].setdefault("risk_pass_iters", 2)

    # HITL disambiguation gate (demo)
    if args.interactive:
        state = disambiguate_and_choose_cli(
            llm_reason=llm_reason,
            search_fn=tavily_search_fn,
            state=state,
            max_rounds=3,
        )

    app = build_graph()
    final_state = app.invoke(state)

    print("\nDone. Topic status:")
    for k, t in final_state.get("topics", {}).items():
        print(f"- {k}: {t.get('status')} | missing: {t.get('missing_fields')}")

    print(format_report(final_state))

    print("\nTop leads:")
    for l in sorted(final_state.get("leads", []), key=lambda x: float(x.get("priority", 0.0)), reverse=True)[:5]:
        print(f"- (p={l.get('priority')}) depth={l.get('depth')} status={l.get('status')} :: {l.get('title')}")


if __name__ == "__main__":
    main()