from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from tools import local_graph_check, save_to_graph, search_tool, fetch_page_content
from brain import get_llm, get_mini_llm
import time

main_llm = get_llm()
mini_llm = get_mini_llm()

class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    source_type: str
    iterations: int
    gap: str  # what is missing — used for targeted web search

# Phrases the writer uses when local context is incomplete.
# Judge catches these and triggers a targeted web search rather than showing a half-answer.
_HEDGING_PHRASES = [
    "does not specify", "does not contain", "cannot confirm", "not mentioned",
    "not provided", "not detailed", "not available in the context",
    "cannot determine", "no specific", "context does not", "context doesn't",
    "not found in", "no information", "is not clear", "unclear from",
    "the provided context", "does not include", "not included in",
]

def analyst_node(state: AgentState):
    print("[ANALYST] Checking Vector Memory...")
    start = time.time()
    res, ctx = local_graph_check(state['query'])

    if res == "LOCAL_FOUND" and ctx.strip():
        elapsed = time.time() - start
        print(f"[ANALYST] Local data found in {elapsed:.2f}s")
        return {
            "source_type": "Local",
            "context": [ctx],
            "gap": "",
            "iterations": state.get("iterations", 0)
        }

    return {
        "source_type": "Web",
        "context": [],
        "gap": "",
        "iterations": state.get("iterations", 0)
    }

def research_node(state: AgentState):
    current_iter = state.get("iterations", 0) + 1
    print(f"[RESEARCHER] Search Attempt {current_iter}...")
    start = time.time()

    try:
        query_lower = state['query'].lower()
        has_year = any(yr in query_lower for yr in ["2024", "2025", "2026", "2027"])
        time_signals = {"latest", "recent", "current", "today", "now", "this year", "this week",
                        "this month", "score", "standings", "winner", "results", "update", "news"}
        is_time_sensitive = any(sig in query_lower for sig in time_signals) or has_year
        date_suffix = " April 2026" if is_time_sensitive and not has_year else ""

        try:
            gap = state.get("gap", "")
            prev_answer = state.get("answer", "")

            if gap and current_iter == 1:
                # Judge or analyst identified a specific gap — search for it directly
                rewrite_prompt = (
                    f"Convert to a focused web search query (8 words max). Output ONLY the query.\n\n"
                    f"Original question: {state['query']}\n"
                    f"Missing information: {gap}"
                )
            elif "PARTIAL_INFO" in prev_answer:
                rewrite_prompt = (
                    f"A previous search gave this incomplete answer:\n{prev_answer}\n\n"
                    f"The user asked: {state['query']}\n\n"
                    f"Generate a specific web search query (10 words max) to find the MISSING information. "
                    f"Focus on the gaps, not what was already found. Output ONLY the search query."
                )
            else:
                rewrite_prompt = (
                    f"Convert the following question into a concise web search query (10 words max). "
                    f"Remove instruction words like 'describe', 'provide', 'identify', 'explain'. "
                    f"Keep the core subject, key terms, and any dates or proper nouns. "
                    f"Output ONLY the search query, nothing else.\n\nQuestion: {state['query']}"
                )
            search_q = mini_llm.invoke(rewrite_prompt).content.strip().strip('"') + date_suffix
            print(f"[RESEARCHER] Search query: {search_q}")
        except Exception:
            search_q = state['query'] + date_suffix

        raw = search_tool.invoke({"query": search_q})

        if isinstance(raw, str):
            import json as _json
            try:
                results = _json.loads(raw)
            except Exception:
                results = [{"url": "Web", "content": raw}]
        else:
            results = raw

        valid_results = [r for r in results if r.get('content')]

        print("[RESEARCHER] Fetching full page content...")
        enriched = []
        fetched = 0
        for r in valid_results:
            url = r.get('url', '')
            snippet = r.get('content', '')
            entry = f"Source: {url}\nContent: {snippet}"
            if fetched < 5:
                page_text = fetch_page_content(url)
                if page_text:
                    entry += f"\nFull Content: {page_text}"
                    fetched += 1
            enriched.append(entry)
        full_text = "\n\n".join(enriched)

        elapsed = time.time() - start
        print(f"[RESEARCHER] Web search completed in {elapsed:.2f}s")

    except Exception as e:
        print(f"[RESEARCHER] Search failed: {e}")
        return {"iterations": current_iter}

    error_signals = ["error", "httperror", "client error", "server error", "status code"]
    full_text_lower = full_text.lower()
    has_real_content = len(full_text) > 200 and not any(sig in full_text_lower[:300] for sig in error_signals)

    if has_real_content:
        source_url = valid_results[0].get('url', 'Web') if valid_results else "Web"
        try:
            raw_sentences = []
            for r in valid_results:
                content = r.get('content', '').strip()
                if content:
                    raw_sentences.append(f"{content} [src: {r.get('url', '')}]")
            if raw_sentences:
                save_to_graph("\n".join(raw_sentences), source_url=source_url)
        except Exception:
            pass

        try:
            extract_prompt = (
                "Extract factual triplets in format [Subject] --(RELATION)--> [Object] from the text below.\n\n"
                f"<text>\n{full_text[:6000]}\n</text>"
            )
            triplets = mini_llm.invoke(extract_prompt).content
            save_to_graph(triplets, source_url=source_url)
        except Exception:
            pass

    existing_context = state.get("context", [])
    merged_context = existing_context + [full_text] if full_text else existing_context
    return {"context": merged_context, "source_type": "Web", "iterations": current_iter, "gap": ""}

def writer_node(state: AgentState):
    print("[WRITER] Drafting Answer...")
    context_str = "\n".join(state['context'])
    if len(context_str) > 24000:
        context_str = context_str[:24000]

    source_type = state.get("source_type", "Web")
    # Hybrid = web path but local context is already loaded (from analyst or previous local answer)
    has_local_context = any(
        "Internal:" in c or "| SOURCE:" in c
        for c in state.get("context", [])
    )
    is_hybrid = source_type == "Web" and has_local_context

    if source_type == "Local":
        sys_msg = SystemMessage(content=(
            "You are a factual research assistant. Answer using ONLY the provided context. "
            "State everything the context says about the topic clearly and completely. "
            "If a specific detail is absent, mention it briefly at the end — do not make it the focus of the answer. "
            "Add a 'Sources:' section at the end only if source URLs are present in the context. "
            "OUTPUT RULE: Write MISSING_INFO as the very first word ONLY if the context has absolutely "
            "zero facts about the subject. Otherwise never use it."
        ))
    elif is_hybrid:
        sys_msg = SystemMessage(content=(
            "You are a factual research assistant. The context contains TWO types of information: "
            "local/internal data AND fresh web search results. "
            "Combine BOTH to give one complete, well-cited answer. "
            "Use internal/local data for precise stored facts. "
            "Use web data to fill gaps, provide comparisons, forecasts, or recent information. "
            "Do not say 'the context does not contain' — synthesize everything into one complete answer. "
            "Always end with a 'Sources:' section listing all cited URLs and document names. "
            "OUTPUT RULE: Write MISSING_INFO as the very first word ONLY if the context has zero relevant facts."
        ))
    else:
        sys_msg = SystemMessage(content=(
            "You are a factual research assistant. Answer using ONLY the provided context. "
            "CRITICAL: Always compose the best possible answer from what is available. "
            "OUTPUT RULES — follow exactly:\n"
            "1. Zero facts about the topic: write MISSING_INFO as the very first word.\n"
            "2. Topic covered but missing a LARGE portion of what was asked: "
            "write PARTIAL_INFO as the very first word, then give the best answer from available facts.\n"
            "3. Minor gaps: answer directly, note limitations inline.\n"
            "Always end with a 'Sources:' section listing each URL on its own line."
        ))

    human_msg = HumanMessage(content=(
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION:\n{state['query']}"
    ))
    ans = main_llm.invoke([sys_msg, human_msg]).content
    return {"answer": ans}

def judge_node(state: AgentState):
    ans = state.get("answer", "")
    iters = state.get("iterations", 0)
    source_type = state.get("source_type", "Web")

    # For Local answers on the FIRST attempt only: catch hedging in the opening response
    # (first 400 chars). Complete answers state the main fact upfront without hedging.
    # Incomplete answers hedge in the first sentence. Checking only the opening prevents
    # false positives on qualifications/caveats at the end of otherwise complete answers.
    if source_type == "Local" and iters == 0:
        opening = ans[:400].lower()
        hedging_found = next((p for p in _HEDGING_PHRASES if p in opening), None)
        if hedging_found:
            print(f"[JUDGE] Local answer incomplete (detected: '{hedging_found}') — triggering targeted web search...")
            # Build a gap description from the query for targeted search
            try:
                gap_prompt = (
                    f"The following answer is incomplete:\n{ans}\n\n"
                    f"Original question: {state['query']}\n\n"
                    f"In one sentence, describe specifically what information is missing. Output ONLY that sentence."
                )
                gap = mini_llm.invoke(gap_prompt).content.strip()
            except Exception:
                gap = state['query']
            return {"iterations": iters, "answer": ans, "gap": gap, "source_type": "Web"}

    # For Web/Hybrid answers: existing PARTIAL_INFO / MISSING_INFO retry logic
    needs_retry = "MISSING_INFO" in ans or "PARTIAL_INFO" in ans
    if needs_retry and iters < 3:
        tag = "PARTIAL_INFO" if "PARTIAL_INFO" in ans else "MISSING_INFO"
        print(f"[JUDGE] {tag} — Triggering deeper search (attempt {iters + 1})...")
        return {"iterations": iters, "answer": ans, "gap": ""}

    if "PARTIAL_INFO" in ans:
        ans = ans.replace("PARTIAL_INFO\n", "").replace("PARTIAL_INFO", "").strip()

    if "MISSING_INFO" in ans:
        print("[JUDGE] Search exhausted. No sufficient information found.")
        ans = (
            "I was unable to find sufficient information to answer this question from available sources. "
            "This topic may require access to specialized documents, paywalled content, or databases "
            "not reachable through web search. Please try rephrasing the question or consulting the source directly."
        )

    # Save synthesized answer to graph so future runs retrieve a direct answer
    # instead of raw fragments that cause the writer to hedge again.
    if state.get("source_type") == "Web" and len(ans) > 50:
        try:
            save_to_graph(
                f"Q: {state['query']}\nA: {ans}",
                source_url="Synthesized"
            )
        except Exception:
            pass

    print("[JUDGE] Answer Approved.")
    return {"answer": ans, "iterations": 4}

builder = StateGraph(AgentState)
builder.add_node("analyst", analyst_node)
builder.add_node("research", research_node)
builder.add_node("write", writer_node)
builder.add_node("judge", judge_node)

builder.set_entry_point("analyst")
builder.add_conditional_edges("analyst", lambda x: "write" if x["source_type"] == "Local" else "research")
builder.add_edge("research", "write")
builder.add_edge("write", "judge")
builder.add_conditional_edges("judge", lambda x: "research" if x["iterations"] < 3 else END)

agent_app = builder.compile()
