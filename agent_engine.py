from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from tools import local_graph_check, save_to_graph, search_tool, fetch_page_content
from brain import get_llm, get_mini_llm

# Both now point to gpt-4o-mini after the model swap in brain.py
main_llm = get_llm()
mini_llm = get_mini_llm()

class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    source_type: str
    iterations: int

def analyst_node(state: AgentState):
    print("[ANALYST] Checking Vector Memory...")
    res, ctx = local_graph_check(state['query'])

    if res == "LOCAL_FOUND" and ctx.strip():
        return {"source_type": "Local", "context": [ctx], "iterations": state.get("iterations", 0)}

    return {"source_type": "Web", "context": [], "iterations": state.get("iterations", 0)}

def research_node(state: AgentState):
    current_iter = state.get("iterations", 0) + 1
    print(f"[RESEARCHER] Search Attempt {current_iter}...")

    # Step 1: Search — if this fails, nothing to return
    try:
        # Rewrite the user query into a clean keyword search query.
        # Natural language instructions ("Provide the specific...", "Describe the history of...")
        # confuse search engines and return irrelevant results.
        query_lower = state['query'].lower()
        has_year = any(yr in query_lower for yr in ["2024", "2025", "2026", "2027"])
        time_signals = {"latest", "recent", "current", "today", "now", "this year", "this week",
                        "this month", "score", "standings", "winner", "results", "update", "news"}
        is_time_sensitive = any(sig in query_lower for sig in time_signals) or has_year
        date_suffix = " April 2026" if is_time_sensitive and not has_year else ""

        try:
            prev_answer = state.get("answer", "")
            if "PARTIAL_INFO" in prev_answer:
                # Previous search was incomplete — generate a targeted query for the MISSING parts
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
            search_q = state['query'] + date_suffix  # fallback to original query

        raw = search_tool.invoke({"query": search_q})

        if isinstance(raw, str):
            import json as _json
            try:
                results = _json.loads(raw)
            except Exception:
                results = [{"url": "Web", "content": raw}]
        else:
            results = raw

        # Filter out empty results
        valid_results = [r for r in results if r.get('content')]

        # Enrich top 5 results by fetching full page content.
        # Serper only returns 1-2 sentence snippets; full pages give the writer
        # enough detail to produce complete answers.
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
    except Exception as e:
        print(f"[RESEARCHER] Search failed: {e}")
        return {"iterations": current_iter}

    # Step 2: Extract and save to graph — non-critical, never blocks the writer
    # Only save if we have real content (not HTTP errors or empty results)
    error_signals = ["error", "httperror", "client error", "server error", "status code"]
    full_text_lower = full_text.lower()
    has_real_content = len(full_text) > 200 and not any(sig in full_text_lower[:300] for sig in error_signals)

    if has_real_content:
        source_url = valid_results[0].get('url', 'Web') if valid_results else "Web"
        try:
            # Save raw snippets directly — these preserve the prose the writer needs
            # for complex/narrative questions. Triplets alone are too sparse.
            raw_sentences = []
            for r in valid_results:
                content = r.get('content', '').strip()
                if content:
                    raw_sentences.append(f"{content} [src: {r.get('url', '')}]")
            if raw_sentences:
                save_to_graph("\n".join(raw_sentences), source_url=source_url)
        except Exception:
            pass  # non-critical

        try:
            # Also save LLM-extracted triplets for precise factual lookups
            extract_prompt = (
                "Extract factual triplets in format [Subject] --(RELATION)--> [Object] from the text below.\n\n"
                f"<text>\n{full_text[:6000]}\n</text>"
            )
            triplets = mini_llm.invoke(extract_prompt).content
            save_to_graph(triplets, source_url=source_url)
        except Exception:
            pass  # non-critical

    # Append new results to any existing context from previous searches
    # so the writer always gets the full accumulated picture across retries
    existing_context = state.get("context", [])
    merged_context = existing_context + [full_text] if full_text else existing_context
    return {"context": merged_context, "source_type": "Web", "iterations": current_iter}

def writer_node(state: AgentState):
    print("[WRITER] Drafting Answer...")
    context_str = "\n".join(state['context'])
    # Cap context to stay within gpt-4o-mini's 8k token limit
    # (~24k chars leaves ~2k tokens for system msg + question + response)
    if len(context_str) > 24000:
        context_str = context_str[:24000]

    source_type = state.get("source_type", "Web")

    if source_type == "Local":
        sys_msg = SystemMessage(content=(
            "You are a factual research assistant. Answer the user's question using ONLY the provided context. "
            "Do not use any outside knowledge. "
            "If the context contains relevant facts — even partial ones — compose the best possible answer from them. "
            "If the answer is incomplete due to limited context, state what is known and note the limitation. "
            "If the context includes source URLs, add a 'Sources:' section at the end. If not, omit it entirely. "
            "OUTPUT RULE: Write the literal text MISSING_INFO (no other words, just that token) on the very first line "
            "ONLY if the context contains absolutely zero facts related to the topic. Otherwise never use it."
        ))
    else:
        sys_msg = SystemMessage(content=(
            "You are a factual research assistant. Answer the user's question using ONLY the provided context. "
            "Do not use any outside knowledge. "
            "CRITICAL: Even if context is thin, ALWAYS compose the best possible answer from what is available. "
            "Partial answers with stated limitations are far better than refusals. "
            "OUTPUT RULES — follow exactly:\n"
            "1. If the context is completely empty or has zero facts related to the topic: "
            "write MISSING_INFO as the very first word, nothing before it.\n"
            "2. If the context covers the topic but is missing a LARGE portion of what was asked "
            "(e.g. asked for 10 facts but only 2-3 are present): "
            "write PARTIAL_INFO as the very first word on its own line, then give the best answer from available facts.\n"
            "3. For minor gaps or thin-but-relevant context: answer directly, note limitations inline — "
            "do NOT use PARTIAL_INFO or MISSING_INFO.\n"
            "Always end with a 'Sources:' section listing each URL from the context on its own line."
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

    needs_retry = "MISSING_INFO" in ans or "PARTIAL_INFO" in ans

    if needs_retry and iters < 3:
        tag = "PARTIAL_INFO" if "PARTIAL_INFO" in ans else "MISSING_INFO"
        print(f"[JUDGE] {tag} — Triggering deeper search (attempt {iters + 1})...")
        return {"iterations": iters}  # keep in the research loop

    # Strip PARTIAL_INFO marker from final answer before showing to user
    if "PARTIAL_INFO" in ans:
        ans = ans.replace("PARTIAL_INFO\n", "").replace("PARTIAL_INFO", "").strip()

    # If MISSING_INFO persists after all retries, writer found truly nothing —
    # strip the marker and present a clean "not found" message.
    if "MISSING_INFO" in ans:
        print("[JUDGE] Search exhausted. No sufficient information found.")
        ans = (
            "I was unable to find sufficient information to answer this question from available sources. "
            "This topic may require access to specialized documents, paywalled content, or databases "
            "not reachable through web search. Please try rephrasing the question or consulting the source directly."
        )

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