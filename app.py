import sys
import io
import os
import time
import asyncio
import tempfile
import chainlit as cl
from agent_engine import agent_app, classify_intent, handle_chat, extract_intent
from ingest import ingest_file, check_similarity_before_ingest
from brain import get_mini_llm

_mini_llm = get_mini_llm()

# Unambiguous Hinglish signals — words that appear in Hinglish but almost never in plain English
_HINGLISH_SIGNALS = {
    "kya", "hai", "hain", "karo", "batao", "bata", "kaise", "kyun", "kyunki",
    "nahi", "nai", "hoga", "hogi", "wala", "wali", "matlab", "samjhao",
    "likho", "dijiye", "bataiye", "chahiye", "aur", "mujhe", "humein",
    "iska", "uska", "yeh", "woh", "theek", "bilkul", "zaroor", "shukriya",
}

def _detect_hinglish(text: str) -> bool:
    """Returns True if the query contains clear Hinglish markers."""
    words = set(text.lower().split())
    return bool(words & _HINGLISH_SIGNALS)

def _translate_to_hinglish(answer: str) -> str:
    """Translate an English answer into natural Hinglish using mini_llm."""
    prompt = (
        "Translate the following answer into natural Hinglish — a mix of Hindi and English "
        "as spoken in everyday Indian conversation. Keep technical terms, numbers, percentages, "
        "proper nouns, and document names in English. Do NOT use pure Hindi or Devanagari script. "
        "Output only the translated answer, nothing else.\n\n"
        f"Answer:\n{answer}"
    )
    from langchain_core.messages import HumanMessage
    result = _mini_llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()

# ── Capture agent's print output to build the thought trace ──────────────────
class StdoutCapture:
    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self._buf = io.StringIO()
        return self._buf
    def __exit__(self, *args):
        sys.stdout = self._old

def parse_trace(log: str) -> list:
    steps = []
    for line in log.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('[ANALYST]'):
            steps.append({'node': 'Analyst',    'msg': line[len('[ANALYST]'):].strip()})
        elif line.startswith('[RESEARCHER]'):
            steps.append({'node': 'Researcher', 'msg': line[len('[RESEARCHER]'):].strip()})
        elif line.startswith('[SYNTHESIZER]'):
            steps.append({'node': 'Synthesizer','msg': line[len('[SYNTHESIZER]'):].strip()})
        elif line.startswith('[WRITER]'):
            steps.append({'node': 'Writer',     'msg': line[len('[WRITER]'):].strip()})
        elif line.startswith('[JUDGE]'):
            steps.append({'node': 'Judge',      'msg': line[len('[JUDGE]'):].strip()})
    return steps

# ── Welcome screen ────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=(
        "## Strategic Intelligence Assistant\n\n"
        "I combine your **private document vault** with **live web search** "
        "to deliver cited, verified answers — no hallucinations.\n\n"
        "Upload a **PDF, DOCX, or TXT** file to add it to your private vault instantly. "
        "Then ask me anything about it — or anything else."
    )).send()

# ── Single message handler — handles uploads + queries ───────────────────────
@cl.on_message
async def main(message: cl.Message):
    # ── Handle file uploads first ─────────────────────────────────────────────
    if message.elements:
        supported = ('.pdf', '.docx', '.txt')
        ingested = []
        skipped  = []

        for element in message.elements:
            if not element.name.lower().endswith(supported):
                skipped.append(element.name)
                continue

            await cl.Message(content=f"Analysing **{element.name}**...").send()

            try:
                suffix = os.path.splitext(element.name)[1]
                if element.path:
                    tmp_path = element.path  # Chainlit already wrote it to disk
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(element.content)
                        tmp_path = tmp.name

                # Check if exact same filename already exists in vault
                from tools import get_vault_documents
                existing_names = [d["file_name"] for d in get_vault_documents()]
                same_name_exists = element.name in existing_names

                if same_name_exists:
                    cl.user_session.set("pending_version", {
                        "tmp_path":      tmp_path,
                        "file_name":     element.name,
                        "similar_doc":   element.name,
                        "chainlit_owns": bool(element.path),
                    })
                    await cl.Message(content=(
                        f"**'{element.name}'** already exists in your vault.\n\n"
                        f"Do you want to save this as a new version? "
                        f"Reply **yes** to update, or **no** to keep the existing version."
                    )).send()
                    if not message.content.strip():
                        return
                    continue

                # Smart version detection — check similarity before ingesting
                similar_doc = await cl.make_async(check_similarity_before_ingest)(tmp_path, element.name)

                if similar_doc:
                    # Pause — ask user for confirmation
                    cl.user_session.set("pending_version", {
                        "tmp_path":     tmp_path,
                        "file_name":    element.name,
                        "similar_doc":  similar_doc,
                        "chainlit_owns": bool(element.path),
                    })
                    await cl.Message(content=(
                        f"**'{element.name}'** looks similar to an existing document in your vault: "
                        f"**'{similar_doc}'**.\n\n"
                        f"Is this an updated version of **'{similar_doc}'**? "
                        f"Reply **yes** to save it as a new version, or **no** to add it as a separate document."
                    )).send()
                    if not message.content.strip():
                        return
                    continue  # don't ingest yet — wait for confirmation

                # No similarity match — ingest normally
                chunks = await cl.make_async(ingest_file)(tmp_path, element.name)
                if not element.path:
                    os.unlink(tmp_path)

                if chunks > 0:
                    ingested.append(f"**{element.name}** — {chunks} chunks indexed")
                else:
                    ingested.append(f"**{element.name}** — already in vault, skipped")
            except Exception as e:
                ingested.append(f"**{element.name}** — failed: {e}")

        if ingested:
            lines = ["**Vault updated — ready to query:**\n"] + [f"- {r}" for r in ingested]
            if skipped:
                lines += [f"\n*Skipped (unsupported format): {', '.join(skipped)}*"]
            await cl.Message(content="\n".join(lines)).send()

        # If no text query came with the upload, stop here
        if not message.content.strip():
            return

    # ── Answer the query ──────────────────────────────────────────────────────
    import re as _re
    query = _re.sub(r'[\s \t​  ]+', ' ', message.content).strip()
    if not query:
        return

    start_total = time.time()

    # Load last 4 turns of history from session (needed for classify_intent too)
    history = cl.user_session.get("chat_history", [])
    last_cited_docs = cl.user_session.get("last_cited_docs", [])

    # Handle version confirmation — if a similar doc was found during upload
    pending_version = cl.user_session.get("pending_version", None)
    _yes = {"yes", "yeah", "yep", "sure", "go ahead", "ok", "okay", "do it", "please", "y"}
    _no  = {"no", "nope", "nah", "n", "separate", "new", "different"}
    if pending_version:
        reply = query.strip().lower().rstrip("!.")
        if reply in _yes:
            cl.user_session.set("pending_version", None)
            pv = pending_version
            # Ingest under the similar doc's filename so it becomes a new version
            await cl.Message(content=f"Saving as a new version of **'{pv['similar_doc']}'**...").send()
            chunks = await cl.make_async(ingest_file)(pv["tmp_path"], pv["similar_doc"])
            if not pv.get("chainlit_owns"):
                try:
                    os.unlink(pv["tmp_path"])
                except Exception:
                    pass
            await cl.Message(content=f"Done — **'{pv['similar_doc']}'** updated to a new version ({chunks} chunks added).").send()
            return
        elif reply in _no:
            cl.user_session.set("pending_version", None)
            pv = pending_version
            await cl.Message(content=f"Saving **'{pv['file_name']}'** as a new separate document...").send()
            chunks = await cl.make_async(ingest_file)(pv["tmp_path"], pv["file_name"])
            if not pv.get("chainlit_owns"):
                try:
                    os.unlink(pv["tmp_path"])
                except Exception:
                    pass
            await cl.Message(content=f"Done — **'{pv['file_name']}'** added as a new document ({chunks} chunks indexed).").send()
            return

    # Handle web search confirmation — if last answer was a NotFound prompt and user says yes
    pending_web_query = cl.user_session.get("pending_web_query", "")
    _yes_signals = {"yes", "yeah", "yep", "sure", "go ahead", "search", "ok", "okay", "do it", "please"}
    if pending_web_query and query.strip().lower().rstrip("!.") in _yes_signals:
        cl.user_session.set("pending_web_query", "")
        query = pending_web_query  # replay the original query
        # Force web search by injecting into history as web_only
        history = cl.user_session.get("chat_history", [])
        initial_state = {
            "query": query, "query_en": "", "lang": "en", "clean_question": "",
            "search_pref": "web_only", "format_pref": "no_preference", "tone": "neutral",
            "doc_filter": "", "doc_filter_2": "", "version_pref": "", "matched_doc": "",
            "history": history, "last_cited_docs": last_cited_docs, "question_type": "factual",
            "grouped_docs": "{}", "context": [],
            "answer": "", "source_type": "", "iterations": 0, "gap": "", "disc_topic": "",
        }
        with StdoutCapture() as buf:
            result = await cl.make_async(agent_app.invoke)(initial_state)
        total_time = time.time() - start_total
        raw_log = buf.getvalue()
        trace   = parse_trace(raw_log)
        source  = result.get('source_type', 'Web')
        iters   = min(result.get('iterations', 0), 3)
        contexts = result.get('context', [])
        has_local = any("Internal:" in c or "| SOURCE:" in c for c in contexts)
        is_hybrid = (source == "Web") and has_local
        final_answer = result.get('answer', 'No answer was generated.')
        if _detect_hinglish(message.content):
            final_answer = await cl.make_async(_translate_to_hinglish)(final_answer)
        history.append(f"Q: {query}")
        history.append(f"A: {final_answer[:500]}")
        cl.user_session.set("chat_history", history[-8:])
        seen_nodes = {}
        ordered_nodes = []
        for step in trace:
            n = step['node']
            if n not in seen_nodes:
                seen_nodes[n] = []
                ordered_nodes.append(n)
            seen_nodes[n].append(step['msg'])
        for node in ordered_nodes:
            msgs = seen_nodes[node]
            async with cl.Step(name=node, type="run") as s:
                s.input  = query if node == 'Analyst' else msgs[0]
                s.output = '\n'.join(msgs)
        attempt_str = f"{iters} search attempt{'s' if iters != 1 else ''}"
        badge = "**[ WEB RESEARCH ]**"
        time_note = f"*{attempt_str} in {total_time:.1f}s*"
        msg = cl.Message(content="")
        await msg.send()
        await msg.stream_token(f"{badge}  {time_note}\n\n---\n\n")
        for i, word in enumerate(final_answer.split(" ")):
            await msg.stream_token(word + (" " if i < len(final_answer.split(" ")) - 1 else ""))
            await asyncio.sleep(0.012)
        await msg.update()
        return

    # Intent check — skip pipeline entirely for casual conversation
    if await cl.make_async(classify_intent)(query, history) == "CHAT":
        intent_data = await cl.make_async(extract_intent)(query)
        response = await cl.make_async(handle_chat)(query, intent_data.get("tone", "neutral"), history)
        await cl.Message(content=response).send()
        history.append(f"Q: {query}")
        history.append(f"A: {response[:300]}")
        cl.user_session.set("chat_history", history[-8:])
        return

    initial_state = {
        "query":           query,
        "query_en":        "",
        "lang":            "en",
        "clean_question":  "",
        "search_pref":     "no_preference",
        "format_pref":     "no_preference",
        "tone":            "neutral",
        "doc_filter":      "",
        "doc_filter_2":    "",
        "version_pref":    "",
        "matched_doc":     "",
        "history":         history,
        "last_cited_docs": last_cited_docs,
        "question_type":   "factual",
        "grouped_docs":    "{}",
        "context":         [],
        "answer":          "",
        "source_type":     "",
        "iterations":      0,
        "gap":             "",
        "disc_topic":      "",
    }

    with StdoutCapture() as buf:
        result = await cl.make_async(agent_app.invoke)(initial_state)

    total_time = time.time() - start_total
    raw_log    = buf.getvalue()
    trace      = parse_trace(raw_log)

    source      = result.get('source_type', 'Web')
    iters       = min(result.get('iterations', 0), 3)
    contexts    = result.get('context', [])
    matched_doc = result.get('matched_doc', '')
    version_pref = result.get('version_pref', '')

    has_local = any("Internal:" in c or "| SOURCE:" in c for c in contexts)
    is_hybrid = (source == "Web") and has_local

    # Track cited documents as an ordered list (most recent last), deduped
    last_cited = cl.user_session.get("last_cited_docs", [])
    newly_cited = []
    for c in contexts:
        for line in c.split("\n"):
            if "Internal:" in line:
                try:
                    doc_name = line.split("Internal:")[1].split("(")[0].strip()
                    if doc_name and doc_name not in newly_cited:
                        newly_cited.append(doc_name)
                except Exception:
                    pass
    # For MultiDoc results, grouped_docs holds both doc names — track them both
    grouped_raw = result.get("grouped_docs", "{}")
    if grouped_raw and grouped_raw != "{}":
        try:
            import json as _j
            for key in _j.loads(grouped_raw).keys():
                fn = key.split(" (v")[0].strip()  # strip version suffix e.g. "doc.pdf (v1)"
                if fn and fn not in newly_cited:
                    newly_cited.append(fn)
        except Exception:
            pass
    if matched_doc and matched_doc not in newly_cited:
        newly_cited.append(matched_doc)
    # Append new docs to end; remove duplicates preserving order; keep last 10
    for d in newly_cited:
        if d in last_cited:
            last_cited.remove(d)
        last_cited.append(d)
    last_cited = last_cited[-10:]
    cl.user_session.set("last_cited_docs", last_cited)
    if matched_doc:
        cl.user_session.set("last_matched_doc", matched_doc)

    # ── Thought Trace — one Chainlit Step per pipeline node ──────────────────
    seen_nodes    = {}
    ordered_nodes = []
    for step in trace:
        n = step['node']
        if n not in seen_nodes:
            seen_nodes[n] = []
            ordered_nodes.append(n)
        seen_nodes[n].append(step['msg'])

    for node in ordered_nodes:
        msgs = seen_nodes[node]
        async with cl.Step(name=node, type="run") as s:
            s.input  = query if node == 'Analyst' else msgs[0]
            s.output = '\n'.join(msgs)

    # ── Build header badge + timing ───────────────────────────────────────────
    _v = version_pref if version_pref and version_pref not in ("latest", "") else ""
    if is_hybrid:
        doc_note  = f" · {matched_doc}" if matched_doc else ""
        v_note    = f" ({_v})" if _v else ""
        badge     = "**[ HYBRID — VAULT + WEB ]**"
        time_note = f"*Vault + web combined in {total_time:.1f}s{doc_note}{v_note}*"
    elif source == "Decomposed":
        badge     = "**[ DECOMPOSED QUERY ]**"
        time_note = f"*Multi-part answer in {total_time:.1f}s*"
    elif source == "VersionDiff":
        doc_note  = f" · {matched_doc}" if matched_doc else ""
        badge     = "**[ VERSION COMPARISON ]**"
        time_note = f"*Compared versions in {total_time:.1f}s{doc_note}*"
    elif source == "Local" or source == "MultiDoc":
        doc_note  = f" · {matched_doc}" if matched_doc else ""
        v_note    = f" ({_v})" if _v else ""
        badge     = "**[ LOCAL VAULT ]**" if source == "Local" else "**[ MULTI-DOC SYNTHESIS ]**"
        time_note = f"*Local memory in {total_time:.1f}s{doc_note}{v_note}*"
    else:
        attempt_str = f"{iters} search attempt{'s' if iters != 1 else ''}"
        badge       = "**[ WEB RESEARCH ]**"
        time_note   = f"*{attempt_str} in {total_time:.1f}s*"

    final_answer = result.get('answer', 'No answer was generated.')

    # Translate answer to Hinglish if the original query was in Hinglish
    if _detect_hinglish(message.content):
        final_answer = await cl.make_async(_translate_to_hinglish)(final_answer)

    # If NotFound — store the original query so user can confirm web search
    if source == "NotFound" or (not source and "search the web" in final_answer.lower()):
        cl.user_session.set("pending_web_query", query)
    else:
        cl.user_session.set("pending_web_query", "")

    # ── Update conversation history (keep last 4 turns) ───────────────────────
    history.append(f"Q: {query}")
    history.append(f"A: {final_answer[:500]}")  # truncate long answers to save context
    cl.user_session.set("chat_history", history[-8:])  # 4 turns = 8 entries (Q+A each)

    # ── Stream the answer token by token ─────────────────────────────────────
    msg = cl.Message(content="")
    await msg.send()

    header = f"{badge}  {time_note}\n\n---\n\n"
    await msg.stream_token(header)

    words = final_answer.split(" ")
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        await msg.stream_token(token)
        await asyncio.sleep(0.012)

    await msg.update()
