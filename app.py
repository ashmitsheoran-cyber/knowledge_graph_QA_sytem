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

# ── Clickable source citations — build sidebar elements from retrieved context ──
# Presentation-layer only: reads what the engine already returned in result['context'].
# Never imports or alters retrieval logic. Pages absent from the context are recovered
# read-only from ChromaDB metadata (which always carries the page).
_MAX_SOURCES = 10

def _parse_sources(contexts):
    """Extract [{chunk, file, page}] from retrieved context. Robust to ALL engine
    context shapes:
      A) per-line : 'FACT: <chunk> | SOURCE: Internal: <file> (vN), Page N'
      B) block    : 'FACT: <c1>\\n\\nFACT: <c2>\\n| SOURCE: Internal: <file>'  (one source covers all)
      C) multi-doc: '--- <file> ---\\n<chunk>\\n\\n<chunk>'
    Read-only — never touches retrieval."""
    import re
    sources, seen = [], set()

    def _src_to_file_page(src):
        src = (src or "").strip()
        if not src.startswith("Internal:"):
            return None, None
        body = src.split("Internal:", 1)[1]
        file = body.split("(v")[0].split(", Page")[0].strip().rstrip(",").strip()
        m = re.search(r"Page\s+([0-9]+)", body)
        return (file or None), (m.group(1) if m else None)

    def _add(chunk, file, page):
        chunk = (chunk or "").strip()
        if not chunk or len(chunk) < 12 or not file or file == "vault_metadata":
            return
        key = (chunk[:120], file)
        if key not in seen:
            seen.add(key)
            sources.append({"chunk": chunk, "file": file, "page": page})

    for ctx in contexts or []:
        if not ctx:
            continue
        if "FACT:" in ctx:
            # Formats A & B. A trailing SOURCE covers any FACT chunks lacking their own.
            all_src = re.findall(r"\|\s*SOURCE:\s*(Internal:[^\n]+)", ctx)
            global_src = all_src[-1] if all_src else None
            for seg in ctx.split("FACT:")[1:]:
                if "| SOURCE:" in seg:
                    chunk_part, src_part = seg.split("| SOURCE:", 1)
                    src = src_part.split("\n", 1)[0].strip()
                else:
                    chunk_part, src = seg, global_src
                file, page = _src_to_file_page(src)
                if file:
                    _add(chunk_part, file, page)
        elif "---" in ctx:
            parts = re.split(r"---\s*(.+?)\s*---", ctx)
            i = 1
            while i < len(parts):
                fname = parts[i].strip()
                body  = parts[i + 1] if i + 1 < len(parts) else ""
                for chunk in body.split("\n\n"):
                    _add(chunk, fname, None)
                i += 2
    return sources

def _recover_pages(sources):
    """Fill in any missing page by matching the chunk back to ChromaDB metadata
    (which always has the page). Read-only — one .get() per cited file."""
    need = {s["file"] for s in sources if s["page"] is None}
    if not need:
        return sources
    try:
        from tools import vault_collection
    except Exception:
        return sources
    page_maps = {}
    for file in need:
        try:
            res = vault_collection.get(
                where={"$and": [{"file_name": {"$eq": file}}, {"is_latest": {"$eq": "true"}}]},
                include=["documents", "metadatas"])
            docs  = res.get("documents") or []
            metas = res.get("metadatas") or []
            page_maps[file] = {d.strip(): meta.get("page") for d, meta in zip(docs, metas)}
        except Exception:
            page_maps[file] = {}
    for s in sources:
        if s["page"] is None:
            s["page"] = page_maps.get(s["file"], {}).get(s["chunk"].strip())
    return sources

def _answer_citations(answer, known_files):
    """Parse each inline '(Source: …)' citation into {claim, file, pages}. Robust to
    nested parens — version markers like '(v1)' and filenames containing parens
    (e.g. 'text (1).pdf') — via paren-DEPTH span scanning. The file is matched by
    substring against the known vault filenames (no fragile filename parsing). Page
    numbers are read after the 'Page' keyword (so the file's own digits aren't grabbed).
    Web/URL citations (no 'Internal:') are skipped."""
    import re
    answer = answer or ""
    # 1) Find balanced "(Source: … )" spans, counting paren depth so an inner ")" from
    #    "(v1)" or a filename doesn't end the span early.
    spans, i = [], 0
    while True:
        s = answer.find("(Source:", i)
        if s < 0:
            break
        depth, j = 0, s
        while j < len(answer):
            if answer[j] == "(":
                depth += 1
            elif answer[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        e = j if j < len(answer) else len(answer) - 1
        spans.append((s, e))
        i = e + 1
    # 2) Parse each span.
    kf = sorted([k for k in (known_files or []) if k], key=len, reverse=True)  # longest first
    out, prev = [], 0
    for (s, e) in spans:
        claim = re.sub(r"\s+", " ", answer[prev:s]).strip(" -*•\t")
        prev  = e + 1
        cit   = answer[s:e + 1]
        if "Internal:" not in cit:
            continue                                   # web/URL citation — skip
        file = next((f for f in kf if f in cit), "")
        pm   = re.search(r"\bPages?\b", cit)
        pages = re.findall(r"\d+", cit[pm.end():]) if pm else []   # page nums come AFTER 'Page'
        if file:
            out.append({"claim": claim, "file": file, "pages": pages})
    return out

def build_source_actions(contexts, answer):
    """Build click-to-open source buttons (cl.Action) — ONE per inline citation.
    If a citation names pages, the button opens one chunk PER cited page (fetched
    read-only from ChromaDB); a page-less citation opens the single best-matching
    chunk. Actions (not side elements) → nothing auto-opens; the sidebar opens only
    on click via 'view_source'. Engine untouched; fully guarded."""
    import numpy as np
    pool      = _parse_sources(contexts)        # retrieved pool (also the known-file list)
    citations = _answer_citations(answer, {s["file"] for s in pool})
    if not citations and not pool:
        return []
    try:
        from tools import embedding_model as emb, vault_collection
    except Exception:
        emb, vault_collection = None, None

    # Preload each cited file's latest chunks once: file -> {page -> [chunks]} and all chunks.
    page_maps, all_chunks = {}, {}
    for f in {c["file"] for c in citations if c.get("file")}:
        pm, allc = {}, []
        if vault_collection is not None:
            try:
                res = vault_collection.get(
                    where={"$and": [{"file_name": {"$eq": f}}, {"is_latest": {"$eq": "true"}}]},
                    include=["documents", "metadatas"])
                for d, mt in zip(res.get("documents") or [], res.get("metadatas") or []):
                    pm.setdefault(mt.get("page"), []).append(d)
                    allc.append(d)
            except Exception:
                pass
        page_maps[f], all_chunks[f] = pm, allc

    def _best(chunks, claim):
        chunks = [c for c in chunks if c]
        if not chunks:
            return None
        if len(chunks) == 1 or emb is None or not claim:
            return chunks[0]
        ce = np.array(emb.encode(chunks)); q = emb.encode([claim])[0]
        sims = (ce @ q) / (np.linalg.norm(ce, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        return chunks[int(np.argmax(sims))]

    actions = []
    if citations:
        for i, c in enumerate(citations, 1):
            f, claim, pages = c["file"], c.get("claim", ""), c.get("pages") or []
            pm = page_maps.get(f, {})
            items, seen = [], set()
            for pg in pages:                                  # one chunk per cited page
                try:    cand = pm.get(int(pg)) or []
                except Exception: cand = []
                ch = _best(cand, claim)
                if ch and ch[:120] not in seen:
                    seen.add(ch[:120]); items.append({"page": str(pg), "chunk": ch})
            if not items:                                     # page-less citation → single best chunk
                src_pool = [s["chunk"] for s in pool if s["file"] == f] \
                           or all_chunks.get(f) or [s["chunk"] for s in pool]
                ch = _best(src_pool, claim)
                if ch:
                    pg = next((str(p) for p, cl in pm.items() if ch in cl), "")
                    items.append({"page": pg, "chunk": ch})
            if not items:
                continue
            pages_lbl = ", ".join(it["page"] for it in items if it["page"])
            if len(items) > 1 and pages_lbl:   suffix = f" (pp. {pages_lbl})"
            elif items[0]["page"]:             suffix = f" (p.{items[0]['page']})"
            else:                              suffix = ""
            actions.append(cl.Action(
                name="view_source",
                payload={"file": f, "items": items},
                label=f"📄 Source {i} · {f}{suffix}",
                tooltip="View the exact passage(s) this answer drew from",
            ))
    else:
        # No inline citations at all → top chunks vs the whole answer (graceful fallback).
        ranked = pool
        if emb is not None and answer and pool:
            ce = np.array(emb.encode([s["chunk"] for s in pool])); q = emb.encode([answer[:1000]])[0]
            sims = (ce @ q) / (np.linalg.norm(ce, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
            ranked = [pool[k] for k in np.argsort(-sims)]
        for i, s in enumerate(_recover_pages(ranked[:3]), 1):
            pg = str(s["page"]) if s.get("page") not in (None, "", "?") else ""
            actions.append(cl.Action(
                name="view_source",
                payload={"file": s["file"], "items": [{"page": pg, "chunk": s["chunk"]}]},
                label=f"📄 Source {i} · {s['file']}" + (f" (p.{pg})" if pg else ""),
                tooltip="View the passage this answer drew from",
            ))
    return actions[:_MAX_SOURCES]


_PDF_HL_DIR = "_pdf_highlights"   # runtime cache of highlighted PDFs (safe to delete)

def _mark_chunk(pg, chunk):
    """Highlight a chunk's CONTIGUOUS text on a page as clean, readable bands.
    Aligns the chunk against the page's real word stream (no scattered gaps),
    unions the matched words per line, and uses light opacity so the text under
    the highlight stays readable. Falls back to phrase search if alignment fails."""
    import re, collections, fitz
    norm   = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    pwords = pg.get_text("words")                         # (x0,y0,x1,y1, word, block, line, wno)
    if not pwords or not chunk:
        return False
    pnorm = [norm(w[4]) for w in pwords]
    cnorm = [t for t in (norm(w) for w in chunk.split()) if t]
    if not cnorm:
        return False
    # Locate the chunk's start in the page's word stream (anchor on first 5 tokens).
    anchor, start = cnorm[:5], -1
    for i in range(len(pnorm) - len(anchor) + 1):
        if pnorm[i:i + len(anchor)] == anchor:
            start = i
            break
    if start < 0:                                         # anchor failed → first distinctive token
        distinctive = next((t for t in cnorm if len(t) >= 7), None)
        if distinctive and distinctive in pnorm:
            start = pnorm.index(distinctive)
    if start >= 0:
        span  = pwords[start:min(start + len(cnorm), len(pwords))]
        lines = collections.OrderedDict()
        for w in span:                                    # union the matched words per text line
            key = (w[5], w[6])
            r   = fitz.Rect(w[:4])
            lines[key] = (lines[key] | r) if key in lines else r
        for r in lines.values():
            a = pg.add_highlight_annot(r)
            a.set_opacity(0.40)                           # light → text stays readable
            a.update()
        return True
    # Fallback: light phrase-window highlights (still readable, even if less contiguous).
    ws, done = chunk.split(), False
    for k in range(0, max(1, len(ws) - 5), 6):
        for r in pg.search_for(" ".join(ws[k:k + 6])):
            a = pg.add_highlight_annot(r); a.set_opacity(0.40); a.update(); done = True
    return done

def _highlighted_pdf(file, items):
    """Return (full_pdf_path, first_page): the WHOLE document with the cited chunk(s)
    highlighted CLEANLY on their page(s). The highlight is RASTERIZED into the cited
    page (rendered to an image and laid back in) so it shows exactly like the readable
    image — never patchy in the PDF viewer. Other pages stay original. One navigable
    file. None if no source PDF on disk."""
    src = os.path.join("docs", file)
    if not os.path.exists(src):
        return None
    try:
        import fitz, hashlib
        by_page = {}
        for it in items:
            ps = str(it.get("page", ""))
            if ps.isdigit():
                by_page.setdefault(int(ps), []).append(it.get("chunk", "") or "")
        if not by_page:
            return None
        os.makedirs(_PDF_HL_DIR, exist_ok=True)
        pages = sorted(by_page)
        key   = hashlib.md5(f"{file}|{[(p, by_page[p]) for p in pages]}|v5".encode()).hexdigest()[:12]
        out   = os.path.join(_PDF_HL_DIR, f"{key}.pdf")
        if not os.path.exists(out):
            doc = fitz.open(src)
            for p in pages:
                if p < 1 or p > len(doc):
                    continue
                pg = doc[p - 1]                              # metadata page is 1-indexed
                for chunk in by_page[p]:
                    _mark_chunk(pg, chunk)                   # clean per-line highlight annots
                pix = pg.get_pixmap(dpi=150)                 # render page INCLUDING the highlight
                for a in list(pg.annots() or []):           # drop annots so they don't double-draw
                    pg.delete_annot(a)
                pg.insert_image(pg.rect, pixmap=pix, overlay=True)   # lay the clean render back in
            doc.save(out)
            doc.close()
        return out, pages[0]
    except Exception as e:
        print(f"[UI] highlighted pdf failed for '{file}' (falling back to text): {e}")
        return None


@cl.action_callback("view_source")
async def _on_view_source(action: cl.Action):
    """Open the cited source in the side panel — fires ONLY on click (no auto-open).
    Preferred: the REAL PDF, opened at the chunk's page with the passage highlighted.
    Fallback (no PDF on disk): the chunk text, one passage per cited page."""
    p     = action.payload or {}
    file  = p.get("file", "Source")
    items = p.get("items") or []
    if not items and "chunk" in p:                       # backward-compat with old payload
        items = [{"page": p.get("page", ""), "chunk": p.get("chunk", "")}]

    # ── Preferred path: ONE full document, cited chunk(s) cleanly highlighted ──
    res = _highlighted_pdf(file, items)
    if res:
        pdf_path, first_page = res
        await cl.ElementSidebar.set_title(f"{file} · p.{first_page}")
        await cl.ElementSidebar.set_elements([
            cl.Pdf(name=file, path=pdf_path, page=first_page, display="side")
        ])
        return

    # ── Fallback: text passages (docs whose PDF isn't on disk) ──
    elements, pages = [], []
    for j, it in enumerate(items, 1):
        pg    = it.get("page", "")
        chunk = it.get("chunk", "")
        head  = f"**{file}**" + (f" — Page {pg}" if pg else "")
        elements.append(cl.Text(name=f"passage_{j}", content=f"{head}\n\n---\n\n{chunk}", display="side"))
        if pg:
            pages.append(pg)
    title = file + (f" · pp. {', '.join(pages)}" if len(pages) > 1 else (f" · p.{pages[0]}" if pages else ""))
    await cl.ElementSidebar.set_title(title)
    await cl.ElementSidebar.set_elements(elements)

# ── Welcome screen ────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    import shutil
    shutil.rmtree(_PDF_HL_DIR, ignore_errors=True)   # clear stale highlighted-PDF cache
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

    # ── Clickable source citations — open the exact retrieved chunk in a sidebar ──
    # Additive + fully guarded: if anything here fails, the answer still renders.
    try:
        src_actions = build_source_actions(contexts, final_answer)
        if src_actions:
            await msg.stream_token("\n\n---\n\n*📎 Sources — click a button below to view the exact passage:*")
            msg.actions = src_actions
    except Exception as _src_e:
        print(f"[UI] Source citation build failed (answer unaffected): {_src_e}")

    await msg.update()
