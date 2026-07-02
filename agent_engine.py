from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from tools import local_graph_check, save_to_graph, search_tool, fetch_page_content, find_matching_doc, find_relevant_docs, find_relevant_docs_per_file, get_chunks_by_doc, get_version_info, get_chunks_for_version, get_vault_documents, identify_best_doc_for_query, get_all_chunks_for_doc, search_top_chunks, find_doc_by_literal_terms, vault_collection
from brain import get_llm, get_mini_llm
import time

main_llm = get_llm()
mini_llm = get_mini_llm()

class AgentState(TypedDict):
    query: str
    query_en: str        # English translation of query (same as query if already English)
    lang: str            # "en" or "hinglish"
    clean_question: str  # query stripped of user instructions
    search_pref: str     # "local_only", "web_only", "no_preference"
    format_pref: str     # "brief", "detailed", "bullet_points", "no_preference"
    tone: str            # "neutral", "casual", "frustrated", "sad", "excited"
    doc_filter: str      # document name fragment if user refers to a specific doc
    version_pref: str    # "latest", "v1", "old", etc. — empty = default (latest)
    matched_doc: str     # resolved exact file_name after fuzzy matching
    history: List[str]   # last 4 Q&A turns for follow-up context
    last_cited_docs: List[str]  # ordered list of docs cited this session (most recent last)
    question_type: str   # comparison | factual | analytical | summary | list | recommendation
    grouped_docs: str    # JSON — {doc_name: [chunks]} for multi-doc synthesis
    context: List[str]
    answer: str
    source_type: str
    iterations: int
    gap: str
    disc_topic: str   # stripped topic query used for discovery embedding (set by analyst, used by writer)
    doc_filter_2: str # second document filter for two-doc comparison queries
    rescued: bool     # True once judge_node has run a literal-keyword vault rescue (prevents re-entry)

# Cache for the keyword sweep's full-vault text fetch — query-independent, invalidated when
# vault count changes (any ingestion bumps the count).
_kw_vault_cache: tuple = (None, None)  # (total_count, raw_result)

# Phrases the writer uses when local context is incomplete.
# Judge catches these and triggers a targeted web search rather than showing a half-answer.
_HINGLISH_SIGNALS = {
    "kya", "hai", "hain", "ka", "ki", "ke", "mein", "ko", "se", "par",
    "karo", "karo", "batao", "bata", "kaise", "kyun", "kyunki", "aur",
    "nahi", "nai", "tha", "thi", "the", "hoga", "hogi", "wala", "wali",
    "matlab", "matlab", "samjhao", "likho", "dijiye", "bataiye", "chahiye",
}

_HINGLISH_HEDGING = [
    "context mein nahi", "mention nahi", "nahi bataya", "pata nahi chalta",
    "available nahi", "nahi milta", "clear nahi", "nahi hai", "nahi hain",
    "context mein nahi hai", "information nahi", "nahi diya",
]

_HEDGING_PHRASES = [
    "does not specify", "does not contain", "cannot confirm", "not mentioned",
    "not provided", "not detailed", "not available in the context",
    "cannot determine", "no specific", "context does not", "context doesn't",
    "not found in", "no information", "is not clear", "unclear from",
    "the provided context", "does not include", "not included in",
    "does not explicitly", "does not explicitly state", "does not state",
    "no information provided", "there is no information",
]

_CHAT_PATTERNS = {
    "hello", "hi", "hey", "thanks", "thank you", "thank you so much",
    "great", "perfect", "awesome", "nice", "cool", "ok", "okay", "got it",
    "bye", "goodbye", "see you", "good morning", "good afternoon", "good evening",
    "how are you", "who are you", "what are you", "what can you do",
    "what functions can you perform", "what can you help with",
    "what do you do", "what are your capabilities", "tell me about yourself",
    "what are you capable of", "how can you help", "what do you know",
}

def classify_intent(query: str, history: list = None) -> str:
    """Returns 'CHAT' or 'QUERY'. Rule-based first, LLM fallback for ambiguous cases."""
    q = query.strip().lower().rstrip("!?.")
    if q in _CHAT_PATTERNS:
        return "CHAT"
    try:
        history_ctx = ""
        if history:
            history_ctx = "Recent conversation:\n" + "\n".join(history[-4:]) + "\n\n"
        result = mini_llm.invoke(
            f"{history_ctx}"
            "You are classifying a user message as CHAT or QUERY. Think carefully before deciding.\n\n"
            "CHAT: anything the assistant can handle from its own understanding — greetings, reactions, "
            "praise ('good boy', 'well done', 'nice'), thanks, expressions of feeling, small talk, "
            "opinions, hypotheticals, or anything with no factual answer to retrieve.\n\n"
            "QUERY: needs information to be looked up — factual questions, document questions, "
            "follow-up questions that reference something from the conversation history "
            "('what about the previous one?', 'compare them', 'and for Apple?').\n\n"
            "Key insight: if the message is a reaction or acknowledgement with no question in it, "
            "it is CHAT even if it sounds unusual. If it references prior conversation to ask for "
            "more information, it is QUERY.\n\n"
            "When genuinely unsure, classify as CHAT.\n"
            "Output ONLY one word: CHAT or QUERY.\n\n"
            f"Message: {query}"
        ).content.strip().upper()
        return "CHAT" if "CHAT" in result else "QUERY"
    except Exception:
        return "QUERY"

def handle_chat(query: str, tone: str = "neutral", history: list = None) -> str:
    """Lightweight conversational response — adapts to user tone, uses history for follow-ups."""
    tone_instruction = {
        "frustrated": "The user seems frustrated. Acknowledge their frustration warmly and offer to help.",
        "sad":        "The user seems sad or down. Be warm, empathetic, and gently encouraging.",
        "excited":    "The user is excited or enthusiastic. Match their energy and be upbeat.",
        "casual":     "Keep it friendly and relaxed.",
        "neutral":    "Be friendly and natural.",
    }.get(tone, "Be friendly and natural.")

    history_ctx = ""
    if history:
        history_ctx = "Recent conversation:\n" + "\n".join(history[-4:]) + "\n\n"

    try:
        return mini_llm.invoke(
            f"{history_ctx}"
            f"You are a warm, intelligent Strategic Intelligence Assistant with your own personality. "
            f"{tone_instruction} "
            f"If the message is a follow-up or inference from the conversation above, answer it directly using that context. "
            f"If asked what you can do, mention: answering questions from private documents, "
            f"live web search with citations, and cross-document reasoning. "
            f"Keep it under 3 sentences. Do not sound robotic.\n\nMessage: {query}"
        ).content.strip()
    except Exception:
        return "Hey! Ask me anything — I'm here to help."

def extract_intent(query: str) -> dict:
    """
    Extracts structured intent from the user's query using LLM understanding.
    Returns clean_question, search_pref, format_pref, tone.
    Never raises — always returns safe defaults.
    """
    try:
        raw = mini_llm.invoke(
            "Analyze the following user message and extract these 8 fields. "
            "Reply in exactly this format, one field per line, nothing else:\n"
            "clean_question: <the actual question, stripped of any instructions or preferences>\n"
            "search_pref: <one of: local_only, web_only, no_preference>\n"
            "format_pref: <one of: brief, detailed, bullet_points, no_preference>\n"
            "tone: <one of: neutral, casual, frustrated, sad, excited>\n"
            "doc_filter: <name or fragment of the FIRST specific document the user refers to, empty string if none>\n"
            "doc_filter_2: <name or fragment of a SECOND specific document explicitly mentioned, empty string if only one or zero documents mentioned>\n"
            "version_pref: <one of: latest, v1, v2, v3, original, old, or empty string if not specified>\n"
            "question_type: <one of: comparison, factual, analytical, summary, list, recommendation>\n\n"
            "Rules:\n"
            "- search_pref=local_only if user says things like 'only use local', 'don't search web', 'from our data only'\n"
            "- search_pref=web_only ONLY if user uses explicit words like 'search online', 'look it up on the web', 'search the internet', 'google it' — dates, years, and event names alone do NOT qualify\n"
            "- when in doubt, use no_preference — the system will decide whether to search or not\n"
            "- format_pref=brief if user says 'short', 'brief', 'quick answer', 'in one line'\n"
            "- format_pref=bullet_points if user says 'list', 'bullet points', 'points'\n"
            "- tone=frustrated if user uses caps, exclamation marks in frustration, or expresses annoyance\n"
            "- tone=sad if user expresses sadness, disappointment, or distress\n"
            "- tone=excited if user uses enthusiastic language\n"
            "- tone=casual if conversational but not emotional\n"
            "- doc_filter: extract the document name or keyword if (1) user says 'in the X report', 'from the X file', 'in that PDF', etc., OR (2) the query is specifically ABOUT a named model, organization, or publication (e.g. 'how does Llama 3 handle X' → 'Llama 3', 'what does the WIPO report say' → 'WIPO', 'explain constitutional AI' → 'constitutional AI', 'what are GPT-4 capabilities' → 'GPT-4', 'what is RBI methodology' → 'RBI') — empty string for general concepts or topics without a specific named document\n"
            "- doc_filter_2: ONLY set if user explicitly names a SECOND different document in the same message (e.g. 'compare the RBI report with the Budget Speech') — empty string if only one document is mentioned or the comparison is within a single document's topic\n"
            "- version_pref: extract if user says 'old version', 'v1', 'original', 'latest', 'May version' — empty string if not specified\n"
            "- question_type=comparison if asking to compare, contrast, differentiate, or evaluate differences between two or more things\n"
            "- question_type=summary if asking to summarize, overview, give key points, or describe what a document/topic is about\n"
            "- question_type=list if asking to enumerate, list all, name all items of a category\n"
            "- question_type=recommendation if asking what should be done, what is best, what is recommended, or for an opinion/verdict\n"
            "- question_type=analytical if asking why, how something works, what caused something, explain a concept, or analyze a situation\n"
            "- question_type=factual for everything else — what is, when was, how many, who, where\n\n"
            f"Message: {query}"
        ).content.strip()

        result = {
            "clean_question":  query,
            "search_pref":     "no_preference",
            "format_pref":     "no_preference",
            "tone":            "neutral",
            "doc_filter":      "",
            "doc_filter_2":    "",
            "version_pref":    "",
            "question_type":   "factual",
        }
        for line in raw.split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if key in result:
                    result[key] = val
        return result
    except Exception:
        return {
            "clean_question": query,
            "search_pref":    "no_preference",
            "format_pref":    "no_preference",
            "tone":           "neutral",
            "doc_filter":     "",
            "doc_filter_2":   "",
            "version_pref":   "",
            "question_type":  "factual",
        }

def decompose_query(query: str) -> list:
    """
    Detects if query has 2+ distinct sub-questions requiring separate retrieval.
    Returns list of sub-questions if multi-part, empty list if single question.
    Conservative — only splits when genuinely necessary.
    """
    try:
        raw = mini_llm.invoke(
            "Does the following question contain 2 or more DISTINCT sub-questions that need separate answers? "
            "Be conservative — only split if the parts are genuinely independent (different topics, different docs, or different time periods).\n"
            "Compound questions like 'what is X and why is it important' are SINGLE — don't split those.\n\n"
            "If YES: list each sub-question on its own line, prefixed with 'Q: '. Each must be a complete, standalone question.\n"
            "If NO: output exactly: SINGLE\n\n"
            f"Question: {query}"
        ).content.strip()
        if "SINGLE" in raw or not any(line.startswith("Q:") for line in raw.split("\n")):
            return []
        parts = [line[2:].strip() for line in raw.split("\n") if line.startswith("Q:")]
        return parts if len(parts) >= 2 else []
    except Exception:
        return []


def _detect_and_translate(query: str):
    """Returns (lang, query_en). One LLM call only if Hinglish detected."""
    words = set(query.lower().split())
    is_hinglish = len(words & _HINGLISH_SIGNALS) >= 2
    if not is_hinglish:
        return "en", query
    try:
        translated = mini_llm.invoke(
            f"Translate this Hinglish query to English. Output ONLY the translated query, nothing else.\n\n{query}"
        ).content.strip()
        print(f"[ANALYST] Hinglish detected — translated to: {translated}")
        return "hinglish", translated
    except Exception:
        return "hinglish", query  # fallback: use original, retrieval degrades gracefully

# Anchor words that signal the user is referring to something from earlier in the session
# without naming it explicitly. Checked as whole words/phrases inside the query.
_IMPLICIT_REF_ANCHORS = {
    "that", "both", "they",
    "the other", "the first", "the second", "the last", "the third",
    "the same", "previous", "aforementioned", "it", "its",
    "the one", "the file", "the document", "the report", "the paper",
    "the pdf", "the version", "this one", "that one",
}

# Short-form acronym → full phrase expansion. Used by both discovery and broad scan
# so embedding and keyword sweep get meaningful signal from short terms.
_STATIC_EXPANSIONS = {
    "ai":   "artificial intelligence",
    "ml":   "machine learning",
    "llm":  "large language model",
    "llms": "large language models",
    "nlp":  "natural language processing",
    "cv":   "computer vision",
    "dl":   "deep learning",
    "rl":   "reinforcement learning",
    "rlhf": "reinforcement learning human feedback",
    "rag":  "retrieval augmented generation",
    "gpt":  "generative pre-trained transformer",
    "rbi":  "Reserve Bank of India",
    "gdpr": "general data protection regulation",
    "wef":  "World Economic Forum",
    "eu":   "European Union",
    "imf":  "International Monetary Fund",
    "bis":  "Bank for International Settlements",
}

def _has_implicit_ref(query: str) -> bool:
    """Returns True if query contains an implicit reference anchor but no explicit doc name.
    Single-word anchors use whole-word matching to avoid substring false positives
    (e.g. 'it' inside 'with', 'its' inside 'distribution').
    Multi-word anchors use substring matching.
    """
    q = query.lower()
    words = set(q.split())
    for anchor in _IMPLICIT_REF_ANCHORS:
        if ' ' in anchor:
            if anchor in q:  # multi-word: substring is fine
                return True
        else:
            if anchor in words:  # single-word: exact word match only
                return True
    return False


# Terms stripped before scoring filenames against query — pure structural/discovery words
_DISCOVERY_STOPWORDS = {
    "which", "file", "files", "has", "have", "info", "about", "the", "a", "an",
    "in", "is", "are", "was", "what", "where", "document", "documents", "doc",
    "docs", "pdf", "paper", "papers", "how", "many", "list", "all", "data",
    "information", "mention", "mentions", "contains", "contain", "talks", "talk",
    "discusses", "discuss", "covers", "cover", "related", "concerning", "and",
    "or", "that", "with", "for", "on", "from", "does", "do", "any", "me",
    "tell", "show", "find", "get", "can", "you", "give", "know", "want", "of",
    "to", "my", "our", "its", "this", "these", "those",
}

def _fn_keyword_score(filename: str, query_terms: list) -> int:
    """Score a filename by how many query_terms appear as exact word tokens in it."""
    import re as _re
    parts = set(_re.split(r'[\s\-_\.\,\(\)\[\]]+', filename.lower()))
    return sum(1 for t in query_terms if t in parts)

def analyst_node(state: AgentState):
    import re as _re
    import json as _json
    # Normalize query: collapse whitespace variants (double spaces, tabs, non-breaking spaces)
    state = dict(state)
    state["query"] = _re.sub(r'[\s \t​  ]+', ' ', state["query"]).strip()

    print("[ANALYST] Checking Vector Memory...")
    start = time.time()

    # ── Implicit reference resolution ────────────────────────────────────────
    # Discovery signals ("which file", "what document", "how many files") take priority —
    # these are meta-queries about the vault, answered from vault metadata only.
    _DISCOVERY_SIGNALS_EARLY = {
        "which file", "which document", "which doc", "which paper",
        "which pdf", "where is this", "where can i find", "what file",
        "what document", "what paper", "which one has", "which one contains",
        "which one talks", "which one discusses",
        "how many files", "how many documents", "how many docs", "how many pdfs",
        "list all files", "list all documents", "list all docs",
    }
    query_lower_early = state["query"].lower()
    is_discovery_query = any(s in query_lower_early for s in _DISCOVERY_SIGNALS_EARLY)

    last_cited = state.get("last_cited_docs", [])
    pre_resolved_doc = ""
    implicit_ref_fired = False

    # Special case: "which file had that info?" — discovery + implicit ref together.
    # The answer is last_cited_docs[-1] directly; no content search needed.
    if last_cited and is_discovery_query and _has_implicit_ref(state["query"]):
        doc_name = last_cited[-1]
        print(f"[ANALYST] Source attribution query — last cited doc: '{doc_name}'")
        ctx = f"FACT: The information came from '{doc_name}'.\n| SOURCE: Internal: {doc_name}"
        intent = extract_intent(state["query"])
        return {
            "source_type": "Local", "context": [ctx], "grouped_docs": "{}",
            "gap": "", "lang": "en", "query_en": state["query"],
            "clean_question": intent.get("clean_question", state["query"]),
            "search_pref": intent.get("search_pref", "no_preference"),
            "format_pref": intent.get("format_pref", "no_preference"),
            "tone": intent.get("tone", "neutral"),
            "doc_filter": "", "version_pref": "",
            "matched_doc": doc_name,
            "question_type": "factual",
            "iterations": state.get("iterations", 0)
        }

    # Regular implicit ref — only fires when no discovery signal and no explicit filename.
    # Broad-scope queries ("all papers", "every file", "across all docs") span multiple docs —
    # skip implicit ref so broad-scan can route correctly downstream.
    _BROAD_SCOPE_SIGS = {"across all", "from all", "every paper", "every file",
                         "every document", "every doc"}
    _BROAD_SCOPE_WORDS_EARLY = {"papers", "files", "documents", "docs", "pdfs", "reports"}
    _q_early = state["query"].lower()
    _is_broad_scope_early = (
        any(s in _q_early for s in _BROAD_SCOPE_SIGS)
        or ("all " in _q_early and any(w in _q_early for w in _BROAD_SCOPE_WORDS_EARLY))
    )
    if last_cited and _has_implicit_ref(state["query"]) and not is_discovery_query and not _is_broad_scope_early:
        raw_q_lower = state["query"].lower()
        explicit_present = any(
            d["file_name"].lower().rsplit('.', 1)[0] in raw_q_lower
            for d in get_vault_documents()
        )
        if not explicit_present:
            query_words = set(raw_q_lower.split())

            # "both" → retrieve from last 2 distinct cited docs, route to MultiDoc.
            # Scan backwards through last_cited to find the last 2 UNIQUE docs,
            # so repeated citations of the same doc don't break the pair.
            if "both" in query_words and len(last_cited) >= 2:
                distinct_pair = []
                for d in reversed(last_cited):
                    if d not in distinct_pair:
                        distinct_pair.append(d)
                    if len(distinct_pair) == 2:
                        break
                if len(distinct_pair) == 2:
                    doc1, doc2 = distinct_pair[1], distinct_pair[0]  # chronological order
                else:
                    doc1, doc2 = None, None
                if doc1 and doc2:
                    chunks1 = get_all_chunks_for_doc(doc1, max_chunks=8)
                    chunks2 = get_all_chunks_for_doc(doc2, max_chunks=8)
                    if chunks1 and chunks2:
                        intent_b = extract_intent(state["query"])
                        print(f"[ANALYST] 'Both' resolved → '{doc1}' + '{doc2}'")
                        return {
                            "source_type": "MultiDoc",
                            "context": [],
                            "grouped_docs": _json.dumps({doc1: chunks1, doc2: chunks2}),
                            "gap": "", "lang": "en", "query_en": state["query"],
                            "clean_question": intent_b.get("clean_question", state["query"]),
                            "search_pref": intent_b.get("search_pref", "no_preference"),
                            "format_pref": intent_b.get("format_pref", "no_preference"),
                            "tone": intent_b.get("tone", "neutral"),
                            "doc_filter": "", "version_pref": "",
                            "matched_doc": doc1,
                            "question_type": intent_b.get("question_type", "factual"),
                            "iterations": state.get("iterations", 0)
                        }

            # "the other" → resolve to second-to-last cited doc
            if "the other" in raw_q_lower and len(last_cited) >= 2:
                pre_resolved_doc = last_cited[-2]
                print(f"[ANALYST] 'The other' resolved → '{pre_resolved_doc}'")
            else:
                pre_resolved_doc = last_cited[-1]
            implicit_ref_fired = True
            print(f"[ANALYST] Implicit reference resolved → '{pre_resolved_doc}'")

    # ── Vault meta-query: count / list ──────────────────────────────────────
    # Answers questions about the vault itself (how many docs, list all files)
    # directly from vault metadata — bypasses all semantic search.
    # Discovery guard: "how many files mention X" has a topic → falls through
    # to the normal discovery path instead of triggering here.
    _q_vmeta = state["query"].lower()
    _VMETA_DISCOVERY_GUARD = bool(_re.search(
        r'\b(mention|discuss|about|cover|contain|related|regarding|deal|talk|say|on topic)\b',
        _q_vmeta
    ))
    _VMETA_COUNT = bool(_re.search(
        r'\bhow many\b.{0,40}\b(docs?|files?|documents?|papers?|pdfs?)\b',
        _q_vmeta
    ))
    _VMETA_LIST = bool(_re.search(
        r'\b(list|show|give).{0,20}\ball\b.{0,20}\b(docs?|files?|documents?|papers?)\b'
        r'|\bwhat\b.{0,20}\b(docs?|files?|documents?|papers?)\b.{0,30}\b(do you have|you have|in (the )?vault|available)\b'
        r'|\bwhat (are|were).{0,10}(all|every).{0,10}\b(docs?|files?|documents?|papers?)\b'
        r'|\bwhat.{0,15}(in|inside).{0,10}(your|the).{0,10}vault\b'
        r'|\b(total|all).{0,10}\b(docs?|files?|documents?|papers?)\b.{0,20}\b(vault|have|available)\b',
        _q_vmeta
    ))
    if (_VMETA_COUNT or _VMETA_LIST) and not _VMETA_DISCOVERY_GUARD:
        _vault_docs_meta = get_vault_documents()
        _n_docs = len(_vault_docs_meta)
        _doc_list_str = "\n".join(
            f"{i + 1}. {d['file_name']}" for i, d in enumerate(_vault_docs_meta)
        )
        _flat_ctx_meta = (
            f"FACT: Your document vault contains exactly {_n_docs} documents.\n\n"
            f"FACT: Complete list of all {_n_docs} documents in the vault:\n{_doc_list_str}\n"
            f"| SOURCE: Internal: vault_metadata"
        )
        print(f"[ANALYST] Vault meta-query — injecting metadata ({_n_docs} docs)")
        return {
            "source_type": "Local", "context": [_flat_ctx_meta], "grouped_docs": "{}",
            "gap": "", "lang": "en", "query_en": state["query"],
            "clean_question": state["query"],
            "search_pref": "no_preference", "format_pref": "no_preference",
            "tone": "neutral", "doc_filter": "", "version_pref": "",
            "matched_doc": "", "question_type": "list", "disc_topic": "",
            "iterations": state.get("iterations", 0),
        }

    # Extract intent — tone, preferences, clean question, doc filter, version
    intent        = extract_intent(state['query'])
    clean_q       = intent["clean_question"]
    search_pref   = intent["search_pref"]
    format_pref   = intent["format_pref"]
    tone          = intent["tone"]
    doc_filter    = intent.get("doc_filter", "")
    version_pref  = intent.get("version_pref", "")
    question_type = intent.get("question_type", "factual")
    doc_filter_2  = intent.get("doc_filter_2", "")

    # Sanitize: gpt-4o-mini sometimes returns "empty" / "none" as a literal string
    # instead of a blank — this would break matching and badge display.
    _INTENT_NULL = {"empty", "none", "n/a", "null", "na", "not specified"}
    version_pref = "" if version_pref.lower() in _INTENT_NULL else version_pref
    doc_filter   = "" if doc_filter.lower()   in _INTENT_NULL else doc_filter
    doc_filter_2 = "" if doc_filter_2.lower() in _INTENT_NULL else doc_filter_2

    if tone not in ("neutral", "casual"):
        print(f"[ANALYST] Tone detected: {tone}")

    # Resolve doc_filter to exact file_name via fuzzy matching
    matched_doc   = find_matching_doc(doc_filter)   if doc_filter   else ""
    matched_doc_2 = find_matching_doc(doc_filter_2) if doc_filter_2 else ""
    if matched_doc:
        print(f"[ANALYST] Document filter resolved: '{doc_filter}' → '{matched_doc}' (v_pref: '{version_pref or 'latest'}')")
    if matched_doc_2:
        print(f"[ANALYST] Second document resolved: '{doc_filter_2}' → '{matched_doc_2}'")

    # Fallback: verbatim filename check — catches cases where user typed the filename directly
    if not matched_doc:
        raw_q_lower = state['query'].lower()
        for d in get_vault_documents():
            fn = d["file_name"]
            fn_base = fn.lower().rsplit('.', 1)[0]
            if fn_base in raw_q_lower or fn.lower() in raw_q_lower:
                matched_doc = fn
                print(f"[ANALYST] Filename matched verbatim in query: '{fn}'")
                break

    # Apply implicit reference resolution if no explicit doc was found
    if not matched_doc and pre_resolved_doc:
        matched_doc = pre_resolved_doc

    # Override question_type to "summary" for meta-questions about a file's content.
    # extract_intent often classifies "what is this file about?" as "factual", which
    # skips direct retrieval. Forcing "summary" ensures all chunks are fetched.
    _META_FILE_SIGNALS = {
        "what is this", "what is the file", "tell me about this", "tell me about the",
        "what does this contain", "what does the file", "what is it about",
        "about this file", "about that file", "about this doc", "about the file",
        "what's in this", "what is in this", "describe this file", "describe the file",
        "overview of this", "context of this", "explain this file", "explain the file",
        "what kind of file", "what type of file", "what is this document",
        "what does this document", "about this document", "about the document",
    }
    if matched_doc and question_type not in ("comparison", "list") and \
            any(s in clean_q.lower() for s in _META_FILE_SIGNALS):
        question_type = "summary"
        print(f"[ANALYST] Meta-file query — overriding question_type to 'summary' for '{matched_doc}'")

    lang, query_en = _detect_and_translate(clean_q)

    # Implicit ref override — if implicit ref resolved matched_doc to a session-carried doc
    # but per-file finds a clearly dominant different doc at 0.85, prefer the content match.
    # Only fires when implicit ref actually determined matched_doc (matched_doc == pre_resolved_doc)
    # and not for comparison/summary where following session context is the correct behavior.
    if (implicit_ref_fired and matched_doc and matched_doc == pre_resolved_doc
            and question_type not in ("comparison", "summary")):
        verify_pf = find_relevant_docs_per_file(query_en, threshold=0.85, chunks_per_doc=6)
        if verify_pf:
            best_cs     = verify_pf[0]["cumulative_score"]
            second_cs   = verify_pf[1]["cumulative_score"] if len(verify_pf) >= 2 else 0.0
            best_verify = verify_pf[0]["file_name"]
            if best_verify != matched_doc and (len(verify_pf) == 1 or best_cs >= 2.0 * second_cs):
                matched_doc = best_verify
                implicit_ref_fired = False
                print(f"[ANALYST] Implicit ref override — per-file: '{best_verify}' (cs={best_cs:.2f}) over '{pre_resolved_doc}'")

    # Version comparison queries — fetch chunks from all relevant versions and compare
    _VERSION_COMPARE_SIGNALS = {
        "differ", "difference", "compare version", "what changed", "what's different",
        "vs v", "v1 vs", "vs v2", "old version vs", "how different", "changed between",
        "how has it changed", "compare the two versions", "compare both versions",
        "all versions", "versions the same", "all the same", "same across",
        "identical", "versions match", "are the versions", "versions differ",
        "versions different", "across versions", "between versions",
    }
    if matched_doc and any(s in clean_q.lower() for s in _VERSION_COMPARE_SIGNALS):
        import re as _re
        v_nums = sorted(set(int(m) for m in _re.findall(r'v(\d+)', clean_q.lower())))
        if len(v_nums) < 2:
            info = get_version_info(matched_doc)
            if info["count"] >= 2:
                v_nums = [v["version"] for v in info["versions"]]

        if len(v_nums) >= 2:
            ctx_parts = []
            for vn in v_nums:
                chunks = get_chunks_for_version(matched_doc, vn, query_en)
                if chunks:
                    ctx_parts.append(
                        f"=== {matched_doc} — Version {vn} ===\n"
                        + "\n".join(f"FACT: {c}" for c in chunks)
                    )
            if ctx_parts:
                ctx = "\n\n".join(ctx_parts)
                v_label = " vs ".join(f"v{v}" for v in v_nums)
                print(f"[ANALYST] Version comparison: {v_label} for '{matched_doc}'")
                return {
                    "source_type": "VersionDiff", "context": [ctx], "grouped_docs": "{}",
                    "gap": "", "lang": lang, "query_en": query_en,
                    "clean_question": clean_q, "search_pref": search_pref,
                    "format_pref": format_pref, "tone": tone,
                    "doc_filter": doc_filter, "version_pref": version_pref,
                    "matched_doc": matched_doc,
                    "question_type": question_type,
                "iterations": state.get("iterations", 0)
                }

    # Version metadata queries — answer directly from ChromaDB metadata, no semantic search needed
    _version_signals = {
        "how many versions", "version history", "versions does", "versions are there",
        "how many times", "updated how", "how often updated",
        "which version", "what version", "when was", "when were", "when did",
        "updated in", "uploaded in", "version from", "version was updated",
        "upload date", "upload time", "which one is latest", "which is the latest",
        "most recent version", "oldest version", "first version", "last version",
    }
    if matched_doc and any(s in clean_q.lower() for s in _version_signals):
        info = get_version_info(matched_doc)
        if info["count"] > 0:
            timestamps = [v["upload_timestamp"] for v in info["versions"]]
            all_same_date = len(set(timestamps)) == 1
            v_lines = [
                f"  - v{v['version']} — uploaded {v['upload_timestamp']}"
                + (" (current/latest)" if v["is_latest"] else " (older)")
                for v in info["versions"]
            ]
            same_date_note = (
                f" All {info['count']} versions share the same upload date ({timestamps[0]})."
                if all_same_date and info["count"] > 1 else ""
            )
            ctx = (
                f"FACT: '{matched_doc}' has {info['count']} version(s) in the vault.{same_date_note}\n"
                + "\n".join(v_lines)
                + f" | SOURCE: Internal: {matched_doc}"
            )
            print(f"[ANALYST] Version metadata query — {info['count']} version(s) found for '{matched_doc}'")
            return {
                "source_type": "Local", "context": [ctx], "grouped_docs": "{}",
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": matched_doc,
                "question_type": question_type,
                "iterations": state.get("iterations", 0)
            }

    # Multi-named-doc synthesis guard — when 3+ vault docs are explicitly named,
    # route directly to synthesize_node. The decomposer uses a single doc_filter
    # for ALL sub-questions, so every sub-question retrieves from the first named
    # doc only — causing wrong or hallucinated answers for all other named docs.
    # This guard intercepts before decompose_query fires.
    _vault_all = get_vault_documents()
    _named_in_q = [d for d in _vault_all if d["file_name"].lower() in clean_q.lower()]
    if len(_named_in_q) >= 3:
        _grouped_named = {}
        for _nd in _named_in_q:
            _fn = _nd["file_name"]
            _c = get_all_chunks_for_doc(_fn, max_chunks=5)
            if _c:
                _grouped_named[_fn] = _c
        if len(_grouped_named) >= 3:
            print(f"[ANALYST] Multi-named-doc synthesis — {len(_grouped_named)} docs")
            return {
                "source_type": "MultiDoc", "context": [], "grouped_docs": _json.dumps(_grouped_named),
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": _named_in_q[0]["file_name"],
                "question_type": question_type,
                "disc_topic": "introduction overview key findings",
                "iterations": state.get("iterations", 0)
            }

    # Query decomposition — handle multi-part questions before regular retrieval.
    # GUARD: skip when a single doc is already resolved (matched_doc set). Fragmenting a
    # single-doc query routes each fragment through weaker semantic-only retrieval and
    # floods the keyword sweep with noise words ("how LARGE is its vocabulary..."), missing
    # niche facts that the WHOLE query finds via local_graph_check's keyword sweep on the
    # resolved doc. Multi-doc queries (no single matched_doc) still decompose normally.
    sub_questions = decompose_query(clean_q) if not matched_doc else []
    if sub_questions:
        print(f"[ANALYST] Decomposed into {len(sub_questions)} sub-questions")

        # When no doc was explicitly named, identify the dominant document via per-file
        # evaluation so sub-questions search within the right doc rather than competing
        # for slots in the shared pool. Only fires when one doc is clearly dominant
        # (cumulative score ≥ 2× the next candidate) — conservative to avoid restricting
        # multi-doc queries. If user already named a doc (matched_doc set), use that.
        decompose_doc = matched_doc
        _decompose_cands = []  # saved when no dominant doc — reused in sub-q fallback (no extra calls)
        if not decompose_doc:
            pre_rel = find_relevant_docs_per_file(query_en, threshold=0.65, chunks_per_doc=8)
            if len(pre_rel) == 1:
                decompose_doc = pre_rel[0]["file_name"]
                print(f"[ANALYST] Decompose — single relevant doc: '{decompose_doc}'")
            elif len(pre_rel) >= 2:
                top_cs    = pre_rel[0]["cumulative_score"]
                second_cs = pre_rel[1]["cumulative_score"]
                if top_cs >= 2.0 * second_cs:
                    decompose_doc = pre_rel[0]["file_name"]
                    print(f"[ANALYST] Decompose — dominant doc: '{decompose_doc}' ({top_cs:.2f} vs {second_cs:.2f})")
                else:
                    _decompose_cands = [d["file_name"] for d in pre_rel[:3]]

        sub_contexts = []
        for i, sq in enumerate(sub_questions):
            sq_ctx = ""
            if decompose_doc:
                # Dominant doc confirmed — targeted per-doc search avoids shared-pool crowding
                _sq_chunks = search_top_chunks(sq, decompose_doc, n=8)
                if _sq_chunks:
                    sq_ctx = ("\n\n".join(f"FACT: {c}" for c in _sq_chunks)
                              + f"\n| SOURCE: Internal: {decompose_doc}")
                    print(f"[ANALYST] Part {i+1} answered from '{decompose_doc}' (targeted)")
            if not sq_ctx.strip() and _decompose_cands:
                # No dominant doc — try top candidates from full-query per-file scoring.
                # Reuses already-computed results; no extra ChromaDB calls.
                for _cand in _decompose_cands:
                    _cand_chunks = search_top_chunks(sq, _cand, n=6)
                    if _cand_chunks:
                        sq_ctx = ("\n\n".join(f"FACT: {c}" for c in _cand_chunks)
                                  + f"\n| SOURCE: Internal: {_cand}")
                        print(f"[ANALYST] Part {i+1} answered from '{_cand}' (candidate)")
                        break
            if not sq_ctx.strip():
                _, sq_ctx = local_graph_check(sq, doc_filter="", version_pref=version_pref)
                if sq_ctx.strip():
                    print(f"[ANALYST] Part {i+1} answered from vault")
                else:
                    print(f"[ANALYST] Part {i+1} — no local data found")
            if sq_ctx.strip():
                sub_contexts.append(f"=== Part {i+1}: {sq} ===\n{sq_ctx}")
        if sub_contexts:
            return {
                "source_type": "Decomposed", "context": ["\n\n".join(sub_contexts)], "grouped_docs": "{}",
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": matched_doc,
                "question_type": question_type,
                "iterations": state.get("iterations", 0)
            }

    # If user explicitly wants web only, skip local check
    if search_pref == "web_only":
        return {
            "source_type": "Web", "context": [], "gap": "",
            "lang": lang, "query_en": query_en,
            "clean_question": clean_q, "search_pref": search_pref,
            "format_pref": format_pref, "tone": tone,
            "doc_filter": doc_filter, "version_pref": version_pref,
            "matched_doc": matched_doc,
            "iterations": state.get("iterations", 0)
        }

    # Content-based doc routing
    _DISCOVERY_SIGNALS = {
        "which file", "which document", "which doc", "which paper",
        "which pdf", "where is this", "where can i find", "what file",
        "what document", "what paper", "which one has", "which one contains",
        "which one talks", "which one discusses",
        "how many files", "how many documents", "how many docs", "how many pdfs",
        "list all files", "list all documents", "list all docs",
    }
    # Use raw query for discovery detection — extract_intent strips the "which file" framing
    # from clean_q, making is_discovery False even when the user clearly asked a discovery question.
    is_discovery = any(s in state["query"].lower() for s in _DISCOVERY_SIGNALS)

    # Discovery queries always bypass the matched_doc gate.
    # Reason: extract_intent may set matched_doc from a topic word (e.g. "world bank" → some doc),
    # which would skip the vault-wide scan entirely and fall into local_graph_check + web cache.
    # Discovery is about FINDING docs — it must always scan the full vault.
    if is_discovery:
        # Strip discovery meta-phrases to embed the pure topic, not the meta-question structure.
        # "Which files mention climate change?" → "climate change"
        # extract_intent's clean_question may retain the "which files mention" framing; this
        # deterministic pass removes it so the embedding targets topic semantics rather than
        # meta-question space (which pushes semantically related docs like WEF / Paris Agreement
        # past the 0.80 threshold even when they are genuinely about the topic).
        # Uses \b word boundaries + explicit plural forms to avoid partial word removal
        # (e.g. "which file" substring match would leave "s" behind in "which files").
        _disc_raw = clean_q if clean_q else state["query"]
        _disc_work = _disc_raw.lower()
        _disc_strip_sigs = sorted(
            list(_DISCOVERY_SIGNALS) + [
                "which files", "which documents", "which papers", "which pdfs",
                "what files", "what documents", "what papers",
            ],
            key=len, reverse=True,
        )
        for _sig in _disc_strip_sigs:
            _disc_work = _re.sub(r'\b' + _re.escape(_sig) + r'\b', ' ', _disc_work)
        for _meta in ("in my vault", "in the vault", " mention ", " mentions ",
                      " contain ", " contains ", " discuss ", " discusses ",
                      " cover ", " covers ", " include ", " includes "):
            _disc_work = _disc_work.replace(_meta, " ")
        disc_query = " ".join(_disc_work.split()).strip(" ?.,") or _disc_raw
        # Expand short abbreviations so the embedding has meaningful signal and
        # keyword sweep fires correctly (requires words >= 4 chars to match chunks).
        # Static lookup first — instant, deterministic, no LLM cost.
        # LLM fallback handles unknown short terms not in the table.
        # _STATIC_EXPANSIONS is defined at module level.
        if len(disc_query) <= 4:
            _disc_q_lower = disc_query.lower().strip()
            if _disc_q_lower in _STATIC_EXPANSIONS:
                _expanded_static = _STATIC_EXPANSIONS[_disc_q_lower]
                print(f"[ANALYST] Discovery term expanded (static): '{disc_query}' → '{_expanded_static}'")
                disc_query = _expanded_static
            else:
                try:
                    _expanded = mini_llm.invoke(
                        "Expand this short abbreviation to its full proper name suitable for "
                        "searching research documents. Output ONLY the full name, nothing else.\n"
                        "Examples: 'nlp' → 'natural language processing'  "
                        "|  'rbi' → 'Reserve Bank of India'  |  'imf' → 'International Monetary Fund'\n"
                        f"Term: {disc_query}"
                    ).content.strip()
                    if _expanded and len(_expanded) > len(disc_query):
                        print(f"[ANALYST] Discovery term expanded (LLM): '{disc_query}' → '{_expanded}'")
                        disc_query = _expanded
                except Exception:
                    pass
        # Per-file evaluation: every doc is scored independently so no file can be
        # crowded out by other docs dominating a shared top-N result pool.
        # Both passes always run and merge — 1.20 used to fire only on zero results,
        # permanently missing files whose distance fell between 0.80 and 1.20.
        # Writer's adaptive verifier filters any false positives from the wider net.
        all_relevant_080 = find_relevant_docs_per_file(disc_query, threshold=0.80, chunks_per_doc=10)
        all_relevant_120 = find_relevant_docs_per_file(disc_query, threshold=1.20, chunks_per_doc=10)
        all_relevant_160 = find_relevant_docs_per_file(disc_query, threshold=1.60, chunks_per_doc=10)
        _seen = {d["file_name"] for d in all_relevant_080}
        all_relevant = list(all_relevant_080)
        for d in all_relevant_120:
            if d["file_name"] not in _seen:
                all_relevant.append(d)
                _seen.add(d["file_name"])
        _n_after_120 = len(all_relevant)
        for d in all_relevant_160:
            if d["file_name"] not in _seen:
                all_relevant.append(d)
                _seen.add(d["file_name"])
        # Keyword sweep: catches docs where exact query terms appear in chunk text
        # but semantic distance exceeds all thresholds (vocabulary mismatch).
        _disc_kws = [w.lower() for w in disc_query.split() if len(w) >= 4]
        if _disc_kws:
            try:
                global _kw_vault_cache
                _kw_total = vault_collection.count()
                if _kw_vault_cache[0] == _kw_total:
                    _kw_r = _kw_vault_cache[1]
                else:
                    _kw_r = vault_collection.get(
                        where={"is_latest": {"$eq": "true"}},
                        include=["documents", "metadatas"]
                    )
                    _kw_vault_cache = (_kw_total, _kw_r)
                _vault_map = {d["file_name"]: d for d in get_vault_documents()}
                for _kw_chunk, _kw_meta in zip(
                    _kw_r.get("documents") or [], _kw_r.get("metadatas") or []
                ):
                    _fn = _kw_meta.get("file_name", "")
                    if not _fn or _fn in _seen or _fn not in _vault_map:
                        continue
                    _cl = _kw_chunk.lower()
                    if all(kw in _cl or kw.rstrip("s") in _cl for kw in _disc_kws):
                        _seen.add(_fn)
                        _d = _vault_map[_fn]
                        all_relevant.append({
                            "file_name": _fn,
                            "version_number": _d.get("version_number", 1),
                            "best_distance": 1.61,
                            "cumulative_score": 0.05,
                            "sample_chunk": _kw_chunk[:120] + "...",
                        })
                        print(f"[ANALYST] Keyword sweep added: '{_fn}'")
            except Exception:
                pass

        if all_relevant:
            print(f"[ANALYST] Discovery — {len(all_relevant_080)} at 0.80, {_n_after_120 - len(all_relevant_080)} from 1.20, {len(all_relevant) - _n_after_120} from 1.60")
        else:
            # Both thresholds empty — return explicit "not found" so writer says
            # "nothing found" rather than falling through to history hallucination.
            print(f"[ANALYST] Discovery — no documents found for '{disc_query}'")
            ctx = (
                f"VAULT SEARCH RESULT (internal documents only — do NOT include web sources or chat history as files):\n"
                f"FACT: No internal documents were found matching this topic.\n"
                f"| SOURCE: Internal vault"
            )
            return {
                "source_type": "Local", "context": [ctx], "grouped_docs": "{}",
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": "",
                "question_type": question_type,
                "iterations": state.get("iterations", 0),
                "disc_topic": disc_query,
            }
        if all_relevant:
            matched_doc = all_relevant[0]["file_name"]
            print(f"[ANALYST] Discovery — {len(all_relevant)} relevant doc(s) found, best: '{matched_doc}'")
            if len(all_relevant) == 1:
                ctx = (
                    f"VAULT SEARCH RESULT (internal documents only — do NOT include web sources or chat history as files):\n"
                    f"FACT: Exactly 1 internal document contains relevant information: **{matched_doc}**."
                    f"\n| SOURCE: Internal: {matched_doc}"
                )
            else:
                lines = [
                    f"VAULT SEARCH RESULT (internal documents only — do NOT include web sources or chat history as files):",
                    f"FACT: Exactly {len(all_relevant[:15])} internal document(s) contain relevant information (ranked by relevance):"
                ]
                for i, d in enumerate(all_relevant[:15]):
                    lines.append(f"  {i+1}. {d['file_name']}")
                lines.append(f"| SOURCE: Internal vault")
                ctx = "\n".join(lines)
            return {
                "source_type": "Local", "context": [ctx], "grouped_docs": "{}",
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": matched_doc,
                "question_type": question_type,
                "iterations": state.get("iterations", 0),
                "disc_topic": disc_query,
            }
        else:
            # Fallback: cumulative semantic scoring
            best = identify_best_doc_for_query(disc_query)
            if best:
                matched_doc = best
                print(f"[ANALYST] Discovery fallback — best: '{matched_doc}'")

    # Two explicit documents named — user said "compare X report with Y report".
    # Both resolved above. Only fires when extract_intent found BOTH doc names.
    # Does NOT fire for single-doc comparisons (e.g. WIPO country comparisons).
    if question_type == "comparison" and matched_doc and matched_doc_2 and matched_doc != matched_doc_2:
        chunks1 = get_all_chunks_for_doc(matched_doc,   max_chunks=8)
        chunks2 = get_all_chunks_for_doc(matched_doc_2, max_chunks=8)
        if chunks1 and chunks2:
            print(f"[ANALYST] Two-doc comparison: '{matched_doc}' + '{matched_doc_2}'")
            return {
                "source_type": "MultiDoc", "context": [], "grouped_docs": _json.dumps({matched_doc: chunks1, matched_doc_2: chunks2}),
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": matched_doc,
                "question_type": question_type,
                "iterations": state.get("iterations", 0)
            }

    # Broad multi-doc scan: "summarize key findings across all AI papers" /
    # "what do all the climate files say?" — queries that span multiple docs
    # but carry no discovery signal ("which file") and no specific doc name.
    # Routes through per-file evaluation (every doc scored independently, no
    # shared-pool crowding) then packages all relevant docs into grouped_docs
    # for MultiDoc synthesis. Guards: no matched_doc, not comparison.
    _BROAD_DOC_WORDS = {"papers", "paper", "files", "file", "documents",
                        "document", "docs", "doc", "pdfs", "pdf",
                        "reports", "report"}
    _q_lower_broad = state["query"].lower()
    is_broad_scan = (
        not matched_doc
        and question_type != "comparison"
        and (
            any(s in _q_lower_broad for s in ("across all", "from all",
                                               "every paper", "every file",
                                               "every document", "every doc"))
            or ("all " in _q_lower_broad
                and any(w in _q_lower_broad for w in _BROAD_DOC_WORDS))
        )
    )
    if is_broad_scan:
        # Extract core topic before per-file search — the full query ("Summarize key findings
        # across all AI papers") is a task description that embeds toward survey articles,
        # not individual paper chunks. Searching on "AI research" finds all 9 AI papers;
        # searching on the verbose query finds maybe 1. Fallback to query_en if LLM fails.
        _TOPIC_ACTION_WORDS = {
            "summary", "summarize", "summarization", "summarizing",
            "analyze", "analysis", "analyzing", "compare", "comparison",
            "comparing", "review", "overview", "describe", "explain",
            "tell", "show", "give", "list", "find", "what",
        }
        try:
            _broad_topic = mini_llm.invoke(
                "Extract the SUBJECT DOMAIN (1-4 words) the user wants to search across documents.\n"
                "RULES:\n"
                "- Output the TOPIC (what the documents are ABOUT), NOT the action verb\n"
                "- NEVER output: 'summarize', 'summary', 'analyze', 'compare', 'review', 'overview'\n"
                "- Prefer full forms: 'AI' → 'artificial intelligence', 'ML' → 'machine learning'\n"
                "Examples:\n"
                "  'Summarize key findings across all AI papers' → 'artificial intelligence'\n"
                "  'Analyze all climate change documents' → 'climate change'\n"
                "  'What do all the finance reports say?' → 'finance'\n"
                "  'Give me summaries of all the LLM papers' → 'large language model'\n"
                "  'Compare all NLP models' → 'natural language processing'\n"
                f"Query: {query_en}"
            ).content.strip()
            # Guard: if action word slipped through, fall back to raw query for word extraction
            _broad_search_q = (
                query_en if (not _broad_topic or _broad_topic.lower() in _TOPIC_ACTION_WORDS)
                else _broad_topic
            )
        except Exception:
            _broad_search_q = query_en
        # Expand any acronyms inside the extracted topic and strip generic filler words
        # ("AI files" → "artificial intelligence", "ML papers" → "machine learning").
        _generic_topic_words = {"files", "file", "papers", "paper", "documents",
                                "document", "docs", "doc", "reports", "report",
                                "all", "every", "each", "any", "some", "many",
                                "various", "different", "my", "your", "our"}
        _broad_words = [
            _STATIC_EXPANSIONS.get(w.lower(), w)
            for w in _broad_search_q.split()
            if w.lower() not in _generic_topic_words
        ]
        if not _broad_words:
            print(f"[ANALYST] Broad scan: no meaningful topic after filtering — skipping")
        else:
            _broad_search_q = " ".join(_broad_words)
            print(f"[ANALYST] Broad scan topic: '{_broad_search_q}'")
            # Threshold ladder — abstract topics have higher L2 distances; widen until enough docs found.
            _broad_pf = find_relevant_docs_per_file(_broad_search_q, threshold=0.80, chunks_per_doc=6)
            if len(_broad_pf) < 2:
                _broad_pf = find_relevant_docs_per_file(_broad_search_q, threshold=1.20, chunks_per_doc=6)
            if len(_broad_pf) < 2:
                _broad_pf = find_relevant_docs_per_file(_broad_search_q, threshold=1.60, chunks_per_doc=6)
            _broad_pf = _broad_pf[:15]  # cap — prevents all-docs scenario at threshold 1.60
            if len(_broad_pf) >= 2:
                _grouped = {}
                _n_broad = len(_broad_pf)
                _chunks_per_doc = max(2, 30 // _n_broad)
                for _d in _broad_pf:
                    _fn = _d["file_name"]
                    _c = (search_top_chunks(_broad_search_q, _fn, n=_chunks_per_doc)
                          or get_all_chunks_for_doc(_fn, max_chunks=_chunks_per_doc))
                    if _c:
                        _grouped[_fn] = _c
                if len(_grouped) >= 2:
                    print(f"[ANALYST] Broad multi-doc scan — {len(_grouped)} docs, {_chunks_per_doc} chunks/doc")
                    return {
                        "source_type": "MultiDoc", "context": [], "grouped_docs": _json.dumps(_grouped),
                        "gap": "", "lang": lang, "query_en": query_en,
                        "clean_question": clean_q, "search_pref": search_pref,
                        "format_pref": format_pref, "tone": tone,
                        "doc_filter": doc_filter, "version_pref": version_pref,
                        "matched_doc": _broad_pf[0]["file_name"],
                        "question_type": question_type,
                        "disc_topic": _broad_search_q,
                        "iterations": state.get("iterations", 0)
                    }

    if not matched_doc:
        if question_type == "comparison":
            # Comparison queries need up to 2 docs — relax gap condition, take top 2
            relevant_docs = find_relevant_docs(query_en, threshold=0.70)
            if len(relevant_docs) >= 2:
                doc1 = relevant_docs[0]["file_name"]
                doc2 = relevant_docs[1]["file_name"]
                chunks1 = get_all_chunks_for_doc(doc1, max_chunks=8)
                chunks2 = get_all_chunks_for_doc(doc2, max_chunks=8)
                if chunks1 and chunks2:
                    grouped = _json.dumps({doc1: chunks1, doc2: chunks2})
                    print(f"[ANALYST] Comparison — dual-doc retrieval: '{doc1}' + '{doc2}'")
                    return {
                        "source_type": "MultiDoc", "context": [], "grouped_docs": grouped,
                        "gap": "", "lang": lang, "query_en": query_en,
                        "clean_question": clean_q, "search_pref": search_pref,
                        "format_pref": format_pref, "tone": tone,
                        "doc_filter": doc_filter, "version_pref": version_pref,
                        "matched_doc": doc1,
                        "question_type": question_type,
                        "iterations": state.get("iterations", 0)
                    }
            elif len(relevant_docs) == 1:
                matched_doc = relevant_docs[0]["file_name"]
                print(f"[ANALYST] Comparison — single doc found: '{matched_doc}'")
        else:
            # Skip gap-condition auto-ID when user named a doc but find_matching_doc couldn't
            # resolve it. Semantic overlap from the shared pool would route to a wrong doc;
            # per-file analysis below evaluates every doc independently so the right one wins.
            unresolved_doc_ref = bool(doc_filter and not matched_doc)
            if not unresolved_doc_ref:
                relevant_docs = find_relevant_docs(query_en, threshold=0.70)
                if relevant_docs:
                    best_dist   = relevant_docs[0]["best_distance"]
                    second_dist = relevant_docs[1]["best_distance"] if len(relevant_docs) > 1 else 1.0
                    gap         = second_dist - best_dist
                    if len(relevant_docs) == 1 or (best_dist < 0.55 and gap >= 0.08):
                        matched_doc = relevant_docs[0]["file_name"]
                        print(f"[ANALYST] Doc auto-identified: '{matched_doc}' (dist={best_dist:.2f})")
                    else:
                        print(f"[ANALYST] {len(relevant_docs)} relevant docs found")

    # Per-file cumulative fallback — runs only when routing above left matched_doc empty.
    # Catches verbose-chunk docs (e.g. Budget_Speech paragraphs) whose best chunk embeds
    # at 0.70-0.80 and is therefore invisible to find_relevant_docs at 0.70 threshold.
    # Strict 2× ratio guard prevents incorrectly restricting genuinely multi-doc queries.
    # Excluded for comparison queries — those have their own dual-doc synthesis logic.
    # pf_top_doc saves the leading candidate even when ratio < 2×, so the vault-miss retry
    # below can target it specifically rather than searching the full pool again.
    pf_top_doc = None
    pf_top_doc_from_extended = False
    if not matched_doc and question_type != "comparison":
        pf = find_relevant_docs_per_file(query_en, threshold=0.80, chunks_per_doc=10)
        if pf:
            top_cs    = pf[0]["cumulative_score"]
            second_cs = pf[1]["cumulative_score"] if len(pf) >= 2 else 0.0
            if len(pf) == 1 or top_cs >= 2.0 * second_cs:
                matched_doc = pf[0]["file_name"]
                print(f"[ANALYST] Per-file fallback — dominant doc: '{matched_doc}' (cs={top_cs:.2f})")
            else:
                pf_top_doc = pf[0]["file_name"]
                print(f"[ANALYST] Per-file hint: '{pf_top_doc}' (ratio {top_cs:.2f}/{second_cs:.2f} < 2×)")
        else:
            # 0.80 pass returned nothing — extended search at 0.85 catches docs whose
            # best chunk embeds at 0.81–0.85 (e.g. niche papers, specialised vocabulary).
            # Lower dominance ratio (1.5×) reflects wider tolerance but still guards
            # against incorrectly restricting genuinely multi-doc queries.
            pf2 = find_relevant_docs_per_file(query_en, threshold=0.85, chunks_per_doc=10)
            if pf2:
                top_cs2    = pf2[0]["cumulative_score"]
                second_cs2 = pf2[1]["cumulative_score"] if len(pf2) >= 2 else 0.0
                if len(pf2) == 1 or top_cs2 >= 1.5 * second_cs2:
                    pf_top_doc = pf2[0]["file_name"]
                    pf_top_doc_from_extended = True
                    print(f"[ANALYST] Extended per-file (0.85): hint='{pf_top_doc}' (cs={top_cs2:.2f})")

    # Literal-term fallback — semantic per-file found no candidate doc. The embedding model
    # is blind to rare acronyms/tokens (e.g. "RLAIF" does not embed near Constitutional AI's
    # chunks, though the literal string is there), so scan the vault for LITERAL occurrences of
    # the query's distinctive terms. FAILURE-PATH ONLY: gated on no matched_doc AND no pf_top_doc,
    # so it can never override a successful semantic match — it only acts on queries that would
    # otherwise miss. local_graph_check(doc_filter=...) then runs its own keyword sweep on the doc.
    if not matched_doc and not pf_top_doc and question_type != "comparison":
        _lit_doc = find_doc_by_literal_terms(query_en)
        if _lit_doc:
            matched_doc = _lit_doc
            print(f"[ANALYST] Literal-term resolve → '{matched_doc}' (semantic found nothing)")

    # Direct retrieval path — fires for implicit refs and explicit summary requests.
    # Uses get_all_chunks_for_doc (sequential get, no n_results cap) to cover the full document.
    # Bypasses local_graph_check Phase 0 (n_results=15 shared pool) entirely.
    use_direct_retrieval = (
        implicit_ref_fired or
        (matched_doc and question_type == "summary")
    )
    if use_direct_retrieval and matched_doc:
        if question_type == "summary":
            # Summary: sequential coverage of the full doc — position matters, not relevance
            direct_chunks = get_all_chunks_for_doc(matched_doc, max_chunks=25)
        else:
            # Factual/analytical/list: semantic search within the doc so niche sections
            # (e.g. quantization on page 45 of a 92-page paper) are found regardless of
            # where they appear. get_all_chunks_for_doc only covers the first N pages.
            direct_chunks = search_top_chunks(query_en, matched_doc, n=20)
        if direct_chunks:
            flat_ctx = "\n\n".join(
                f"FACT: {chunk}" for chunk in direct_chunks
            ) + f"\n| SOURCE: Internal: {matched_doc}"
            print(f"[ANALYST] Direct doc retrieval for '{matched_doc}' ({len(direct_chunks)} chunks, type={question_type})")
            return {
                "source_type": "Local", "context": [flat_ctx], "grouped_docs": "{}",
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": matched_doc,
                "question_type": question_type,
                "iterations": state.get("iterations", 0)
            }

    retrieval_query = query_en
    res, ctx = local_graph_check(retrieval_query, doc_filter=matched_doc, version_pref=version_pref)

    # Vault-miss retry: if the shared-pool search failed (or returned nothing useful) AND
    # per-file analysis identified a likely document, re-run local_graph_check restricted
    # to that document. This catches cases where the query embeds at dist 0.70-0.80 and
    # Budget_Speech (or similar verbose docs) ranks outside the shared-pool top-15.
    if pf_top_doc and not matched_doc:
        pf_doc_in_ctx = pf_top_doc in ctx
        vault_miss = (res == "SEARCH_REQUIRED") or (
            res == "LOCAL_FOUND" and "Internal:" not in ctx
        ) or (pf_top_doc_from_extended and res == "LOCAL_FOUND" and not pf_doc_in_ctx)
        if vault_miss:
            print(f"[ANALYST] Vault miss — retrying in per-file hint doc: '{pf_top_doc}'")
            res2, ctx2 = local_graph_check(retrieval_query, doc_filter=pf_top_doc, version_pref=version_pref)
            if res2 == "LOCAL_FOUND":
                res, ctx = res2, ctx2
                matched_doc = pf_top_doc

    if res == "LOCAL_FOUND" and ctx.strip():
        elapsed = time.time() - start

        # If matched_doc is still empty, extract the most-cited doc from context
        # so the badge and history show which doc actually answered the question
        if not matched_doc:
            doc_counts: dict = {}
            for line in ctx.split("\n"):
                if "Internal:" in line:
                    try:
                        fn = line.split("Internal:")[1].split("(")[0].strip()
                        doc_counts[fn] = doc_counts.get(fn, 0) + 1
                    except Exception:
                        pass
            if doc_counts:
                matched_doc = max(doc_counts, key=doc_counts.get)
                print(f"[ANALYST] Primary doc inferred from context: '{matched_doc}'")

        # Check how many distinct docs contributed — if 2+, use multi-doc synthesis
        grouped = get_chunks_by_doc(retrieval_query, doc_filter=matched_doc if doc_filter else None, version_pref=version_pref)
        if len(grouped) >= 2:
            print(f"[ANALYST] Multi-doc query — {len(grouped)} documents relevant: {list(grouped.keys())}")
            return {
                "source_type": "MultiDoc", "context": [ctx], "grouped_docs": _json.dumps(grouped),
                "gap": "", "lang": lang, "query_en": query_en,
                "clean_question": clean_q, "search_pref": search_pref,
                "format_pref": format_pref, "tone": tone,
                "doc_filter": doc_filter, "version_pref": version_pref,
                "matched_doc": matched_doc,
                "question_type": question_type,
                "iterations": state.get("iterations", 0)
            }

        print(f"[ANALYST] Local data found in {elapsed:.2f}s")
        return {
            "source_type": "Local", "context": [ctx], "grouped_docs": "{}",
            "gap": "", "lang": lang, "query_en": query_en,
            "clean_question": clean_q, "search_pref": search_pref,
            "format_pref": format_pref, "tone": tone,
            "doc_filter": doc_filter, "version_pref": version_pref,
            "matched_doc": matched_doc,
            "iterations": state.get("iterations", 0)
        }

    # Local search found nothing.
    # If user explicitly asked for web, go there. Otherwise ask for confirmation.
    if search_pref == "web_only":
        route = "Web"
    else:
        route = "NotFound"  # local-first: ask user before hitting web

    return {
        "source_type": route, "context": [], "grouped_docs": "{}",
        "gap": "", "lang": lang, "query_en": query_en,
        "clean_question": clean_q, "search_pref": search_pref,
        "format_pref": format_pref, "tone": tone,
        "doc_filter": doc_filter, "version_pref": version_pref,
        "matched_doc": matched_doc,
        "iterations": state.get("iterations", 0)
    }

def research_node(state: AgentState):
    current_iter = state.get("iterations", 0) + 1
    print(f"[RESEARCHER] Search Attempt {current_iter}...")
    start = time.time()

    try:
        query_lower = state.get('query_en', state['query']).lower()
        has_year = any(yr in query_lower for yr in ["2024", "2025", "2026", "2027"])
        time_signals = {"latest", "recent", "current", "today", "now", "this year", "this week",
                        "this month", "score", "standings", "winner", "results", "update", "news"}
        is_time_sensitive = any(sig in query_lower for sig in time_signals) or has_year
        date_suffix = " April 2026" if is_time_sensitive and not has_year else ""

        try:
            gap = state.get("gap", "")
            prev_answer = state.get("answer", "")

            q_en = state.get('query_en', state['query'])
            if gap and current_iter == 1:
                rewrite_prompt = (
                    f"Convert to a focused web search query (8 words max). Output ONLY the query.\n\n"
                    f"Original question: {q_en}\n"
                    f"Missing information: {gap}"
                )
            elif "PARTIAL_INFO" in prev_answer:
                rewrite_prompt = (
                    f"A previous search gave this incomplete answer:\n{prev_answer}\n\n"
                    f"The user asked: {q_en}\n\n"
                    f"Generate a specific web search query (10 words max) to find the MISSING information. "
                    f"Focus on the gaps, not what was already found. Output ONLY the search query."
                )
            else:
                rewrite_prompt = (
                    f"Convert the following question into a concise web search query (10 words max). "
                    f"Remove instruction words like 'describe', 'provide', 'identify', 'explain'. "
                    f"Keep the core subject, key terms, and any dates or proper nouns. "
                    f"Output ONLY the search query, nothing else.\n\nQuestion: {q_en}"
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

def synthesize_node(state: AgentState):
    import json as _json
    print("[SYNTHESIZER] Multi-doc synthesis — generating per-document answers...")
    grouped = _json.loads(state.get("grouped_docs", "{}"))
    q_en = state.get('query_en', state['query'])

    # If the query is vague (e.g. "tell me about both", "summarize both"),
    # replace it with a rich per-doc question so mini_llm has a real task.
    _VAGUE_SIGNALS = {
        "tell me about", "what about", "about both", "about them",
        "about all", "summarize both", "summarize them", "summarize all",
        "overview of both", "both of them", "both docs",
        # Broad multi-doc signals: per-doc mini_llm would say NOT_COVERED if asked
        # "across all AI papers" against a single doc's chunks — replace with the
        # generic per-doc summary prompt so each doc's own findings are extracted.
        "across all", "from all", "every paper", "every file",
        "every document", "every doc", "all papers", "all documents",
        "all files", "all docs", "all reports",
    }
    q_lower = q_en.lower()
    # Force broad mode for 5+ docs regardless of query wording — per-doc mini_llm
    # calls are unreliable for large sets (rate limits, NOT_COVERED false negatives).
    _n_docs = len(grouped)
    is_vague = (
        _n_docs >= 5
        or len(q_lower.split()) <= 5
        or any(s in q_lower for s in _VAGUE_SIGNALS)
        # Comparisons ALWAYS take the flat-context path (handled in the comparison branch
        # below), never the per-doc gpt-4o-mini extractor. That extractor is unreliable on
        # messy spec/figure chunks — it drops exact values (returns NOT_COVERED) or misreads
        # figure axes (e.g. a perplexity tick "1.8" read as "1.8T tokens"). The 70B writer
        # extracts exact figures from both docs' own chunks far more reliably.
        or state.get("question_type") == "comparison"
    )
    per_doc_q = (
        "Provide a complete summary of this document: what is it about, "
        "what are its main topics and key findings, and what is its purpose?"
        if is_vague else q_en
    )

    # Dynamic chunk cap: total budget of 45 distributed evenly, floor of 3.
    # 5 docs → 9 chunks each, 13 docs → 3 chunks each, 15 docs → 3 chunks each.
    _syn_chunks_per_doc = max(3, 45 // _n_docs)

    # ── Broad summary mode ────────────────────────────────────────────────────
    # For vague/overview queries ("summarize all AI papers", "give an overview"),
    # skip per-doc mini_llm calls entirely. Those calls fail consistently because:
    # (a) search_top_chunks("Provide a complete summary...") fetches bibliography/
    #     acknowledgment chunks that contain overview language but no real content,
    #     causing mini_llm to return NOT_COVERED; (b) N mini_llm calls for N docs
    #     risks hitting rate limits.
    # Instead, fetch semantically relevant chunks per doc and pass as rich flat
    # context — writer synthesizes everything in one main_llm call.
    if is_vague:
        print(f"[SYNTHESIZER] Broad summary mode — building flat context ({_n_docs} docs)")
        _disc_q = state.get("disc_topic") or q_en
        _qt = state.get("question_type", "factual")
        _flat_parts = []
        for doc_name, chunks in grouped.items():
            if _qt == "comparison":
                # Comparison needs each entity's OWN specs in BOTH columns. Two problems
                # otherwise: (1) 22 chunks/doc × 2 docs blows past the writer's 24k char cap,
                # starving the second doc → its column comes back "not stated"; (2) the
                # comparison query skews retrieval toward the OTHER entity's mentions.
                # Fix: bound chunks so both docs fit, and lead with positional (abstract/intro)
                # chunks — where parameter counts, scale, and architecture live — then add
                # semantic matches. Keeps both columns concrete and symmetric.
                _cap = max(6, 24 // _n_docs)
                _pos = chunks[:max(4, _cap // 2)]
                # The raw comparison question ranks figure-bearing chunks too low
                # (proven: misses 2.0T / 4096 / 128k-vocab) -> writer says "Not specified"
                # or fabricates. Augment with a generic spec-seeking query; interleave so
                # neither intent is starved; then merge/cap EXACTLY as before (same count).
                from itertools import zip_longest
                _SPEC_MAGNET = (
                    "model parameters number of tokens trained context length "
                    "sequence length vocabulary size tokenizer architecture"
                )
                _sem_q   = search_top_chunks(_disc_q, doc_name, n=_cap)
                _sem_mag = search_top_chunks(_SPEC_MAGNET, doc_name, n=_cap)
                _sem, _sem_seen = [], set()
                for _a, _b in zip_longest(_sem_q, _sem_mag):
                    for _x in (_a, _b):
                        if _x is not None and _x not in _sem_seen:
                            _sem_seen.add(_x)
                            _sem.append(_x)
                _seen, _merged = set(), []
                for _c in (_pos + _sem):
                    if _c not in _seen:
                        _seen.add(_c)
                        _merged.append(_c)
                _use = _merged[:_cap]
            else:
                _top = search_top_chunks(_disc_q, doc_name, n=_syn_chunks_per_doc)
                _use = _top if _top else chunks[:_syn_chunks_per_doc]
            _flat_parts.append(f"--- {doc_name} ---\n" + "\n\n".join(_use))
        return {"context": ["\n\n".join(_flat_parts)], "source_type": "MultiDoc"}

    # ── Specific question mode ────────────────────────────────────────────────
    # For focused questions ("what do all AI papers say about attention?"),
    # run per-doc mini_llm for precise per-document answers.
    mini_answers = []
    for doc_name, chunks in grouped.items():
        _top = search_top_chunks(q_en, doc_name, n=_syn_chunks_per_doc)
        chunks_text = "\n\n".join(_top if _top else chunks[:_syn_chunks_per_doc])
        try:
            mini = mini_llm.invoke(
                f"Based ONLY on the following excerpts from '{doc_name}', "
                f"answer this question as directly as possible. "
                f"If the excerpts don't contain relevant information, reply with exactly: NOT_COVERED\n\n"
                f"Excerpts:\n{chunks_text}\n\n"
                f"Question: {q_en}"
            ).content.strip()
            if mini and mini != "NOT_COVERED":
                mini_answers.append(f"**From {doc_name}:**\n{mini}")
                print(f"[SYNTHESIZER] Got answer from: {doc_name}")
            else:
                print(f"[SYNTHESIZER] NOT_COVERED: {doc_name}")
        except Exception as _e:
            print(f"[SYNTHESIZER] Error ({doc_name}): {type(_e).__name__}: {str(_e)[:80]}")

    if mini_answers:
        combined = "\n\n---\n\n".join(mini_answers)
        return {"context": [combined], "source_type": "MultiDoc"}

    # Fallback — build flat context from grouped so writer has real content.
    print("[SYNTHESIZER] Fallback to flat context.")
    _flat_parts = []
    for _d, _cks in grouped.items():
        _ftop = search_top_chunks(q_en, _d, n=_syn_chunks_per_doc)
        _flat_parts.append(f"--- {_d} ---\n" + "\n\n".join(_ftop if _ftop else _cks[:_syn_chunks_per_doc]))
    return {"context": ["\n\n".join(_flat_parts)], "source_type": "MultiDoc"}


def writer_node(state: AgentState):
    import re as _re
    print("[WRITER] Drafting Answer...")

    context_parts = state['context']
    context_str   = "\n".join(context_parts)

    # ── Discovery enrichment + content verification ──────────────────────────
    # Global fix: when analyst returns a VAULT SEARCH RESULT (file name list only),
    # (1) run a secondary search at 0.70 threshold to catch files the 0.80 pass missed,
    # (2) fetch actual chunks from every candidate file,
    # (3) use mini_llm to deterministically verify which file BEST answers the query.
    # This makes discovery answers content-grounded and consistent across all queries.
    if "VAULT SEARCH RESULT" in context_str and "FACT:" in context_str:
        # _disc_embed: stripped topic (from disc_topic) — used ONLY for embedding + keyword searches.
        #   Using raw query here would embed "which files mention X" meta-structure, pushing papers
        #   past the similarity threshold even when they're genuinely about the topic.
        # _disc_verifier_q: raw user query — used ONLY for the verifier prompt.
        #   Needed so the adaptive standard can detect "mention" framing: "which files mention X"
        #   → any-occurrence YES; "what files discuss X" → strict direct-discussion YES.
        #   These two variables must NEVER be swapped — each has a specific purpose.
        _disc_embed = state.get("disc_topic") or state.get("query_en") or state.get("query", "")
        _disc_verifier_q = state.get("query") or state.get("query_en") or _disc_embed

        # Files already found by discovery (0.80 threshold)
        found_files = _re.findall(r'^\s*\d+\.\s+(.+?\.(?:pdf|docx|txt))\s*$', context_str, _re.IGNORECASE | _re.MULTILINE)
        if not found_files:
            found_files = _re.findall(r'\*\*(.+?\.(?:pdf|docx|txt))\*\*', context_str, _re.IGNORECASE)

        # Filename keyword fallback — catches docs whose verbose/paragraph-style chunks
        # embed with higher distance against short discovery queries (e.g. Budget_Speech
        # paragraphs vs "india's budget"). Checks filename tokens directly so any doc
        # with a descriptive name is included regardless of chunk embedding quality.
        _fn_stopwords = {
            "which", "file", "files", "have", "info", "about", "what", "does",
            "list", "show", "tell", "find", "give", "document", "documents",
            "information", "mention", "discusses", "covers", "contains", "related",
        }
        disc_kw = {w.strip("'-.").lower() for w in _disc_embed.split()
                   if len(w.strip("'-.")) >= 4} - _fn_stopwords
        if disc_kw:
            for d in get_vault_documents():
                fn = d["file_name"]
                if fn in found_files:
                    continue
                fn_tokens = set(fn.lower().replace("_", " ").replace("-", " ").replace(".", " ").split())
                if disc_kw & fn_tokens:
                    found_files.append(fn)
                    print(f"[WRITER] Filename keyword fallback added: '{fn}'")

        # Fetch chunks from every candidate — no cap, 3 chunks each.
        # Batched verifier (groups of 15) keeps prompt size manageable regardless
        # of how many docs the analyst finds.
        file_chunks = {}
        for fn in found_files:
            chunks = (search_top_chunks(_disc_embed, fn, n=3)
                      if _disc_embed else get_all_chunks_for_doc(fn, max_chunks=3))
            if chunks:
                file_chunks[fn] = chunks

        if file_chunks:
            # Multi-file relevance verifier — filters to only genuinely relevant files.
            # Ask YES/NO per file; filter found_files to only the YES set.
            # This eliminates irrelevant files added by the loose secondary/keyword passes
            # while keeping all genuinely relevant ones regardless of ranking.
            if _disc_verifier_q:
                file_list = list(file_chunks.keys())
                # Adaptive standard: "which files MENTION X" → any occurrence counts.
                # Other queries → strict "directly discusses" standard.
                _mention_words = {"mention", "mentions", "contain", "contains",
                                  "cover", "covers", "include", "includes", "have"}
                _is_mention_q = bool(_mention_words & set(_disc_verifier_q.lower().split()))
                _verifier_standard = (
                    "output exactly 'YES' if the content mentions or references this topic in ANY way "
                    "(even a single sentence counts), or 'NO' ONLY if the topic is completely absent."
                    if _is_mention_q else
                    "output exactly 'YES' if its content directly discusses the question's topic, "
                    "or 'NO' if it is unrelated or only mentions the topic in passing."
                )
                # Batched verifier — groups of 15 keep prompt size reliable regardless
                # of how many docs the analyst found. Results from all batches combined.
                _BATCH = 15
                verified = []
                for _b0 in range(0, len(file_list), _BATCH):
                    _batch = file_list[_b0: _b0 + _BATCH]
                    _exc = [
                        f"FILE {i+1}: {fn}\n" + "\n".join(f"  - {c[:300]}" for c in file_chunks[fn][:4])
                        for i, fn in enumerate(_batch)
                    ]
                    _vp = (
                        f"For each file below, {_verifier_standard}\n"
                        f"Output ONLY one YES or NO per line, same order as files. Nothing else.\n\n"
                        f"QUESTION: {_disc_verifier_q}\n\n"
                        + "\n\n".join(_exc)
                    )
                    try:
                        _raw = mini_llm.invoke([HumanMessage(content=_vp)]).content.strip()
                        _vds = [ln.strip().upper() for ln in _raw.split('\n')
                                if ln.strip().upper() in ('YES', 'NO')]
                        if len(_vds) == len(_batch):
                            verified.extend(fn for fn, v in zip(_batch, _vds) if v == 'YES')
                        else:
                            # Count mismatch — include whole batch (non-fatal)
                            verified.extend(_batch)
                            print(f"[WRITER] Verifier batch mismatch — including all {len(_batch)}")
                    except Exception:
                        verified.extend(_batch)  # failure is non-fatal
                if verified:
                    excluded = len(file_list) - len(verified)
                    found_files = [f for f in found_files if f in verified]
                    print(f"[WRITER] Verifier: {len(verified)}/{len(file_list)} relevant, excluded {excluded}")
                else:
                    print(f"[WRITER] Verifier said NO to all — keeping top-ranked")

            # Always rebuild context header so count matches what verifier approved.
            # Old code only rebuilt for 2+ files, leaving "Exactly 3 documents" in header
            # even after filtering down to 1 — this fixes that misleading mismatch.
            found_with_chunks = [fn for fn in found_files if fn in file_chunks]
            if found_with_chunks:
                hdr_lines = [
                    "VAULT SEARCH RESULT (internal documents only — do NOT include web sources or chat history as files):",
                    f"FACT: Exactly {len(found_with_chunks)} internal document(s) contain relevant information (ranked by relevance):",
                ]
                for i, fn in enumerate(found_with_chunks):
                    hdr_lines.append(f"  {i+1}. {fn}")
                hdr_lines.append("| SOURCE: Internal vault")
                context_str = "\n".join(hdr_lines)
                print(f"[WRITER] Context header rebuilt: {len(found_with_chunks)} file(s)")

            # Build enriched context: verified ranking + actual chunk content
            chunk_blocks = "\n".join(
                f"\n--- CONTENT FROM: {fn} ---\n" + "\n".join(f"  • {c[:400]}" for c in file_chunks[fn])
                for fn in found_files if fn in file_chunks
            )
            context_str = (
                context_str
                + f"\n\nVERIFIED RANKING (most relevant first): {', '.join(found_files)}"
                + "\n\nDOCUMENT CONTENT (use to answer the question — top-ranked file is most relevant):\n"
                + chunk_blocks
            )
            print(f"[WRITER] Discovery enriched: {len(file_chunks)} file(s), top='{found_files[0] if found_files else '?'}')")

    if len(context_str) > 24000:
        context_str = context_str[:24000]

    source_type   = state.get("source_type", "Web")
    has_local_context = any(
        "Internal:" in c or "| SOURCE:" in c
        for c in state.get("context", [])
    )
    is_hybrid = source_type == "Web" and has_local_context

    lang          = state.get("lang", "en")
    tone          = state.get("tone", "neutral")
    format_pref   = state.get("format_pref", "no_preference")
    question_type = state.get("question_type", "factual")
    history       = state.get("history", [])

    lang_instruction = (
        "\nLANGUAGE: The user wrote in Hinglish. Write your answer in Hinglish "
        "(mix of Hindi and English, using English script — no Devanagari). "
        "Technical terms, URLs, and proper nouns stay in English."
        if lang == "hinglish" else ""
    )

    tone_instruction = {
        "frustrated": "\nTONE: The user seems frustrated. Acknowledge this briefly and warmly before answering.",
        "sad":        "\nTONE: The user seems down. Be warm and encouraging in how you frame the answer.",
        "excited":    "\nTONE: The user is excited. Match their enthusiasm — be energetic and positive.",
        "casual":     "\nTONE: Keep it friendly and conversational, not overly formal.",
        "neutral":    "",
    }.get(tone, "")

    # User-stated format preference takes priority; question_type shapes the answer when user gave none
    if format_pref != "no_preference":
        shape_instruction = {
            "brief":         "\nFORMAT: Keep the answer short and to the point — 3-5 sentences maximum.",
            "bullet_points": "\nFORMAT: Present the answer as bullet points.",
            "detailed":      "\nFORMAT: Give a thorough, detailed answer.",
        }.get(format_pref, "")
    else:
        shape_instruction = {
            "comparison": (
                "\nFORMAT: Lead with the single most important difference in ONE sentence. "
                "Then present the comparison as a proper 3-column markdown table: header row "
                "(| Attribute | [first entity] | [second entity] |), separator row (|---|---|---|), "
                "and one data row per attribute, each row on its own line. "
                "Fill EVERY cell with SPECIFIC, CONCRETE data drawn from THAT entity's own excerpts — "
                "exact numbers, parameter counts, named techniques, dates, measured results. "
                "BOTH columns must be equally detailed: each document's excerpts contain that document's "
                "own details, so extract them for its column — never leave one column thin while the other is rich. "
                "NEVER fill a cell with a vague one-word term such as 'Higher', 'Lower', 'Limited', 'Fixed', "
                "'Better', 'Larger', or 'Not explicitly stated'. If you cannot find a concrete value for one "
                "entity on a given attribute, DROP that entire row rather than padding it with a vague placeholder "
                "— keep only attributes where BOTH entities have real data. "
                "GROUNDING: every figure you write MUST appear verbatim in that entity's own provided excerpts — "
                "never supply a number from your own training knowledge; if a value is not in the excerpts, treat "
                "it as absent (and drop the row per the rule above rather than inventing or approximating it). "
                "After the final table row, leave a BLANK LINE, then write a one-sentence verdict followed by your "
                "synthesis as ordinary paragraphs. The verdict and synthesis must NEVER appear inside the table or "
                "as a table row — no pipes, no empty cells, plain prose only."
            ),
            "summary":         "\nFORMAT: Open with a one-sentence overview, then cover the key themes or findings in organised sections or bullet points. End with the single most important takeaway.",
            "list":            "\nFORMAT: Present as a clean numbered or bulleted list. Each item on its own line. No prose padding.",
            "recommendation":  "\nFORMAT: Lead with your verdict or recommendation in the first sentence. Then explain the reasoning. Be decisive — do not hedge.",
            "analytical":      "\nFORMAT: First state what the facts show, then explain why or how. Use clear logical steps. End with the implication or conclusion.",
            "factual":         "",
        }.get(question_type, "")

    extra = lang_instruction + tone_instruction + shape_instruction

    # Persona — injected into every prompt. Consistent identity, consistent voice.
    _PERSONA = (
        "You are ARIA — Analytical Research & Intelligence Assistant. "
        "You are sharp, direct, and confident. You have studied the user's documents thoroughly "
        "and speak about them with authority. You have opinions, make reasoned judgements, "
        "and always keep the conversation productive. You are an analyst, not a search engine. "
        "When you know something, say it plainly. When you don't, say what you'd need — "
        "never leave the user at a dead end.\n\n"
    )

    # Reasoning instruction — grounded strictly in context facts, never beyond them.
    _REASONING = (
        "\n\nREASONING RULE: After stating the facts, add one synthesis paragraph — "
        "what do these facts mean together, what is the implication, what should the user take away. "
        "This reasoning must be grounded only in facts present in the context. "
        "Never speculate beyond what the context shows."
    )

    # Web-cache guard — injected into every VAULT-GROUNDED writer message (Local, MultiDoc,
    # Decomposed, VersionDiff) but NOT the web/hybrid messages. local_graph_check blends cached
    # web facts (from knowledge_graph.json) into context carrying their source URLs; without this
    # the writer cites stale blogs/wikis (e.g. kanerika.com, en.wikipedia.org) as if they were vault.
    _WEB_CACHE_GUARD = (
        "\n\nWEB-CACHE GUARD: This is a vault-grounded answer. IGNORE any retrieved fact whose SOURCE is a "
        "web URL (begins with http, https, or www) — those are stale cached web entries, NOT your vault. "
        "Do not repeat their content and never cite a URL as a source here; rely only on 'Internal:' vault excerpts."
    )

    has_discovery = "VAULT SEARCH RESULT" in context_str
    discovery_rule = (
        "\n\nRULE 5 — Discovery listing: the context contains a VAULT SEARCH RESULT with a VERIFIED RANKING. "
        "List ALL files from the VERIFIED RANKING line in your answer — every single one, in order. "
        "Do NOT include URLs, DOI links, http addresses, arXiv IDs, or any web citations found inside "
        "document chunk text as vault file names — those are references within documents, not vault files. "
        "When citing sources inline in your answer, cite ONLY the vault filenames from the VERIFIED RANKING. "
        "Never cite arXiv IDs, DOI links, or URLs as primary sources. "
        "CRITICAL: Do not add file names from conversation history that are NOT in the VERIFIED RANKING — "
        "your listing must include every file from the VERIFIED RANKING and nothing else."
    ) if has_discovery else ""

    if source_type == "Decomposed":
        sys_msg = SystemMessage(content=(
            f"{_PERSONA}"
            "The user asked a multi-part question. You have context for each part separately, "
            "labelled 'Part 1', 'Part 2', etc. Answer each part directly using its context. "
            "Then add a brief 'Overall' paragraph connecting the answers — what do they mean together? "
            "Cite sources inline as (Source: name). Build a coherent answer, not a list of facts."
            f"{_REASONING}"
            f"{_WEB_CACHE_GUARD}"
            f"\nOUTPUT RULE: Write MISSING_INFO as the very first word ONLY if ALL parts have zero relevant context.{extra}"
        ))
    elif source_type == "VersionDiff":
        sys_msg = SystemMessage(content=(
            f"{_PERSONA}"
            "You are comparing versions of the same document. Context is labelled by version. "
            "Identify what is NEW, what was REMOVED or CHANGED, and what STAYED THE SAME. "
            "Structure: (1) What changed — cite actual content, not vague statements. "
            "(2) What stayed the same. (3) Verdict: minor update or significant revision?\n"
            "Never say 'it was updated' without saying what specifically changed."
            f"{_REASONING}"
            f"{_WEB_CACHE_GUARD}"
            f"\nOUTPUT RULE: Write MISSING_INFO as the very first word ONLY if neither version has relevant content.{extra}"
        ))
    elif source_type == "MultiDoc":
        sys_msg = SystemMessage(content=(
            f"{_PERSONA}"
            "You have answers extracted from MULTIPLE documents, each prefixed with its source. "
            "Synthesize into ONE well-reasoned final answer. Compare, contrast, and connect. "
            "Where documents agree, state it confidently. Where they differ, highlight it with your analysis. "
            "Always cite which document each fact comes from. "
            "CRITICAL: Answer only from what is in the CONTEXT section — silently skip any gaps, never mention them. "
            "CRITICAL: Ignore any file names or topics from the conversation history — use ONLY the document excerpts provided in CONTEXT."
            " Do NOT ask the user for clarification — always synthesize and answer directly from what is provided."
            f"{_REASONING}"
            f"{_WEB_CACHE_GUARD}"
            f"\nOUTPUT RULE: Write MISSING_INFO as the very first word ONLY if ALL documents returned no relevant info.{extra}"
        ))
    elif source_type == "Local":
        sys_msg = SystemMessage(content=(
            f"{_PERSONA}"
            "You have retrieved relevant excerpts from the user's vault. Follow these rules:\n\n"
            "RULE 1 — Facts in context: state them directly and confidently. No hedging. "
            "No 'I believe', 'it seems', 'the context suggests' — if it's there, say it plainly. "
            "Acronym rule: if asked what an acronym stands for, look at the document title. "
            "A title like 'XYZ: Synergizing Alpha and Beta' means XYZ stands for Alpha and Beta — "
            "state it directly. NEVER say 'the document does not explicitly define the acronym' "
            "when the expansion is visible in the title or any text — that IS the definition. "
            "NEVER speculate about why something might not be defined — "
            "if you cannot find the answer, say so in one plain sentence and stop.\n\n"
            "RULE 2 — Synthesize: after the facts, tell the user what they mean. "
            "What is the implication? What should they take away? One analytical paragraph, "
            "grounded only in what the context shows.\n\n"
            "RULE 3 — Genuine gaps: if something is truly not in the context, say so in ONE sentence. "
            "Do NOT ask for clarification before answering — always give the best possible answer from "
            "available context first, then note any gap. "
            "If the SPECIFIC detail the user asked for is absent even though related background is present, "
            "state plainly in one or two sentences what the documents do and do not say — do NOT pad with "
            "paragraphs of 'this suggests', 'this implies', or 'it is likely' speculation around the missing detail. "
            "Do NOT end on 'the context does not provide'. Do NOT suggest the user check the file themselves.\n\n"
            "RULE 4 — Authority: YOU are the system that has already read, indexed, and analysed these "
            "documents. NEVER tell the user to 'examine the file', 'check the document', 'consult the source', "
            "or 'gather more information' themselves — that is your job, not theirs. If you need more "
            "context, ask a targeted question about what aspect they want to explore.\n\n"
            "Cite sources inline as (Source: filename) where relevant. "
            "OUTPUT RULE: Write MISSING_INFO as the very first word ONLY if the context has absolutely "
            f"zero facts about the subject.{_WEB_CACHE_GUARD}{discovery_rule}{extra}"
        ))
    elif is_hybrid:
        sys_msg = SystemMessage(content=(
            f"{_PERSONA}"
            "You have TWO sources: internal documents AND live web data. "
            "Synthesize into one authoritative answer — not a summary of each separately. "
            "Use internal data for precise vault facts. Use web data for recent context or broader picture. "
            "Reason across both: where do they agree, where do they differ, what does the combined view show?"
            f"{_REASONING}"
            "\nEnd with a 'Sources:' section listing all cited document names and URLs. "
            f"OUTPUT RULE: Write MISSING_INFO as the very first word ONLY if the context has zero relevant facts.{extra}"
        ))
    else:
        sys_msg = SystemMessage(content=(
            f"{_PERSONA}"
            "State the key facts from the context, then interpret them — "
            "what do they mean, what do they imply, what should the reader take away? "
            "Be direct. Have a view. Build an answer a sharp person finds genuinely useful."
            f"{_REASONING}"
            "\nOUTPUT RULES:\n"
            "1. Zero facts about the topic: write MISSING_INFO as the very first word.\n"
            "2. Topic covered but missing a large portion: write PARTIAL_INFO as the very first word, "
            "then give the best answer from available facts.\n"
            "3. Minor gaps: answer directly, note the gap in one sentence, suggest a next step.\n"
            f"End with a 'Sources:' section listing each URL on its own line.{extra}"
        ))

    q_display = state.get('clean_question') or (state.get('query_en') if lang == "hinglish" else state['query'])

    history_str = ""
    if history:
        history_str = "CONVERSATION HISTORY (most recent last):\n" + "\n".join(history) + "\n\n"

    human_msg = HumanMessage(content=(
        f"{history_str}"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION:\n{q_display}"
    ))
    ans = main_llm.invoke([sys_msg, human_msg]).content
    return {"answer": ans}

def judge_node(state: AgentState):
    ans = state.get("answer", "")
    iters = state.get("iterations", 0)
    source_type = state.get("source_type", "Web")

    web_allowed = state.get("search_pref") == "web_only"

    # ── Universal vault rescue ────────────────────────────────────────────────
    # Any vault-grounded answer that HEDGES or returns MISSING_INFO may be a retrieval
    # MISS, not a true gap — the fact can be literally present but unfound by the path that
    # ran (discovery file-list, decompose fragments, or a mis-resolved doc). Resolve the doc
    # by LITERAL keyword (catches acronyms the embedder is blind to, e.g. RLAIF), re-retrieve
    # the WHOLE query against it (semantic + keyword sweep), and redraft via writer_node.
    # Gated on failure + a resolvable doc, and the redraft is accepted ONLY if it stops
    # hedging — so this acts solely on already-failing answers and can never regress a good one.
    if (not state.get("rescued") and iters == 0
            and source_type in ("Local", "Decomposed", "MultiDoc")):
        _all_hedge = _HEDGING_PHRASES + _HINGLISH_HEDGING
        _opening = ans[:600].lower()
        if ("MISSING_INFO" in ans) or any(p in _opening for p in _all_hedge):
            _q_en = state.get("query_en", state.get("query", ""))
            # Candidate docs tried IN ORDER: the already-resolved matched_doc FIRST (trusted
            # when correct — e.g. a single-doc query that merely hedged on a flooded sweep),
            # then the literal-term resolver (rescues acronym cases where matched_doc was empty
            # or mis-inferred, e.g. discovery routed RLAIF to 2304.00501 which lacks the term).
            # A candidate is accepted only if its redraft stops hedging — a wrong doc lacks the
            # query's terms, so its redraft hedges and we fall through to the next candidate.
            _cands = []
            for _c in (state.get("matched_doc", ""), find_doc_by_literal_terms(_q_en)):
                if _c and _c not in _cands:
                    _cands.append(_c)
            for _rdoc in _cands:
                try:
                    _rstatus, _rctx = local_graph_check(_q_en, doc_filter=_rdoc)
                except Exception:
                    continue
                if _rstatus != "LOCAL_FOUND" or "Internal:" not in _rctx:
                    continue
                _rstate = dict(state)
                _rstate.update({
                    "context": [_rctx], "source_type": "Local",
                    "matched_doc": _rdoc, "grouped_docs": "{}", "rescued": True,
                })
                print(f"[JUDGE] Vault rescue — trying doc '{_rdoc}', redrafting…")
                try:
                    _redraft = writer_node(_rstate).get("answer", ans)
                except Exception:
                    continue
                if _redraft and "MISSING_INFO" not in _redraft \
                        and not any(p in _redraft[:600].lower() for p in _all_hedge):
                    ans = _redraft
                    source_type = "Local"
                    print(f"[JUDGE] Vault rescue accepted → '{_rdoc}'")
                    break
                print(f"[JUDGE] Rescue redraft from '{_rdoc}' still weak — trying next")

    # For Local answers on the FIRST attempt only: catch hedging in the opening.
    # Only triggers web if user explicitly asked for web — otherwise approve as-is.
    if source_type == "Local" and iters == 0:
        opening = ans[:600].lower()
        all_hedging = _HEDGING_PHRASES + _HINGLISH_HEDGING
        hedging_found = next((p for p in all_hedging if p in opening), None)
        if hedging_found and web_allowed:
            print(f"[JUDGE] Local answer incomplete — triggering targeted web search...")
            try:
                q_en = state.get('query_en', state['query'])
                gap_prompt = (
                    f"The following answer is incomplete:\n{ans}\n\n"
                    f"Original question: {q_en}\n\n"
                    f"In one sentence, describe specifically what information is missing. Output ONLY that sentence."
                )
                gap = mini_llm.invoke(gap_prompt).content.strip()
            except Exception:
                gap = state.get('query_en', state['query'])
            return {"iterations": iters, "answer": ans, "gap": gap, "source_type": "Web"}
        elif hedging_found:
            print(f"[JUDGE] Local answer incomplete but web not requested — approving as-is.")

    # PARTIAL_INFO / MISSING_INFO retry — only if web is explicitly allowed
    needs_retry = "MISSING_INFO" in ans or "PARTIAL_INFO" in ans
    if needs_retry and iters < 3 and web_allowed:
        tag = "PARTIAL_INFO" if "PARTIAL_INFO" in ans else "MISSING_INFO"
        print(f"[JUDGE] {tag} — Triggering deeper search (attempt {iters + 1})...")
        return {"iterations": iters, "answer": ans, "gap": ""}

    if "PARTIAL_INFO" in ans:
        ans = ans.replace("PARTIAL_INFO\n", "").replace("PARTIAL_INFO", "").strip()

    if "MISSING_INFO" in ans:
        if web_allowed:
            print("[JUDGE] Search exhausted. No sufficient information found.")
            ans = (
                "I was unable to find sufficient information to answer this question from available sources. "
                "This topic may require access to specialized documents, paywalled content, or databases "
                "not reachable through web search. Please try rephrasing the question or consulting the source directly."
            )
        else:
            ans = ans.replace("MISSING_INFO\n", "").replace("MISSING_INFO", "").strip()
            if not ans:
                ans = "I couldn't find relevant information about this in your document vault. Reply with **yes** if you'd like me to search the web."

    # Save synthesized answer in English — always, regardless of query language.
    # Graph stays English-only so future queries in any language retrieve it correctly.
    if state.get("source_type") == "Web" and len(ans) > 50:
        try:
            q_en = state.get('query_en', state['query'])
            save_to_graph(
                f"Q: {q_en}\nA: {ans}",
                source_url="Synthesized"
            )
        except Exception:
            pass

    # Translate final answer to Hinglish if that's what the user wrote in
    if state.get("lang") == "hinglish":
        try:
            ans = mini_llm.invoke(
                "Translate the following answer to Hinglish (Hindi+English mix, English script only, "
                "no Devanagari). Keep technical terms, URLs, numbers, and proper nouns in English. "
                f"Output ONLY the translated answer.\n\n{ans}"
            ).content.strip()
        except Exception:
            pass  # fallback: return English answer

    # Strip code-block fences from tables so they render as actual markdown tables.
    # Detects: ``` (optional lang tag) followed by pipe-table lines followed by ```.
    # Only matches blocks whose content is entirely pipe-table rows — safe for real code blocks.
    import re as _re
    ans = _re.sub(r'```[^\n]*\n((?:\|[^\n]*\n)+)```', r'\1', ans)

    # Lift any "phantom summary row" out of a markdown table: the writer sometimes places
    # the trailing verdict/synthesis prose INSIDE the table as a row whose first cell holds
    # the sentence and the remaining cells are empty (| prose |  |  |). Convert such rows to
    # normal paragraphs so the table ends cleanly and the prose renders separately.
    def _lift_phantom_rows(_text):
        _out = []
        for _ln in _text.split("\n"):
            _s = _ln.strip()
            if _s.startswith("|") and _s.endswith("|") and _s.count("|") >= 2:
                _cells = [c.strip() for c in _s.strip("|").split("|")]
                if len(_cells) >= 2:
                    _first, _rest = _cells[0], _cells[1:]
                    _is_sep = all(c and set(c) <= set("-: ") for c in _cells)
                    if not _is_sep and _first and all(not c for c in _rest) and \
                            (len(_first) > 30 or _first.rstrip().endswith(".")):
                        _out.append("")        # blank line closes the table
                        _out.append(_first)    # prose rendered as a paragraph
                        continue
            _out.append(_ln)
        return "\n".join(_out)
    ans = _lift_phantom_rows(ans)

    print("[JUDGE] Answer Approved.")
    return {"answer": ans, "iterations": 4}

def notfound_node(state: AgentState):
    """Fires when local search finds nothing and user hasn't explicitly asked for web."""
    q = state.get("clean_question") or state.get("query", "")
    last_cited = state.get("last_cited_docs", [])
    history = state.get("history", [])

    # Implicit ref unresolved — offer session candidates instead of a dead-end.
    if last_cited and _has_implicit_ref(state.get("query", "")):
        recent = last_cited[-3:]
        if len(recent) == 1:
            clarification = f"**{recent[0]}**"
        elif len(recent) == 2:
            clarification = f"**{recent[0]}** or **{recent[1]}**"
        else:
            clarification = ", ".join(f"**{d}**" for d in recent[:-1]) + f", or **{recent[-1]}**"
        ans = (
            f"I wasn't sure which document you were referring to. "
            f"Based on your session, you may mean: {clarification}.\n\n"
            f"Could you name the document, or rephrase your question?"
        )
        print("[NOTFOUND] Implicit ref unresolved — offering session candidates.")
        return {"answer": ans, "iterations": 4}

    # Reasoning fallback — vault has nothing, but ARIA still responds intelligently
    # using general knowledge, conversation history, and session context.
    print("[NOTFOUND] No vault data — invoking reasoning fallback...")

    history_ctx = ""
    if history:
        history_ctx = "CONVERSATION HISTORY (most recent last):\n" + "\n".join(history[-4:]) + "\n\n"

    vault_docs_hint = ""
    if last_cited:
        vault_docs_hint = f"\n\nDOCUMENTS ACTIVE IN SESSION: {', '.join(last_cited[-5:])}"

    try:
        ans = main_llm.invoke([
            SystemMessage(content=(
                "You are ARIA — Analytical Research & Intelligence Assistant. "
                "You are sharp, direct, and confident. You have opinions and make reasoned judgements. "
                "You are an analyst, not a search engine. Never leave the user at a dead end.\n\n"
                "SITUATION: A search of the user's private document vault returned NO results for this query. "
                "You have no vault excerpts to cite. But you still have:\n"
                "  (a) General knowledge up to your training cutoff\n"
                "  (b) The conversation history shown above\n"
                "  (c) Awareness of which documents the user has been discussing this session\n\n"
                "DECISION LOGIC — choose exactly ONE of these responses:\n"
                "  1. Answerable from general knowledge → answer it directly and concisely. "
                "End with a single italicised line: *Note: This is from general knowledge, not your vault.*\n"
                "  2. Query is ambiguous or likely refers to a specific document → ask ONE smart "
                "clarifying question. Reference the active documents by name if relevant "
                "('Did you mean X from [doc]?').\n"
                "  3. Genuinely unknown even from general knowledge → briefly name what TYPE of source "
                "would contain this, then ask: 'Would you like me to search the web for this?'\n\n"
                "HARD RULE: Never end on 'I couldn't find'. Always give the user a concrete next step "
                "or a direct answer. One of the three options above will always work."
                f"{vault_docs_hint}"
            )),
            HumanMessage(content=f"{history_ctx}QUESTION: {q}")
        ]).content.strip()
        print("[NOTFOUND] Reasoning fallback — answer generated.")
    except Exception as e:
        print(f"[NOTFOUND] Reasoning fallback failed: {e}")
        ans = (
            f"I couldn't find **\"{q}\"** in your document vault. "
            "Would you like me to **search the web**? "
            "Reply with **yes** or repeat your question with *'search the web'* added."
        )
    return {"answer": ans, "iterations": 4}


builder = StateGraph(AgentState)
builder.add_node("analyst",    analyst_node)
builder.add_node("research",   research_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("notfound",   notfound_node)
builder.add_node("write",      writer_node)
builder.add_node("judge",      judge_node)

builder.set_entry_point("analyst")

def _analyst_route(x):
    if x["source_type"] == "MultiDoc":
        return "synthesize"
    if x["source_type"] in ("Local", "VersionDiff", "Decomposed"):
        return "write"
    if x["source_type"] == "NotFound":
        return "notfound"
    return "research"

builder.add_conditional_edges("analyst", _analyst_route)
builder.add_edge("synthesize", "write")
builder.add_edge("notfound",   END)
builder.add_edge("research",   "write")
builder.add_edge("write",      "judge")
builder.add_conditional_edges("judge", lambda x: "research" if x["iterations"] < 3 else END)

agent_app = builder.compile()
