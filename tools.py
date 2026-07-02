import os
import json
import re
import uuid
import http.client
import requests
import chromadb
from datetime import datetime
from brain import get_mini_llm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

mini_llm = get_mini_llm()

# --- NEW SERPER SEARCH TOOL ---
class SerperSearchTool:
    def invoke(self, params):
        query = params.get("query", "")
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
            "q": query,
            "gl": "us",
            "hl": "en",
            "autocorrect": True
        })
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            
            contexts = []
            
            # 1. Grab Google Answer Box if it exists
            if "answerBox" in data:
                ans = data["answerBox"].get("answer") or data["answerBox"].get("snippet")
                if ans:
                    contexts.append({
                        "content": ans,
                        "url": "Google Answer Box"
                    })
            
            # 2. Grab top 8 Organic search snippets
            if "organic" in data:
                for result in data["organic"][:8]:
                    contexts.append({
                        "content": result.get("snippet", ""),
                        "url": result.get("link", "")
                    })
                    
            return contexts
        except Exception as e:
            print(f"[SERPER] Search Error: {e}")
            return []

search_tool = SerperSearchTool()

# URLs from these domains can't be scraped meaningfully
_SKIP_DOMAINS = {
    "youtube.com", "youtu.be", "instagram.com", "twitter.com", "x.com",
    "facebook.com", "tiktok.com", "reddit.com", "linkedin.com",
    "google.com", "maps.google", "play.google",
}

_BOT_SIGNALS = [
    "enable javascript", "javascript is required", "javascript is disabled",
    "does not support javascript", "please enable", "captcha", "robot",
    "access denied", "403 forbidden", "subscribe to read", "sign in to read",
    "create an account", "paywall", "subscription required",
]

def fetch_page_content(url: str, max_chars: int = 2500) -> str:
    """Fetch a URL and return clean plain text. Returns empty string if blocked or bot-detected."""
    try:
        if not url or url in ("Google Answer Box", "Web"):
            return ""
        domain = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
        if any(skip in domain for skip in _SKIP_DOMAINS):
            return ""
        if url.lower().endswith(".pdf"):
            return ""
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            allow_redirects=True
        )
        if resp.status_code != 200:
            return ""
        html = resp.text
        # Reject bot-detection / paywalled pages early
        html_lower = html.lower()
        if any(sig in html_lower[:3000] for sig in _BOT_SIGNALS):
            return ""
        # Strip scripts, styles, nav, header, footer noise
        html = re.sub(r'<(script|style|nav|header|footer|aside)[^>]*>.*?</(script|style|nav|header|footer|aside)>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        # Try to extract main content area first
        for tag in ['article', 'main', 'div id="content"', 'div class="content"', 'div class="article"']:
            match = re.search(rf'<{tag}[^>]*>(.*?)</{tag.split()[0]}>', html, re.DOTALL | re.IGNORECASE)
            if match and len(match.group(1)) > 500:
                html = match.group(1)
                break
        # Strip remaining tags
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        # Final bot-detection check on plain text
        text_lower = text.lower()
        if any(sig in text_lower[:500] for sig in _BOT_SIGNALS):
            return ""
        return text[:max_chars]
    except Exception:
        return ""
# ------------------------------

# Initialize ChromaDB for Semantic Memory
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_graph")
vault_collection = chroma_client.get_or_create_collection(name="internal_vault")
doc_index_collection = chroma_client.get_or_create_collection(name="doc_index")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Single-entry cache for find_relevant_docs_per_file — same query hits ChromaDB once per query.
_frdpf_cache: tuple = (None, None, None)  # (query_str, total_count, raw_result)

# Cache for get_vault_documents() — query-independent, invalidated when vault count changes.
_vault_docs_cache: tuple = (None, None)  # (total_count, result_list)

# Cache for find_doc_by_literal_terms' full-vault text fetch — query-independent,
# invalidated when vault count changes (any ingestion bumps the count).
_litterm_cache: tuple = (None, None)  # (total_count, raw_result)

# Cache for the page-1 title-region index — query-independent, count-invalidated.
_title_index_cache: tuple = (None, None)  # (total_count, {file_name: title_region})

# Generic words dropped before title matching in find_matching_doc's Pass 4. Digits and
# short numeric tokens ("2"/"3"/"4") are KEPT — they are the version discriminators the
# embedder and the literal-term resolver both lose.
_TITLE_STOP = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "with",
    "paper", "papers", "report", "reports", "model", "models", "study", "studies",
    "technical", "system", "systems", "approach", "method", "methods", "herd", "family",
    "introducing", "introduction", "towards", "via", "using", "framework", "document", "documents",
}

JSON_GRAPH_FILE = "knowledge_graph.json"

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "has", "had", "have", "its",
    "this", "that", "with", "from", "into", "onto", "upon", "also", "been",
    "will", "what", "when", "where", "which", "while", "about", "their",
    "there", "then", "than", "them", "they", "some", "such", "does", "did",
    "not", "but", "can", "all", "any", "who", "how", "very", "just", "both",
    "each", "more", "most", "over", "after", "before", "between", "through",
    "during", "describe", "explain", "tell", "give", "list", "name", "find",
    "show", "identify", "define", "provide", "mention", "discuss"
}

# Generic terms that alone don't indicate topical relevance — overlap must include
# at least one keyword NOT in this set so "india + 2026 + april" can't match
# an unrelated India/2026 entry.
_GENERIC_TERMS = {
    "india", "china", "usa", "europe", "world", "global", "new", "latest",
    "2024", "2025", "2026", "2027", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "week", "month", "year", "day", "time", "news", "update", "report",
    "according", "released", "published", "announced", "said", "stated",
    # Generic qualifiers/modifiers that appear across all topics
    "major", "minor", "new", "key", "big", "top", "main", "first", "last",
    "significant", "important", "latest", "recent", "specific", "general",
    "various", "several", "many", "few", "one", "two", "three", "four", "five",
    # Generic legal/document suffixes that appear everywhere
    "act", "law", "bill", "code", "rule", "case", "court", "section", "clause",
    # Generic action/project words that appear across all domains
    "mission", "program", "project", "operation", "launch", "system", "initiative",
    "objective", "primary", "conducted", "scheduled", "planned", "announced",
    "target", "goal", "aim", "purpose", "role", "part", "phase", "stage"
}

def get_vault_documents():
    """Returns list of unique documents currently in vault (latest versions only)."""
    global _vault_docs_cache
    try:
        total = vault_collection.count()
        if _vault_docs_cache[0] == total:
            return _vault_docs_cache[1]
        result = vault_collection.get(where={"is_latest": "true"})
        if not result or not result.get("metadatas"):
            _vault_docs_cache = (total, [])
            return []
        seen = {}
        for meta in result["metadatas"]:
            fn = meta.get("file_name", meta.get("source", "unknown"))
            if fn not in seen:
                seen[fn] = {
                    "file_name":        fn,
                    "version_number":   meta.get("version_number", 1),
                    "upload_timestamp": meta.get("upload_timestamp", "unknown"),
                }
        docs = list(seen.values())
        _vault_docs_cache = (total, docs)
        return docs
    except Exception:
        return []


def build_doc_index_for_file(file_name: str) -> bool:
    """
    Compute centroid embedding for a document and upsert to doc_index_collection.
    Centroid = mean of all is_latest chunk embeddings for this file.
    Called at ingest time so doc_index stays in sync with vault_collection automatically.
    id = file_name → upsert replaces previous centroid on re-ingest or version update.
    """
    import numpy as np
    try:
        result = vault_collection.get(
            where={"$and": [{"file_name": {"$eq": file_name}}, {"is_latest": {"$eq": "true"}}]},
            include=["embeddings"]
        )
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return False
        centroid = np.mean(embeddings, axis=0).tolist()
        doc_index_collection.upsert(
            ids=[file_name],
            embeddings=[centroid],
            metadatas=[{"file_name": file_name}]
        )
        return True
    except Exception as e:
        print(f"[DOC_INDEX] Failed for '{file_name}': {e}")
        return False


def find_relevant_docs_by_centroid(user_query: str, top_k: int = 15) -> list:
    """
    Stage 1 of two-stage retrieval: query the doc-level centroid index.
    Returns top-K documents ranked by centroid similarity — one result per document,
    regardless of corpus size. Scales to 1000s of docs with a single vector search.
    Returns list of {file_name, distance} sorted by distance ascending.
    Returns [] when doc_index is empty (safe: caller falls back to find_relevant_docs).
    """
    n_docs = doc_index_collection.count()
    if n_docs == 0:
        return []
    try:
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        safe_k = min(top_k, n_docs)
        results = doc_index_collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_k
        )
        if not results or not results["metadatas"][0]:
            return []
        return [
            {"file_name": meta.get("file_name", ""), "distance": results["distances"][0][i]}
            for i, meta in enumerate(results["metadatas"][0])
            if meta.get("file_name")
        ]
    except Exception:
        return []


def find_relevant_docs(user_query: str, threshold: float = 0.80, n_results: int = 30) -> list:
    """
    Searches the vault for documents relevant to user_query.
    Returns list of dicts: {file_name, version_number, best_distance, sample_chunk}
    Only includes docs where at least one chunk scores below threshold (strict).
    Groups by document and picks the best (closest) chunk per doc.
    n_results: how many chunks to pull from ChromaDB — use higher values for discovery queries
    to ensure all docs get coverage.
    """
    if vault_collection.count() == 0:
        return []
    try:
        safe_n = min(n_results, vault_collection.count())
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        results = vault_collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_n,
            where={"is_latest": "true"}
        )
        if not results or not results['documents'][0]:
            return []

        doc_scores = {}
        for i, doc in enumerate(results['documents'][0]):
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            fn   = meta.get("file_name", meta.get("source", "unknown"))
            ver  = meta.get("version_number", 1)

            if dist > threshold:
                continue  # not relevant enough

            if fn not in doc_scores:
                doc_scores[fn] = {
                    "file_name":       fn,
                    "version_number":  ver,
                    "best_distance":   dist,
                    "cumulative_score": 0.0,
                    "sample_chunk":    doc[:120] + "..." if len(doc) > 120 else doc,
                }
            else:
                if dist < doc_scores[fn]["best_distance"]:
                    doc_scores[fn]["best_distance"] = dist
            # Accumulate relevance score for every qualifying chunk of this doc
            doc_scores[fn]["cumulative_score"] += (1.0 - dist)

        return sorted(doc_scores.values(), key=lambda x: x["best_distance"])
    except Exception:
        return []


def get_chunks_by_doc(user_query: str, doc_filter=None, version_pref=None, threshold=0.70) -> dict:
    """
    Returns relevant chunks grouped by document for multi-doc synthesis.
    Only includes docs where at least one chunk scores below threshold.
    Returns: {"file_name (vN)": [chunk_text, ...]}
    """
    if vault_collection.count() == 0:
        return {}
    try:
        where_clause = _build_where_clause(doc_filter, version_pref)
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        results = vault_collection.query(
            query_embeddings=[query_embedding],
            n_results=40,
            where=where_clause
        )
        if not results or not results['documents'][0]:
            return {}

        doc_chunks = {}
        for i, doc in enumerate(results['documents'][0]):
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            fn  = meta.get("file_name", meta.get("source", "unknown"))
            ver = meta.get("version_number", 1)
            if dist > threshold:
                continue
            key = f"{fn} (v{ver})"
            if key not in doc_chunks:
                doc_chunks[key] = []
            doc_chunks[key].append(doc)

        # Cap at 6 chunks per doc to keep mini_llm calls manageable
        return {k: v[:6] for k, v in doc_chunks.items()}
    except Exception:
        return {}


def get_all_chunks_for_doc(doc_name: str, max_chunks: int = 12) -> list:
    """
    Fetches up to max_chunks text chunks from a specific document using vault_collection.get()
    — no query embedding, no n_results limit, no distance threshold.
    Used for implicit-ref summarize requests where we just need the doc's content.
    """
    try:
        result = vault_collection.get(
            where={"$and": [
                {"file_name": {"$eq": doc_name}},
                {"is_latest": {"$eq": "true"}}
            ]},
            include=["documents"]
        )
        docs = result.get("documents") or []
        return docs[:max_chunks]
    except Exception:
        return []


def search_top_chunks(user_query: str, doc_name: str, n: int = 20) -> list:
    """
    Semantic search within a single known document with no n_results cap constraint.
    Used for direct retrieval on factual/analytical/list queries when matched_doc is set —
    bypasses local_graph_check's shared-pool limit while keeping semantic ranking.
    Returns list of chunk text strings ordered by relevance (closest first).
    """
    if vault_collection.count() == 0:
        return []
    try:
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        safe_n = min(n, vault_collection.count())
        results = vault_collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_n,
            where={"$and": [
                {"file_name": {"$eq": doc_name}},
                {"is_latest": {"$eq": "true"}}
            ]}
        )
        if not results or not results["documents"][0]:
            return []
        return results["documents"][0]
    except Exception:
        return []


def find_doc_by_literal_terms(user_query: str, max_docs_per_term: int = 2) -> str:
    """
    Literal-keyword doc resolver — for when SEMANTIC retrieval is blind to a term.
    The embedding model (all-MiniLM-L6-v2) cannot connect rare acronyms/tokens to the
    document that defines them (e.g. 'RLAIF' does not embed near Constitutional AI's
    chunks, though the literal string appears there). This scans the vault for LITERAL
    occurrences of the query's DISTINCTIVE tokens (those appearing in <= max_docs_per_term
    documents — high signal, like 'RLAIF' in 2 docs, not noise like 'introduced' in 20)
    and returns the document with the most such hits. Returns '' if nothing distinctive
    resolves cleanly. Read-only; uses the same is_latest filter as all retrieval.

    Intended ONLY as a failure-path fallback — call it when semantic resolution found
    no matched_doc, so it can never override a successful semantic match.
    """
    global _litterm_cache
    import re as _re
    if vault_collection.count() == 0:
        return ""
    tokens = {w.lower() for w in _re.findall(r'[A-Za-z0-9]+', user_query)
              if len(w) >= 4 and w.lower() not in _STOPWORDS}
    if not tokens:
        return ""
    try:
        total = vault_collection.count()
        if _litterm_cache[0] == total:
            r = _litterm_cache[1]
        else:
            r = vault_collection.get(
                where={"is_latest": {"$eq": "true"}},
                include=["documents", "metadatas"],
            )
            _litterm_cache = (total, r)
    except Exception:
        return ""
    tok_docs: dict = {}   # token -> set of file_names containing it
    doc_tok:  dict = {}   # file_name -> {token: chunk_hit_count}
    for ch, mt in zip(r.get("documents") or [], r.get("metadatas") or []):
        fn = mt.get("file_name", "")
        if not fn:
            continue
        cl = ch.lower()
        for t in tokens:
            if t in cl or t.rstrip("s") in cl:
                tok_docs.setdefault(t, set()).add(fn)
                d = doc_tok.setdefault(fn, {})
                d[t] = d.get(t, 0) + 1
    # Distinctive tokens only: appear in a small number of docs (rare = high signal).
    rare = {t for t, ds in tok_docs.items() if 1 <= len(ds) <= max_docs_per_term}
    if not rare:
        return ""
    scored = {fn: sum(c for t, c in tc.items() if t in rare) for fn, tc in doc_tok.items()}
    scored = {f: s for f, s in scored.items() if s > 0}
    return max(scored, key=scored.get) if scored else ""


def find_relevant_docs_per_file(user_query: str, threshold: float = 0.65, chunks_per_doc: int = 10) -> list:
    """
    Discovery-safe retrieval: evaluates EVERY doc in the vault independently.
    Each doc gets its own semantic search so no file can be crowded out by
    other docs dominating a shared top-N result pool.
    Returns list sorted by cumulative_score descending (most relevant first).
    """
    global _frdpf_cache
    docs = get_vault_documents()
    if not docs:
        return []
    total = vault_collection.count()
    if total == 0:
        return []
    try:
        if _frdpf_cache[0] == user_query and _frdpf_cache[1] == total:
            r = _frdpf_cache[2]
        else:
            query_embedding = embedding_model.encode([user_query])[0].tolist()
            # include=["metadatas","distances"] skips ~7.8MB of chunk text —
            # this function only needs file_name + distance for scoring.
            r = vault_collection.query(
                query_embeddings=[query_embedding],
                n_results=total,
                where={"is_latest": {"$eq": "true"}},
                include=["metadatas", "distances"],
            )
            _frdpf_cache = (user_query, total, r)
        if not r or not r["metadatas"][0]:
            return []
    except Exception:
        return []

    from collections import defaultdict
    file_dists = defaultdict(list)
    for dist, meta in zip(r["distances"][0], r["metadatas"][0]):
        fn = meta.get("file_name", "")
        if fn:
            file_dists[fn].append(dist)

    doc_map = {d["file_name"]: d for d in docs}
    results = []
    for fn, dists_only in file_dists.items():
        if fn not in doc_map:
            continue
        best_dist = min(dists_only)
        if best_dist > threshold:
            continue
        cumulative = sum(1.0 - d for d in dists_only if d <= threshold)
        results.append({
            "file_name":        fn,
            "version_number":   doc_map[fn]["version_number"],
            "best_distance":    best_dist,
            "cumulative_score": cumulative,
            "sample_chunk":     "",
        })

    return sorted(results, key=lambda x: -x["cumulative_score"])


def _build_title_index() -> dict:
    """
    {file_name: title_region} for every is_latest doc — title_region is the first 240
    chars of that doc's earliest page-1 chunk (where the title sits). One read-only
    vault_collection.get(); count-invalidated cache. Returns {} on any failure.

    The lowest chunk_id on page 1 is used (not chunk_id==0): ingest skips splits < 50
    chars, so a doc whose first split is tiny would have its title at chunk_id 1.
    """
    global _title_index_cache
    try:
        total = vault_collection.count()
        if _title_index_cache[0] == total:
            return _title_index_cache[1]
        result = vault_collection.get(
            where={"$and": [
                {"is_latest": {"$eq": "true"}},
                {"page": {"$eq": 1}},
            ]},
            include=["documents", "metadatas"],
        )
        docs  = result.get("documents")  or []
        metas = result.get("metadatas")  or []
        best_cid: dict = {}
        index: dict = {}
        for text, meta in zip(docs, metas):
            fn = meta.get("file_name", meta.get("source", ""))
            if not fn:
                continue
            try:
                cid = int(meta.get("chunk_id", 999))
            except (TypeError, ValueError):
                cid = 999
            if fn not in best_cid or cid < best_cid[fn]:
                best_cid[fn] = cid
                index[fn] = (text or "")[:240]
        _title_index_cache = (total, index)
        return index
    except Exception:
        return {}


def _match_doc_by_title(doc_filter: str):
    """
    Failure-path title resolver (Pass 4 of find_matching_doc). Matches doc_filter against
    each doc's page-1 title region for docs whose FILENAME is an opaque arXiv/register
    number ("Llama 2" -> 2307.09288v2.pdf, "Executive Order" -> 2023-24283.pdf).

    Phrase pass first — the contiguous significant-token phrase as a substring of exactly
    one title (this is what makes "llama 2" unique; a stray "2" elsewhere cannot tie it).
    Token pass fallback — unique-max-with-margin over matched-token counts. Returns None on
    ties, so it never misroutes. Read-only.
    """
    index = _build_title_index()
    if not index:
        return None

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    sig = [t for t in _norm(doc_filter).split() if t not in _TITLE_STOP]
    if not sig:
        return None

    regions = {fn: _norm(region) for fn, region in index.items()}

    # Phrase pass: contiguous significant-token phrase, unique substring match.
    phrase = " ".join(sig)
    phrase_hits = [fn for fn, reg in regions.items() if phrase in reg]
    if len(phrase_hits) == 1:
        return phrase_hits[0]

    # Token pass: over the phrase-hit pool (or all docs if none), count matched tokens;
    # return the unique max only if it beats the runner-up by >= 1; else None.
    pool = phrase_hits if phrase_hits else list(regions)
    sig_set = set(sig)
    scored = sorted(
        ((len(sig_set & set(regions[fn].split())), fn) for fn in pool),
        key=lambda x: -x[0],
    )
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return None
    top = scored[0][0]
    runner = scored[1][0] if len(scored) > 1 else 0
    if top - runner >= 1:
        return scored[0][1]
    return None


def find_matching_doc(doc_filter: str):
    """
    Match doc_filter to a vault document. Three-pass strategy:
    1. Filename substring match (exact string in filename)
    2. Word-overlap match (keywords in filename)
    3. Semantic content match (search vault chunks for the filter term)
    Pass 3 handles cases like "Apple report" → FY24_Q1_Consolidated_Financial_Statements.pdf
    """
    if not doc_filter:
        return None
    docs = get_vault_documents()
    if not docs:
        return None
    doc_filter_lower = doc_filter.lower()

    # Pass 1: substring match
    for doc in docs:
        if doc_filter_lower in doc["file_name"].lower():
            return doc["file_name"]

    # Pass 2: word-overlap match.
    # Version-aware: if the filter contains a small number like "2" in "Llama 2",
    # that number must appear as a STANDALONE WORD TOKEN in the matched filename —
    # NOT as a substring. "2" inside "2024" or "2307" doesn't count.
    # This prevents "Llama 2" from matching a Llama 3 file that has "llama" but no "2" token.
    # Years (≥100) are excluded — "2024" is context, not a version identifier.
    # All non-version queries (no small numbers in filter) behave exactly as before.
    _PASS2_DOC_STOP = {"report", "reports", "paper", "papers", "document", "documents",
                       "file", "files", "pdf", "the", "a", "an", "of", "and", "in"}
    filter_words = set(doc_filter_lower.replace("_", " ").replace("-", " ").split()) - _PASS2_DOC_STOP
    filter_version_nums = {w for w in filter_words if w.isdigit() and int(w) < 100}
    best_match, best_score = None, 0
    for doc in docs:
        fn_words = set(doc["file_name"].lower().replace("_", " ").replace("-", " ").replace(".", " ").split())
        score = len(filter_words & fn_words)
        if score > best_score:
            best_score, best_match = score, doc["file_name"]
    if best_score >= 2:
        if filter_version_nums:
            fn_tokens = set(best_match.lower().replace("_", " ").replace("-", " ").replace(".", " ").split())
            if filter_version_nums.issubset(fn_tokens):
                return best_match
            # Version mismatch — matched the wrong version family. Fall through to Pass 3.
        else:
            return best_match

    # Pass 3: semantic content match.
    # Stage A: clear dominant winner (dist < 0.60 and gap ≥ 0.05 from second-best).
    # Stage B: cumulative tiebreaker — when Stage A fails (close gap between candidates),
    #          score each doc by how many of its chunks match the filter term.
    #          A paper entirely ABOUT "Llama 2" accumulates far more qualifying chunks
    #          than a paper that merely mentions it — this correctly separates arxiv IDs
    #          from named files when both have similar per-chunk distances.
    try:
        query_embedding = embedding_model.encode([doc_filter])[0].tolist()
        results = vault_collection.query(
            query_embeddings=[query_embedding],
            n_results=30,
            where={"is_latest": "true"}
        )
        if not results or not results['documents'][0]:
            return None

        doc_best: dict = {}
        for i, dist in enumerate(results['distances'][0]):
            meta = results['metadatas'][0][i]
            fn = meta.get("file_name", meta.get("source", ""))
            if fn and (fn not in doc_best or dist < doc_best[fn]):
                doc_best[fn] = dist

        if not doc_best:
            return None

        sorted_docs = sorted(doc_best.items(), key=lambda x: x[1])
        best_fn   = sorted_docs[0][0]
        best_dist = sorted_docs[0][1]
        second_best_dist = sorted_docs[1][1] if len(sorted_docs) > 1 else 1.0
        gap = second_best_dist - best_dist

        # Stage A: unambiguous winner
        if best_dist < 0.60 and (len(doc_best) == 1 or gap >= 0.05):
            return best_fn

        # Stage B: close contest — use per-file cumulative scoring to pick the doc
        # whose content is most saturated with the filter term
        top_candidates = [fn for fn, dist in sorted_docs if dist < 0.75]
        if top_candidates:
            per_file = find_relevant_docs_per_file(doc_filter, threshold=0.75, chunks_per_doc=10)
            for result in per_file:
                if result["file_name"] in top_candidates:
                    return result["file_name"]
    except Exception:
        pass

    # Pass 4 (failure-path only): title-region match. Reached ONLY when passes 1-3
    # returned nothing, so it cannot change any currently-resolving result. Resolves
    # opaque-filename docs ("Llama 2" -> 2307.09288v2.pdf, "Executive Order" ->
    # 2023-24283.pdf). Unique-max-with-margin → None on ties (never misroutes).
    title_match = _match_doc_by_title(doc_filter)
    if title_match:
        return title_match

    return None


def _build_where_clause(matched_doc: str = None, version_pref: str = None) -> dict:
    """Build ChromaDB where clause from doc filter and version preference."""
    conditions = []

    if matched_doc:
        conditions.append({"file_name": {"$eq": matched_doc}})

    if version_pref and version_pref not in ("latest", "current", "newest", ""):
        # Parse version number hints
        vp = version_pref.lower()
        version_num = None
        if vp in ("v1", "version 1", "original", "first", "old", "oldest", "previous"):
            version_num = 1
        elif vp in ("v2", "version 2", "second"):
            version_num = 2
        elif vp in ("v3", "version 3", "third"):
            version_num = 3
        else:
            import re as _re
            m = _re.search(r'v(\d+)', vp)
            if m:
                version_num = int(m.group(1))

        if version_num:
            conditions.append({"version_number": {"$eq": version_num}})
        else:
            conditions.append({"is_latest": {"$eq": "true"}})
    else:
        conditions.append({"is_latest": {"$eq": "true"}})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def get_version_info(doc_name: str) -> dict:
    """Returns version count and per-version metadata for a document."""
    try:
        result = vault_collection.get(
            where={"file_name": {"$eq": doc_name}},
            include=["metadatas"]
        )
        if not result or not result.get("metadatas"):
            return {"count": 0, "versions": []}
        seen = {}
        for meta in result["metadatas"]:
            v = int(meta.get("version_number", 1))
            if v not in seen:
                seen[v] = {
                    "version": v,
                    "is_latest": meta.get("is_latest", "false") == "true",
                    "upload_timestamp": meta.get("upload_timestamp", "unknown"),
                }
        versions = sorted(seen.values(), key=lambda x: x["version"])
        return {"count": len(versions), "versions": versions}
    except Exception:
        return {"count": 0, "versions": []}


def identify_best_doc_for_query(user_query: str, n: int = 30) -> str:
    """
    Finds the single best-matching document for a query using cumulative relevance scoring.
    No distance threshold — every doc competes. Used for discovery queries.
    Each doc's score = sum of (1 - distance) across its chunks in top-N results.
    """
    if vault_collection.count() == 0:
        return ""
    try:
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        results = vault_collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where={"is_latest": "true"}
        )
        if not results or not results["documents"][0]:
            return ""
        doc_scores: dict = {}
        for i, dist in enumerate(results["distances"][0]):
            meta = results["metadatas"][0][i]
            fn = meta.get("file_name", meta.get("source", ""))
            if not fn:
                continue
            doc_scores[fn] = doc_scores.get(fn, 0.0) + (1.0 - dist)
        if not doc_scores:
            return ""
        return max(doc_scores, key=doc_scores.get)
    except Exception:
        return ""


def get_chunks_for_version(doc_name: str, version_num: int, user_query: str, n: int = 12) -> list:
    """Returns top semantic chunks from a specific version of a document."""
    where_clause = {"$and": [
        {"file_name": {"$eq": doc_name}},
        {"version_number": {"$eq": version_num}}
    ]}
    try:
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        results = vault_collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where_clause
        )
        if not results or not results['documents'][0]:
            return []
        return results['documents'][0]
    except Exception:
        return []


def local_graph_check(user_query, doc_filter=None, version_pref=None):
    query_clean = re.sub(r'[^\w\s]', '', user_query.lower())
    # Keep words >= 3 chars that aren't stopwords.
    query_keywords = {w for w in query_clean.split() if len(w) >= 3 and w not in _STOPWORDS}

    where_clause = _build_where_clause(doc_filter, version_pref)

    # --- PHASE 0: INTERNAL VAULT — highest priority, checked before web cache ---
    # Internal documents (PDFs, DOCX, TXT) ingested via ingest.py live here.
    # Uses both vector search AND keyword matching — tabular/numeric content
    # (financial statements, tables) doesn't embed well, so keyword scan is essential.
    if vault_collection.count() > 0:
        try:
            query_embedding = embedding_model.encode([user_query])[0].tolist()
            _n_vault = 20 if doc_filter else 15
            vault_results = vault_collection.query(
                query_embeddings=[query_embedding],
                n_results=_n_vault,
                where=where_clause
            )
            vault_facts = []
            seen_vault = set()
            if vault_results['documents'][0]:
                for i, doc in enumerate(vault_results['documents'][0]):
                    meta = vault_results['metadatas'][0][i]
                    fn  = meta.get('file_name', meta.get('source', 'document'))
                    ver = meta.get('version_number', 1)
                    src = f"Internal: {fn} (v{ver}), Page {meta.get('page', '?')}"
                    dist = vault_results['distances'][0][i]

                    # Vector hit
                    if dist < 0.90 and doc not in seen_vault:
                        vault_facts.append((dist, f"FACT: {doc} | SOURCE: {src}"))
                        seen_vault.add(doc)

                    # Keyword fallback — catches tabular content that embeds poorly
                    elif doc not in seen_vault:
                        doc_lower = doc.lower()
                        kw_hits = sum(1 for kw in query_keywords if kw in doc_lower)
                        if kw_hits >= 2:
                            vault_facts.append((1.0, f"FACT: {doc} | SOURCE: {src}"))
                            seen_vault.add(doc)

            if vault_facts:
                vault_facts.sort(key=lambda x: x[0])
                _out_cap = 20 if doc_filter else 12
                vault_lines = [line for _, line in vault_facts[:_out_cap]]
            else:
                vault_lines = []

            # For doc-specific queries: always prepend page 1-2 chunks (title + abstract).
            # Semantic search finds methodology chunks but misses the intro where acronyms
            # are defined and paper framing is established. Page 1-2 covers this regardless
            # of how the query embeds.
            if doc_filter:
                try:
                    intro_r = vault_collection.get(
                        where={"$and": [
                            {"file_name": {"$eq": doc_filter}},
                            {"is_latest": {"$eq": "true"}},
                            {"page": {"$lte": 2}}
                        ]},
                        include=["documents", "metadatas"]
                    )
                    intro_docs  = intro_r.get("documents") or []
                    intro_metas = intro_r.get("metadatas") or []
                    intro_lines = []
                    for _doc, _meta in zip(intro_docs, intro_metas):
                        if _doc not in seen_vault:
                            _fn  = _meta.get("file_name", doc_filter)
                            _ver = _meta.get("version_number", 1)
                            _src = f"Internal: {_fn} (v{_ver}), Page {_meta.get('page', 1)}"
                            intro_lines.append(f"FACT: {_doc} | SOURCE: {_src}")
                            seen_vault.add(_doc)
                    vault_lines = intro_lines + vault_lines
                except Exception:
                    pass

            # Keyword sweep for doc-specific queries: fetch chunks containing exact
            # query terms that semantic search may have ranked below the top-N cutoff.
            # Targets niche sections of long papers (e.g. "quantization" on page 45
            # of a 92-page paper) where the chunk embeds poorly relative to the query
            # but the exact term is present.
            if doc_filter and query_keywords:
                try:
                    kw_r = vault_collection.get(
                        where={"$and": [
                            {"file_name": {"$eq": doc_filter}},
                            {"is_latest": {"$eq": "true"}}
                        ]},
                        include=["documents", "metadatas"]
                    )
                    kw_added = 0
                    for _kw_doc, _kw_meta in zip(
                        kw_r.get("documents") or [], kw_r.get("metadatas") or []
                    ):
                        if kw_added >= 5 or _kw_doc in seen_vault:
                            continue
                        _kw_lower = _kw_doc.lower()
                        if any(kw in _kw_lower for kw in query_keywords):
                            _fn  = _kw_meta.get("file_name", doc_filter)
                            _ver = _kw_meta.get("version_number", 1)
                            _src = f"Internal: {_fn} (v{_ver}), Page {_kw_meta.get('page', '?')}"
                            vault_lines.append(f"FACT: {_kw_doc} | SOURCE: {_src}")
                            seen_vault.add(_kw_doc)
                            kw_added += 1
                except Exception:
                    pass

        except Exception:
            vault_lines = []
    else:
        vault_lines = []

    phase1_facts = {}  # fact_text -> (composite_score, source)

    # --- PHASE 1: JSON KEYWORD SCAN — ranked by composite score ---
    if os.path.exists(JSON_GRAPH_FILE):
        with open(JSON_GRAPH_FILE, 'r', encoding='utf-8') as f:
            try:
                graph_data = json.load(f)
                for entry in graph_data:
                    source = entry.get('source', 'Local')
                    for fact in entry.get('data', []):
                        fact_clean = re.sub(r'[^\w\s]', '', fact.lower())
                        fact_words = set(fact_clean.split())
                        matched_kws = query_keywords.intersection(fact_words)
                        overlap = len(matched_kws)
                        # Require >= 2 matches AND at least one topical keyword
                        # (not just dates/places like "india", "april", "2026").
                        # This prevents generic country+year co-occurrences from
                        # pulling in unrelated entries.
                        has_topical = any(k not in _GENERIC_TERMS for k in matched_kws)
                        # Scale minimum overlap with query length — longer queries need more matches
                        # to avoid accidental word collisions across unrelated facts.
                        min_overlap = 3 if len(query_keywords) >= 6 else 2
                        if overlap >= min_overlap and has_topical:
                            # Keyword overlap is primary; length breaks ties so
                            # rich snippets beat sparse triplets at equal overlap.
                            composite = overlap * 1000 + min(len(fact), 500)
                            if fact not in phase1_facts or composite > phase1_facts[fact][0]:
                                phase1_facts[fact] = (composite, source)
            except:
                pass

    # --- PHASE 2: VECTOR SEARCH — always runs alongside Phase 1 ---
    # Phase 1 finds keyword-matched facts; Phase 2 finds semantically similar facts
    # that may use different vocabulary (answer-terms vs query-terms). Combining both
    # gives the writer the most complete context.
    phase2_facts = {}  # fact_text -> source
    if collection.count() > 0:
        try:
            query_embedding = embedding_model.encode([user_query])[0].tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=12)
            if results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    if results['distances'][0][i] < 0.70 and doc not in phase1_facts:
                        meta = results['metadatas'][0][i]
                        phase2_facts[doc] = meta.get('source', 'Local')
        except Exception:
            pass

    # Merge all three phases: vault first (highest authority), then web cache
    if vault_lines or phase1_facts or phase2_facts:
        p1_ranked = sorted(phase1_facts.items(), key=lambda x: x[1][0], reverse=True)
        p2_ranked = sorted(phase2_facts.items(), key=lambda x: len(x[0]), reverse=True)

        context_lines = list(vault_lines)
        # Dedup: vault chunks now also live in the KG (Phase 1) after the backfill, so the
        # same chunk can surface from both the vault (Phase 0) and the graph. Keep the vault
        # copy — it carries the page — and skip identical repeats. Literal-dup removal only;
        # ranking and selection are unchanged.
        _vault_texts = {
            _l.split("FACT: ", 1)[1].split(" | SOURCE:", 1)[0]
            for _l in vault_lines if _l.startswith("FACT: ") and " | SOURCE:" in _l
        }
        for fact, (_, src) in p1_ranked[:15]:
            if fact in _vault_texts:
                continue
            context_lines.append(f"FACT: {fact} | SOURCE: {src}")
        for fact, src in p2_ranked[:8]:
            if fact in _vault_texts:
                continue
            context_lines.append(f"FACT: {fact} | SOURCE: {src}")

        return "LOCAL_FOUND", "\n".join(context_lines)

    return "SEARCH_REQUIRED", ""

def save_to_graph(extracted_text, source_url="Local Knowledge Base"):
    lines = [t.strip() for t in extracted_text.split('\n')]
    candidates = []
    for line in lines:
        clean_line = re.sub(r'^\d+[\.\)\-]\s*', '', line).strip()
        if len(clean_line) > 10:
            candidates.append(clean_line)

    if not candidates:
        return

    # Load existing facts to deduplicate
    existing_facts = set()
    current_data = []
    if os.path.exists(JSON_GRAPH_FILE):
        try:
            with open(JSON_GRAPH_FILE, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            for entry in current_data:
                for fact in entry.get('data', []):
                    existing_facts.add(fact.strip())
        except:
            pass

    # Only save facts not already in the graph
    valid_facts = [f for f in candidates if f not in existing_facts]

    if not valid_facts:
        return  # Nothing new to save

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    embeddings = embedding_model.encode(valid_facts).tolist()
    ids = [str(uuid.uuid4()) for _ in valid_facts]
    metadatas = [{"source": source_url, "timestamp": timestamp_str} for _ in valid_facts]
    collection.add(documents=valid_facts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    new_entry = {"timestamp": timestamp_str, "source": source_url, "data": valid_facts}
    current_data.append(new_entry)
    with open(JSON_GRAPH_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, indent=2)

