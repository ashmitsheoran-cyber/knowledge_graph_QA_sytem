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
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

def local_graph_check(user_query):
    query_clean = re.sub(r'[^\w\s]', '', user_query.lower())
    # Keep words >= 3 chars that aren't stopwords.
    query_keywords = {w for w in query_clean.split() if len(w) >= 3 and w not in _STOPWORDS}

    # --- PHASE 0: INTERNAL VAULT — highest priority, checked before web cache ---
    # Internal documents (PDFs, DOCX, TXT) ingested via ingest.py live here.
    # Uses both vector search AND keyword matching — tabular/numeric content
    # (financial statements, tables) doesn't embed well, so keyword scan is essential.
    if vault_collection.count() > 0:
        try:
            query_embedding = embedding_model.encode([user_query])[0].tolist()
            vault_results = vault_collection.query(query_embeddings=[query_embedding], n_results=15)
            vault_facts = []
            seen_vault = set()
            if vault_results['documents'][0]:
                for i, doc in enumerate(vault_results['documents'][0]):
                    meta = vault_results['metadatas'][0][i]
                    src = f"Internal: {meta.get('source', 'document')}, Page {meta.get('page', '?')}"
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
                vault_lines = [line for _, line in vault_facts[:12]]
            else:
                vault_lines = []
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
        for fact, (_, src) in p1_ranked[:15]:
            context_lines.append(f"FACT: {fact} | SOURCE: {src}")
        for fact, src in p2_ranked[:8]:
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