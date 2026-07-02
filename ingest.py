import os
import json
import uuid
import numpy as np
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize DB and Embedding Model (Must match tools.py)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vault_collection = chroma_client.get_or_create_collection(name="internal_vault")
doc_index_collection = chroma_client.get_or_create_collection(name="doc_index")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

DOCS_DIR = "./docs"
GRAPH_FILE = "knowledge_graph.json"   # mirrored vault chunks for the KG retrieval phase (Phase 1)


def extract_text_from_pdf(file_path):
    import fitz
    text_data = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                text_data.append({"text": text, "page": page_num + 1})
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text_data


def extract_text_from_docx(file_path):
    import docx
    text_data = []
    try:
        doc = docx.Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if full_text:
            text_data.append({"text": full_text, "page": 1})
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text_data


def _get_next_version(file_name: str) -> int:
    """
    Returns the next version number for a given file_name.
    Checks existing chunks in vault to find the current max version.
    """
    try:
        existing = vault_collection.get(where={"file_name": file_name})
        if not existing or not existing.get("metadatas"):
            return 1
        versions = [
            int(m.get("version_number", 1))
            for m in existing["metadatas"]
            if m.get("version_number")
        ]
        return max(versions) + 1 if versions else 1
    except Exception:
        return 1


def _mark_old_versions_stale(file_name: str):
    """
    Marks all existing chunks of a file as is_latest=false before adding a new version.
    ChromaDB does not support bulk update, so we fetch IDs and update one by one.
    """
    try:
        existing = vault_collection.get(where={"file_name": file_name})
        if not existing or not existing.get("ids"):
            return
        for chunk_id, meta in zip(existing["ids"], existing["metadatas"]):
            if meta.get("is_latest") == "true":
                updated_meta = {**meta, "is_latest": "false"}
                vault_collection.update(ids=[chunk_id], metadatas=[updated_meta])
    except Exception as e:
        print(f"[INGEST] Warning: could not mark old versions stale: {e}")


def _update_doc_index(file_name: str):
    """
    Recompute centroid for file_name from its current is_latest chunks and upsert
    to doc_index_collection. Called at the end of every ingest_file invocation so
    the doc-level index stays in sync with vault_collection automatically.
    Uses file_name as the ChromaDB id — upsert replaces previous centroid on re-ingest.
    """
    try:
        result = vault_collection.get(
            where={"$and": [{"file_name": {"$eq": file_name}}, {"is_latest": {"$eq": "true"}}]},
            include=["embeddings"]
        )
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return
        centroid = np.mean(embeddings, axis=0).tolist()
        doc_index_collection.upsert(
            ids=[file_name],
            embeddings=[centroid],
            metadatas=[{"file_name": file_name}]
        )
        print(f"[INGEST] Doc index updated: '{file_name}' ({len(embeddings)} chunks)")
    except Exception as e:
        print(f"[INGEST] Doc index update failed for '{file_name}': {e}")


def upsert_doc_to_graph(file_name: str) -> int:
    """Mirror a document's current (is_latest) chunks into knowledge_graph.json as a
    SINGLE entry {source: "Internal: <file>", data: [chunks]} — NO timestamp (ChromaDB
    owns time). Lets the KG retrieval phase (local_graph_check Phase 1) surface vault
    facts by exact keyword. Idempotent: replaces THIS file's entry, preserves every other
    entry (other docs AND web-cache facts). Read-only on ChromaDB."""
    try:
        res = vault_collection.get(
            where={"$and": [{"file_name": {"$eq": file_name}}, {"is_latest": {"$eq": "true"}}]},
            include=["documents"]
        )
        seen, data = set(), []
        for d in (res.get("documents") or []):
            c = (d or "").strip()
            if len(c) >= 12 and c not in seen:
                seen.add(c)
                data.append(c)
        if not data:
            return 0
        graph = []
        if os.path.exists(GRAPH_FILE):
            try:
                with open(GRAPH_FILE, "r", encoding="utf-8") as f:
                    graph = json.load(f)
            except Exception:
                graph = []
        src = f"Internal: {file_name}"
        graph = [e for e in graph if e.get("source") != src]   # drop this file's old entry
        graph.append({"source": src, "data": data})            # ...and re-add the latest chunks
        with open(GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        return len(data)
    except Exception as e:
        print(f"[INGEST] KG mirror failed for '{file_name}': {e}")
        return 0


def check_similarity_before_ingest(file_path: str, file_name: str, threshold: float = 0.85):
    """
    Computes average embedding of new doc and compares to all existing docs in vault.
    Returns the name of the most similar existing doc if similarity > threshold, else None.
    This enables smart version detection for renamed files.
    """
    import numpy as np

    # Extract text from new file
    extracted = []
    if file_name.lower().endswith('.pdf'):
        extracted = extract_text_from_pdf(file_path)
    elif file_name.lower().endswith('.docx'):
        extracted = extract_text_from_docx(file_path)
    elif file_name.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            extracted = [{"text": f.read(), "page": 1}]

    if not extracted:
        return None

    # Sample up to 20 chunks from new doc for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    sample_chunks = []
    for page in extracted:
        sample_chunks.extend(splitter.split_text(page["text"]))
        if len(sample_chunks) >= 20:
            break
    sample_chunks = sample_chunks[:20]
    if not sample_chunks:
        return None

    new_embeddings = embedding_model.encode(sample_chunks)
    new_avg = np.mean(new_embeddings, axis=0)

    # Get all existing docs and their average embeddings
    try:
        existing = vault_collection.get(where={"is_latest": "true"}, include=["embeddings", "metadatas"])
        if not existing or not existing.get("metadatas") or not existing.get("embeddings"):
            return None

        # Group embeddings by file_name
        doc_embeddings = {}
        for meta, emb in zip(existing["metadatas"], existing["embeddings"]):
            fn = meta.get("file_name", meta.get("source", ""))
            if fn == file_name:
                continue  # skip self (same filename = already handled by version logic)
            if fn not in doc_embeddings:
                doc_embeddings[fn] = []
            doc_embeddings[fn].append(emb)

        if not doc_embeddings:
            return None

        # Find the most similar existing doc
        best_match, best_score = None, 0.0
        for fn, embs in doc_embeddings.items():
            existing_avg = np.mean(embs[:20], axis=0)
            # Cosine similarity
            score = float(
                np.dot(new_avg, existing_avg) /
                (np.linalg.norm(new_avg) * np.linalg.norm(existing_avg) + 1e-9)
            )
            if score > best_score:
                best_score, best_match = score, fn

        if best_score >= threshold:
            print(f"[INGEST] Similarity check: '{file_name}' is {best_score:.2f} similar to '{best_match}'")
            return best_match
        return None

    except Exception as e:
        print(f"[INGEST] Similarity check failed: {e}")
        return None


def ingest_file(file_path: str, file_name: str) -> int:
    """
    Ingest a single file into the vault. Returns number of new chunks added.
    Handles versioning: same file_name = new version, old chunks marked is_latest=false.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    extracted_pages = []
    if file_name.lower().endswith('.pdf'):
        extracted_pages = extract_text_from_pdf(file_path)
    elif file_name.lower().endswith('.docx'):
        extracted_pages = extract_text_from_docx(file_path)
    elif file_name.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            extracted_pages = [{"text": f.read(), "page": 1}]

    if not extracted_pages:
        return 0

    # Determine version for this upload
    version_number = _get_next_version(file_name)
    is_new_version = version_number > 1

    # If this is a new version, mark all old chunks as stale
    if is_new_version:
        print(f"[INGEST] New version detected for '{file_name}' — marking v{version_number - 1} as stale...")
        _mark_old_versions_stale(file_name)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc_id = str(uuid.uuid4())  # single doc_id shared across all chunks of this upload

    documents, metadatas, ids = [], [], []

    for page_data in extracted_pages:
        for i, chunk in enumerate(text_splitter.split_text(page_data["text"])):
            chunk = chunk.replace('\n', ' ').strip()
            if len(chunk) < 50:
                continue
            documents.append(chunk)
            metadatas.append({
                "source":           file_name,   # kept for backwards compat with tools.py
                "file_name":        file_name,
                "doc_id":           doc_id,
                "page":             page_data["page"],
                "chunk_id":         i,
                "upload_timestamp": timestamp_str,
                "version_number":   version_number,
                "is_latest":        "true",
                "type":             "internal"
            })
            ids.append(str(uuid.uuid4()))

    if not documents:
        return 0

    # Deduplication: skip chunks already present in this exact version
    existing = vault_collection.get(where={"$and": [{"file_name": {"$eq": file_name}}, {"is_latest": {"$eq": "true"}}]})
    existing_docs = set(existing.get("documents") or [])

    new_docs, new_metas, new_ids = [], [], []
    for doc, meta, vid in zip(documents, metadatas, ids):
        if doc not in existing_docs:
            new_docs.append(doc)
            new_metas.append(meta)
            new_ids.append(vid)

    if new_docs:
        embeddings = embedding_model.encode(new_docs).tolist()
        vault_collection.add(
            documents=new_docs,
            embeddings=embeddings,
            metadatas=new_metas,
            ids=new_ids
        )
        v_label = f"v{version_number}" + (" (NEW VERSION)" if is_new_version else "")
        print(f"[INGEST] '{file_name}' {v_label} — {len(new_docs)} chunks added.")

    _update_doc_index(file_name)
    upsert_doc_to_graph(file_name)   # mirror latest chunks into the KG (Phase 1 retrieval)
    return len(new_docs)


def process_documents():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[INGEST] Created {DOCS_DIR} directory. Add your files and run again.")
        return

    files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if not files:
        print(f"[INGEST] No supported documents found in {DOCS_DIR}.")
        return

    total = 0
    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"[INGEST] Processing: {file_name}...")
        added = ingest_file(file_path, file_name)
        total += added
        if not added:
            print(f"  -> Already in vault, skipped.")

    print("\n" + "="*50)
    print(f"[INGEST] COMPLETE! Total new chunks stored: {total}")
    print("="*50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into the vault.")
    parser.add_argument("--file", type=str, help="Ingest a single file by path")
    parser.add_argument("--bulk", action="store_true", help="Ingest all files in ./docs/ folder")
    args = parser.parse_args()

    if args.file:
        file_path = args.file
        file_name = os.path.basename(file_path)
        print(f"[INGEST] Ingesting single file: {file_name}")
        added = ingest_file(file_path, file_name)
        if added:
            print(f"[INGEST] Done — {added} new chunks added.")
        else:
            print(f"[INGEST] Already in vault or no content found.")
    elif args.bulk:
        process_documents()
    else:
        print("Usage:")
        print("  Single file : python ingest.py --file path/to/file.pdf")
        print("  Bulk import : python ingest.py --bulk")
