import os
import uuid
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize DB and Embedding Model (Must match tools.py)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Create a NEW collection specifically for internal documents
vault_collection = chroma_client.get_or_create_collection(name="internal_vault")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

DOCS_DIR = "./docs"

def extract_text_from_pdf(file_path):
    import fitz  # PyMuPDF
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
            text_data.append({"text": full_text, "page": 1}) # Word docs don't have strict pages like PDFs
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text_data

def process_documents():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[INGEST] Created {DOCS_DIR} directory. Please add your files and run again.")
        return

    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.docx', '.txt'))]
    if not files:
        print(f"[INGEST] No supported documents found in {DOCS_DIR}.")
        return

    # We use LangChain's splitter to break large pages into overlapping semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    total_chunks_added = 0

    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"[INGEST] Processing: {file_name}...")
        
        # 1. Extract Text
        extracted_pages = []
        if file_name.endswith('.pdf'):
            extracted_pages = extract_text_from_pdf(file_path)
        elif file_name.endswith('.docx'):
            extracted_pages = extract_text_from_docx(file_path)
        elif file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_pages = [{"text": f.read(), "page": 1}]

        # 2. Chunk Text and Prepare Metadata
        documents = []
        metadatas = []
        ids = []
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        for page_data in extracted_pages:
            chunks = text_splitter.split_text(page_data["text"])
            
            for i, chunk in enumerate(chunks):
                # Clean up the chunk slightly
                chunk = chunk.replace('\n', ' ').strip()
                if len(chunk) < 50:
                    continue # Skip tiny, useless fragments

                documents.append(chunk)
                metadatas.append({
                    "source": file_name,
                    "page": page_data["page"],
                    "chunk_id": i,
                    "timestamp": timestamp_str,
                    "type": "internal"
                })
                ids.append(str(uuid.uuid4()))

        # 3. Deduplicate against existing vault entries, then embed and save
        if documents:
            existing = vault_collection.get(where={"source": file_name})
            existing_docs = set(existing.get("documents") or [])
            new_docs, new_metas, new_ids, new_embeds_input = [], [], [], []
            for doc, meta, vid in zip(documents, metadatas, ids):
                if doc not in existing_docs:
                    new_docs.append(doc)
                    new_metas.append(meta)
                    new_ids.append(vid)
                    new_embeds_input.append(doc)

            if new_docs:
                embeddings = embedding_model.encode(new_embeds_input).tolist()
                vault_collection.add(
                    documents=new_docs,
                    embeddings=embeddings,
                    metadatas=new_metas,
                    ids=new_ids
                )
                total_chunks_added += len(new_docs)
                print(f"  -> Added {len(new_docs)} new chunks ({len(documents) - len(new_docs)} duplicates skipped).")
            else:
                print(f"  -> All chunks already in vault, skipped.")

    print("\n" + "="*50)
    print(f"[INGEST] COMPLETE! Total chunks securely stored: {total_chunks_added}")
    print("="*50)

if __name__ == "__main__":
    process_documents()