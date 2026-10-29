import random
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

db_path = "data/chroma_db"
used_chunks_path = "data/used_chunks.json"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings
)

def _load_used_ids() -> set:
    if os.path.exists(used_chunks_path):
        with open(used_chunks_path, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()
    return set()

def _save_used_ids(used_ids: set):
    os.makedirs(os.path.dirname(used_chunks_path), exist_ok=True)
    with open(used_chunks_path, "w", encoding="utf-8") as f:
        json.dump(list(used_ids), f)

def get_random_chunk() -> tuple[str, str]:
    db_data = vectorstore.get()
    ids = db_data.get("ids", [])
    documents = db_data.get("documents", [])
    if not documents:
        raise ValueError("no chunks available in the database, run ingest.py first.")

    used_ids = _load_used_ids()
    available = [(i, d) for i, d in zip(ids, documents) if i not in used_ids]

    if not available:
        used_ids = set()
        available = list(zip(ids, documents))

    chunk_id, chunk_text = random.choice(available)
    used_ids.add(chunk_id)
    _save_used_ids(used_ids)

    return chunk_id, chunk_text