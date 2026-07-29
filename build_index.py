import os
import json
import time
import glob
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = "faiss_index_8bit.index"
DEFAULT_METADATA_PATH = "faiss_metadata.jsonl"
DEFAULT_DOCS_DIR = "documents"


def load_documents(docs_dir: str) -> list:
    documents = []
    txt_files = sorted(glob.glob(os.path.join(docs_dir, "**/*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{docs_dir}'")

    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            documents.append({
                "doc_id": len(documents) + 1,
                "text": text,
                "source": os.path.relpath(path, docs_dir),
                "meta": {"filename": os.path.basename(path)}
            })
    return documents


def load_documents_from_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    raise ValueError(f"Unsupported JSON format in '{path}'")


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def build_index(model_name: str, docs_dir: str, index_path: str, metadata_path: str, json_path: str = None):
    print(f"1. Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name, device="cpu")
    print(f"   Model loaded. Vector dimension: {model.get_sentence_embedding_dimension()}")

    if json_path and os.path.isfile(json_path):
        print(f"2. Loading documents from JSON '{json_path}'...")
        raw_docs = load_documents_from_json(json_path)
    else:
        print(f"2. Loading documents from '{docs_dir}'...")
        raw_docs = load_documents(docs_dir)
    print(f"   Loaded {len(raw_docs)} documents")

    print("3. Chunking documents...")
    chunks = []
    for doc in raw_docs:
        text = doc.get("text", "")
        if not text.strip():
            continue
        text_chunks = chunk_text(text)
        doc_id = doc.get("doc_id", len(chunks) + 1)
        for chunk_idx, text in enumerate(text_chunks, start=1):
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_idx,
                "text": text,
                "source": doc.get("source", doc.get("filename", "")),
                "meta": doc.get("meta", {})
            })
    print(f"   Created {len(chunks)} chunks")

    print("4. Encoding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    print(f"   Encoded {len(chunks)} vectors with dimension {dim}")

    print("5. Building 8-bit FAISS index...")
    sq_index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
    sq_index.train(embeddings)
    sq_index.add(embeddings)
    print(f"   Index built with {sq_index.ntotal} vectors")

    print(f"6. Saving index to '{index_path}'...")
    faiss.write_index(sq_index, index_path)

    print(f"7. Saving metadata to '{metadata_path}'...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "meta": chunk["meta"]
            }, ensure_ascii=False) + "\n")

    index_size = os.path.getsize(index_path) / (1024 * 1024)
    meta_size = os.path.getsize(metadata_path) / (1024 * 1024)
    print("=========================================")
    print(f"✅ Build Complete!")
    print(f"Index size:   {index_size:.1f} MB")
    print(f"Metadata size: {meta_size:.1f} MB")
    print(f"Vectors:      {len(chunks):,}")
    print(f"Dimension:    {dim}")
    print("=========================================")


def main():
    parser = argparse.ArgumentParser(description="Build 8-bit FAISS index with MiniLM embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--docs", default=DEFAULT_DOCS_DIR, help="Directory containing .txt documents")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Output FAISS index path")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH, help="Output metadata JSONL path")
    parser.add_argument("--json", default=None, help="Optional JSON file containing documents")
    args = parser.parse_args()

    if not args.json and not os.path.isdir(args.docs):
        print(f"Error: Neither JSON file '{args.json}' nor documents directory '{args.docs}' exists.")
        return

    build_index(args.model, args.docs, args.index, args.metadata, args.json)


if __name__ == "__main__":
    main()
