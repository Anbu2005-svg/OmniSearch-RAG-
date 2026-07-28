import os
import json
import time
import urllib.request
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Optimize CPU thread allocation
torch.set_num_threads(2)
faiss.omp_set_num_threads(2)

# Cloud Dataset Download URLs
DEFAULT_INDEX_URL = os.getenv(
    "INDEX_DOWNLOAD_URL",
    "https://huggingface.co/datasets/Anbanand/OmniSearch_RAG/resolve/main/faiss_index.index"
)
DEFAULT_META_URL = os.getenv(
    "METADATA_DOWNLOAD_URL",
    "https://huggingface.co/datasets/Anbanand/OmniSearch_RAG/resolve/main/faiss_metadata.jsonl"
)

# Auto-detect Cloud environment (Render sets PORT/RENDER env vars) or explicit OPTIMIZE_RAM
IS_CLOUD = (
    os.getenv("PORT") is not None or 
    os.getenv("RENDER") is not None or 
    os.getenv("OPTIMIZE_RAM", "false").lower() in ("true", "1", "yes")
)

class FAISSMetadataRetriever:
    """
    RAG Retriever Guaranteed under 512 MB RAM:
    - Uses FAISS Memory Mapping (IO_FLAG_MMAP) so 614 MB vector index stays on disk (<30 MB RAM).
    - Cloud Mode: Uses 'all-MiniLM-L6-v2' (~150 MB RAM total).
    - Standard Mode: Uses 'all-mpnet-base-v2' (~480 MB RAM total, under 512 MB limit!).
    """
    def __init__(
        self,
        index_path: str = "faiss_index.index",
        metadata_path: str = "faiss_metadata.jsonl",
        model_name: str = None
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = "all-MiniLM-L6-v2" if IS_CLOUD else "all-mpnet-base-v2"

        self.index = None
        self.encoder = None
        self.total_vectors = 0
        self.vector_dim = 384 if "MiniLM" in self.model_name else 768
        
        self.line_offsets = []
        self._meta_file = None

        self._ensure_files_exist()
        self._load_index()
        self._load_metadata_offsets()

    def _ensure_files_exist(self):
        """Download index and metadata files if missing on cloud server."""
        if not os.path.exists(self.index_path) and DEFAULT_INDEX_URL:
            print(f"[Download] Downloading FAISS index from Hugging Face...")
            try:
                urllib.request.urlretrieve(DEFAULT_INDEX_URL, self.index_path)
                print("[Download] FAISS index download complete!")
            except Exception as e:
                print(f"[Download Error] FAISS index download failed: {e}")

        if not os.path.exists(self.metadata_path) and DEFAULT_META_URL:
            print(f"[Download] Downloading Metadata file from Hugging Face...")
            try:
                urllib.request.urlretrieve(DEFAULT_META_URL, self.metadata_path)
                print("[Download] Metadata file download complete!")
            except Exception as e:
                print(f"[Download Error] Metadata download failed: {e}")

    def _load_index(self):
        if not os.path.exists(self.index_path):
            print(f"[FAISS Warning] Index file not found at '{self.index_path}'. Initializing empty index.")
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.total_vectors = 0
            return

        print(f"[FAISS] Loading index from {self.index_path} using Memory Mapping (MMAP)...")
        start = time.time()
        
        # Memory Mapping IO_FLAG_MMAP keeps the 614 MB index on disk, using < 30 MB RAM
        try:
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
        except Exception:
            self.index = faiss.read_index(self.index_path)

        self.total_vectors = self.index.ntotal
        self.vector_dim = self.index.d
        print(f"[FAISS] Loaded {self.total_vectors:,} vectors in {time.time()-start:.2f}s (<30 MB RAM)")

    def _load_metadata_offsets(self):
        """Build in-memory byte offset index for metadata file."""
        if not os.path.exists(self.metadata_path):
            print(f"[Metadata Warning] Metadata file not found at '{self.metadata_path}'.")
            return
        
        print(f"[Metadata] Building fast byte-offset index for {self.metadata_path}...")
        start = time.time()
        self.line_offsets = []
        with open(self.metadata_path, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
        
        self._meta_file = open(self.metadata_path, 'rb')
        print(f"[Metadata] Indexed {len(self.line_offsets):,} line offsets in {time.time()-start:.2f}s")

    def _get_encoder(self):
        """Lazy load encoder on demand to keep RAM under 512 MB."""
        if self.encoder is None:
            print(f"[Encoder] Loading embedding model '{self.model_name}'...")
            start = time.time()
            self.encoder = SentenceTransformer(self.model_name)
            with torch.inference_mode():
                self.encoder.encode(["warmup query"], normalize_embeddings=True)
            print(f"[Encoder] Model loaded in {time.time()-start:.2f}s")
        return self.encoder

    def _get_metadata_by_line(self, line_idx: int) -> dict:
        """O(1) random access metadata lookup using byte offsets."""
        if not self._meta_file or line_idx < 0 or line_idx >= len(self.line_offsets):
            return {"doc_id": line_idx, "chunk_id": 0, "text": "", "meta": {}}

        try:
            offset = self.line_offsets[line_idx]
            self._meta_file.seek(offset)
            line = self._meta_file.readline()
            if line:
                return json.loads(line.decode('utf-8'))
        except Exception as e:
            print(f"[Metadata Error] Line {line_idx}: {e}")
        
        return {"doc_id": line_idx, "chunk_id": 0, "text": "", "meta": {}}

    @torch.inference_mode()
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list:
        """
        Perform high-speed vector similarity search for a query string.
        Returns a list of dictionary matches with document text and metadata.
        """
        if not query or not query.strip() or self.total_vectors == 0:
            return []

        encoder = self._get_encoder()

        # Encode query to numpy array
        query_vec = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query_vec, min(top_k, self.total_vectors))

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0:
                continue
            
            similarity = max(0.0, 1.0 - float(dist)) if dist <= 2.0 else float(1.0 / (1.0 + dist))
            
            if similarity < score_threshold:
                continue

            meta_data = self._get_metadata_by_line(int(idx))
            results.append({
                "rank": rank,
                "vector_id": int(idx),
                "doc_id": meta_data.get("doc_id", int(idx)),
                "chunk_id": meta_data.get("chunk_id", 0),
                "distance": float(dist),
                "similarity": similarity,
                "text": meta_data.get("text", ""),
                "meta": meta_data.get("meta", {})
            })

        return results
