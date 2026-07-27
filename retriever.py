import os
import json
import time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Optimize CPU thread allocation
torch.set_num_threads(2)
faiss.omp_set_num_threads(2)

class FAISSMetadataRetriever:
    """
    Ultra-Lightweight RAG Retriever (<150 MB RAM):
    - Uses ultra-compact 'all-MiniLM-L6-v2' model (~80 MB RAM vs 440 MB for mpnet).
    - Lazy model loading so server binds port $PORT instantly within Render 512MB limit.
    - O(1) byte-offset disk seeking for instant document metadata lookups (<1ms).
    """
    def __init__(
        self,
        index_path: str = "faiss_index.index",
        metadata_path: str = "faiss_metadata.jsonl",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = os.getenv("EMBEDDING_MODEL", model_name)

        self.index = None
        self.encoder = None
        self.total_vectors = 0
        self.vector_dim = 384 if "MiniLM" in self.model_name else 768
        
        self.line_offsets = []
        self._meta_file = None

        self._load_index()
        self._load_metadata_offsets()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            print(f"[FAISS Warning] Index file not found at '{self.index_path}'. Initializing empty index.")
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.total_vectors = 0
            return

        print(f"[FAISS] Loading index from {self.index_path}...")
        start = time.time()
        self.index = faiss.read_index(self.index_path)
        self.total_vectors = self.index.ntotal
        self.vector_dim = self.index.d
        print(f"[FAISS] Loaded {self.total_vectors:,} vectors with dimension {self.vector_dim} in {time.time()-start:.2f}s")

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
        """Lazy load encoder on demand to keep startup memory < 50 MB."""
        if self.encoder is None:
            print(f"[Encoder] Lazy loading embedding model '{self.model_name}' (~80 MB RAM)...")
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

        # Get lazy-loaded encoder
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
