import os
import json
import time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Optimize CPU thread allocation
torch.set_num_threads(4)
faiss.omp_set_num_threads(4)

class FAISSMetadataRetriever:
    """
    High-Performance RAG Retriever:
    - O(1) byte-offset disk seeking for instant document metadata lookups (<1ms).
    - PyTorch inference mode + thread optimization for high-speed query vector encoding.
    - Graceful fallback for cloud hosting when large index files are hosted externally.
    """
    def __init__(
        self,
        index_path: str = "faiss_index.index",
        metadata_path: str = "faiss_metadata.jsonl",
        model_name: str = "all-mpnet-base-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        self.index = None
        self.encoder = None
        self.total_vectors = 0
        self.vector_dim = 768
        
        self.line_offsets = []
        self._meta_file = None

        self._load_index()
        self._load_metadata_offsets()
        self._load_encoder()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            print(f"[FAISS Warning] Index file not found at '{self.index_path}'. Initializing empty IndexFlatL2(768).")
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

    def _load_encoder(self):
        print(f"[Encoder] Loading embedding model '{self.model_name}'...")
        start = time.time()
        self.encoder = SentenceTransformer(self.model_name)
        
        # Pre-warm PyTorch model
        with torch.inference_mode():
            self.encoder.encode(["warmup query"], normalize_embeddings=True)
            
        print(f"[Encoder] Model loaded & pre-warmed in {time.time()-start:.2f}s")

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

        # Encode query to 768-dim numpy array
        query_vec = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
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
