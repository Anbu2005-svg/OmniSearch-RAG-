import os
import gc
import json
import time
import urllib.request
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# Cloud Dataset Download URLs
DEFAULT_INDEX_URL = os.getenv(
    "INDEX_DOWNLOAD_URL",
    "https://huggingface.co/datasets/Anbanand/OmniSearch_RAG/resolve/main/faiss_index_8bit.index"
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
    RAG Retriever optimized for low-RAM deployments (384-dim MiniLM):
    - Uses 8-bit FAISS Index (~76 MB for 200K chunks).
    - Cloud Mode: Uses INT8 Dynamic Quantization on 'all-MiniLM-L6-v2' (~55 MB).
    """
    def __init__(
        self,
        index_path: str = "faiss_index_8bit_50k.index",
        metadata_path: str = "faiss_metadata_50k.jsonl",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        self.index = None
        self.encoder = None
        self.total_vectors = 0
        self.vector_dim = 384
        
        self.line_offsets = []
        self._meta_file = None

        self._ensure_files_exist()
        self._load_index()
        self._load_metadata_offsets()

    def _download_file(self, url: str, dest_path: str, chunk_size: int = 2 * 1024 * 1024):
        """Stream download in 2MB chunks to avoid memory spikes."""
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        gc.collect()

    def _ensure_files_exist(self):
        """Download index and metadata files with streaming to preserve RAM."""
        if not os.path.exists(self.index_path) and DEFAULT_INDEX_URL:
            print(f"[Download] Streaming FAISS index from Hugging Face...")
            try:
                self._download_file(DEFAULT_INDEX_URL, self.index_path)
                print("[Download] FAISS index download complete!")
            except Exception as e:
                print(f"[Download Error] FAISS index download failed: {e}")

        if not os.path.exists(self.metadata_path) and DEFAULT_META_URL:
            print(f"[Download] Streaming Metadata file from Hugging Face...")
            try:
                self._download_file(DEFAULT_META_URL, self.metadata_path)
                print("[Download] Metadata file download complete!")
            except Exception as e:
                print(f"[Download Error] Metadata download failed: {e}")
        gc.collect()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            print(f"[FAISS Warning] Index file not found at '{self.index_path}'. Initializing empty index.")
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.total_vectors = 0
            return

        print(f"[FAISS] Loading index from {self.index_path}...")
        start = time.time()
        
        # Enforce single thread to prevent memory multiplication
        faiss.omp_set_num_threads(1)
        gc.collect()
        
        try:
            # IO_FLAG_MMAP maps the index file directly to disk space without consuming physical RAM
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        except Exception:
            try:
                self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            except Exception as e:
                print(f"[FAISS MMAP Failed] Falling back to standard read: {e}")
                self.index = faiss.read_index(self.index_path)

        self.total_vectors = self.index.ntotal
        self.vector_dim = self.index.d
        print(f"[FAISS] Loaded {self.total_vectors:,} vectors (dim={self.vector_dim}) in {time.time()-start:.2f}s")
        gc.collect()

    def _load_metadata_offsets(self):
        """Build ultra-compact in-memory byte offset index using array('Q') (<1.6MB RAM)."""
        import array
        if not os.path.exists(self.metadata_path):
            print(f"[Metadata Warning] Metadata file not found at '{self.metadata_path}'.")
            return
        
        print(f"[Metadata] Building compact byte-offset index for {self.metadata_path}...")
        start = time.time()
        # array('Q') stores unsigned 64-bit integers directly in C contiguous buffer, taking only 8 bytes per item
        self.line_offsets = array.array('Q')
        with open(self.metadata_path, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
        
        gc.collect()
        print(f"[Metadata] Indexed {len(self.line_offsets):,} line offsets in {time.time()-start:.2f}s")

    def _get_encoder(self):
        """Lazy load encoder and apply INT8 dynamic quantization for cloud deployments."""
        if self.encoder is None:
            # Auto-align encoder with actual FAISS index dimension
            if self.vector_dim == 768 and "MiniLM" in self.model_name:
                self.model_name = "all-mpnet-base-v2"
            elif self.vector_dim == 384 and "mpnet" in self.model_name:
                self.model_name = "all-MiniLM-L6-v2"

            print(f"[Encoder] Loading embedding model '{self.model_name}' (vector_dim={self.vector_dim})...")
            start = time.time()
            self.encoder = SentenceTransformer(self.model_name, device='cpu')
            
            # Apply INT8 dynamic quantization for cloud environments (Render / low-RAM hosts)
            if IS_CLOUD:
                print("[Encoder] Applying INT8 dynamic quantization to reduce RAM...")
                self.encoder[0].auto_model = torch.quantization.quantize_dynamic(
                    self.encoder[0].auto_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                if hasattr(self.encoder[0], 'max_seq_length'):
                    self.encoder[0].max_seq_length = 128
                gc.collect()
            
            with torch.inference_mode():
                self.encoder.encode(["warmup query"], normalize_embeddings=True, show_progress_bar=False)
            print(f"[Encoder] Model loaded and pre-warmed in {time.time()-start:.2f}s")
        return self.encoder

    def _get_metadata_by_line(self, line_idx: int) -> dict:
        """O(1) random access metadata lookup using byte offsets."""
        if line_idx < 0 or line_idx >= len(self.line_offsets):
            return {"doc_id": line_idx, "chunk_id": 0, "text": "", "meta": {}}

        if not os.path.exists(self.metadata_path):
            return {"doc_id": line_idx, "chunk_id": 0, "text": "", "meta": {}}

        try:
            offset = self.line_offsets[line_idx]
            with open(self.metadata_path, 'rb') as f:
                f.seek(offset)
                line = f.readline()
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
