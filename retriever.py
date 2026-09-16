import os
import gc
import json
import time
import array
import hashlib
import logging
import urllib.request
import numpy as np
import faiss

faiss.omp_set_num_threads(1)
logger = logging.getLogger("omnisearch")

# Cloud Dataset Download URLs
DEFAULT_INDEX_URL = os.getenv(
    "INDEX_DOWNLOAD_URL",
    "https://huggingface.co/datasets/Anbanand/OmniSearch_RAG/resolve/main/faiss_index_8bit.index"
)
DEFAULT_META_URL = os.getenv(
    "METADATA_DOWNLOAD_URL",
    "https://huggingface.co/datasets/Anbanand/OmniSearch_RAG/resolve/main/faiss_metadata.jsonl"
)

# ONNX model files from HuggingFace (all-MiniLM-L6-v2)
ONNX_MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
ONNX_TOKENIZER_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", "onnx_model")

# SEC-07: SHA256 checksums for integrity verification of downloaded files
# These hashes should be updated when the model version changes
EXPECTED_HASHES = {
    "model.onnx": os.getenv("ONNX_MODEL_SHA256", ""),  # Set via env var or leave empty to skip
    "tokenizer.json": os.getenv("ONNX_TOKENIZER_SHA256", ""),
}


class LightEncoder:
    """
    Ultra-lightweight sentence encoder using ONNX Runtime (~30 MB RAM).
    Replaces PyTorch + sentence-transformers (~250 MB RAM).
    Implements the same encode pipeline: tokenize → transformer → mean pooling → normalize.
    """

    def __init__(self, model_dir=ONNX_MODEL_DIR, max_seq_length=128):
        self.model_dir = model_dir
        self.max_seq_length = max_seq_length
        self.session = None
        self.tokenizer = None

    def _ensure_files(self):
        """Download ONNX model and tokenizer if not present."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "model.onnx")
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")

        if not os.path.exists(model_path):
            logger.info("[ONNX] Downloading model.onnx (~25 MB)...")
            self._download(ONNX_MODEL_URL, model_path)
            # SEC-07: Verify integrity
            if not self._verify_hash(model_path, EXPECTED_HASHES.get("model.onnx", "")):
                raise RuntimeError("ONNX model integrity check failed — possible tampering detected")
            logger.info("[ONNX] model.onnx downloaded and verified.")

        if not os.path.exists(tokenizer_path):
            logger.info("[ONNX] Downloading tokenizer.json...")
            self._download(ONNX_TOKENIZER_URL, tokenizer_path)
            if not self._verify_hash(tokenizer_path, EXPECTED_HASHES.get("tokenizer.json", "")):
                raise RuntimeError("Tokenizer integrity check failed — possible tampering detected")
            logger.info("[ONNX] tokenizer.json downloaded and verified.")

    def _download(self, url, dest, chunk_size=2 * 1024 * 1024):
        """Stream download in 2 MB chunks to avoid RAM spikes."""
        import requests
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        gc.collect()

    @staticmethod
    def _verify_hash(filepath: str, expected_hash: str) -> bool:
        """SEC-07: Verify SHA256 hash of downloaded file."""
        if not expected_hash:
            return True  # Skip verification if no hash provided
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        actual = sha256.hexdigest()
        if actual != expected_hash:
            logger.error(f"[SEC-07] Hash mismatch for {filepath}: expected={expected_hash[:16]}... got={actual[:16]}...")
            os.remove(filepath)  # Remove potentially tampered file
            return False
        logger.info(f"[SEC-07] Hash verified for {os.path.basename(filepath)}")
        return True

    def load(self):
        """Lazy-load the ONNX model and tokenizer."""
        if self.session is not None:
            return

        self._ensure_files()

        import onnxruntime as ort
        from tokenizers import Tokenizer

        start = time.time()

        # Load tokenizer (Rust-based, ~5 MB RAM)
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.tokenizer.enable_padding(length=self.max_seq_length)

        # Load ONNX session (CPU only, ~30 MB RAM)
        model_path = os.path.join(self.model_dir, "model.onnx")
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=['CPUExecutionProvider']
        )

        # Cache input names for fast lookup
        self._input_names = {inp.name for inp in self.session.get_inputs()}

        # Pre-warm with a dummy query
        self.encode(["warmup"], normalize_embeddings=True)

        print(f"[ONNX Encoder] Loaded and pre-warmed in {time.time() - start:.2f}s")
        gc.collect()

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        """Encode texts to embeddings using ONNX Runtime (same output as SentenceTransformer)."""
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # Build feed dict (only include inputs the model expects)
        feeds = {}
        if 'input_ids' in self._input_names:
            feeds['input_ids'] = input_ids
        if 'attention_mask' in self._input_names:
            feeds['attention_mask'] = attention_mask
        if 'token_type_ids' in self._input_names:
            feeds['token_type_ids'] = np.zeros_like(input_ids, dtype=np.int64)

        # Run ONNX inference
        outputs = self.session.run(None, feeds)

        # Mean pooling over token embeddings (same as sentence-transformers)
        token_embeddings = outputs[0].astype(np.float32)  # (batch, seq_len, hidden_dim)
        seq_len = token_embeddings.shape[1]
        mask = attention_mask[:, :seq_len, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings


class FAISSMetadataRetriever:
    """
    RAG Retriever optimized for low-RAM deployments (384-dim MiniLM):
    - Uses 8-bit FAISS Index (memory-mapped).
    - Uses ONNX Runtime encoder (~30 MB) instead of PyTorch (~250 MB).
    - Total RAM: ~100 MB (fits in Render 512 MB free tier).
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
        """Lazy load ONNX encoder (~30 MB RAM vs ~250 MB for PyTorch)."""
        if self.encoder is None:
            print(f"[Encoder] Loading ONNX encoder for '{self.model_name}' (dim={self.vector_dim})...")
            self.encoder = LightEncoder(model_dir=ONNX_MODEL_DIR, max_seq_length=128)
            self.encoder.load()
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
                    # SEC-15: Safe JSON deserialization with validation
                    data = json.loads(line.decode('utf-8'))
                    if not isinstance(data, dict):
                        return {"doc_id": line_idx, "chunk_id": 0, "text": "", "meta": {}}
                    # Ensure expected keys exist and are correct types
                    return {
                        "doc_id": data.get("doc_id", line_idx),
                        "chunk_id": data.get("chunk_id", 0),
                        "text": str(data.get("text", ""))[:5000],  # Cap text length
                        "meta": data.get("meta", {}) if isinstance(data.get("meta"), dict) else {},
                    }
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
            logger.warning(f"[Metadata] Invalid data at line {line_idx}: {type(e).__name__}")
        except Exception as e:
            logger.warning(f"[Metadata Error] Line {line_idx}: {type(e).__name__}")

        return {"doc_id": line_idx, "chunk_id": 0, "text": "", "meta": {}}

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list:
        """
        Perform high-speed vector similarity search for a query string.
        Returns a list of dictionary matches with document text and metadata.
        """
        if not query or not query.strip() or self.total_vectors == 0:
            return []

        encoder = self._get_encoder()

        # Encode query to numpy array
        query_vec = encoder.encode([query], normalize_embeddings=True)
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
