import os
import re
import json
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from retriever import FAISSMetadataRetriever

# Load backend .env configuration
load_dotenv(override=True)

# Common domain dictionary for fast rule-based spelling correction
COMMON_TYPOS = {
  "qualty": "quality",
  "qulity": "quality",
  "directiv": "directive",
  "directve": "directive",
  "emision": "emission",
  "emisions": "emissions",
  "gasolin": "gasoline",
  "diesl": "diesel",
  "petrl": "petrol",
  "europen": "european",
  "parliament": "parliament",
  "environmnt": "environment",
  "environmntal": "environmental",
  "standard": "standard",
  "stndard": "standard",
  "rqeuirement": "requirement",
  "rqeuirements": "requirements",
}

class RAGEngine:
    """
    High-Performance RAG Orchestrator:
    - Auto Query Correction & Refinement step (fixes typos before vector search).
    - Sub-millisecond vector retrieval via FAISS & O(1) metadata seeker.
    - Low-latency cloud LLM synthesis with fast fallback.
    """
    def __init__(self, retriever: FAISSMetadataRetriever = None):
        self.retriever = retriever or FAISSMetadataRetriever()

    def correct_and_refine_query(self, query: str) -> str:
        """
        Fast auto-correction step: fixes spelling errors, typos, and normalizes queries.
        """
        if not query or not query.strip():
            return ""

        corrected_words = []
        words = query.strip().split()
        for w in words:
            clean_word = re.sub(r'[^\w\s]', '', w).lower()
            if clean_word in COMMON_TYPOS:
                corrected_words.append(COMMON_TYPOS[clean_word])
            else:
                corrected_words.append(w)

        corrected = " ".join(corrected_words)
        return corrected

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        llm_provider: str = "",
        api_key: str = "",
        api_url: str = "",
        model_name: str = ""
    ) -> Dict[str, Any]:
        """
        Processes a user query by first auto-correcting typos, performing FAISS retrieval, and synthesizing an answer.
        """
        # Reload env in case it changed
        load_dotenv(override=True)

        provider = (llm_provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        url = (api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")).strip()
        key = (api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()
        model = (model_name or os.getenv("MODEL_NAME", "llama3")).strip()

        # 1. AI Auto-Correction Step
        corrected_query = self.correct_and_refine_query(query)
        search_target = corrected_query if corrected_query else query

        # 2. Retrieve context chunks using corrected query
        context_chunks = self.retriever.search(search_target, top_k=top_k, score_threshold=score_threshold)
        
        if not context_chunks:
            return {
                "answer": f"No relevant documents found matching '{search_target}' in the index.",
                "sources": [],
                "query": query,
                "corrected_query": search_target
            }

        # 3. Synthesize answer using selected provider
        if "/v1" in url or provider in ("openai_compatible", "openai"):
            answer = self._call_openai_compatible(search_target, context_chunks, key, url, model)
        elif provider == "ollama":
            answer = self._call_ollama(search_target, context_chunks, key, url, model)
        else:
            answer = self._local_synthesize(search_target, context_chunks)

        return {
            "answer": answer,
            "sources": context_chunks,
            "query": query,
            "corrected_query": search_target
        }

    def _local_synthesize(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Structured context summary fallback."""
        top_doc = chunks[0]
        header = f"**Retrieved Context Summary** (Top match Doc ID #{top_doc['doc_id']}, Similarity: {top_doc['similarity']:.2%}):\n\n"
        
        body_points = []
        for i, chunk in enumerate(chunks[:3], 1):
            snippet = chunk['text'].strip()
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            body_points.append(f"**[Source {i} - Doc #{chunk['doc_id']} Chunk #{chunk['chunk_id']}]**\n> \"{snippet}\"")

        summary_text = "\n\n".join(body_points)
        footer = "\n\n*Tip: Connect your Ollama Cloud / API key in backend `.env` for full AI response generation.*"
        return header + summary_text + footer

    def _call_ollama(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        api_key: str,
        endpoint_url: str,
        model_name: str
    ) -> str:
        """Call Ollama Native Generate API with fast timeout."""
        if not endpoint_url:
            endpoint_url = "http://localhost:11434"

        base_endpoint = endpoint_url.rstrip("/")
        if not base_endpoint.endswith("/api/generate"):
            generate_url = f"{base_endpoint}/api/generate"
        else:
            generate_url = base_endpoint

        context_str = "\n\n".join([f"Document {c['doc_id']}:\n{c['text']}" for c in chunks])
        prompt = (
            f"You are a helpful RAG assistant. Answer the user's question accurately using ONLY the context provided below.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model_name or "llama3",
            "prompt": prompt,
            "stream": False
        }

        try:
            res = requests.post(generate_url, json=payload, headers=headers, timeout=5)
            res.raise_for_status()
            data = res.json()
            return data.get("response", str(data))
        except Exception:
            return self._local_synthesize(query, chunks)

    def _call_openai_compatible(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        api_key: str,
        base_url: str,
        model_name: str
    ) -> str:
        """Call OpenAI-compatible cloud APIs with fast timeout."""
        if not base_url:
            base_url = "https://api.openai.com/v1"

        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        context_str = "\n\n".join([f"Document {c['doc_id']}:\n{c['text']}" for c in chunks])
        system_msg = "You are an intelligent RAG query engine. Answer questions using only the provided context documents."
        user_msg = f"Context:\n{context_str}\n\nUser Question: {query}"

        headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model_name or "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.2
        }

        try:
            res = requests.post(endpoint, json=payload, headers=headers, timeout=5)
            res.raise_for_status()
            data = res.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except Exception:
            return self._local_synthesize(query, chunks)
