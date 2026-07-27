import sys
import io
import argparse

# Ensure UTF-8 output encoding for Windows command line
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from rag_engine import RAGEngine, FAISSMetadataRetriever

def main():
    parser = argparse.ArgumentParser(description="CLI Tool for querying the FAISS RAG Index")
    parser.add_argument("query", type=str, nargs="?", help="The search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold")
    
    args = parser.parse_args()
    
    query = args.query
    if not query:
        query = input("Enter your search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    print(f"\nSearching RAG Dataset for: '{query}' (top_k={args.top_k})\n" + "-"*50)
    
    retriever = FAISSMetadataRetriever()
    engine = RAGEngine(retriever=retriever)
    
    result = engine.generate_answer(query, top_k=args.top_k, score_threshold=args.threshold)
    
    print("\n--- [RAG SYNTHESIZED RESPONSE] ---")
    print(result["answer"])
    
    print(f"\n--- [RETRIEVED SOURCES ({len(result['sources'])} found)] ---")
    for s in result["sources"]:
        print(f"\nRank {s['rank']} | Doc ID: {s['doc_id']} | Chunk: {s['chunk_id']} | Similarity: {s['similarity']:.4f}")
        print(f"Snippet: {s['text'][:200]}...")

if __name__ == "__main__":
    main()
