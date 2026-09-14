FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY retriever.py .
COPY rag_engine.py .
COPY server.py .

# Copy data files (these must be present in the repo or volume)
# For HuggingFace Spaces, upload these via Git LFS or the UI
COPY faiss_index.index* ./
COPY faiss_metadata.jsonl* ./

# Expose port (default 7860, or dynamically assigned by host)
EXPOSE 7860 80 8080 10000

# Start FastAPI server, respecting the PORT environment variable if provided by the host
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
