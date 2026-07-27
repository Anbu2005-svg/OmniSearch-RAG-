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

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Start FastAPI server on HuggingFace Spaces port
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
