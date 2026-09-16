FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SEC-08: Create non-root user BEFORE copying application files
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy application code
COPY retriever.py .
COPY rag_engine.py .
COPY server.py .

# Copy bundled lightweight 50k dataset files
COPY faiss_index_8bit_50k.index* ./
COPY faiss_metadata_50k.jsonl* ./
COPY faiss_index_8bit.index* ./
COPY faiss_metadata.jsonl* ./

# Pre-download ONNX model during build (cached in image layer, ~25 MB)
RUN python -c "from retriever import LightEncoder; e = LightEncoder(); e._ensure_files(); print('[Docker] ONNX model pre-downloaded')"

# SEC-08: Set ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (default 10000 for SnapDeploy/Render, or dynamically assigned by host)
EXPOSE 10000

# Start FastAPI server, respecting the PORT environment variable if provided by the host
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
