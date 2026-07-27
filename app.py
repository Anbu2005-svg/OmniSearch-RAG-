import os
import time
import pandas as pd
import streamlit as st
from retriever import FAISSMetadataRetriever
from rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="RAG Intelligence Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 24px;
        color: #f8fafc;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache retriever and engine initialization
@st.cache_resource(show_spinner="Loading FAISS Index & Embedding Model...")
def load_rag_components():
    retriever = FAISSMetadataRetriever(
        index_path="faiss_index.index",
        metadata_path="faiss_metadata.jsonl",
        model_name="all-mpnet-base-v2"
    )
    engine = RAGEngine(retriever=retriever)
    return retriever, engine

try:
    retriever, engine = load_rag_components()
    index_loaded = True
except Exception as e:
    index_loaded = False
    load_error = str(e)

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/database.png", width=64)
    st.title("⚙️ RAG Settings")
    
    st.subheader("Retrieval Controls")
    top_k = st.slider("Top-K Documents", min_value=1, max_value=20, value=5, step=1)
    score_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    
    st.divider()
    st.subheader("🌐 Cloud LLM Model Settings")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "openai_compatible", "local"],
        format_func=lambda x: {
            "ollama": "🦙 Ollama (Local / Cloud API)",
            "openai_compatible": "☁️ Cloud LLM API (Groq / OpenAI / OpenRouter)",
            "local": "📄 Context Synthesizer (Offline)"
        }[x]
    )
    
    api_key = ""
    api_url = ""
    model_name = "llama3"

    if llm_provider == "ollama":
        api_url = st.text_input("Ollama Endpoint URL", value="http://localhost:11434", help="Default: http://localhost:11434 or custom Ollama Cloud URL")
        api_key = st.text_input("Ollama API Key (Optional)", type="password", help="Enter API Key if using protected Ollama Cloud host")
        model_name = st.text_input("Ollama Model Name", value="llama3", help="e.g. llama3, mistral, gemma:7b, qwen2")
    
    elif llm_provider == "openai_compatible":
        api_url = st.text_input("Base API Endpoint", value="https://api.openai.com/v1", help="e.g. https://api.groq.com/openai/v1 or https://openrouter.ai/api/v1")
        api_key = st.text_input("Cloud API Key", type="password")
        model_name = st.text_input("Cloud Model Name", value="gpt-3.5-turbo", help="e.g. llama-3.1-70b-versatile, gpt-4o-mini, mistral-large")

    st.divider()
    st.info("💡 **FAISS Vector Store**: 200,000 document chunks indexed with 768-dim embeddings.")

# Header
st.title("🔍 RAG Application")
st.markdown("Retrieval-Augmented Generation over 200,000 document chunks using FAISS vector search & Cloud LLMs.")

# Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Chunks</div>
        <div class="metric-value">{retriever.total_vectors if index_loaded else 0:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vector Dim</div>
        <div class="metric-value">{retriever.vector_dim if index_loaded else 768} d</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">FAISS Size</div>
        <div class="metric-value">614.4 MB</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    status_color = "#4ade80" if index_loaded else "#f87171"
    status_text = "Ready" if index_loaded else "Error"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">FAISS Index</div>
        <div class="metric-value" style="color: {status_color}; font-size: 22px;">● {status_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

if not index_loaded:
    st.error(f"Failed to load FAISS index: {load_error}")
    st.stop()

# Sample Query Shortcuts
st.subheader("💡 Sample Search Queries")
sample_cols = st.columns(3)

query_input = ""
if sample_cols[0].button("⛽ Petrol & Diesel Quality Directive"):
    query_input = "quality of petrol and diesel fuels directive"
if sample_cols[1].button("🔬 Ultra-fine Particle Emissions"):
    query_input = "ultra-fine particle emissions from GDI engines"
if sample_cols[2].button("🚨 Emergency Fuel Quality Standards"):
    query_input = "emergency fuel availability and quality exemptions"

# Search Input Box
user_query = st.text_input(
    "Enter your question or search query:",
    value=query_input if query_input else "",
    placeholder="e.g. European Parliament report on fuel quality standards...",
    key="search_input"
)

if user_query:
    with st.spinner("Retrieving relevant document chunks and generating answer..."):
        start_time = time.time()
        res = engine.generate_answer(
            query=user_query,
            top_k=top_k,
            score_threshold=score_threshold,
            llm_provider=llm_provider,
            api_key=api_key,
            api_url=api_url,
            model_name=model_name
        )
        latency = time.time() - start_time

    # Results Display
    st.success(f"Generated response in {latency:.3f} seconds ({len(res['sources'])} source chunks retrieved)")
    
    st.markdown("### 📝 AI Generated Answer")
    st.markdown(res["answer"])
    
    st.divider()
    
    # Source Inspector
    st.markdown("### 📚 Source Document Citations")
    if res["sources"]:
        for src in res["sources"]:
            with st.expander(f"Rank #{src['rank']} | Doc ID: {src['doc_id']} | Chunk: {src['chunk_id']} | Similarity: {src['similarity']:.2%}"):
                st.write(f"**Similarity Score:** `{src['similarity']:.4f}` | **Distance:** `{src['distance']:.4f}`")
                st.write("**Text Excerpt:**")
                st.info(src['text'])
                if src.get('meta'):
                    st.json(src['meta'])
    else:
        st.warning("No document chunks passed the similarity threshold filter.")
