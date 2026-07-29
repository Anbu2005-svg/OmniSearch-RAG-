import gradio as gr
from server import app as api_app

demo = gr.Interface(
    fn=lambda: "OmniSearch RAG API is actively running! The FastAPI endpoints are available at the root URL.",
    inputs=None,
    outputs="text",
    title="OmniSearch RAG Backend API",
    description="This Space hosts the FastAPI backend for OmniSearch. Use the /api/search endpoint for RAG queries.",
)

app = gr.mount_gradio_app(api_app, demo, path="/ui")
