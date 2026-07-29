import gradio as gr
from server import app as api_app

# Create a simple Gradio UI to satisfy Hugging Face's requirements
def status_check():
    return "OmniSearch RAG API is actively running! The FastAPI endpoints are available at the root URL."

demo = gr.Interface(
    fn=status_check,
    inputs=None,
    outputs="text",
    title="OmniSearch RAG Backend API",
    description="This Space hosts the FastAPI backend for OmniSearch. Use the /api/search endpoint for RAG queries."
)

# Mount the Gradio UI at /ui, while keeping your FastAPI app at the root (/)
app = gr.mount_gradio_app(api_app, demo, path="/ui")
