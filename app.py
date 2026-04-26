import os
from shiftlog_gym.server.app import app
from observatory.gradio_app import demo
import gradio as gr

# Automatically detect Hugging Face Space root path for correct subdomain resolution
# (Fixes "400 Bad Request" and "Failed to fetch" on HF)
root_path = os.environ.get("GRADIO_ROOT_PATH", "")

# Fuse OpenEnv API with the Gradio UI natively
app = gr.mount_gradio_app(app, demo, path="/", root_path=root_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
