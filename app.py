from shiftlog_gym.server.app import app
from observatory.gradio_app import demo
import gradio as gr

# Fuse OpenEnv API with the Gradio UI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
