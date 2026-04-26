from shiftlog_gym.server.app import app
from observatory.gradio_app import demo
import gradio as gr
from fastapi.responses import RedirectResponse

@app.get("/")
def redirect_root():
    return RedirectResponse(url="/dashboard")

# Fuse OpenEnv API with the Gradio UI at /dashboard to fix HF Space static routing bugs
app = gr.mount_gradio_app(app, demo, path="/dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
