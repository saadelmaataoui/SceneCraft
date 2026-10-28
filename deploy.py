"""One public URL on Modal, serving the Gradio app directly.

modal deploy modal_audio.py   # once, or whenever that file changes
modal deploy deploy.py

One container on purpose: this plus modal_audio.py's scaledown_window keep
GPU containers warm and the on-disk cache (scenecraft/cache.py) consistent
for the judging window. Set min_containers back to 0 after judging.
"""
import modal

app = modal.App("scenecraft")
jobs = modal.Volume.from_name("scenecraft-jobs", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "fonts-dejavu-core")
         .pip_install("gradio==5.9.1", "requests", "openai", "modal", "numpy", "scipy",
                       "soundfile", "pyloudnorm")
         .add_local_python_source("app", "scenecraft"))


@app.function(image=image, volumes={"/data": jobs},
              secrets=[modal.Secret.from_name("meta-model-api")],
              min_containers=1, max_containers=1, timeout=900)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os

    from gradio.routes import mount_gradio_app
    from fastapi import FastAPI

    os.environ.setdefault("SCENECRAFT_CACHE_DIR", "/data/cache")
    from app import demo

    return mount_gradio_app(FastAPI(), demo, path="/")
