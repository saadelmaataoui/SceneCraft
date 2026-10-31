"""GPU service: AudioGen (sound generation) + Audiobox Aesthetics (ranking).

Deploy once: modal deploy modal_audio.py

Two images on purpose: audiocraft pins torch 2.1, audiobox_aesthetics needs
torch 2.2 or later. Never install both libraries in one image.
"""
import io

import modal

app = modal.App("scenecraft-audio")
weights = modal.Volume.from_name("scenecraft-weights", create_if_missing=True)

# audiocraft 1.3.0 hard-pins av==11.0.0, which is sdist-only on PyPI now and
# needs FFmpeg headers to build. Install its deps ourselves with an av that
# ships wheels, then audiocraft with --no-deps so pip can't swap av back.
gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "av==12.3.0",
        "torch==2.1.0", "torchaudio==2.1.0", "torchvision==0.16.0", "torchtext==0.16.0",
        "xformers<0.0.23", "numpy<2",
        "transformers>=4.31.0,<4.46",  # newer releases drop torch 2.1
        "spacy==3.7.5",  # 3.8 pulls a numpy-2 thinc
        "click<8.2",  # spacy imports click.parser; newer typer no longer pulls click in
        "einops", "flashy>=0.0.1", "hydra-core>=1.1", "hydra_colorlog", "julius",
        "num2words", "sentencepiece", "huggingface_hub", "tqdm", "demucs", "librosa",
        "torchmetrics", "encodec", "protobuf", "soundfile",
    )
    .pip_install("audiocraft==1.3.0", extra_options="--no-deps")
    .env({"HF_HOME": "/weights/hf"})
)

judge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("audiobox_aesthetics", "soundfile", "requests")  # it imports requests without declaring it
    .env({"HF_HOME": "/weights/hf"})
)


def wav_bytes(x, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


# GPU lists are fallbacks in order: Modal takes the first type it has free,
# so an L4 shortage doesn't leave calls queued forever.
@app.cls(gpu=["L4", "A10G", "T4"], image=gen_image, volumes={"/weights": weights}, scaledown_window=600)
class Foley:
    @modal.enter()
    def load(self):
        from audiocraft.models import AudioGen
        self.model = AudioGen.get_pretrained("facebook/audiogen-medium")

    @modal.method()
    def takes(self, prompt: str, seconds: float, k: int = 4) -> list[bytes]:
        self.model.set_generation_params(duration=seconds)
        wavs = self.model.generate([prompt] * k)  # k different takes: [k, 1, samples] at 16 kHz
        return [wav_bytes(w[0].cpu().numpy(), self.model.sample_rate) for w in wavs]


@app.cls(gpu=["T4", "L4", "A10G"], image=judge_image, volumes={"/weights": weights}, scaledown_window=600)
class Judge:
    @modal.enter()
    def load(self):
        from audiobox_aesthetics.infer import initialize_predictor
        self.predictor = initialize_predictor()

    @modal.method()
    def score(self, clips: list[bytes]) -> list[dict]:
        import soundfile as sf
        import torch
        batch = []
        for c in clips:
            x, sr = sf.read(io.BytesIO(c), dtype="float32")
            batch.append({"path": torch.from_numpy(x)[None], "sample_rate": sr})
        return self.predictor.forward(batch)  # one dict per clip: CE, CU, PC, PQ