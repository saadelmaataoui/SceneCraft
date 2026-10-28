"""GPU service: AudioGen (sound generation) + Audiobox Aesthetics (ranking).

Deploy once: modal deploy modal_audio.py

Two images on purpose: audiocraft pins torch 2.1, audiobox_aesthetics needs
torch 2.2 or later. Never install both libraries in one image.
"""
import io

import modal

app = modal.App("scenecraft-audio")
weights = modal.Volume.from_name("scenecraft-weights", create_if_missing=True)

gen_image = (modal.Image.debian_slim(python_version="3.10")
             .apt_install("ffmpeg", "git")
             .pip_install("audiocraft", "soundfile")
             .env({"HF_HOME": "/weights/hf"}))
judge_image = (modal.Image.debian_slim(python_version="3.10")
               .pip_install("audiobox_aesthetics", "soundfile")
               .env({"HF_HOME": "/weights/hf"}))


def wav_bytes(x, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


@app.cls(gpu="L4", image=gen_image, volumes={"/weights": weights}, scaledown_window=600)
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


@app.cls(gpu="T4", image=judge_image, volumes={"/weights": weights}, scaledown_window=600)
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
