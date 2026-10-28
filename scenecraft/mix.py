"""Place, pan, vary, master.

Three details make it sound edited rather than generated: trim each take
so its attack is sample zero (the hit lands on its frame), vary pitch and
level slightly per hit, and pan by where the object is on screen.
"""
import io
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

from scenecraft.sync import FPS

SR = 48_000


def load(wav_bytes):
    """AudioGen output (16 kHz mono) -> 48 kHz float32."""
    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return resample_poly(x, SR, sr).astype(np.float32)


def trim_to_transient(x, thresh_db=-30):
    """Start the take at its attack so the hit lands on its frame."""
    peak = np.abs(x).max()
    if peak == 0:
        return x
    start = int(np.argmax(np.abs(x) > peak * 10 ** (thresh_db / 20)))
    x = x[max(0, start - int(0.005 * SR)):].copy()
    fade = min(int(0.002 * SR), len(x))
    x[:fade] *= np.linspace(0, 1, fade)
    return x


def place(bus, x, t, pan, gain_db):
    i = int(t * SR)
    if i >= len(bus):
        return
    j = min(len(bus), i + len(x))
    g, a = 10 ** (gain_db / 20), (pan + 1) * np.pi / 4  # constant-power pan
    bus[i:j, 0] += x[: j - i] * g * np.cos(a)
    bus[i:j, 1] += x[: j - i] * g * np.sin(a)


def render(events, takes, textures, ambience, duration_s):
    """takes: {source_id: [48 kHz arrays]}; textures: {source_id: (array, per-frame envelope)}."""
    n = max(1, int(duration_s * SR))
    stems = defaultdict(lambda: np.zeros((n, 2), np.float32))
    rng, count = np.random.default_rng(7), defaultdict(int)
    for e in events:
        sid = e["source_id"]
        if sid not in takes or not takes[sid]:
            continue
        x = trim_to_transient(takes[sid][count[sid] % len(takes[sid])])  # round-robin takes
        count[sid] += 1
        rate = 2 ** (rng.uniform(-1, 1) / 12)                            # up to 1 semitone either way
        x = resample_poly(x, 1000, round(1000 * rate)).astype(np.float32)
        place(stems[sid], x, e["t"], e["pan"], e["gain_db"] + rng.uniform(-1.5, 1.5))
    for sid, (tex, env) in textures.items():
        level = np.interp(np.arange(n) / SR, np.arange(len(env)) / FPS, env)
        tiled = np.resize(tex, n)
        stems[sid] += (tiled * level * 0.5)[:, None]
    if ambience is not None and len(ambience):
        stems["ambience"] += (np.resize(ambience, n) * 10 ** (-18 / 20))[:, None]
    master = sum(stems.values()) if stems else np.zeros((n, 2), np.float32)
    loudness = pyln.Meter(SR).integrated_loudness(master)
    k = 10 ** ((-16.0 - loudness) / 20) if np.isfinite(loudness) and loudness > -70 else 1.0
    return {s: v * k for s, v in stems.items()}, np.clip(master * k, -1, 1)


@dataclass
class MixResult:
    out_path: str
    qc_path: str
    stem_paths: dict
    master_path: str
    url: str


def _mux(video_path, audio_path, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_path],
        check=True, capture_output=True,
    )


def render_clip(clip, cue, envelopes, takes_bytes, out_dir):
    """clip: scenecraft.ingest.Clip. cue: dict from sync.build_cue_sheet.
    envelopes: dict from sync.build_cue_sheet.
    takes_bytes: {source_id: [wav bytes]}, PLUS an "ambience" key holding a
    single-element list (keep=1 from sourcing.source) for the room tone bed.
    """
    os.makedirs(out_dir, exist_ok=True)

    takes_bytes = dict(takes_bytes)  # don't mutate the caller's dict
    ambience_bytes = takes_bytes.pop("ambience", [])

    takes = {sid: [load(w) for w in wavs] for sid, wavs in takes_bytes.items()}
    textures = {}
    for sid, env in envelopes.items():
        tex_takes = takes.get(sid)
        if tex_takes:
            textures[sid] = (tex_takes[0], env)
    ambience = load(ambience_bytes[0]) if ambience_bytes else np.zeros(SR, np.float32)

    stems, master = render(cue["events"], takes, textures, ambience, cue["clip"]["duration_s"])

    stem_paths = {}
    for sid, wave in stems.items():
        path = os.path.join(out_dir, f"{sid}.wav")
        sf.write(path, wave, SR)
        stem_paths[sid] = path

    master_path = os.path.join(out_dir, "mix.wav")
    sf.write(master_path, master, SR)

    out_path = os.path.join(out_dir, "out.mp4")
    qc_path = os.path.join(out_dir, "qc.mp4")
    # What users download: original picture + new mix.
    _mux(clip.master_path, master_path, out_path)
    # What QC watches: the timecoded proxy + the same mix, so issues come back with readable times.
    _mux(clip.muse_path, master_path, qc_path)

    return MixResult(out_path=out_path, qc_path=qc_path, stem_paths=stem_paths,
                      master_path=master_path, url=out_path)
