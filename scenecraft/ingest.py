"""ffmpeg: constant-frame-rate master + two proxies, ffprobe sizes.

Phones record variable frame rate, which makes events drift against the
picture. Convert to a constant rate first, then derive both proxies from
that master. Uploads are capped at 15 seconds.
"""
import json
import os
import shutil
import subprocess
from dataclasses import dataclass

MAX_SECONDS = 15
SAM_FPS = 15


@dataclass
class Clip:
    id: str
    master_path: str
    sam_path: str
    muse_path: str
    width: int
    height: int
    n_frames: int
    duration_s: float


def _run(*args):
    subprocess.run(args, check=True, capture_output=True)


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream.get("nb_frames", 0))


def prepare(take_path: str, out_dir: str) -> Clip:
    """Build master.mp4, sam.mp4, muse.mp4 for one upload into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    clip_id = os.path.splitext(os.path.basename(take_path))[0]

    master_path = os.path.join(out_dir, "master.mp4")
    sam_path = os.path.join(out_dir, "sam.mp4")
    muse_path = os.path.join(out_dir, "muse.mp4")

    # 1. Constant-frame-rate master, first 15 s only, no audio (picture only).
    _run("ffmpeg", "-y", "-i", take_path, "-t", str(MAX_SECONDS),
         "-vf", "fps=30,scale=-2:1080", "-an",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", master_path)

    # 2. Clean proxy for SAM 3.1. 15 fps halves the frame bill and is plenty
    #    for footsteps.
    _run("ffmpeg", "-y", "-i", master_path,
         "-vf", f"fps={SAM_FPS},scale=-2:720",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", sam_path)

    # 3. Proxy for Muse Spark with the clock burned in, so it can read times
    #    off the picture.
    fontfile = ""
    if os.name == "nt" and shutil.which("ffmpeg"):
        win_font = r"C\:/Windows/Fonts/arial.ttf"
        if os.path.exists(r"C:\Windows\Fonts\arial.ttf"):
            fontfile = f":fontfile={win_font}"
    _run("ffmpeg", "-y", "-i", master_path,
         "-vf", (f"fps={SAM_FPS},scale=-2:480,"
                 r"drawtext=text='%{pts\:hms}':x=12:y=12:fontsize=26:"
                 f"fontcolor=white:box=1:boxcolor=black@0.6{fontfile}"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", muse_path)

    width, height, n_frames = _probe(sam_path)
    duration_s = n_frames / SAM_FPS if n_frames else 0.0

    return Clip(
        id=clip_id, master_path=master_path, sam_path=sam_path, muse_path=muse_path,
        width=width, height=height, n_frames=n_frames, duration_s=duration_s,
    )
