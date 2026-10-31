"""sha256(video) + stage + version -> stored result.

Re-running an example costs nothing; bump PIPELINE_VERSION when you change
a stage's logic so stale cached output doesn't leak through.
"""
import hashlib
import os
import pickle

PIPELINE_VERSION = "2"
CACHE_DIR = os.environ.get("SCENECRAFT_CACHE_DIR", ".cache")


def video_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key_path(video_hash_hex: str, stage: str) -> str:
    return os.path.join(CACHE_DIR, video_hash_hex, f"{stage}.v{PIPELINE_VERSION}.pkl")


def get(video_hash_hex: str, stage: str):
    path = _key_path(video_hash_hex, stage)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def put(video_hash_hex: str, stage: str, value) -> None:
    path = _key_path(video_hash_hex, stage)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
