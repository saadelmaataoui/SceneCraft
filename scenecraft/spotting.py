"""Meta Model API: muse-spark-1.3 runs a spotting session on the clip.

A spotting session is where a foley supervisor watches the cut and decides
what needs sound. Muse Spark does that here and returns it in a schema we
control.
"""
import json
import os

from openai import OpenAI

MODEL = "muse-spark-1.3"

_client = None


def get_client():
    """Built lazily: the installed openai SDK raises immediately if no API
    key is present, so constructing this at import time would crash any
    script that merely imports scenecraft before MODEL_API_KEY is set."""
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    return _client

SOURCE = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "sam_concept", "category", "mode", "sound_prompt", "approx_times_s"],
    "properties": {
        "id": {"type": "string"},
        "sam_concept": {"type": "string", "description": "one short noun phrase, e.g. 'shoe'"},
        "category": {"type": "string", "enum": ["feet", "moves", "specifics"]},
        "mode": {"type": "string", "enum": ["impact", "continuous"]},
        "sound_prompt": {"type": "string"},
        "approx_times_s": {"type": "array", "items": {"type": "number"}},
    },
}
SPOTTING = {
    "type": "object", "additionalProperties": False, "required": ["scene", "sources"],
    "properties": {
        "scene": {
            "type": "object", "additionalProperties": False,
            "required": ["location", "acoustics", "ambience_prompt"],
            "properties": {k: {"type": "string"} for k in ["location", "acoustics", "ambience_prompt"]},
        },
        "sources": {"type": "array", "items": SOURCE},
    },
}

BRIEF = """You are a foley supervisor running a spotting session on a silent clip.
The clock burned into the top-left corner is the clip time. Read every time from it.
List each on-screen thing that should make a sound:
- sam_concept: ONE short noun phrase a segmentation model can find ("shoe", "door", "mug"). Never a list.
- category: feet (footsteps), moves (cloth and body movement), specifics (props and objects).
- mode: impact for discrete hits, continuous for sound that lasts while it moves.
- sound_prompt: ONE isolated hit (impact) or a steady texture (continuous). Name the material,
  surface, intensity and mic distance. End with: isolated, dry, no music, no speech.
- approx_times_s: when each hit happens, in seconds. Empty for continuous sources.
scene.ambience_prompt: the room tone or background for this place, ending with: no music, no speech."""


def upload(path):
    with open(path, "rb") as f:
        return get_client().files.create(purpose="user_data", file=f).id


def spot(muse_mp4):
    resp = get_client().responses.create(
        model=MODEL,
        instructions=BRIEF,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Spot this clip."},
            {"type": "input_video", "file_id": upload(muse_mp4)},
        ]}],
        text={"format": {"type": "json_schema", "name": "SpottingList", "strict": True, "schema": SPOTTING}},
    )
    usage = getattr(resp, "usage", None)
    return json.loads(resp.output_text), usage
