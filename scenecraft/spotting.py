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
        "mode": {"type": "string", "enum": ["impact", "continuous"],
                 "description": "impact: a visible contact instant (hit, tap, set down). "
                                "continuous: sound lasting while it moves (rub, slide, roll)."},
        "sound_prompt": {"type": "string",
                         "description": "one sentence for a text-to-audio model: attack, material, "
                                        "object, action, surface, intensity, mic distance; ends with "
                                        "'no music, no speech'; no visual words."},
        "approx_times_s": {"type": "array", "items": {"type": "number"},
                           "description": "hit times read off the burned-in clock; never invent a "
                                          "rhythm; empty for continuous."},
    },
}
SPOTTING = {
    "type": "object", "additionalProperties": False, "required": ["scene", "sources"],
    "properties": {
        "scene": {
            "type": "object", "additionalProperties": False,
            "required": ["location", "acoustics", "ambience_prompt"],
            "properties": {
                "location": {"type": "string"},
                "acoustics": {"type": "string"},
                "ambience_prompt": {"type": "string",
                                    "description": "room tone from the acoustics only; never visual "
                                                   "words; ends with 'no music, no speech'"},
            },
        },
        "sources": {"type": "array", "items": SOURCE},
    },
}

BRIEF = """You are a foley supervisor running a spotting session on a silent clip.
The clock burned into the top-left corner is the clip time. Read every time from it.
List each sound the clip needs. Split by SOUND, not by object: if the same hands
both clap and rub, emit two sources (e.g. "hands_clap" impact + "hands_rub" continuous)
sharing one sam_concept.
- sam_concept: ONE short noun phrase a segmentation model can find ("shoe", "door", "mug"). Never a list.
- category: feet (footsteps), moves (cloth and body movement), specifics (props and objects).
- mode: impact ONLY for a visible contact instant (hit, tap, set down, snap shut).
  continuous for sound that lasts while something moves (rub, slide, zip, roll).
  When unsure, watch the motion: instant stop = impact, sustained motion = continuous.
- sound_prompt: one sentence written for a text-to-audio model. Name the attack, material,
  object, action, surface, intensity and mic distance (close-miked or distant).
  Impact: start with "Single", describe ONE hit, end with "one hit only, followed by silence".
  Continuous: describe a steady loopable texture with "no impacts".
  End every prompt with: no music, no speech.
  Never use visual words (close-up, white, studio, video, frame, on screen): describe only sound.
- approx_times_s: read each hit time off the burned-in clock. Never invent an even rhythm;
  list only hits you actually see. Empty for continuous sources.
scene.ambience_prompt: the room tone for this place from its ACOUSTICS only
(e.g. "faint dead-room tone, very dry, no echo, no music, no speech").
Never mention colors, objects, or what the room looks like."""


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
