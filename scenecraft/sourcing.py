"""Calls the Modal audio service: four AudioGen takes per sound, scored and
ranked by Audiobox Aesthetics, the top three kept as round-robin variants."""
import re

import modal

Foley = modal.Cls.from_name("scenecraft-audio", "Foley")
Judge = modal.Cls.from_name("scenecraft-audio", "Judge")

SECONDS = {"impact": 1.5, "continuous": 5.0, "ambience": 10.0}

_CLOSEUP = re.compile(r"\bclose-?ups?\b", re.IGNORECASE)
_VOID = re.compile(r"\bwhite studio void\b", re.IGNORECASE)
_NOISE = re.compile(r",?\s*no music, no speech\.?", re.IGNORECASE)
_WS = re.compile(r"\s+")


def build_prompt(base: str, mode: str) -> str:
    """Normalize a model-written prompt for AudioGen: acoustic wording, a
    mode-specific suffix (one clean hit vs. a steady texture), and a single
    noise-free ending. Already-clean prompts pass through unchanged."""
    p = _WS.sub(" ", (base or "").strip().strip('"')).strip()
    p = _CLOSEUP.sub("close-miked", p)
    p = _VOID.sub("dead room", p)
    p = _WS.sub(" ", _NOISE.sub("", p)).strip().rstrip(".")
    if not p:
        p = "Single neutral foley hit" if mode == "impact" else "Neutral room texture"
    low = p.lower()
    if mode == "impact" and "one hit only" not in low:
        p += ". One hit only, followed by silence"
    elif mode == "continuous" and "no impacts" not in low:
        p += ". Steady loopable texture, no impacts"
    return p + ", no music, no speech."


async def source(prompt, mode, keep=3):
    prompt = build_prompt(prompt, mode)
    clips = await Foley().takes.remote.aio(prompt, SECONDS[mode], 4)
    scores = await Judge().score.remote.aio(clips)
    ranked = sorted(zip(scores, clips), key=lambda sc: sc[0]["PQ"] + sc[0]["CU"], reverse=True)
    return [clip for _, clip in ranked[:keep]]  # best takes become round-robin variants
