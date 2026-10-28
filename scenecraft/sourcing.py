"""Calls the Modal audio service: four AudioGen takes per sound, scored and
ranked by Audiobox Aesthetics, the top three kept as round-robin variants."""
import modal

Foley = modal.Cls.from_name("scenecraft-audio", "Foley")
Judge = modal.Cls.from_name("scenecraft-audio", "Judge")

SECONDS = {"impact": 1.5, "continuous": 5.0, "ambience": 10.0}


async def source(prompt, mode, keep=3):
    clips = await Foley().takes.remote.aio(prompt, SECONDS[mode], 4)
    scores = await Judge().score.remote.aio(clips)
    ranked = sorted(zip(scores, clips), key=lambda sc: sc[0]["PQ"] + sc[0]["CU"], reverse=True)
    return [clip for _, clip in ranked[:keep]]  # best takes become round-robin variants
