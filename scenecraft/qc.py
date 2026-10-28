"""Muse Spark watches and listens to our own mix: a re-recording mixer's
sign-off, plus one fix pass.

Trust QC for meaning (wrong material, missing source, a sound that should
not be there). Frame-level timing comes from SAM, not from QC. Stop after
one pass so a run always finishes.
"""
import json

from scenecraft import mix, sourcing, sync
from scenecraft.spotting import get_client, upload

MODEL = "muse-spark-1.3"

QC_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["score", "issues"],
    "properties": {
        "score": {"type": "integer", "description": "1-10: would a sound editor sign this off?"},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["source_id", "time_s", "problem", "fix_prompt"],
            "properties": {
                "source_id": {"type": "string"},
                "time_s": {"type": "number"},
                "problem": {"type": "string", "enum": ["wrong_sound", "missing", "unwanted",
                                                        "too_loud", "too_quiet", "off_sync"]},
                "fix_prompt": {"type": "string", "description": "new sound prompt if the sound is wrong, else empty"},
            },
        }},
    },
}

BRIEF = """You are a re-recording mixer signing off a foley mix against picture.
Watch and listen to this clip. The known sound sources are listed below. Score the mix
1-10 and list any problems, each tied to a source_id and a time in seconds."""


def review(qc_mp4, cue):
    sources_desc = json.dumps([{"id": s["id"], "sound_prompt": s["sound_prompt"]} for s in cue["sources"]])
    resp = get_client().responses.create(
        model=MODEL,
        instructions=BRIEF,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": f"Known sources: {sources_desc}"},
            {"type": "input_video", "file_id": upload(qc_mp4)},
        ]}],
        text={"format": {"type": "json_schema", "name": "QcReview", "strict": True, "schema": QC_SCHEMA}},
    )
    return json.loads(resp.output_text), getattr(resp, "usage", None)


async def _apply_fix(issue, cue, envelopes, takes_bytes):
    sid = issue["source_id"]
    problem = issue["problem"]
    source = next((s for s in cue["sources"] if s["id"] == sid), None)
    if source is None:
        return

    if problem == "wrong_sound" and issue.get("fix_prompt"):
        takes_bytes[sid] = await sourcing.source(issue["fix_prompt"], source["mode"])
    elif problem in ("too_loud", "too_quiet"):
        delta = -4.0 if problem == "too_loud" else 4.0
        for e in cue["events"]:
            if e["source_id"] == sid:
                e["gain_db"] += delta
    elif problem == "unwanted":
        cue["events"] = [e for e in cue["events"]
                          if not (e["source_id"] == sid and abs(e["t"] - issue["time_s"]) <= 0.25)]
    elif problem == "off_sync":
        # Simplified re-snap: nudge this source's events toward the reported time
        # within a tight 0.15s window (full re-snap needs the original SAM motion
        # peaks, which aren't threaded through to this stage).
        for e in cue["events"]:
            if e["source_id"] == sid and abs(e["t"] - issue["time_s"]) <= 0.15:
                e["t"] = issue["time_s"]
                e["frame"] = round(issue["time_s"] * sync.FPS)
    # "missing" sources are surfaced in the UI only; adding one mid-run is out of scope.


async def review_and_fix(clip, cue, envelopes, takes_bytes, out_dir, mix_result):
    """Runs QC against an already-rendered mix_result, applies one fix pass
    if there are issues (re-rendering only then), and scores again.
    Returns (cue, mix_result, qc_usage_list)."""
    review_before, usage_before = review(mix_result.qc_path, cue)
    cue["qc"]["before"] = review_before["score"]
    cue["qc"]["issues"] = review_before["issues"]
    usages = [usage_before]

    fixable = [i for i in review_before["issues"] if i["problem"] != "missing"]
    if not fixable:
        cue["qc"]["after"] = review_before["score"]
        return cue, mix_result, usages

    for issue in fixable:
        await _apply_fix(issue, cue, envelopes, takes_bytes)

    mix_result = mix.render_clip(clip, cue, envelopes, takes_bytes, out_dir)
    review_after, usage_after = review(mix_result.qc_path, cue)
    cue["qc"]["after"] = review_after["score"]
    usages.append(usage_after)
    return cue, mix_result, usages
