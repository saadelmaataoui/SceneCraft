"""Stage order, concurrency, and progress — as an async generator.

Each yield is a dict of whatever changed; keys not present mean "no change
yet" for that output. app.py converts these dicts into Gradio component
updates. Caching is coarse (whole-run, keyed by the uploaded video's
sha256) so clicking a precomputed example is near-instant.
"""
import asyncio
import os
import time

from scenecraft import cache, ingest, mix, qc, sourcing, spotting, sync, tracking

# Meta Model API list prices (see the build plan's #budget section).
PRICE_IN_PER_TOKEN = 1.25e-6
PRICE_OUT_PER_TOKEN = 4.25e-6
PRICE_PER_SAM_FRAME = 0.20 / 1000
SAM_MIN_FRAMES = 50


def _usage_cost(usage):
    if not usage:
        return 0.0
    pt = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    return pt * PRICE_IN_PER_TOKEN + ct * PRICE_OUT_PER_TOKEN


def _sam_cost(n_frames, n_concepts):
    billed = max(SAM_MIN_FRAMES, n_frames) * n_concepts
    return billed * PRICE_PER_SAM_FRAME


async def run(take_path: str, jobs_dir: str = ".jobs"):
    """Yields dicts with any of: status, cost, video, cue, stem_paths, qc_issues, done.
    Modal's GPU time (AudioGen + Audiobox) isn't metered client-side; see #budget
    for the per-clip estimate — the running cost total here covers Meta API usage only.
    """
    t0 = time.monotonic()
    video_hash = cache.video_hash(take_path)
    cached = cache.get(video_hash, "final")
    if cached is not None:
        yield {"status": "Loaded from cache — this clip has run before.", **cached, "done": True}
        return

    out_dir = os.path.join(jobs_dir, video_hash)
    cost = 0.0

    clip = await asyncio.to_thread(ingest.prepare, take_path, out_dir)
    yield {"status": f"Ingest done ({clip.duration_s:.1f}s, {clip.n_frames} frames)"}

    spot, spot_usage = await asyncio.to_thread(spotting.spot, clip.muse_path)
    cost += _usage_cost(spot_usage)
    n_sources = len(spot["sources"])
    yield {"status": f"Spotting done — {n_sources} sound source(s) found.", "cost": cost}

    sam_file = await asyncio.to_thread(spotting.upload, clip.sam_path)

    async def _track(s):
        return s["id"], await asyncio.to_thread(tracking.track, sam_file, s["sam_concept"])

    async def _source(s):
        return s["id"], await sourcing.source(s["sound_prompt"], s["mode"])

    (tracks_pairs, takes_pairs), ambience_takes = await asyncio.gather(
        asyncio.gather(
            asyncio.gather(*[_track(s) for s in spot["sources"]]),
            asyncio.gather(*[_source(s) for s in spot["sources"]]),
        ),
        sourcing.source(spot["scene"]["ambience_prompt"], "ambience", keep=1),
    )
    tracks_by_source = dict(tracks_pairs)
    takes_bytes = dict(takes_pairs)
    takes_bytes["ambience"] = ambience_takes
    cost += _sam_cost(clip.n_frames, n_sources)
    yield {"status": "Tracking + sourcing done.", "cost": cost}

    cue, envelopes = sync.build_cue_sheet(clip, spot, tracks_by_source)
    yield {"status": "Sync done — cue sheet built.", "cue": cue}

    mix_result = await asyncio.to_thread(mix.render_clip, clip, cue, envelopes, takes_bytes, out_dir)
    yield {"status": "Mixed.", "video": mix_result.out_path, "stem_paths": mix_result.stem_paths, "cue": cue}

    cue, mix_result, qc_usages = await qc.review_and_fix(
        clip, cue, envelopes, takes_bytes, out_dir, mix_result)
    cost += sum(_usage_cost(u) for u in qc_usages)
    before, after = cue["qc"]["before"], cue["qc"]["after"]
    elapsed = time.monotonic() - t0
    status = f"QC: {before} -> {after}. Done in {elapsed:.0f}s, ~${cost:.3f} in Meta API usage."

    payload = {
        "status": status, "cost": cost, "video": mix_result.out_path,
        "cue": cue, "stem_paths": mix_result.stem_paths, "qc_issues": cue["qc"]["issues"],
    }
    cache.put(video_hash, "final", {k: v for k, v in payload.items() if k != "status"})
    yield {**payload, "done": True}
