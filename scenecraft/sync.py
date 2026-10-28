"""Where the hits are.

A sudden stop makes a sound: a heel strike, a mug set down, a door meeting
its frame. So an impact is a peak in deceleration of the tracked box.
Continuous sources get a level that follows speed. Then each of Muse
Spark's rough times snaps to the strongest motion peak near it.
"""
import numpy as np
from scipy.signal import savgol_filter, find_peaks

FPS = 15  # frame rate of sam.mp4: SAM frame indexes are proxy frames


def motion(track, w, h):
    """Box track -> per-frame centre x, size, speed. Gaps are interpolated."""
    f = np.array([r[0] for r in track])
    b = np.array([r[1:] for r in track], dtype=float)
    frames = np.arange(f[0], f[-1] + 1)
    x1, y1, x2, y2 = (np.interp(frames, f, b[:, i]) for i in range(4))
    cx, bottom = (x1 + x2) / 2, y2              # bottom edge: where contact happens
    area = (x2 - x1) * (y2 - y1) / (w * h)      # share of the frame: a distance cue
    if len(frames) >= 7:
        cx, bottom = savgol_filter(cx, 7, 2), savgol_filter(bottom, 7, 2)
    speed = np.hypot(np.gradient(cx), np.gradient(bottom)) * FPS / h  # frame-heights per second
    return frames, cx, area, speed


def detect_impacts(track, w, h, min_gap_s=0.18):
    frames, cx, area, speed = motion(track, w, h)
    if len(frames) < 7:
        return []
    decel = np.clip(-np.gradient(speed), 0, None)
    if decel.max() == 0:
        return []
    peaks, _ = find_peaks(decel, height=0.25 * decel.max(), distance=max(1, round(min_gap_s * FPS)))
    return [{
        "t": frames[p] / FPS,
        "frame": int(frames[p]),
        "pan": float(np.clip(cx[p] / w * 2 - 1, -1, 1)),
        "gain_db": float(np.clip(10 * np.log10(area[p] / 0.02 + 1e-6), -12, 3)),
        "strength": float(decel[p] / decel.max()),
    } for p in peaks]


def snap(approx_times, detected, window_s=0.35):
    """Muse Spark says roughly when; SAM motion says exactly when."""
    events, used = [], set()
    for t in approx_times:
        near = [i for i, d in enumerate(detected) if abs(d["t"] - t) <= window_s and i not in used]
        if near:
            i = max(near, key=lambda i: detected[i]["strength"])
            used.add(i)
            events.append({**detected[i], "origin": "snapped"})
        else:
            events.append({"t": t, "frame": round(t * FPS), "pan": 0.0, "gain_db": -6.0,
                            "strength": 0.5, "origin": "muse_only"})
    events += [{**d, "origin": "sam_only"} for i, d in enumerate(detected)
               if i not in used and d["strength"] > 0.6]
    return sorted(events, key=lambda e: e["t"])


def envelope(track, w, h, n_frames):
    """Continuous sources (cloth, rolling): the level follows speed."""
    frames, _, _, speed = motion(track, w, h)
    env = np.zeros(n_frames)
    frames = frames[frames < n_frames]
    env[frames] = np.clip(speed[: len(frames)] / 0.5, 0, 1)  # 0.5 frame-heights/s = full level
    return env


def _merged_impacts(obj_tracks, w, h):
    """Impacts for a source: union of every tracked object's deceleration peaks."""
    detected = []
    for track in obj_tracks.values():
        detected.extend(detect_impacts(track, w, h))
    detected.sort(key=lambda d: d["t"])
    return detected


def _largest_track(obj_tracks):
    return max(obj_tracks.values(), key=len)


def build_cue_sheet(clip, spot, tracks_by_source):
    """Ties sync together: snapped impacts for 'impact' sources, an envelope
    for 'continuous' sources. Returns (cue_sheet dict, envelopes dict)."""
    w, h, n_frames = clip.width, clip.height, clip.n_frames
    events = []
    envelopes = {}
    sources_out = []

    for s in spot["sources"]:
        obj_tracks = tracks_by_source.get(s["id"], {})
        track_ids = sorted(obj_tracks.keys())
        sources_out.append({**s, "track_ids": track_ids, "takes": []})

        if not obj_tracks:
            continue

        if s["mode"] == "impact":
            detected = _merged_impacts(obj_tracks, w, h)
            for e in snap(s.get("approx_times_s", []), detected):
                events.append({"source_id": s["id"], **e})
        else:
            envelopes[s["id"]] = envelope(_largest_track(obj_tracks), w, h, n_frames)

    events.sort(key=lambda e: e["t"])
    cue = {
        "clip": {"id": clip.id, "duration_s": clip.duration_s, "sam_fps": FPS, "w": w, "h": h},
        "scene": spot["scene"],
        "sources": sources_out,
        "events": events,
        "mix": {},
        "qc": {"before": None, "after": None, "issues": []},
    }
    return cue, envelopes
