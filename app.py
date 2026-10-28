"""SceneCraft — Gradio frontend.

Drop a silent clip, or click one of the three examples. ~50 lines of UI,
per the hackathon plan's decision to use Gradio instead of a custom
split-screen UI: a drag-and-drop upload box, a progress bar, and a video
player, with zero custom video-encoding work.
"""
import html
import os

import gradio as gr

from scenecraft import pipeline

JOBS_DIR = os.path.abspath(".jobs")
EXAMPLES = [p for p in [
    "clips/gravel.mp4", "clips/door_slam.mp4", "clips/keys_drop.mp4",
] if os.path.exists(p)]

ORIGIN_COLOR = {"snapped": "#35e28a", "sam_only": "#35e2ff", "muse_only": "#ff9f45"}
CATEGORY_ORDER = ["feet", "moves", "specifics"]


def timeline_html(cue):
    """A static, non-interactive read of where every hit landed — a
    lightweight substitute for the old custom-UI timeline lanes."""
    if not cue or not cue.get("events"):
        return ""
    duration = max(cue["clip"]["duration_s"], 0.1)
    cat_by_source = {s["id"]: s.get("category", "specifics") for s in cue["sources"]}
    rows = []
    for cat in CATEGORY_ORDER:
        ticks = []
        for e in cue["events"]:
            if cat_by_source.get(e["source_id"]) != cat:
                continue
            pct = min(99, max(0, e["t"] / duration * 100))
            color = ORIGIN_COLOR.get(e.get("origin"), "#999")
            title = f"{e['source_id']} @ {e['t']:.2f}s ({e.get('origin')})"
            ticks.append(
                f'<span title="{html.escape(title)}" style="position:absolute;left:{pct:.1f}%;'
                f'top:2px;bottom:2px;width:3px;border-radius:2px;background:{color}"></span>'
            )
        rows.append(
            f'<div style="display:grid;grid-template-columns:5.5rem 1fr;gap:8px;align-items:center;'
            f'font:11px monospace;color:#9BA2AF;text-transform:uppercase;margin:4px 0">'
            f'<span>{cat}</span>'
            f'<div style="position:relative;height:16px;background:#1A1E26;border-radius:4px">{"".join(ticks)}</div>'
            f'</div>'
        )
    legend = " &nbsp; ".join(
        f'<span style="color:{c}">&#9632;</span> {name}'
        for name, c in [("snapped", ORIGIN_COLOR["snapped"]),
                        ("sam only", ORIGIN_COLOR["sam_only"]),
                        ("muse only", ORIGIN_COLOR["muse_only"])]
    )
    return (f'<div style="font-size:11px;color:#9BA2AF;margin-bottom:4px">{legend}</div>'
            + "".join(rows))


def stems_html(stem_paths):
    """One <audio> player per stem — Gradio has no multi-file gr.Audio, and
    the stem count varies per clip, so plain HTML is the simplest fit."""
    if not stem_paths:
        return ""
    players = []
    for source_id, path in stem_paths.items():
        abspath = os.path.abspath(path)
        players.append(
            f'<div style="margin:6px 0"><div style="font:12px monospace;color:#9BA2AF;'
            f'margin-bottom:2px">{html.escape(source_id)}</div>'
            f'<audio controls style="width:100%" src="/file={abspath}"></audio></div>'
        )
    return "".join(players)


def qc_rows(issues):
    return [[i["source_id"], round(i["time_s"], 2), i["problem"]] for i in (issues or [])]


async def on_run(video_path, progress=gr.Progress()):
    if not video_path:
        yield "Drop a clip first.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    status = video = timeline = cue = stems = rows = gr.update()
    async for step in pipeline.run(video_path, jobs_dir=JOBS_DIR):
        if "status" in step:
            cost_bit = f" · ~${step['cost']:.3f}" if "cost" in step else ""
            status = step["status"] + cost_bit
        if "video" in step:
            video = step["video"]
        if "cue" in step:
            cue = step["cue"]
            timeline = timeline_html(step["cue"])
        if "stem_paths" in step:
            stems = stems_html(step["stem_paths"])
        if "qc_issues" in step:
            rows = qc_rows(step["qc_issues"])
        progress(1.0 if step.get("done") else 0.5, desc=status if isinstance(status, str) else "Working...")
        yield status, video, timeline, cue, stems, rows


with gr.Blocks(title="SceneCraft") as demo:
    gr.Markdown(
        "# SceneCraft\n"
        "Drop a silent clip. Muse Spark spots what should make sound, SAM 3.1 tracks it "
        "frame by frame, AudioGen records takes that Audiobox Aesthetics ranks, and the mix "
        "is synced to the motion — then Muse Spark checks its own work."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Video(label="Silent clip (max 15s)", sources=["upload"])
            if EXAMPLES:
                gr.Examples(examples=EXAMPLES, inputs=inp,
                            label="No clip on hand? Try one of these.")
            run_btn = gr.Button("Add sound", variant="primary")
            status_md = gr.Markdown()
        with gr.Column():
            out_video = gr.Video(label="SceneCraft mix")
            timeline_out = gr.HTML(label="Timeline")
            stems_out = gr.HTML(label="Stems")
            cue_json = gr.JSON(label="Cue sheet", open=False)
            qc_table = gr.Dataframe(label="QC issues", headers=["source", "time_s", "problem"])

    run_btn.click(on_run, inputs=inp,
                  outputs=[status_md, out_video, timeline_out, cue_json, stems_out, qc_table])

if __name__ == "__main__":
    os.makedirs(JOBS_DIR, exist_ok=True)
    demo.queue().launch(allowed_paths=[JOBS_DIR])
