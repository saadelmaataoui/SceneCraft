"""Meta Model API: sam-3.1 streams a box and mask per object per frame.

One request per concept: Meta's docs say lists like "shoe, door" return zero
matches, so each sound source is tracked with its own request. Timing only
needs the boxes, not the decoded masks.
"""
import re

from scenecraft.spotting import get_client

FRAME = re.compile(r"<(\d+)f>")
BOX = re.compile(r"(\d+)<\|box;([^|]+)\|>")  # boxes are all the timing needs


def track(file_id, concept, on_line=None):
    """Returns {object_id: [(frame, x1, y1, x2, y2), ...]} in sam.mp4 pixels."""
    tracks = {}

    def handle(line):
        m = FRAME.match(line)
        if not m:
            return
        if on_line:
            on_line(concept, line)  # a UI can decode masks with @meta-sam/parser
        frame = int(m.group(1))
        for obj, kv in BOX.findall(line):
            d = dict(p.split("=", 1) for p in kv.split(";"))
            tracks.setdefault(int(obj), []).append(
                (frame, float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])))

    with get_client().responses.stream(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},  # one noun phrase, never a list
            {"type": "input_video", "file_id": file_id},
        ]}],
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                for line in event.delta.splitlines():  # one delta per frame line
                    handle(line.strip())

    for t in tracks.values():
        t.sort()  # frame indexes can skip and arrive out of order
    return tracks
