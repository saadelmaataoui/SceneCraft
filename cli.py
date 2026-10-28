"""Test harness: python cli.py clips/gravel.mov -> prints progress, writes out.mp4.

Build and debug the pipeline here first; app.py and deploy.py are thin
layers on top of the same scenecraft.pipeline.run.
"""
import argparse
import asyncio
import json
import sys

from scenecraft import pipeline


async def main(take_path: str, jobs_dir: str):
    async for step in pipeline.run(take_path, jobs_dir=jobs_dir):
        if "status" in step:
            cost = f" (~${step['cost']:.3f})" if "cost" in step else ""
            print(f"[{step['status']}]{cost}", file=sys.stderr)
        if step.get("done"):
            print(f"out.mp4: {step['video']}")
            print(f"cue sheet: {json.dumps(step['cue'], default=str)[:2000]}...")
            return step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", help="Path to a silent clip, e.g. clips/gravel.mov")
    parser.add_argument("--jobs-dir", default=".jobs")
    args = parser.parse_args()
    asyncio.run(main(args.clip, args.jobs_dir))
