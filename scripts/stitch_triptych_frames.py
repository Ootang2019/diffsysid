#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from diffsysid.render import compose_labeled_strip, maybe_make_gif_from_frames, maybe_make_mp4_from_frames

BG = (18, 20, 24)
TEXT = (235, 235, 235)
SUBTEXT = (180, 180, 180)
LABEL_BG = (0, 0, 0)



def main():
    p = argparse.ArgumentParser(description="Stitch multiple per-view frame folders into one triptych video/gif.")
    p.add_argument("--frames-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--subtitle", type=str, default="")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--gap", type=int, default=10)
    p.add_argument("--stem", type=str, default="triptych")
    args = p.parse_args()

    if len(args.frames_dirs) != len(args.labels):
        raise ValueError("frames-dirs and labels must have the same length")

    frame_lists = []
    for d in args.frames_dirs:
        frames = sorted(d.glob("frame_*.png"))
        if not frames:
            raise FileNotFoundError(f"No frames found in {d}")
        frame_lists.append(frames)

    count = min(len(frames) for frames in frame_lists)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stitched_dir = args.output_dir / "frames"
    stitched_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        images = [Image.open(frame_lists[j][idx]).convert("RGB") for j in range(len(frame_lists))]
        frame = compose_labeled_strip(images, args.labels, args.title, args.subtitle, gap=args.gap, background=BG, text=TEXT, subtext=SUBTEXT, label_bg=LABEL_BG)
        frame.save(stitched_dir / f"frame_{idx:04d}.png")
        for img in images:
            img.close()

    mp4_path = maybe_make_mp4_from_frames(stitched_dir, args.output_dir / f"{args.stem}.mp4", args.fps)
    gif_path = maybe_make_gif_from_frames(stitched_dir, args.output_dir / f"{args.stem}.gif", args.fps)
    manifest = {
        "fps": args.fps,
        "title": args.title,
        "subtitle": args.subtitle,
        "source_frames_dirs": [str(x) for x in args.frames_dirs],
        "labels": args.labels,
        "stitched_frames": count,
        "artifacts": {
            "frames_dir": str(stitched_dir),
            "mp4": None if mp4_path is None else str(mp4_path),
            "gif": None if gif_path is None else str(gif_path),
        },
    }
    (args.output_dir / f"{args.stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
