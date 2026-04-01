from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
]


def load_font(size: int, candidates: list[str] | tuple[str, ...] | None = None):
    for name in candidates or DEFAULT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(str(name), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def draw_panel(draw: ImageDraw.ImageDraw, box, title: str, font, small_font, *, fill, text, muted, subtitle: str | None = None, outline=(70, 74, 84), radius: int = 14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)
    x0, y0, *_ = box
    draw.text((x0 + 14, y0 + 10), title, fill=text, font=font)
    if subtitle:
        draw.text((x0 + 14, y0 + 40), subtitle, fill=muted, font=small_font)


def maybe_make_mp4_from_gif(gif_path: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    mp4_path = gif_path.with_suffix(".mp4")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(gif_path), "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(mp4_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp4_path


def maybe_make_mp4_from_frames(frames_dir: Path, mp4_path: Path, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"), "-vf", "format=yuv420p", "-movflags", "+faststart", str(mp4_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp4_path


def maybe_make_gif_from_frames(frames_dir: Path, gif_path: Path, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    palette = gif_path.with_name(gif_path.stem + "_palette.png")
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"), "-vf", "palettegen", str(palette)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"), "-i", str(palette), "-lavfi", "paletteuse", str(gif_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return gif_path


def draw_banner(img: Image.Image, title: str, subtitle: str, *, background=(18, 20, 24), text=(235, 235, 235), subtext=(180, 180, 180), top_pad: int = 64, title_size: int = 24, subtitle_size: int = 18) -> Image.Image:
    w, h = img.size
    out = Image.new("RGB", (w, h + top_pad), background)
    out.paste(img, (0, top_pad))
    draw = ImageDraw.Draw(out)
    font = load_font(title_size)
    small = load_font(subtitle_size)
    draw.text((14, 10), title, fill=text, font=font)
    draw.text((14, 10 + subtitle_size + 8), subtitle, fill=subtext, font=small)
    return out


def compose_labeled_strip(images: list[Image.Image], labels: list[str], title: str, subtitle: str, *, gap: int = 10, background=(18, 20, 24), text=(235, 235, 235), subtext=(180, 180, 180), label_bg=(0, 0, 0), top_pad: int = 74, bottom_pad: int = 22, title_size: int = 28, subtitle_size: int = 20, label_size: int = 20) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    w = sum(widths) + gap * (len(images) - 1)
    h = max(heights)
    out = Image.new("RGB", (w, h + top_pad + bottom_pad), background)
    draw = ImageDraw.Draw(out)
    font = load_font(title_size)
    small = load_font(subtitle_size)
    label_font = load_font(label_size)
    draw.text((16, 14), title, fill=text, font=font)
    draw.text((16, 44), subtitle, fill=subtext, font=small)

    x = 0
    for img, label in zip(images, labels, strict=True):
        out.paste(img, (x, top_pad))
        draw.rounded_rectangle((x + 12, top_pad + 12, x + 142, top_pad + 48), radius=10, fill=label_bg)
        draw.text((x + 24, top_pad + 18), label, fill=text, font=label_font)
        x += img.width + gap
    return out
