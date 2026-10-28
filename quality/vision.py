"""Image and video inputs: pictures and clips drawn here (PIL, ffmpeg), sent as data URLs, and the replies checked for
what they show. Each case passes when every group of its keywords has one member in the reply (case-insensitive).
Videos need ffmpeg on this host (their cases are skipped without it)."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
import time
import urllib.error
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, strip_think


def _png(im) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _font(size: int):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _images() -> list[tuple[str, list[str], str, list[tuple[str, ...]]]]:
    """(name, image data URLs, question, keyword groups)."""
    from PIL import Image, ImageDraw
    out = []
    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((60, 80, 260, 280), fill=(220, 30, 30))
    d.rectangle((360, 120, 560, 320), fill=(30, 60, 220))
    d.text((150, 380), "GLM 42", fill=(0, 0, 0), font=_font(48))
    out.append(("shapes", [_png(im)], "Describe the shapes and their colours, and quote any text in the image.",
                [("circle",), ("red",), ("square", "rectangle"), ("blue",), ("42",)]))
    im = Image.new("RGB", (1100, 260), (245, 245, 235))
    ImageDraw.Draw(im).text((40, 90), "TensorFold on DGX Spark", fill=(10, 10, 60), font=_font(64))
    out.append(("text", [_png(im)], "What does the text in this image say? Quote it exactly.",
                [("tensorfold",), ("spark",)]))
    im = Image.new("RGB", (720, 360), "white")
    d = ImageDraw.Draw(im)
    for x in (60, 290, 520):
        d.polygon([(x + 70, 60), (x, 280), (x + 140, 280)], fill=(20, 150, 40))
    out.append(("count", [_png(im)], "How many triangles are in this image, and what colour are they? Answer briefly.",
                [("three", "3"), ("green",)]))
    a = Image.new("RGB", (320, 320), (230, 20, 20))
    b = Image.new("RGB", (320, 320), (250, 210, 0))
    out.append(("two images", [_png(a), _png(b)], "These are two solid-colour images. Name the colour of the first and "
                "then of the second.", [("red",), ("yellow",)]))
    return out


def _clip(frames, fps: int) -> str | None:
    """An mp4 data URL of ``frames`` (PIL images) at ``fps`` (None without ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(Path(tmp) / f"f{i:04d}.png")
        mp4 = Path(tmp) / "clip.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(Path(tmp) / "f%04d.png"),
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(mp4)], check=True)
        return "data:video/mp4;base64," + base64.b64encode(mp4.read_bytes()).decode()


def _videos() -> list[tuple[str, str | None, str, list[tuple[str, ...]]]]:
    from PIL import Image, ImageDraw
    out = []
    frames = []
    for i in range(40):  # 4 s at 10 fps: an orange ball crossing from left to right
        f = Image.new("RGB", (448, 336), (200, 230, 255))
        x = 20 + i * 9
        ImageDraw.Draw(f).ellipse((x, 130, x + 70, 200), fill=(255, 140, 0))
        frames.append(f)
    out.append(("video motion", _clip(frames, 10), "In this video, which direction does the ball move: left to right or "
                "right to left? Answer in a few words.", [("left to right", "to the right", "rightward")]))
    # a little seeded noise: perfectly flat frames are a degenerate input the tower reads unreliably (as any bf16 run)
    import random
    rng = random.Random(5)
    frames = []
    for i in range(40):
        base = (20, 40, 200) if i < 20 else (20, 170, 40)
        f = Image.new("RGB", (448, 336), base)
        px = f.load()
        for _ in range(4000):
            x, y = rng.randrange(448), rng.randrange(336)
            px[x, y] = tuple(min(255, max(0, c + rng.randint(-40, 40))) for c in base)
        frames.append(f)
    out.append(("video colours", _clip(frames, 10), "This video's background changes colour once. What colour is it at "
                "the start, and what colour at the end?", [("blue",), ("green",)]))
    return out


def run(client: Client) -> Result:
    t0 = time.time()
    passed, total, notes, details = 0, 0, [], {}
    cases = [(n, [{"type": "image_url", "image_url": {"url": u}} for u in urls], q, kw) for n, urls, q, kw in _images()]
    for n, url, q, kw in _videos():
        if url is None:
            notes.append(f"{n}: skipped (no ffmpeg on this host)")
            continue
        cases.append((n, [{"type": "video_url", "video_url": {"url": url}}], q, kw))
    for name, parts, question, groups in cases:
        total += 1
        try:
            r = client.chat([{"role": "user", "content": parts + [{"type": "text", "text": question}]}], max_tokens=160)
            text = strip_think(r["choices"][0]["message"]["content"] or "")
        except Exception as e:  # a refusal or a transport error: the case fails with the reason
            text = f"<error: {e}>"
        low = text.lower()
        miss = [g for g in groups if not any(k in low for k in g)]
        ok = not miss
        passed += ok
        details[name] = {"reply": text, "missing": [list(g) for g in miss]}
        if not ok:
            notes.append(f"{name}: missing {['/'.join(g) for g in miss]} in {text[:160]!r}")
    return Result("vision", passed / max(1, total), total, passed, time.time() - t0, notes, details)


def concurrent(client: Client, streams: int = 16) -> Result:
    """The image and video cases ``streams`` at once (each case repeated in turn): every reply equal to the same request
    alone, and its keywords there. A case the server refuses alone (HTTP 400, e.g. video on a server without it) has no
    reply to compare and is left out of the batch with a note; the vision check fails it."""
    t0 = time.time()
    cases = [(n, [{"type": "image_url", "image_url": {"url": u}} for u in urls], q, kw) for n, urls, q, kw in _images()]
    cases += [(n, [{"type": "video_url", "video_url": {"url": url}}], q, kw) for n, url, q, kw in _videos() if url is not None]

    def ask(case) -> str:
        name, parts, question, _ = case
        r = client.chat([{"role": "user", "content": parts + [{"type": "text", "text": question}]}], max_tokens=160)
        return strip_think(r["choices"][0]["message"]["content"] or "")

    alone, refused = {}, []
    for c in cases:
        try:
            alone[c[0]] = ask(c)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            refused.append(c[0])
    cases = [c for c in cases if c[0] in alone]
    if not cases:
        raise RuntimeError(f"every case refused alone: {', '.join(refused)}")
    batch = [cases[i % len(cases)] for i in range(streams)]
    with ThreadPoolExecutor(streams) as ex:
        together = list(ex.map(ask, batch))
    passed, notes = 0, []
    for case, text in zip(batch, together):
        miss = [g for g in case[3] if not any(k in text.lower() for k in g)]
        same = text == alone[case[0]]
        passed += same and not miss
        if not same:
            notes.append(f"{case[0]}: differs from alone: {text[:100]!r} vs {alone[case[0]][:100]!r}")
        elif miss:
            notes.append(f"{case[0]}: missing {['/'.join(g) for g in miss]}")
    notes.insert(0, f"{streams} image/video requests together vs alone")
    if refused:
        notes.insert(1, f"refused alone (HTTP 400), not run together: {', '.join(refused)}")
    return Result("vision-concurrent", passed / streams, streams, passed, time.time() - t0, notes, {"alone": alone, "together": together})
