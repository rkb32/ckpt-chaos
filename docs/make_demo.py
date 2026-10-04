"""Render docs/demo.gif from real recorded output.

docs/demo_naive.txt and docs/demo_atomic.txt are exactly what `ckpt-chaos run` printed (see the commands
below), docs/demo_*.exit hold the exit codes. Only the waiting time is shortened and the command is typed
out; the text of every result line is the recorded one.

    pip install pillow
    python docs/make_demo.py
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BG, FG, DIM = (13, 17, 23), (201, 209, 217), (139, 148, 158)
GREEN, RED, YELLOW, PROMPT = (63, 185, 80), (255, 123, 114), (210, 153, 34), (126, 231, 135)
PAD, FONT_SIZE = 18, 15
FONT_PATHS = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/System/Library/Fonts/Menlo.ttc"]
VERDICTS = {"PASS": GREEN, "HARD_FAIL": RED, "SILENT_DIVERGENCE": YELLOW, "LOST_WORK": YELLOW, "CONTROL_FAIL": RED}

CMD = ['ckpt-chaos run --step-regex "resumed from step (\\d+)" --result-file "{out}/result.json" \\',
       "    -- python examples/naive_json.py --out {out}"]
SCENES = [
    ("naive", "# the script overwrites state.json in place", CMD),
    ("atomic", "# the same script, writing a temp file and renaming it", [CMD[0], CMD[1] + " --atomic"]),
]


def load_font() -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, FONT_SIZE)
    raise SystemExit("no monospace font found; add one to FONT_PATHS")


FONT = load_font()
CHAR_W = FONT.getlength("M")
LINE_H = FONT_SIZE + 6


def scene_lines(name: str, comment: str, cmd: list[str]) -> list[tuple[str, tuple]]:
    out = (HERE / f"demo_{name}.txt").read_text(encoding="utf-8").rstrip().splitlines()
    code = (HERE / f"demo_{name}.exit").read_text().split(":")[-1].strip()
    lines = [(comment, DIM), ("$ " + cmd[0], FG), (cmd[1] if cmd[1].startswith(" ") else "$ " + cmd[1], FG)]
    lines += [(l, DIM if re.match(r"^\s*(\d/3|\d+ file events)", l) else FG) for l in out]
    lines += [("", FG), ("$ echo $?", FG), (code, GREEN if code == "0" else RED)]
    return lines


WORDS = "|".join(VERDICTS)


def verdict_spans(text: str) -> list[tuple[int, str]]:
    """(column, verdict) for a result row ('#3 ... HARD_FAIL') or the summary line ('HARD_FAIL: 4  PASS: 8')."""
    if text.startswith(("#", "no crash")):
        m = re.search(rf"({WORDS})\s*$", text)
        return [(m.start(1), m.group(1))] if m else []
    if re.match(rf"^({WORDS}): \d+", text):
        return [(m.start(), m.group(1)) for m in re.finditer(rf"({WORDS})(?=:)", text)]
    return []


def render(lines: list[tuple[str, tuple]], width: int, height: int, cursor: str = "") -> Image.Image:
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    for i, (text, color) in enumerate(lines):
        y = PAD + i * LINE_H
        if text.startswith("$ "):
            d.text((PAD, y), "$", font=FONT, fill=PROMPT)
            d.text((PAD + 2 * CHAR_W, y), text[2:], font=FONT, fill=color)
            continue
        d.text((PAD, y), text, font=FONT, fill=color)
        for start, word in verdict_spans(text):  # colour only the verdict words
            d.text((PAD + start * CHAR_W, y), word, font=FONT, fill=VERDICTS[word])
    if cursor:
        d.rectangle([PAD + cursor_x(lines) , PAD + (len(lines) - 1) * LINE_H + 2, PAD + cursor_x(lines) + CHAR_W,
                     PAD + len(lines) * LINE_H - 2], fill=FG)
    return img


def cursor_x(lines) -> float:
    return len(lines[-1][0]) * CHAR_W if lines else 0


def build() -> None:
    scenes = [scene_lines(*s) for s in SCENES]
    cols = max(len(t) for sc in scenes for t, _ in sc)
    rows = max(len(sc) for sc in scenes)
    width, height = int(PAD * 2 + cols * CHAR_W) + 4, PAD * 2 + rows * LINE_H
    frames: list[Image.Image] = []
    times: list[int] = []

    def add(lines, ms, cursor=""):
        frames.append(render(lines, width, height, cursor))
        times.append(ms)

    for sc in scenes:
        shown: list[tuple[str, tuple]] = [sc[0]]
        add(shown, 700)
        for k in (1, 2):  # type the two command lines, a few characters per frame
            text, color = sc[k]
            body = text[2:] if text.startswith("$ ") else text
            prefix = "$ " if text.startswith("$ ") else ""
            for n in range(0, len(body) + 1, 5):
                add(shown + [(prefix + body[:n], color)], 35, cursor="x")
            shown.append(sc[k])
        add(shown, 350)
        i = 3
        while i < len(sc):  # progress lines, then the table row by row
            text, color = sc[i]
            slow = text.strip().startswith(("1/3", "2/3", "3/3")) or "file events" in text
            shown.append(sc[i])
            add(shown, 450 if slow else (700 if text.startswith("crash point") else 110))
            if text.startswith("2/3"):
                add(shown, 900)  # the real run spends its time here
            i += 1
        add(shown, 3200)
    frames[0].save(HERE / "demo.gif", save_all=True, append_images=frames[1:], duration=times, loop=0, optimize=True,
                   disposal=1)
    print(f"{len(frames)} frames, {width}x{height}px, {(HERE / 'demo.gif').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
