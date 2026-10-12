"""Redraw the TraceCMR logo files (mark + wordmark) with the wordmark set in a given font, as plain SVG paths.

The wordmark is converted to outlines so the logo looks the same everywhere, with or without the font installed.
Shaping (kerning) is done with HarfBuzz; letter spacing is applied after shaping.

    pip install uharfbuzz fonttools
    python scripts/make_wordmark.py --font InterDisplay-Medium.ttf

Writes docs/assets/tracecmr-logo.svg (dark text, for light backgrounds), tracecmr-logo-dark.svg (light text, for
dark backgrounds) and the copies the app uses in src/tracecmr/app/resources/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "docs" / "assets" / "tracecmr-icon.svg"
INK = {"dark": "#F6F5F2", "light": "#0E1318"}  # text colour for a dark / light background
RED = "#D0646F"


def mark_markup(ink: str) -> str:
    """The stacked-slices mark at 0..64 (from the icon file, without its tile and offset)."""
    s = ICON.read_text()
    g = s[s.index("<g"):s.rindex("</g>") + 4]
    g = re.sub(r'<g transform="[^"]*">', "<g>", g, count=1)
    return g.replace('fill="#F6F5F2"', f'fill="{ink}"')


def wordmark_path(font_file: Path, text: str, cap_height: float, x0: float, baseline: float,
                  tracking_em: float) -> tuple[str, float]:
    """SVG path data for `text`, cap height `cap_height`, starting at x0 on `baseline`. Returns (d, end_x)."""
    blob = hb.Blob.from_file_path(str(font_file))
    face = hb.Face(blob)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})

    tt = TTFont(str(font_file))
    glyphs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    upm = tt["head"].unitsPerEm
    cap = getattr(tt["OS/2"], "sCapHeight", 0) or 0.727 * upm
    scale = cap_height / cap
    track = tracking_em * upm

    pen = SVGPathPen(glyphs)
    x = 0.0
    for i, (info, pos) in enumerate(zip(buf.glyph_infos, buf.glyph_positions)):
        name = order[info.codepoint]
        tp = TransformPen(pen, (scale, 0, 0, -scale, x0 + (x + pos.x_offset) * scale,
                                baseline - pos.y_offset * scale))
        glyphs[name].draw(tp)
        x += pos.x_advance + (track if i < len(buf.glyph_infos) - 1 else 0)
    d = pen.getCommands()
    d = re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.2f}", d)
    return d, x0 + x * scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", type=Path, required=True)
    ap.add_argument("--text", default="TraceCMR")
    ap.add_argument("--cap-height", type=float, default=24.0)
    ap.add_argument("--gap", type=float, default=16.0, help="space between the mark (ends at x=58) and the text")
    ap.add_argument("--tracking", type=float, default=-0.015, help="letter spacing in em (negative = tighter)")
    args = ap.parse_args()

    baseline = 32 + args.cap_height / 2  # text centred on the mark's height
    d, end = wordmark_path(args.font, args.text, args.cap_height, 58 + args.gap, baseline, args.tracking)
    width = round(end + 3, 1)
    for kind, name in (("dark", "tracecmr-logo-dark.svg"), ("light", "tracecmr-logo.svg")):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 64" role="img" '
               f'aria-label="TraceCMR">{mark_markup(INK[kind])}<path d="{d}" fill="{INK[kind]}"/></svg>')
        for folder in (ROOT / "docs" / "assets", ROOT / "src" / "tracecmr" / "app" / "resources"):
            (folder / name).write_text(svg)
        print(f"wrote {name} ({width} x 64)")


if __name__ == "__main__":
    main()
