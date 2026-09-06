"""Figure 1 -- three system behaviors at one mistake, drawn in staff notation.

Verified exact case (Polytune x MAESTRO, piece 01-03_R1_2014_..--4, t=178.95 s;
see verify_two_scenario.py, claim holds): the reference contains ONE inserted
note (extra @ MIDI 68 = G-sharp 4); the system outputs TWO events, extra @ 68
plus missed @ 69 (A4) -- its encoding of a wrong note.

Layout: ONE two-staff system (score above, played notes below) whose three
consecutive measures are the three scenarios (a), (b), (c), read left to
right like music. Every measure repeats the same passage: the score has a rest
where the player adds G-sharp 4, so the inserted note is visible by
comparison. System claims are drawn on top:

(a) Silent: the system makes no claim.
(b) Counterfactual: the extra claim (blue box) sits on the played G-sharp 4,
    a correct insertion claim; the missed A4 claim (hollow orange note on the
    score staff, i.e. a note the system says the score contained and the
    player omitted) sits two beats later, an unrelated false deletion claim.
(c) As measured: the missed A4 claim is co-located with the extra claim
    (bracket), the system's "played 68 instead of 69" encoding: found but
    misnamed (raw HM=1).
Below the system, one report box: (b) and (c) publish the IDENTICAL per-class
report (extra: 1 TP; missed: 1 FP).

The passage itself (E4, B4 context notes, 4/4) is schematic; the mistake's
pitches, classes and co-location are the measured case. The caption states it.

Why notation rather than a piano roll: both systems the letter evaluates open
with staff-notation figures (Polytune Fig. 1; LadderSym Fig. 1), so the
reader meets the mistake as a musician sees it; the piano-roll version read as
a bar chart (author's review, 2026-09-05). Why one system rather than a 2x2
grid of panels: one clef and one label pair serve all three scenarios, so the
staves are larger and the figure a third shorter.

IEEE graphics rules honoured (proposal/review-evidence/dossier-parts/21-*):
FG-016/017 in-figure text in Times New Roman, embedded as a TrueType subset;
FG-019/020 all type 8 pt at placed size; FG-021 bare (a)/(b)/(c) centered
below each measure, descriptive wording only in the LaTeX caption; FG-026 no
transparency; FG-028 greyscale-legible (claims are boxes + hollow vs filled
heads, not colour alone); FG-031 no baked-in caption or frame around the
figure; music glyphs are vector outlines (no font). The gate (verify_shipped
[8]) rejects Type 3 fonts in this PDF.

Pipeline (pure Python, no system installs): MusicXML (built here) -> verovio
(engraving to SVG) -> svglib/reportlab (SVG to PDF) -> PyMuPDF (place the
system, draw boxes, labels, bracket, report box; strip unused font
resources).

Run:  pip install verovio svglib reportlab pymupdf
      python fig1_two_scenario.py
"""
from __future__ import annotations

import io
import os
import re
from typing import Dict, List, Tuple

import pymupdf as fitz
import verovio
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig1_two_scenario.pdf")

# palette shared with the letter's earlier figure versions
C_EXTRA = "#3b6fb0"      # system extra (insertion claim): blue box
C_MISS = "#c46a10"       # system missed (deletion claim): hollow orange note + box
C_GRAY = "#555555"
C_TEXT = "#000000"

# IEEE single column = 3.5 in = 252 pt
COL_W = 252.0
LABEL_MARGIN = 26.0      # left strip for "score"/"played"
RIGHT_PAD = 2.0
FONT_PT = 8.0
TNR_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
]

# scenario per measure: beat of the claimed-missing A4 on the score staff
# (None: no claim; 4: two beats after the inserted note; 2: co-located)
MEASURES = [None, 4, 2]


def hex_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


# ------------------------------------------------------------------ MusicXML
def _note(step: str, octave: int, alter: int | None = None, color: str | None = None,
          hollow: bool = False) -> str:
    alt = f"<alter>{alter}</alter>" if alter else ""
    acc = "<accidental>sharp</accidental>" if alter == 1 else ""
    col = f' color="{color}"' if color else ""
    head = ""
    stem = "<stem>up</stem>"
    if color or hollow:
        filled = ' filled="no"' if hollow else ""
        head = f'<notehead{filled}{col}>normal</notehead>'
        stem = f'<stem{col}>up</stem>'
    return (f"<note{col}><pitch><step>{step}</step>{alt}<octave>{octave}</octave></pitch>"
            f"<duration>1</duration><type>quarter</type>{acc}{stem}{head}</note>")


def _rest() -> str:
    return "<note><rest/><duration>1</duration><type>quarter</type></note>"


def musicxml() -> str:
    """Two parts, three 4/4 measures of quarter notes, no time signature drawn.

    Score staff (P1), every measure: E4, rest, B4, rest, with the system's
    claimed-missing A4 as a hollow orange note at MEASURES[m] (2 = co-located
    with the inserted note, 4 = two beats later, None = no claim).
    Played staff (P2), every measure: E4, G#4 (MIDI 68, the inserted note),
    B4, rest.
    """
    attrs = ("<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
             "<clef><sign>G</sign><line>2</line></clef></attributes>")

    def score_measure(i: int, missed_beat: int | None) -> str:
        beats = [_note("E", 4), _rest(), _note("B", 4), _rest()]
        if missed_beat is not None:
            beats[missed_beat - 1] = _note("A", 4, color=C_MISS, hollow=True)
        return f'<measure number="{i + 1}">{attrs if i == 0 else ""}{"".join(beats)}</measure>'

    def played_measure(i: int) -> str:
        beats = [_note("E", 4), _note("G", 4, alter=1), _note("B", 4), _rest()]
        return f'<measure number="{i + 1}">{attrs if i == 0 else ""}{"".join(beats)}</measure>'

    p1 = "".join(score_measure(i, mb) for i, mb in enumerate(MEASURES))
    p2 = "".join(played_measure(i) for i in range(len(MEASURES)))
    return ('<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list>'
            '<score-part id="P1"><part-name></part-name></score-part>'
            '<score-part id="P2"><part-name></part-name></score-part></part-list>'
            f'<part id="P1">{p1}</part><part id="P2">{p2}</part></score-partwise>')


# ------------------------------------------------------------------ engraving
def engrave(xml: str) -> str:
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": 2600, "pageHeight": 600, "scale": 50, "breaks": "none",
        "adjustPageHeight": True, "adjustPageWidth": True,
        "header": "none", "footer": "none", "mnumInterval": 0,
        "spacingStaff": 5, "spacingSystem": 0,
        "pageMarginLeft": 10, "pageMarginRight": 10,
        "pageMarginTop": 10, "pageMarginBottom": 10,
    })
    if not tk.loadData(xml):
        raise RuntimeError("verovio failed to load the MusicXML")
    if tk.getPageCount() != 1:
        raise RuntimeError("the three measures did not fit one system")
    svg = tk.renderToSVG(1)
    # part-name placeholders come out as empty <text>; drop every text node so
    # the notation PDF carries no font at all (labels are set later in TNR)
    return re.sub(r"<text.*?</text>", "", svg, flags=re.S)


class Geometry:
    """Staff lines, barlines and noteheads, in verovio viewBox units."""

    def __init__(self, svg: str) -> None:
        # verovio emits one <g class="staff"> per measure and staff; group the
        # blocks by their staff-line heights (identical across measures) and
        # keep noteheads in left-to-right order
        starts = [s.start() for s in re.finditer(r'<g id="[^"]*" class="staff">', svg)]
        starts.append(len(svg))
        by_lines: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        for a, b in zip(starts, starts[1:]):
            blk = svg[a:b]
            horiz = [(int(x2) - int(x1), int(y1)) for x1, y1, x2, y2 in re.findall(
                r'<path d="M(\d+) (\d+) L(\d+) (\d+)"[^>]*/>', blk) if y1 == y2 and x2 > x1]
            # the five longest horizontal strokes are the staff lines (ledger
            # lines are shorter)
            ys = tuple(sorted(y for _, y in sorted(horiz, reverse=True)[:5]))
            heads = [(int(x), int(y)) for x, y in re.findall(
                r'<g class="notehead"[^>]*>\s*<use [^>]*translate\((-?\d+), (-?\d+)\)', blk)]
            by_lines.setdefault(ys, []).extend(heads)
        self.staves: List[Dict] = [
            {"lines": list(ys), "heads": sorted(heads)}
            for ys, heads in sorted(by_lines.items())]
        if len(self.staves) != 2:
            raise RuntimeError(f"expected 2 staves, parsed {len(self.staves)}")
        # barlines: vertical paths at least one staff tall (stems are 3.5
        # spaces, barlines 4 or, for the system line, more)
        staff_h = self.staves[0]["lines"][-1] - self.staves[0]["lines"][0]
        vert = [(int(y2) - int(y1), int(x1)) for x1, y1, x2, y2 in re.findall(
            r'<path d="M(\d+) (\d+) L(\d+) (\d+)"[^>]*/>', svg) if x1 == x2 and int(y2) > int(y1)]
        xs = sorted({x for h, x in vert if h >= 0.95 * staff_h})
        # thick+thin final barline: keep distinct positions further than a
        # staff space apart
        self.barlines: List[int] = []
        for x in xs:
            if not self.barlines or x - self.barlines[-1] > self.space:
                self.barlines.append(x)
        if len(self.barlines) != len(MEASURES) + 1:
            raise RuntimeError(f"expected {len(MEASURES) + 1} barlines, found {self.barlines}")

    @property
    def space(self) -> int:
        ln = self.staves[0]["lines"]
        return ln[1] - ln[0]


def svg_to_pdf(svg: str) -> fitz.Document:
    tmp = io.BytesIO()
    renderPDF.drawToFile(svg2rlg(io.StringIO(svg)), tmp)
    doc = fitz.open("pdf", tmp.getvalue())
    pg = doc[0]
    # svglib selects a default Type1 font (an empty BT..Tf..ET block) even when
    # nothing is drawn with it; an unembedded font resource would fail IEEE's
    # font check, so drop text blocks that show no glyphs, then the resource.
    if not pg.get_text().strip():
        pg.clean_contents()
        xref = pg.get_contents()[0]
        stream = doc.xref_stream(xref)
        stream = re.sub(rb"BT\b.*?\bET\b",
                        lambda m: m.group(0) if re.search(rb"T[jJ]\b", m.group(0)) else b"",
                        stream, flags=re.S)
        doc.update_stream(xref, stream)
        doc.xref_set_key(pg.xref, "Resources/Font", "null")
    return doc


def _fit(us: List[float], pts: List[float]) -> Tuple[float, float]:
    """Least-squares a, b with pt = a*u + b."""
    n = len(us)
    mu, mp = sum(us) / n, sum(pts) / n
    a = sum((u - mu) * (p - mp) for u, p in zip(us, pts)) / sum((u - mu) ** 2 for u in us)
    return a, mp - a * mu


# ------------------------------------------------------------------ compose
class System:
    """The engraved three-measure system and its coordinate maps."""

    def __init__(self) -> None:
        self.svg = engrave(musicxml())
        self.geo = Geometry(self.svg)
        self.doc = svg_to_pdf(self.svg)
        self.rect: fitz.Rect | None = None
        self._calibrate()

    def _calibrate(self) -> None:
        """Affine maps viewBox units -> notation-page points, fitted on the ten
        staff lines (y) and the system barlines (x) as actually drawn."""
        h_seg, v_seg = [], []
        for d in self.doc[0].get_drawings():
            for it in d["items"]:
                if it[0] != "l":
                    continue
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.05 and abs(p2.x - p1.x) > 1:
                    h_seg.append((abs(p2.x - p1.x), p1.y))
                elif abs(p1.x - p2.x) < 0.05 and abs(p2.y - p1.y) > 1:
                    v_seg.append((abs(p2.y - p1.y), p1.x))

        def longest_distinct(segs: List[Tuple[float, float]], n: int, merge: float = 0.0) -> List[float]:
            # svglib emits each stroke several times; keep one entry per
            # coordinate (merging positions closer than `merge`), ranked by length
            best: Dict[float, float] = {}
            for length, coord in segs:
                key = round(coord, 2)
                near = next((k for k in best if abs(k - key) <= merge), None)
                key = near if near is not None else key
                best[key] = max(best.get(key, 0.0), length)
            return sorted(k for k, _ in sorted(best.items(), key=lambda kv: -kv[1])[:n])

        pdf_lines = longest_distinct(h_seg, 10)
        svg_lines = sorted(y for st in self.geo.staves for y in st["lines"])
        if len(pdf_lines) != 10 or len(svg_lines) != 10:
            raise RuntimeError(f"staff-line calibration failed: {len(pdf_lines)} pdf, {len(svg_lines)} svg")
        self._ay, self._by = _fit(svg_lines, pdf_lines)
        staff_h_pt = pdf_lines[4] - pdf_lines[0]
        space_pt = pdf_lines[1] - pdf_lines[0]
        tall = [(h, x) for h, x in v_seg if h >= 0.95 * staff_h_pt]
        pdf_bars = longest_distinct(tall, len(self.geo.barlines), merge=0.8 * space_pt)
        if len(pdf_bars) != len(self.geo.barlines):
            raise RuntimeError(f"barline calibration failed: {pdf_bars} vs {self.geo.barlines}")
        self._ax, self._bx = _fit([float(x) for x in self.geo.barlines], pdf_bars)
        bbox = fitz.Rect()
        for d in self.doc[0].get_drawings():
            bbox |= d["rect"]
        self.bbox = bbox + (-1.5, -1.5, 1.5, 1.5)

    def place(self, page: fitz.Page, x0: float, y0: float, width: float) -> float:
        s = width / self.bbox.width
        self.rect = fitz.Rect(x0, y0, x0 + width, y0 + self.bbox.height * s)
        page.show_pdf_page(self.rect, self.doc, 0, clip=self.bbox)
        return self.rect.height

    def placed_height(self, width: float) -> float:
        return self.bbox.height * width / self.bbox.width

    def pt(self, ux: float, uy: float) -> Tuple[float, float]:
        assert self.rect is not None
        s = self.rect.width / self.bbox.width
        return (self.rect.x0 + (self._ax * ux + self._bx - self.bbox.x0) * s,
                self.rect.y0 + (self._ay * uy + self._by - self.bbox.y0) * s)

    def head(self, staff: int, measure: int, beat: int) -> Tuple[int, int]:
        """Notehead (x, y) of the note on `beat` of `measure` (0-based) on
        `staff` (0 score, 1 played); notes come out of verovio in time order."""
        heads = self.geo.staves[staff]["heads"]
        idx = 0
        for m, mb in enumerate(MEASURES):
            beats = ([1, 3] if mb is None else sorted([1, 3, mb])) if staff == 0 else [1, 2, 3]
            if m == measure:
                return heads[idx + beats.index(beat)]
            idx += len(beats)
        raise IndexError(measure)


def main() -> None:
    fontfile = next((p for p in TNR_CANDIDATES if os.path.exists(p)), None)
    if fontfile is None:
        raise SystemExit("Times New Roman TTF not found; set TNR_CANDIDATES")
    font = fitz.Font(fontfile=fontfile)

    sysm = System()
    notation_w = COL_W - LABEL_MARGIN - RIGHT_PAD
    top = 1.5 + FONT_PT + 1.0          # room for the "missed" labels above the top staff
    sys_h = sysm.placed_height(notation_w)
    y_labels = top + sys_h + FONT_PT + 4.5       # (a)/(b)/(c) baseline
    box_h = FONT_PT + 5.0
    box_top = y_labels + 5.0
    bottom = box_top + box_h + 1.5
    doc = fitz.open()
    page = doc.new_page(width=COL_W, height=bottom)

    def text(x: float, y: float, s: str, size: float = FONT_PT, color: str = C_TEXT,
             align: str = "left") -> None:
        w = font.text_length(s, fontsize=size)
        if align == "center":
            x -= w / 2
        elif align == "right":
            x -= w
        page.insert_text((x, y), s, fontsize=size, fontname="TNR", fontfile=fontfile,
                         color=hex_rgb(color))

    def box(staff: int, measure: int, beat: int, color: str, pad: float = 0.0,
            dashes: str | None = None, width: float = 0.7,
            left_pad: float | None = None) -> fitz.Rect:
        ux, uy = sysm.head(staff, measure, beat)
        sp = sysm.geo.space
        has_accidental = staff == 1 and beat == 2
        lp = pad if left_pad is None else left_pad
        x0, y0 = sysm.pt(ux - (2.6 if has_accidental else 0.9) * sp - lp * sp, uy - (1.45 + pad) * sp)
        x1, y1 = sysm.pt(ux + (1.7 + pad) * sp, uy + (1.45 + pad) * sp)
        r = fitz.Rect(x0, y0, x1, y1)
        page.draw_rect(r, color=hex_rgb(color), width=width, dashes=dashes)
        return r

    sysm.place(page, LABEL_MARGIN, top, notation_w)
    # the reference truth, marked in every measure: the inserted G#4 on the
    # played staff (dashed gray, drawn first so claim boxes sit on top of it)
    truth: Dict[int, fitz.Rect] = {}
    for measure in range(len(MEASURES)):
        truth[measure] = box(1, measure, 2, C_GRAY, pad=0.35, left_pad=0.05,
                             dashes="[1.2 1.0] 0", width=0.6)
    # staff labels, right-aligned in the left strip
    for staff, lab in ((0, "score"), (1, "played")):
        ys = sysm.geo.staves[staff]["lines"]
        _, y = sysm.pt(0, (ys[0] + ys[-1]) / 2)
        text(LABEL_MARGIN - 3, y + FONT_PT * 0.35, lab, color=C_GRAY, align="right")

    # claims: (b) = measure 1, (c) = measure 2
    boxes: Dict[str, fitz.Rect] = {}
    boxes["b_extra"] = box(1, 1, 2, C_EXTRA)
    boxes["b_miss"] = box(0, 1, MEASURES[1], C_MISS)
    boxes["c_extra"] = box(1, 2, 2, C_EXTRA)
    boxes["c_miss"] = box(0, 2, MEASURES[2], C_MISS)
    # bracket tying the two co-located claims in (c)
    bx = max(boxes["c_extra"].x1, boxes["c_miss"].x1) + 4.5
    for (p1, p2) in (((bx, boxes["c_miss"].y0), (bx, boxes["c_extra"].y1)),
                     ((bx - 4, boxes["c_miss"].y0), (bx, boxes["c_miss"].y0)),
                     ((bx - 4, boxes["c_extra"].y1), (bx, boxes["c_extra"].y1))):
        page.draw_line(p1, p2, color=hex_rgb(C_GRAY), width=0.6)
    # class labels: "extra" under its box, "missed" above the stem tip; the
    # truth box is named once, under measure (a)
    r = truth[0]
    text((r.x0 + r.x1) / 2, r.y1 + FONT_PT * 0.95, "inserted", color=C_GRAY, align="center")
    for r in (boxes["b_extra"], boxes["c_extra"]):
        text((r.x0 + r.x1) / 2, truth[1].y1 + FONT_PT * 0.95, "extra", color=C_EXTRA, align="center")
    for measure, r in ((1, boxes["b_miss"]), (2, boxes["c_miss"])):
        ux, uy = sysm.head(0, measure, MEASURES[measure])
        _, y_tip = sysm.pt(ux, uy - 3.6 * sysm.geo.space)
        half = font.text_length("missed", fontsize=FONT_PT) / 2
        cx = min((r.x0 + r.x1) / 2, COL_W - 1.0 - half)
        text(cx, y_tip - 1.5, "missed", color=C_MISS, align="center")

    # bare measure labels (a)/(b)/(c) centered under each measure (FG-021)
    bars_pt = [sysm.pt(x, 0)[0] for x in sysm.geo.barlines]
    for i, lab in enumerate(("(a)", "(b)", "(c)")):
        text((bars_pt[i] + bars_pt[i + 1]) / 2, y_labels, lab, align="center")

    # one-line report box (no fill: FG-031), centered under the system
    msg = "identical published report for (b) and (c):   extra: 1 TP     missed: 1 FP"
    w = font.text_length(msg, fontsize=FONT_PT) + 12
    rb = fitz.Rect(COL_W / 2 - w / 2, box_top, COL_W / 2 + w / 2, box_top + box_h)
    page.draw_rect(rb, color=hex_rgb("#444444"), width=0.6)
    text(COL_W / 2, box_top + box_h - 4.5, msg, align="center")

    doc.subset_fonts()
    doc.save(OUT, garbage=4, deflate=True)
    doc.close()

    chk = fitz.open(OUT)
    pg = chk[0]
    fonts = pg.get_fonts()
    print("wrote", OUT, "size", pg.rect, "fonts", fonts)
    pg.get_pixmap(dpi=300).save(OUT[:-4] + ".png")
    for f in fonts:
        if f[1] == "n/a":
            raise SystemExit(f"unembedded font in figure: {f}")


if __name__ == "__main__":
    main()
