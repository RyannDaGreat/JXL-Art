"""
Read the rendered equations back out of art/out/<piece>.png and check every line is well formed.

    python3.10 art/experiments/verify_equations.py equations       # v1 (names written downward)
    python3.10 art/experiments/verify_equations.py equations_v2    # v2 (names inline)
    python3.10 art/experiments/verify_equations.py equations_v4    # v4 (one = per line, tall parens read as bars)
"""
import importlib.util
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]   # dump root (portable)
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
BY_BITS = {tuple(int(b) for row in gen.FONT[ch] for b in row): ch for ch in gen.EQ_GLYPHS}
TALL_BARS = {(1, 0, 0) * 5: "(", (0, 0, 1) * 5: ")"}     # what a tall paren shows in the text rows
assert not set(TALL_BARS) & set(BY_BITS)
BY_BITS.update(TALL_BARS)
LAYOUT = {  # piece -> (cell bands holding text, glyph y0 within the band); default: every band from 1
    "equations": (range(1, 32, gen.EQ_ROWS), gen.GLYPH_Y0), "equations_crt": (range(1, 32, gen.EQ_ROWS), gen.GLYPH_Y0),
    "equations_v4": (range(2, 32, 2), gen.PIXEL * (gen.TALL_TEXT_ROW - gen.CELL_H // gen.PIXEL))}


def max_channel(im):
    """Pure function. Greyscale image of max(R, G, B), so any coloured glyph counts as lit."""
    r, g, b = im.split()
    from PIL import ImageChops
    return ImageChops.lighter(ImageChops.lighter(r, g), b)


def read_cell(px, cx0, cy0, glyph_y0=gen.GLYPH_Y0):
    """Query (reads pixels). Character drawn in the cell whose top-left is (cx0, cy0); ' ' when blank, '?' if unknown."""
    bits = tuple(int(px[cx0 + gen.GLYPH_X0 + c * gen.PIXEL + 1, cy0 + glyph_y0 + r * gen.PIXEL + 1] > 60)
                 for r in range(5) for c in range(3))
    return BY_BITS.get(bits, " " if not any(bits) else "?")


def well_formed(line, one_equals=False):
    """
    Pure function. True when parentheses balance and tokens alternate legally; a function name
    (a whole word, or in v1 just its first letter) must be followed by "("; with `one_equals`
    the line must contain exactly one "=".

    Examples:
        >>> well_formed("S(Y+1)·(2-Z)"), well_formed("SIN(Y+1)·EXP(2-Z)")
        (True, True)
        >>> well_formed("(Y+"), well_formed("SI(Y)")
        (False, False)
        >>> well_formed("Y=2", True), well_formed("Y=2=3", True), well_formed("Y+2", True)
        (True, False, False)
    """
    if one_equals and line.count("=") != 1:
        return False
    line = re.sub("|".join(gen.EQ_WORDS), "F", line)
    line = re.sub("[" + "".join(w[0] for w in gen.EQ_WORDS) + "]", "F", line)
    depth, expect_operand, prev_fn = 0, True, False
    for ch in line:
        if ch == "(":
            if not (expect_operand or prev_fn):
                return False
            depth += 1
            expect_operand = True
        elif ch == ")":
            if expect_operand or depth == 0:
                return False
            depth -= 1
        elif ch in "YZ123":
            if not expect_operand:
                return False
            expect_operand = False
        elif ch in "+-·/^=":
            if expect_operand:
                return False
            expect_operand = True
        elif ch == "F":
            if not expect_operand:
                return False
            prev_fn = True
            continue
        else:
            return False
        prev_fn = False
    return depth == 0 and not expect_operand


def rendered_lines(piece):
    """Query (reads the PNG). Non-empty text lines of the piece, one per decoder group and cell band."""
    im = Image.open(ROOT / f"art/out/{piece}.png").convert("RGB")
    px = max_channel(im).load()
    bands, glyph_y0 = LAYOUT.get(piece, (range(1, 32), gen.GLYPH_Y0))
    lines = []
    for group_y0 in range(0, im.height, 1024):
      for group_x0 in range(0, im.width, 1024):
        for k in bands:
            y0 = group_y0 + gen.Y_OFFSET + 32 * k
            if y0 + 32 <= group_y0 + 1024:
                line = "".join(read_cell(px, group_x0 + 16 * i, y0, glyph_y0) for i in range(64)).rstrip()
                if line:
                    lines.append(line)
    return lines


if __name__ == "__main__":
    piece = sys.argv[1] if len(sys.argv) > 1 else "equations"
    lines = rendered_lines(piece)
    one_equals = piece == "equations_v4"
    bad = [line for line in lines if not well_formed(line, one_equals)]
    for line in lines:
        print(("OK  " if well_formed(line, one_equals) else "BAD ") + line)
    print(f"\n{piece}: {len(lines)} lines, {len(bad)} malformed")
    assert not bad
