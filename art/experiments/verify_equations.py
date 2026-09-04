"""Read the rendered equations back out of art/out/equations.png and check every line is well formed."""
import importlib.util
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]   # dump root (portable)
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
im = Image.open(ROOT / "art/out/equations.png").convert("L")
px = im.load()
bits_of = lambda ch: tuple(int(b) for row in gen.FONT[ch] for b in row)
by_bits = {bits_of(ch): ch for ch in gen.EQ_GLYPHS}


def read_cell(cx0, cy0):
    """Query. Character drawn in the cell whose top-left is (cx0, cy0), or ' ' when blank."""
    bits = tuple(int(px[cx0 + gen.GLYPH_X0 + c * gen.PIXEL + 1, cy0 + gen.GLYPH_Y0 + r * gen.PIXEL + 1] > 127)
                 for r in range(5) for c in range(3))
    return by_bits.get(bits, " " if not any(bits) else "?")


def well_formed(line):
    """
    Pure function. True when parentheses balance and tokens alternate legally (fn must be followed by "(").

    Examples:
        >>> well_formed("S(Y+1)*(2-Z)")
        True
        >>> well_formed("(Y+")
        False
    """
    depth, expect_operand, prev_fn = 0, True, False
    for ch in line:
        if ch == "(":
            if not (expect_operand or prev_fn): return False
            depth += 1; expect_operand = True
        elif ch == ")":
            if expect_operand or depth == 0: return False
            depth -= 1
        elif ch in "YZ123":
            if not expect_operand: return False
            expect_operand = False
        elif ch in "+-*/^=":
            if expect_operand: return False
            expect_operand = True
        elif ch in "SCT":
            if not expect_operand: return False
            prev_fn = True; continue
        else:
            return False
        prev_fn = False
    return depth == 0 and not expect_operand


if __name__ == "__main__":
    lines = []
    for group_x0 in (0, 1024):
        for k in range(1, 32, gen.EQ_ROWS):            # expression bands: RT == 0
            y0 = gen.Y_OFFSET + 32 * k
            if y0 + 32 <= 1024:
                lines.append("".join(read_cell(group_x0 + 16 * i, y0) for i in range(64)).rstrip())
    bad = [line for line in lines if not well_formed(line)]
    for line in lines:
        print(("OK  " if well_formed(line) else "BAD ") + line)
    print(f"\n{len(lines)} lines, {len(bad)} malformed")
    assert not bad
