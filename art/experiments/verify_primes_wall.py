"""
Read every number of a prime wall back (digits and colour) and check it against trial division.

    python3.10 art/experiments/verify_primes_wall.py                       # primes_wall
    python3.10 art/experiments/verify_primes_wall.py --wall primes_wall_4k
"""
import importlib.util
from pathlib import Path

import fire
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wall", ROOT / "art/gen_primes_wall.py")
walls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walls)
BY_BITS = {tuple(int(b) for row in rows for b in row): ch for ch, rows in walls.DIGIT_FONT.items()}
CELL_W, GLYPH_X0, LIT, BRIGHT = 4, 1, 12, 200


def is_prime(n):
    """
    Pure function. Trial division.

    Examples:
        >>> [n for n in range(1, 20) if is_prime(n)]
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def read_number(px, x0, y0, digits, glyph_y0):
    """Query (reads pixels). (value, bright) of the number whose first digit cell starts at (x0, y0); value None if unreadable."""
    text, bright = "", None
    for k in range(digits):
        cx = x0 + CELL_W * k
        pts = [(cx + GLYPH_X0 + c, y0 + glyph_y0 + r) for r in range(5) for c in range(3)]
        bits = tuple(int(max(px[p]) > LIT) for p in pts)
        if not any(bits):
            continue
        ch = BY_BITS.get(bits)
        if ch is None:
            return None, None
        text += ch
        bright = max(max(px[p]) for p in pts) > BRIGHT
    return (int(text) if text else None), bright


def verify(wall="primes_wall"):
    """Command. Prints the check and asserts every number and colour is right."""
    w = walls.WALLS[wall]
    im = Image.open(ROOT / f"art/out/{w.name}.png").convert("RGB")
    px = im.load()
    glyph_y0 = w.cell_h - 5
    per_side = w.canvas[0] // walls.gen.GROUP
    wrong, checked, primes = [], 0, 0
    for g in range(w.groups):
        gx0, gy0 = (g % per_side) * walls.gen.GROUP, (g // per_side) * walls.gen.GROUP
        for row in range(w.rows):
            for j in range(w.per_row):
                n = w.group_size * g + w.per_row * row + j + 1
                value, bright = read_number(px, gx0 + w.cell_px * j, gy0 + walls.FIRST_Y + w.cell_h * row, w.digits, glyph_y0)
                checked += 1
                if value != n or bright != is_prime(n):
                    wrong.append((n, value, bright))
                primes += bool(bright)
            edge = [(x, y) for x in range(gx0 + w.per_row * w.cell_px, gx0 + walls.gen.GROUP) for y in range(gy0 + walls.FIRST_Y + w.cell_h * row, gy0 + walls.FIRST_Y + w.cell_h * (row + 1))]
            if any(max(px[p]) > LIT for p in edge):
                wrong.append(("edge", g, row, None))
    print(f"{w.name}: {checked} numbers checked, {primes} bright, {len(wrong)} wrong: {wrong[:10]}")
    assert not wrong


if __name__ == "__main__":
    fire.Fire(verify)
