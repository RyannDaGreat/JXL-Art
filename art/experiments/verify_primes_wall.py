"""
Read every number of art/out/primes_wall.png back (digits and colour) and check it against trial division.

    python3.10 art/experiments/verify_primes_wall.py
"""
import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wall", ROOT / "art/gen_primes_wall.py")
wall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wall)
BY_BITS = {tuple(int(b) for row in rows for b in row): ch for ch, rows in wall.DIGIT_FONT.items()}
CELL_W, CELL_H, GLYPH_X0, GLYPH_Y0 = 4, 10, 1, 5


def is_prime(n):
    """
    Pure function. Trial division.

    Examples:
        >>> [n for n in range(1, 20) if is_prime(n)]
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def read_number(px, x0, y0):
    """Query (reads pixels). (value, bright) of the number whose first digit cell starts at (x0, y0); value None if unreadable."""
    digits, bright = "", None
    for k in range(wall.DIGITS):
        cx = x0 + CELL_W * k
        pts = [(cx + GLYPH_X0 + c, y0 + GLYPH_Y0 + r) for r in range(5) for c in range(3)]
        bits = tuple(int(max(px[p]) > 12) for p in pts)
        if not any(bits):
            continue
        ch = BY_BITS.get(bits)
        if ch is None:
            return None, None
        digits += ch
        bright = max(max(px[p]) for p in pts) > 200
    return (int(digits) if digits else None), bright


if __name__ == "__main__":
    im = Image.open(ROOT / "art/out/primes_wall.png").convert("RGB")
    px = im.load()
    wrong, checked, primes = [], 0, 0
    for row in range(wall.ROWS):
        for j in range(wall.PER_ROW):
            n = row * wall.PER_ROW + j + 1
            value, bright = read_number(px, wall.CELLS * CELL_W * j, wall.FIRST_Y + CELL_H * row)
            checked += 1
            if value != n or bright != is_prime(n):
                wrong.append((n, value, bright))
            primes += bool(bright)
    print(f"primes_wall: {checked} numbers checked, {primes} bright, {len(wrong)} wrong: {wrong[:10]}")
    assert not wrong
