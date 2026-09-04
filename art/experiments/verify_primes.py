"""
Read the primes piece back from its decoded PNG and check every band: the label reads n, its
colour says prime iff n is prime, the sieve marks sit exactly at the proper divisors and the
prime bar agrees.

    python3.10 art/experiments/verify_primes.py                 # art/out/primes.png
    python3.10 art/experiments/verify_primes.py some/other.png
"""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]   # dump root (portable)
spec = importlib.util.spec_from_file_location("primes", ROOT / "art/gen_primes.py")
primes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primes)
BY_ROWS = {primes.digit_rows(d): d for d in range(10)}


def is_prime(n):
    """
    Pure function. Trial division.

    Examples:
        >>> [n for n in range(1, 30) if is_prime(n)]
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def close(colour, target, tolerance=3):
    """
    Pure function. True when two RGB triples agree within `tolerance` per component.

    Examples:
        >>> close((255, 205, 70), (254, 206, 70)), close((0, 0, 0), (7, 8, 14))
        (True, False)
    """
    return all(abs(a - b) <= tolerance for a, b in zip(colour, target))


def read_glyph(px, x0, y0):
    """Query (reads pixels). Digit at glyph column x0 / band row y0, None when blank, '?' when unknown."""
    rows = tuple(sum((1 << (2 - c)) for c in range(3) if not close(px[x0 + c, y0 + r], primes.BACKGROUND)) for r in range(primes.DIGIT_ROWS))
    if not any(rows):
        return None
    return BY_ROWS.get(rows, "?")


def check_band(px, n):
    """Query (reads pixels). List of problems found in band n (empty when the band is right)."""
    y = primes.Y0 + primes.BAND * (n - 1)
    problems = []
    digits = [read_glyph(px, primes.LABEL_X + k * primes.LABEL_PITCH, y) for k in range(3)]
    shown = int("".join(str(d) for d in digits if d is not None)) if "?" not in digits and any(d is not None for d in digits) else None
    if shown != n:
        problems.append(f"label reads {digits}")
    lit = next((px[primes.LABEL_X + 10 + c, y + r] for r in range(primes.DIGIT_ROWS) for c in range(3)
                if not close(px[primes.LABEL_X + 10 + c, y + r], primes.BACKGROUND)), None)
    expected = primes.ONE if n == 1 else primes.PRIME if is_prime(n) else primes.COMPOSITE
    if lit is None or not close(lit, expected):
        problems.append(f"label colour {lit}, expected {expected}")
    count = sum(n % d == 0 for d in range(2, n))
    units = [not close(px[primes.BAR_X + primes.BAR_UNIT * j + 3, y + 2], primes.BACKGROUND) for j in range(count + 1)]
    if is_prime(n):
        if not (units[0] and close(px[primes.BAR_X + 3, y + 2], primes.PRIME)):
            problems.append("prime stub missing")
    elif units != [True] * count + [False]:
        problems.append(f"bar shows {units.index(False) if False in units else '>' + str(count)} units, expected {count}")
    for k in range(primes.LABEL_X // primes.BAND):
        d = k + 2
        mark = not close(px[primes.BAND * k + 3, y + 2], primes.BACKGROUND)
        if mark != (n % d == 0 and n >= 2 * d):
            problems.append(f"mark for d={d} {'present' if mark else 'missing'}")
            break
    return problems


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "art/out/primes.png"
    px = Image.open(path).convert("RGB").load()
    bad = {n: check_band(px, n) for n in range(1, primes.NUMBERS + 1)}
    bad = {n: p for n, p in bad.items() if p}
    for n, p in list(bad.items())[:20]:
        print(f"BAD {n}: {'; '.join(p)}")
    shown_primes = sum(is_prime(n) for n in range(1, primes.NUMBERS + 1))
    print(f"primes: {primes.NUMBERS} numbers checked ({shown_primes} primes), {len(bad)} bands wrong")
    assert not bad
