"""
primes_wall: a dense grid of the integers 1 .. N_MAX (1 px font, PER_ROW per row, right-aligned) with
the primes bright and the composites faint. Primality is computed inside the tree: one counter
channel per prime divisor d <= sqrt(N_MAX) holds n mod d (advancing at every number start along the
row and by PER_ROW at every row start down column 0), a number is composite when any counter is 0.
The trivial hit n = d (d itself) can only happen in the first two rows (n <= 100), where every
composite has a factor 2, 3, 5 or 7, so only those four counters are tested there (and the cells
1 .. 7 are spelled out); from n = 101 on all counters are tested.

    python3.10 art/gen_primes_wall.py          # writes art/trees/primes_wall.tree
"""
import importlib.util
from functools import reduce
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
If, Leaf, Set, chain, runs_tree = gen.If, gen.Leaf, gen.Set, gen.chain, gen.runs_tree
gen.apply_geometry(1)                    # 4x10 digit cells, glyph at xm 1..3, ym 5..9

CANVAS = (1024, 1024)
FIRST_Y = 20                             # top margin: the first band starts here (a multiple of CELL_H)
PER_ROW = 50                             # numbers per row (5 digit cells of 4 px = 20 px each)
DIGITS = 4                               # digit cells per number; a blank cell follows
CELLS = DIGITS + 1
ROWS = (CANVAS[1] - FIRST_Y) // 10       # bands of 10 rows
N_MAX = ROWS * PER_ROW
DIVISORS = [d for d in range(2, int(N_MAX ** 0.5) + 1) if all(d % q for q in range(2, d))]
PRIME_COLOUR, COMPOSITE_COLOUR = (255, 200, 60), (26, 26, 26)   # composites faint: the wall stays dense
DIGIT_FONT = {"0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
              "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
              "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
              "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "001", "001", "001"],
              "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"]}
ORDER = list("0123456789")
NAMES = ["Q"] + [f"C{d}" for d in DIVISORS] + ["M", "cc", "Dp", "o", "t", "h", "th", "Z", "D", "P", "R", "G", "B"]
assert len(DIVISORS) <= 19, "a channel can only look back 19 channels (Prev19): the counters must all precede M directly"


def prev(here, there):
    """
    Pure function. Property name of channel `there` as seen from channel `here` (NAMES order).

    Examples:
        >>> prev("M", "C2") == f"Prev{len(DIVISORS)}", prev("t", "o")
        (True, 'Prev')
    """
    k = NAMES.index(here) - NAMES.index(there)
    return "Prev" + (str(k) if k > 1 else "")


def at_number_start(prev_cc, prev_dp, advance, hold):
    """
    Pure function. `advance` on the first pixel of a number (xm == 0 of its first digit cell), else
    `hold` (a small subtree; it appears twice).

    Examples:
        >>> gen.render(at_number_start("Prev2", "Prev", Set(1), Set(0))).splitlines()[0]
        'if Prev2 > 9'
    """
    return If(prev_cc, gen.cc_xm_gt(0), hold, If(prev_dp, 0, hold, advance))


def number_counter():
    """
    Pure function. Q channel: 10 * (x mod 20) + ym, so a number's first pixel column has Q <= 9;
    the same shape as cell_counter with a 20 px cell (numbers are 5 digit cells wide).

    Examples:
        >>> gen.render(number_counter()).splitlines()[1]
        '  if W > 189'
    """
    last = gen.CELL_H * (CELLS * gen.CELL_W - 1)
    along_row = If("W", last - 1, Leaf("W", -last), Leaf("W", gen.CELL_H))
    down_column0 = If("N", gen.CELL_H - 2, Leaf("N", -(gen.CELL_H - 1)), Leaf("N", 1))
    return If("x", 0, along_row, If("y", 0, down_column0, Set(gen.CELL_H - gen.Y_OFFSET)))


def digit_position(prev_cc):
    """
    Pure function. Dp channel: digit cell index 0 .. CELLS - 1 within the number, 0 at x == 0.

    Examples:
        >>> gen.render(digit_position("Prev")).splitlines()[0]
        'if x > 0'
    """
    return If("x", 0, If(prev_cc, gen.cc_xm_gt(0), Leaf("W", 0), If("W", CELLS - 2, Set(0), Leaf("W", 1))), Set(0))


def counter(d, prev_q):
    """
    Pure function. Counter channel for divisor d: n mod d (1 for n = 1). Along the row it steps at
    number starts (Q <= 9), down column 0 it steps by PER_ROW at band starts (rows y > FIRST_Y with
    ym == 0, i.e. Q == 0).

    Examples:
        >>> print(gen.render(counter(3, "Prev")).splitlines()[1])
          if Prev > 9
    """
    r = PER_ROW % d
    step_x = If("W", d - 2, Leaf("W", 1 - d), Leaf("W", 1))
    step_y = If("N", d - 1 - r, Leaf("N", r - d), Leaf("N", r)) if r else Leaf("N", 0)
    column = If("y", FIRST_Y, If(prev_q, 0, Leaf("N", 0), step_y), Set(1))
    return If("x", 0, If(prev_q, gen.CELL_H - 1, Leaf("W", 0), step_x), column)


def digit(prev_cc, prev_dp, prev_lower, row_step, init):
    """
    Pure function. A decimal digit channel holding 0 .. 9; a value of 10 .. 19 right after a step
    means "wrapped, carry out" and is read by the next digit through `prev_lower` on the same
    pixel; holds normalise it back. `row_step` is the digit's own increment at a row start
    (PER_ROW's digit), `init` its value for n = 1.

    Examples:
        >>> gen.render(digit("Prev2", "Prev", None, 0, 1)).splitlines()[0]
        'if x > 0'
    """
    norm = lambda pred: If(pred, 9, Leaf(pred, -10), Leaf(pred, 0))
    carry = lambda pred, base: If(prev_lower, 9, Leaf(pred, base + 1), Leaf(pred, base)) if prev_lower else Leaf(pred, base + 1)
    along = at_number_start(prev_cc, prev_dp, carry("W", 0), norm("W"))
    at_row = carry("N", row_step) if prev_lower else Leaf("N", row_step)
    column = If("y", FIRST_Y, If(prev_cc, 0, norm("N"), at_row), Set(init))
    return If("x", 0, along, column)


def composite_flag(prev_counters):
    """
    Pure function. M channel: 1 when some counter is 0 (read as |counter| == 0 through the Abs
    properties, one split each), else 0. Evaluated on every pixel (the counters hold their values
    across a number). Rows 0 and 1 (n <= 100) test only the divisors <= 7 so that n = d is not a
    hit; the first seven cells of row 0 (n = 1 .. 7) are spelled out: 1, 4, 6 composite.

    Examples:
        >>> gen.render(composite_flag(["Prev", "Prev2"])).splitlines()[0]
        'if y > 39'
    """
    width = CELLS * gen.CELL_W
    chain_of = lambda ps: reduce(lambda t, p: If(p + "Abs", 0, t, Set(1)), reversed(ps), Set(0))
    small = [p for p, d in zip(prev_counters, DIVISORS) if d <= 7]
    head = If("x", 2 * width - 1, If(prev_counters[0] + "Abs", 0, Set(0), Set(1)), If("x", width - 1, Set(0), Set(1)))
    return If("y", FIRST_Y + 19, chain_of(prev_counters),
              If("y", FIRST_Y + 9, chain_of(small), If("x", 7 * width - 1, chain_of(small), head)))


def significance(prev_th, prev_h, prev_t):
    """
    Pure function. Z channel: index of the number's first non-zero digit (0 thousands .. 3 ones).

    Examples:
        >>> gen.render(significance("Prev", "Prev2", "Prev3")).splitlines()[0]
        'if Prev > 0'
    """
    return If(prev_th, 0, Set(0), If(prev_h, 0, Set(1), If(prev_t, 0, Set(2), Set(3))))


def digit_to_draw(prev_dp, prev_z, prev_digits):
    """
    Pure function. D channel: the digit of the current digit cell (leading zeros and the blank
    cell are BLANK), read off the digit channels with one runs_tree each.

    Examples:
        >>> gen.render(digit_to_draw("Prev7", "Prev6", ["Prev5", "Prev4", "Prev3", "Prev2"])).splitlines()[0]
        'if y > 19'
    """
    copy = lambda p: runs_tree(list(range(10)), Set, p)
    th, h, t, o = prev_digits
    cases = [(CELLS - 2, Set(gen.BLANK)), (2, copy(o)), (1, If(prev_z, 2, Set(gen.BLANK), copy(t))), (0, If(prev_z, 1, Set(gen.BLANK), copy(h)))]
    return If("y", FIRST_Y - 1, chain(prev_dp, cases, If(prev_z, 0, Set(gen.BLANK), copy(th))), Set(gen.BLANK))


def build():
    """
    Pure function. (tree source, channel names) for the wall.

    Examples:
        >>> src, names = build(); names[:3], names[-5:]
        (['Q', 'C2', 'C3'], ['D', 'P', 'R', 'G', 'B'])
    """
    gen.FONT.update(DIGIT_FONT)
    counters = [f"C{d}" for d in DIVISORS]
    trees = {"Q": number_counter(), "cc": gen.cell_counter(), "Dp": digit_position(prev("Dp", "cc"))}
    for d, c in zip(DIVISORS, counters):
        trees[c] = counter(d, prev(c, "Q"))
    tens_step = PER_ROW // 10 % 10
    assert PER_ROW % 10 == 0 and PER_ROW < 100, "row starts add PER_ROW as a single tens step"
    trees["o"] = digit(prev("o", "cc"), prev("o", "Dp"), None, 0, 1)
    trees["t"] = digit(prev("t", "cc"), prev("t", "Dp"), None, tens_step, 0)
    trees["h"] = digit(prev("h", "cc"), prev("h", "Dp"), prev("h", "t"), 0, 0)
    trees["th"] = digit(prev("th", "cc"), prev("th", "Dp"), prev("th", "h"), 0, 0)
    trees["t"] = digit(prev("t", "cc"), prev("t", "Dp"), prev("t", "o"), tens_step, 0)
    trees["M"] = composite_flag([prev("M", c) for c in counters])
    trees["Z"] = significance(prev("Z", "th"), prev("Z", "h"), prev("Z", "t"))
    trees["D"] = digit_to_draw(prev("D", "Dp"), prev("D", "Z"), [prev("D", n) for n in ("th", "h", "t", "o")])
    trees["P"] = gen.pattern_channel(ORDER, prev("P", "D"), prev("P", "cc"), flip=False)
    tone = lambda k: If(prev("R" if k == 0 else "GB"[k - 1], "M"), 0, Set(COMPOSITE_COLOUR[k] - (COMPOSITE_COLOUR[0] if k else 0)),
                        Set(PRIME_COLOUR[k] - (PRIME_COLOUR[0] if k else 0)))
    trees["R"] = gen.colour_channel(prev("R", "P"), prev("R", "cc"), on=tone(0))
    trees["G"], trees["B"] = tone(1), tone(2)
    tree = gen.simplify(gen.dispatch([trees[n] for n in NAMES]))
    header = f"""/* primes_wall.tree — GENERATED by art/gen_primes_wall.py; edit the generator, not this file.
   {CANVAS[0]}x{CANVAS[1]}: the integers 1 .. {N_MAX} in a dense grid ({PER_ROW} per row, 1 px font, right-aligned),
   primes {PRIME_COLOUR}, composites {COMPOSITE_COLOUR}. One counter channel per prime divisor
   {DIVISORS[0]} .. {DIVISORS[-1]} holds n mod d; composite = some counter is 0 (rows 0-1 test only 2 3 5 7, so n = d is no hit).
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(NAMES))}
     Q number-pixel counter (10 * (x mod 20) + ym); C<d> counters; M composite; cc cell counter (4x10 digit
     cells); Dp digit index 0..4 in the number; o t h th decimal
     digits (10..19 = carry out); Z first non-zero digit; D digit to draw; P glyph-row pattern;
     R brightness; G B RCT 3 deltas.
   Nodes: {gen.count_nodes(tree)} */
Width {CANVAS[0]}
Height {CANVAS[1]}
RCT 3
HiddenChannel {len(NAMES) - 3}
"""
    return header + gen.render(tree) + "\n", NAMES


def generate(out=None):
    """Command. Writes the tree (default art/trees/primes_wall.tree) and prints the node count."""
    src, names = build()
    out = Path(out) if out else ROOT / "art/trees/primes_wall.tree"
    out.write_text(src)
    print(f"{out.name}: {CANVAS[0]}x{CANVAS[1]}, {N_MAX} numbers, {len(DIVISORS)} divisor channels, {src.split('Nodes: ')[1].split(' ')[0]} nodes, {len(names)} channels")


if __name__ == "__main__":
    fire.Fire(generate)
