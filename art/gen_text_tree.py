#!/usr/bin/env python3
"""
Generates the text pieces: a grid of pseudo-random characters from a tiny pixel font, chosen by
a Rule-30 cellular automaton, optionally masked to a shape (only whole characters are drawn).

    python3.10 art/gen_text_tree.py                                  # text.tree (32 glyphs, 1024x1024)
    python3.10 art/gen_text_tree.py --mask lemniscate --name text_infinity          # 2048x1024

Trees are built from a tiny DSL (If / Leaf tuples rendered to jxl_from_tree syntax) so every
repeated structure (counters, bit accumulation, run-length lookups) is one function.

Cell layout: CELL_W x CELL_H px cells; one hidden counter holds cc = CELL_H * xm + ym, so both
cell coordinates are thresholds on one value (xm > k <=> cc > CELL_H*k + CELL_H-1; ym > k inside
a known xm column <=> cc > CELL_H*xm + k). Bands of cells start at Y_OFFSET; the first
BLANK_BANDS bands stay blank so the first drawn line samples the automaton 60 steps below the
seed row (a Weyl seed sampled every 16 px always has a near-return with ~3% differing bits at
some distance, and fewer steps left visible echoes). Orientation 4 flips the image so the spare
margin shows at the bottom; the font is emitted upside down so letters display upright.
Glyph pixels are PIXEL x PIXEL at (GLYPH_X0, GLYPH_Y0) in the cell. The character value V is
assembled in column xm == SAMPLE_X from consecutive Rule-30 rows (Wolfram's Rule-30 PRNG), one
bit per row; a negative V means "blank cell" (there is no blank glyph).

Glyph rows are stored as 3-bit patterns in hidden channel P (one run-length tree per glyph row,
computed at xm == GLYPH_X0, copied right); at the start of glyph columns 1 and 2 the consumed
high bit is subtracted so the colour channel lights a pixel with a single threshold.
Character order is optimised (random restarts + 2-opt) to minimise pattern changes.

Canvases wider than one JPEG XL group (1024 px) decode as independent groups with local x, so
the seed row and the mask field branch on the group id g (21 = left, 22 = right).
"""
import math
import random
from fractions import Fraction
from pathlib import Path

import fire

TREE_DIR = Path(__file__).resolve().parent / "trees"

FONT = {  # 3x5 glyphs, 5 rows of 3 bits
    "A": ["010", "101", "111", "101", "101"], "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"], "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"], "F": ["111", "100", "110", "100", "100"],
    "G": ["011", "100", "101", "101", "011"], "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"], "J": ["001", "001", "001", "101", "010"],
    "K": ["101", "101", "110", "101", "101"], "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"], "N": ["110", "101", "101", "101", "101"],
    "O": ["010", "101", "101", "101", "010"], "P": ["110", "101", "110", "100", "100"],
    "Q": ["010", "101", "101", "010", "001"], "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"], "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"], "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"], "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"], "Z": ["111", "001", "010", "100", "111"],
    "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"], "!": ["010", "010", "010", "000", "010"],
    "?": ["110", "001", "010", "000", "010"], ".": ["000", "000", "000", "000", "010"],
    ",": ["000", "000", "000", "010", "100"], "{": ["011", "010", "110", "010", "011"],
    "}": ["110", "010", "011", "010", "110"], "(": ["010", "100", "100", "100", "010"],
    ")": ["010", "001", "001", "001", "010"], "+": ["000", "010", "111", "010", "000"], " ": ["000"] * 5,
    "-": ["000", "000", "111", "000", "000"], "·": ["000", "000", "010", "000", "000"],
    "/": ["001", "001", "010", "100", "100"], "=": ["000", "111", "000", "111", "000"],
    "^": ["010", "101", "000", "000", "000"], "3": ["111", "001", "011", "001", "111"],
}
CHARSET_32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "!?.,{}"
CHARSET_16 = "ABDEFHIKNOPRTVXY"     # 16 letters chosen by local search (art/experiments/charset_search.py) for the fewest glyph-row changes
GLYPH_ROWS, GLYPH_COLS, PIXEL = 5, 3, 4
GLYPH_X0, GLYPH_Y0 = 2, 12
SAMPLE_X = 1                           # column where V is assembled (x = 1 + 16k, past the CA's edge column)
CELL_W, CELL_H = 16, 32
Y_OFFSET = 20                          # first cell band (bands repeat every CELL_H rows from here)
BLANK_BANDS = 1                        # leading bands left blank: first drawn line is 60 Rule-30 steps from the seed
FLIP = True                            # Orientation 4 (vertical flip): the spare margin shows at the bottom
BLANK = -1                             # V value of a blank cell (any negative value)
CA_ON, CA_THRESHOLD = 1024, 511        # Rule-30 "on" value and the test "> 511" used everywhere
WEYL_MOD, WEYL_STEPS = 1024, (633, 411)   # seed sequences v = (v + step) mod 1024; one step per group
GROUP, FIRST_GROUP = 1024, 21          # JPEG XL group width and the id of the first group (ids follow in raster order)
SEED_SHIFT = 37                        # with > 2 groups, group k starts its seed row 37 * k values along the Weyl sequence (odd: never a whole cell)
WHITE = 255
LOGO_TEXT = "JXL-RS "                  # jxl_rs: every run of drawn cells spells this from its left edge
LOGO_TONE_SPLIT = 3                    # jxl_rs: counter values 1..3 (JXL) in RUST_ORANGE, the rest in RUST_TAN
RUST_ORANGE, RUST_TAN = (247, 76, 0), (222, 165, 132)   # Ferris orange; GitHub's Rust language colour
OPTIMIZER_RESTARTS = 40

# ---------------------------------------------------------------- tree DSL (pure functions)


def If(prop, split, then, other):
    """
    Pure function. Decision node: `then` when prop > split, else `other`.

    Examples:
        >>> If("x", 3, Set(1), Set(0))
        ('if', 'x', 3, ('leaf', 'Set', 1), ('leaf', 'Set', 0))
    """
    return ("if", prop, split, then, other)


def Leaf(pred, offset=0):
    """
    Pure function. Predictor leaf.

    Examples:
        >>> Leaf("W", 1)
        ('leaf', 'W', 1)
    """
    return ("leaf", pred, offset)


def Set(value):
    """
    Pure function. Constant leaf (the Set predictor predicts 0).

    Examples:
        >>> Set(255)
        ('leaf', 'Set', 255)
    """
    return Leaf("Set", value)


def render(tree, depth=0):
    """
    Pure function. jxl_from_tree text for a tree.

    Examples:
        >>> print(render(If("x", 3, Set(1), Leaf("W", -2))))
        if x > 3
          - Set 1
          - W -2
    """
    pad = "  " * depth
    if tree[0] == "leaf":
        return f"{pad}- {tree[1]} {tree[2]}"
    return "\n".join([f"{pad}if {tree[1]} > {tree[2]}", render(tree[3], depth + 1), render(tree[4], depth + 1)])


def count_nodes(tree):
    """
    Pure function. Decision + leaf nodes.

    Examples:
        >>> count_nodes(If("x", 3, Set(1), Set(0)))
        3
    """
    return 1 if tree[0] == "leaf" else 1 + count_nodes(tree[3]) + count_nodes(tree[4])


def simplify(tree, bounds=None):
    """
    Pure function. Removes splits whose outcome is already fixed by ancestor splits on the same
    property. libjxl validates trees and refuses to decode one containing such a split (the file
    still encodes), so every tree goes through this before rendering.

    Examples:
        >>> print(render(simplify(If("x", 5, If("x", 5, Set(1), Set(2)), Set(3)))))
        if x > 5
          - Set 1
          - Set 3
    """
    if tree[0] == "leaf":
        return tree
    bounds = bounds or {}
    _, prop, split, then, other = tree
    lo, hi = bounds.get(prop, (-2 ** 31, 2 ** 31 - 1))       # prop is known to lie in [lo, hi]
    if lo > split:
        return simplify(then, bounds)
    if hi <= split:
        return simplify(other, bounds)
    return If(prop, split, simplify(then, {**bounds, prop: (split + 1, hi)}), simplify(other, {**bounds, prop: (lo, split)}))


def chain(prop, cases, default):
    """
    Pure function. Nested ifs: cases are (split, subtree) pairs tested in order (largest split
    first); `default` is reached when no split is exceeded.

    Examples:
        >>> print(render(chain("y", [(7, Set(2)), (3, Set(1))], Set(0))))
        if y > 7
          - Set 2
          if y > 3
            - Set 1
            - Set 0
    """
    tree = default
    for split, subtree in reversed(cases):
        tree = If(prop, split, subtree, tree)
    return tree


def runs_tree(values, leaf_of, prop="Prev"):
    """
    Pure function. Balanced tree over an index (property `prop`) selecting leaf_of(values[i]);
    one split per run boundary of equal consecutive values.

    Examples:
        >>> print(render(runs_tree([1, 1, 0, 0], lambda v: Set(v))))
        if Prev > 1
          - Set 0
          - Set 1
    """
    def build(lo, hi):
        boundaries = [i for i in range(lo, hi) if values[i] != values[i + 1]]
        if not boundaries:
            return leaf_of(values[lo])
        b = boundaries[len(boundaries) // 2]
        return If(prop, b, build(b + 1, hi), build(lo, b))
    return build(0, len(values) - 1)


def dispatch(channel_trees):
    """
    Pure function. Selects a tree per channel index with an `if c > k` chain.

    Examples:
        >>> print(render(dispatch([Set(0), Set(1)])))
        if c > 0
          - Set 1
          - Set 0
    """
    tree = channel_trees[0]
    for k, subtree in enumerate(channel_trees[1:]):
        tree = If("c", k, subtree, tree)
    return tree


def per_group(two_groups, left, right):
    """
    Pure function. `left` in the first group, `right` in the second; just `left` for one group.

    Examples:
        >>> per_group(True, Set(1), Set(2))
        ('if', 'g', 21, ('leaf', 'Set', 2), ('leaf', 'Set', 1))
        >>> per_group(False, Set(1), Set(2))
        ('leaf', 'Set', 1)
    """
    return If("g", FIRST_GROUP, right, left) if two_groups else left


# ---------------------------------------------------------------- reusable channel trees


def cell_counter():
    """
    Pure function. Counter channel cc = CELL_H * (x mod CELL_W) + ((y + CELL_H - Y_OFFSET) mod CELL_H):
    +CELL_H per pixel along a row (wrapping at the last column), +1 per row down column 0.

    Examples:
        >>> print(render(cell_counter()))
        if x > 0
          if W > 479
            - W -480
            - W 32
          if y > 0
            if N > 30
              - N -31
              - N 1
            - Set 12
    """
    last_column = CELL_H * (CELL_W - 1)
    along_row = If("W", last_column - 1, Leaf("W", -last_column), Leaf("W", CELL_H))
    down_column0 = If("N", CELL_H - 2, Leaf("N", -(CELL_H - 1)), Leaf("N", 1))
    return If("x", 0, along_row, If("y", 0, down_column0, Set(CELL_H - Y_OFFSET)))


def cc_xm_gt(k):
    """
    Pure function. Split value on cc for "xm > k".

    Examples:
        >>> cc_xm_gt(1)
        63
    """
    return CELL_H * (k + 1) - 1


def cc_ym_gt(xm, k):
    """
    Pure function. Split value on cc for "ym > k" inside column xm.

    Examples:
        >>> cc_ym_gt(2, 11)
        75
    """
    return CELL_H * xm + k


def weyl(pred, step):
    """
    Pure function. v = (pred + step) mod WEYL_MOD as a wrap test plus two leaves.

    Examples:
        >>> print(render(weyl("W", 633)))
        if W > 390
          - W -391
          - W 633
    """
    return If(pred, WEYL_MOD - step - 1, Leaf(pred, step - WEYL_MOD), Leaf(pred, step))


def rule30(n_groups=1):
    """
    Pure function. Rule 30 (new = NW xor (N or NE)) with on = CA_ON = 1024. Row 0 and column 0
    are Weyl sequences (period > width, so nothing repeats; the column one injects entropy at
    the left edge because Rule 30 only moves information rightward well and a fixed edge goes
    periodic). Their values 0..1023 feed the rule directly through the same "> 511" thresholds.
    NW and NE are recovered from the NW-N and N-NE properties given N. Every decoder group
    re-seeds at its own corner, so with two groups the second group's seed row uses a different
    step, and with more each group k starts the row SEED_SHIFT * k values further along it.

    Examples:
        >>> render(rule30(1)).splitlines()[0]
        'if y > 0'
        >>> render(rule30(3)).splitlines()[-9:-6]
        ['  if x > 0', '    if W > 390', '      - W -391']
    """
    on, off = Set(CA_ON), Set(0)
    interior = If("N", CA_THRESHOLD,
                  If("NW-N", -CA_ON + CA_THRESHOLD, off, on),
                  If("NW-N", CA_THRESHOLD, If("N-NE", -CA_THRESHOLD - 1, on, off), If("N-NE", -CA_THRESHOLD - 1, off, on)))
    if n_groups > 2:
        corner = lambda k: (k * SEED_SHIFT + 1) * WEYL_STEPS[0] % WEYL_MOD
        corners = chain("g", [(FIRST_GROUP + k - 1, Set(corner(k))) for k in range(n_groups - 1, 0, -1)], Set(corner(0)))
        seed_row = If("x", 0, weyl("W", WEYL_STEPS[0]), corners)
    else:
        seed_row = per_group(n_groups == 2, weyl("W", WEYL_STEPS[0]), weyl("W", WEYL_STEPS[1]))
    return If("y", 0, If("x", 0, interior, weyl("N", WEYL_STEPS[0])), seed_row)


def value_channel(bits, prev_ca, prev_cc, sample_y0, gate=None):
    """
    Pure function. Character-value channel V: in column xm == SAMPLE_X it reads `bits` consecutive
    Rule-30 rows (ym = sample_y0 ...) as bits, most significant first; copies W to the right and
    N downward. `gate(update)` wraps the last bit's update (masks: BLANK when outside). The first
    BLANK_BANDS bands and the unused xm == 0 column hold BLANK.

    Examples:
        >>> render(value_channel(2, "Prev", "Prev2", 10)).splitlines()[0]
        'if Prev2 > 63'
    """
    rows = []
    for i in range(bits):
        w = 1 << (bits - 1 - i)
        update = If(prev_ca, CA_THRESHOLD, Set(w) if i == 0 else Leaf("N", w), Set(0) if i == 0 else Leaf("N", 0))
        if gate is not None and i == bits - 1:
            update = gate(update)
        rows.append((cc_ym_gt(SAMPLE_X, sample_y0 + i - 1), update))
    in_sample_column = chain(prev_cc, [(cc_ym_gt(SAMPLE_X, sample_y0 + bits - 1), Leaf("N", 0))] + rows[::-1], Leaf("N", 0))
    in_sample_column = If("y", Y_OFFSET + BLANK_BANDS * CELL_H - 1, in_sample_column, Set(BLANK))
    return chain(prev_cc, [(cc_xm_gt(SAMPLE_X), Leaf("W", 0)), (cc_xm_gt(SAMPLE_X - 1), in_sample_column)], Set(BLANK))


def text_counter_channel(period, prev_cc, decide_row, gate):
    """
    Pure function. Value channel V that counts 1 .. period along each run of drawn cells, so a
    run spells a fixed word: the previous cell's value arrives as W through the xm == 0 copy
    column, and a blank (negative) or group-edge (0) W restarts at 1. `gate(update)` blanks
    cells outside the mask. Decided at (SAMPLE_X, decide_row), copied right and down.

    Examples:
        >>> render(text_counter_channel(7, "Prev", 11, lambda update: update)).splitlines()[0]
        'if Prev > 63'
    """
    step = If("W", period - 1, Set(1), If("W", 0, Leaf("W", 1), Set(1)))
    in_column = If(prev_cc, cc_ym_gt(SAMPLE_X, decide_row), Leaf("N", 0),
                   If(prev_cc, cc_ym_gt(SAMPLE_X, decide_row - 1), gate(step), Leaf("N", 0)))
    in_column = If("y", Y_OFFSET + BLANK_BANDS * CELL_H - 1, in_column, Set(BLANK))
    return chain(prev_cc, [(cc_xm_gt(SAMPLE_X), Leaf("W", 0)), (cc_xm_gt(SAMPLE_X - 1), in_column)], Leaf("W", 0))


# ---------------------------------------------------------------- glyph lookup


def glyph_row_patterns(rows):
    """
    Pure function. 5 rows of '010' strings -> tuple of 5 ints 0..7 (left pixel = bit 2).

    Examples:
        >>> glyph_row_patterns(["010", "101", "111", "101", "101"])
        (2, 5, 7, 5, 5)
    """
    return tuple(int(row, 2) for row in rows)


def decoded_glyph(ch, flip=FLIP):
    """
    Pure function. Glyph rows in decode order: reversed when the image is displayed flipped.

    Examples:
        >>> decoded_glyph("L", flip=True)[0], decoded_glyph("L", flip=False)[0]
        ('111', '100')
    """
    return FONT[ch][::-1] if flip else FONT[ch]


def path_cost(order, dist):
    """
    Pure function. Sum of dist over consecutive pairs.

    Examples:
        >>> path_cost("abc", lambda a, b: 1)
        2
    """
    return sum(dist(a, b) for a, b in zip(order, order[1:]))


def two_opt(order, dist):
    """
    Pure function. Improves an open path by segment reversals until no gain.

    Examples:
        >>> two_opt(["a", "c", "b"], lambda p, q: abs(ord(p) - ord(q)))
        ['a', 'b', 'c']
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if path_cost(candidate, dist) < path_cost(order, dist):
                    order, improved = candidate, True
    return order


def best_order(items, dist, restarts=OPTIMIZER_RESTARTS, seed=0):
    """
    Near-pure function (uses a seeded RNG). Best of `restarts` random open paths, each improved
    by 2-opt.

    Examples:
        >>> "".join(best_order("cab", lambda p, q: abs(ord(p) - ord(q)), restarts=3)) in ("abc", "cba")
        True
    """
    rng = random.Random(seed)
    items = list(items)
    best = None
    for _ in range(restarts):
        rng.shuffle(items)
        order = two_opt(list(items), dist)
        if best is None or path_cost(order, dist) < path_cost(best, dist):
            best = order
    return best


def peel_across(at_glyph_start, prev_cc, early=0):
    """
    Pure function. Pattern-channel layout: `at_glyph_start` (the glyph row's 3-bit pattern) at
    xm == GLYPH_X0 - early, copied right with W; the consumed high bit is removed `early` pixels
    before glyph columns 1 and 2 (early=1 is what the CRT glow class needs).

    Examples:
        >>> render(peel_across(Set(5), "Prev")).splitlines()[0]
        'if Prev > 95'
    """
    start = GLYPH_X0 - early
    x1, x2 = GLYPH_X0 + PIXEL - early, GLYPH_X0 + 2 * PIXEL - early
    across = chain(prev_cc, [(cc_xm_gt(x2), Leaf("W", 0)), (cc_xm_gt(x2 - 1), If("W", 1, Leaf("W", -2), Leaf("W", 0))),
                             (cc_xm_gt(x1), Leaf("W", 0)), (cc_xm_gt(x1 - 1), If("W", 3, Leaf("W", -4), Leaf("W", 0)))], Leaf("W", 0))
    return chain(prev_cc, [(cc_xm_gt(start), across), (cc_xm_gt(start - 1), at_glyph_start)], Leaf("N", 0))


def pattern_channel(order, prev_value, prev_cc, early=0, flip=FLIP):
    """
    Pure function. Hidden channel P holding the current glyph row's 3-bit pattern: computed at
    xm == GLYPH_X0 - early from the value per glyph row (0 in the cell's margin rows and for blank
    cells, i.e. negative values), then laid out by peel_across.

    Examples:
        >>> render(pattern_channel("AB", "Prev", "Prev2")).splitlines()[0]
        'if Prev2 > 95'
    """
    patterns = {ch: glyph_row_patterns(decoded_glyph(ch, flip)) for ch in order}
    per_row = [runs_tree([patterns[ch][r] for ch in order], Set) for r in range(GLYPH_ROWS)]
    start = GLYPH_X0 - early
    row_cases = [(cc_ym_gt(start, GLYPH_Y0 + r * PIXEL - 1), per_row[r]) for r in range(GLYPH_ROWS - 1, -1, -1)]
    return peel_across(If(prev_value, -1, chain(prev_cc, row_cases, Set(0)), Set(0)), prev_cc, early)


def tall_frame(ch):
    """
    Pure function. Row patterns of glyph `ch` over the TALL_FRAME glyph rows of the tall layout:
    font rows from TALL_TEXT_ROW; parens are TALL_PARENS[opener] rows high, centred on the text,
    caps 010 and bars 100 (open) or 001 (close).

    Examples:
        >>> tall_frame("(")[9:14], tall_frame("[")[7], tall_frame("]")[8], tall_frame("]")[15], tall_frame("+")[10]
        ((2, 4, 4, 4, 2), 2, 1, 2, 2)
    """
    frame = [0] * TALL_FRAME
    openers = {**{o: o for o in TALL_PARENS}, **{c: o for o, c in TALL_CLOSERS.items()}}
    if ch in openers:
        n = TALL_PARENS[openers[ch]]
        top = TALL_TEXT_ROW + (GLYPH_ROWS - n) // 2
        bar = 0b100 if ch in TALL_PARENS else 0b001
        frame[top:top + n] = [0b010] + [bar] * (n - 2) + [0b010]
    else:
        frame[TALL_TEXT_ROW:TALL_TEXT_ROW + GLYPH_ROWS] = glyph_row_patterns(FONT[ch])
    return tuple(frame)


def tall_pattern_channel(order, prev_token, prev_cc, prev_rt):
    """
    Pure function. P for the tall layout (two bands per line, see build_equations): the decide
    band (RT 1) draws frame rows TALL_FIRST_ROW.. below its decision row and, above it, rows 16..
    of the previous line's token, which the token channel still holds there; the draw band (RT 0)
    draws frame rows 8..15. Row cases only where the frame row changes.

    Examples:
        >>> render(tall_pattern_channel(list("{[()]}Y"), "Prev", "Prev3", "Prev2")).splitlines()[0]
        'if Prev3 > 95'
    """
    frames = {ch: tall_frame(ch) for ch in order}
    rows_per_band = CELL_H // PIXEL

    def band(frame_rows):                    # frame row per glyph row of the band -> tree over ym, then token
        rows = [[frames[ch][fr] for ch in order] for fr in frame_rows]
        cases = [(cc_ym_gt(GLYPH_X0, PIXEL * r - 1), runs_tree(rows[r], Set, prev_token))
                 for r in range(len(rows) - 1, 0, -1) if rows[r] != rows[r - 1]]
        return chain(prev_cc, cases, runs_tree(rows[0], Set, prev_token))

    decide = band([r if r >= TALL_FIRST_ROW else 2 * rows_per_band + r for r in range(rows_per_band)])
    draw = band([rows_per_band + r for r in range(rows_per_band)])
    return peel_across(If(prev_token, -1, If(prev_rt, 0, decide, draw), Set(0)), prev_cc)


def colour_channel(prev_pattern, prev_cc, on=None):
    """
    Pure function. R channel: glyph column c is lit (value `on`, default WHITE) when the (already
    shifted) pattern exceeds 3, 1, 0; outside the glyph columns the pixel is black.

    Examples:
        >>> render(colour_channel("Prev", "Prev2")).splitlines()[0]
        'if Prev2 > 447'
    """
    on = Set(WHITE) if on is None else on
    lit = lambda split: If(prev_pattern, split, on, Set(0))
    x0, x1, x2, x3 = (GLYPH_X0 + c * PIXEL for c in range(4))
    return chain(prev_cc, [(cc_xm_gt(x3 - 1), Set(0)), (cc_xm_gt(x2 - 1), lit(0)), (cc_xm_gt(x1 - 1), lit(1)),
                           (cc_xm_gt(x0 - 1), lit(3))], Set(0))


# ---------------------------------------------------------------- mask (whole cells only)
# A cell is blank unless a field S is inside a band at the cell's sample point. Channels cannot
# be added, so S must be separable: Gerono's figure-eight y^2 = x^2 - x^4/a^2 is, and so are
# its cousins y^2 = x^2 - |x|^p/a^(p-2). Lower p thickens the tips (the quartic's tips are half
# as thick as its tops) but shallows the lobes' minimum so the holes close; p = 3.5 balances:
#   S = (|u|^p / a^(p-2) - u^2 + k v^2) / SCALE,  u = x - cx, v = y - cy,  p = power
# and the band -eps_in < S <= eps_out is a thick lemniscate that crosses itself at 45 degrees
# (a level-set band bulges at the crossing, where the gradient vanishes: the knot is ~1.3x the
# stroke for bold strokes, worse for thin ones). Centring the band makes membership one
# |S| <= half-width test via PrevAbs.
# Predictors are linear, so the polynomials are grown by rounded chord slopes over PIECE-px
# pieces aligned to the centre; the value is exact at breakpoints and off by a few px between.
LEMNISCATE = dict(power=3.5, a=0.44, k=1.4, eps_in=100000, eps_out=100000, scale=16, piece_x=128, piece_y=128)
# a: lobe tips at cx +- a*W; k: vertical squash; eps in px^2 units (stroke ~ 2 eps / |grad S|)
# The crossing: a level-set band bulges where the gradient vanishes, so within |u| < REACH * W
# the strokes are drawn instead as two straight lines v = +-s u of constant width, i.e. the band
# |L| <= w of the linear field L = den |v| - num |u| (num/den ~ s), fitted so the line passes
# through the centre of the polynomial band at |u| = REACH * W and has the same height there.
XBAND = dict(reach=0.22, max_denominator=9)


def lemniscate_profile(canvas):
    """
    Pure function. (f, g, cx, cy): f(u) = |u|^p / a^(p-2) - u^2 and g(v) = k v^2 in px^2 units.

    Examples:
        >>> f, g, cx, cy = lemniscate_profile((2048, 1024))
        >>> round(f(901)), round(g(10))
        (0, 140)
    """
    cx, cy, a, k = canvas[0] // 2, canvas[1] // 2, round(LEMNISCATE["a"] * canvas[0]), LEMNISCATE["k"]
    pw = LEMNISCATE["power"]
    return (lambda u: abs(u) ** pw / a ** (pw - 2) - u ** 2), (lambda v: k * v ** 2), cx, cy


def xband_geometry(canvas):
    """
    Pure function. (reach, num, den, w): at |u| = reach the polynomial band spans v in
    [v_in, v_out]; the crossing line v = (num/den) |u| goes through the middle of that span and
    |L| <= w, L = den |v| - num |u|, has the same height there.

    Examples:
        >>> reach, num, den, w = xband_geometry((2048, 1024))
        >>> reach, 0 < num / den < 1, w > 0
        (451, True, True)
    """
    f, g, _, _ = lemniscate_profile(canvas)
    k = LEMNISCATE["k"]
    reach = round(XBAND["reach"] * canvas[0])
    s0 = f(reach)                                        # S on the u axis (negative inside a lobe)
    v_in = math.sqrt(max(0.0, -LEMNISCATE["eps_in"] - s0) / k)
    v_out = math.sqrt((LEMNISCATE["eps_out"] - s0) / k)
    slope = Fraction((v_in + v_out) / 2 / reach).limit_denominator(XBAND["max_denominator"])
    return reach, slope.numerator, slope.denominator, round(slope.denominator * (v_out - v_in) / 2)


def xband_field(canvas):
    """
    Pure function. Field channel L = den |y - cy| - num |x - cx| grown with +-num / +-den steps
    (per group: |u| = x' in the right group, GROUP - x' in the left one).

    Examples:
        >>> render(xband_field((1024, 1024))).splitlines()[0]
        'if x > 0'
    """
    _, num, den, _ = xband_geometry(canvas)
    cx, cy = canvas[0] // 2, canvas[1] // 2
    ys = If("y", cy, Leaf("N", den), Leaf("N", -den))
    if canvas[0] > GROUP:
        xs = per_group(True, Leaf("W", num), Leaf("W", -num))
        base = per_group(True, Set(den * cy - num * GROUP), Set(den * cy))
    else:
        xs, base = If("x", cx, Leaf("W", -num), Leaf("W", num)), Set(den * cy - num * cx)
    return If("x", 0, xs, If("y", 0, ys, base))


def lemniscate_geometry(canvas):
    """
    Pure function. (cx, cy, a, k, mid, half_width) in field units for a canvas.

    Examples:
        >>> lemniscate_geometry((2048, 1024))[:3]
        (1024, 512, 901)
    """
    w, h = canvas
    sc = LEMNISCATE["scale"]
    mid = (LEMNISCATE["eps_out"] - LEMNISCATE["eps_in"]) / (2 * sc)
    half_width = (LEMNISCATE["eps_out"] + LEMNISCATE["eps_in"]) // (2 * sc)
    return w // 2, h // 2, round(LEMNISCATE["a"] * w), LEMNISCATE["k"], mid, half_width


def chord_pieces(profile, breakpoints, size):
    """
    Pure function. (split, slope) per piece between consecutive breakpoints clipped to [0, size):
    slope = rounded chord slope of `profile`; "prop > split" selects the piece.

    Examples:
        >>> chord_pieces(lambda t: t * t / 4, [-4, 0, 4, 8], 8)
        [(-1, 1), (3, 3)]
    """
    edges = sorted({0, size} | {b for b in breakpoints if 0 < b < size})
    return [(a - 1, round((profile(b) - profile(a)) / (b - a))) for a, b in zip(edges, edges[1:])]


def piece_chain(prop, pred, pieces):
    """
    Pure function. `pred + slope` selected by prop from chord_pieces output.

    Examples:
        >>> print(render(piece_chain("x", "W", [(-1, 1), (3, 3)])))
        if x > 3
          - W 3
          - W 1
    """
    return chain(prop, [(split, Leaf(pred, slope)) for split, slope in reversed(pieces[1:])], Leaf(pred, pieces[0][1]))


def lemniscate_field(canvas):
    """
    Pure function. Field channel S for the lemniscate mask; with two groups the crossing sits on
    the group boundary, so the left group holds u in [-1024, 0) and the right one u in [0, 1024).

    Examples:
        >>> render(lemniscate_field((1024, 1024))).splitlines()[0]
        'if x > 0'
    """
    cx, cy, a, k, mid, _ = lemniscate_geometry(canvas)
    sc, px, py = LEMNISCATE["scale"], LEMNISCATE["piece_x"], LEMNISCATE["piece_y"]
    pw = LEMNISCATE["power"]
    f = lambda u: (abs(u) ** pw / a ** (pw - 2) - u ** 2) / sc
    g = lambda v: k * v ** 2 / sc
    ys = piece_chain("y", "N", chord_pieces(lambda y: g(y - cy), [cy + j * py for j in range(-8, 9)], canvas[1]))

    def half(centre, size):        # chain over local x in [0, size) with u = x - centre
        fx = lambda x: f(x - centre)
        xs = piece_chain("x", "W", chord_pieces(fx, [centre + j * px for j in range(-16, 17)], size))
        return xs, Set(round(fx(0) + g(-cy) - mid))

    if canvas[0] > GROUP:
        (xs_l, base_l), (xs_r, base_r) = half(GROUP, GROUP), half(0, canvas[0] - GROUP)
        xs, base = per_group(True, xs_l, xs_r), per_group(True, base_l, base_r)
    else:
        xs, base = half(cx, canvas[0])
    return If("x", 0, xs, If("y", 0, ys, base))


def lemniscate_gate(canvas, prev_field, prev_xband):
    """
    Pure function. gate(update) for value_channel: within |u| < reach a cell is drawn when
    |L| <= w (straight crossing strokes), elsewhere when |S| <= half-width (the lobes).

    Examples:
        >>> render(lemniscate_gate((2048, 1024), "Prev2", "Prev")(Set(1))).splitlines()[0]
        'if g > 21'
    """
    half_width = lemniscate_geometry(canvas)[5]
    reach, _, _, w = xband_geometry(canvas)
    cx = canvas[0] // 2

    def gate(update):
        lobes = If(prev_field + "Abs", half_width, Set(BLANK), update)
        crossing = If(prev_xband + "Abs", w, Set(BLANK), update)
        if canvas[0] > GROUP:
            return per_group(True, If("x", GROUP - reach - 1, crossing, lobes), If("x", reach - 1, lobes, crossing))
        return If("x", cx + reach, lobes, If("x", cx - reach - 1, crossing, lobes))
    return gate


# ---------------------------------------------------------------- CRT look (v2)
# Scanlines: a period-SCAN row counter darkens every SCAN-th line of everything. Glow: class
# channel C is 3 on lit glyph pixels and otherwise the left neighbour minus one, a phosphor
# trail that fades to the right (2 px), plus a 1 px rim before each lit glyph column; a
# symmetric halo would need the next cell's value or a second font lookup.
# Colour: the R channel holds brightness; RCT 3 makes G = R + PHOSPHOR_G and B = R - 255 (= 0),
# so bright pixels are amber-yellow and dim ones green. (A superellipse tube vignette was tried
# and removed: banded, not smooth, and ~+120 B; see concerns.md.)
CRT = dict(scan=4, dark=0.55, levels=(8, 60, 120, 215), phosphor_g=25)


def scanline_counter():
    """
    Pure function. Row counter y mod CRT["scan"]; value 0 marks a dark line.

    Examples:
        >>> render(scanline_counter()).splitlines()[0]
        'if y > 0'
    """
    return If("y", 0, If("N", CRT["scan"] - 2, Set(0), Leaf("N", 1)), Set(0))


def class_channel(prev_pattern, prev_cc):
    """
    Pure function. Glow class C: 3 on lit glyph pixels, W - 1 elsewhere (fading trail), 2 on the
    pixel before a lit glyph column. Relies on pattern_channel computing the pattern one pixel
    before the glyph and peeling a bit on the LAST pixel of each column (crt=True layout), so on
    that pixel "lit" is W > 2 and the pattern's top bit already belongs to the next column.

    Examples:
        >>> render(class_channel("Prev", "Prev2")).splitlines()[0]
        'if Prev2 > 447'
    """
    trail = If("W", 0, Leaf("W", -1), Set(0))
    lit, rim = Set(3), Set(2)
    x0 = GLYPH_X0
    bit = [3, 1, 0]                       # "top bit set" threshold after 0, 1, 2 peels
    cases = [(cc_xm_gt(x0 + 3 * PIXEL - 1), trail)]
    for c in range(GLYPH_COLS - 1, -1, -1):
        last = x0 + (c + 1) * PIXEL - 1
        next_bit = If(prev_pattern, bit[c + 1], rim, trail) if c + 1 < GLYPH_COLS else trail
        cases.append((cc_xm_gt(last - 1), If("W", 2, lit, next_bit)))
        cases.append((cc_xm_gt(x0 + c * PIXEL - 1), If(prev_pattern, bit[c], lit, trail)))
    cases.append((cc_xm_gt(x0 - 2), If(prev_pattern, 3, rim, trail)))
    return chain(prev_cc, cases, trail)


def brightness_channel(prev_class, prev_scan):
    """
    Pure function. R channel of the CRT look: brightness by glow class, dimmed on dark scanlines.

    Examples:
        >>> render(brightness_channel("Prev", "Prev2")).splitlines()[0]
        'if Prev > 2'
    """
    levels = CRT["levels"]
    per_class = [If(prev_scan, 0, Set(v), Set(round(v * CRT["dark"]))) for v in levels]
    return chain(prev_class, [(k - 1, per_class[k]) for k in range(len(levels) - 1, 0, -1)], per_class[0])


# ---------------------------------------------------------------- random equations (a CFG)
# Tokens are generated left to right, one per cell, by a counter automaton: matched parentheses
# only need the nesting depth (the Dyck-1 language), not a stack. State channel S holds
# 8 * class + depth so the token class is a threshold band and transitions are `W + constant`
# (no per-depth duplication). Classes: OPEN, OP, the function-name state(s), CLOSE, OPERAND,
# BLANK. v1 writes function names downward (one FUNC state; hanging rows copy the successor
# letter of the glyph above); v2 writes them inline (states FN1 FN2 FN3; the token channel
# copies the successor letter of the glyph to the left) and colours tokens by kind.
# Grammar:  E -> operand | ( E ) | fn ( E ) | E op E  (op: + - · / ^, = only at depth 0)
# "(" needs depth < EQ_MAX_DEPTH and room, ")" needs depth > 0, and the last cells of a line
# force closes so every line balances. Each decoder group is one line of 64 cells.
EQ_WORDS = ("SIN", "COS", "TAN", "EXP")      # first and middle letters must be unique (successor map)
EQ_KINDS = {"parens": "()", "operators": "+-·/^=", "variables": "YZ", "digits": "123",
            "functions": "".join(sorted(set("".join(EQ_WORDS))))}
EQ_GLYPHS = "".join(EQ_KINDS.values())
EQ_MAX_DEPTH = 3
EQ_ROWS = 3                                  # v1: cell rows per expression (1 expression + 2 hanging)
EQ_RT_INIT = 1                               # v1 row type before the first band: bands go 2, 0, 1, ...
EQ_END_V = 23                                # random_length: V > 23 after a top-level operand ends the equation (1 in 4)
EQ_MIN_CELLS = 8                             # random_length: no equation ends before this many tokens
EQ_EQUALS_V = 15                             # one_equals: V > 15 after a top-level operand writes the "=" (1 in 2)
TALL_PARENS = {"{": 13, "[": 9, "(": 5}      # tall_parens: open paren glyph by depth after opening 1..3 -> height in glyph rows
TALL_CLOSERS = {"{": "}", "[": "]", "(": ")"}
TALL_TEXT_ROW = 9                            # tall_parens: frame glyph row of the text's first row (draw band row 1); parens centre on it
TALL_FIRST_ROW = 5                           # tall_parens: first frame row a decide band can draw; tokens are decided at pixel row 4*5 - 1
TALL_FRAME = 24                              # tall_parens: frame = decide band glyph rows 0..7, draw band 8..15, next decide band 16..23
EQ_PALETTE = {"parens": (170, 170, 170), "operators": (255, 140, 50), "variables": (110, 190, 255),
              "digits": (200, 150, 255), "functions": (255, 220, 90)}   # v2 syntax colours on black


def eq_classes(inline):
    """
    Pure function. Class index per state name.

    Examples:
        >>> eq_classes(False)["BLANK"], eq_classes(True)["FN3"]
        (5, 4)
    """
    names = ["OPEN", "OP"] + (["FN1", "FN2", "FN3"] if inline else ["FUNC"]) + ["CLOSE", "OPERAND", "BLANK"]
    return {n: i for i, n in enumerate(names)}


def eq_successors():
    """
    Pure function. Next letter of each non-final word letter.

    Examples:
        >>> eq_successors()["S"], eq_successors()["A"]
        ('I', 'N')
    """
    successor = {}
    for word in EQ_WORDS:
        for a, b in zip(word, word[1:]):
            assert successor.get(a, b) == b, "ambiguous letter successor"
            successor[a] = b
    return successor


def eq_row_type(prev_cc, period=EQ_ROWS, init=EQ_RT_INIT):
    """
    Pure function. RT channel: cycles 0 .. period-1 per cell band (0 = expression row; v1 uses
    1 and 2 for the hanging rows, v2 uses 1 for a blank spacer row), copied right.

    Examples:
        >>> render(eq_row_type("Prev3")).splitlines()[0]
        'if x > 0'
    """
    step = If("N", period - 2, Set(0), Leaf("N", 1))
    return If("x", 0, Leaf("W", 0), If("y", 0, If(prev_cc, 0, Leaf("N", 0), step), Set(init)))


def eq_state(prev_v, prev_cc, decide_row, line_cells, inline, first_line_y, prev_rt=None, random_length=False,
             one_equals=False, tall=False):
    """
    Pure function. State channel S (see above), decided once per cell at xm == SAMPLE_X on
    `decide_row`, copied right (W) and down (N); reset at x == 0 (each decoder group is one
    line) and blank above `first_line_y`. Random choices use the 5-bit value V. With
    `random_length` a top-level operand may end the equation once EQ_MIN_CELLS tokens are drawn.
    `one_equals` stores depth as 2*depth + a with a = 1 once the single "=" is written: "=" is
    offered only before it, ending only after it, and the forced closes keep room for "=" + operand.
    `prev_rt` row types blank the state on spacer rows, or with `tall` decide on RT 1 bands and
    copy it down through the RT 0 draw bands.

    Examples:
        >>> render(eq_state("Prev", "Prev2", 11, 64, True, 52)).splitlines()[0]
        'if x > 0'
    """
    C = eq_classes(inline)
    base = lambda cls: 8 * C[cls]
    fn_first = "FN1" if inline else "FUNC"
    fn_cells = 5 if inline else 3            # cells a function needs after its first: letters, "(", operand, ")"
    stride = 2 if one_equals else 1          # depth is stored as stride * depth (the "after =" bit sits below it)
    reserve = 1 if one_equals else 0         # forced closes start earlier so that "=" and an operand still fit
    cells_left_le = lambda k, then, other: If("x", CELL_W * (line_cells - k), then, other)

    def expect_operand(b):                   # W in [b, b + 8): depth = (W - b) // stride
        to_operand, to_open, to_func = Leaf("W", base("OPERAND") - b), Leaf("W", stride - b), Leaf("W", base(fn_first) - b)
        shallow = lambda then: If("W", b + stride * EQ_MAX_DEPTH - 1, to_operand, then)
        with_fn = If(prev_v, 23, shallow(to_open), If(prev_v, 19, shallow(to_func), to_operand))
        without_fn = If(prev_v, 23, shallow(to_open), to_operand)
        return cells_left_le(EQ_MAX_DEPTH + 3, to_operand, cells_left_le(EQ_MAX_DEPTH + fn_cells + reserve, without_fn, with_fn))

    def expect_operator(b):
        to_close, to_op, to_blank = Leaf("W", base("CLOSE") - stride - b), Leaf("W", base("OP") - b), Set(base("BLANK"))
        to_equals = Leaf("W", 1 - b)         # OPEN class at depth 0 with the "after =" bit: drawn as "=" (a real "(" has depth > 0)
        deep = lambda then, other: If("W", b + stride - 1, then, other)            # depth > 0
        after_equals = (lambda then, other: If("W", b, then, other)) if one_equals else (lambda then, other: then)
        may_end = If("x", CELL_W * EQ_MIN_CELLS, If(prev_v, EQ_END_V, to_blank, to_op), to_op) if random_length else to_op
        forced = after_equals(cells_left_le(2, to_blank, to_op), to_equals)
        free = after_equals(may_end, If(prev_v, EQ_EQUALS_V, to_equals, to_op))
        return cells_left_le(EQ_MAX_DEPTH + 1 + 2 * reserve, deep(to_close, forced), deep(If(prev_v, 21, to_close, to_op), free))

    cases = [(base("BLANK") - 1, Set(base("BLANK"))), (base("OPERAND") - 1, expect_operator(base("OPERAND"))),
             (base("CLOSE") - 1, expect_operator(base("CLOSE")))]
    if inline:
        cases += [(base("FN3") - 1, Leaf("W", stride - base("FN3"))), (base("FN2") - 1, Leaf("W", 8)), (base("FN1") - 1, Leaf("W", 8))]
    else:
        cases += [(base("FUNC") - 1, Leaf("W", stride - base("FUNC")))]
    cases += [(base("OP") - 1, expect_operand(base("OP")))]
    decide = If("y", first_line_y - 1, chain("W", cases, expect_operand(0)), Set(base("BLANK")))
    if prev_rt is not None:
        decide = If(prev_rt, 0, decide, Leaf("N", 0)) if tall else If(prev_rt, 0, Set(base("BLANK")), decide)
    sample = If(prev_cc, cc_ym_gt(SAMPLE_X, decide_row), Leaf("N", 0),
                If(prev_cc, cc_ym_gt(SAMPLE_X, decide_row - 1), decide, Leaf("N", 0)))
    cell = chain(prev_cc, [(cc_xm_gt(SAMPLE_X), Leaf("W", 0)), (cc_xm_gt(SAMPLE_X - 1), sample)], Leaf("W", 0))
    return If("x", 0, cell, Set(base("OP")))


def eq_token(order, prev_s, prev_v, prev_cc, decide_row, inline, prev_rt=None, one_equals=False, tall=False):
    """
    Pure function. Token channel T: the glyph index for the cell, picked from the state's class
    (random details from V). Word letters after the first copy the successor letter of the glyph
    to the left (inline) or above (v1 hanging rows, when the row type is not 0). Blank cells are
    BLANK (negative); the first cell of a line reads a garbage W but never needs it. `one_equals`
    draws the OPEN state at depth 0 as "=" and drops "=" from the operators; `tall` picks {, [, (
    and }, ], ) by depth and copies tokens down through the RT 0 draw bands.

    Examples:
        >>> render(eq_token(list(EQ_GLYPHS), "Prev", "Prev2", "Prev3", 11, True)).splitlines()[0]
        'if y > 0'
    """
    C = eq_classes(inline)
    base = lambda cls: 8 * C[cls]
    stride = 2 if one_equals else 1
    g = lambda ch: Set(order.index(ch))
    pick = lambda cases, default: chain(prev_v, [(k, g(ch)) for k, ch in cases], g(default))
    operators = pick([(17, "+"), (12, "-"), (7, "·"), (3, "/")], "^")
    operator = operators if one_equals else If(prev_s, base("OP"), operators, pick([(17, "+"), (12, "-"), (7, "·"), (5, "/")], "="))
    by_depth = lambda b, glyphs, first: chain(prev_s, [(b + stride * (k + first) - 1, g(glyphs[k])) for k in range(len(glyphs) - 1, 0, -1)], g(glyphs[0]))
    opens = by_depth(0, list(TALL_PARENS), 1) if tall else g("(")                          # depth after opening 1..3
    closes = by_depth(base("CLOSE"), list(TALL_CLOSERS.values()), 0) if tall else g(")")  # depth after closing 0..2
    open_or_equals = If(prev_s, stride - 1, opens, g("=")) if one_equals else opens
    first = [w[0] for w in EQ_WORDS]
    first_letter = pick([(19 + k, first[k]) for k in range(len(first) - 1, 0, -1)], first[0])
    successor = eq_successors()
    def succ(prop):        # successor letter of the glyph at `prop` (W or N); blank neighbours stay blank
        table = runs_tree([order.index(successor[ch]) if ch in successor else BLANK for ch in order], Set, prop=prop)
        return If(prop, -1, table, Set(BLANK))
    cases = [(base("BLANK") - 1, Set(BLANK)), (base("OPERAND") - 1, pick([(13, "Y"), (8, "Z"), (5, "1"), (2, "2")], "3")),
             (base("CLOSE") - 1, closes)]
    if inline:
        cases += [(base("FN3") - 1, succ("W")), (base("FN2") - 1, succ("W")), (base("FN1") - 1, first_letter)]
    else:
        cases += [(base("FUNC") - 1, first_letter)]
    cases += [(base("OP") - 1, operator)]
    expression = chain(prev_s, cases, open_or_equals)
    if prev_rt is None:
        decision = expression
    elif tall:
        decision = If(prev_rt, 0, expression, Leaf("N", 0))     # RT 0 draw bands copy the decide band's tokens down
    else:
        decision = If(prev_rt, 0, succ("N"), expression)
    sample = If(prev_cc, cc_ym_gt(SAMPLE_X, decide_row), Leaf("N", 0),
                If(prev_cc, cc_ym_gt(SAMPLE_X, decide_row - 1), decision, Leaf("N", 0)))
    # the xm == 0 column copies the previous cell's glyph so inline word letters can read it as W;
    # row 0 is BLANK so the copy chains above the first decision row start from blank, not 0
    cell = chain(prev_cc, [(cc_xm_gt(SAMPLE_X), Leaf("W", 0)), (cc_xm_gt(SAMPLE_X - 1), sample)], Leaf("W", 0))
    return If("y", 0, cell, Set(BLANK))


def eq_kinds(parens="()"):
    """
    Pure function. EQ_KINDS with the parens kind replaced (tall_parens draws six paren glyphs).

    Examples:
        >>> eq_kinds("{[()]}")["parens"], list(eq_kinds())
        ('{[()]}', ['parens', 'operators', 'variables', 'digits', 'functions'])
    """
    return {**EQ_KINDS, "parens": parens}


def kind_ends(order, kinds=EQ_KINDS):
    """
    Pure function. Last glyph index of each kind's block, in `kinds` order (the order must keep
    each kind's glyphs contiguous).

    Examples:
        >>> kind_ends(list(EQ_GLYPHS))
        [1, 7, 9, 12, 22]
    """
    ends, pos = [], 0
    for glyphs in kinds.values():
        assert set(order[pos:pos + len(glyphs)]) == set(glyphs), "glyph order must keep kinds contiguous"
        pos += len(glyphs)
        ends.append(pos - 1)
    return ends


def colour_index_channel(order, prev_token, prev_pattern, prev_cc, kinds=EQ_KINDS):
    """
    Pure function. K channel (v2): 0 on unlit pixels, else 1 + kind index of the cell's glyph,
    read off the token with thresholds at the kind blocks.

    Examples:
        >>> render(colour_index_channel(list(EQ_GLYPHS), "Prev2", "Prev", "Prev3")).splitlines()[0]
        'if Prev3 > 447'
    """
    ends = kind_ends(order, kinds)
    kind = chain(prev_token, [(ends[i - 1], Set(i + 1)) for i in range(len(ends) - 1, 0, -1)], Set(1))
    lit = lambda split: If(prev_pattern, split, kind, Set(0))
    x0, x1, x2, x3 = (GLYPH_X0 + c * PIXEL for c in range(4))
    return chain(prev_cc, [(cc_xm_gt(x3 - 1), Set(0)), (cc_xm_gt(x2 - 1), lit(0)), (cc_xm_gt(x1 - 1), lit(1)),
                           (cc_xm_gt(x0 - 1), lit(3))], Set(0))


def palette_channel(prev_kind, component):
    """
    Pure function. One colour channel from the kind index: black for 0, EQ_PALETTE otherwise.

    Examples:
        >>> render(palette_channel("Prev", 0)).splitlines()[0]
        'if Prev > 4'
    """
    colours = [EQ_PALETTE[kind][component] for kind in EQ_KINDS]
    return chain(prev_kind, [(k, Set(colours[k])) for k in range(len(colours) - 1, -1, -1)], Set(0))


def kinds_order(kinds=EQ_KINDS):
    """
    Near-pure (seeded optimiser). Glyph order with each kind's glyphs contiguous, each block
    ordered for few glyph-row changes; the parens block keeps its given order (tall parens have
    no font glyph and their bars already match).

    Examples:
        >>> o = kinds_order(); set(o[:2]) == {"(", ")"} and len(o) == len(EQ_GLYPHS)
        True
    """
    dist = lambda p, q: sum(a != b for a, b in zip(glyph_row_patterns(FONT[p]), glyph_row_patterns(FONT[q])))
    return [ch for kind, glyphs in kinds.items() for ch in (list(glyphs) if kind == "parens" else best_order(glyphs, dist))]


def build_equations(canvas, crt=False, inline=False, gap=0, name=None, row_gap=0, random_length=False, one_equals=False, tall_parens=False):
    """
    Pure function. (tree source, channel names, glyph order) for the random-equations piece.
    inline=True is v2: names written inline and syntax-coloured (no CRT variant). `gap` cells at
    the end of each decoder group stay blank (space between side-by-side equations). A
    1024-wide canvas is one group, i.e. one equation per line. `row_gap` blank cell rows are
    left between equation rows (inline only; v1 already has its two hanging rows). `random_length`
    lets equations end early at random (see eq_state) instead of filling the line. `one_equals`
    writes exactly one "=" per line (eq_state). `tall_parens` (inline, row_gap 1) draws parens sized
    by depth on a two-band layout: the RT 1 band decides tokens low enough to draw big parens'
    tops under its decision row, the RT 0 band draws the text (tall_pattern_channel).

    Examples:
        >>> build_equations((2048, 1024))[1]
        ['cc', 'A', 'V', 'RT', 'S', 'T', 'P', 'R', 'G', 'B']
        >>> build_equations((2048, 1024), inline=True)[1]
        ['cc', 'A', 'V', 'S', 'T', 'P', 'K', 'R', 'G', 'B']
    """
    assert not (crt and inline)
    assert not tall_parens or (inline and row_gap == 1), "tall parens need the inline two-band layout"
    bits = 5
    decide_row = PIXEL * TALL_FIRST_ROW - 1 if tall_parens else GLYPH_Y0 - 1
    sample_y0 = decide_row - bits + 1
    kinds = eq_kinds("{[()]}" if tall_parens else "()")
    if inline:
        order = kinds_order(kinds)
    else:
        dist = lambda p, q: sum(a != b for a, b in zip(glyph_row_patterns(FONT[p]), glyph_row_patterns(FONT[q])))
        order = best_order(EQ_GLYPHS, dist)
    n_groups = -(-canvas[0] // GROUP) * -(-canvas[1] // GROUP)
    line_cells = min(canvas[0], GROUP) // CELL_W - gap
    first_line_y = Y_OFFSET + CELL_H            # the first band is blank (too close to the seed row)

    spaced = inline and row_gap > 0
    names = ["cc", "A", "V"] + (["RT"] if not inline or spaced else []) + ["S", "T", "P"] + (["K"] if inline else ["sl", "C"] if crt else []) + ["R", "G", "B"]
    prev = lambda here, there: "Prev" + (str(names.index(here) - names.index(there)) if names.index(here) - names.index(there) > 1 else "")
    trees = {"cc": cell_counter(), "A": rule30(n_groups)}
    trees["V"] = value_channel(bits, prev("V", "A"), prev("V", "cc"), sample_y0)
    if not inline:
        trees["RT"] = eq_row_type(prev("RT", "cc"))
    elif spaced:
        trees["RT"] = eq_row_type(prev("RT", "cc"), period=row_gap + 1, init=row_gap)   # bands 0, 2, 4 .. are type 0
    trees["S"] = eq_state(prev("S", "V"), prev("S", "cc"), decide_row, line_cells, inline, first_line_y,
                          prev("S", "RT") if spaced else None, random_length, one_equals, tall_parens)
    trees["T"] = eq_token(order, prev("T", "S"), prev("T", "V"), prev("T", "cc"), decide_row, inline,
                          prev("T", "RT") if tall_parens or not inline else None, one_equals, tall_parens)
    if tall_parens:
        trees["P"] = tall_pattern_channel(order, prev("P", "T"), prev("P", "cc"), prev("P", "RT"))
    else:
        trees["P"] = pattern_channel(order, prev("P", "T"), prev("P", "cc"), early=1 if crt else 0, flip=False)
    if inline:
        trees["K"] = colour_index_channel(order, prev("K", "T"), prev("K", "P"), prev("K", "cc"), kinds)
        for i, ch in enumerate("RGB"):
            trees[ch] = palette_channel(prev(ch, "K"), i)
    elif crt:
        trees["sl"] = scanline_counter()
        trees["C"] = class_channel(prev("C", "P"), prev("C", "cc"))
        trees["R"] = brightness_channel(prev("R", "C"), prev("R", "sl"))
        trees["G"], trees["B"] = Set(CRT["phosphor_g"]), Set(-WHITE)
    else:
        trees["R"] = colour_channel(prev("R", "P"), prev("R", "cc"))
        trees["G"] = trees["B"] = Set(0)
    tree = simplify(dispatch([trees[n] for n in names]))
    name = name or "equations" + ("_v2" if inline else "_crt" if crt else "")
    header = f"""/* {name}.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.
   {canvas[0]}x{canvas[1]}: random well-formed equations (matched parentheses, max depth {EQ_MAX_DEPTH}) from a counter
   automaton; one token per cell, {line_cells} cells per equation, one equation per decoder group and cell row;
   function names {"inline, tokens coloured by kind" if inline else "SIN/COS/TAN written downward"}{"; one = per line; parens sized by depth over two bands per line" if tall_parens else ""}.
   Glyph order: {"".join(order)!r}
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(names))}
     cc cell counter; A Rule 30; V 5 random bits; S = {'8*class + 2*depth (+1 after the =)' if one_equals else '8*class + depth'}; T glyph index; P glyph-row pattern;
     {"K colour index (0 unlit, else 1 + kind); R G B palette lookups." if inline else "RT row type (0 expression, 1-2 hanging); R draws."}
   Nodes: {count_nodes(tree)} */
Width {canvas[0]}
Height {canvas[1]}
{"" if inline else "RCT 3"}
HiddenChannel {len(names) - 3}
"""
    return header + render(tree) + "\n", names, order


# ---------------------------------------------------------------- assembling a piece


def build_piece(charset, mask, canvas, crt=False, text=None, name=None):
    """
    Pure function. (tree source text, channel names, character order) for a text piece.
    crt=True adds scanlines, a phosphor trail glow and green-amber colour. `text` replaces the
    Rule 30 glyph choice by a per-run counter so every run of drawn cells spells it, coloured in
    the two Rust tones (the jxl_rs logo).

    Examples:
        >>> src, names, order = build_piece(CHARSET_16, None, (1024, 1024))
        >>> names
        ['cc', 'A', 'V', 'P', 'R', 'G', 'B']
        >>> build_piece(CHARSET_16, None, (1024, 1024), crt=True)[1]
        ['cc', 'A', 'V', 'P', 'sl', 'C', 'R', 'G', 'B']
        >>> src.splitlines()[0]
        '/* text.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.'
        >>> build_piece(None, "lemniscate", (2048, 1024), text=LOGO_TEXT)[1:]
        (['S', 'L', 'cc', 'V', 'P', 'R', 'G', 'B'], [' ', 'J', 'X', 'L', '-', 'R', 'S', ' '])
    """
    assert text or (len(charset) & (len(charset) - 1) == 0 and " " not in charset), "power-of-two size, no blank"
    assert mask in (None, "lemniscate") and canvas[1] <= GROUP and canvas[0] <= 2 * GROUP
    assert not (text and crt)
    two_groups = canvas[0] > GROUP
    if text:
        order = [" "] + list(text)           # index = counter value; 0 is never drawn
    else:
        bits = len(charset).bit_length() - 1
        sample_y0 = GLYPH_Y0 - bits
        dist = lambda p, q: sum(a != b for a, b in zip(glyph_row_patterns(FONT[p]), glyph_row_patterns(FONT[q])))
        order = best_order(charset, dist)

    names = (["S", "L"] if mask else []) + ["cc"] + ([] if text else ["A"]) + ["V", "P"] + (["sl", "C"] if crt else []) + ["R", "G", "B"]
    prev = lambda here, there: "Prev" + (str(names.index(here) - names.index(there)) if names.index(here) - names.index(there) > 1 else "")
    trees = {}
    if mask:
        trees["S"] = lemniscate_field(canvas)
        trees["L"] = xband_field(canvas)
    trees["cc"] = cell_counter()
    gate = lemniscate_gate(canvas, prev("V", "S"), prev("V", "L")) if mask else None
    if text:
        trees["V"] = text_counter_channel(len(text), prev("V", "cc"), GLYPH_Y0 - 1, gate or (lambda update: update))
    else:
        trees["A"] = rule30(2 if two_groups else 1)
        trees["V"] = value_channel(bits, prev("V", "A"), prev("V", "cc"), sample_y0, gate)
    trees["P"] = pattern_channel(order, prev("P", "V"), prev("P", "cc"), early=1 if crt else 0)
    if text:
        tone = lambda here, k: If(prev(here, "V"), LOGO_TONE_SPLIT, Set(RUST_TAN[k] - (RUST_TAN[0] if k else 0)),
                                  Set(RUST_ORANGE[k] - (RUST_ORANGE[0] if k else 0)))
        trees["R"] = colour_channel(prev("R", "P"), prev("R", "cc"), on=tone("R", 0))
        trees["G"], trees["B"] = tone("G", 1), tone("B", 2)   # RCT 3 adds R; unlit pixels (R = 0) clamp to black
    elif crt:
        trees["sl"] = scanline_counter()
        trees["C"] = class_channel(prev("C", "P"), prev("C", "cc"))
        trees["R"] = brightness_channel(prev("R", "C"), prev("R", "sl"))
        trees["G"], trees["B"] = Set(CRT["phosphor_g"]), Set(-WHITE)   # RCT 3: G = R + 40, B = 0
    else:
        trees["R"] = colour_channel(prev("R", "P"), prev("R", "cc"))
        trees["G"] = trees["B"] = Set(0)      # RCT 3 adds R to these channels: grey = white/black

    tree = simplify(dispatch([trees[n] for n in names]))
    name = name or (f"text_{mask}" if mask else "text") + ("_crt" if crt else "")
    header = f"""/* {name}.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.
   {f"{canvas[0]}x{canvas[1]}: every run of drawn cells spells {text!r} (3x5 font) in Rust colours." if text else
     f"{canvas[0]}x{canvas[1]} grid of pseudo-random characters (3x5 font, {len(charset)} glyphs) chosen by Rule 30."}
   Character order ({"counter value" if text else "CA value"} 0..{len(order) - 1}): {"".join(order)!r}
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(names))}
     cc = {CELL_H}*xm + ym cell counter; {f"V letter counter 1..{len(text)} per run of drawn cells" if text else
     f"A Rule 30 (on = {CA_ON}); V character value from A in column xm == {SAMPLE_X}, rows ym {sample_y0}..{sample_y0 + bits - 1}"}
     (negative = blank cell); P glyph-row pattern (bit 2 = left pixel), shifted per glyph column; R glyph
     pixels; G and B are {"per-tone" if text else "0"} deltas (RCT 3 adds R).
     {f"S mask field ({mask}) and L crossing field: a cell is drawn when |L| <= w near the centre or |S| <= half-width elsewhere." if mask else ""}
     {"sl scanline counter; C glow class (3 lit, trail 2, 1); R brightness by class and scanline, G = R + 25, B = 0." if crt else ""}
   Nodes: {count_nodes(tree)} */
Width {canvas[0]}
Height {canvas[1]}
{"Orientation 4" if FLIP else ""}
RCT 3
HiddenChannel {len(names) - 3}
"""
    return header + render(tree) + "\n", names, order


def generate(charset16=False, mask=None, name=None, width=None, height=1024, crt=False, equations=False, inline=False, gap=0, row_gap=0,
             random_length=False, one_equals=False, tall_parens=False, logo=False):
    """
    Command. Writes art/trees/<name>.tree (default text.tree / text_<mask>[_crt].tree / equations.tree);
    prints node count. Default canvas: 1024x1024 without a mask, 2048x1024 with one or for equations.

    Examples:
        >>> # generate()                                                       -> text.tree
        >>> # generate(mask="lemniscate", name="text_infinity")  -> text_infinity.tree (2048x1024)
        >>> # generate(mask="lemniscate", crt=True, name="text_infinity_v2")   -> CRT look
        >>> # generate(mask="lemniscate", logo=True)                            -> jxl_rs.tree (spells JXL-RS)
        >>> # generate(equations=True)                                         -> equations.tree
        >>> # generate(equations=True, inline=True, gap=12, row_gap=1, random_length=True)  -> equations_v2.tree (two per row)
        >>> # generate(equations=True, inline=True, width=1024, name="equations_v3")  -> one equation per line
        >>> # generate(equations=True, inline=True, gap=12, row_gap=1, random_length=True, one_equals=True, tall_parens=True, name="equations_v4")
    """
    charset = CHARSET_16 if charset16 else CHARSET_32
    canvas = (width or (2 * GROUP if (mask or equations) else GROUP), height)
    if equations:
        src, names, order = build_equations(canvas, crt, inline, gap, name, row_gap, random_length, one_equals, tall_parens)
        default_name = "equations" + ("_v2" if inline else "_crt" if crt else "")
    else:
        default_name = "jxl_rs" if logo else ("text_" + mask if mask else "text") + ("_crt" if crt else "")
        src, names, order = build_piece(charset, mask, canvas, crt, LOGO_TEXT if logo else None, name or default_name)
    out = TREE_DIR / f"{name or default_name}.tree"
    out.write_text(src)
    nodes = src.split("Nodes: ")[1].split(" ")[0]
    print(f"{out.name}: {canvas[0]}x{canvas[1]}, {len(charset)} glyphs, {nodes} nodes, channels {names}, order {''.join(order)!r}")


if __name__ == "__main__":
    fire.Fire(generate)
