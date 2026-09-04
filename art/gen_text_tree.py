#!/usr/bin/env python3
"""
Generates the text pieces: a grid of pseudo-random characters from a tiny pixel font, chosen by
a Rule-30 cellular automaton, optionally masked to a shape (only whole characters are drawn).

    python3.10 art/gen_text_tree.py                                  # text.tree (32 glyphs)
    python3.10 art/gen_text_tree.py --charset16 --mask rings         # text_rings.tree
    python3.10 art/gen_text_tree.py --charset16 --mask hexagons --name text_infinity

Trees are built from a tiny DSL (If / Leaf tuples rendered to jxl_from_tree syntax) so every
repeated structure (counters, bit accumulation, run-length lookups) is one function.

Cell layout: CELL_W x CELL_H px cells; one hidden counter holds cc = CELL_H * xm + ym, so both
cell coordinates are thresholds on one value (xm > k <=> cc > CELL_H*k + CELL_H-1; ym > k inside
a known xm column <=> cc > CELL_H*xm + k). The first cell row starts at Y_OFFSET so even the
first line samples the automaton 60 steps below the seed row: a Weyl seed sampled every 16 px
always has a near-return (only ~3% of bits differ) at some distance, and fewer steps left
visible echoes in the first line. Glyph pixels are PIXEL x PIXEL
at (GLYPH_X0, GLYPH_Y0) in the cell. The character value V is assembled in column xm == SAMPLE_X
from consecutive Rule-30 rows (Wolfram's Rule-30 PRNG), one bit per row; value 0 is the blank.

Glyph rows are stored as 3-bit patterns in hidden channel P (one run-length tree per glyph row,
computed at xm == GLYPH_X0, copied right); at the start of glyph columns 1 and 2 the consumed
high bit is subtracted so the colour channel lights a pixel with a single threshold.
Character order is optimised (random restarts + 2-opt) to minimise pattern changes.
"""
import random
from pathlib import Path

import fire

TREE_DIR = Path(__file__).resolve().parent / "trees"

FONT = {  # 3x5 glyphs, 5 rows of 3 bits
    " ": ["000", "000", "000", "000", "000"], "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"], "C": ["011", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "110"], "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"], "G": ["011", "100", "101", "101", "011"],
    "H": ["101", "101", "111", "101", "101"], "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "010"], "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"], "M": ["101", "111", "111", "101", "101"],
    "N": ["110", "101", "101", "101", "101"], "O": ["010", "101", "101", "101", "010"],
    "P": ["110", "101", "110", "100", "100"], "Q": ["010", "101", "101", "010", "001"],
    "R": ["110", "101", "110", "101", "101"], "S": ["011", "100", "010", "001", "110"],
    "T": ["111", "010", "010", "010", "010"], "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"], "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"], "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"], "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"], "!": ["010", "010", "010", "000", "010"],
    "?": ["110", "001", "010", "000", "010"], ".": ["000", "000", "000", "000", "010"],
}
CHARSET_32 = " " + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "01!?."
CHARSET_16 = " ABDEFHIKNOPRTUV"        # blank + 15 letters chosen (local search) for the fewest glyph-row changes
GLYPH_ROWS, GLYPH_COLS, PIXEL = 5, 3, 4
GLYPH_X0, GLYPH_Y0 = 2, 12
SAMPLE_X = 1                           # column where V is assembled (x = 1 + 16k, past the CA's edge column)
CELL_W, CELL_H = 16, 32
Y_OFFSET = 20                          # first cell row (bands repeat every CELL_H rows from here)
BLANK_BANDS = 1                        # leading bands left blank so the first drawn line is 60 Rule-30 steps below the seed
FLIP = True                            # Orientation 4 (vertical flip): the first decoded line, which has had the
                                       # fewest Rule-30 steps since the seed row, is displayed at the bottom
CA_ON, CA_THRESHOLD = 1024, 511        # Rule-30 "on" value and the test "> 511" used everywhere
WEYL_STEP, WEYL_MOD = 633, 1024        # seed sequences v = (v + 633) mod 1024 along row 0 and down column 0
WHITE = 255
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


def runs_tree(values, leaf_of):
    """
    Pure function. Balanced tree over an index (property "Prev") selecting leaf_of(values[i]);
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
        return If("Prev", b, build(b + 1, hi), build(lo, b))
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


def rule30():
    """
    Pure function. Rule 30 (new = NW xor (N or NE)) with on = CA_ON = 1024. Row 0 and column 0
    are Weyl sequences (v + 633) mod 1024 (period > width, so nothing repeats; the column one
    injects entropy at the left edge because Rule 30 only moves information rightward well and
    a fixed edge goes periodic). Their values 0..1023 feed the rule directly through the same
    "> 511" thresholds. NW and NE are recovered from the NW-N and N-NE properties given N.

    Examples:
        >>> render(rule30()).splitlines()[0]
        'if y > 0'
    """
    on, off = Set(CA_ON), Set(0)
    interior = If("N", CA_THRESHOLD,
                  If("NW-N", -CA_ON + CA_THRESHOLD, off, on),
                  If("NW-N", CA_THRESHOLD, If("N-NE", -CA_THRESHOLD - 1, on, off), If("N-NE", -CA_THRESHOLD - 1, off, on)))
    weyl = lambda pred: If(pred, WEYL_MOD - WEYL_STEP - 1, Leaf(pred, WEYL_STEP - WEYL_MOD), Leaf(pred, WEYL_STEP))
    return If("y", 0, If("x", 0, interior, weyl("N")), weyl("W"))


def value_channel(bits, prev_ca, prev_cc, sample_y0, gate=None):
    """
    Pure function. Character-value channel V: in column xm == SAMPLE_X it reads `bits` consecutive
    Rule-30 rows (ym = sample_y0 ...) as bits, most significant first; copies W to the right and
    N downward. `gate(update)` wraps the last bit's update (masks: Set 0 when outside). The first
    BLANK_BANDS bands stay 0 (blank).

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
    in_sample_column = If("y", Y_OFFSET + BLANK_BANDS * CELL_H - 1, in_sample_column, Set(0))
    return chain(prev_cc, [(cc_xm_gt(SAMPLE_X), Leaf("W", 0)), (cc_xm_gt(SAMPLE_X - 1), in_sample_column)], Leaf("N", 0))


# ---------------------------------------------------------------- glyph lookup


def glyph_row_patterns(rows):
    """
    Pure function. 5 rows of '010' strings -> tuple of 5 ints 0..7 (left pixel = bit 2).

    Examples:
        >>> glyph_row_patterns(["010", "101", "111", "101", "101"])
        (2, 5, 7, 5, 5)
    """
    return tuple(int(row, 2) for row in rows)


def decoded_glyph(ch):
    """
    Query (reads FLIP). Glyph rows in decode order: reversed when the image is displayed flipped.

    Examples:
        >>> decoded_glyph("L")[0] if FLIP else decoded_glyph("L")[-1]
        '111'
    """
    return FONT[ch][::-1] if FLIP else FONT[ch]


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
    Pure function. Improves a path (fixed first element) by segment reversals until no gain.

    Examples:
        >>> two_opt(["a", "c", "b"], lambda p, q: abs(ord(p) - ord(q)))
        ['a', 'b', 'c']
    """
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if path_cost(candidate, dist) < path_cost(order, dist):
                    order, improved = candidate, True
    return order


def best_order(items, start, dist, restarts=OPTIMIZER_RESTARTS, seed=0):
    """
    Near-pure function (uses a seeded RNG). Best of `restarts` random paths from `start`,
    each improved by 2-opt.

    Examples:
        >>> best_order("abc", "a", lambda p, q: abs(ord(p) - ord(q)), restarts=3)
        ['a', 'b', 'c']
    """
    rng = random.Random(seed)
    rest = [i for i in items if i != start]
    best = None
    for _ in range(restarts):
        rng.shuffle(rest)
        order = two_opt([start] + rest, dist)
        if best is None or path_cost(order, dist) < path_cost(best, dist):
            best = order
    return best


def pattern_channel(order, prev_cc):
    """
    Pure function. Hidden channel P holding the current glyph row's 3-bit pattern: computed at
    xm == GLYPH_X0 from the value (Prev) per glyph row (0 in the cell's margin rows), copied right
    with W; the consumed high bit is removed at the start of glyph columns 1 and 2.

    Examples:
        >>> render(pattern_channel(" A", "Prev2")).splitlines()[0]
        'if Prev2 > 95'
    """
    patterns = {ch: glyph_row_patterns(decoded_glyph(ch)) for ch in order}
    per_row = [runs_tree([patterns[ch][r] for ch in order], Set) for r in range(GLYPH_ROWS)]
    row_cases = [(cc_ym_gt(GLYPH_X0, GLYPH_Y0 + r * PIXEL - 1), per_row[r]) for r in range(GLYPH_ROWS - 1, -1, -1)]
    at_glyph_start = chain(prev_cc, row_cases, Set(0))
    x1, x2 = GLYPH_X0 + PIXEL, GLYPH_X0 + 2 * PIXEL
    across = chain(prev_cc, [(cc_xm_gt(x2), Leaf("W", 0)), (cc_xm_gt(x2 - 1), If("W", 1, Leaf("W", -2), Leaf("W", 0))),
                             (cc_xm_gt(x1), Leaf("W", 0)), (cc_xm_gt(x1 - 1), If("W", 3, Leaf("W", -4), Leaf("W", 0)))], Leaf("W", 0))
    return chain(prev_cc, [(cc_xm_gt(GLYPH_X0), across), (cc_xm_gt(GLYPH_X0 - 1), at_glyph_start)], Leaf("N", 0))


def colour_channel(prev_pattern, prev_cc):
    """
    Pure function. R channel: glyph column c is lit when the (already shifted) pattern exceeds
    3, 1, 0; outside the glyph columns the pixel is black.

    Examples:
        >>> render(colour_channel("Prev", "Prev2")).splitlines()[0]
        'if Prev2 > 447'
    """
    lit = lambda split: If(prev_pattern, split, Set(WHITE), Set(0))
    x0, x1, x2, x3 = (GLYPH_X0 + c * PIXEL for c in range(4))
    return chain(prev_cc, [(cc_xm_gt(x3 - 1), Set(0)), (cc_xm_gt(x2 - 1), lit(0)), (cc_xm_gt(x1 - 1), lit(1)),
                           (cc_xm_gt(x0 - 1), lit(3))], Set(0))


# ---------------------------------------------------------------- masks (whole cells only)
# A cell is blank unless a field S lies in a band (lo, hi] at the cell's sample point. Channels
# cannot be added, so S must be separable, S = f(x) + g(y), grown with W +slope along rows and
# N +slope down column 0. "rings": f, g quadratic (two round loops) with piecewise-constant chord
# slopes, exact at the piece boundaries. "diamonds": f = ||x - cx| - d|, g = |y - cy| built with
# +-1 steps that flip at fixed x / y thresholds: two rhombus rings, one 11-node channel.
# "hexagons": like diamonds but |y - cy| steps by 2 beyond a knee, giving wider flatter loops.
MASK_CENTER = (512, 512)     # sample-point coordinates; the glyph centre sits ~9 px right and below
RINGS = dict(offset=192, r_in=150, r_out=250, scale=8, x_breaks=list(range(0, 1025, 128)), y_breaks=[0, 256, 384, 512, 640, 768, 1024])
DIAMONDS = dict(offset=224, r_in=150, r_out=230, knee_x=None, knee_y=None)
HEXAGONS = dict(offset=208, r_in=150, r_out=250, knee_x=None, knee_y=96)   # wide hexagonal loops that overlap at the centre


def chord_slopes(profile, breaks):
    """
    Pure function. (split, integer slope) per piece (breaks[k], breaks[k+1]]: the chord slope,
    so the accumulated value is exact at every break.

    Examples:
        >>> chord_slopes(lambda t: t * t, [0, 4, 8])
        [(-1, 4), (3, 12)]
    """
    return [(a - 1, round((profile(b) - profile(a)) / (b - a))) for a, b in zip(breaks, breaks[1:])]


def slope_chain(prop, pred, pieces):
    """
    Pure function. `pred + slope` selected by prop; pieces are (split, slope) in increasing order.

    Examples:
        >>> print(render(slope_chain("x", "W", [(-1, 4), (3, 12)])))
        if x > 3
          - W 12
          - W 4
    """
    return chain(prop, [(split, Leaf(pred, s)) for split, s in reversed(pieces[1:])], Leaf(pred, pieces[0][1]))


def rings_channel():
    """
    Pure function. Field channel for two round loops: f = (|x-cx| - d)^2 / scale, g = (y-cy)^2 / scale.

    Examples:
        >>> render(rings_channel()).splitlines()[0]
        'if x > 0'
    """
    cx, cy = MASK_CENTER
    f = lambda x: (abs(x - cx) - RINGS["offset"]) ** 2 / RINGS["scale"]
    g = lambda y: (y - cy) ** 2 / RINGS["scale"]
    xs = slope_chain("x", "W", chord_slopes(f, RINGS["x_breaks"]))
    ys = slope_chain("y", "N", chord_slopes(g, RINGS["y_breaks"]))
    return If("x", 0, xs, If("y", 0, ys, Set(round(f(0) + g(0)))))


def rings_band():
    """
    Pure function. (lo, hi] band of the rings field.

    Examples:
        >>> rings_band()
        (2812, 7812)
    """
    return RINGS["r_in"] ** 2 // RINGS["scale"], RINGS["r_out"] ** 2 // RINGS["scale"]


def polygon_channel(spec):
    """
    Pure function. Field channel S = f(|x - cx| - d) + g(|y - cy|) - mid, where f and g are
    piecewise linear with slope 1 up to a knee and slope 2 beyond (knee None: plain |.|), built
    with +-1/+-2 steps whose sign flips at x = cx-d, cx, cx+d and at y = cy. "mid" centres the
    band on 0 so membership is a single |S| test.

    Examples:
        >>> render(polygon_channel(DIAMONDS)).splitlines()[0]
        'if x > 0'
    """
    cx, cy = MASK_CENTER
    d, kx, ky = spec["offset"], spec["knee_x"], spec["knee_y"]
    def ramp(t, knee):
        return t if knee is None or t <= knee else knee + 2 * (t - knee)
    def slope_cases(centre, knee):     # step per pixel while moving right, by distance from `centre`
        if knee is None:
            return [(centre, 1)], -1
        return [(centre + knee, 2), (centre, 1), (centre - knee, -1)], -2
    right_loop, _ = slope_cases(cx + d, kx)
    left_loop, default = slope_cases(cx - d, kx)   # `default` = the step far to the left of a centre
    xs = chain("x", [(s, Leaf("W", v)) for s, v in right_loop] + [(cx, Leaf("W", default))] + [(s, Leaf("W", v)) for s, v in left_loop], Leaf("W", default))
    y_cases, y_default = slope_cases(cy, ky)
    ys = chain("y", [(s, Leaf("N", v)) for s, v in y_cases], Leaf("N", y_default))
    mid = (spec["r_in"] + spec["r_out"]) // 2
    return If("x", 0, xs, If("y", 0, ys, Set(ramp(abs(0 - (cx - d)), kx) + ramp(cy, ky) - mid)))


def polygon_band(spec):
    """
    Pure function. Half-width of the centred band: inside when |S| <= half-width.

    Examples:
        >>> polygon_band(DIAMONDS)
        40
    """
    return (spec["r_out"] - spec["r_in"]) // 2


def diamonds_channel():
    """Pure function. Two rhombus rings (see polygon_channel)."""
    return polygon_channel(DIAMONDS)


def diamonds_band():
    """Pure function. Band half-width of the diamonds mask."""
    return polygon_band(DIAMONDS)


def hexagons_channel():
    """Pure function. Two hexagonal rings (|y| steps grow to 2 beyond the knee: flatter, wider loops)."""
    return polygon_channel(HEXAGONS)


def hexagons_band():
    """Pure function. Band half-width of the hexagons mask."""
    return polygon_band(HEXAGONS)


MASKS = {"rings": (rings_channel, rings_band), "diamonds": (diamonds_channel, diamonds_band),
         "hexagons": (hexagons_channel, hexagons_band)}


def prev_abs(prev):
    """
    Pure function. The absolute-value property of a Prev reference.

    Examples:
        >>> prev_abs("Prev"), prev_abs("Prev3")
        ('PrevAbs', 'Prev3Abs')
    """
    return prev + "Abs"


# ---------------------------------------------------------------- assembling a piece


def build_piece(charset, mask):
    """
    Pure function. (tree source text, channel names, character order) for a text piece.

    Examples:
        >>> src, names, order = build_piece(CHARSET_16, None)
        >>> names
        ['cc', 'A', 'V', 'P', 'R', 'G', 'B']
        >>> src.splitlines()[0]
        '/* text.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.'
    """
    assert charset[0] == " " and len(charset) & (len(charset) - 1) == 0, "blank first, power-of-two size"
    assert mask in (None, *MASKS)
    bits = len(charset).bit_length() - 1
    sample_y0 = GLYPH_Y0 - bits
    dist = lambda p, q: sum(a != b for a, b in zip(glyph_row_patterns(FONT[p]), glyph_row_patterns(FONT[q])))
    order = best_order(list(charset), " ", dist)

    names = (["S"] if mask else []) + ["cc", "A", "V", "P", "R", "G", "B"]
    prev = lambda here, there: "Prev" + (str(names.index(here) - names.index(there)) if names.index(here) - names.index(there) > 1 else "")
    trees = {}
    if mask:
        field, band = MASKS[mask]
        trees["S"] = field()
    trees["cc"] = cell_counter()
    trees["A"] = rule30()
    gate = None
    if mask == "rings":
        lo, hi = band()
        gate = lambda update: If(prev("V", "S"), hi, Set(0), If(prev("V", "S"), lo, update, Set(0)))
    elif mask:
        gate = lambda update: If(prev_abs(prev("V", "S")), band(), Set(0), update)
    trees["V"] = value_channel(bits, prev("V", "A"), prev("V", "cc"), sample_y0, gate)
    trees["P"] = pattern_channel(order, prev("P", "cc"))
    trees["R"] = colour_channel(prev("R", "P"), prev("R", "cc"))
    trees["G"] = trees["B"] = Set(0)          # RCT 3 adds R to these channels: grey = white/black

    tree = dispatch([trees[n] for n in names])
    name = f"text_{mask}" if mask else "text"
    header = f"""/* {name}.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.
   Grid of pseudo-random characters (3x5 font, {len(charset)} glyphs) chosen by Rule 30.
   Character order (CA value 0..{len(charset) - 1}): {"".join(order)!r}
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(names))}
     cc = {CELL_H}*xm + ym cell counter; A Rule 30 (on = {CA_ON}); V character value from A in column
     xm == {SAMPLE_X}, rows ym {sample_y0}..{sample_y0 + bits - 1}; P glyph-row pattern (bit 2 = left pixel), shifted per
     glyph column; R glyph pixels; G and B are 0 deltas (RCT 3 adds R to them).
     {f"S mask field ({mask}); a cell is blank unless the field is inside its band at the cell's sample point." if mask else ""}
   Nodes: {count_nodes(tree)} */
Width 1024
Height 1024
{"Orientation 4" if FLIP else ""}
RCT 3
HiddenChannel {len(names) - 3}
"""
    return header + render(tree) + "\n", names, order


def generate(charset16=False, mask=None, name=None):
    """
    Command. Writes art/trees/<name>.tree (default text.tree / text_<mask>.tree); prints node count.

    Examples:
        >>> # generate()                                  -> text.tree (32 glyphs)
        >>> # generate(charset16=True, mask="diamonds", name="text_infinity")
    """
    charset = CHARSET_16 if charset16 else CHARSET_32
    src, names, order = build_piece(charset, mask)
    out = TREE_DIR / f"{name or ('text_' + mask if mask else 'text')}.tree"
    out.write_text(src)
    nodes = src.split("Nodes: ")[1].split(" ")[0]
    print(f"{out.name}: {len(charset)} glyphs, {nodes} nodes, channels {names}, order {''.join(order)!r}")


if __name__ == "__main__":
    fire.Fire(generate)
