#!/usr/bin/env python3
"""
Generates the text pieces: a grid of pseudo-random characters from a tiny pixel font, chosen by
a Rule-30 cellular automaton, optionally masked to a shape (only whole characters are drawn).

    python3.10 art/gen_text_tree.py                                  # text.tree (32 glyphs, 1024x1024)
    python3.10 art/gen_text_tree.py --charset16 --mask hexagons --name text_infinity   # 2048x1024

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
import random
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
}
CHARSET_32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "012!?."
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
GROUP, FIRST_GROUP = 1024, 21          # JPEG XL group width and the id of the first group
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


def rule30(two_groups):
    """
    Pure function. Rule 30 (new = NW xor (N or NE)) with on = CA_ON = 1024. Row 0 and column 0
    are Weyl sequences (period > width, so nothing repeats; the column one injects entropy at
    the left edge because Rule 30 only moves information rightward well and a fixed edge goes
    periodic). Their values 0..1023 feed the rule directly through the same "> 511" thresholds.
    NW and NE are recovered from the NW-N and N-NE properties given N. With two groups the
    second group's seed row uses a different step so the halves are not copies.

    Examples:
        >>> render(rule30(False)).splitlines()[0]
        'if y > 0'
    """
    on, off = Set(CA_ON), Set(0)
    interior = If("N", CA_THRESHOLD,
                  If("NW-N", -CA_ON + CA_THRESHOLD, off, on),
                  If("NW-N", CA_THRESHOLD, If("N-NE", -CA_THRESHOLD - 1, on, off), If("N-NE", -CA_THRESHOLD - 1, off, on)))
    seed_row = per_group(two_groups, weyl("W", WEYL_STEPS[0]), weyl("W", WEYL_STEPS[1]))
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


def pattern_channel(order, prev_value, prev_cc):
    """
    Pure function. Hidden channel P holding the current glyph row's 3-bit pattern: computed at
    xm == GLYPH_X0 from the value per glyph row (0 in the cell's margin rows and for blank cells,
    i.e. negative values), copied right with W; the consumed high bit is removed at the start of
    glyph columns 1 and 2.

    Examples:
        >>> render(pattern_channel("AB", "Prev", "Prev2")).splitlines()[0]
        'if Prev2 > 95'
    """
    patterns = {ch: glyph_row_patterns(decoded_glyph(ch)) for ch in order}
    per_row = [runs_tree([patterns[ch][r] for ch in order], Set) for r in range(GLYPH_ROWS)]
    row_cases = [(cc_ym_gt(GLYPH_X0, GLYPH_Y0 + r * PIXEL - 1), per_row[r]) for r in range(GLYPH_ROWS - 1, -1, -1)]
    at_glyph_start = If(prev_value, -1, chain(prev_cc, row_cases, Set(0)), Set(0))
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


# ---------------------------------------------------------------- mask (whole cells only)
# A cell is blank unless a field S is inside a band at the cell's sample point. Channels cannot
# be added, so S must be separable: S = ||x - cx| - d| + |y - cy| - mid (Manhattan distance to
# the nearer loop centre, centred on the band so membership is one |S| <= half-width test via
# the PrevAbs property), built with +-1 steps whose sign flips at x = cx-d, cx, cx+d and y = cy.
# Two diamond rings result; clipping |y - cy| <= clip with plain y thresholds flattens their
# tops and bottoms into hexagons. Geometry is a fraction of the canvas.
HEXAGONS = dict(offset=0.23, r_in=0.146, r_out=0.264, clip=0.41)   # x fractions of width, clip of height


def hexagon_geometry(canvas):
    """
    Pure function. (cx, cy, d, r_in, r_out, clip) in pixels for a canvas.

    Examples:
        >>> hexagon_geometry((2048, 1024))
        (1024, 512, 471, 299, 541, 420)
    """
    w, h = canvas
    return (w // 2, h // 2, round(HEXAGONS["offset"] * w), round(HEXAGONS["r_in"] * w),
            round(HEXAGONS["r_out"] * w), round(HEXAGONS["clip"] * h))


def hexagon_field(canvas):
    """
    Pure function. Field channel S for the hexagon mask (see above); with two groups the loop
    centre sits at local x = GROUP - d in the left group and x = d in the right group.

    Examples:
        >>> render(hexagon_field((1024, 1024))).splitlines()[0]
        'if x > 0'
    """
    cx, cy, d, r_in, r_out, _ = hexagon_geometry(canvas)
    mid = (r_in + r_out) // 2
    two = canvas[0] > GROUP
    step = lambda centre: If("x", centre - 1, Leaf("W", 1), Leaf("W", -1))
    if two:
        xs = per_group(True, step(GROUP - d), step(d))
        base = per_group(True, Set(GROUP - d + cy - mid), Set(d + cy - mid))
    else:
        xs = chain("x", [(cx + d, Leaf("W", 1)), (cx, Leaf("W", -1)), (cx - d, Leaf("W", 1))], Leaf("W", -1))
        base = Set(cx - d + cy - mid)
    ys = If("y", cy, Leaf("N", 1), Leaf("N", -1))
    return If("x", 0, xs, If("y", 0, ys, base))


def hexagon_gate(canvas, prev_field):
    """
    Pure function. gate(update) for value_channel: BLANK unless |S| <= half-width and |y - cy| <= clip.

    Examples:
        >>> render(hexagon_gate((2048, 1024), "Prev")(Set(1))).splitlines()[0]
        'if y > 932'
    """
    _, cy, _, r_in, r_out, clip = hexagon_geometry(canvas)
    half_width = (r_out - r_in) // 2
    return lambda update: If("y", cy + clip, Set(BLANK),
                             If("y", cy - clip - 1, If(prev_field + "Abs", half_width, Set(BLANK), update), Set(BLANK)))


# ---------------------------------------------------------------- assembling a piece


def build_piece(charset, mask, canvas):
    """
    Pure function. (tree source text, channel names, character order) for a text piece.

    Examples:
        >>> src, names, order = build_piece(CHARSET_16, None, (1024, 1024))
        >>> names
        ['cc', 'A', 'V', 'P', 'R', 'G', 'B']
        >>> src.splitlines()[0]
        '/* text.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.'
    """
    assert len(charset) & (len(charset) - 1) == 0 and " " not in charset, "power-of-two size, no blank"
    assert mask in (None, "hexagons") and canvas[1] <= GROUP and canvas[0] <= 2 * GROUP
    bits = len(charset).bit_length() - 1
    sample_y0 = GLYPH_Y0 - bits
    two_groups = canvas[0] > GROUP
    dist = lambda p, q: sum(a != b for a, b in zip(glyph_row_patterns(FONT[p]), glyph_row_patterns(FONT[q])))
    order = best_order(charset, dist)

    names = (["S"] if mask else []) + ["cc", "A", "V", "P", "R", "G", "B"]
    prev = lambda here, there: "Prev" + (str(names.index(here) - names.index(there)) if names.index(here) - names.index(there) > 1 else "")
    trees = {}
    if mask:
        trees["S"] = hexagon_field(canvas)
    trees["cc"] = cell_counter()
    trees["A"] = rule30(two_groups)
    gate = hexagon_gate(canvas, prev("V", "S")) if mask else None
    trees["V"] = value_channel(bits, prev("V", "A"), prev("V", "cc"), sample_y0, gate)
    trees["P"] = pattern_channel(order, prev("P", "V"), prev("P", "cc"))
    trees["R"] = colour_channel(prev("R", "P"), prev("R", "cc"))
    trees["G"] = trees["B"] = Set(0)          # RCT 3 adds R to these channels: grey = white/black

    tree = dispatch([trees[n] for n in names])
    name = f"text_{mask}" if mask else "text"
    header = f"""/* {name}.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.
   {canvas[0]}x{canvas[1]} grid of pseudo-random characters (3x5 font, {len(charset)} glyphs) chosen by Rule 30.
   Character order (CA value 0..{len(charset) - 1}): {"".join(order)!r}
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(names))}
     cc = {CELL_H}*xm + ym cell counter; A Rule 30 (on = {CA_ON}); V character value from A in column
     xm == {SAMPLE_X}, rows ym {sample_y0}..{sample_y0 + bits - 1} (negative = blank cell); P glyph-row pattern (bit 2 =
     left pixel), shifted per glyph column; R glyph pixels; G and B are 0 deltas (RCT 3 adds R).
     {f"S mask field ({mask}); a cell is blank unless |S| <= half-width and |y - cy| <= clip at its sample point." if mask else ""}
   Nodes: {count_nodes(tree)} */
Width {canvas[0]}
Height {canvas[1]}
{"Orientation 4" if FLIP else ""}
RCT 3
HiddenChannel {len(names) - 3}
"""
    return header + render(tree) + "\n", names, order


def generate(charset16=False, mask=None, name=None, width=None, height=1024):
    """
    Command. Writes art/trees/<name>.tree (default text.tree / text_<mask>.tree); prints node count.
    Default canvas: 1024x1024 without a mask, 2048x1024 with one.

    Examples:
        >>> # generate()                                                       -> text.tree
        >>> # generate(charset16=True, mask="hexagons", name="text_infinity")  -> text_infinity.tree
    """
    charset = CHARSET_16 if charset16 else CHARSET_32
    canvas = (width or (2 * GROUP if mask else GROUP), height)
    src, names, order = build_piece(charset, mask, canvas)
    out = TREE_DIR / f"{name or ('text_' + mask if mask else 'text')}.tree"
    out.write_text(src)
    nodes = src.split("Nodes: ")[1].split(" ")[0]
    print(f"{out.name}: {canvas[0]}x{canvas[1]}, {len(charset)} glyphs, {nodes} nodes, channels {names}, order {''.join(order)!r}")


if __name__ == "__main__":
    fire.Fire(generate)
