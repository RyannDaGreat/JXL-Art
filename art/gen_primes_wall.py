"""
Prime walls: a dense grid of consecutive integers (1 px font) with the primes bright and the
composites faint. Primality is computed inside the tree: one counter channel per prime divisor
d <= sqrt(N_MAX) holds n mod d, stepping at every number start along the row, by PER_ROW at every
band start down column 0, and by GROUP_SIZE per group index in the top margin (a countdown from the
group id); a number is composite when some counter is 0. Three presets (WALLS):

    python3.10 art/gen_primes_wall.py                        # primes_wall     1024^2,  5000 numbers
    python3.10 art/gen_primes_wall.py --wall primes_wall_v2  # 8 px rows,      6125 numbers, < 1 KB
    python3.10 art/gen_primes_wall.py --wall primes_wall_4k  # 4096x2048, 8 groups, 33600 numbers, 2 KB budget

The trivial hit n = d happens only for n <= max divisor, i.e. in the first SMALL_ROWS rows of group
0, where every composite has a factor among the SMALL divisors, so only those counters are tested
there and the first cells (n <= max small divisor) are spelled out. A channel can look back only
19 channels, so the counters are split into blocks [Q, 19 counters, M] (one group) or
[Q, E, 16 counters, M] (several groups): Q a number-pixel counter (its column-0 value in the
margin is the group countdown), E the small-test region flag,
M the running composite flag (each M also reads the previous block's M).
"""
import importlib.util
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
If, Leaf, Set, chain, runs_tree = gen.If, gen.Leaf, gen.Set, gen.chain, gen.runs_tree
gen.apply_geometry(1)                    # 4 px digit cells; build() sets the band height per wall

FIRST_Y = 20                             # top margin: the first band starts here; rows 1..19 apply the group countdown
BLOCK_SINGLE, BLOCK_MULTI = 19, 16       # counters per block: [Q, counters, M] or, with a region flag E, [Q, E, counters, M] (19 = the Prev reach)
PRIME_COLOUR, COMPOSITE_COLOUR = (255, 200, 60), (26, 26, 26)   # composites faint: the wall stays dense
DIGIT_FONT = {"0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
              "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
              "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
              "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "001", "001", "001"],
              "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"]}
ORDER = list("0123456789")


def primes_below(n):
    """
    Pure function. Primes <= n.

    Examples:
        >>> primes_below(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    return [d for d in range(2, n + 1) if all(d % q for q in range(2, int(d ** 0.5) + 1))]


@dataclass(frozen=True)
class Wall:
    """A wall preset: canvas, numbers per group row, digit cells per number, band height, byte budget."""
    name: str
    canvas: tuple
    per_row: int
    digits: int
    cell_h: int
    limit: int = 1024

    @property
    def cells(self):
        return self.digits + 1               # digit cells plus a blank one

    @property
    def groups(self):
        return (self.canvas[0] // gen.GROUP) * (self.canvas[1] // gen.GROUP)

    @property
    def rows(self):
        return (gen.GROUP - FIRST_Y) // self.cell_h

    @property
    def group_size(self):
        return self.rows * self.per_row

    @property
    def n_max(self):
        return self.group_size * self.groups

    @property
    def divisors(self):
        return primes_below(int(self.n_max ** 0.5))

    @property
    def small_rows(self):
        """Rows of group 0 that can contain a trivial hit n = d."""
        return -(-self.divisors[-1] // self.per_row)

    @property
    def small(self):
        """Divisors that suffice for n <= small_rows * per_row."""
        return primes_below(int((self.small_rows * self.per_row) ** 0.5))

    @property
    def cell_px(self):
        return self.cells * gen.CELL_W

    @property
    def block(self):
        return BLOCK_SINGLE if self.groups == 1 else BLOCK_MULTI

    def blocks(self):
        """Divisors per counter block, in order."""
        return [self.divisors[k:k + self.block] for k in range(0, len(self.divisors), self.block)]


WALLS = {
    "primes_wall": Wall("primes_wall", (1024, 1024), 50, 4, 10),
    "primes_wall_v2": Wall("primes_wall_v2", (1024, 1024), 49, 4, 8),
    "primes_wall_4k": Wall("primes_wall_4k", (4096, 2048), 42, 5, 10, limit=2048),
}


def layout(w):
    """
    Pure function. Channel names of a wall: counter blocks, then the digit and glyph channels.

    Examples:
        >>> names = layout(WALLS["primes_wall"]); names[:3], names[18:21], names[-6:]
        (['Q1', 'C2', 'C3'], ['C61', 'C67', 'M1'], ['Z', 'D', 'P', 'R', 'G', 'B'])
    """
    names = []
    for k, block in enumerate(w.blocks()):
        names += [f"Q{k + 1}"] + ([f"E{k + 1}"] if w.groups > 1 else []) + [f"C{d}" for d in block] + [f"M{k + 1}"]
    return names + ["cc", "Dp"] + ["o", "t", "h", "th", "tk"][:w.digits] + ["Z", "D", "P", "R", "G", "B"]


def prev_of(names):
    """Pure function. prev(here, there) -> the Prev property naming channel `there` from channel `here` (<= 19 apart)."""
    def prev(here, there):
        k = names.index(here) - names.index(there)
        assert 1 <= k <= 19, (here, there, k)
        return "Prev" + (str(k) if k > 1 else "")
    return prev


def number_counter(period_px, countdown):
    """
    Pure function. Counter channel: CELL_H * (x mod period_px) + ym along the grid rows (Q with
    the number width, or the cell counter with CELL_W), so a period's first pixel column has a
    value <= CELL_H - 1. In the margin rows (y < FIRST_Y) column 0 holds `countdown` instead: the
    group index at y == 0 counting down per row (the first Q), or a 0/1 flag read from an earlier
    counter (later blocks and the cell counter of multi-group walls).

    Examples:
        >>> gen.render(number_counter(20, Set(0))).splitlines()[1]
        '  if W > 189'
    """
    last = gen.CELL_H * (period_px - 1)
    along_row = If("W", last - 1, Leaf("W", -last), Leaf("W", gen.CELL_H))
    band_rows = If("N", gen.CELL_H - 2, Leaf("N", -(gen.CELL_H - 1)), Leaf("N", 1))
    column = If("y", FIRST_Y, band_rows, If("y", FIRST_Y - 1, Set(0), countdown))
    return If("x", 0, along_row, column)


def group_countdown(w):
    """
    Pure function. Column-0 margin value of the first Q: the group index G (from g, ids in raster
    order) plus one at y == 0, minus one per row down to 0, so it is positive on rows 1 .. G and
    the group step runs G times.

    Examples:
        >>> gen.render(group_countdown(WALLS["primes_wall_4k"])).splitlines()[:2]
        ['if y > 0', '  if N > 0']
    """
    index = chain("g", [(gen.FIRST_GROUP + k - 1, Set(k + 1)) for k in range(w.groups - 1, 0, -1)], Set(1))
    return If("y", 0, If("N", 0, Leaf("N", -1), Set(0)), index)


def region_flag(w):
    """
    Pure function. E channel: 1 in the small-test region (group 0, the first small_rows bands).

    Examples:
        >>> gen.render(region_flag(WALLS["primes_wall"])).splitlines()[0]
        'if g > 21'
    """
    return If("g", gen.FIRST_GROUP, Set(0), If("y", FIRST_Y + w.small_rows * gen.CELL_H - 1, Set(0), Set(1)))


def counter(w, d, prev_q):
    """
    Pure function. Counter channel for divisor d: n mod d (1 for n = 1). Steps: +1 at number
    starts (Q < CELL_H) along the row; +PER_ROW at band starts (Q == 0) down column 0; +GROUP_SIZE
    per margin row while the countdown (Q in column 0, y < FIRST_Y) is positive.

    Examples:
        >>> print(gen.render(counter(WALLS["primes_wall"], 3, "Prev")).splitlines()[1])
          if Prev > 9
    """
    step = lambda pred, k: (If(pred, d - 1 - k % d, Leaf(pred, k % d - d), Leaf(pred, k % d)) if k % d else Leaf(pred, 0))
    margin = If("y", 0, If(prev_q, 0, step("N", w.group_size), Leaf("N", 0)), Set(1)) if w.groups > 1 else Set(1)
    column = If("y", FIRST_Y, If(prev_q, 0, Leaf("N", 0), step("N", w.per_row)), margin)
    return If("x", 0, If(prev_q, gen.CELL_H - 1, Leaf("W", 0), step("W", 1)), column)


def or_flag(w, prev_counters, prev_e, prev_m_before, block):
    """
    Pure function. M channel of a block: 1 when one of its counters is 0 (|counter| == 0, one
    split each) or the previous block's M is 1. In the small-test region (flag `prev_e`, or a y
    test when there is one group) only the small divisors count and the cells n <= max(small) of
    row 0 are spelled out.

    Examples:
        >>> gen.render(or_flag(WALLS["primes_wall"], ["Prev", "Prev2"], None, None, 0)).splitlines()[0]
        'if y > 39'
    """
    base = Set(0) if prev_m_before is None else If(prev_m_before, 0, Set(1), Set(0))
    any_zero = lambda ps: reduce(lambda t, p: If(p + "Abs", 0, t, Set(1)), reversed(ps), base)
    full = any_zero(prev_counters)
    small = [p for p, d in zip(prev_counters, w.blocks()[block]) if d in w.small]
    if block == 0:
        head_n = w.small[-1]                                   # cells 1 .. head_n of row 0 are spelled out
        composite = [int(n == 1 or any(n % d == 0 for d in range(2, n))) for n in range(1, head_n + 1)]
        head = runs_tree(composite, Set, "x")
        head = _scale_x(head, w.cell_px)
        region = If("x", head_n * w.cell_px - 1, any_zero(small), If("y", FIRST_Y + gen.CELL_H - 1, any_zero(small), head))
    else:
        region = any_zero(small) if small else base
    if prev_e is None:
        return If("y", FIRST_Y + w.small_rows * gen.CELL_H - 1, full, region)
    return If(prev_e, 0, region, full)


def _scale_x(tree, cell_px):
    """
    Pure function. Rewrites the x splits of a cell-index runs_tree to pixel thresholds.

    Examples:
        >>> gen.render(_scale_x(runs_tree([1, 0, 0], Set, "x"), 20))
        'if x > 19\\n  - Set 0\\n  - Set 1'
    """
    if tree[0] == "leaf":
        return tree
    _, prop, split, then, other = tree
    return If(prop, (split + 1) * cell_px - 1, _scale_x(then, cell_px), _scale_x(other, cell_px))


def at_number_start(prev_cc, prev_dp, advance, hold):
    """
    Pure function. `advance` on the first pixel of a number (xm == 0 of its first digit cell), else
    `hold` (a small subtree; it appears twice).

    Examples:
        >>> gen.render(at_number_start("Prev2", "Prev", Set(1), Set(0))).splitlines()[0]
        'if Prev2 > 9'
    """
    return If(prev_cc, gen.cc_xm_gt(0), hold, If(prev_dp, 0, hold, advance))


def digit_position(w, prev_cc):
    """
    Pure function. Dp channel: digit cell index 0 .. cells - 1 within the number, 0 at x == 0.

    Examples:
        >>> gen.render(digit_position(WALLS["primes_wall"], "Prev")).splitlines()[0]
        'if x > 0'
    """
    return If("x", 0, If(prev_cc, gen.cc_xm_gt(0), Leaf("W", 0), If("W", w.cells - 2, Set(0), Leaf("W", 1))), Set(0))


def digit(w, prev_cc, prev_dp, prev_lower, row_inc, group_inc, init):
    """
    Pure function. A decimal digit channel holding 0 .. 9; a value of 10 .. 19 right after a step
    means "wrapped, carry out", read by the next digit through `prev_lower` on the same pixel and
    normalised by the next step or hold. Steps: +1 (with carry in) at number starts, +row_inc at
    band starts (cc == 0 in column 0), +group_inc per margin countdown row (cc > 0 there).

    Examples:
        >>> gen.render(digit(WALLS["primes_wall"], "Prev2", "Prev", None, 0, 0, 1)).splitlines()[0]
        'if x > 0'
    """
    sub = lambda pred, k: If(pred, 9, Leaf(pred, k - 10), Leaf(pred, k))          # drop a carry flag, add k
    step = lambda pred, k: (If(prev_lower, 9, sub(pred, k + 1), sub(pred, k)) if prev_lower else sub(pred, k))
    along = at_number_start(prev_cc, prev_dp, step("W", 0) if prev_lower else sub("W", 1), sub("W", 0))
    margin = If("y", 0, If(prev_cc, 0, step("N", group_inc), sub("N", 0)), Set(init)) if w.groups > 1 else Set(init)
    column = If("y", FIRST_Y, If(prev_cc, 0, sub("N", 0), step("N", row_inc)), margin)
    return If("x", 0, along, column)


def significance(prev_digits_high_to_low):
    """
    Pure function. Z channel: index of the number's first non-zero digit (0 = the highest).

    Examples:
        >>> gen.render(significance(["Prev", "Prev2", "Prev3"])).splitlines()[0]
        'if Prev > 0'
    """
    ps = prev_digits_high_to_low
    return reduce(lambda t, kp: If(kp[1], 0, Set(kp[0]), t), reversed(list(enumerate(ps[:-1]))), Set(len(ps) - 1))


def digit_to_draw(w, prev_dp, prev_z, prev_digits_high_to_low):
    """
    Pure function. D channel: the digit of the current digit cell (leading zeros, the blank cell
    and the partial number past the last full cell of a row are BLANK), read off the digit
    channels with one runs_tree each.

    Examples:
        >>> gen.render(digit_to_draw(WALLS["primes_wall"], "Prev7", "Prev6", ["Prev5", "Prev4", "Prev3", "Prev2"])).splitlines()[0]
        'if y > 19'
    """
    copy = lambda p: runs_tree(list(range(10)), Set, p)
    ps = prev_digits_high_to_low
    cases = [(w.cells - 2, Set(gen.BLANK))] + [(k - 1, If(prev_z, k, Set(gen.BLANK), copy(ps[k])) if k < len(ps) - 1 else copy(ps[k]))
                                              for k in range(len(ps) - 1, 0, -1)]
    digits = chain(prev_dp, cases, If(prev_z, 0, Set(gen.BLANK), copy(ps[0])))
    return If("y", FIRST_Y - 1, If("x", w.per_row * w.cell_px - 1, Set(gen.BLANK), digits), Set(gen.BLANK))


def build(w):
    """
    Pure function. (tree source, channel names) for wall `w`.

    Examples:
        >>> src, names = build(WALLS["primes_wall"]); len(names), names[-1]
        (33, 'B')
    """
    gen.apply_geometry(1)
    gen.CELL_H, gen.GLYPH_Y0 = w.cell_h, w.cell_h - gen.GLYPH_ROWS
    gen.FONT.update(DIGIT_FONT)
    names = layout(w)
    prev = prev_of(names)
    ds = w.divisors
    trees = {}
    n_blocks = len(w.blocks())
    for k, block in enumerate(w.blocks()):
        q, e, m = f"Q{k + 1}", f"E{k + 1}", f"M{k + 1}"
        flag = If(prev(q, f"Q{k}"), 0, Set(1), Set(0)) if (k and w.groups > 1) else Set(0)
        trees[q] = number_counter(w.cell_px, group_countdown(w) if k == 0 else flag)
        if w.groups > 1:
            trees[e] = region_flag(w)
        for d in block:
            trees[f"C{d}"] = counter(w, d, prev(f"C{d}", q))
        trees[m] = or_flag(w, [prev(m, f"C{d}") for d in block], prev(m, e) if w.groups > 1 else None, prev(m, f"M{k}") if k else None, k)
    last_q, last_m = f"Q{n_blocks}", f"M{n_blocks}"
    trees["cc"] = number_counter(gen.CELL_W, If(prev("cc", last_q), 0, Set(1), Set(0))) if w.groups > 1 else gen.cell_counter()
    trees["Dp"] = digit_position(w, prev("Dp", "cc"))
    digit_names = ["o", "t", "h", "th", "tk"][:w.digits]
    row_incs = [int(c) for c in str(w.per_row).zfill(w.digits)[::-1]]
    group_incs = [int(c) for c in str(w.group_size).zfill(w.digits)[::-1]]
    assert len(str(w.n_max)) <= w.digits and len(str(w.group_size)) <= w.digits
    for i, name in enumerate(digit_names):
        lower = prev(name, digit_names[i - 1]) if i else None
        trees[name] = digit(w, prev(name, "cc"), prev(name, "Dp"), lower, row_incs[i], group_incs[i], 1 if i == 0 else 0)
    high_to_low = digit_names[::-1]
    trees["Z"] = significance([prev("Z", n) for n in high_to_low])
    trees["D"] = digit_to_draw(w, prev("D", "Dp"), prev("D", "Z"), [prev("D", n) for n in high_to_low])
    trees["P"] = gen.pattern_channel(ORDER, prev("P", "D"), prev("P", "cc"), flip=False)
    tone = lambda ch, k: If(prev(ch, last_m), 0, Set(COMPOSITE_COLOUR[k] - (COMPOSITE_COLOUR[0] if k else 0)),
                            Set(PRIME_COLOUR[k] - (PRIME_COLOUR[0] if k else 0)))
    trees["R"] = gen.colour_channel(prev("R", "P"), prev("R", "cc"), on=tone("R", 0))
    trees["G"], trees["B"] = tone("G", 1), tone("B", 2)
    tree = gen.simplify(gen.dispatch([trees[n] for n in names]))
    header = f"""/* {w.name}.tree — GENERATED by art/gen_primes_wall.py --wall {w.name}; edit the generator, not this file.
   {w.canvas[0]}x{w.canvas[1]}: the integers 1 .. {w.n_max} in a dense grid ({w.per_row} per group row, {w.groups} group(s), {w.cell_h} px rows,
   1 px font, right-aligned), primes {PRIME_COLOUR}, composites {COMPOSITE_COLOUR}. One counter channel per prime
   divisor {ds[0]} .. {ds[-1]} holds n mod d; composite = some counter is 0 (the first {w.small_rows} row(s) of group 0 test
   only {w.small} so n = d is no hit). Byte budget {w.limit}.
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(names))}
     Q<k> number-pixel counter / group countdown; E<k> small-test region; C<d> counters; M<k> composite so far;
     cc cell counter ({gen.CELL_W}x{gen.CELL_H} digit cells); Dp digit index; {" ".join(digit_names)} decimal digits (10..19 = carry out);
     Z first non-zero digit; D digit to draw; P glyph-row pattern; R brightness; G B RCT 3 deltas.
   Nodes: {gen.count_nodes(tree)} */
Width {w.canvas[0]}
Height {w.canvas[1]}
RCT 3
HiddenChannel {len(names) - 3}
"""
    return header + gen.render(tree) + "\n", names


def generate(wall="primes_wall", out=None):
    """Command. Writes the tree (default art/trees/<wall>.tree) and prints the node count."""
    w = WALLS[wall]
    src, names = build(w)
    out = Path(out) if out else ROOT / f"art/trees/{w.name}.tree"
    out.write_text(src)
    print(f"{out.name}: {w.canvas[0]}x{w.canvas[1]}, {w.n_max} numbers, {len(w.divisors)} divisor channels, "
          f"{src.split('Nodes: ')[1].split(' ')[0]} nodes, {len(names)} channels, budget {w.limit} B")


if __name__ == "__main__":
    fire.Fire(generate)
