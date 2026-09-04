"""
primes: count 1, 2, 3, ... down the image, sieve every number inside the decoder's MA tree and
colour the primes. Writes art/trees/primes.tree (edit this generator, never the tree).

    python3.10 art/gen_primes.py            # -> art/trees/primes.tree (then python3.10 art/build.py)

Layout (1024x1024, one decoder group, everything at 1 px): row 0 is a seed row; from Y0 down, every
BAND rows are one number n = 1, 2, 3, ... (NUMBERS of them). The left part is a sieve of
Eratosthenes: divisor d = 2, 3, 4, ... has a BAND-wide block of columns; column 0 of the block holds
d (seeded along row 0 by W + 1 at block starts, copied down), column 1 is a counter that at each
band's first row goes N - 1 while N > 1, hits 0 after 1, and restarts from d - 1 (the divisor is NW,
one row up in column 0) after a hit, so it is 0 exactly when d divides n; columns 2 .. BAND-1 copy it
so a 4x4 square can be drawn on hits. The trivial divisor d = n would also hit, so a counter stays
INACTIVE until the activation wave reaches it: channel Act copies NW, i.e. Act = 1 iff y - x >= R0,
a diagonal that, with block width = band height, passes the counter of block d exactly at band
n = d + 1 (R0 = Y0 + 2 BAND - 1); the counter then starts at d - 1 = (-(d + 1)) mod d and the first
hit is n = 2d. Channel M accumulates "some counter is 0" left to right, so at the label column it is
1 for composites. The label: digits from channel Dg, whose column 0 counts n mod 10 per band, column
1 counts the tens (incremented when W, the fresh ones digit, is 0) and every column to the right
copies WW, so the row alternates ones/tens; the glyph starts at LABEL_X + 5 (odd: tens) and
LABEL_X + 10 (even: ones) read the right digit from one shared pattern table (channel P, 3x5 digit
font, one runs_tree per glyph row, peeled across the 3 columns with the period-5 counter c5). The
hundreds glyph at LABEL_X is a "1" drawn once y passes band 100. Proper divisors stop at n / 2, so the
sieve only needs blocks up to x ~ 510 and the right half shows a bar chart instead: channel Cn counts
the hits along the row (one per block, read at the block's first mark column) and from BAR_X counts
down once every BAR_UNIT px (where c5 and xm are both 0, i.e. x = 0 mod 30), so a bar of BAR_UNIT px
per proper divisor is lit while Cn > 0; primes (Cn stays 0, M = 0) get a gold stub of one unit.
Channel L classifies pixels (label, mark, bar, stub); R G B colour them: primes gold, composites grey
(and 1, which is neither), sieve marks in a blue-to-teal gradient by x, bars teal.
Verify with art/experiments/verify_primes.py (reads digits, colours, marks and the bar back).
"""
import importlib.util
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
If, Leaf, Set, chain, runs_tree = gen.If, gen.Leaf, gen.Set, gen.chain, gen.runs_tree
render, simplify, dispatch, count_nodes = gen.render, gen.simplify, gen.dispatch, gen.count_nodes

CANVAS = (1024, 1024)
BAND = 6                                 # rows per number = columns per divisor block (the wave has slope 1)
Y0 = 4                                   # first band; row 0 seeds the divisors, rows 1..3 are blank
NUMBERS = (CANVAS[1] - Y0) // BAND       # 170
R0 = Y0 + 2 * BAND - 1                   # Act = 1 iff y - x >= R0: block d activates at band d + 1
INACTIVE = -2                            # counter value before activation (never 0)
DIGIT_ROWS = 5
LABEL_X, LABEL_PITCH = 510, 5            # hundreds glyph column (even, multiple of 5); tens at +5 (odd -> tens), ones at +10 (even -> ones)
BAR_X, BAR_UNIT = 540, BAND * 5           # divisor-count bars: BAR_UNIT px per proper divisor; BAR_X is a multiple of BAR_UNIT
MARK_ROWS = (1, 4)                       # rows of the band (inclusive) that show a hit square
Y100 = Y0 + BAND * 99                    # first row of band 100
Y10 = Y0 + BAND * 9                      # first row of band 10: the tens glyph is blank above it
DIGITS = ["111101101101111", "010110010010111", "111001111100111", "111001111001111", "101101111001001",
          "111100111001111", "111100111101111", "111001001001001", "111101111101111", "111101111001111"]
PRIME, COMPOSITE, ONE = (255, 205, 70), (96, 96, 112), (60, 60, 72)
BACKGROUND = (7, 8, 14)
MARK_GRADIENT = ((70, 105, 215), (55, 160, 215), (60, 205, 170))   # marks by third of the sieve width
BAR = (40, 120, 130)
CHANNELS = ["xm", "ym", "c5", "Act", "Sv", "M", "Cn", "Dg", "P", "L", "R", "G", "B"]


def digit_rows(digit):
    """
    Pure function. The 3-bit row patterns of a digit glyph (left pixel = bit 2).

    Examples:
        >>> digit_rows(1)
        (2, 6, 2, 2, 7)
    """
    rows = DIGITS[digit]
    return tuple(int(rows[3 * r:3 * r + 3], 2) for r in range(DIGIT_ROWS))


def mod_counter(period, first_value):
    """
    Pure function. Column-0 row counter y -> (y + first_value) mod period, copied right with W.

    Examples:
        >>> print(render(mod_counter(6, 2)))
        if x > 0
          - W 0
          if y > 0
            if N > 4
              - Set 0
              - N 1
            - Set 2
    """
    return If("x", 0, Leaf("W", 0), If("y", 0, If("N", period - 2, Set(0), Leaf("N", 1)), Set(first_value)))


def column_counter(period):
    """
    Pure function. x mod period, counted along every row with W + 1.

    Examples:
        >>> render(column_counter(6)).splitlines()[1:3]
        ['  if W > 4', '    - Set 0']
    """
    return If("x", 0, If("W", period - 2, Set(0), Leaf("W", 1)), Set(0))


def activation_wave():
    """
    Pure function. Act = 1 iff y - x >= R0: column 0 is 1 from row R0 down, every other pixel copies NW.

    Examples:
        >>> render(activation_wave()).splitlines()[0]
        'if y > 0'
    """
    return If("y", 0, If("x", 0, Leaf("NW", 0), If("y", R0 - 1, Set(1), Set(0))), Set(0))


def sieve_channel(prev_xm, prev_ym, prev_act):
    """
    Pure function. Sv: seed row d per block (W + 1 at block starts), then column 0 copies d down,
    column 1 counts down to the hit value 0 (restarting from NW - 1 = d - 1 after a hit or on
    activation) and the other columns copy it.

    Examples:
        >>> render(sieve_channel("Prev4", "Prev3", "Prev")).splitlines()[0]
        'if y > 0'
    """
    seed = If("x", 0, If(prev_xm, 0, Leaf("W", 0), Leaf("W", 1)), Set(2))
    step = If("N", 1, Leaf("N", -1), If("N", 0, Set(0), Leaf("NW", -1)))
    counter = If(prev_ym, 0, Leaf("N", 0), If(prev_act, 0, step, Set(INACTIVE)))
    body = If(prev_xm, 1, Leaf("W", 0), If(prev_xm, 0, counter, Leaf("N", 0)))
    return If("y", 0, body, seed)


def composite_flag(prev_sv):
    """
    Pure function. M = 1 once a counter to the left is 0 on this row (copied right), else 0.

    Examples:
        >>> render(composite_flag("Prev")).splitlines()[0]
        'if x > 0'
    """
    return If("x", 0, If(prev_sv, 0, Leaf("W", 0), If(prev_sv, -1, Set(1), Leaf("W", 0))), Set(0))


def divisor_count(prev_xm, prev_c5, prev_sv):
    """
    Pure function. Cn: number of hits so far along the row (one per block, read where xm == 2)
    up to the label, then from BAR_X + BAR_UNIT down by one every BAR_UNIT px (x = 0 mod 30: c5 == xm
    == 0), stopping at 0, so BAR_UNIT px are lit per divisor.

    Examples:
        >>> render(divisor_count("Prev6", "Prev4", "Prev2")).splitlines()[0]
        'if x > 569'
    """
    hit = If(prev_sv, 0, Leaf("W", 0), If(prev_sv, -1, Leaf("W", 1), Leaf("W", 0)))
    count = If("x", 0, If(prev_xm, 2, Leaf("W", 0), If(prev_xm, 1, hit, Leaf("W", 0))), Set(0))
    tick = If(prev_xm, 0, Leaf("W", 0), If(prev_c5, 0, Leaf("W", 0), If("W", 0, Leaf("W", -1), Set(0))))
    return If("x", BAR_X + BAR_UNIT - 1, tick, If("x", LABEL_X - 1, Leaf("W", 0), count))


def digit_channel(prev_ym):
    """
    Pure function. Dg: column 0 = n mod 10 (steps at band-first rows), column 1 = tens digit
    (steps when the fresh ones digit W is 0), columns to the right copy WW (ones, tens, ones, ...).

    Examples:
        >>> render(digit_channel("Prev5")).splitlines()[0]
        'if x > 1'
    """
    step = If("N", 8, Set(0), Leaf("N", 1))
    ones = If("y", 0, If(prev_ym, 0, Leaf("N", 0), step), Set(0))
    tens = If("y", 0, If(prev_ym, 0, Leaf("N", 0), If("W", 0, Leaf("N", 0), step)), Set(0))
    return If("x", 1, Leaf("WW", 0), If("x", 0, tens, ones))


def glyph_table(prev_ym, rows_of):
    """
    Pure function. Pattern by band row for a glyph: rows_of(r) gives the subtree of glyph row r;
    the band rows past DIGIT_ROWS are 0.

    Examples:
        >>> render(glyph_table("Prev", lambda r: Set(r))).splitlines()[0]
        'if Prev > 4'
    """
    return chain(prev_ym, [(DIGIT_ROWS - 1, Set(0))] + [(r - 1, rows_of(r)) for r in range(DIGIT_ROWS - 1, 0, -1)], rows_of(0))


def pattern_channel(prev_dg, prev_c5, prev_ym):
    """
    Pure function. P: at a glyph start column (c5 == 0 inside the label) the 3-bit row pattern of
    the digit Dg (or of "1" for the hundreds glyph once past band 100), peeled by one bit at the
    next two columns, 0 in the gaps and outside the label.

    Examples:
        >>> render(pattern_channel("Prev", "Prev5", "Prev6")).splitlines()[0]
        'if x > 509'
    """
    digits = glyph_table(prev_ym, lambda r: runs_tree([digit_rows(d)[r] for d in range(10)], Set, prev_dg))
    hundreds = If("y", Y100 - 1, glyph_table(prev_ym, lambda r: Set(digit_rows(1)[r])), Set(0))
    start = If("x", LABEL_X + LABEL_PITCH - 1, digits, hundreds)
    peel = If(prev_c5, 1, If("W", 1, Leaf("W", -2), Leaf("W", 0)), If("W", 3, Leaf("W", -4), Leaf("W", 0)))
    label = If(prev_c5, 2, Set(0), If(prev_c5, 0, peel, start))
    return If("x", LABEL_X - 1, If("x", LABEL_X + 3 * LABEL_PITCH - 1, Set(0), label), Set(0))


def lit_class(prev_xm, prev_ym, prev_c5, prev_sv, prev_cn, prev_p):
    """
    Pure function. L: 4 on the prime stub (first bar unit, no divisors), 3 on bar pixels (Cn > 0),
    2 on lit label pixels (the pattern bit of the glyph column; the tens glyph stays dark below
    band 10 and nothing is lit above band 1), 1 on sieve-mark pixels (counter 0, block columns
    2.., MARK_ROWS), else 0. Bars and stubs use MARK_ROWS too.

    Examples:
        >>> render(lit_class("Prev9", "Prev8", "Prev7", "Prev5", "Prev3", "Prev")).splitlines()[0]
        'if x > 509'
    """
    lit = lambda split: If(prev_p, split, Set(2), Set(0))
    glyph = If(prev_c5, 1, lit(0), If(prev_c5, 0, lit(1), lit(3)))
    label = If("y", Y0 - 1, If("x", LABEL_X + 2 * LABEL_PITCH - 1, glyph, If("y", Y10 - 1, glyph, Set(0))), Set(0))
    stub = If("x", BAR_X + BAR_UNIT - 1, Set(0), Set(4))
    bar = If(prev_ym, MARK_ROWS[1], Set(0), If(prev_ym, MARK_ROWS[0] - 1, If(prev_cn, 0, Set(3), stub), Set(0)))
    hit = If(prev_sv, 0, Set(0), If(prev_sv, -1, If(prev_ym, MARK_ROWS[1], Set(0), If(prev_ym, MARK_ROWS[0] - 1, Set(1), Set(0))), Set(0)))
    mark = If(prev_xm, 1, hit, Set(0))
    return If("x", LABEL_X - 1, If("x", BAR_X - 1, bar, label), mark)


def colour_channel(k, prev_l, prev_m):
    """
    Pure function. One of R G B (component k) from the lit class, the composite flag and the row.

    Examples:
        >>> render(colour_channel(0, "Prev", "Prev5")).splitlines()[0]
        'if Prev > 3'
    """
    not_one = lambda then: If("y", Y0 + BAND - 1, then, Set(ONE[k]))          # band 1 is neither
    label = not_one(If(prev_m, 0, Set(COMPOSITE[k]), Set(PRIME[k])))
    stub = If("y", Y0 + BAND - 1, If(prev_m, 0, Set(BACKGROUND[k]), Set(PRIME[k])), Set(BACKGROUND[k]))
    third = LABEL_X // 3
    mark = chain("x", [(2 * third - 1, Set(MARK_GRADIENT[2][k])), (third - 1, Set(MARK_GRADIENT[1][k]))], Set(MARK_GRADIENT[0][k]))
    return chain(prev_l, [(3, stub), (2, Set(BAR[k])), (1, label), (0, mark)], Set(BACKGROUND[k]))


def build():
    """
    Pure function. (tree source text, channel names) for the primes piece.

    Examples:
        >>> src, names = build()
        >>> names[-3:], src.splitlines()[0]
        (['R', 'G', 'B'], '/* primes.tree — GENERATED by art/gen_primes.py; edit the generator, not this file.')
    """
    names = CHANNELS
    prev = lambda here, there: "Prev" + (str(names.index(here) - names.index(there)) if names.index(here) - names.index(there) > 1 else "")
    trees = {"xm": column_counter(BAND), "ym": mod_counter(BAND, (-Y0) % BAND), "c5": column_counter(LABEL_PITCH),
             "Act": activation_wave()}
    trees["Sv"] = sieve_channel(prev("Sv", "xm"), prev("Sv", "ym"), prev("Sv", "Act"))
    trees["M"] = composite_flag(prev("M", "Sv"))
    trees["Cn"] = divisor_count(prev("Cn", "xm"), prev("Cn", "c5"), prev("Cn", "Sv"))
    trees["Dg"] = digit_channel(prev("Dg", "ym"))
    trees["P"] = pattern_channel(prev("P", "Dg"), prev("P", "c5"), prev("P", "ym"))
    trees["L"] = lit_class(prev("L", "xm"), prev("L", "ym"), prev("L", "c5"), prev("L", "Sv"), prev("L", "Cn"), prev("L", "P"))
    for k, ch in enumerate("RGB"):
        trees[ch] = colour_channel(k, prev(ch, "L"), prev(ch, "M"))
    tree = simplify(dispatch([trees[n] for n in names]))
    header = f"""/* primes.tree — GENERATED by art/gen_primes.py; edit the generator, not this file.
   {CANVAS[0]}x{CANVAS[1]}: n = 1 .. {NUMBERS} down the image, one per {BAND} rows; a sieve of Eratosthenes on the left
   (divisor d = 2, 3, .. per {BAND}-px block: column 0 holds d, column 1 counts -n mod d, 0 = hit, activated by the
   diagonal y - x >= {R0} so d = n never fires), labels at x = {LABEL_X} (gold = prime, grey = composite), from {BAR_X} a bar
   of {BAR_UNIT} px per proper divisor (gold stub = prime).
   Channels: {", ".join(f"c{i} {n}" for i, n in enumerate(names))}
     xm x mod {BAND}; ym band row; c5 x mod {LABEL_PITCH}; Act wave; Sv sieve; M composite flag; Cn divisor count / bar countdown;
     Dg digits (ones, tens alternating); P digit row pattern; L lit class (1 mark, 2 label, 3 bar, 4 stub); R G B colours.
   Nodes: {count_nodes(tree)} */
Width {CANVAS[0]}
Height {CANVAS[1]}
HiddenChannel {len(names) - 3}
"""
    return header + render(tree) + "\n", names


def generate(out=None):
    """
    Command. Writes the tree (default art/trees/primes.tree) and prints its node count.

    Examples:
        >>> # generate()                      -> art/trees/primes.tree
        >>> # generate("/tmp/x/primes.tree")  -> anywhere else (e.g. a scratchpad before it fits under 1 KB)
    """
    src, names = build()
    path = Path(out) if out else ROOT / "art/trees/primes.tree"
    path.write_text(src)
    print(f"{path}: {src.split('Nodes: ')[1].split(' ')[0]} nodes, channels {names}")


if __name__ == "__main__":
    fire.Fire(generate)
