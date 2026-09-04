#!/usr/bin/env python3
"""
Generates art/trees/text.tree: a 64x32 grid of pseudo-random ASCII characters (3x5 pixel font,
32-character set) chosen by a Rule-30 cellular automaton.

Why a generator: the font lookup is a decision tree over (pixel position, character value).
For each of the 15 glyph pixels the tree branches on the character value with one split per
run of equal bits, so the order of the 32 characters decides the tree size. We order them by a
short Hamming-distance path (nearest neighbour + 2-opt) and emit the runs as nested "if"s.

Layout: cell 16 px wide (x counter starts at 8 so the first partial cell is blank) and 32 px
tall; glyph pixels are 4x4 at xm 2..13, ym 12..31. The character value (0..31) is assembled at
xm == 0 from five consecutive Rule-30 rows (ym 7..11), Wolfram's classic Rule-30 PRNG. The
glyph sits low in the cell so even the first text line samples the automaton 7+ steps after
the seed row (sampling rows 1..5 left visible near-repeats from the quasi-periodic seed).

Usage: python3.10 art/gen_text_tree.py   (then python3.10 art/build.py text)
"""
from pathlib import Path

import fire

TREE_PATH = Path(__file__).resolve().parent / "trees" / "text.tree"

# 3x5 font: 5 rows of 3 bits per character. Index 0 must be the blank so V=0 draws nothing.
FONT = {
    " ": ["000", "000", "000", "000", "000"],
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "G": ["011", "100", "101", "101", "011"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "010"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["110", "101", "101", "101", "101"],
    "O": ["010", "101", "101", "101", "010"],
    "P": ["110", "101", "110", "100", "100"],
    "Q": ["010", "101", "101", "010", "001"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "!": ["010", "010", "010", "000", "010"],
    "?": ["110", "001", "010", "000", "010"],
    ".": ["000", "000", "000", "000", "010"],
}
GLYPH_ROWS, GLYPH_COLS = 5, 3
PIXEL = 4                      # screen pixels per glyph pixel
GLYPH_X0, GLYPH_Y0 = 2, 12     # glyph origin inside the cell
SAMPLE_Y0 = GLYPH_Y0 - 5       # the 5 CA rows feeding the character value sit just above the glyph
CELL_W, CELL_H = 16, 32
X_OFFSET = 8                   # x counter starts here so cells straddle the x=0 boundary cone
Y_OFFSET = 20                  # first cell row starts here: 27+ CA steps below the seed row
LAST_FULL_CELL_END = 1024 - X_OFFSET   # pixels beyond this belong to a cut-off cell


def glyph_bits(rows):
    """
    Pure function. Flattens 5 rows of '010' strings into a tuple of 15 ints, row-major.

    Examples:
        >>> glyph_bits(["010", "101", "111", "101", "101"])
        (0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1)
    """
    return tuple(int(b) for row in rows for b in row)


def hamming(a, b):
    """
    Pure function. Number of differing positions between two equal-length bit tuples.

    Examples:
        >>> hamming((0, 1, 1), (1, 1, 0))
        2
    """
    return sum(x != y for x, y in zip(a, b))


def path_cost(order, glyphs):
    """
    Pure function. Total Hamming distance along consecutive glyphs in `order`.

    Examples:
        >>> path_cost([0, 1, 2], {0: (0, 0), 1: (0, 1), 2: (1, 1)})
        2
    """
    return sum(hamming(glyphs[a], glyphs[b]) for a, b in zip(order, order[1:]))


def shortest_hamming_path(glyphs, start):
    """
    Pure function. Nearest-neighbour path from `start` through all keys, improved by 2-opt.
    Deterministic; returns a list of keys. The path (not cycle) cost is what matters because
    run count = transitions + 1 per pixel.

    Examples:
        >>> shortest_hamming_path({'a': (0, 0), 'b': (1, 1), 'c': (0, 1)}, 'a')
        ['a', 'c', 'b']
    """
    remaining = set(glyphs) - {start}
    order = [start]
    while remaining:
        nearest = min(sorted(remaining), key=lambda k: hamming(glyphs[order[-1]], glyphs[k]))
        order.append(nearest)
        remaining.remove(nearest)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if path_cost(candidate, glyphs) < path_cost(order, glyphs):
                    order, improved = candidate, True
    return order


def runs_tree(bits, lo, hi, depth):
    """
    Pure function. Emits nested 'if Prev > k' lines selecting Set 255/0 by runs of equal bits
    in bits[lo..hi] (inclusive); splits at the middle run boundary for a balanced tree.

    Examples:
        >>> print(runs_tree((1, 1, 0, 0), 0, 3, 0))
        if Prev > 1
          - Set 0
          - Set 255
    """
    pad = "  " * depth
    boundaries = [i for i in range(lo, hi) if bits[i] != bits[i + 1]]
    if not boundaries:
        return f"{pad}- Set {255 if bits[lo] else 0}"
    b = boundaries[len(boundaries) // 2]
    return "\n".join([f"{pad}if Prev > {b}", runs_tree(bits, b + 1, hi, depth + 1), runs_tree(bits, lo, b, depth + 1)])


def count_nodes(tree_text):
    """
    Pure function. Number of decision + leaf nodes in emitted tree text.

    Examples:
        >>> count_nodes("if Prev > 1\\n  - Set 0\\n  - Set 255")
        3
    """
    return sum(1 for line in tree_text.splitlines() if line.strip().startswith(("if ", "- ")))


def column_tree(bits_by_pixel, row, depth):
    """
    Pure function. Emits the xm dispatch (3 columns) for one glyph row; each column is a runs tree.

    Examples:
        >>> t = column_tree({p: (1,) * 32 for p in range(15)}, 0, 0)
        >>> t.splitlines()[0]
        'if Prev4 > 9'
    """
    pad = "  " * depth
    col = lambda c: runs_tree(bits_by_pixel[row * GLYPH_COLS + c], 0, len(FONT) - 1, depth + 2)
    return "\n".join([
        f"{pad}if Prev4 > {GLYPH_X0 + 2 * PIXEL - 1}", col(2),
        f"{pad}  if Prev4 > {GLYPH_X0 + PIXEL - 1}", col(1), col(0),
    ])


def glyph_tree(bits_by_pixel, depth):
    """
    Pure function. Emits the ym dispatch over the 5 glyph rows (rows 4,3,2 then 1,0).

    Examples:
        >>> glyph_tree({p: (0,) * 32 for p in range(15)}, 0).splitlines()[0]
        'if Prev3 > 19'
    """
    pad = "  " * depth
    y = lambda r: GLYPH_Y0 + r * PIXEL - 1        # "ym > y(r)" means glyph row >= r
    return "\n".join([
        f"{pad}if Prev3 > {y(2)}",
        f"{pad}  if Prev3 > {y(4)}", column_tree(bits_by_pixel, 4, depth + 2),
        f"{pad}    if Prev3 > {y(3)}", column_tree(bits_by_pixel, 3, depth + 3), column_tree(bits_by_pixel, 2, depth + 3),
        f"{pad}  if Prev3 > {y(1)}", column_tree(bits_by_pixel, 1, depth + 2), column_tree(bits_by_pixel, 0, depth + 2),
    ])


def indent(text, depth):
    """
    Pure function. Prefixes every line with 2*depth spaces.

    Examples:
        >>> indent("a\\nb", 1)
        '  a\\n  b'
    """
    return "\n".join("  " * depth + line for line in text.splitlines())


def render_tree(order):
    """
    Pure function. Full text.tree source for the given character order (index = CA value).

    Examples:
        >>> src = render_tree(list(FONT))
        >>> src.splitlines()[0]
        '/* text.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.'
    """
    glyphs = {ch: glyph_bits(FONT[ch]) for ch in FONT}
    bits_by_pixel = {p: tuple(glyphs[ch][p] for ch in order) for p in range(GLYPH_ROWS * GLYPH_COLS)}
    lookup = glyph_tree(bits_by_pixel, 0)
    header = f"""/* text.tree — GENERATED by art/gen_text_tree.py; edit the generator, not this file.
   64x32 grid of pseudo-random characters from a 32-glyph 3x5 font, chosen by Rule 30.
   Character order (CA value 0..31): {"".join(order)!r}
   Channels (HiddenChannel 4 => c0..c3 state, c4..c6 = R,G,B):
     c0 xm : (x + {X_OFFSET}) mod {CELL_W}     c1 ym : (y + {CELL_H - Y_OFFSET}) mod {CELL_H}
     c2 A  : Rule 30 (on = 100), row 0 is a Weyl sequence (v+633) mod 1024, column 0 is (v+37) mod 100
     c3 V  : character value, bits taken from A at xm == 0, ym = {SAMPLE_Y0}..{SAMPLE_Y0 + 4} (weights 16 8 4 2 1)
     c4 R  : glyph pixel lookup (Prev1 = V, Prev3 = ym, Prev4 = xm); c5, c6 copy R */
Width 1024
Height 1024
HiddenChannel 4
"""
    body = f"""if c > 4
  if Prev > 0
    - Set 255
    - Set 0
  if c > 3
    if x > {LAST_FULL_CELL_END - 1}
      - Set 0
      if Prev3 > {GLYPH_Y0 + GLYPH_ROWS * PIXEL - 1}
        - Set 0
        if Prev3 > {GLYPH_Y0 - 1}
          if Prev4 > {GLYPH_X0 + GLYPH_COLS * PIXEL - 1}
            - Set 0
            if Prev4 > {GLYPH_X0 - 1}
{indent(lookup, 7)}
              - Set 0
          - Set 0
    if c > 2
      /* V: Prev1 = A, Prev2 = ym, Prev3 = xm */
      if Prev3 > 0
        - W 0
        if Prev2 > {SAMPLE_Y0 + 4}
          - N 0
          if Prev2 > {SAMPLE_Y0 + 3}
            if Prev > 49
              - N +1
              - N 0
            if Prev2 > {SAMPLE_Y0 + 2}
              if Prev > 49
                - N +2
                - N 0
              if Prev2 > {SAMPLE_Y0 + 1}
                if Prev > 49
                  - N +4
                  - N 0
                if Prev2 > {SAMPLE_Y0}
                  if Prev > 49
                    - N +8
                    - N 0
                  if Prev2 > {SAMPLE_Y0 - 1}
                    if Prev > 49
                      - Set 16
                      - Set 0
                    - N 0
      if c > 1
        /* A: Rule 30; NW/NE recovered from NW-N and N-NE */
        if y > 1
          if x > 0
            if N > 49
              if NW-N > -51
                - Set 0
                - Set 100
              if NW-N > 49
                if N-NE > -50
                  - Set 100
                  - Set 0
                if N-NE > -50
                  - Set 0
                  - Set 100
            if N > 62
              - N -63
              - N +37
          if y > 0
            if N > 511
              - Set 100
              - Set 0
            if W > 390
              - W -391
              - W +633
        if c > 0
          if y > 0
            if N > {CELL_H - 2}
              - Set 0
              - N +1
            - Set {CELL_H - Y_OFFSET}
          if x > 0
            if W > {CELL_W - 2}
              - Set 0
              - W +1
            - Set {X_OFFSET}
"""
    return header + body


def generate():
    """
    Command. Writes art/trees/text.tree and prints the character order and node count.

    Examples:
        >>> # generate()  -> "order: ' .!?1I...'  lookup nodes: 231  total nodes: 300"
    """
    glyphs = {ch: glyph_bits(FONT[ch]) for ch in FONT}
    assert len(glyphs) == 32, len(glyphs)
    bottom_partial_rows = (1024 - Y_OFFSET) % CELL_H
    assert bottom_partial_rows <= GLYPH_Y0, f"cut-off glyph at the bottom: {bottom_partial_rows} rows visible"
    order = shortest_hamming_path(glyphs, " ")
    src = render_tree(order)
    TREE_PATH.write_text(src)
    print(f"order: {''.join(order)!r}  path cost: {path_cost(order, glyphs)}")
    print(f"total nodes: {count_nodes(src)}  -> {TREE_PATH}")


if __name__ == "__main__":
    fire.Fire(generate)
