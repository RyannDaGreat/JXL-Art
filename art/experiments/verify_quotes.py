"""
Read the quotes back out of art/out/quotes.png (1 px geometry, one quote per slot) and check every
quote against the grammar.

    python3.10 art/experiments/verify_quotes.py
"""
import importlib.util
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]   # dump root (portable)
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
gen.apply_geometry(1)
GLYPHS = sorted({ch for w, _ in gen.quote_vocab() for ch in w})
BY_BITS = {tuple(int(b) for row in gen.FONT[ch] for b in row): ch for ch in GLYPHS}
CLASS_OF = {w: c for c, ws in gen.QUOTE_WORDS.items() for w in ws}
SENTENCE = re.compile(r'^"DET (ADJ )?NOUN( (VERB|PREP) DET (ADJ )?NOUN)*\."$')
LIT = 60


def read_cell(px, cx0, cy0):
    """Query (reads pixels). Character in the cell whose top-left is (cx0, cy0); ' ' when blank, '?' if unknown."""
    bits = tuple(int(max(px[cx0 + gen.GLYPH_X0 + c * gen.PIXEL + gen.PIXEL // 2, cy0 + gen.GLYPH_Y0 + r * gen.PIXEL + gen.PIXEL // 2]) > LIT)
                 for r in range(5) for c in range(3))
    return BY_BITS.get(bits, " " if not any(bits) else "?")


def well_formed(line):
    """
    Pure function. True when the line is a quoted sentence of vocabulary words in grammar order.

    Examples:
        >>> well_formed('"THE SILENT MIND DEVOURS EVERY TRUTH OF THE SOUL."')
        True
        >>> well_formed('"THE MIND DEVOURS."'), well_formed('"MIND IS THE SOUL."')
        (False, False)
    """
    m = re.fullmatch(r'"(.*)\."', line)
    if not m:
        return False
    words = m.group(1).split(" ")
    if not all(w in CLASS_OF for w in words):
        return False
    return SENTENCE.fullmatch('"' + " ".join(CLASS_OF[w] for w in words) + '."') is not None


def rendered_quotes():
    """Query (reads the PNG). Non-empty slot texts, one per decoder group, text band and slot."""
    im = Image.open(ROOT / "art/out/quotes.png").convert("RGB")
    px = im.load()
    first_band = -(-gen.QUOTE_SEED_ROWS // gen.CELL_H)
    slot_px = gen.QUOTE_SLOT_CELLS * gen.CELL_W
    quotes = []
    for group_y0 in range(0, im.height, gen.GROUP):
        for k in range(first_band, gen.GROUP // gen.CELL_H):
            y0 = group_y0 + gen.Y_OFFSET + gen.CELL_H * k
            if y0 + gen.CELL_H > group_y0 + gen.GROUP:
                break
            for slot_x0 in range(0, im.width, slot_px):
                text = "".join(read_cell(px, slot_x0 + gen.CELL_W * i, y0) for i in range(gen.QUOTE_SLOT_CELLS)).rstrip()
                if text:
                    quotes.append(text)
    return quotes


if __name__ == "__main__":
    quotes = rendered_quotes()
    bad = [q for q in quotes if not well_formed(q)]
    for q in quotes[::25]:
        print(("OK  " if well_formed(q) else "BAD ") + q)
    for q in bad[:10]:
        print("BAD " + q)
    words = [w for q in quotes for w in re.findall(r"[A-Z]+", q)]
    counts = {w: words.count(w) for w in CLASS_OF}
    print(f"\nquotes: {len(quotes)} quotes, {len(bad)} malformed, {sum(1 for c in counts.values() if c)} distinct words used of {len(CLASS_OF)},"
          f" least used: {sorted(counts, key=counts.get)[:5]}, longest: {max(map(len, quotes))} cells")
    assert not bad
