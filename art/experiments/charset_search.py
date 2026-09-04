"""Local search for the 15-letter set (plus blank) whose glyph-row patterns compress best."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # dump root (portable)
spec = importlib.util.spec_from_file_location("gen", ROOT / "art/gen_text_tree.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REQUIRED = set("AEIO")                     # keep it looking like text
dist = lambda p, q: sum(a != b for a, b in zip(gen.glyph_row_patterns(gen.FONT[p]), gen.glyph_row_patterns(gen.FONT[q])))


def cost(letters):
    """Query (RNG-seeded optimiser). Total pattern transitions for the best order of ' ' + letters."""
    order = gen.best_order([" "] + list(letters), " ", dist, restarts=12)
    return gen.path_cost(order, dist), "".join(order)


current = list("ETAOINSRHDLUCMF")
best_cost, best_order = cost(current)
print("start", "".join(current), best_cost, best_order)
improved = True
while improved:
    improved = False
    for i, out_letter in enumerate(current):
        if out_letter in REQUIRED:
            continue
        for in_letter in LETTERS:
            if in_letter in current:
                continue
            candidate = current[:i] + [in_letter] + current[i + 1:]
            c, order = cost(candidate)
            if c < best_cost:
                best_cost, best_order, current, improved = c, order, candidate, True
                print("improved", "".join(sorted(current)), c, order)
print("best", "".join(sorted(current)), best_cost, best_order)
