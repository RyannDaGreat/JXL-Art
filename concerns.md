# concerns.md — how this project came to be (append-only)

## 2026-09-03 21:45 — kickoff
- Read wtf.html and tools/jxl_from_tree.cc to learn the exact grammar (whitespace tokens, mandatory
  leaf offsets, `HiddenChannel`, `Prev2..Prev19`).
- Found `jxl_from_tree` from Homebrew jpeg-xl 0.11.1 aborting: `Library not loaded:
  /opt/homebrew/opt/imath/lib/libImath-3_1.29.dylib`. Fixed with `brew reinstall jpeg-xl` (0.12.0).

## 2026-09-03 21:55 — MISTAKE: system install without dump protocol
- I ran `brew reinstall jpeg-xl` before writing a `setup.sh`. User interrupted: "this is a dump. if
  you're going to install stuff, you MUST follow FULL dump protocol. That means no WOM problems."
- Root cause: treated a "repair" as not-an-install. Lesson: in a dump, *any* command that changes the
  machine goes into top-level `setup.sh` at the moment it is run.
- Fix: created `setup.sh` (macOS brew path + Linux build-from-source path + pip deps + self-check),
  ran it, and documented the top-level layout rule in the manifest.

## 2026-09-03 22:00 — semantics verified (art/experiments/t1..t4)
- t1: `Set 300` → white, `Set -50` → black. Output clamps; state is unbounded during decoding.
- t2: HiddenChannel counter (x mod 64) invisible, readable via `Prev`. 37 bytes.
- t3: 2048x256, `if x > 1500` never true and a `W +1` ramp restarts at x=1024 → group-local coords.
- t4: 2048x2048 with `g` thresholds 21/22/23 (result recorded below once viewed).

## 2026-09-03 22:05 — CA rule search, first attempt (wrong scoring)
- Brute-forced all 256 skewed rules `new = f(NW,N,W)` from a single seed at (0,0). My scorer
  penalised absolute neighbour correlation and used stride-16 samples on a 256² simulation; the
  "winners" (rules 52/116/...) were actually a filled triangle (all samples correlated) and the
  balanced rules (54, 62, 86, ...) produced NaN correlations, meaning their samples were constant,
  i.e. periodic patterns. Conclusion so far: no single-seed skewed boolean rule is chaotic in the
  bulk with this scorer; need to re-check rule 30 (`NW xor (N or W)`) directly and consider richer
  seeds (seed the whole first row/column) or multi-bit state.

## 2026-09-03 22:40 — first builds
- digits.tree v1 built at 130 B, 16x16 seven-segment 0/1 grid, digits random. VLM check: glyphs read
  clearly; first column almost all "0" (15/16).
- flag.tree v1 built at 261 B: 13 stripes (79 px), canton 780x553, 50 stars in the 6/5 stagger via
  counters + row parity; star = ">= 4 of 5 pentagram half-planes" on hidden linear channels.
  VLM check at 5x zoom: clean five-pointed stars, correct colours. User: "The flag looks great!"
- Group ids confirmed: 2x2 groups are 21,22,23,24 in raster order; 1946x1024 has groups 21 (left), 22.

## 2026-09-03 22:55 — MISTAKE: "NW" used as a property
- `NW` is only a predictor. Recovered it from `NW-N` given N (and NE from `N-NE`).

## 2026-09-03 23:05 — MISTAKE + LESSON: Rule 30 goes periodic near a fixed left boundary
- Sampling the CA at xm=8 instead of 0 did not fix the all-zero first column. Numeric check (crop of
  the raw CA channel, art/experiments/rule30_debug.tree) showed rows 64 apart *identical* in the
  first ~16 columns: Rule 30 is left-permutive, so information flows right at speed 1 but left only
  weakly; a constant boundary makes the boundary region periodic, and the period was 64 = the cell
  pitch. Fix: column 0 runs its own Weyl sequence (v+37 mod 100) downward, injecting entropy.
  Verified with a drawn-vs-sampled comparison script (0 mismatches, column 0 now mixed).
- Encoding switched to on=100/off=0 so boundary values 0..99 and interior values share the
  thresholds 49 / -51 / -50.

## 2026-09-03 23:20 — text piece + LESSON: seed row must not be periodic within the width
- art/gen_text_tree.py generates text.tree (3x5 font, 32 chars, Hamming-path ordering, 287 nodes,
  336 B). VLM check: letters legible, BUT the top ~7 text rows repeated every 25 characters. Cause:
  Rule 30 is translation invariant, so a seed row with spatial period 100 px stays periodic until
  boundary influence (speed 1 px/row from the left) reaches it. Fix: seed period > width,
  v = (v+633) mod 1024 (golden-ratio step), threshold 511. Applied to digits too.
- Measured cost: ~1.2 bytes/node for very repetitive trees (text), ~2.1 bytes/node for the flag
  (many distinct properties/split values). Golf = fewer *distinct* symbols, not just fewer nodes.

## 2026-09-03 23:30 — flag v3 golf attempt (art/experiments/flag_v3.tree): 223 B vs 261 B
- Mirror symmetry: u = xm-65 counter, |u| built with W+-3 flipping at u=0; 3 half-plane channels
  (L2 = ym-3|u|, L4 = 167-3|u|-4ym, L5 = 167+3|u|-4ym) and a 13-node star test replace 4 channels
  and a 29-node ">=4 of 5" tree. Row parity folded into the U counter (flips when ym wraps).
  Region + K channels folded into R; G and B derived from R by thresholds. 13 -> 9 channels.
