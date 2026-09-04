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

## 2026-09-04 00:10 — flag golf outcome
- Probed header cost: HiddenChannel 0/1/2/4/8 => 24/30/36/48/72 B, i.e. 6 B per hidden channel.
  Node removals in repetitive trees saved nothing (v4 = v3 = 223 B); RCT 3 colour deltas saved 1 B
  (blue G 59 -> 60); a 2:1 canvas would save 2 B (rejected: wrong aspect); a single-group
  upsampled version ~8 B (rejected: blur). Final flag.tree 222 B, user approved the look.

## 2026-09-04 00:30 — infinity text, first attempts (too big)
- Two elliptical rings via a separable quadratic field with 64-px chord slopes: 510 B. Lemniscate
  of Gerono (separable quartic): ~570 B. User: "omg why did it make it sooo much bigger ... the code
  looks so repetetive too ... surely theres ways to simplify"; target set to 300 B.
- Root cause: predictors are linear, so curvature only comes from tables of slopes, and each
  distinct multi-hundred constant costs ~2 B. The mask alone was ~+170..230 B.

## 2026-09-04 01:00 — generator rewrite + golf (LESSONS)
- Rewrote gen_text_tree.py around a tree DSL; every channel is one function. Measured variants:
  32 glyphs pixel-lookup 361 B, rows-lookup 325 B; 16 glyphs rows-lookup 268 B.
- Merged xm/ym into one counter cc = 32*xm + ym (-6 B header); sample V at xm == 1 (no x offset,
  no partial-cell test); RCT 3 greyscale (G = B = Set 0); Rule 30 with on = 1024 so Weyl seed rows
  need no threshold row; PrevAbs single-test band gate.
- MISTAKE: first 2-opt optimiser (single start) gave a worse order than before (297 vs 287 nodes);
  fixed with random restarts. MISTAKE: doctest for chord_slopes had the wrong expected splits.
- MISTAKE: polygon field used +1 between the loop centres (sign of `-default`), so only the left
  loop rendered; caught by the montage VLM check, fixed to `default`.
- Charset local search (must keep A E I O): most-frequent letters cost 32 row transitions, the set
  "ABDEFHIKNOPRTUV" costs 24 -> text_infinity 302 -> 287 B (diamonds).
- Final: text.tree 334 B (32 glyphs), text_infinity.tree 296 B (16 glyphs, hexagon loops).

## 2026-09-04 01:40 — first-line echoes and the flip (LESSON)
- Even 28 rows below the seed, the first text line echoed runs 55 cells apart. Math: on the 16-px
  sample lattice a Weyl seed with any odd step has a near-return with only 16/1024 phase error
  (~3% of bits differ) at some distance <= 63 cells (pigeonhole), so the choice of step cannot fix
  it; only more Rule-30 steps can (P(echo) ~ 0.97^(window) — 34% at 28 rows, ~12% at 60 rows).
- Fix: blank the first band (2 nodes) so the first drawn line is 60 steps below the seed, and
  `Orientation 4` (vertical flip) so the spare 64 px margin shows at the bottom.
- MISTAKE: the flip also flipped the glyphs (upside-down letters in the VLM check); fixed by
  emitting the font rows reversed in decode space (`decoded_glyph`).
- MISTAKE: moving Y_OFFSET to 52 made the counter start negative and would not have skipped a
  band anyway (bands are periodic in the counter); a band can only be blanked explicitly.

## 2026-09-04 02:20 — wide infinity, no blank glyph
- User: "plz make it bigger and fit the aspect ratio. so, more letters and not ssquare" / "also
  space should not be one of the chatacters".
- Canvas 2048x1024 = two decoder groups. Seed row uses a different Weyl step per group (633 / 411)
  because identical groups decode to identical halves; the mask field's flip position and base
  value branch on g. Blank cells are now a negative V (pattern channel emits 0 for V <= -1); the
  blank first band and the unused xm == 0 column hold -1.
- Mask simplified to diamonds clipped by y thresholds (flat-topped hexagons); knee slopes and the
  quadratic rings removed from the generator (kept in git history). Loop geometry as fractions
  of the canvas: offset 0.23 W, r_in 0.146 W, r_out 0.264 W, clip 0.41 H.
- 16-letter search without a blank: "ABDEFHIKNOPRTVXY", 20 row transitions (start set had 28).
- Sizes: text 338 B (32 glyphs incl. "2"), text_infinity 315 B (+16 B over the 299 B square
  version: 2 groups ~8 B, group branches, y clip). VLM check: both loops full of letters, upright,
  no blanks inside the strokes, 64 px margin at the bottom from the flip.

## 2026-09-04 03:00 — round loops, full glyph set
- User: "the infinity no longer looks rounded it looks like the HSBC logo" / "also does it even use
  all the letters and !.{} etc". The clipped-diamond mask really was a hexagon pair.
- Mask replaced by elliptical rings: f = (|x-cx|-d)^2/192, g = (y-cy)^2/128 grown by chord slopes
  over 192/128-px pieces aligned to the centres, so slopes are 1, 3, 5, 7 (small, repeated) and the
  value is exact at breakpoints; sagitta over a 192-px chord at r = 520 is ~9 px (invisible at
  cell resolution). Lesson: the polygon look came from the *geometry* (45 deg edges), not from
  chord approximation — coarse chords of a true quadratic look round.
- Font gained "," "{" "}"; CHARSET_32 = A-Z + "!?.,{}" (no digits now). Both pieces use it.
- Sizes: text 346 B; text_infinity 440 B (32 glyphs) / 354 B (--charset16). VLM check: round
  overlapping loops, braces and commas legible, no blank glyphs.

## 2026-09-04 04:00 — a real figure-eight
- User: "an infinity symbol's derivitive in the middle should lok like an X ... it looks like 2
  circles next to each other". Overlapping rings cross at 2*acos(d/r) ~ 32 degrees: a lens.
- Mask replaced by a Gerono-type lemniscate band S = (|u|^p/a^(p-2) - u^2 + k v^2)/16, |S| <= eps.
  Explored (montages): quartic eps 60k had thin tips and a knot 2x the stroke; bold cubic (p = 3)
  closed the holes because its minimum is only 0.148 a^2 (quartic 0.25 a^2); p = 3.5 with
  eps = 100k, a = 0.44 W, k = 1.4 gives open holes, tips ~3 chars, a clear 45-degree X.
- Level-set bands cannot have uniform stroke: thickness = 2 eps / |grad S|, zero gradient at the
  crossing. Bolder strokes make the knot *relatively* smaller (ratio ~ sqrt(a / stroke)).
- Sizes: text_infinity 471 B (32 glyphs; the polynomial needs a chain per group, ~+30 B over
  the ellipses), text 346 B. Removed the ellipse mask code (git history has it).

## 2026-09-04 04:40 — de-glooping the crossing
- User: "it looks so gloopy omg lolol". The level-set knot was a blob with flaring arms.
- Added channel L = 8|v| - 5|u| (linear, separable): inside |u| < 451 px cells are drawn iff
  |L| <= 1027, i.e. two straight strokes of constant width crossing at 2*atan(5/8) = 64 deg; the
  lobes keep the polynomial band. Line slope and width are computed from the polynomial band's
  v-span at the reach so the two bands meet without a jog (matching the *hole* edge matters:
  at a smaller reach the hole has only just opened and the inner edges would not line up).
- text_infinity 524 B (+53 B: one channel, per-group gate with the update subtree 4x).

## 2026-09-04 05:30 — infinity v2: CRT look
- User: "green-amber CRT-like scanlines in foreground and background, with a subtle glow on the
  letters and a vignette for the monitor boundarie". v1 kept as is.
- Glow as a causal phosphor trail (class = lit ? 3 : W - 1) plus a 1 px rim before each glyph
  column, enabled by computing the pattern one pixel early and peeling on a column's last pixel
  (lit there = "W > 2"). Verified numerically: 120, 215x4, 120, 60, 8 across a glyph row.
- Tube: superellipse n = 4 with 256-px chords gave a straight chamfered corner; n = 3 with 128-px
  chords looks rounded. Vignette as three brightness bands + black.
- Colour via RCT 3 constants: G = R + 25, B = R - 255; outside the tube R = -25 so G = 0.
- text_infinity_v2 733 B (v1 524 B): +3 channels (18 B) and ~145 nodes.
