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

## 2026-09-04 05:50 — vignette removed
- User: "plz nix the crt boundary effects like the vignette its not smooth and seems to have added
  a lot of bytes". Removed the tube field channel and the brightness bands: v2 733 -> 602 B.
- Lesson: a vignette needs a smooth brightness ramp, but a tree can only emit a handful of
  discrete values per region (each level multiplies the leaf table), so it bands; and the
  superellipse field costs ~50 nodes of chord slopes + 6 B. Not worth it here.

## 2026-09-04 06:40 — equations (a CFG)
- User asked whether a context-free grammar is possible; then: random equations with matched
  parentheses, function names written downward. Key realisation: Dyck-1 needs only a depth
  counter, and encoding S = 8*class + depth with `W + constant` transitions keeps the automaton
  at ~45 nodes. End-of-line forcing with x thresholds guarantees balance.
- MISTAKE: the piece inherited the vertical flip, so names hung upward and the top line's
  letters were cut; the row-type cycle also left the last hanging rows incomplete. Fixed with a
  `flip` parameter on the pattern channel and EQ_RT_INIT = 1.
- MISTAKE: a patch script matched the shared header block twice and aborted without writing;
  rebuilt from the old generator before noticing. Unique anchors now.
- Verified mechanically (art/experiments/verify_equations.py): all 20 lines well formed.
- equations 534 B, equations_crt 603 B.

## 2026-09-04 08:00 — equations v2 (inline names, syntax colours) and a decoder-validity lesson
- User: "use a dot not a *", "can we syntax-highlight it? can any way to make sin and tan inline
  instead of the columnar thing? new version".
- LESSON (cost me an hour): a tree with a split whose outcome is fixed by an ancestor split on
  the same property encodes but does not decode. My new function-room threshold coincided with
  the existing one in v1 (both 6 cells), nesting `x > 928` inside `x > 928`. Bisected by
  toggling code paths; fixed generally with `simplify()` in the DSL.
- MISTAKE: v2's token channel kept BLANK in the xm == 0 column, so inline word letters copied a
  blank instead of the previous glyph ("S   "); the read-back verifier caught it. Copying W there
  then leaked a 0 (= "(") along the top partial band; a y == 0 guard fixed that.
- MISTAKE: the verifier thresholded luminance at 127 and read grey parentheses (120) as blank;
  now max(R,G,B) > 60, and parens were brightened to (170,170,170).
- Sizes: equations 559 B, equations_crt 630 B, equations_v2 671 B. Both verified.

## 2026-09-04 08:40 — v2 gap, v3 one-per-line
- User: "one equation per line in another forked version? and add some space between them in the
  dense version". A line cannot span decoder groups (no state crosses the boundary), so v3 is
  the single-group 1024x1024 canvas (30 lines of 64 tokens); v2 keeps two per row with a `gap`
  of 4 blank cells at the end of each group.
- MISTAKE: adding EXP made "E" the first glyph in v1's order, and the successor lookup treated a
  blank neighbour (-1) like index 0, filling hanging rows with "X"; guarded with `prop > -1`.
- Sizes: v2 675 B, v3 658 B, v1 561 B, crt 630 B. All verified.

## 2026-09-04 09:10 — v2 vertical spacing
- User repeated "add some space between them in the dense version"; read as vertical too. Added
  `row_gap`: a row-type channel (period row_gap + 1) blanks the state on spacer rows. v2 now has a
  blank row between lines (15 per group) plus the 4-cell horizontal gap: 698 B. Verified 30/30.

## 2026-09-04 09:25 — v2 still read as a wall of text
- User: "this is a wall of text i cant see where one equation starts and another ends i want more
  space between them". Rendered gap/row_gap variants (8/2, 12/2, 12/3; all 698 B since only
  thresholds change) and chose 12/2: two clear columns, 10 lines per group. 12/3 looked empty.

## 2026-09-04 09:40 — random-length equations
- User: "can we make the equations themself shorter, only 1 empty-space between lines", then
  "actually, make the equations of random length". Added `random_length`: in expect_operator at
  depth 0, past EQ_MIN_CELLS, V > EQ_END_V sends the state to BLANK. Row gap back to 1, gap 12.

## 2026-09-04 10:05 — equations_v4: one "=" per line, parens sized by depth
- User: "not all lines hace exactly one = now" then "make that a  neww version. also can we make
  outer parentheis bigger and bigger". Design notes: the phase ("=" written yet) must live in S
  because no later channel can feed back into S within a row; storing it as the low bit under
  2*depth keeps every class threshold and W+constant transition intact (only depth thresholds
  scale). Tall parens can only extend below the decision row, so v4 decides tokens at row 19 of the
  odd band and draws in the even band plus the top of the next odd band; the frame is centred so
  parens grow symmetrically around the text.

## 2026-09-04 10:40 — v4 at 4K
- User: "plz make it 4k (i.e. more equations same relative size same filesize)". Measured with a
  g-reading debug tree: ids 21..24 / 25..28 for 4096x2048, +15 B for 8 groups vs 2. Chose
  4096x2048 over 3840x2160 (partial groups break the grammar's forced closes and clip parens).
  Per-group seeds via a chain over g; the two-group pieces keep their old seeding (bytes unchanged).
- First 4K build: 7 of 8 tiles identical and degenerate ("3^3^3^..."): the seed row's x split had
  its branches swapped (If takes `then` when prop > split), so x > 0 got the constant corner value.
  Caught by the pairwise tile comparison, not by the grammar verifier (a constant CA still yields
  well-formed equations). Lesson: always check tiles pairwise when adding groups.

## 2026-09-04 11:10 — jxl_rs logo
- User: "for the infiinity -letter one if you can get it to say jxl-rs and use rust-like colors you
  won the jxl-rs logo contest retroactively xD". New piece jxl_rs: the lemniscate mask with a
  per-run letter counter instead of the CA (uppercase JXL-RS, the 3x5 font has no lowercase),
  two Rust tones via RCT 3 deltas split on the counter.
- jxl_rs built: 394 B, 287 runs read back, all spell JXL-RS except 8 that cross x = 1024 (restart at J).

## 2026-09-04 11:30 — jxl_rs_crt
- User: "Very nice, I do miss the scanlines / An old terminal orange scanline pattern would go hard".
  Reused the CRT channels (sl, C, brightness) with G = R - 105 / -50 by word and B = 0.

## 2026-09-04 11:50 — chromatic aberration on jxl_rs_crt
- User: "add a lil chromatic abberatin plz". A channel cannot read a neighbour of an earlier channel,
  so a real shift is impossible; instead the glow class (which already encodes "pixel before lit"
  and "pixels after lit") drives per-channel brightness: red full on the rim, green/blue full on the
  first trail pixel. Rim became class 4 to tell it from the first trail pixel (both were 2).

## 2026-09-04 12:30 — quotes (madlibs grammar)
- User: "take that equation generator and turn it into a madlibs-style thing ... randomly generate
  pseudo-intellectual philosophical quotes"; "do it right"; notify when done. Budget analysis: the
  letter table costs ~2 nodes per letter (~2.8 B), the grammar ~100 nodes, glyphs ~130, so the
  vocabulary is limited to ~35 words of <= 7 letters. Word ends are one (length, class) table
  because the vocabulary is sorted by length then class; the blank cell between words carries a
  class marker so each grammar edge stores its target word list once.
- The fork session's worktree (.claude/worktrees/primes) got committed as an embedded repo by
  `git add -A`; removed from the index and ignored.

## 2026-09-04 13:30 — quotes v2 (1 px font)
- First build (PIXEL 4, one quote per line, 977 B) rendered "DEATH ." because the full stop was
  chosen at the blank marker cell; v2 decides it at the noun's last letter. User then asked for a
  1-px font, tighter lines, more quotes and a bigger vocabulary: geometry made switchable
  (apply_geometry), slots of 64 cells with a slot counter channel, VERB/PREP share a marker, and the
  adjective decision moved to the determiner's last letter so each word list is stored once.
- Forked a subagent for the primes piece (user: "fork a subagent to do the prime thing").
- Quotes v2 measured: 60 words = 1340 B (1063 nodes); letters cost ~2.5 B each and the word-choice
  chains ~150 B. A bug had doubled the token table (line_start = the whole decision tree). v3: S builds
  its own random bits so picks are single N + base leaves; V channel dropped; vocabulary class-major.

## 2026-09-04 12:55 — primes (fork)
- The obvious sieve (counter n mod d compared with the divisor through NW-N) marks n = d as composite
  too; two encodings were tried before settling on the down-counter + activation wave (a diagonal
  y - x >= R0, exact because block width = band height). Lesson: when a counter needs "skip the
  first cycle", gate it spatially instead of encoding a phase in its value.
- Digits: a shared pattern table needs the same channel value at each glyph start; alternating
  ones/tens along the row with WW and choosing odd/even start columns (pitch 5) made one table serve
  two glyphs; a third would need another trick (hundreds is a fixed "1" from band 100).
- First bar build showed count - 1 units: BAR_X itself is 0 mod 30, so the countdown ticked on the
  bar's first pixel; the countdown now starts one unit later. Caught by verify_primes.py.
- First render showed "02", "03" and partial "0" glyphs in rows 0..3: the tens glyph is now blanked
  below band 10 and the label above band 1 (in L, 2 y tests) instead of duplicating the pattern table.
- Left half of the width was empty (proper divisors end at n / 2): moved the label to 510 and added the
  divisor-count bar chart, which also makes primes readable from afar.

## 2026-09-04 14:40 — primes_wall
- User asked for a densely packed wall of only the primes. Analysed at length: dense packing needs
  a stall or queue fed back from the primality test into the divisor counters on the next pixel,
  and a channel can only read its own neighbours and earlier channels at the same pixel, so it is
  not expressible (PrevErr turned out to be the clamped-gradient error, not the leaf offset, so it
  cannot smuggle a previous pixel's flag either). Built the closest honest piece: a dense grid of
  all integers with primes bright and composites faint (902 B, 5000 numbers, verified 0 wrong).
- Bugs on the way: Prev only reaches 19 channels back (channel order redesigned around it); the
  "start counters at 1 - 2d so n = d is not a hit" trick breaks when column 0 jumps by 50 modularly
  (the value never leaves the negative regime) — replaced by testing only 2, 3, 5, 7 for n <= 100.

## 2026-09-04 16:30 — primes_wall_v2 and primes_wall_4k
- "16x more primes" is not reachable: divisors up to 283 (59 channels) cost ~2.4 KB. Built the largest
  fits: v2 (6125 numbers, 994 B) and 4k (4096x2048, 33600 numbers, 1945 B under 2 KB).
- Bugs: Prev reaches only 19 channels back (blocks with their own Q/E/M; later Qs read the previous Q;
  the digits read the cell counter, which carries the countdown flag for multi-group walls); the group
  countdown was off by one (G - 1 steps, so group 1 showed 1, 2, 3 .. with wrong colours — the user
  spotted a bright 4 and 12); the first generalised layout cost +100 B on the original wall until
  single-group walls went back to one 19-counter block with a y test instead of a region channel.
- Edge bug: a partial extra number was drawn past the last full cell of each row (and duplicated the
  next row's first number); D is now blank there and the verifier checks the edge. Sizes: {'primes_wall': 959, 'primes_wall_v2': 1013, 'primes_wall_4k': 1918}.
