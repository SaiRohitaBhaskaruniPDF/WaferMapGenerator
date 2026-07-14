# Synthetic Wafer Map Generator — Spec vs. Implementation Audit

This sheet reproduces the specification text and, under each item, answers:
**Implemented?**, **Where** (file / function), and **How**.

Legend: ✅ Implemented · 🟡 Partial · ⬜ Not yet (by design / placeholder)

---

## Goal
Generate synthetic wafer maps for demonstration, benchmarking and Software QA purposes.

> **Status:** ✅ The whole pipeline lives in `generator.generate()`, which both the
> Streamlit manual form (`app.py`) and the chat agent (`llm_agent.py`) call, so
> both entry points produce identical data. Exports: die-level CSV, per-test CSV,
> PNG/SVG/JPEG/TIFF wafer maps, and STDF per lot per insertion.

---

## Parameters — Must Have

### 1. Wafer diameter — 150 mm, 200 mm, 300 mm
- **Implemented:** ✅
- **Where:** `geometry.py` → `STANDARD_DIAMETERS`, `snap_diameter()`; UI `app.py` diameter selectbox.
- **How:** Only the three legal sizes are offered in the form. Fuzzy/chat input is
  snapped to the nearest legal size. Diameter becomes `radius = diameter/2`, which
  bounds the die grid in `compute_die_grid()`.

### 2. Edge Exclusion — 1 mm to 10 mm
- **Implemented:** ✅
- **Where:** `geometry.py` → `EDGE_EXCLUSION_MIN/MAX`, used in `compute_die_grid()`; UI slider in `app.py`.
- **How:** The usable radius is reduced: `active_radius = radius - edge_exclusion`.
  A die is kept only if its center is within `active_radius`.

### 3. Die size — 1×1 mm to 25×35 mm, aspect 1:2–2:1, scribe street
- **Implemented:** ✅ (incl. the "nice to have" variable street 0.05–0.2 mm)
- **Where:** `geometry.py` → `validate_die_size()`, `clamp_die_size()`, `clamp_street()`,
  `STREET_MIN/MAX/DEFAULT`; `WaferConfig.pitch_x/pitch_y`.
- **How:**
  - Size box: short side ≤ 25 mm, long side ≤ 35 mm (`DIE_MAX_SHORT_MM`, `DIE_MAX_LONG_MM`).
  - Aspect ratio checked against `ASPECT_MIN=0.5` (1:2) and `ASPECT_MAX=2.0` (2:1),
    so 6×3 and 3×6 pass, 12×3 fails.
  - Scribe street default 0.1 mm, clamped to 0.05–0.2 mm. Die spacing =
    `pitch = die size + street`.
  - `validate_die_size()` is used by the strict manual form; `clamp_*` quietly
    corrects fuzzy chat/LLM input.

### 4. Flat vs Notch — auto-select (150 = flat, 200/300 = notch)
- **Implemented:** ✅
- **Where:** `geometry.py` → `auto_edge_type()`; drawn in `renderer.py`.
- **How:** `return "flat" if diameter <= 150.0 else "notch"`. The user never picks
  this; it is derived from diameter.

### 5. Wafer orientation — flat/notch bottom/right/left/top, 90° only
- **Implemented:** ✅
- **Where:** `geometry.py` → `WaferConfig.edge_orientation` (`down|up|left|right`);
  UI radio in `app.py`; chat parser `llm_agent._parse_edge_orientation()`;
  rendered in `renderer.py` (`_NOTCH_PARAMS`, `_FLAT_PARAMS`).
- **How:** A single orientation string is threaded UI → config → renderer. The
  renderer places the notch/flat on the requested side (90° increments only).

### 6. Wafer Quantity — 25 max, any 1..24
- **Implemented:** ✅
- **Where:** `llm_agent.py` → `MAX_WAFERS = 25`, `LOT_SIZE_PRESETS`, `_clamp_wafers()`;
  UI lot-size preset + custom slider in `app.py`; loop in `generator.generate()`.
- **How:** FOUP-aware presets (25 standard, 13 thin/bonded, or custom partial lot).
  Any request is clamped to 1–25. Generation loops `for w in range(req.num_wafers)`.

### 7. Wafer numbers — sequential 1..25
- **Implemented:** ✅
- **Where:** `generator.py` → `WaferResult.number`, `wafer_id`.
- **How:** `number = w + 1`; ID is `f"{lot_id}_{w+1:02d}"` (e.g. `LOT_001_01`).
  Flows to the CSV as `WaferNumber`.

### 8. Yield — direct % or defect density Y = e^(-A·D)
- **Implemented:** ✅ (defects/in² correctly ignored, per spec)
- **Where:** `yield_model.py` → `poisson_yield()`, `resolve_target_yield()`,
  `apply_yield_target()`; `geometry.py` → `WaferConfig.die_area_cm2`; UI yield-mode radio.
- **How:** Three modes: `signature` (whatever the pattern gives), `direct`
  (use the % as-is), `defect_density` (compute `Y = e^(-A·D)`, A = die area in cm²).
  `apply_yield_target()` then kills/revives random dies to hit the number while
  keeping the visual pattern intact.

### 9. Test insertions — CP1 / CP1+CP2 / CP1+CP2+CP3, retest cascade
- **Implemented:** ✅
- **Where:** `yield_model.py` → `INSERTION_TEMPS`, `RETEST_SURVIVAL_MIN/MAX`,
  `cascade_insertions()`; `generator.py` (num_insertions clamp + CP1 from yield);
  `signatures.py` → `CP2_FAIL_BIN=31`, `CP3_FAIL_BIN=32`; UI radio in `app.py`.
- **How:**
  - CP = Circuit Probe; temps labeled CP1 25 °C, CP2 −40 °C, CP3 125 °C.
  - CP1 P/F comes from the yield model + signature.
  - CP2 keeps 90–99.9% of CP1 passers; CP3 keeps 90–99.9% of CP2 passers.
  - Only prior passers are eligible (`if d[4] == PASS_BIN`).
  - Survival is always ≤ 0.999, so CP2 can never beat CP1 (the rare pathology is ignored).
  - Dies lost at CP2/CP3 get distinct bins so the loss is visible on the map.

### 10. Number of bins — hardbins 16/64/256, softbins ×4/×16/×64
- **Implemented:** ✅ (softbin→hardbin mapping intentionally loose, per spec)
- **Where:** `binning.py` → `HARDBIN_CHOICES`, `SOFTBIN_MULTIPLIERS`, `build_bin_map()`,
  `map_wafer_bins()`; UI selectboxes.
- **How:** Internal signature bins (1..33) are translated to the chosen hardbin/softbin
  space. PASS → (1,1). Fail causes cycle through `2..hardbin_count` (so with 16
  hardbins the ~32 fail causes share bins; with 256 each gets its own). Softbin is
  derived deterministically from the hardbin block.

### 11. Test items — count, P/F vs parametric, data shapes, names
- **Implemented:** ✅ (incl. the "nice to have" verbose naming)
- **Where:** `test_items.py` → `TEST_COUNT_CHOICES`, `TestPlan`, `VALUE_SHAPES`,
  `_shaped_value()`, `make_test_names()`, `generate_die_results()`,
  `estimate_result_count()`; UI "Test items" expander.
- **How:**
  - **11a Count:** orders of magnitude only — `(100, 1000, 10k, 100k, 1M)`.
  - **11b P/F vs parametric:** split by `parametric_pct` (default 50, 10% steps).
    Pass/Fail reports 0.0/1.0; parametric reports a real value from the RNG.
    (Note: values are currently **uniform**, not Gaussian — a documented simplification.)
  - **11b-2 Data shapes:** `uniform`, `exponential` (10^rng), `quantized` (0.2 steps),
    `signed` (−1..+1), `scientific` (X.XXe±YY), `constant` (one value).
  - **11c Names:** `simple` (PARAM_0001, zero-padded), `obnoxious` (long fixed prefix,
    differs only at the end — a UI checker), `chunked` (8-char gibberish chunks so the
    front differs too). Verbose lengths 31/63/127/255.

### 12. Yield Patterns
- **12a Steve's list:** 🟡 *Your doc placeholder.* The code already implements **37
  named spatial signatures** in `signatures.py` → `SIGNATURE_NAMES` (edge ring, center
  cluster, donut, bull's-eye, scratch families, quadrant, wedge, spokes, rings, etc.).
  These are selectable individually or layered (`compose_signatures()`). **Action for
  you:** paste Steve's actual list here and cross-check against `SIGNATURE_NAMES`.
- **12b Repeaters:** ✅ — `signatures.py` → `assign_reticle()` (bin 12);
  `geometry.py` → `auto_stepping_field()` auto-derives the stepping field from die size.
  Same die position (dieX % dpr_x, dieY % dpr_y) fails in every field.
- **12b-iii-1 Soft repeaters:** ✅ — `WaferConfig.repeater_fail_rate` (10–100% via UI
  slider) passed to `assign_reticle(fail_rate=...)`. 1.0 = hard, <1.0 = soft.
- **12b-iii-2 Striping:** ✅ — `signatures.py` → `assign_stripe()` (bin 30), four
  variants (top/bottom/left/right). Hard or soft via `WaferConfig.stripe_fail_rate`.
  Models lens-tilt yield loss along one field edge.

---

## Nice To Have

### 13. Lot numbers — FYYWWSSSS (+ split suffixes)
- **13a Format:** ✅ — `fab.py` → `make_lot_id()`. `F` fab letter + `YY` year +
  `WW` ISO work week (handles 53-week years via `isocalendar()[1]`) + `SSSS`
  sequential. Wired via `auto_lot_id` checkbox and multi-lot generation.
- **13b Splits / child lots (.01, .02):** 🟡 **Partial.** `make_lot_id()` accepts a
  `split` argument and appends `.NN`, but it is **not yet wired** into the pipeline or
  UI (no way to request a split lot from the form/chat today). Function ready; exposure missing.

### 14. Lot sequence — different sort-start timestamps
- **Implemented:** ✅
- **Where:** `fab.py` → `LOT_CADENCES`, `lot_schedule()`; UI number-of-lots + cadence;
  `generator.generate()` builds one `LotResult` per scheduled lot.
- **How:** Cadence choices (per month/week/day, multiple/day) set the time gap. Lots are
  dated working backwards from now so data looks like recent history for trend charts.
  Each lot's FYYWWSSSS reflects its own date.

### 15. Test Time — 1–600 seconds per touchdown
- **Implemented:** ✅
- **Where:** `fab.py` → `wafer_test_seconds()`; UI slider; used by `stdf_writer.py`
  and timestamps in `generator.py`.
- **How:** `touchdowns = ceil(dies / sites)`, total time = touchdowns × seconds/touchdown.
  Drives per-die elapsed time and wafer start/finish timestamps.

### 16. Multi-site — 1–16 sites, GDPW table, layout patterns
- **Implemented:** ✅
- **Where:** `fab.py` → `_GDPW_SITE_TABLE`, `auto_site_count()`, `SITE_PATTERNS`,
  `_block_dims()`, `assign_sites()`; UI multi-site checkbox + layout selectbox.
- **How:**
  - Site count from Gross Die Per Wafer: <200→1, 200–399→2, 400–799→4, 800–1599→8, 1600+→16.
  - Layouts: side by side, top & bottom, block (2×2 / 2×4), checkerboard, diagonal.
  - Each die is stamped with its site via position inside the repeating site array.
    Checkerboard/diagonal scatter the numbering to ease probe-card wire routing.

### 17. Site-to-Site Yield Loss (S2S)
- **Implemented:** ✅ (matches the suggested method exactly)
- **Where:** `yield_model.py` → `s2s_factors()`, `apply_s2s()`; `signatures.py` →
  `S2S_FAIL_BIN=33`; `generator.py` applies it after the yield target; UI checkbox;
  factors surfaced in `app.py` (lines ~327–330).
- **How:**
  - Actual yield = yield model × S2S factor.
  - S2S factor per site in 0.0–1.0.
  - Healthy: all factors > 0.95 (random-looking across the fixture).
  - Problem: one random site dragged to 0.40–0.80, so its loss is clearly visible.
  - Casualties get bin 33 so the site loss is distinguishable on the map.

### 18. Repair — virgin vs repaired good die
- **Implemented:** ⬜ **Not implemented (placeholder, per spec).**
- **Where:** Only referenced in planning notes
  (`.cursor/plans/wafer_map_spec_implementation_72ae3690.plan.md`); no code in the
  generator, signatures, binning, or exporters.
- **What it would need (future work):**
  - A concept of repairable fails (e.g. memory rows/columns) with spare elements.
  - Split good die into **Virgin good** (passed with no repair) and **Repaired good**
    (passed only after using a spare), plus the resulting ratios.
  - Dedicated bin(s) and CSV/STDF columns so yield analysis can differentiate
    Good / Virgin / Repaired.
  - This is intentionally deferred; the spec itself marks it "requires more thought."

---

## Summary table

| # | Item | Status |
|---|------|--------|
| 1 | Wafer diameter | ✅ |
| 2 | Edge exclusion | ✅ |
| 3 | Die size / aspect / scribe street | ✅ (+ variable street) |
| 4 | Flat vs notch (auto) | ✅ |
| 5 | Wafer orientation | ✅ |
| 6 | Wafer quantity | ✅ |
| 7 | Wafer numbers | ✅ |
| 8 | Yield (direct + defect density) | ✅ |
| 9 | Test insertions (CP cascade) | ✅ |
| 10 | Number of bins | ✅ |
| 11 | Test items (count/shapes/names) | ✅ |
| 12a | Steve's list | 🟡 (paste list; 37 signatures already exist) |
| 12b | Repeaters + auto stepping field | ✅ |
| 12b-iii-1 | Soft repeaters | ✅ |
| 12b-iii-2 | Striping | ✅ |
| 13a | Lot numbers FYYWWSSSS | ✅ |
| 13b | Splits / child lots | 🟡 (function ready, not wired) |
| 14 | Lot sequence | ✅ |
| 15 | Test time | ✅ |
| 16 | Multi-site | ✅ |
| 17 | Site-to-site yield loss | ✅ |
| 18 | Repair | ⬜ (placeholder, deferred by spec) |

**Bottom line:** every Must-Have (1–12) is implemented. Of the Nice-To-Haves,
13a, 14, 15, 16, and 17 are done; only **13b (split-lot exposure)** is partial and
**18 (Repair)** is an intentional placeholder.
