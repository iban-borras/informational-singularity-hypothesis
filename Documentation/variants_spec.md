# HSI Level 0 — Variant Degradation Models

**Date:** 2025-12-13
**Status:** 7 active variants (B, D, E, F, G, H, I)

## Update: Algorithm Corrections & Variant I

**Key changes (Dec 2025):**
- **ADDED Variant I** — Inverse of E (tests pass-order asymmetry)
- **FIXED all simplification rules** — Degradation to No-Res (`→0`), never deletion (`→∅`)
- **FIXED Variants E and I** — Two-phase degradation with accumulation between phases

**Eliminated variants:**
- **Variant A** — Incompatible with stratified order principle (fundamental to HSI)
- **Variant C** — Incompatible with micro-state reality principle (ontological commitment of HSI)

**Active variants:** B, D, E, F, G, H, I (all scientifically aligned with the HSI paradigm)

---

## Core Framework

All variants operate on the same Level 0 framework: an **ordered cascade degradation** where each Absolute (symbol `1`) encapsulates all previous states and degrades them sequentially. Parentheses denote stratified containment. The observable for iteration α is produced from a decay frame built from the accumulation of all prior observables.

**HSI Core Principle:**
> "Order is the only way Nothingness can contradict itself. Reality emerges from primordial order through stratified containment of the Absolute."

## Markwon-ready figure (Absolute integration sequence)

Note: To render the math below with Markwon you need the Math plugin enabled. The LaTeX figure environment is omitted; use this block directly.

$$
\begin{aligned}
&\begin{array}{ccccccccccccc}
0 & & 1 & & 0 & & 1 & & 1 & & 0 & & \cdots \\
\emptyset_0 & \rightarrow & |\emptyset_0| & \rightarrow & \emptyset_0 & \rightarrow & |\emptyset_0 & \rightarrow & |\emptyset_0| & \rightarrow & \emptyset_0| & \rightarrow & \cdots \\
& \perp & & \perp & & \perp & & \perp & & \perp & & \perp &
\end{array}
\end{aligned}
$$

_Figure: Informational Singularity sequence with progressive integration of the Absolute. Absoluteness states ($|\emptyset_0|$, $|\emptyset_0|$, $\emptyset_0|$) indicate increasing integration of prior states folded into themselves._

<!-- ASCII fallback (for environments without math rendering) -->
```
0     1       0     1      1       0      ...
Ø0 -> |Ø0| -> Ø0 -> |Ø0 -> |Ø0| -> Ø0| -> ...
   ⟂       ⟂    ⟂      ⟂      ⟂
```

Alternatives:

```
0     10      0     10     10      0      ...
Ø0 -> |Ø0| -> Ø0 -> |Ø0 -> |Ø0| -> Ø0| -> ...
   ⟂       ⟂    ⟂      ⟂      ⟂

0     01      0     01     01      0      ...
Ø0 -> |Ø0| -> Ø0 -> |Ø0 -> |Ø0| -> Ø0| -> ...
   ⟂       ⟂    ⟂      ⟂      ⟂
```
### Why the Absolute matters (ABS modes)

The Absolute token appended to the outer decay frame controls how prior content interacts at the boundary. We consider three semantics:
- ABS = "1" (symmetric baseline): minimal injection, good as control.
- ABS = "10" (directional 1→0): introduces asymmetry likely to bias cancellations at the boundary.
- ABS = "01" (directional 0→1): the dual of 10, used to test symmetry.

In our streaming collapse, the base simplification (degrade 01→0, 10→0, compress runs) may partially consume these boundary tokens depending on the frontier of the accumulated state. This is precisely what can seed or suppress emergent order. Exposing ABS lets us test whether φ‑alignment emerges robustly or depends on a specific boundary orientation.


## Common framework (applies to all variants)

Notation:
- Accα: accumulation string after iteration α−1 (concatenation of all previous Φ values)
- Dα: decay frame used at iteration α
- Φα: observable result at iteration α

Per iteration α ≥ 1:
1) Update accumulation
   - `Accα ← Accα−1 + Φα−1` (for α=1, Acc0 is the initial seed, typically `"0"`)
2) Build decay frame
   - `Dα = (Accα)1`  — Absolute (1) contains every prior state in stratified parentheses
3) Collapse using the variant‑specific procedure on `Dα`
4) Finalize observable
   - Return the resulting string as `Φα`; variants may apply a final global simplify to enforce Level‑0 observability

Base simplification (used unless a variant overrides it):
- Degrade alternating pairs to No-Res: replace every `01` and `10` with `0` (not empty!)
- Compress runs: replace `0+ → 0` and `1+ → 1`
- Iterate until stable (no more patterns to degrade)
- Always return at least `0` (No-Res) — never empty string
- These operations are pure string rules; they do not depend on frequency or time.

**IMPORTANT (Dec 2025 fix):** According to HSI ontology, No-Res (`0`) is the fundamental state.
Nothing ever truly disappears — it degrades to No-Res. The simplification result is never empty.

**Philosophical note on `1+ → 1`:** One might argue that each Absolute (`1`) contains unique history
and compressing `111 → 1` loses information, unlike `000 → 0` (merging identical No-Res). However,
in HSI's real execution flow, No-Res is always present (we start from R₀ = `0`) and absorbs all
adjacent Absolutes. Thus, `1+ → 1` compression never affects the final result — the primordial `0`
guarantees eventual collapse to No-Res regardless. We keep the simpler symmetric rule for pragmatic reasons.

Implementation anchors:
- Main driver: `level0/generator.py`
- Helper names: `_collapse_global_ignore_parentheses`, `_collapse_inside_parentheses`, `_collapse_inside_parentheses_local`, `_collapse_inside_parentheses_one_local`, `_simplify_base`, `_simplify_variant_d`, `_simplify_variant_e`

## Variant B — Stratified inside→out (local‑only) + global finalize ⭐ GOLD STANDARD

**Scientific Value:** Gold standard for HSI. Fully compatible with all core principles.

**Goal:** Preserve structural order during degradation; only the innermost contents are simplified at each pass.

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) Repeat until stabilization or 1‑symbol state:
   - Replace every innermost `( ... )` by simplifying its contents only (outer structure untouched)
   - **Accumulate intermediate states** (ontological reality of micro-states)
3) Finalize with one global simplify of the whole state to get `Φα`

**Properties:**
- ✅ Preserves stratified order (inside→out collapse)
- ✅ Accumulates ALL micro-states (ontologically real)
- ✅ Produces clean Level 0 observable
- ✅ **Validated as ontologically correct**

**Why it survives:** Fully aligned with principle that "order emerges through stratified containment."

---

## Variant D — Minimal asymmetry (simultaneous 10→0, 01→0) with stratified passes

**Scientific Value:** Tests whether symmetry-breaking affects order emergence.

**Goal:** Introduce a minimal, directional asymmetry in the local rule and study its global effect.

**Custom simplify `E_d`:**
- Degrade `10 → 0` and `01 → 0` simultaneously (both to No-Res)
- Then compress runs `0+→0`, `1+→1`
- Iterate until stable

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) Repeat until stabilization or 1‑symbol state:
   - Collapse innermost parentheses using `E_d`
   - **Accumulate intermediate states**
3) Final global simplify with `E_d`
4) Output the stabilized state as `Φα`

**Properties:**
- ✅ Preserves stratified order
- ✅ Accumulates ALL micro-states
- ✅ Both patterns degrade to No-Res (HSI ontology: nothing disappears)
- ✅ Tests robustness of order emergence to asymmetric rules

**Why it survives:** Tests legitimate scientific hypothesis: "Is order emergence robust to symmetry-breaking?"

---

## Variant E — Ordered two‑phase degradation: all `01→0`, then all `10→0`

**Scientific Value:** Tests whether pass order affects order emergence.

**Goal:** Explore the effect of fixed ordered phases across the contents before run compression.

**Custom simplify `E_e` (two phases with accumulation between them):**
- Phase 1: Degrade all `01 → 0`, compress runs, iterate until stable, **accumulate**
- Phase 2: Degrade all `10 → 0`, compress runs, iterate until stable, **accumulate**

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) **Phase 1** - Repeat until stabilization:
   - Collapse innermost parentheses using only `01→0` rule
   - **Accumulate intermediate states**
3) **Phase 2** - Repeat until stabilization:
   - Collapse innermost parentheses using only `10→0` rule
   - **Accumulate intermediate states**
4) Output the stabilized state as `Φα`

**Properties:**
- ✅ Preserves stratified order
- ✅ Accumulates ALL micro-states (including between phases)
- ✅ Enforces deterministic phase order (emergence before collapse)
- ✅ Tests whether sequence of simplifications matters

**Why it survives:** Tests legitimate scientific hypothesis: "Does pass order affect emergent structure?"

---

## Variant F — Hybrid: stabilize inside→out, then single global pass

**Scientific Value:** Tests hybrid stabilization strategy (stratified + final global closure).

**Goal:** Combine structural stabilization with a final global closure.

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) Repeat until stabilization or 1‑symbol state:
   - Collapse innermost parentheses (using Base simplify)
   - **Accumulate intermediate states**
3) If the result is longer than 1 symbol, apply one global simplify
4) Output as `Φα`

**Properties:**
- ✅ Preserves stratified order (until stable)
- ✅ Accumulates ALL micro-states
- ✅ Final global closure (single pass)
- ✅ Positioned between B and A in terms of global intervention

**Why it survives:** Tests legitimate scientific hypothesis: "Does a single final global pass affect order differently than continuous global passes?"

---

## Variant G — Stratified without final global (ABS‑aware)

**Scientific Value:** Preserves raw stratified structure for analysis.

**Goal:** Preserve the stratified outcome strictly; avoid closing effects of a final global simplify.

**Algorithm (per iteration):**
1) Build `Dα = (Accα)ABS` with ABS ∈ {"1","10","01"}
2) Repeat until stabilization or 1‑symbol state:
   - Replace innermost `( … )` simplifying contents only
   - **Accumulate intermediate states**
3) Output stabilized state as `Φα` (⚠️ **no final global**)

**Properties:**
- ✅ Preserves stratified order
- ✅ Accumulates ALL micro-states
- ⚠️ No final cleaning (may contain parentheses in observable)
- ✅ Purely stratified outcome

**Why it survives:** Unique value in preserving **raw stratified structure** before any global closure. Useful for studying persistence of inner order.

---

## Variant H — Temporal with per‑tick global soft‑closure (ABS‑aware)

**Scientific Value:** Tests continuous global feedback effects.

**Goal:** Combine fine temporal resolution with gentle global closure at each step.

**Algorithm (per iteration):**
1) Build `Dα = (Accα)ABS`
2) Loop:
   - One innermost replacement (simplify contents only)
   - **Accumulate intermediate states**
   - If result length > 1, apply one global simplify (soft closure)
   - Stop on stabilization or 1‑symbol state
3) Output as `Φα`

**Properties:**
- ⚠️ Inside→out with per-tick global interference
- ✅ Accumulates ALL micro-states
- ✅ Continuous global feedback
- ✅ Positioned between B and A

**Why it survives:** Tests legitimate scientific hypothesis: "Does continuous global feedback affect order emergence?"

---

## Invariants and guarantees

**All surviving variants (B, D, E, F, G, H) satisfy:**
- ✅ **Ordered cascade:** every `1` encapsulates all prior states in stratified order `(Accα)1`
- ✅ **Determinism:** for a given variant and input, the process is deterministic
- ✅ **Monotone information recording:** Accumulation grows with iterations; Φα is always computed from all prior Φ
- ✅ **Micro-state reality:** All variants accumulate intermediate states (ontologically real)
- ✅ **Stratified order preservation:** All variants collapse inside→out (respecting nesting depth)
- ⚠️ **Level 0 observability:** Most variants produce clean observable; Variant G may contain parentheses

**HSI Alignment:**
All surviving variants are **scientifically compatible** with the core principle:
> "Order is the only way Nothingness can contradict itself."

---

## Variant I — Inverse of E: all `10→0`, then all `01→0`

**Scientific Value:** Tests whether pass order asymmetry affects order emergence (inverse of E).

**Goal:** Explore the inverse effect of E: collapse before emergence.

**Custom simplify `E_i` (two phases with accumulation between them):**
- Phase 1: Degrade all `10 → 0`, compress runs, iterate until stable, **accumulate**
- Phase 2: Degrade all `01 → 0`, compress runs, iterate until stable, **accumulate**

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) **Phase 1** - Repeat until stabilization:
   - Collapse innermost parentheses using only `10→0` rule
   - **Accumulate intermediate states**
3) **Phase 2** - Repeat until stabilization:
   - Collapse innermost parentheses using only `01→0` rule
   - **Accumulate intermediate states**
4) Output the stabilized state as `Φα`

**Properties:**
- ✅ Preserves stratified order
- ✅ Accumulates ALL micro-states (including between phases)
- ✅ Enforces deterministic phase order (collapse before emergence)
- ✅ Tests whether sequence of simplifications matters (inverse of E)

**Why it survives:** Tests legitimate scientific hypothesis: "Is the effect of pass order symmetric or asymmetric?"

---

## Future candidates (J–Z, not implemented yet)

These are design sketches to help plan extensions; they are not active in the codebase.
- J — Windowed finalize: global simplify applied only to a sliding window over the outermost structure
- K — Asymmetric runs: compress `0+` and `1+` with different policies (e.g., thresholds) without adding randomness
- L — Multi‑Absolute frame: `Dα = (Accα)11` to test double‑encapsulation pressure
- M — Alternating finalizer: finalize every k iterations instead of every iteration
- … (reserve N–Z for additional controlled decay models consistent with the HSI principles)

---

## Change log (relevant to this spec)

### 2025-12-13
- **ADDED Variant I:** Inverse of E (10→0 first, then 01→0) for pass-order asymmetry testing
- **FIXED E and I algorithms:** Two-phase degradation with accumulation between phases (not simultaneous)
- **FIXED all variants:** Degradation to No-Res (`→0`), never deletion (`→∅`). HSI ontology: nothing disappears.
- **7 active variants:** B, D, E, F, G, H, I

### 2025-11-22
- **ELIMINATED Variant A:** Incompatible with stratified order principle (fundamental to HSI)
- **ELIMINATED Variant C:** Incompatible with micro-state reality principle (ontological commitment)
- **ADDED Variants G & H:** Experimental variants for testing raw structure preservation and continuous feedback
- **UPDATED all descriptions:** Added scientific value, "why it survives" rationale, and alignment notes
- **6 surviving variants:** B (gold standard), D (asymmetry), E (pass-order), F (hybrid), G (raw structure), H (feedback)

### 2025‑08‑21
- B and C switched to local‑only simplification inside parentheses, followed by a single global simplify at the end of each iteration (ensures Level 0 observability while preserving order during the stratified phase).

## Cross‑reference (implementation hints)

**Active variants:**
- **B:** `_collapse_inside_parentheses_local` + final `_collapse_global_ignore_parentheses`
- **D:** `_collapse_inside_parentheses_local(..., simplify_fn=_simplify_variant_d)` + final global
- **E:** Two-phase: `_simplify_variant_e_phase1` (01→0), then `_simplify_variant_e_phase2` (10→0)
- **F:** `_collapse_inside_parentheses` until steady, then `_collapse_global_ignore_parentheses`
- **G:** `_collapse_inside_parentheses_local` (no final global)
- **H:** `_collapse_inside_parentheses_one_local` + per-tick `_collapse_global_ignore_parentheses`
- **I:** Two-phase: `_simplify_variant_i_phase1` (10→0), then `_simplify_variant_i_phase2` (01→0)

**Eliminated variants:**
- ~~**A:**~~ `_collapse_global_ignore_parentheses`
- ~~**C:**~~ `_collapse_inside_parentheses_one_local` + final global

