# HSI Level 0 — Variant Degradation Models (v33)

**Version:** v33 (HSI v32 Aligned)
**Date:** 2025-11-22
**Status:** 6 active variants (B, D, E, F, G, H)

## v33 Update: Variant Elimination

**Eliminated variants:**
- **Variant A** — Incompatible with stratified order principle (fundamental to HSI v32)
- **Variant C** — Incompatible with micro-state reality principle (ontological commitment of v32)

**Surviving variants:** B, D, E, F, G, H (all scientifically aligned with v32 paradigm)

---

## Core Framework

All variants operate on the same Level 0 framework: an **ordered cascade degradation** where each Absolute (symbol `1`) encapsulates all previous states and degrades them sequentially. Parentheses denote stratified containment. The observable for iteration α is produced from a decay frame built from the accumulation of all prior observables.

**HSI v32 Core Principle:**
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

In our streaming collapse, the base simplification (delete 01/10, compress runs) may partially consume these boundary tokens depending on the frontier of the accumulated state. This is precisely what can seed or suppress emergent order. Exposing ABS lets us test whether φ‑alignment emerges robustly or depends on a specific boundary orientation.


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
- Remove alternating pairs: delete every occurrence of `01` and `10` (balanced cancellation)
- Compress runs: replace `0+ → 0` and `1+ → 1`
- These operations are pure string rules; they do not depend on frequency or time.

Implementation anchors:
- Main driver: `level0/generator.py`
- Helper names: `_collapse_global_ignore_parentheses`, `_collapse_inside_parentheses`, `_collapse_inside_parentheses_local`, `_collapse_inside_parentheses_one_local`, `_simplify_base`, `_simplify_variant_d`, `_simplify_variant_e`

## Variant B — Stratified inside→out (local‑only) + global finalize ⭐ GOLD STANDARD

**Scientific Value:** Gold standard for HSI v32. Fully compatible with all core principles.

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
- ✅ **Validated as ontologically correct** (HSI v32)

**Why it survives:** Fully aligned with v32 principle that "order emerges through stratified containment."

---

## Variant D — Minimal asymmetry (10→∅, 01→0) with stratified passes

**Scientific Value:** Tests whether symmetry-breaking affects order emergence.

**Goal:** Introduce a minimal, directional asymmetry in the local rule and study its global effect.

**Custom simplify `E_d`:**
- Delete `10`
- Replace `01 → 0`
- Then compress runs `0+→0`, `1+→1`

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) Repeat until stabilization or 1‑symbol state:
   - Collapse innermost parentheses using `E_d`
   - **Accumulate intermediate states**
3) Output the stabilized state as `Φα` (no extra finalization beyond the stratified passes)

**Properties:**
- ✅ Preserves stratified order
- ✅ Accumulates ALL micro-states
- ✅ Introduces directional asymmetry (breaks 0↔1 symmetry)
- ✅ Tests robustness of order emergence to asymmetric rules

**Why it survives:** Tests legitimate scientific hypothesis: "Is order emergence robust to symmetry-breaking?"

---

## Variant E — Ordered two‑pass removal: all `01`, then all `10` (then runs)

**Scientific Value:** Tests whether pass order affects order emergence.

**Goal:** Explore the effect of fixed ordered passes across the contents before run compression.

**Custom simplify `E_e` per pass:**
- Pass 1: delete all `01`
- Pass 2: delete all `10`
- Then compress runs `0+→0`, `1+→1`

**Algorithm (per iteration):**
1) Build `Dα = (Accα)1`
2) Repeat until stabilization or 1‑symbol state:
   - Collapse innermost parentheses using `E_e`
   - **Accumulate intermediate states**
3) Output the stabilized state as `Φα`

**Properties:**
- ✅ Preserves stratified order
- ✅ Accumulates ALL micro-states
- ✅ Enforces deterministic pass order (more aggressive than base)
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

## Invariants and guarantees (v33)

**All surviving variants (B, D, E, F, G, H) satisfy:**
- ✅ **Ordered cascade:** every `1` encapsulates all prior states in stratified order `(Accα)1`
- ✅ **Determinism:** for a given variant and input, the process is deterministic
- ✅ **Monotone information recording:** Accumulation grows with iterations; Φα is always computed from all prior Φ
- ✅ **Micro-state reality:** All variants accumulate intermediate states (ontologically real)
- ✅ **Stratified order preservation:** All variants collapse inside→out (respecting nesting depth)
- ⚠️ **Level 0 observability:** Most variants produce clean observable; Variant G may contain parentheses

**HSI v32 Alignment:**
All surviving variants are **scientifically compatible** with the v32 core principle:
> "Order is the only way Nothingness can contradict itself."

---

## Future candidates (I–Z, not implemented yet)

These are design sketches to help plan extensions; they are not active in the codebase.
- I — Stack‑accurate innermost in O(n): replace regex with an explicit stack machine; same semantics, higher throughput
- J — Windowed finalize: global simplify applied only to a sliding window over the outermost structure
- K — Asymmetric runs: compress `0+` and `1+` with different policies (e.g., thresholds) without adding randomness
- L — Multi‑Absolute frame: `Dα = (Accα)11` to test double‑encapsulation pressure
- M — Alternating finalizer: finalize every k iterations instead of every iteration
- … (reserve N–Z for additional controlled decay models consistent with the HSI principles)

---

## Change log (relevant to this spec)

### v33 (2025-11-22)
- **ELIMINATED Variant A:** Incompatible with stratified order principle (fundamental to HSI v32)
- **ELIMINATED Variant C:** Incompatible with micro-state reality principle (ontological commitment of v32)
- **ADDED Variants G & H:** Experimental variants for testing raw structure preservation and continuous feedback
- **UPDATED all descriptions:** Added scientific value, "why it survives" rationale, and v32 alignment notes
- **6 surviving variants:** B (gold standard), D (asymmetry), E (pass-order), F (hybrid), G (raw structure), H (feedback)

### 2025‑08‑21
- B and C switched to local‑only simplification inside parentheses, followed by a single global simplify at the end of each iteration (ensures Level 0 observability while preserving order during the stratified phase).

## Cross‑reference (implementation hints)

**Active variants (v33):**
- **B:** `_collapse_inside_parentheses_local` + final `_collapse_global_ignore_parentheses`
- **D:** `_collapse_inside_parentheses(..., simplify_fn=_simplify_variant_d)`
- **E:** `_collapse_inside_parentheses(..., simplify_fn=_simplify_variant_e)`
- **F:** `_collapse_inside_parentheses` until steady, then `_collapse_global_ignore_parentheses`
- **G:** `_collapse_inside_parentheses_local` (no final global)
- **H:** `_collapse_inside_parentheses_one_local` + per-tick `_collapse_global_ignore_parentheses`

**Eliminated variants:**
- ~~**A:**~~ `_collapse_global_ignore_parentheses` (removed in v33)
- ~~**C:**~~ `_collapse_inside_parentheses_one_local` + final global (removed in v33)

