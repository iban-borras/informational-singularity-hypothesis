# ISH Level 0 — Variant Specification Guide

## Overview

The ISH computational model tests the core mechanism — **binary annihilation with nesting** — through 7 experimental variants and 5 control baselines. All variants operate on the same framework: an ordered cascade where each Absolute (`1`) encapsulates all prior states in stratified parentheses, and collapse proceeds inside→out.

**Core principle:**
> "Order is the only way Nothingness can contradict itself. Reality emerges from primordial order through stratified containment of the Absolute."

---

## Base Mechanism

All variants share:
- **Decay frame:** `Dα = (Accα)1` — the Absolute contains every prior state
- **Simplification:** degrade alternating pairs (`01→0`, `10→0`), compress runs, iterate until stable
- **Accumulation:** `Accα ← Accα−1 + Φα−1` — monotone information recording
- **Ontological constraint:** degradation always produces No-Res (`0`), never deletion — nothing disappears

---

## Experimental Variants

### Variant B ⭐ Gold Standard
**Stratified inside→out + global finalize**

Collapse innermost parentheses first, accumulate intermediate states, finalize with one global simplify. Fully preserves stratified order.

### Variant D — Asymmetric
**Simultaneous `10→0` and `01→0` with dedicated function**

Tests whether introducing minimal directional asymmetry in the local rule affects order emergence. Same stratified structure as B.

### Variant E — Two-Phase (01 first)
**Phase 1: all `01→0` until stable → Phase 2: all `10→0` until stable**

Tests whether the order of application matters. Accumulates between phases.

### Variant F — Hybrid (with feedback)
**Stratified collapse + single global final pass**

Combines structural stabilization with a final global closure. This single line of code difference (`return sf(collapsed)` vs `return collapsed`) produces a complete regime change across all metrics.

### Variant G — Raw Structure
**Stratified without final global simplify**

Preserves raw stratified output for analysis. May contain residual parentheses.

### Variant H — Continuous Feedback
**Inside→out with per-tick global simplify**

Tests continuous global feedback: after each innermost replacement, a global simplify is applied. Maximum feedback intensity.

### Variant I — Inverse of E (10 first)
**Phase 1: all `10→0` until stable → Phase 2: all `01→0` until stable**

The mirror of E. Tests pass-order symmetry. Produces identical statistical signatures (LZ → 1/φ) but 7× more inter-scale information flow than B.

---

## Worked Example: Variant B Collapse Sequence

### Ontological reading of the binary sequence

Each bit has a precise ontological meaning. The following diagram maps the first clean bits of Φ₀ (from the structural accumulation `"0(0)1010"`) to the generative process they encode:

```
Clean bits:  0      0      1      0      1      0
Ontology:    Ø₀    (Ø₀)   |Ø₀|    Ø₀    |Ø₀|    Ø₀
             seed  ╰─decay frame─╯  ╰─collapse─╯  return
                        ⟂              ⟂
```

Where:
- **`0` = Ø₀** — Pure Nothingness (No-Res): the absence of all properties
- **`1` = |Ø₀|** — Absolute Nothingness: Nothingness that, by becoming origin, acquires absoluteness
- **`⟂`** — Irreducible ontological tension: the contradiction between being nothing and being absolute

Every `01` or `10` adjacency is a ⟂-tension that annihilates to `0` (No-Res). Adjacent `00` pairs carry no tension — Nothingness is consistent with itself. Nothing disappears — it degrades. The final `0` records the return to No-Res after collapse — the fundamental consequence of every annihilation.

### Step-by-step trace (ABS = 1)

**Seed:** `R₀ = 0` (Nothingness as origin)

**Simplification rules:** `01 → 0`, `10 → 0` (⟂-annihilation), `0⁺ → 0`, `1⁺ → 1` (run compression). Iterate until stable.

---

#### Iteration α = 0

**1. Accumulate seed:** Acc ← `"0"`

**2. Decay frame:** `(Acc)ABS = (0)1`

Ontological reading: Nothingness `(Ø₀)` confronts the Absolute `|Ø₀|`

**3. Inside→out collapse (while loop — each pass appends state to Acc):**

```
Pass 1: append "(0)1" to Acc     → Acc = "0(0)1"
        simplify innermost (0) → "0"
        result: "01"             ← ⟂-tension exposed

Pass 2: append "01" to Acc       → Acc = "0(0)101"
        simplify "01" → "01"     ← no parentheses, stable
        state == previous → EXIT LOOP
```

**4. Final global simplify:** `01 → 0` (⟂-annihilation → return to No-Res)

**5. Record collapsed state:** The `else` fallback appends `"0"` → Acc = `"0(0)1010"`

**Structural accumulation:** `"0(0)1010"` (8 chars) → **clean bits: `001010`** (6 bits)

---

#### Iteration α = 1

**1. Accumulate:** Acc += `"0"` (current_state) → Acc = `"0(0)10100"`

> Note: the snapshot saved at end of α=0 captures `"0(0)1010"` (before this append).

**2. Decay frame:** `(0(0)10100)1`

The entire history is nested inside the Absolute — each iteration contains all predecessors.

**3. Inside→out collapse:**

```
Pass 1: append "(0(0)10100)1"    → grows Acc
        innermost (0) → "0"
        result: "(0010100)1"

Pass 2: append "(0010100)1"      → grows Acc
        (0010100): simplify "0010100" → "0"
        result: "01"

Pass 3: append "01"              → grows Acc
        "01" → "01" (stable)
        state == previous → EXIT LOOP
```

**4. Final global:** `01 → 0`

**5. Record collapsed state:** append `"0"` → Acc grows further

**Structural accumulation at snapshot:** 30 chars → **clean bits: 22 bits**

---

### Key observations

1. **The result is always `0`** — guaranteed by the Uniqueness Lemma (the sequence always starts with `0`, and `01 → 0`)
2. **The information lives in the accumulation**, not the final value — Φ captures every intermediate micro-state, including the final collapsed `0`
3. **Each iteration nests deeper** — the decay frame `(Accα)1` encapsulates the entire prior history, creating progressively richer structure
4. **The final `0`** — records the return to No-Res after collapse; this is the fundamental ontological consequence: every tension resolves back to Nothingness
5. **Exponential growth** — clean bits grow as ~2ⁿ per iteration (6 → 22 → 82 → ...)

---

## Control Variants

| Variant | Source | Expected LZ | Purpose |
|---------|--------|-------------|---------|
| **A** | Mersenne Twister PRNG | ≈ 1.0 | Null hypothesis — pure randomness |
| **J** | Binary expansion of π | ≈ 1.0 | Sanity check — transcendental constant |
| **K** | Wolfram Rule 30 | ≈ 1.0 | Deterministic pseudorandomness |
| **L** | Logistic map (r=3.99) | ≈ 0.88 | Deterministic chaos |
| **M** | Fibonacci word | ≈ 0.55 | **Positive control** — has φ by construction |

---

## Two Families

The experimental results reveal two distinct families determined by a single parameter — **global feedback**:

| Property | 1/φ Family (B, D, E, G, I) | 1/√3 Family (F, H) |
|----------|---------------------------|---------------------|
| LZ ratio | ≈ 0.615 → 1/φ | ≈ 0.57 → 1/√3 |
| TE ratio | 3.78–27.46× (significant) | 0.92× (no signal) |
| mean_children | φ+1 (exact) | φ+1 (F) / degenerate (H) |
| Mechanism | Local collapse, hierarchy preserved | Global feedback destroys inter-scale structure |

---

## Invariants

All experimental variants satisfy:
- ✅ **Ordered cascade:** every `1` encapsulates all prior states
- ✅ **Determinism:** same input → same output
- ✅ **Monotone accumulation:** information only grows
- ✅ **Micro-state reality:** all intermediate states are accumulated
- ✅ **Stratified order:** collapse proceeds inside→out

---

*For implementation details, see the [source code](https://github.com/iban-borras/informational-singularity-hypothesis).*
*For complete numerical results, see the Experimental Results Summary.*

