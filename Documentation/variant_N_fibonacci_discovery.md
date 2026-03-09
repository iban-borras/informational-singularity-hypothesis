# Variant N — Ontological Collapse and the Fibonacci Skeleton

**Authors:** Iban Borràs Serret & Sophia (Augment Agent)
**Date:** 2026-03-08 (proposal) / 2026-03-09 (Fibonacci discovery)
**Status:** Verified result — ready for OSF inclusion

---

## 1. Philosophical Motivation

In Variant B, the seed Ø₀ is **preserved** inside the decay frame: `(0)1`.
The original Nothingness remains as a record while the Absolute is generated alongside it.

Variant N proposes a fundamentally different ontology: the Absolute **wraps** Nothingness.
The frame is `1(Acc)` instead of `(Acc)1`. This is not merely a syntactic inversion —
it captures the idea that Nothingness does not *generate* the Absolute; rather,
the Absolute *envelops* all that exists.

### Key ontological distinction

- **B (preservative):** Ø₀ *generates* |Ø₀| → frame `(Ø₀)|Ø₀|` → bits: `(0)1`
  - Reading: "Nothingness produces Absolute alongside itself"
- **N (transformative):** |Ø₀| *wraps* Ø₀ → frame `|Ø₀|(Ø₀)` → bits: `1(0)`
  - Reading: "The Absolute envelops all prior Nothingness"

The `1(...)` structure is an **atomic ontological unit**: the leading `1` and the
parenthesized content form an indivisible entity. When the inner tensions resolve,
the **entire unit** collapses — including the `1` itself. This is the "return to
Ø₀": once all internal contradictions are resolved, nothing remains.

### Symbolic trace (first cycle)

```
0         1(0)      0
Ø₀   →  |Ø₀|(Ø₀)  →  Ø₀
seed    atomic frame   collapse (return to Ø₀)
           ⟂              ⟂
```

## 2. The Atomic 1(...) Frame

The core innovation of Variant N is the **atomic frame** `1(content)`:

1. The `1` (Absolute) and the parenthesized `(content)` are **one indivisible unit**
2. First, simplify `content` using standard `_simplify_base` rules
3. Then, the entire unit resolves: `1 + simplified_content` → `_simplify_base("1" + result)`
4. This means `1(0)` → simplify inner `0` → resolve `"10"` → `"0"` (⟂-annihilation)

This differs from Variant B where `(0)1` collapses the `(0)` first (→ `0`) and then
the outer `01` globally. In N, the `1` is structurally bound to its content —
it cannot exist without what it wraps.

| Variant | Frame | Collapse strategy |
|---------|-------|-------------------|
| **B** (canonical) | `(Acc)1` | Stratified inside→out, each nesting level resolves independently |
| **N** (ontological) | `1(Acc)` | Atomic — `1(content)` collapses as an indivisible unit |

### Nested atomic frames

When the accumulation contains nested `1(...)` patterns (from previous iterations),
the collapse proceeds **inside→out** just like B, but each `1(...)` unit is atomic:

```
1(0 1(0) 0)
       ↑ innermost: 1(0) → "10" → "0"
→ 1(0 0 0) → 1(0) → "10" → "0"
```

The run compression rule (three consecutive Ø₀ = one Ø₀) is applied between
atomic collapses: `1(000)` → `1(0)` → `0`.

## 3. Exact Algorithm

### Per-iteration procedure

```
Input:  Acc (accumulated string, initialized to "0")
Output: updated Acc after iteration

1. Build frame:    state = "1(" + Acc + ")"
2. Collapse loop:  while state changes:
   a. Append state to Acc    (record intermediate micro-state)
   b. Collapse all innermost atomic frames:
      regex: 1\(([01]+)\) → _simplify_base("1" + _simplify_base(inner))
      then compress runs in the result
   c. If result is single char → exit loop
3. If final state ≠ last appended → append final state to Acc
4. Result: Acc now contains the full ontological trace
```

### Simplification rules (same as all variants)

- `01 → 0` (⟂-annihilation: Nothingness meets Absolute)
- `10 → 0` (⟂-annihilation: Absolute meets Nothingness)
- `0+ → 0` (run compression: multiple Nothingness = one Nothingness)
- `1+ → 1` (run compression: multiple Absolute = one Absolute)
- Iterate until stable. Result is never empty (minimum: `"0"`).

### Collapse function

```python
def _collapse_n_atomic(state: str) -> str:
    """Collapse innermost ATOMIC 1(...) frames."""
    def _collapse_match(match):
        inner = match.group(1)
        inner_simplified = _simplify_base(inner)
        return _simplify_base("1" + inner_simplified)
    result = re.sub(r'1\(([01]+)\)', _collapse_match, state)
    result = _compress_runs(result)
    return result
```

The regex `1\(([01]+)\)` only matches `1(` followed by **pure digits** (no nested
parens) followed by `)`. This ensures inside→out ordering: innermost frames are
collapsed first, exactly like Variant B.

## 4. Verified Traces

### α = 0 → 4 clean bits ✓

```
Acc = "0"
Frame: "1(0)"

Pass 1: append "1(0)" → Acc = "01(0)"
  1(0): inner "0" → resolve "10" → "0"
  → collapsed to "0" (single char, exit)

Append collapsed "0" → Acc = "01(0)0"
Clean bits: "0100" (4 bits)
```

### α = 1 → 12 clean bits ✓

```
Acc = "01(0)0"
Frame: "1(01(0)0)"

Pass 1: append "1(01(0)0)" → Acc = "01(0)01(01(0)0)"
  inner 1(0) → "0", so frame becomes "1(00)" → compress → "1(0)"
  → "1(0)" (still multi-char)

Pass 2: append "1(0)" → Acc = "01(0)01(01(0)0)1(0)"
  1(0) → "0"
  → collapsed to "0" (exit)

Append collapsed "0" → Acc = "01(0)01(01(0)0)1(0)0"
Clean bits: "010010100100" (12 bits)
```

### α = 2 → 33 clean bits

```
Acc = "01(0)01(01(0)0)1(0)0"
Frame: "1(01(0)01(01(0)0)1(0)0)"

Pass 1: inner 1(0) frames collapse → "1(01(0)0)" (via run compression)
Pass 2: inner 1(0) → "1(0)"
Pass 3: 1(0) → "0"

Clean bits: "010010100100101001010010010100100" (33 bits)
```

## 5. Growth Analysis — Convergence to φ²

Running the algorithm for 12 iterations reveals a striking pattern:

```
α= 0:          1 →          4 bits  (×4.0000)
α= 1:          4 →         12 bits  (×3.0000)
α= 2:         12 →         33 bits  (×2.7500)
α= 3:         33 →         88 bits  (×2.6667)
α= 4:         88 →        232 bits  (×2.6364)
α= 5:        232 →        609 bits  (×2.6250)
α= 6:        609 →       1596 bits  (×2.6207)
α= 7:       1596 →       4180 bits  (×2.6190)
α= 8:       4180 →      10945 bits  (×2.6184)
α= 9:      10945 →      28656 bits  (×2.6182)
α=10:      28656 →      75024 bits  (×2.6181)
α=11:      75024 →     196417 bits  (×2.6181)
```

**The growth ratio converges to φ² = φ + 1 ≈ 2.6180** (where φ = (1+√5)/2).

This is structurally analogous to Fibonacci recurrence: `F(n) = F(n-1) + F(n-2)`,
where F(n)/F(n-1) → φ. The growth ratio approaches φ² because the two branches —
accumulated history (Acc) and collapse intermediates — feed back into each other,
and the frame `1(Acc)` effectively doubles the self-reference depth.

---

## 6. The Fibonacci Discovery

### 6.1 Initial observation

After running N for 17 iterations (24M bits), the 2D raster plot was visually **indistinguishable** from Variant M (Fibonacci Word — our positive control for φ detection). This was unexpected: N was designed as an alternative HSI mechanism, not as a known mathematical sequence.

### 6.2 Verification

We generated both sequences and compared them bit by bit:

```
N clean bits:     010010100100101001010010010100100...
Fibonacci word:   101101011011010110101101101011011...
```

Every bit is the exact inverse. We verified formally:

```python
n_complement = ''.join('1' if b == '0' else '0' for b in n_bits)
match = (n_complement == fibonacci_word)  # True — 4180/4180 bits
```

**Result: Variant N produces the exact bitwise complement of the Fibonacci word.**

This was verified for 4,180 clean bits (8 iterations) with 100% match.

### 6.3 Statistical confirmation

| Metric | Variant N | Fibonacci word | Relationship |
|--------|-----------|----------------|--------------|
| Ratio of 1s | 0.3818 | 0.6182 | N ≈ 1/φ², Fib ≈ 1/φ |
| Ratio of 0s | 0.6182 | 0.3818 | Complementary |
| Compression ratio | 16.37% | ~16% | Identical structure |
| Fractal dimension | 0.978 | ~0.98 | Identical |

The Fibonacci word is generated by the substitution σ: 0→01, 1→0. Its complement follows σ': 0→10, 1→0. The atomic collapse rules of N, combined with accumulation, implement precisely this substitution.

## 7. What This Tells Us About Variant B

### 7.1 Variant N as a minimal proof of concept

Variant N is the **simplest possible implementation** of the HSI generative idea. It takes the philosophical starting point — Nothingness self-observing — and implements it in the most naïve way: the Absolute wraps the accumulated history as a single indivisible unit, and the whole thing collapses at once.

This simplicity is deliberate. N answers the question: *what is the minimum mechanism that can emerge from the HSI axioms?* The answer turns out to be the Fibonacci word — a beautiful, perfectly ordered crystal, but one that contains no genuine complexity. Everything in N is predetermined by the initial act of placing `1` before the content.

N is valuable precisely because it is too simple. It reveals what is **missing** when you strip the mechanism to its bare minimum, and thereby illuminates why each element of Variant B is necessary.

### 7.2 Why each element of Variant B is necessary

The contrast between N and B can be decomposed into three structural differences, each with a clear ontological justification:

**1. Position of the Absolute: residue, not agent**

| | N: `1(Acc)` | B: `(Acc)1` |
|---|---|---|
| Ontological role | The Absolute **pre-exists** and acts on Nothingness | The Absolute **emerges** as a residue of self-contradiction |
| Problem | Violates the premise: only Nothingness exists. Where does the `1` come from? | Consistent: `1` appears at the end because it is what remains after self-annihilation |
| Result | Deterministic crystal (Fibonacci) | Fractal complexity |

In N, writing `1(Acc)` presupposes the Absolute — it is already there before anything happens. But the foundational premise states that **only Nothingness exists**. The `1` cannot be an external agent; it must arise from the process itself. In B, the `1` appears *after* the accumulated contradiction, as its consequence. The Absolute is not a cause — it is what is left when Nothingness fails to be Nothing.

**2. Collapse strategy: stratified vs. atomic**

| | N: atomic | B: stratified inside→out |
|---|---|---|
| Mechanism | `1(content)` resolves as one indivisible unit | Each nesting layer resolves independently before feeding the next |
| Analogy | A building demolished by a single explosion | A building deconstructed floor by floor, each floor's materials reused |
| Information generated | Minimal — one substitution step per frame | Maximal — each layer creates new boundary conditions for the next |
| Result | Quasi-periodic (one frequency) | Fractal (hierarchy of frequencies) |

The atomic collapse of N is equivalent to saying that self-observation happens instantaneously and completely. But genuine self-reference cannot be instantaneous — the act of observing creates a new state, which must itself be observed, creating a cascade. The stratified collapse of B captures this: the innermost contradiction resolves first, its resolution creates a new context, and the next layer resolves within that new context. Each resolution is ontologically real and contributes to the accumulated history.

**3. Accumulation of intermediate states**

In N, the collapse produces a single result per frame — the intermediate states within the atomic unit are not preserved. In B, every intermediate state during the inside→out collapse is accumulated. These micro-states are not computational artefacts; they represent the ontological reality that **every moment of the self-observation process exists**.

This is the difference between a photograph (N: one final state) and a film (B: the entire process recorded). The fractal structure of B emerges precisely because the process itself — not just its endpoint — is real.

### 7.3 The skeleton analogy

The Fibonacci word is the **simplest possible quasi-periodic sequence** — a crystal governed by φ. When we strip the HSI mechanism of its hierarchical collapse (replacing stratified inside→out with atomic resolution), the skeleton that remains is this crystal.

**Fibonacci is the skeleton. Variant B adds the flesh.**

The stratified collapse of B creates a hierarchy of resolutions on top of this skeleton. Each layer of nesting adds a new scale of structure. The result is not a crystal but a fractal — the difference between a single note and music.

The analogy is precise: Nothingness vibrating alone can only produce one note — the Fibonacci note. When it folds upon itself in layers (Variant B), harmonics appear. And harmonics upon harmonics create the music.

## 8. Relationship with Control Variant M

Variant M (Fibonacci Word) was designed as a **positive control** for φ detection: a sequence with φ encoded by construction, to validate that our analysis pipeline correctly detects φ where it exists.

The discovery that N = NOT(M) means:

1. **N is not an independent experimental variant** — it is mathematically equivalent to the positive control (up to bit inversion, which preserves all structural properties).
2. **This validates the analysis pipeline**: N and M show identical compression ratios, fractal dimensions, and spectral properties, as expected for complementary sequences.
3. **This was not designed** — it emerged from the mechanism itself, providing independent confirmation that φ is intrinsic to the simplification rules.

## 9. Summary

| Question | Answer |
|----------|--------|
| What does N generate? | The bitwise complement of the Fibonacci word |
| Is this a bug? | No — verified mathematically, 100% match |
| Why does it happen? | Atomic collapse implements the Fibonacci substitution rule |
| What does it tell us about B? | Fibonacci is the skeleton; stratified collapse adds fractal complexity |
| Why can't the Absolute be an agent? | It produces a crystal, not emergence — all information is pre-determined |
| Is N useful? | Yes — as an independent derivation of the Fibonacci word from HSI axioms |

---

*"Nothingness cannot be Nothingness without 'knowing' that it is Nothingness — and that 'knowing' is the first fissure."*

— From the discussion that led to this discovery (Iban & Sophia, March 2026)

---

**Files:**
- Implementation: `level0/generator.py` (function `_collapse_n_atomic`)
- Trace script: `_trace_variant_n.py`
- Generated data: `results/level0/phi_snapshots/var_N/`
- Comparison control: `results/level0/phi_snapshots/var_M/` (Fibonacci word)

