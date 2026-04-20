# HSI v2 Phase 1 Live Findings

## Status

**Document type:** living findings log  
**Language:** English  
**Scope:** HSI v2 Phase 1 gate (`P_m`, projections, fibers, moments) plus derived coherence and transport readouts  
**Last updated:** 2026-04-17  
**Current status:** first Phase 1 gate passed, comparative reports consolidated, strong-null batch integrated, first protocol-robustness sweep completed, and adjacent-iteration divergence scans extended to `B/E/I/F`

This document records the live scientific reading of the first executable HSI v2 Phase 1 runs. It is not a paper draft and not a decision log. Its role is to preserve what the code is currently telling us, what that strengthens, what it does not strengthen, and what must happen next.

The guiding principle remains strict:

> First demonstrate a non-trivial projective hierarchy.  
> Only then move toward richer inter-scale formalization.

---

## Scope of the Current Batch

The current Phase 1 implementation operates on:

- locally stable pattern spaces `P_m`
- adjacent-scale projections
- fibers under explicit projection policies
- moments `S_q(m)` for `q = 0, 1, 2, 3`
- comparative summaries from stored Phase 1 artifacts
- a derived tail-coherence readout over the last edges of the tower
- a derived transport-persistence readout over consecutive edges of the tower

The current default protocol is:

- observable sampling: prefix, consecutive segments
- segment protocol: `3 x 1,000,000 bits`
- scale ladder: `m = 8, 12, 16, 20, 24, 28, 32`
- projection policies:
  - `prefix` as main policy
  - `suffix` as mandatory control
- `P_m` admission rule:
  - `min_count_floor = 16`
  - `min_count_rate = 1e-6`
  - `min_segment_support = 2`
  - `cv_max = 1.5`

Current runs inspected in this document:

- `B @ 20`
- `E @ 20`
- `I @ 20`
- `F @ 20`
- `A @ 20`
- `J @ 20`
- `K @ 20`
- `L @ 20`
- `M @ 20`
- `N @ 17`
- `B-shuffled @ 20`
- `B-same-density @ 20`
- `B-markov1 @ 20`
- `B @ 20` under `2 x 1,000,000 bits`
- `B-markov1 @ 20` under `2 x 1,000,000 bits`
- `B @ 20` under `4 x 750,000 bits`
- `B-markov1 @ 20` under `4 x 750,000 bits`
- adjacent-iteration divergence scans for:
  - `B @ 20` versus `B @ 21`
  - `E @ 20` versus `E @ 21`
  - `E @ 21` versus `E @ 22`
  - `E @ 22` versus `E @ 23`
  - `I @ 20` versus `I @ 21`
  - `I @ 21` versus `I @ 22`
  - `I @ 22` versus `I @ 23`
  - `F @ 20` versus `F @ 21`
  - `F @ 21` versus `F @ 22`
  - `F @ 22` versus `F @ 23`

The most important Phase 1 contrast set is:

- `B` as canonical HSI reference
- `F` as feedback-affected contrast
- `A` as baseline falsation control
- `M` and `N` as structured non-HSI contrast controls

The current reporting layer now includes:

- `phase1_report`, which compares tower width, retention, survival, entropy, and moment ratios
- `phase1_coherence_report`, which compares the last part of the tower through tail survival, tail entropy, `q`-spread, and `prefix/suffix` gaps
- `phase1_transport_report`, which compares how much branching mass is transported into children that still branch one edge later
- both tail-derived reports now expose `mean`, `min`, and `last` summaries for the relevant tail observables, so that abrupt top-of-tower collapse is not hidden by a simple tail average

---

## Main Findings

### 1. `B` and `E` sustain a wide, coherent projective tower

`B` and `E` remain indistinguishable in the current Phase 1 observable stack.

Observed behavior:

- large and stable growth of `P_m` across the scale ladder
- near-perfect fiber survival across almost the entire tower
- non-trivial fiber widths at all scales
- non-trivial child entropy even at the top of the ladder
- a real `prefix/suffix` difference, especially in entropy and singleton fraction

Interpretation:

- this strongly reinforces the idea that `B` and `E` belong to the same structural family
- the statistical commutativity intuition from v1 survives into the first real v2 tower
- the hierarchy is not merely surviving; it remains comparatively rich deep into the ladder

### 2. `F` does not collapse, but it becomes much more rigid than `B`

`F` is not weak in this Phase 1 observable. It clearly sustains a real tower.

Observed behavior:

- full fiber survival at high scales
- much narrower `P_m` than `B`
- much lower child entropy than `B`
- moment ratios much closer to `1`
- much smaller `prefix/suffix` gap than `B`

Interpretation:

- this is consistent with the hypothesis that `F` preserves local/projective structure while losing richer inter-scale organization
- `F` is not equivalent to a random or degenerate control
- but it is also not equivalent to `B`
- the tower of `F` looks more rigid, thinner, and less hierarchically expressive

This is one of the most important current findings.

### 3. `A` falsifies cleanly

`A` behaves exactly as a baseline falsation control should behave.

Observed behavior:

- massive apparent richness at low scales
- catastrophic collapse by `m = 20`
- zero retained structure at higher scales
- zero meaningful fiber survival at the top of the ladder

Interpretation:

- the current `P_m` rule is not hallucinating projective hierarchy in a random baseline
- the method is therefore behaving in a scientifically healthy way

### 4. `M` and `N` form a compact crystalline class

`M` and `N` are, in the current observable, essentially indistinguishable.

Observed behavior:

- very small but fully persistent `P_m`
- perfect survival
- almost no meaningful `prefix/suffix` asymmetry
- very low child entropy
- moment ratios close to a smooth crystalline regime

Interpretation:

- these controls are extremely valuable
- they show that a fully persistent tower is not enough by itself to support HSI
- a compact crystal-like tower can exist without supporting the richer HSI reading
- this prevents us from confusing "persistent hierarchy" with "HSI-specific hierarchy"

### 5. `I` supports robustness, but in a different regime from `B`

`I` clearly survives the gate, but it is not simply a weaker `B`.

Observed behavior:

- narrower tower than `B`
- stronger high-order moment ratios than `B` at the upper part of the ladder
- higher child entropy than `F`, lower overall breadth than `B`

Interpretation:

- `I` appears to belong to the same broad family as `B/E`, but in a shifted internal regime
- this is a useful robustness result, because it suggests the gate is not only passing one single sequence archetype

### 6. Tail coherence now separates coherent, rigid, crystalline, collapsed, and trivial towers

The derived tail-coherence readout does not replace the Phase 1 tower itself. It sharpens the interpretation of the last part of the tower using only already generated Phase 1 artifacts.

Observed class split:

- `B/E/I` classify as `coherent`
- `F` classifies as `rigid`
- `M/N` classify as `crystalline`
- `A/J/L` classify as `collapsed`
- `K` classifies as `trivial`

Interpretation:

- the current program no longer distinguishes only between persistence and collapse
- it now distinguishes between different kinds of persistence
- this is especially important because it separates the broad HSI-like towers from compact crystal-like towers without invoking any premature geometry
- it also reinforces the working idea that `F` keeps a tower, but in a more rigid regime than `B/E/I`

### 7. Transport persistence reinforces the same class split from a more inter-scale angle

The derived transport-persistence readout asks a stricter question than plain local branching:

- when branching mass appears on one edge of the tower,
- how much of that mass reaches children that still branch on the next edge?

Observed class split:

- `B/E/I` classify as `coherent`
- `F` classifies as `rigid`
- `M/N` classify as `crystalline`
- `A/J/L` classify as `collapsed`
- `K` classifies as `trivial`

Interpretation:

- this strengthens the distinction between local branching richness and the transport of branching structure across edges
- `B/E/I` do not only branch locally; they transport branching mass in a non-trivial way across the last part of the ladder
- `F` still transports structure, but in a much more rigid regime than `B/E/I`
- `M/N` again show why persistence by itself is not enough: they preserve a compact crystal-like regime without reading as HSI-like coherence
- the fact that coherence and transport now point to the same class split is one of the strongest current arguments for continuing the v2 program

### 8. The full strong-null Phase 1 batch sharpens the claim

The current full Phase 1 null batch now includes:

- `B-shuffled`
- `B-same-density`
- `B-markov1`

Observed behavior:

- `B-shuffled` and `B-same-density` show large apparent low-scale richness, but collapse completely at the top of the tower
- both of these nulls reach:
  - `kept@last = 0`
  - tail coherence = `0`
  - transport persistence = `0`
- `B-markov1` does **not** collapse completely
- instead, `B-markov1` generates:
  - a much broader tower than `B`
  - much lower final retention
  - much weaker top-of-tower survival
  - much weaker top-of-tower transport quality
  - almost no signed `prefix/suffix` asymmetry

Interpretation:

- the discriminant is no longer correctly phrased as "tower versus no tower"
- the discriminant is now better phrased as:
  - compact, selective, directionally structured projective persistence
  - versus broad, diffuse, weakly selective persistence
- randomization and density-preserving nulls fail the gate entirely
- `Markov-1` survives only in a much more diffuse regime that does not read as HSI-like

This is one of the most important refinements of the current program.

### 9. The first protocol-robustness sweep preserves the `B` vs `B-markov1` split

The first robustness sweep has now compared:

- `3 x 1,000,000 bits`
- `2 x 1,000,000 bits`
- `4 x 750,000 bits`

Observed behavior:

- `B` remains stable across all three segment layouts
- in particular, the top-of-tower behavior of `B` remains in the same regime:
  - high retention
  - near-perfect tail survival
  - non-trivial child entropy
  - non-zero directional asymmetry
- `B-markov1` also remains stable across layouts, but in its own regime:
  - very large breadth ratio
  - low final retention
  - degraded tail survival
  - degraded transport persistence
  - near-zero signed asymmetry

Interpretation:

- the current `B` versus `B-markov1` separation is not a fragile artifact of one particular segment protocol
- the observed contrast appears structurally stable under modest changes in segmentation
- this significantly strengthens the claim that the Phase 1 gate is detecting a real difference in tower regime rather than a sampling accident

### 10. Adjacent-iteration divergence now splits `B`, `E/I`, and `F` into distinct observable-stability regimes

The dedicated first-divergence scans now show that adjacent-iteration robustness is not uniform across the structured variants.

Observed behavior:

- `B @ 20` versus `B @ 21` shows:
  - no observable-bit mismatch through `303,000,000` bits
  - no observable-bit mismatch through `1,000,000,000` bits
  - no observable-bit mismatch through `3,000,000,000` bits
  - and no observable-bit mismatch before the current end of `B @ 20`
- the first measured `B` difference is therefore a `length_mismatch` at `11,198,388,932` observable bits
- `E @ 20` versus `E @ 21` shows its first true `bit_mismatch` at `1,395,248,513`
- `I @ 20` versus `I @ 21` shows its first true `bit_mismatch` at `1,526,912,785`
- `F @ 20` versus `F @ 21` shows a much earlier `length_mismatch` at `127,016,258`

Interpretation:

- `B` currently reads as observable prefix extension rather than observable rewriting
- `E` and `I` also show very long frozen prefixes, but not indefinitely: they eventually enter a true adjacent-iteration rewriting regime within the first `3G` bits
- `F` remains stable only over a much shorter adjacent-iteration prefix, which is consistent with its already more rigid and less expressive Phase 1 tower
- adjacent-iteration robustness is therefore becoming a meaningful discriminator inside the structured family, not only a yes/no sanity check

### 11. The newly added tail of iteration 21 remains in-family for `E`, `I`, and `F`

The first post-divergence Phase 1 attempt revealed an important boundary condition:

- for `E @ 20` versus `E @ 21`, the first measured `bit_mismatch` occurs at the last observable bit currently available in `E @ 20`
- for `I @ 20` versus `I @ 21`, the first measured `bit_mismatch` also occurs at the last observable bit currently available in `I @ 20`
- for `F @ 20` versus `F @ 21`, the first measured difference is already a `length_mismatch`

This means that a shared "post-divergence window" does not currently exist for the shorter iteration in any of these three cases.

Observed behavior on the newly added `iteration 21` tail:

- `E @ 21`, sampled from its first measured divergence onward, still reproduces the canonical `E/B` regime:
  - `k@last = 1208`
  - `ret_last = 0.8622`
  - tail coherence and transport remain in the coherent range
- `I @ 21`, sampled from its first measured divergence onward, still reproduces the canonical `I` regime:
  - `k@last = 565`
  - `ret_last = 0.9930`
  - tail coherence and transport remain in the coherent range
- `F @ 21`, sampled from its first measured divergence onward, still reproduces the canonical `F` regime:
  - `k@last = 227`
  - `ret_last = 1.0000`
  - tail coherence and transport remain in the rigid range

Interpretation:

- the newly added tail does not currently show an obvious regime break for `E`, `I`, or `F`
- what differs between these variants is not that the extension tail suddenly changes class, but how long exact adjacent-iteration freezing persists before extension or rewriting appears
- the next meaningful iteration-robustness question is therefore no longer "does the post-divergence tail collapse?"
- it is:
  - where does a shared window beyond the shorter iteration first become available?
  - and does the in-family regime still persist there?

### 12. The `21 -> 22` scans reinforce the same terminal-boundary pattern for `E/I/F`

The next adjacent-iteration layer now shows:

- `E @ 21` versus `E @ 22`:
  - first measured `bit_mismatch` at `2,790,497,025`
- `I @ 21` versus `I @ 22`:
  - no `bit_mismatch` through `3,000,000,000`
  - first measured `bit_mismatch` in the uncapped scan at `3,053,825,569`
- `F @ 21` versus `F @ 22`:
  - first measured `length_mismatch` at `254,032,518`

Observed practical consequence:

- for `E @ 21`, the first measured `bit_mismatch` again occurs at the final observable bit currently available in the shorter iteration
- for `I @ 21`, the first measured `bit_mismatch` also turns out to sit at the final observable bit currently available in the shorter iteration
- for `F @ 21`, the first measured difference again appears as a terminal `length_mismatch`

Interpretation:

- the current adjacent-iteration process is not yet yielding a shared post-divergence window for `E`, `I`, or `F`
- instead, the repeated pattern is:
  - a long exact frozen prefix
  - followed by terminal extension or terminal rewriting at the boundary of the shorter iteration
- this makes the current family reading sharper:
  - `B`: deep prefix extension over the full currently compared extent
  - `E/I`: long frozen prefixes ending in terminal bit rewriting
  - `F`: shorter frozen prefixes ending in terminal length extension

### 13. The `22 -> 23` layer confirms that `E/I/F` still do not yield a shared post-divergence window

The next adjacent-iteration layer now shows:

- `E @ 22` versus `E @ 23`:
  - no `bit_mismatch` through `3,000,000,000`
  - first `bit_mismatch` in the uncapped scan at `5,580,994,049`
- `I @ 22` versus `I @ 23`:
  - no `bit_mismatch` through `3,000,000,000`
  - first `bit_mismatch` in the uncapped scan at `6,107,651,137`
- `F @ 22` versus `F @ 23`:
  - first `bit_mismatch` already within `3G`, at `508,065,037`

Observed practical consequence:

- for `E @ 22`, the first measured `bit_mismatch` again sits at the final observable boundary of the shorter iteration
- for `I @ 22`, the same thing happens again
- for `F @ 22`, the first difference is no longer a `length_mismatch`, but the first `bit_mismatch` still sits at the final observable boundary of the shorter iteration

Interpretation:

- `E`, `I`, and `F` still do not provide a shared post-divergence window in the currently tested adjacent pairs
- the repeated family pattern is now best phrased as:
  - `B`: deep observable-prefix extension
  - `E/I`: terminal bit rewriting after a very long frozen prefix
  - `F`: short frozen prefixes, first ending in terminal length extension and then in terminal bit rewriting

### 14. Current data horizon: `E/I` stop at `iter23`, while `F` confirms the terminal-divergence pattern one step further

The current repository does not yet contain `iter24` snapshots for:

- `E`
- `I`

So the adjacent-iteration chain for those variants currently stops at:

- `E @ 22` versus `E @ 23`
- `I @ 22` versus `I @ 23`

By contrast, `F` does have `iter24`, and the next scan shows:

- `F @ 23` versus `F @ 24`:
  - first `bit_mismatch` at `1,016,130,073`
  - with empty right-context on the shorter iteration, again indicating a terminal-boundary divergence

Interpretation:

- the `E/I` story is currently blocked by available data horizon, not by methodological indecision
- `F` strengthens the current family picture by extending the same terminal-divergence logic one step further

### 15. The first lightweight `matched-LZ` null survives at the tail, but fails to reproduce HSI-style selectivity

The first operational `matched-LZ` null is now implemented as a lightweight block-shuffle surrogate:

- exact bit density is preserved
- observed blocks are shuffled at a tuned block size
- the block size is selected to approximate normalized `LZ76` over three calibration windows

For `B @ 20`, the selected calibration parameters were:

- `selected_block_bits = 4096`
- `target_lz_normalized = 0.0316`
- `achieved_lz_normalized = 0.0424`

The resulting Phase 1 tower is striking:

- `kept@first = 61`
- `kept@mid = 412`
- `kept@last = 1214`
- `retention@last = 0.0849`
- `breadth ratio = 19.9016`

Tail persistence remains high:

- coherence:
  - `surv mean = 0.9792`
  - `surv last = 0.9902`
- transport:
  - `active mean = 0.3669`
  - `active last = 0.3287`

Interpretation:

- unlike `shuffled` and `same-density`, the lightweight `matched-LZ` null does **not** collapse
- unlike `Markov-1`, it does **not** mainly express itself as a massively broad diffuse tower
- but it still fails to reproduce the key HSI feature:
  - the tower is **not** strongly selective relative to its candidate space
  - for nearly the same `kept@last` as observed `B`, it carries a vastly larger candidate pool and therefore a much smaller `retention@last`

This sharpens the current Phase 1 reading:

- persistence alone is not enough
- transport persistence alone is not enough
- the current discriminant must include **selective filtration across scales**, not only top-of-tower survival

---

## Contrast Controls

### `J`

`J` does not sustain the tower. It behaves as a weak control in this Phase 1 observable.

### `K`

`K` is particularly informative:

- it remains trivial across scales
- it shows that low effective complexity or strong determinism alone is not enough

This matters because it tells us that rigid structure by itself is not sufficient. The gate is detecting something more specific than mere compressibility.

### `L`

`L` does not survive as a meaningful tower. Deterministic chaos is not reproducing the current projective-hierarchy signal.

---

## What Currently Strengthens HSI

- `B` sustains a wide and highly coherent tower across scales.
- `E` matches `B`, extending the statistical commutativity result into Phase 1.
- `I` confirms that the coherent class is not limited to a single HSI sequence archetype.
- `F` remains structured, but in a narrower regime, supporting the distinction between local structure and richer inter-scale organization.
- `A`, `J`, and `L` do not reproduce the same behavior.
- `K` shows that rigid determinism alone is not enough.
- `M` and `N` provide a clean crystalline contrast class, helping us avoid premature over-interpretation.
- the derived tail-coherence readout independently separates coherent HSI-like towers from rigid, crystalline, collapsed, and trivial regimes.
- the derived transport-persistence readout reaches the same class split from a more explicitly inter-scale observable.
- the full strong-null Phase 1 batch now shows:
  - complete top-of-tower collapse for `shuffled` and `same-density`
  - persistent but clearly non-HSI-like diffuse hierarchy for `Markov-1`
- the first protocol-robustness sweep preserves the `B` versus `B-markov1` separation across three segment layouts.
- the first adjacent-iteration divergence scans now show an even stronger stability result for `B`:
  - no observable-bit mismatch for `B@20` versus `B@21` through `303,000,000` bits
  - no observable-bit mismatch through `1,000,000,000` bits
  - no observable-bit mismatch through `3,000,000,000` bits
  - and, in the first uncapped scan, no observable-bit mismatch before the current end of `B@20`
- the first measured `B@20` versus `B@21` difference is therefore a `length_mismatch`, not a `bit_mismatch`
- the present best reading is that `B@20` behaves as an observable prefix of `B@21`
- `E@20` versus `E@21` also preserves an extremely long frozen prefix, with the first true bit rewrite only at `1,395,248,513`
- `I@20` versus `I@21` behaves similarly, with the first true bit rewrite at `1,526,912,785`
- `F@20` versus `F@21` diverges far earlier, with the first measured difference already at `127,016,258`
- the first newly added tail sampled in `E@21`, `I@21`, and `F@21` remains in the same Phase 1 regime as the earlier canonical window

---

## What Does Not Yet Strengthen HSI

- The mere existence of a persistent tower does not uniquely support HSI, because `M/N` also sustain a tower.
- The mere existence of a persistent tower does not uniquely support HSI, because `Markov-1` can also sustain a persistent but diffuse tower.
- The current Phase 1 observable does not yet prove inter-scale information flow in the v1 sense.
- The current gate does not yet distinguish all structured non-HSI regimes from HSI-like regimes.
- The tail-coherence readout is still derived from current Phase 1 artifacts; it is not yet a full inter-scale causal-flow observable.
- The transport-persistence readout is stronger than plain local richness, but it is still derived from Phase 1 fibers rather than from a final causal-flow construction.
- The current reports are strong enough for interpretation, but they are not yet the final word on inter-scale organization.
- Tail-derived summaries should now be read with three complementary views:
  - `mean` for the overall tail regime
  - `min` for worst-edge fragility
  - `last` for strict top-of-tower survival

---

## What Would Weaken or Falsify the Current Direction

Any of the following would be serious trouble:

- if richer runs showed `B` collapsing toward the `A/J/L` class
- if `F` became indistinguishable from `B` across fiber richness and moment structure under stronger tests
- if `M/N` turned out to be indistinguishable from `B` not only in persistence but also in hierarchical richness under improved observables
- if the current effects disappeared under modest changes in segment protocol or scale ladder
- if adjacent `B` iterations started to show early observable rewriting once first-divergence scans moved beyond the current frozen extent
- if `E/I` turned out to lose their HSI-like tower immediately after their first measured bit mismatch
- if `F` unexpectedly recovered a `B`-like deep frozen-prefix regime under stronger adjacent-iteration tests
- if newly added tails beyond the frozen prefix started to flip `E/I/F` into different tower classes than the canonical ones already measured

At the moment, none of these have happened.

---

## Working Interpretation

The current best reading is that the system contains at least two separable structural layers:

1. **Local/projective structural layer**
   What survives within scales and under adjacent-scale projections.

2. **Richer inter-scale organizational layer**
   What differentiates a wide, expressive, non-crystalline hierarchy from a rigid or crystal-like one.

Current evidence suggests:

- `B/E` express both layers strongly
- `I` also belongs to the coherent side of the second layer, though in a shifted regime
- `F` clearly preserves the first layer
- `F` appears weaker in the second, and currently reads as a rigid hierarchy rather than a coherent one
- `M/N` preserve a compact crystalline form of the first layer and do not read as HSI-like coherence
- `Markov-1` preserves persistence in a broad, low-retention, weakly asymmetric regime that is clearly distinct from `B`
- the lightweight `matched-LZ` null preserves tail persistence for `B/E/I`, but does so in a weakly selective regime with inflated candidate space and does not read as HSI-like compact filtration
- the current minimal summary of this distinction can be written as `SPI_simple = retention@last * surv_last`, which stays high only when persistence and selectivity survive together
- adjacent `B` iterations currently preserve an extremely long observable prefix exactly, which points to nested extension rather than local rewriting
- `E` and `I` preserve very long adjacent-iteration prefixes, but still enter true rewriting within the first `3G` observable bits
- `F` preserves only a much shorter adjacent-iteration prefix and currently reads as the shallowest stability regime among the structured survivors
- the first observable tail newly added in `E@21`, `I@21`, and `F@21` currently remains in-family rather than breaking into a new tower regime
- `A/J/L` fail the first layer at useful scales
- `K` represents trivial persistence without meaningful hierarchy

The current taxonomy is therefore:

- **coherent hierarchies:** `B/E/I`
- **rigid hierarchy:** `F`
- **crystalline persistence:** `M/N`
- **collapsed controls:** `A/J/L`
- **trivial deterministic control:** `K`

This does **not** yet complete the theory, but it is enough to justify continuing Phase 1 as a real scientific program rather than a speculative extension.

---

## Immediate Next Step

The next operational step should be:

1. use `phase1_report` and `phase1_coherence_report` as the stable experimental basis for Block III drafting
2. include `phase1_transport_report` as the strongest current derived readout of inter-edge persistence
3. preserve the current class split (`coherent`, `rigid`, `crystalline`, `collapsed`, `trivial`) as a provisional empirical taxonomy
4. preserve the refined strong-null reading:
   - random collapse (`shuffled`, `same-density`)
   - broad diffuse persistence (`Markov-1`)
   - persistent but weakly selective filtration (`matched-LZ`)
   - compact HSI-like persistence (`B/E/I`)
5. move to the next robustness layer:
   - first-divergence scans for `E/I/F` against the next adjacent iteration (`21` versus `22`)
   - determine whether the first divergence again sits at the terminal edge of the shorter iteration
   - only once a shared post-divergence window exists, compare whether the tower remains in-family or breaks sharply
6. keep all further interpretation constrained by the current result:
   - no premature geometry
   - no claim that persistence alone is enough
   - no claim that coherence or transport already replace causal inter-scale flow

---

## Editorial Note for Sofia

This document is intentionally written as a live scientific reading, not as publication prose.

Its purpose is to preserve:

- what the code actually shows
- what is already strong enough to say
- what must still remain provisional
- and where the paper narrative should be careful not to overclaim

It now includes both:

- the direct Phase 1 tower findings
- the derived coherence and transport taxonomies built from those artifacts

This means it is now suitable as working support for Block III drafting, provided the text remains careful about one point:

- the current result already supports a taxonomy of tower classes
- but it does **not** yet amount to a final theory of inter-scale flow

For the future paper, the key message to preserve is:

> Phase 1 has now shown that a non-trivial projective hierarchy is real in the HSI family, but that not all persistent hierarchies are equal. The current evidence already separates random collapse, rigid crystalline structure, and a richer HSI-like hierarchy.
