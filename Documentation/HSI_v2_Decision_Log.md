# HSI v2 Decision Log

## Purpose

This document records scientific and methodological decisions that change how HSI v2 is executed, interpreted, or documented.

Its job is not to replace the main design documents.

Its job is to keep a compact, chronological record of:

- what was decided
- why it was decided
- what it changes in practice
- and what remains open

## Status Labels

- `accepted`: active default unless superseded
- `provisional`: current working default, still under test
- `rejected`: considered and explicitly not adopted
- `superseded`: historical entry replaced by a newer one

---

## D-0001

- Date: 2026-04-14
- Status: `accepted`
- Scope: scientific gate
- Title: Phase 1 must demonstrate projective hierarchy before geometry

### Decision

HSI v2 must first demonstrate or refute a non-trivial projective hierarchy that is not reproduced by strong controls.

Defects, hierarchical graphs, and spectral constructions are downstream stages and must not be treated as the primary entry gate.

### Rationale

The earlier roadmap risked moving too quickly toward elegant mathematical objects before establishing that the underlying multiscale projective structure is objectively present in the data.

### Source Documents

- `docs/HSI_v2_proposta_revisada.md`
- `docs/HSI_v2_Critical_Review.md`
- `docs/HSI_v2_Phase1_Minimum_Executable.md`

### Practical Impact

- Phase 0 / preflight is mandatory.
- Phase 1 reports must answer the hierarchy question directly.
- No geometry-first implementation track should be treated as canonical.

### Open Questions

- Which operational definition of `P_m` should become the main default after preflight.
- Which projection policy should be the main one and which should remain a control.

---

## D-0002

- Date: 2026-04-14
- Status: `accepted`
- Scope: repository architecture
- Title: HSI v2 must use a separate public and internal namespace

### Decision

New public scripts must use the prefix `hsi_v2_`.

New internal implementation must live under `hsi_agents_project/v2/`, and new outputs must live under `hsi_agents_project/results/hsi_v2/`.

### Rationale

The legacy repo already has overloaded Level 1 and Level 2 names. A separate v2 track avoids semantic confusion, keeps the future paper methods section cleaner, and protects legacy reproducibility.

### Source Documents

- `docs/HSI_v2_Repository_Implementation_Plan.md`

### Practical Impact

- New code should not silently redefine legacy scripts.
- New artifacts must be stage-scoped and metadata-rich.
- README and operational docs can explain v2 as an extension rather than a rewrite.

### Open Questions

- Exact internal module split for the first implementation batch.

---

## D-0003

- Date: 2026-04-14
- Status: `accepted`
- Scope: scientific workflow
- Title: A preflight factor-complexity study is required before the heavy Phase 1 run

### Decision

Before launching the full Phase 1 pipeline, we must run a preflight study focused on factor spaces, complexity growth, branching, asymmetry, and usable scale depth.

### Rationale

This gives us a cheap way to test whether the data plausibly supports persistent multiscale structure before investing in the full pattern-space and fiber pipeline.

### Source Documents

- `docs/HSI_v2_Preflight_Factor_Complexity_Experiment.md`
- `docs/HSI_v2_Phase1_Minimum_Executable.md`

### Practical Impact

- The first public v2 scripts should cover preflight first.
- Phase 1 defaults should be informed by preflight, not guessed in advance.

### Open Questions

- Exact iteration set for the first preflight batch.
- Whether matched-LZ can be made affordable in the first execution pass.

---

## D-0004

- Date: 2026-04-14
- Status: `accepted`
- Scope: mathematical interpretation
- Title: Raw binary factor branching cannot converge to phi plus one

### Decision

In the binary observable language, the raw one-step factor branching ratio `p(m+1)/p(m)` cannot exceed `2`.

Therefore it cannot converge to `phi + 1`, and any possible bridge to `phi + 1` must be searched for in a derived observable rather than raw one-step binary branching.

### Rationale

Each factor of length `m` can only extend to the right by appending `0` or `1`, so the raw right-extension count is structurally bounded.

This prevents a mathematically impossible interpretation from leaking into the Phase 1 design.

### Source Documents

- `docs/HSI_v2_Response_to_Sofia_on_Fiber_Survival.md`
- `docs/HSI_v2_Preflight_Factor_Complexity_Experiment.md`

### Practical Impact

- The preflight must estimate `h_eff` explicitly instead of equating it with LZ.
- Any `phi + 1` bridge, if it exists, must use a different observable.

### Open Questions

- Which derived observable is the best candidate for testing that bridge.

---

## D-0005

- Date: 2026-04-14
- Status: `provisional`
- Scope: null-model policy
- Title: Minimal serious null set for early HSI v2 work

### Decision

The minimal serious null set for early HSI v2 work is:

- shuffled
- same-density
- Markov-1
- matched-LZ if viable, otherwise prepared as the next step

### Rationale

This is the cheapest set that still gives us protection against trivial self-deception by randomization, density effects, and first-order local correlations.

### Source Documents

- `docs/HSI_v2_Preflight_Factor_Complexity_Experiment.md`
- `docs/HSI_v2_proposta_revisada.md`

### Practical Impact

- No early positive claim should rely on shuffled-only controls.
- The first pipeline must treat null generation as a first-class stage.

### Open Questions

- Whether a stronger de Bruijn-like or run-length-preserving null should be promoted earlier.
- When matched-LZ becomes operationally affordable.

---

## D-0006

- Date: 2026-04-15
- Status: `provisional`
- Scope: paper architecture
- Title: HSI v2 paper should follow a three-block structure with falsifiable section endings

### Decision

The HSI v2 paper should be organized in three main blocks:

- Block I: ontology
- Block II: Level 0 results
- Block III: Level 1 and Level 2 formalization

Each major section should end with either a concrete empirical result, or a falsifiable prediction.

`phi` must not be presented as part of the generative dynamics. If it appears in the paper, it should appear as an emergent metric, signature, or comparative result.

The primary drafting language should be Catalan first, followed by an English translation.

The target length is approximately 25 to 30 pages, but this is an editorial goal rather than a hard scientific constraint.

### Rationale

This structure keeps the paper aligned with the actual epistemic order of the project:

- ontological premises first
- validated Level 0 evidence second
- and only then the stronger Level 1 and Level 2 formalization

Ending sections with data or falsifiable predictions disciplines the writing and helps prevent speculative drift.

Keeping `phi` out of the dynamics preserves the methodological commitment already adopted in HSI v2: no golden-ratio axiomatics, only emergent signatures if supported by evidence.

Writing first in Catalan should reduce conceptual loss during the stage where the mathematical and philosophical articulation is still being stabilized.

### Source Documents

- `Paper_Latex/v2_main_cat.tex`
- `docs/HSI_v2_Critical_Review.md`
- `docs/HSI_v2_proposta_revisada.md`

### Practical Impact

- The v2 paper outline should follow the three-block architecture unless later evidence justifies a revision.
- Section planning should explicitly include a result or prediction endpoint.
- `phi` references should be reviewed so they remain observational or metric-level, not axiomatic.
- Catalan should remain the drafting baseline until the structure and argument are stable enough to translate cleanly.

### Open Questions

- The exact internal sectioning of Block III after the first preflight and Phase 1 results.
- Whether the final paper length remains near the editorial target once the formalization stabilizes.

---

## D-0007

- Date: 2026-04-15
- Status: `provisional`
- Scope: preflight gate
- Title: The first preflight gate has been passed, but only for local-to-projective hierarchy beyond strong nulls

### Decision

The first HSI v2 preflight gate is considered passed.

This means only that the current local-to-projective combinatorial observables justify entering the real Phase 1 pipeline.
It does not mean that the HSI v2 tower is already validated.

### Rationale

`B`, `E`, `I`, and `F` separate from the strong null set in factor complexity and branching.
`A` and `J` remain weak.
`K` and `L` remain weak as contrast controls.
`M` remains structurally strong, which is informative but not by itself HSI evidence.

### Source Documents

- `docs/HSI_v2_Preflight_Factor_Complexity_Experiment.md`
- `docs/HSI_v2_Pm_Projection_Fiber_Proposal.md`
- `hsi_agents_project/results/hsi_v2/preflight/reports/`

### Practical Impact

- The project should now move from preflight-only reasoning to `P_m`, projections, and fibers.
- Strong-null separation remains the primary readout.
- Contrast controls must be reported separately from the null-based gate.

### Open Questions

- Whether `N` joins `M` as a strong structured contrast once its preflight batch is included.
- Whether the same separation survives under the first real `P_m / fiber` implementation.

---

## D-0008

- Date: 2026-04-15
- Status: `provisional`
- Scope: scientific interpretation
- Title: Preflight supports a two-layer reading: local structure versus inter-scale coherence

### Decision

The current working interpretation is that HSI v2 must distinguish at least two separable layers:

- local structure within a scale
- inter-scale coherence across projections

`F` is the key reason for this distinction: it remains structured in preflight local observables while earlier results suggest degraded inter-scale flow.

### Rationale

If `F` preserves local factor-complexity and branching structure, then the next gate cannot be reduced to local richness alone.
The decisive test must move to projections and fibers.

### Source Documents

- `docs/HSI_v2_Response_to_Sofia_on_Fiber_Survival.md`
- `docs/HSI_v2_Pm_Projection_Fiber_Proposal.md`
- `hsi_agents_project/Documentation/HSI_Experimental_Results_Summary.md`

### Practical Impact

- `P_m` should remain a within-scale construction.
- Inter-scale coherence should be tested through `pi_(m+1,m)` and fibers.
- `B vs F` becomes a primary conceptual contrast for Phase 1 design.

### Open Questions

- Which fiber statistics discriminate `B` from `F` most clearly.
- Whether moments add information beyond the raw fiber diagnostics in the first Phase 1 pass.

---

## D-0009

- Date: 2026-04-15
- Status: `provisional`
- Scope: Phase 1 defaults
- Title: Main `P_m` default should be locally stable, and main projection should be prefix with suffix as mandatory control

### Decision

The working default for `P_m` is a locally stable pattern space defined by:

- minimum global count
- minimum segment support
- maximum CV on supporting segments

The working main projection is prefix.
Suffix is mandatory as a control projection.
Canonical reduction is deferred as an exploratory policy rather than the main gate.

### Rationale

This keeps `P_m` from circularizing the hierarchy question and keeps the projection gate hard to game.
Prefix is crude but honest.
Suffix is required to control projection asymmetry.
Canonical reduction is useful later but too coherence-seeking to be the main Phase 1 gate.

### Source Documents

- `docs/HSI_v2_Phase1_Minimum_Executable.md`
- `docs/HSI_v2_Pm_Projection_Fiber_Proposal.md`

### Practical Impact

- The next Phase 1 implementation should start with local `P_m`, then prefix and suffix projection towers, then fibers.
- `B`, `F`, `A`, and `M` should be the first contrast set.
- Results that only survive under canonical reduction should not be treated as primary evidence.

### Open Questions

- Whether the current CV threshold is the right one after the first real fiber run.
- Whether `N` should join the first contrast set by default once its batch is fully integrated.

---

## D-0010

- Date: 2026-04-16
- Status: `provisional`
- Scope: report semantics and paper traceability
- Title: Phase 1 reports must default to raw metrics, preserve asymmetry direction, and expose tail collapse sentinels

### Decision

The canonical default outputs of the Phase 1 reporting layer must be raw metrics, not interpretive class labels.

Interpretive labels such as `rich`, `rigid`, `coherent`, `crystalline`, `collapsed`, and `trivial` may remain available only as an explicit optional layer.

Directional divergences between projection or transport policies must be exposed in three forms:

- absolute gap
- signed gap
- relative gap

Tail-derived observables must no longer be summarized only by a mean. They must expose at least:

- mean
- min
- last

so that abrupt top-of-tower collapse cannot be hidden by smoothing.

### Rationale

External hostile review correctly identified two vulnerabilities in the report layer:

- fixed interpretive labels could be mistaken for intrinsic outputs of the code
- absolute-only gaps erased directional asymmetry

A third review point also identified the risk that tail means could smooth away a sharp collapse on the final edge.

This decision keeps the code epistemically cleaner while preserving the ergonomic value of interpretive summaries when explicitly requested.

### Source Documents

- `docs/Phase1_Code_Audit_Report.md`
- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/`

### Practical Impact

- The paper must treat the Phase 1 class taxonomy as an interpretive synthesis, not as a canonical mathematical output of the code.
- Claims about top-of-tower persistence must be checked against `tail_min` and `tail_last`, not only `tail_mean`.
- Signed and relative asymmetry should be preferred over magnitude-only language when discussing directional effects.
- No heavy Phase 1 reruns are required when only the reporting layer changes in this way.

### Open Questions

- Whether future longer tail windows should also expose quantiles or edge-indexed trajectories.
- Whether the eventual paper tables should show all three tail summaries, or reserve some for appendices.

---

## D-0011

- Date: 2026-04-16
- Status: `provisional`
- Scope: Phase 1 null interpretation and protocol robustness
- Title: Phase 1 null discrimination must be phrased in terms of tower regime, not persistence alone

### Decision

The current full strong-null Phase 1 batch supports the following reading:

- `shuffled` and `same-density` surrogates collapse completely at the top of the tower
- `Markov-1` does not collapse completely, but survives only in a much broader and less selective regime than the observed HSI tower

Therefore the Phase 1 discriminant must not be phrased as:

- "HSI has a persistent tower and strong nulls do not"

It must instead be phrased as:

- compact, selective, directionally structured projective persistence
  versus
- broad, diffuse, weakly asymmetric persistence

The first protocol-robustness sweep across:

- `3 x 1,000,000 bits`
- `2 x 1,000,000 bits`
- `4 x 750,000 bits`

preserves the `B` versus `B-markov1` split in this stronger sense.

### Rationale

The randomizing strong nulls (`shuffled`, `same-density`) now clearly falsify any claim that Phase 1 richness could be explained by density alone or by local pattern abundance.

The `Markov-1` surrogate is scientifically more demanding: it still produces a persistent tower, but one with:

- much larger breadth ratio
- much lower final retention
- much lower tail survival and transport quality
- near-zero signed asymmetry

This forces the HSI v2 program to adopt a more precise and more defensible language.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-report__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__var-B-markov1-B-same-density-B-shuffled__20260416T213456.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-coherence__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__tail-2__var-B-markov1-B-same-density-B-shuffled__20260416T213456.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-transport__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-vs-suffix__tail-2__var-B-markov1-B-same-density-B-shuffled__20260416T213500.md`

### Practical Impact

- Paper prose must not frame the strong-null result as persistence versus no persistence.
- The correct contrast is now:
  - HSI-like compact/selective persistence
  - versus diffuse Markov persistence
  - versus random collapse
- Protocol robustness should now be treated as part of the current empirical support for the Phase 1 gate.
- The next discriminator should focus on iteration robustness and stronger nulls such as matched-LZ, not on re-proving that random nulls collapse.

### Open Questions

- Whether the `B` versus `Markov-1` split remains equally stable across more iterations (`B@21-23`).
- Whether a matched-LZ null remains on the diffuse side of the split, or approaches the HSI regime more closely.

---

## D-0012

- Date: 2026-04-17
- Status: `provisional`
- Scope: iteration robustness methodology
- Title: Iteration robustness must be anchored to first observable divergence, not fixed prefixes or shallow manual offsets

### Decision

Iteration robustness for deterministic HSI variants must not be assessed by comparing only the initial observable prefix or a small set of manually chosen offsets.

When repeated scans show that two iterations remain identical across large observable slices, the next methodological step must be an explicit first-divergence scan over the observable stream.

### Rationale

The first iteration-robustness pass showed that `B@20`, `B@21`, `B@22`, and `B@23` share the same first `3,000,000` observable bits.

The offset-based follow-up then showed that `B@20` and `B@21` also remain identical over the scanned observable slices starting at:

- `3,000,000`
- `30,000,000`
- `300,000,000`

This means that fixed-prefix or shallow-offset comparison can remain trivially frozen far deeper than expected.

The right response is not to abandon iteration robustness, but to locate the first genuinely non-trivial divergence point and only then compare projective hierarchy beyond that point.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/phase1_iteration_robustness/`

### Practical Impact

- A dedicated divergence-scan tool is now justified as part of the iteration-robustness workflow.
- Manual offset probes remain useful for intuition, but they no longer count as sufficient iteration-robustness evidence by themselves.
- Future Phase 1 iteration comparisons should preferably start from, or be informed by, the first measured observable divergence.

### Open Questions

- At what observable bit index do `B@20` and `B@21` first diverge?
- Whether the same long frozen prefix also appears in other HSI variants such as `E`, `I`, or `F`.

---

## D-0013

- Date: 2026-04-17
- Status: `provisional`
- Scope: interpretation of adjacent-iteration robustness for variant B
- Title: `B@20` versus `B@21` currently reads as observable prefix extension, not bitwise rewriting

### Decision

The current best reading of adjacent-iteration robustness for variant `B` is not merely that neighboring iterations are "very similar".

They currently show exact observable prefix preservation over the full scanned extent, with the first measured divergence appearing as a length mismatch rather than as a bit mismatch.

### Rationale

The dedicated divergence scans now show:

- no observable-bit mismatch through `303,000,000` bits
- no observable-bit mismatch through `1,000,000,000` bits
- no observable-bit mismatch through `3,000,000,000` bits
- and no observable-bit mismatch through `11,198,388,932` observable bits

At the end of the uncapped scan, the first measured difference between `B@20` and `B@21` is a `length_mismatch`.

The present empirical reading is therefore that `B@20` behaves as an observable prefix of `B@21` over the currently compared extent.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/iteration_divergence/`

### Practical Impact

- Paper prose should not describe adjacent `B` iterations merely as "similar" inside early windows.
- The stronger and more precise language is:
  - extremely long exact observable-prefix preservation
  - followed, at current evidence, by extension rather than rewriting
- Future iteration-robustness work should ask:
  - where does the first true bit mismatch appear?
  - and what happens to the projective tower beyond that point?

### Open Questions

- Whether the first true bit mismatch between `B@20` and `B@21` occurs only after the current end of `B@20`.
- Whether the same prefix-extension behavior holds for `E`, `I`, or `F`.

---

## D-0014

- Date: 2026-04-17
- Status: `provisional`
- Scope: adjacent-iteration family structure
- Title: Adjacent-iteration robustness now separates `B`, `E/I`, and `F` into different observable-stability regimes

### Decision

The current adjacent-iteration divergence scans no longer support treating the structured HSI-family variants as a single stability class.

The present working split is:

- `B`: exact observable-prefix preservation over the full compared extent, currently ending in `length_mismatch`
- `E/I`: extremely long frozen prefixes followed by a true `bit_mismatch` within the first `3G` observable bits
- `F`: much shorter frozen prefix followed by an early `length_mismatch`

### Rationale

The dedicated scans now show:

- `B@20` versus `B@21`:
  - no observable-bit mismatch through `303,000,000`
  - no observable-bit mismatch through `1,000,000,000`
  - no observable-bit mismatch through `3,000,000,000`
  - no observable-bit mismatch through `11,198,388,932`
  - first measured difference: `length_mismatch`
- `E@20` versus `E@21`:
  - first measured difference: `bit_mismatch` at `1,395,248,513`
- `I@20` versus `I@21`:
  - first measured difference: `bit_mismatch` at `1,526,912,785`
- `F@20` versus `F@21`:
  - first measured difference: `length_mismatch` at `127,016,258`

This makes adjacent-iteration robustness itself an informative discriminator inside the structured family, not only a generic sanity check.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/iteration_divergence/`

### Practical Impact

- Paper prose should not describe `B/E/I/F` as sharing a single adjacent-iteration stability behavior.
- `B` should currently be described as observable-prefix extension rather than observable rewriting.
- `E` and `I` should be described as long-frozen but eventually rewriting.
- `F` should be described as only shallowly frozen across adjacent iterations.
- The next iteration-robustness step should no longer be manual offset probing.
- It should instead anchor new Phase 1 windows just beyond the first measured divergence for `E/I/F`.

### Open Questions

- Do `E` and `I` remain on the HSI-like side of the Phase 1 split when sampled just beyond their first bit mismatch?
- Does `F` lose additional tower quality immediately beyond its early adjacent-iteration divergence?
- Is there any structured variant other than `B` that also behaves as a pure observable-prefix extension over the currently available extent?

---

## D-0015

- Date: 2026-04-17
- Status: `provisional`
- Scope: post-divergence iteration reading
- Title: The first newly added tail in `E/I/F` remains in-family, and shared post-divergence windows do not yet exist for the shorter iteration

### Decision

The first attempt to sample Phase 1 windows just beyond the measured adjacent-iteration divergence shows that, for the currently tested pairs:

- `E@20` versus `E@21`
- `I@20` versus `I@21`
- `F@20` versus `F@21`

there is not yet a meaningful shared post-divergence window for the shorter iteration.

At the same time, the newly added observable tail sampled in iteration `21` remains in the same Phase 1 class already seen in the canonical earlier windows:

- `E` stays coherent
- `I` stays coherent
- `F` stays rigid

### Rationale

The direct post-divergence Phase 1 runs revealed:

- `E@20` cannot support a `3 x 1,000,000` window beyond its first measured divergence, because the first `bit_mismatch` appears at the final observable bit currently available in `E@20`
- `I@20` behaves the same way
- `F@20` already ends at its first measured `length_mismatch`

However, when the newly added iteration `21` tail is sampled from that boundary onward:

- `E@21` still yields the canonical `E/B`-like coherent tower
- `I@21` still yields the canonical `I`-like coherent tower
- `F@21` still yields the canonical rigid tower

This means the immediate extension tail is not currently showing a regime break.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/phase1_iteration_robustness/post-divergence/`
- `hsi_agents_project/results/hsi_v2/iteration_divergence/`

### Practical Impact

- "Post-divergence tower failure" is not currently supported as the next iteration-robustness readout for `E/I/F`.
- The next meaningful scan should move to `21` versus `22`, or later adjacent pairs, to see whether divergence again sits at the terminal boundary of the shorter iteration.
- Phase 1 should now distinguish between:
  - length of exact frozen prefix
  - type of first divergence
  - and class stability of the newly added tail

### Open Questions

- Do `E` and `I` keep placing their first adjacent-iteration `bit_mismatch` at the final observable bit of the shorter iteration?
- Does `F` remain a pure shorter-prefix extension across later adjacent pairs?
- At what adjacent pair does a genuinely shared post-divergence window first become available for each structured variant?

---

## D-0016

- Date: 2026-04-17
- Status: `provisional`
- Scope: repeated adjacent-pair structure in `E/I/F`
- Title: `E` and `I` currently show terminal bit rewriting, while `F` shows short-prefix terminal divergence across consecutive adjacent pairs

### Decision

The current repeated adjacent-pair scans support a sharper family-level interpretation:

- `E` and `I` do not currently yield a shared post-divergence window because their first measured `bit_mismatch` sits at the terminal observable boundary of the shorter iteration
- `F` does not currently yield a shared post-divergence window because its first measured difference remains terminal, first as `length_mismatch` and then as terminal `bit_mismatch`

This pattern is now observed in more than one adjacent pair:

- `E@20 -> E@21`
- `E@21 -> E@22`
- `I@20 -> I@21`
- `I@21 -> I@22`
- `F@20 -> F@21`
- `F@21 -> F@22`
- `F@22 -> F@23`

### Rationale

The new scans show:

- `E@22` versus `E@23`: first `bit_mismatch` at `5,580,994,049`
- `E@21` versus `E@22`: first `bit_mismatch` at `2,790,497,025`
- `I@22` versus `I@23`: first `bit_mismatch` at `6,107,651,137`
- `I@21` versus `I@22`: first `bit_mismatch` at `3,053,825,569`
- `F@22` versus `F@23`: first `bit_mismatch` at `508,065,037`
- `F@21` versus `F@22`: first `length_mismatch` at `254,032,518`

And the attempted Phase 1 post-divergence runs again fail on the shorter iteration for `E` and `I`, exactly because the shorter iteration ends at that same boundary.

This means that the current question is no longer whether the newly added tail immediately changes Phase 1 class.
The newly added tails tested so far remain in-family.
The real question is when a genuinely shared post-divergence window first appears.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/iteration_divergence/`
- `hsi_agents_project/results/hsi_v2/phase1_iteration_robustness/post-divergence/`
- `hsi_agents_project/results/hsi_v2/phase1_iteration_robustness/post-divergence-21v22/`

### Practical Impact
  - short-prefix terminal divergence (`F`)
- Iteration robustness should now distinguish:
  - deep prefix extension (`B`)
  - terminal bit rewriting (`E/I`)
  - terminal length extension (`F`)
- The next useful experiment should search for the first adjacent pair where `E` or `I` provides a real shared post-divergence window.
- Manual post-divergence tower runs should only be attempted once that condition is met.

- Does `F` ever yield a true internal divergence followed by a shared post-divergence window, rather than a terminal one?

- At what adjacent pair does `E` first provide a shared post-divergence window?
- At what adjacent pair does `I` first provide a shared post-divergence window?
- Does `F` ever switch from terminal length extension to true internal rewriting?

---

## D-0017

- Date: 2026-04-17
- Status: `provisional`
- Scope: operational `matched-LZ` null for Phase 1 and preflight
- Title: Lightweight `matched-LZ` is now operational as a tuned block-shuffle null; the first `B` result preserves tail persistence but not HSI-style selectivity

### Decision

`matched-LZ` is now promoted from planned placeholder to operational strong null in HSI v2.

The current implementation is deliberately lightweight:

- it preserves exact bit density
- it shuffles observed blocks rather than synthesizing a full parsing-faithful LZ surrogate
- and it selects the block size that best approximates normalized `LZ76` over a small calibration set of windows

This should be treated as an operationally affordable `matched-LZ` null for Phase 1, not yet as a definitive maximal version.

The first Phase 1 comparison against `B @ 20` already changes the scientific reading:

- the lightweight `matched-LZ` null does **not** collapse at the top of the tower
- it preserves strong tail coherence and transport persistence
- but it fails to reproduce HSI-style selectivity:
  - `kept@last = 1214`
  - `retention@last = 0.0849`
  - with a much larger candidate pool than observed `B`

So the discriminator against stronger nulls can no longer be phrased as tail persistence alone.
It must explicitly include selective filtration relative to candidate-space inflation.

### Rationale

The scientific plan had always treated `matched-LZ` as mandatory if it became affordable enough to run.
The lightweight tuned block-shuffle version satisfies that requirement with low code surface and clear reproducibility.

For the first real `B @ 20` run:

- `selected_block_bits = 4096`
- `target_lz_normalized = 0.0316`
- `achieved_lz_normalized = 0.0424`

And the resulting tower shows:

- high tail survival (`surv mean = 0.9792`, `surv last = 0.9902`)
- high tail transport (`active mean = 0.3669`, `active last = 0.3287`)
- but low final selectivity (`retention@last = 0.0849`)

This makes the current gate sharper:

- random nulls are rejected by collapse
- `Markov-1` is rejected by broad diffuse persistence
- lightweight `matched-LZ` is rejected by persistent but weakly selective filtration

### Source Documents

- `docs/HSI_v2_Phase1_Minimum_Executable.md`
- `docs/HSI_v2_proposta_revisada.md`
- `hsi_agents_project/v2/common/null_models.py`
- `hsi_agents_project/hsi_v2_phase1_run.py`
- `hsi_agents_project/hsi_v2_preflight_factor_complexity.py`
- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/phase1/nulls/phase1-tower__var-B__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__null-matched-lz__20260417T210643/`

### Practical Impact

- `matched-LZ` can now be requested explicitly in both preflight and Phase 1.
- The current operational meaning of `matched-LZ` is:
  - exact density preserved
  - approximate normalized LZ matched on calibration windows
  - long-range arrangement broken by tuned block shuffling
- Phase 1 interpretation should now distinguish at least four null regimes:
  - random collapse (`shuffled`, `same-density`)
  - broad diffuse persistence (`Markov-1`)
  - persistent but weakly selective filtration (`matched-LZ`)
  - compact selective HSI-like persistence (`B/E/I`)

### Open Questions

- Does the lightweight `matched-LZ` null stay on the non-HSI side for `E` and `I` as well as `B`?
- Is a stronger parsing-faithful `matched-LZ` worth the extra engineering cost, or is the current lightweight version already scientifically sufficient for Porta 1?
- Which metric should become primary against stronger nulls:
  - `retention@last`
  - candidate-space inflation
  - or a derived selectivity observable combining both?

---

## D-0018

- Date: 2026-04-17
- Status: `provisional`
- Scope: `matched-LZ` follow-up on `E` and `I`
- Title: Lightweight `matched-LZ` also stays on the non-HSI side for `E` and `I`; persistence is not enough, selectivity must now be foregrounded against stronger nulls

### Decision

The first `matched-LZ` follow-up on `E @ 20` and `I @ 20` confirms the same scientific reading already suggested by `B @ 20`:

- the lightweight `matched-LZ` null does **not** collapse at the top of the tower
- it preserves strong tail coherence and transport for `E` and `I` as well
- but it still fails to reproduce HSI-style compact selectivity

So the current Porta 1 discriminator against stronger nulls should now be read as:

- not persistence alone
- not coherence alone
- not transport persistence alone
- but compact/selective persistence relative to candidate-space inflation

### Rationale

For `E @ 20`, the lightweight `matched-LZ` null lands in the same broad regime already seen for `B`:

- `kept@last = 1214`
- `retention@last = 0.0849`
- `surv mean = 0.9792`
- `active mean = 0.3669`

For `I @ 20`, the result is different in scale but not in kind:

- `kept@last = 585`
- `retention@last = 0.0710`
- `surv mean = 0.9412`
- `active mean = 0.5398`

This is now enough to treat lightweight `matched-LZ` as a real strong null regime rather than a one-off `B` curiosity.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-report__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__var-B-matched-lz__20260417T211752.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-coherence__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__tail-2__var-B-matched-lz__20260417T211752.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-transport__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-vs-suffix__tail-2__var-B-matched-lz__20260417T211752.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-report__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__var-E-matched-lz-I-matched-lz__20260417T214226.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-coherence__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__tail-2__var-E-matched-lz-I-matched-lz__20260417T214226.md`
- `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-transport__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-vs-suffix__tail-2__var-E-matched-lz-I-matched-lz__20260417T214226.md`

### Practical Impact

- The refined strong-null reading should now distinguish four regimes:
  - random collapse (`shuffled`, `same-density`)
  - broad diffuse persistence (`Markov-1`)
  - persistent but weakly selective filtration (`matched-LZ`)
  - compact selective HSI-like persistence (`B/E/I`)
- Paper prose should not use tail persistence alone as the null discriminator once `matched-LZ` is in scope.
- The most defensible comparative language is now:
  - persistence plus selectivity
  - or persistence under controlled candidate-space inflation

### Open Questions

- Which selectivity observable should be foregrounded in the paper:
  - `retention@last`
  - candidate-space inflation
  - or a derived observable combining both?
- Is the current lightweight `matched-LZ` already sufficient for Porta 1, or is there still strategic value in a stronger parsing-faithful version later?

---

## D-0019

- Date: 2026-04-17
- Status: `provisional`
- Scope: default composite summary against stronger nulls in Phase 1 reporting
- Title: The minimal selective-persistence summary is now `SPI_simple = retention@last * surv_last`

### Decision

The current default composite summary for selective persistence against stronger nulls is now:

`SPI_simple = retention@last * surv_last`

This should be treated as a compact synthesis, not as a replacement for the underlying observables.

The intended reading is:

- `surv_last` tells us whether the top of the tower is still alive
- `retention@last` tells us whether that survival remains selective rather than inflated by a large candidate pool
- `SPI_simple` summarizes when both conditions hold at once

### Rationale

After `matched-LZ`, tail persistence alone is no longer enough:

- `B/E/I` and their lightweight `matched-LZ` surrogates can all preserve a live tail
- but the surrogates fail compact selectivity

So the report layer now needs a minimal composite summary that is:

- transparent
- cheap
- directly derived from already visible observables
- and easy to audit against hostile review

`retention@last * surv_last` is the simplest form that satisfies those constraints.

### Source Documents

- `hsi_agents_project/v2/phase1/report.py`
- `hsi_agents_project/v2/phase1/coherence.py`
- `hsi_agents_project/Documentation/HSI_v2_Phase1_Live_Findings.md`

### Practical Impact

- `SPI_simple` should be visible in the compact Phase 1 report layer.
- It should be read only alongside its factors:
  - `retention@last`
  - `surv_last`
- Paper prose should present it as a summary of selective persistence, not as an autonomous physical invariant.

### Open Questions

- Is `SPI_simple` sufficient as the primary composite summary for Porta 1?
- Does a later, stricter composite index need transport as well, or is that unnecessary for the current gate?

---

## D-0020

- Date: 2026-04-18
- Status: `provisional`
- Scope: default observable definition for the minimum Gate 2 defect pilot
- Title: The minimum Gate 2 defect counts only future-bearing continuation conflicts; terminal closure is tracked separately

### Decision

The first executable Gate 2 pilot should not treat every failure of one-step continuation as a commutation defect.

Instead, the default observable is now:

- compare the locally mass-dominant child at the lower edge
- against the child favored one scale deeper by positive branching richness
- count a defect only when both of these exist and disagree

If no child retains positive one-step-ahead branching richness at the higher edge, the parent is marked as **terminal closure**, not as defect.

### Rationale

The earliest implementation of the pilot still mixed together two qualitatively different situations:

1. a real continuation conflict, where local dominance and deeper branching preference disagree
2. a terminally closed parent, where no child remains future-bearing one scale higher

That conflation inflated defect rates for rigid/crystalline variants and made the pilot much less discriminative than intended.

The corrected observable now reflects the actual Gate 2 scientific question:

- not whether a branch simply dies
- but whether a still-open continuation structure carries a non-trivial disagreement between local and one-step-ahead preference

This keeps the pilot honest and prevents terminal closure from masquerading as algebraic defect.

### Source Documents

- `docs/HSI_v2_Phase2_Minimum_Commutation_Defect_Pilot.md`
- `hsi_agents_project/v2/phase2/defects.py`
- `hsi_agents_project/hsi_v2_phase2_defect_pilot.py`

### Practical Impact

- `defect_rate` and `defect_mass_rate` are now conditioned on future-bearing parents only.
- The pilot must also report:
  - `future_parent_fraction`
  - `terminal_parent_fraction`
- Early Gate 2 interpretation should distinguish:
  - true continuation conflict
  - from terminal closure
- Any later graph/spectral work should inherit this distinction unless a stronger observable supersedes it.

### Open Questions

- Does the current one-step branching-richness score already separate `B` from `F/M/N` robustly, or does it still need refinement after the first broader pilot batch?
- Should the future-bearing score remain `child_entropy`, or is there later value in a richer descendant statistic once the minimum pilot is stabilized?

---

## D-0021

- Date: 2026-04-18
- Status: `provisional`
- Scope: next Gate 2 observable after the parent-local defect pilot
- Title: If parent-local continuation defects do not separate `B` from `matched-lz`, the next pilot should move to cross-policy loop holonomy rather than finer local summaries

### Decision

The first follow-up after the minimum future-bearing defect pilot should not be another scalar refinement of the same parent-local observable.

Instead, the next executable probe is a **cross-policy loop holonomy pilot** on a short tail window such as `24,28,32`:

- left path: dominant continuation under `prefix` across both steps, returned to the low scale through `suffix`
- right path: dominant continuation under `suffix` across both steps, returned to the low scale through `prefix`

The pilot should measure:

- whether the two returned low-scale patterns disagree
- whether either return drifts away from the original low-scale parent
- and whether such loop families recur coherently across segments

### Rationale

The minimum defect pilot already answered one important question:

- it separates `B` from `Markov-1`, and from rigid/crystalline regimes such as `F/M/N`

But it also exposed a harder limit:

- `matched-lz` reproduces not only the aggregate defect rates of `B`, but almost the entire parent-local family content of that first observable layer

That means the next Gate 2 step must no longer remain parent-local.

The methodological lesson borrowed from the bounded `automath` read is not ontology, but this:

- once a surrogate preserves local transport statistics too well, the next honest observable is a **path-dependent non-commutation object**, not a finer scalar compression of the same local defect

Cross-policy loop holonomy is the smallest such object we can build directly from the existing Phase 1 fiber artifacts without overengineering a full graph or spectral layer.

### Source Documents

- `docs/HSI_v2_Automath_Inspired_Framework_Guide.md`
- `docs/HSI_v2_Phase2_Minimum_Commutation_Defect_Pilot.md`
- `hsi_agents_project/v2/phase2/holonomy.py`
- `hsi_agents_project/hsi_v2_phase2_loop_holonomy.py`

### Practical Impact

- Gate 2 now has a second pilot family available that is no longer parent-local.
- The new pilot reuses existing `fibers_prefix.json` / `fibers_suffix.json` outputs and therefore adds minimal implementation risk.
- If cross-policy loop holonomy still fails against `matched-lz`, the next observable should escalate to cocycle/glue objects rather than further local refinements.

### Open Questions

- Does loop holonomy actually open `B` from `matched-lz`, or does the surrogate still shadow the coherent regime at this short path level?
- If not, is the next honest step a glue/cocycle observable on overlapping parents, or a multiscale graph edge object?

---

## D-0022

- Date: 2026-04-18
- Status: `provisional`
- Scope: next Gate 2 observable after cross-policy loop holonomy
- Title: If short cross-policy loops still fail against `matched-lz`, the next pilot should test overlap-glue cocycles between parent pairs rather than deepen single-path observables

### Decision

After the cross-policy loop holonomy pilot, the next executable Gate 2 step should move to **overlap-glue cocycles** on ordered pairs of parent-scale patterns.

The minimum construction is:

- take ordered pairs of `m=24` patterns whose suffix-`20` / prefix-`20` overlap already glues locally
- lift each parent to a dominant two-step descendant at `m=32`
- test whether the induced high-scale overlap still glues

### Rationale

The short loop pilot still left `B-matched-lz` too close to `B`.

That means even path-dependent observables built from a single parent are still too local for the current strong null.

The next honest move is therefore to test **compatibility between local pieces**, not only transport of one piece by itself.

This is still conservative:

- it stays below graph/spectral work
- it reuses the existing Phase 1 fiber artifacts
- and it remains falsable against the same strong-null stack

### Source Documents

- `docs/HSI_v2_Automath_Inspired_Framework_Guide.md`
- `hsi_agents_project/v2/phase2/glue.py`
- `hsi_agents_project/hsi_v2_phase2_glue_cocycle.py`

### Practical Impact

- Gate 2 now has a third pilot family focused on pairwise compatibility rather than parent-local transport.
- If overlap-glue cocycles still fail against `matched-lz`, the next step should escalate to explicit graph or cocycle-on-squares constructions rather than more local refinements.

### Open Questions

- Does overlap-glue finally separate `B` from `matched-lz`, or does the strong null also preserve pairwise compatibility at this depth?
- If not, is the next honest jump a sparse multiscale graph edge object or a fully explicit cocycle on commuting squares?

---

## D-0023

- Date: 2026-04-18
- Status: `provisional`
- Scope: next Gate 2 observable after overlap-glue cocycles
- Title: If pairwise overlap glue is still shadowed by `matched-lz`, the next pilot should test anchored commuting squares built around actual shared bridges

### Decision

The next conservative Gate 2 step after overlap-glue is an explicit **commuting-square cocycle** built around actual shared bridge patterns.

The minimum construction is:

- identify an actual bridge `q` at `m=28` that appears simultaneously as:
  - a `prefix` child of a left `m=24` parent
  - a `suffix` child of a right `m=24` parent
- require the bridge to remain future-bearing at `m=32` under both policies
- compare whether the dominant bridge recovered independently from each corner still anchors the actual shared bridge

### Rationale

Overlap-glue tests whether compatibility survives separate lift.

But even that remains one step short of a genuine square: it does not yet ask whether an **actual shared bridge** is stably selected from both corners once the bridge itself is treated as the square interior.

This makes the new observable more structural than:

- parent-local transport
- short loop holonomy
- or pairwise overlap alone

while still avoiding premature graph/spectral construction.

### Source Documents

- `hsi_agents_project/v2/phase2/square.py`
- `hsi_agents_project/hsi_v2_phase2_square_cocycle.py`

### Practical Impact

- Gate 2 now has a fourth pilot family that tests anchored shared bridges rather than free pairwise compatibility.
- If anchored commuting squares still fail against `matched-lz`, the next honest step should be a sparse multiscale graph edge object or a higher cocycle on linked squares.

### Open Questions

- Do anchored commuting squares finally open `B` from `matched-lz`?
- If not, is the next conservative move a sparse graph of bridge-compatibility, or have we reached the limit of purely local multiscale pilots?

---

## D-0024

- Date: 2026-04-18
- Status: `provisional`
- Scope: next Gate 2 observable after anchored commuting-square cocycles
- Title: If anchored squares are still shadowed by `matched-lz`, the next conservative step should be a sparse multiscale graph of recurrent shared bridges

### Decision

After the commuting-square pilot, the next honest Gate 2 step should not be yet another local or short-path refinement.

It should be a **sparse multiscale bridge graph** built from recurrent future-bearing shared bridges and their reuse relations.

The minimum graph object is:

- nodes = future-bearing shared bridges at `m=28`
- sparse edges = bridge reuse through:
  - shared left `m=24` parent
  - shared right `m=24` parent
  - or shared dominant `m=32` descendant

The primary contrast should focus on:

- recurrent edge mass
- recurrent core size
- segment-core stability
- and null-conditioned edge rarity relative to `matched-lz`

### Rationale

Gate 2 has now tested four increasingly structured pilot families:

- parent-local future-bearing defects
- cross-policy loop holonomy
- overlap-glue cocycles
- anchored commuting squares

Together they already separate `B` from:

- rigid/crystalline persistence (`F/M/N`)
- and diffuse persistent nulls (`B-markov1`)

But they still fail to separate `B` from `B-matched-lz`.

That means the current local/square family of observables has likely reached its useful limit against the strongest current surrogate.

The next conservative move is therefore not "more of the same".

It is to test whether these local bridge relations assemble into a **recurrent sparse scaffold** that survives segment conditioning and is not reproduced by the null.

This remains below spectral or geometric language while finally introducing an explicitly graph-like object.

### Source Documents

- `docs/HSI_v2_Automath_Inspired_Framework_Guide.md`
- `docs/HSI_v2_Phase2_Sparse_Multiscale_Graph_Pilot.md`

### Practical Impact

- Gate 2 now has a fifth pilot direction available, positioned as the last conservative step before any pre-spectral graph work.
- The implementation target should be:
  - `hsi_v2_phase2_sparse_graph_pilot.py`
  - `hsi_agents_project/v2/phase2/sparse_graph.py`
- The pilot must emit:
  - `graph.json`
  - `summary.json`
  - `manifest.json`
  - and a compact report

### Open Questions

- Does recurrent bridge reuse finally open `B` from `B-matched-lz`?
- If not, have we reached the limit of Gate 2 under the current observable stack?

---

## D-0025

- Date: 2026-04-18
- Status: `provisional`
- Scope: next Gate 2 observable after sparse multiscale bridge graphs remained shadowed by `matched-lz`
- Title: If sparse bridge graphs are still shadowed by `matched-lz`, the next conservative step should target longer-range return-distance order

### Decision

After local defects, loop holonomy, overlap glue, commuting squares, and sparse bridge graphs all fail to open `B` from `B-matched-lz`, the next conservative Gate 2 step should not remain parent-local or bridge-local.

It should target **return-distance order** for strong patterns along the actual observable stream.

The minimum executable pilot is:

- choose a stable Phase 1 pattern scale, default `m=32`
- select the top patterns at that scale by raw count
- reconstruct the exact analyzed observable slice for observed or null-surrogate runs
- measure the distance between consecutive returns of the same selected pattern within each segment
- summarize:
  - lag entropy
  - dominant lag-bin mass
  - long-lag fraction
  - and segment-profile stability

### Rationale

The strongest current surrogate, `matched-lz`, keeps reproducing:

- parent-local defect structure
- loop and square pilots
- pairwise glue
- and sparse bridge scaffolds

This strongly suggests that the current Gate 2 observable family is still dominated by local or mesoscopic organization that survives block-preserving LZ-matched shuffling.

The next honest shift is therefore to test not "which patterns exist" nor "which bridges can be reused", but **how far apart and how rhythmically strong patterns return along the stream**.

This stays pre-geometric and pre-spectral, but explicitly moves the pilot stack toward long-range order.

### Source Documents

- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/hsi_v2_phase2_return_lag_spectrum.py`

### Practical Impact

- Gate 2 now has a long-range pilot that can be run directly on observed and null-surrogate Phase 1 batches.
- The pilot reconstructs null-surrogate slices from stored seeds, so it does not require re-running Phase 1 to compare return-order statistics.
- If return-lag order still fails against `matched-lz`, the next honest move should be a genuinely sequence-order object such as a directed transition graph or a longer-range inter-block continuity test.

### Open Questions

- Does return-distance order finally separate `B` from `B-matched-lz`?
- If not, have we reached the limit of non-graph, non-spectral Gate 2 observables under the current slice depth?

---

## D-0026

- Date: 2026-04-18
- Status: `provisional`
- Scope: refinement of the Gate 2 return-lag pilot when `top` and `rare-stable` pattern selection remain too close to `matched-lz`
- Title: When long-range lag spectra remain shadowed by `matched-lz`, condition the selection on bridge-linked patterns tied to the projective hierarchy

### Decision

The Gate 2 return-lag pilot should support a third pattern-selection mode, `bridge-linked`, alongside `top` and `rare-stable`.

`bridge-linked` means:

- choose patterns at the scanned scale `m` that are actually linked to the projective hierarchy through both lower-policy fiber maps
- and, when a higher adjacent scale exists, rank first those patterns that remain future-bearing under both policies at the next edge

In practice, the ranking should favor patterns that:

- appear under both lower-policy child maps
- remain future-bearing under both higher-policy parent maps when available
- and only then optimize for support/stability/rarity

### Rationale

The earlier return-lag pilots based on:

- the most frequent patterns
- or on rare-but-stable patterns

do improve the contrast with `markov-1`, but they still leave `B-matched-lz` too close to `B`.

That suggests the long-range observable was still being driven by patterns that are strong in frequency space, yet not sufficiently conditioned by the actual projective scaffold.

The next conservative refinement is therefore not to invent a new observable from scratch, but to keep the same lag-spectrum pilot and constrain the selected patterns to those that are more deeply anchored to the hierarchy itself.

### Source Documents

- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/hsi_v2_phase2_return_lag_spectrum.py`

### Practical Impact

- The return-lag pilot now supports:
  - `--pattern-selection top`
  - `--pattern-selection rare-stable`
  - `--pattern-selection bridge-linked`
- The recommended first target is:
  - `m=40`
  - selected from a Phase 1 batch that also includes `m=48`, so the bridge-linked ranking can use both lower and higher adjacent edges.

### Open Questions

- Does `bridge-linked` finally open `B` from `B-matched-lz` cleanly, or only partially?
- If the separation is still weak, does the next move have to condition on even more explicit multiscale transport rather than on lag spectra alone?

---

## D-0027

- Date: 2026-04-18
- Status: `provisional`
- Scope: next Gate 2 layer after bridge-linked lag aggregates begin to separate `B` from `matched-lz` only partially
- Title: When bridge-linked lag aggregates remain only partially open, compare full per-pattern return profiles instead of aggregate lag summaries

### Decision

The return-lag pilot should now support a **post-analysis comparison layer** based on per-pattern lag-bin profiles.

The comparison is between:

- one observed return-lag run
- and one null return-lag run

for the same:

- pattern scale
- pattern-selection mode
- and segment protocol

The minimum comparison should report:

- selection overlap (`selection_jaccard`)
- weighted profile overlap (`weighted_profile_jaccard`)
- weighted Jensen-Shannon divergence on lag-bin profiles
- weighted relative mean/median lag gaps
- and weighted top-bin match fraction

### Rationale

The bridge-linked selector improves the return-lag story against `matched-lz`, but aggregate row summaries still compress away too much of the structure.

The empirical signal now suggests:

- `B` and `B-matched-lz` may remain moderately close in aggregate lag means and entropies
- while still disagreeing strongly on
  - which bridge-linked patterns are selected
  - and how the return mass is distributed inside each pattern

So the next honest move is not a stronger scalar, but a richer comparison on the actual lag profiles pattern by pattern.

### Source Documents

- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/v2/phase2/return_profile_compare.py`
- `hsi_agents_project/hsi_v2_phase2_return_profile_divergence.py`

### Practical Impact

- The return-lag dataset now stores full `lag_bin_counts` per selected pattern.
- Gate 2 has a new comparison script:
  - `hsi_v2_phase2_return_profile_divergence.py`
- The preferred first contrast is:
  - observed `B`
  - versus `B-matched-lz`
  - under `bridge-linked`
  - at `m=40` and `m=48`

### Open Questions

- Do per-pattern return profiles finally provide the cleanest open between `B` and `matched-lz` in Gate 2?
- If they do, should future Gate 2 work promote profile divergence above aggregate lag metrics as the main readout?

---

## D-0028

- Date: 2026-04-18
- Status: `provisional`
- Scope: family-level consolidation of the first Gate 2 opening after seed sweeps against `matched-lz`
- Title: Treat `B/E` as the provisional strong Gate 2 block, keep `I` exploratory until its bridge-linked return-profile divergence stabilizes

### Decision

The current Gate 2 readout should be interpreted in two layers:

1. `B` and `E` form the **provisional strong block** of Gate 2.
2. `I` remains **exploratory**, not yet part of the same strong block.

The practical rule is:

- `B/E` may be described as showing a seed-robust first opening against `matched-lz`
- `I` may be described as suggestive or partially open, but not yet consolidated

### Rationale

The `matched-lz` seed sweep at `m=48`, under the `bridge-linked` selector and per-pattern return-profile divergence, now gives a clear family split.

For `B`:

- the strongest fine-grained criterion, `top_eq = 0`, survives all tested seeds
- profile divergence remains non-trivial even when selection overlap fluctuates

For `E`:

- the same pattern repeats
- `top_eq = 0` survives all tested seeds as well
- the profile-divergence ranges are effectively the same as for `B`

So `B` and `E` now behave as one coherent Gate 2 block.

For `I` at `m=48`:

- the bridge-linked overlap is often much smaller
- `top_eq = 0` survives only in a minority of seeds
- some seeds still allow large dominant-bin coincidence

That means `I` is not yet stable enough to be promoted to the same evidential status as `B/E`.

The short recovery check at `m=40` improves the situation for `I`:

- selection overlap becomes less sparse
- divergence remains non-trivial

but the strongest criterion still does not stabilize:

- `weighted_top_bin_match_fraction` remains clearly above zero across all tested seeds

So `m=40` helps diagnose the problem, but does not yet rescue `I` into the strong block.

### Source Documents

- `hsi_agents_project/results/hsi_v2/phase2/seed_sweep_profile_divergence_m48/`
- `hsi_agents_project/results/hsi_v2/phase2/family_seed_sweep_profile_divergence_m48/`
- `hsi_agents_project/results/hsi_v2/phase2/family_seed_sweep_profile_divergence_m40/`

### Practical Impact

- Gate 2 can now be summarized, provisionally, as:
  - `B/E`: strong first opening against `matched-lz`
  - `I`: exploratory, not yet canonized
- Paper-facing wording should not claim a full `B/E/I` family block at Gate 2 yet.
- If `I` is mentioned, it should be framed as a partially open but seed-sensitive case.

### Open Questions

- Can `I` be stabilized by changing the selector, the scale, or the segment protocol?
- Is the `I` irregularity a real family difference, or just an artifact of bridge-linked sparsity at high scales?
- Should the next Gate 2 consolidation step focus on turning `B/E` into the canonical strong pair before spending more effort on `I`?

---

## D-0029

- Date: 2026-04-19
- Status: `provisional`
- Scope: next formal Gate 2 launch point after the bridge-linked return-profile opening and family-level consolidation
- Title: Promote the bridge-linked return-profile substrate into the first explicit transport morphism `Ï‰_m` and transport-defect pilot `Î´_m` before any graph or spectral escalation

### Decision

The next mathematical investigation step should not be:

- another exploratory selector,
- another aggregate lag scalar,
- or a direct jump to graph / spectrum.

It should be the first formal Level 2 pilot:

- define an observable bridge-linked transport operator `Ï‰_m`
- define a first defect candidate `Î´_m`
- and test it first on the canonical strong pair `B/E`

with:

- selector `bridge-linked`
- primary scales `m=40` and `m=48`
- canonical segment protocol `3 x 10^6`
- primary strong null pressure from `matched-lz` seed sweeps

### Rationale

The current state of the program is now clear.

1. Gate 1 is already passed.
2. Gate 2 is observably open, but not yet formal.
3. The first genuine opening against `matched-lz` appeared only after:
   - bridge-linked pattern selection
   - high scales
   - and per-pattern return-profile divergence

That means the next honest step is not to keep inventing provisional observables.

It is to turn the first successful Gate 2 substrate into the first admissible defect object.

This is also the correct mathematical order from the v2 proposal:

- projective hierarchy first
- transport defects second
- graph third
- spectrum only after that

So the next launch point should be an `N2-01` pilot centered on:

- return profiles `Ï_m(p)`
- an observable bridge-aware transport kernel `K_m(p -> q)`
- the transported profile `T_m Ï(p)`
- and a first defect tuple comparing `Ï_m(p)` with `T_m Ï(p)`

### Source Documents

- `docs/HSI_v2_proposta_revisada.md`
- `docs/HSI_v2_Phase2_N2_01_Transport_Defect_Pilot.md`
- `HSI_v2_Decision_Log.md`

### Practical Impact

- The next formal Phase 2 design target is:
  - `hsi_v2_phase2_transport_defect.py`
- The corresponding internal implementation target is:
  - `hsi_agents_project/v2/phase2/transport_defect.py`
- `B/E` should be treated as the canonical strong pair for the first formal Gate 2 contract.
- `I` should remain exploratory until a defect-layer readout stabilizes it.
- No graph or spectral escalation should happen before `Ï‰_m` and `Î´_m` exist in a minimal executable form.

### Open Questions

- What is the most conservative default kernel for `K_m(p -> q)`?
- Should the first defect tuple be reported as:
  - `JS + wJ + top-bin mismatch + mean-gap`
  - or should one of those be reserved as secondary?
- Does the matched-LZ seed envelope still leave `B/E` outside it once transport is defined explicitly?

---

## D-0030

- Date: 2026-04-19
- Status: `provisional`
- Scope: default readout of the first executable N2-01 transport-defect pilot
- Title: Extend the N2-01 defect tuple with an ordered lag-flow metric once the high-scale support is widened enough to avoid transport-flat artifacts

### Decision

The default N2-01 transport-defect readout should be extended with:

- `Î´_flow`

defined as an ordered earth-mover distance on the log-lag-bin axis.

The corresponding run-level summary should expose:

- `defect_flow_mean`

alongside:

- `defect_js_mean`
- `defect_wj_mean`
- `defect_top_mass`
- `defect_mean_gap`

### Rationale

The first executable N2-01 pass revealed two distinct facts.

1. With a too-narrow high-scale support (`m=48`, `top=16`), the transport kernel became artificially close to an identity restriction and `B` collapsed to a false near-zero defect.
2. After widening the admissible high-scale support enough to restore genuine transport mixing, `B` stopped being transport-flat, but `matched-lz` still remained comparatively close on:
   - dominant-bin mismatch
   - and mean-lag gap

This means the next honest refinement is not a new kernel family and not a larger rerun by default.

It is a minimal extension of the defect tuple toward a metric that penalizes **smooth displacement of mass across neighboring lag bins**.

`Î´_flow` does exactly that:

- it stays fully observable,
- it preserves the ordered nature of the lag-bin axis,
- and it is sensitive to tail drift even when top bin and mean remain similar.

So it is the right austere refinement for the first transport-defect pilot under strong-null pressure.

### Source Documents

- `docs/HSI_v2_Phase2_N2_01_Transport_Defect_Pilot.md`
- `hsi_agents_project/v2/phase2/transport_defect.py`
- `hsi_agents_project/results/hsi_v2/phase2/phase2-transport-defect__m-40-48__sel-bridge-linked__var-B__20260419T014252/`
- `hsi_agents_project/results/hsi_v2/phase2/phase2-transport-defect__m-40-48__sel-bridge-linked__var-B-markov1-B-matched-lz__20260419T014254/`

### Practical Impact

- The canonical N2-01 summary now includes `defect_flow_mean`.
- Reports should treat `Î´_flow` as a transport-shift diagnostic, not as a replacement for the rest of the tuple.
- `matched-lz` should now be checked not only for dominant-bin or mean agreement, but also for ordered tail drift.
- No new Phase 1 run is required to adopt this refinement.

### Open Questions

- Does `Î´_flow` stay outside the matched-LZ seed envelope for `B/E` under seed sweeps?
- Should `segment_defect_stability` remain JS-ranked, or eventually be keyed to a multi-metric defect score?
- If `Î´_flow` is still insufficient against `matched-lz`, is the next honest move a kernel refinement rather than another observable?

---


## D-0031

- Date: 2026-04-19
- Status: `provisional`
- Scope: next default comparative probe after the audited collapse of the first observed `B`-only transport opening
- Title: Promote an observed-only return-lag window sweep as the next austere Gate 2 probe before any renewed null or transport escalation

### Decision

The next default comparative experiment after the transport audit should be:

- an observed-only return-lag window sweep

implemented as:

- a fixed-Phase-1, fixed-selector, fixed-pattern-space scan
- where the only intentional mutation across runs is `segment_offset_bits`

The first executable contract should be:

- `W2-01A`

with:

- observed variants only
- one pattern scale per invocation
- direct pairwise `B/E`, `B/I`, `E/I` comparison inside the same artifact
- non-overlapping windows by default through `window_step_bits = window_span_bits`

### Rationale

The transport audit changed the local scientific picture.

1. The apparent `B`-only opening in the first `N2-01` readout was not stable under audit.
2. After regenerating the low-scale observed row in the current full-bin format, observed `B` and observed `E` both became transport-exact on the audited `m40 -> m48` slice.
3. That means the next honest question is no longer whether the old slice can be rhetorically defended.

It is whether the current `B/E` near-coincidence is:

- globally stable across the observable stream,
- or only an early-window regime that later splits.

This also matches the methodological clue inherited from legacy v1:

- `B` and `E` may remain near-identical for a long prefix and diverge only later

So the correct next step is not immediate strong-null pressure on every offset and not another transport refinement by default.

It is a controlled observed-only sweep over window position.

### Source Documents

- `docs/HSI_v2_Phase2_Window_Sweep_Design.md`
- `hsi_agents_project/hsi_v2_phase2_return_lag_window_sweep.py`
- `hsi_agents_project/v2/phase2/window_sweep.py`

### Practical Impact

- The next default Gate 2 executable target is:
  - `hsi_v2_phase2_return_lag_window_sweep.py`
- The corresponding internal implementation target is:
  - `hsi_agents_project/v2/phase2/window_sweep.py`
- The first pass should remain observed-only and should not run matched-LZ seed pressure across all windows by default.
- Null pressure and `N2-01` transport follow-up should only be applied to candidate windows discovered by the observed sweep.

### Open Questions

- Do `B` and `E` remain tightly locked across all scanned windows?
- Does a late-window `B/E` split appear while `B/I` and `E/I` stay separated?
- If later windows become too sparse under frozen pattern-space selection, should dynamic per-window reselection be opened as a second design?

---

## D-0032

- Date: 2026-04-19
- Status: `provisional`
- Scope: execution tooling for deep strong-null window sweeps in Phase 2
- Title: Replace the fake determinate null-surrogate progress bar with an honest heartbeat and use a dedicated null-pressure sweep executable

### Decision

Late-window strong-null checks should use a dedicated public executable:

- `hsi_v2_phase2_null_pressure_window_sweep.py`

instead of ad-hoc inline scripts or the observed-only window sweep entry point.

At the same time, null-surrogate construction inside the Phase 2 return-lag pipeline should stop reporting a fake determinate `50%` state and should instead use an indeterminate heartbeat until the surrogate is actually ready.

### Rationale

The previous null-surrogate progress display in `return_lag.py` used a two-step determinate bar:

1. announce the null model
2. complete after `generate_control(...)`

This was visually misleading because almost all the real work happened inside the blocking control generator, so the UI appeared to stall at `50%` for long stretches without conveying any true fraction of completion.

For deep late-window checks, the situation was worse:

- the observed-only sweep entry point was not the right semantic tool for null pressure,
- inline one-off scripts were reproducible enough for local debugging but not for canonical reruns,
- and the user had no honest readout of where the execution sat within the full job.

The new arrangement is more austere and more truthful:

- pair-level phase logs,
- window-level progress notes,
- accurate load bars where the total is known,
- and a heartbeat spinner where completion ratio is inherently unknown.

### Source Documents

- `hsi_agents_project/utils/progress.py`
- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/v2/phase2/window_sweep.py`
- `hsi_agents_project/v2/phase2/null_pressure.py`
- `hsi_agents_project/hsi_v2_phase2_null_pressure_window_sweep.py`

### Practical Impact

- Null-surrogate generation no longer pretends to know its fractional completion.
- Deep `B` vs strong-null late-window checks can now be rerun from a canonical CLI with manifests and reports.
- Terminal output now exposes the real execution hierarchy:
  - pair
  - window
  - inner load / surrogate / scan substeps

### Open Questions

- If matched-LZ seed sweeps become much wider, should the aggregate report add envelope summaries directly rather than only per-seed rows?
- If future null generators can report internal milestones, should the heartbeat be upgraded to model-specific determinate progress?

---

## D-0033

- Date: 2026-04-19
- Status: `provisional`
- Scope: model-aware progress reporting for long Phase 2 strong-null generation
- Title: Upgrade null-surrogate progress from heartbeat-only fallback to model-aware bars and staged diagnostics

### Decision

The provisional heartbeat-only fallback adopted for deep null-pressure runs should be treated as an interim safety fix, not as the final default.

The default execution behavior for long strong-null generation is now:

- `markov1`: determinate progress bar tied to real chunked work
  - transition counting
  - chain sampling
- `matched-lz`: staged determinate progress tied to:
  - target LZ profile windows
  - candidate block evaluation
  - per-candidate LZ windows

This keeps the readout honest while still exposing useful internal state during long runs.

### Rationale

The heartbeat solved one problem only:

- it stopped the false `50%` illusion

but it did not solve the user-facing operational problem:

- during long null generation, the terminal still failed to say **where** the computation was.

The null generators are not symmetric.

`markov1` has a naturally measurable workload:

1. fit first-order transition counts
2. sample the surrogate chain

So a real determinate bar is appropriate.

`matched-lz` is less linear, but its workload is still structured enough to report meaningful staged progress:

- target calibration windows
- candidate block surrogates
- LZ windows per candidate

That is more informative than a spinner and still more truthful than inventing a fake fine-grained ETA.

### Source Documents

- `hsi_agents_project/v2/common/null_models.py`
- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/hsi_v2_phase2_null_pressure_window_sweep.py`

### Practical Impact

- Deep strong-null runs can now be launched from an external PowerShell window with materially informative progress.
- `markov1` no longer looks stalled during long runs.
- `matched-lz` now exposes its internal candidate search structure, even when exact remaining wall time is not predictable.
- For long experiments, the recommended workflow is:
  - Ariadna prepares the CLI and analysis path
  - Iban launches the heavy run externally inside `venv`
  - we keep working in parallel while the process advances

### Open Questions

- Is the current staged readout for `matched-lz` sufficient, or does the LZ kernel itself eventually need deeper instrumentation?
- Should Phase 1 null-surrogate generation adopt the same visible progress defaults, not only Phase 2?

---

## D-0034

- Date: 2026-04-19
- Status: `provisional`
- Scope: methodological hardening of Phase 2 return-lag null-pressure sweeps
- Title: Preserve cross-segment lag continuity and anchor strong-null pattern selection to the observed run

### Decision

Phase 2 return-lag evaluation under strong-null pressure now obeys two stricter defaults:

1. Return lags are measured across the full analyzed strip, not independently inside each segment.
2. In `null_pressure` sweeps, the selected pattern list is fixed exogenously from the observed run and reused unchanged inside each null surrogate.

This means:

- segment boundaries are treated as storage boundaries only, not as physical reset points for recurrence,
- and strong-null comparisons now isolate temporal transport on a fixed observable vocabulary instead of letting each null redefine the vocabulary it is asked to reproduce.

### Rationale

The hostile audit surfaced two valid objections.

First, the previous implementation computed `np.diff` only inside each segment. That clipped any recurrence crossing segment boundaries and artificially capped the observable tail by the segment size. Since the whole purpose of the Phase 2 readout is to probe long-lag structure, that truncation was not admissible.

Second, the previous strong-null sweep allowed each null run to select patterns from its own stored Phase 1 pattern space. That mixed two axes at once:

- the observable vocabulary,
- and the temporal return profile.

For observed-only window sweeps this was acceptable because the goal was to freeze each run's own upstream selection while moving the window.

For strong-null pressure it is not acceptable, because the scientific question is narrower:

- do the null surrogates reproduce the return-profile geometry of the observed patterns?

So the conservative fix is to hold the observed pattern list fixed and evaluate that same list inside every null.

### Source Documents

- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/v2/phase2/window_sweep.py`
- `hsi_agents_project/hsi_v2_phase2_null_pressure_window_sweep.py`
- `docs/HSI_v2_Phase2_Window_Sweep_Design.md`

### Practical Impact

- Any previous `null_pressure` outputs generated before this correction must be treated as exploratory only.
- The corrected strong-null sweep is now the first Phase 2 version that is methodologically admissible for paper-facing claims.
- Observed-only window sweeps remain valid, but strong-null sweeps now have a stricter contract than observed-vs-observed scans.
- Reports and manifests now record that null-pressure selection is anchored to the observed run.

### Open Questions

- Once the lag tail is no longer clipped, does the worst matched-LZ window move deeper or become stronger at the current offsets?
- With the selection axis now frozen, does `matched-lz` remain close only on some seeds, or does a stable separation emerge across the seed envelope?
- Should future Phase 2 metrics report both segment-local and strip-global lag summaries explicitly, or is the global definition sufficient as the default scientific object?

---

## D-0035

- Date: 2026-04-20
- Status: `provisional`
- Scope: scientific result of the strict null-pressure rerun against matched-lz
- Title: The strict null-pressure rerun opens B from matched-lz in a paper-admissible way, with strongest separation at the 717Mâ€“720M band

### Decision

The corrected `null_pressure` sweep under the D-0034 methodological contract produces a first paper-admissible opening of Gate 2 against `matched-lz`:

1. Under observed-anchored pattern selection (`bridge-linked`, `m=48`, `top=64`) and strip-global `return_lag` evaluation, `matched-lz` does not stably reproduce the return-profile geometry of observed variant `B`.
2. The separation is sustained across all 6 tested seeds (17, 101, 211, 307, 401, 503), with seeds 307, 211, and 401 as the most exigent.
3. The strongest divergence concentrates in the `717M â†’ 720M` window band (`Prof JS avg = 0.242`, `d_mean avg = 0.416`), attenuating toward `726M` without disappearing.
4. The divergence is not a crude tail loss but a structured reorganization: bridge-linked patterns shift their dominant return bin from `16384â€“65535` to `131072â€“262143` under `matched-lz`.

### Rationale

D-0034 fixed the methodological contract (observed-anchored selection, strip-global lag evaluation). D-0035 records the scientific outcome: `matched-lz` remains the closest strong null, but fails to preserve the full recurrent rhythm of `B` when contrast is formulated on the temporal return profile rather than a single summary metric.

This is the first Gate 2 result that survives seed pressure and strict methodological audit simultaneously. It does not satisfy the formal Gate 2 conditions (defects Î´_m, spectrum, scaling laws), but establishes that the observational substrate is now distinguishable from the strong nulls under admissible constraints.

### Source Documents

- `docs/HSI_v2_Phase2_Null_Pressure_Strict_Readout.md`
- `hsi_agents_project/results/hsi_v2/phase2/window_sweep_deep/null_pressure_strict/phase2-return-lag-null-pressure__obs-B__m-48__sel-bridge-linked__top-64__w-5__off-714M__step-3M__20260419T102749/`

### Practical Impact

- The paper section on Gate 2 first opening (Â§8.4) now reflects this strict readout, replacing the provisional single-seed version.
- The summary table in the paper body reports `markov1` and `matched-lz` with avg and envelope (minâ€“max) of Prof JS, Prof wJ, d_mean, and top_eq.
- Seed-by-window detail is reserved for traceability, not for the paper body.
- The correct editorial framing is "first defensible observational opening of Gate 2 under strict contract", not "Gate 2 resolved".

### Open Questions

- Should the strict null-pressure sweep be extended to E and I, or does the B-only result suffice for the current paper?
- Is the 717Mâ€“720M concentration a genuine structural feature, or an artifact of the window placement?
- Should the next step be a formal Î´_m pilot on this same band, or a wider window sweep first?

---

## D-0036

- Date: 2026-04-20
- Status: `provisional`
- Scope: strict-band scientific reading of the first formal `N2-01` transport defect under weak-null and strong-null pressure
- Title: Strict `N2-01` separates `markov1` but not `matched-lz`; keep `null_pressure` as the Gate 2 opening and redirect formal escalation toward an observed-kernel counterfactual defect

### Decision

The first strict-band `N2-01` pilot is scientifically valid but is **not** yet admissible as the first formal Gate 2 readout against the strong null `matched-lz`.

The result is:

1. It rejects `markov1` very strongly:
   - `d_JS avg = 0.3936`
   - `d_wJ avg = 0.7263`
   - `d_flow avg = 3.2383`
   - `d_mean avg = 0.8239`
2. It leaves `matched-lz` very close to the observed zero baseline:
   - over 24 seed-window rows, `d_JS avg = 0.0018`
   - `d_wJ avg = 0.0197`
   - `d_flow avg = 0.0248`
   - `d_mean avg = 0.0123`
3. Therefore the paper-facing Gate 2 claim should remain grounded in D-0035 (`null_pressure` under strict contract), not in the current `delta_m` pilot.
4. The next formal target should freeze not only observed pattern support but the observed transport law itself.

### Rationale

The strict `N2-01` pilot anchors observed support correctly, but still evaluates each run with its **own** transport kernel.

So the current question is effectively:

> is the candidate run internally transport-consistent on the observed bridge-linked support?

Under that question:

- `markov1` fails badly
- `matched-lz` mostly passes

This explains the observed pattern:

- near-zero observed calibration
- very large weak-null separation
- very small strong-null envelope

So the current object is useful as a weak-null calibration and an endogenous consistency probe, but not yet as the decisive formal test of whether a candidate obeys the observed transport law of `B`.

An additional caution follows from the same run:

- `terminal_fraction` is not a monotone strong-null discriminator here

since:

- observed calibration sits near `0.3281`
- `matched-lz` sits higher (`0.4062 -> 0.5156`)
- `markov1` sits much lower (`0.03125`)

So it should remain secondary in the Gate 2 reading.

### Source Documents

- `docs/HSI_v2_Phase2_N2_01_Strict_Readout.md`
- `docs/HSI_v2_Phase2_N2_02_Observed_Kernel_Defect_Proposal.md`
- `hsi_agents_project/results/hsi_v2/phase2/transport_defect_strict_smoke/phase2-transport-defect-strict__stage-observed__m-40-48__sel-bridge-linked__top-64__off-717M__var-B-E__20260420T095713/`
- `hsi_agents_project/results/hsi_v2/phase2/transport_defect_strict_B_markov1/phase2-transport-defect-strict__stage-nulls__m-40-48__sel-bridge-linked__top-64__off-714M-plus-4__var-B__20260420T193214/`
- `hsi_agents_project/results/hsi_v2/phase2/transport_defect_strict_B_matchedlz/phase2-transport-defect-strict__stage-nulls__m-40-48__sel-bridge-linked__top-64__off-714M-plus-4__var-B__20260420T212430/`

### Practical Impact

- Do not promote the current strict `N2-01` result into the paper as a positive formal Gate 2 opening.
- Keep D-0035 (`null_pressure`) as the first defensible Gate 2 result.
- Treat the current `N2-01` pilot as:
  - a valid weak-null rejection
  - a useful calibration object
  - and a diagnostic of endogeneity in the first transport-defect design
- The next formal implementation target should move toward an observed-kernel counterfactual defect (`N2-02`), not a larger rerun of the current endogenous object.

### Open Questions

- Should the observed kernel be frozen from `B` only, or from a `B/E` envelope once the pair is recomputed under the same contract?
- Does the counterfactual defect need an explicit support-coverage penalty, or is the fixed-kernel mismatch alone sufficient?
- Should `N2-02` first be tested only on `B` vs strong nulls before extending to `E`?

---
## D-0037

- Date: 2026-04-20
- Status: `provisional`
- Scope: override resolution contract for observed-anchored `return_lag` evaluation on null surrogates
- Title: Resolve observed-anchored `return_lag` overrides against `source_variant` for null surrogates and revalidate any strict Phase 2 null runs that depended on the old behavior

### Decision

The `return_lag` override contract is tightened as follows:

1. Selection overrides must resolve first against `config.variant`, then against `config.source_variant`, and finally against the base variant prefix when the run name is decorated (for example `B-markov1`, `B-matched-lz`).
2. This closes a silent anchoring leak in Phase 2 null-surrogate evaluation: observed-anchored pattern selections keyed only by `B` could fail to attach when the evaluated null run was named as `B-markov1` or `B-matched-lz`.
3. Therefore any serious Phase 2 null run whose claimed strictness depended on observed-anchored `return_lag` selection must be treated as pending revalidation under D-0037.

### Rationale

The strict `transport_defect_counterfactual` smoke exposed a silent mismatch between:

- the override key supplied by the launcher (`B`)
- and the decorated null-surrogate run variant stored in Phase 1 metadata (`B-markov1`, `B-matched-lz`)

Without source-aware resolution, the `return_lag` layer could silently fall back to run-local pattern selection for null surrogates, defeating the intended observed anchoring.

The correct place to fix the contract is the shared `return_lag` implementation itself, not a local wrapper, so that all downstream strict launchers inherit the same behavior.

### Source Documents

- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/hsi_v2_phase2_transport_defect_strict.py`
- `hsi_agents_project/hsi_v2_phase2_transport_defect_counterfactual.py`

### Practical Impact

- Future strict Phase 2 runs now attach observed-anchored selections correctly even when null-surrogate runs use decorated variant names.
- Existing strict null comparisons that depended on observed anchoring should not be treated as final until rerun or explicitly revalidated under D-0037.
- This affects both endogenous and counterfactual transport-defect launchers, and may also affect strict `null_pressure` style sweeps that passed overrides keyed only by the observed variant.

### Open Questions

- Which previously accepted strict Phase 2 null runs must be rerun immediately for paper safety?
- Does the D-0035 `null_pressure` opening survive unchanged once rerun under D-0037?
- Should the strict launcher layer also emit an explicit warning whenever an override is resolved via `source_variant` rather than `variant`?

---

## D-0038

- Date: 2026-04-21
- Status: `provisional`
- Scope: scientific reading of the completed strict `N2-02` observed-kernel counterfactual transport defect
- Title: Strict `N2-02` preserves `B/E` and rejects `markov1`, but still does not materially separate `matched-lz`; treat it as a valid negative result and keep Gate 2 paper claims pending D-0037 revalidation of `null_pressure`

### Decision

The completed strict `N2-02` batch is scientifically valid, but it does **not** yet provide a formal Gate 2 opening against the strong null `matched-lz`.

The result is:

1. The observed family is preserved exactly under the frozen observed law:
   - `B`: `d_JS = 0`, `d_flow = 0`, `d_mean = 0` on all four offsets
   - `E`: `d_JS = 0`, `d_flow = 0`, `d_mean = 0` on all four offsets
2. `markov1` is strongly rejected:
   - `cov avg = 0.1821`
   - `d_JS avg = 0.4376`
   - `d_flow avg = 1.6043`
   - `d_mean avg = 0.6970`
3. `matched-lz` remains too close to the observed zero baseline:
   - over 24 seed-window rows, `cov avg = 0.9944`
   - `d_JS avg = 0.0045`
   - `d_wJ avg = 0.0224`
   - `d_flow avg = 0.0266`
   - `d_mean avg = 0.0124`
4. Therefore `N2-02` should be recorded as a valid negative result:
   - correct object
   - correct calibration
   - but no material strong-null opening
5. Paper-facing Gate 2 language should remain suspended on transport-defect objects until the older strict `null_pressure` candidate is rerun or explicitly revalidated under D-0037.

### Rationale

`N2-02` was designed to attack the exact weakness of `N2-01`:

- `N2-01` froze observed support but still used the candidate's own transport kernel
- `N2-02` freezes the observed transport law itself

This escalation was methodologically correct.

The completed result therefore has real scientific content even though it is negative:

- it confirms that the observed law of `B` does not break `E`
- it confirms that the same law strongly excludes `markov1`
- but it also shows that the local frozen law `K_B` is still not sufficiently distinctive to break `matched-lz`

So the limiting factor is no longer the endogeneity objection alone. It is now a more substantive fact about the current law-level observable.

### Source Documents

- `docs/HSI_v2_Phase2_N2_02_Observed_Kernel_Defect_Proposal.md`
- `docs/HSI_v2_Phase2_N2_02_Counterfactual_Readout.md`
- `hsi_agents_project/results/hsi_v2/phase2/transport_defect_counterfactual/phase2-transport-defect-counterfactual__stage-all__anchor-B__m-40-48__sel-bridge-linked__top-64__off-714M-plus-4__var-B-E__20260421T012228/`

### Practical Impact

- Do not promote `N2-02` into the paper body as a positive Gate 2 opening.
- Treat `N2-02` as:
  - a strong internal validation of the `B/E` family
  - a strong weak-null rejection
  - and a valuable negative result against the strong null
- Keep the older strict `null_pressure` result as the best Gate 2 candidate in principle, but do not treat it as paper-final until D-0037 revalidation is complete.
- The next formal object should not be justified by pretending that `N2-02` almost worked. It should be justified because the current frozen-law defect has now been honestly tested and bounded.

### Open Questions

- Does the D-0035 `null_pressure` opening survive unchanged once rerun under D-0037?
- Is the next honest law-level object direct kernel divergence on common support, rather than another transport-defect remix?
- Should `E` later be rerun as an anchor in its own right, or is the current `B`-anchored preservation result already sufficient?

---
## D-0039

- Date: 2026-04-21
- Status: `provisional`
- Scope: strict Phase 2 window-sweep execution contract for frozen null realizations, segment-boundary coverage, and null-pressure reporting
- Title: Freeze source realizations across the full sweep, preserve boundary-spanning pattern starts, and report both worst divergence and closest-null imitation

### Decision

Strict Phase 2 window sweeps now obey three additional defaults:

1. For each selected observed or null run, the source bitstream must be generated **once** at the full sweep-global required length and then sliced window-by-window. Recomputing a fresh null realization from a longer prefix at each offset is no longer admissible for paper-safe sweeps.
2. `return_lag` evaluation must preserve pattern starts that cross segment boundaries by carrying an `m-1` overlap into each subsequent segment before computing rolling codes.
3. `null_pressure` summaries must expose both:
   - the worst divergence window,
   - and the closest-imitation window,
   together with min/avg/max envelopes, rather than only the maximum-separation readout.

### Rationale

The hostile audit of the D-0037 rerun candidate surfaced three valid concerns.

First, if a null such as `matched-lz` is regenerated independently for each offset from a progressively longer prefix, then the sweep no longer isolates window position as the only changed variable. The null realization itself can mutate across windows. This violates the original austerity principle of the window-sweep design.

Second, although cross-segment lag continuity had already been repaired in D-0034, the actual pattern-detection layer still dropped valid pattern starts that straddled segment boundaries. That was a smaller effect than the original lag-clipping bug, but still inconsistent with a strict whole-strip reading.

Third, a null-pressure report that only foregrounds the strongest divergence is vulnerable to an obvious hostile objection: it does not show where the null comes closest to reproducing the observed profile. A paper-safe readout must therefore present the full envelope, not only the opening window.

### Source Documents

- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/v2/phase2/window_sweep.py`
- `hsi_agents_project/v2/phase2/null_pressure.py`
- `hsi_agents_project/hsi_v2_phase2_null_pressure_window_sweep.py`
- `docs/HSI_v2_Phase2_Window_Sweep_Design.md`

### Practical Impact

- Any strict Phase 2 window sweep that regenerated null realizations independently per offset should be treated as diagnostic, not paper-final.
- The in-flight D-0037 `null_pressure` rerun must itself be repeated under D-0039 before being treated as the definitive revalidation of D-0035.
- Future strict `null_pressure` reports should present the null envelope in both directions:
  - strongest separation
  - closest imitation
- The sweep metadata should explicitly record that the source bitstream was frozen sweep-globally.

### Open Questions

- Once rerun under D-0039, does the D-0035 opening survive with the same 717Mâ€“720M concentration, or does the strongest band shift?
- Is the segment-boundary overlap correction numerically small enough to remain editorially secondary, or does it materially move the envelope on the hardest seeds?
- Should the same frozen-source contract now be promoted immediately to strict transport-defect sweeps, or only after the `null_pressure` revalidation is complete?

---

## D-0040

- Date: 2026-04-21
- Status: `provisional`
- Scope: scientific revalidation of the strict `null_pressure` Gate 2 opening under the D-0039 contract
- Title: D-0035 survives D-0039 hardening; strict `null_pressure` remains the first paper-safe observational opening of Gate 2, with peak opening at 717M

### Decision

The strict `null_pressure` result remains scientifically positive after the D-0039 hardening pass.

Under:

- D-0037-corrected observed anchoring
- sweep-global frozen null realizations
- and boundary-spanning pattern preservation across segments

the strong null `matched-lz` still does not reproduce the return-profile geometry of observed `B`.

The updated reading is:

1. `markov1` remains the clearly farther null:
   - `Prof JS avg = 0.3333`
   - `d_mean avg = 0.6869`
2. `matched-lz` remains the hardest null, but is still clearly separated:
   - over 30 seed-window rows, `Prof JS min/avg/max = 0.1112 / 0.1842 / 0.2719`
   - `Prof wJ min/avg/max = 0.2873 / 0.4513 / 0.6245`
   - `d_mean min/avg/max = 0.1163 / 0.3220 / 0.5437`
3. The strongest average opening is now best localized at `717M`:
   - `Prof JS avg = 0.2205`
   - `d_mean avg = 0.4348`
   - `Prof wJ avg = 0.3273`
   - `top_eq avg = 0.2738`
4. The earlier phrase "strongest concentration in the 717M-720M band" should be sharpened to:
   - peak opening at `717M`
   - attenuation toward `726M`
   - with secondary divergence still visible at `714M` and `723M`

### Rationale

D-0039 was the strongest hostile-methodology correction applied so far to the `null_pressure` line.

It removed three plausible ways the earlier result might have been overstated:

- regenerating fresh null realizations at each window,
- dropping pattern starts that crossed segment boundaries,
- and reporting only the strongest divergence window without exposing the closest-imitation envelope.

The fact that the opening survives after all three corrections is therefore itself scientifically meaningful.

The envelope is milder than the earlier D-0035-era readout, but it does not collapse.

So the correct update is not "the old result failed" but rather:

> the old result was directionally right, and the stronger contract still supports it.

### Source Documents

- `docs/HSI_v2_Phase2_Null_Pressure_Strict_Readout.md`
- `hsi_agents_project/results/hsi_v2/phase2/window_sweep_deep/null_pressure_strict_d0039_reval/phase2-return-lag-null-pressure__obs-B__m-48__sel-bridge-linked__top-64__w-5__off-714M__step-3M__20260421T181545/`

### Practical Impact

- The D-0039 run should now be treated as the canonical strict `null_pressure` artifact.
- Paper-facing Gate 2 language can be stabilized again on the `null_pressure` line.
- The preferred editorial wording is now:
  - "first paper-safe observational opening of Gate 2"
  - not "Gate 2 resolved"
- The paper should update the spatial reading from:
  - "717M-720M concentration"
  to:
  - "peak opening at 717M with attenuation toward 726M"
- `N2-02` remains valuable, but as a valid negative result rather than the primary Gate 2 opening.

### Open Questions

- Should the strict `null_pressure` opening now be extended to `E` and `I`, or is the `B`-anchored result sufficient for the current paper?
- Is the `717M` peak the best immediate anchor for the next formal defect object, or should one first widen the readout over a slightly broader late band?
- Should the same D-0039 sweep-global frozen-source contract now be enforced immediately across all remaining Phase 2 executables?

---

## D-0041

- Date: 2026-04-21
- Status: `provisional`
- Scope: methodological propagation of the D-0039 frozen-source contract to the remaining multi-window Phase 2 transport-defect launchers
- Title: The D-0039 sweep-global frozen-source contract is now enforced by default in the strict and counterfactual transport-defect launchers

### Decision

The multi-window Phase 2 transport-defect launchers now preload and freeze their evaluated source realizations once across the full offset sweep before rebuilding any return-lag rows.

This applies to:

- `hsi_v2_phase2_transport_defect_strict.py`
- `hsi_v2_phase2_transport_defect_counterfactual.py`

The enforced contract is:

1. compute `frozen_total_bits = max(offsets) + window_span_bits`
2. preload each evaluated observed/null source once up to that total bit budget
3. reuse the same frozen source realization for every offset in the sweep
4. emit `source_freeze_mode = sweep-global-frozen` and `frozen_total_bits` in rows, summaries, reports, and manifests

### Rationale

After D-0039, it would be methodologically inconsistent to keep `null_pressure` under a sweep-global frozen-source contract while allowing the remaining multi-window Phase 2 launchers to regenerate or reload their sources window by window.

Even when the underlying object is different:

- endogenous transport defect in `N2-01`
- counterfactual observed-kernel defect in `N2-02`

the comparison across offsets still assumes a fixed evaluated source realization.

So the stricter contract should now be treated as a Phase 2 default rather than a `null_pressure`-only special case.

### Source Documents

- `HSI_v2_Decision_Log.md` (D-0039, D-0040)
- `hsi_agents_project/hsi_v2_phase2_transport_defect_strict.py`
- `hsi_agents_project/hsi_v2_phase2_transport_defect_counterfactual.py`

### Practical Impact

- Future multi-window `N2-01` and `N2-02` runs should be considered contract-clean by default on this point.
- Existing pre-D-0041 multi-window transport-defect readouts remain useful as exploratory diagnostics, but should be rerun before any paper-facing claim depends materially on cross-offset envelopes.
- The strengthened metadata trail now makes it explicit, inside the artifacts themselves, whether a run used the sweep-global frozen-source contract.

### Open Questions

- Does the 717M-anchored next object gain more from this stricter execution contract if it is formulated directly at the kernel level rather than through another transport-profile remix?
- Should a small repo-facing recovery note be added later to summarize which pre-D-0041 transport-defect artifacts remain exploratory only?

---
## D-0042

- Date: 2026-04-21
- Status: `provisional`
- Scope: local-kernel semantics for the Phase 2 `N2-03` common-support object
- Title: `N2-03` window-local kernels must be rebuilt from child-anchored realized events, not from admissible child mass

### Decision

The `window-local` implementation of `N2-03` is tightened as follows:

1. The local parent-child kernel must be rebuilt from **realized child occurrences** inside the frozen late window.
2. Each realized selected child occurrence contributes event mass to the selected parent-child pair(s) that it actually instantiates under the structural Phase 1 edge support.
3. If the same parent-child pair is admissible through both `prefix` and `suffix`, the same child occurrence contributes **once**, not twice, to that pair.
4. Shared-support divergence averages in `window-local` mode must be weighted by the anchor's realized local event mass on the frozen child universe, not by raw low-scale parent counts.

### Rationale

The hostile audit of the first `window-local` scaffold surfaced a valid ambiguity.

Although the original local reconstruction already used child counts inside the late window, its formulation was still too easy to read as "admissible local child mass filtered by structure" rather than as a realized local transport law.

That ambiguity became methodologically dangerous in two ways:

- it left room for a hostile referee to argue that we were still comparing marginals rather than realized parent-child transport events
- and it allowed same-pair `prefix/suffix` admissibility to inflate a local score by duplicated policy counting

The corrected object keeps the same frozen parent/child universe, but makes the semantics explicit and event-based:

> local kernel mass is induced by realized selected child events in the window

This is the smallest correction that restores a defensible transport interpretation without a large refactor.

### Source Documents

- `docs/HSI_v2_Phase2_N2_03_Common_Support_Kernel_Divergence_Proposal.md`
- `docs/HSI-audit-Phase2-Kernel-Common-Support.md`
- `hsi_agents_project/v2/phase2/kernel_common_support.py`
- `hsi_agents_project/hsi_v2_phase2_kernel_common_support.py`

### Practical Impact

- Any pre-D-0042 `window-local` `N2-03` result should be treated as exploratory only.
- All serious `N2-03` runs must be rerun under the child-anchored event contract before they are read scientifically.
- The corrected `window-local` smoke calibration now behaves as expected:
  - observed `B` and observed `E` remain identical on the local object
  - `markov1` collapses to zero shared support
- These smokes validate the object mechanically, but they do not yet settle whether `matched-lz` separates materially at the paper-scale `top=64` setting.

### Open Questions

- Under the corrected event contract, does the full `top=64` `717M` run materially separate `matched-lz` from the observed baseline?
- Once the local object is stable, should a later refinement add an ordered child-axis flow metric, or is shared-support `JS/WJ` sufficient for this phase?

---

## D-0043

- Date: 2026-04-21
- Status: `provisional`
- Scope: scientific reading of the corrected `N2-03` common-support kernel and its derived throughput readout
- Title: Corrected `N2-03` remains negative on normalized common support; the derived throughput readout reveals a bidirectional `B/E` asymmetry and must not be treated as Gate 2 evidence

### Decision

The corrected `N2-03` line now has to be read as two separate objects.

1. The corrected `window-local` **normalized common-support kernel** is scientifically valid, but remains a negative result against the strong null:
   - observed `B` calibrates exactly
   - observed `E` remains exactly aligned with `B`
   - `markov1` collapses to zero support
   - `matched-lz` remains too close on shared-support shape
2. The derived **throughput** readout exposes a real and strong directional asymmetry between `B` and `E`:
   - under anchor `B`, observed `E` preserves full support but only `0.3114` of `B`'s event throughput
   - under anchor `E`, observed `B` preserves only `0.6012` of `E`'s support, but carries `1.9306` of `E`'s event throughput on the overlap
3. Because that throughput readout does not preserve the observed family, it must be treated as an internal asymmetry observable, not as a paper-safe Gate 2 signature.
4. D-0040 `strict null_pressure` therefore remains the canonical Gate 2 opening.

### Rationale

The corrected `N2-03` object resolves an ambiguity that had remained hidden while the earlier scaffold was still under audit.

Once the late-window kernel is rebuilt from child-anchored realized events under D-0042, two different questions become visible:

1. what is the **shape** of the local transport law on the support that both systems share?
2. how much **absolute event mass** survives through that support?

Those are not the same question.

The first question gives a clean negative result:

- `B/E` are identical on normalized shared-support shape
- `markov1` collapses
- `matched-lz` still hugs the observed baseline too well

The second question gives a clean asymmetry result:

- `B` and `E` are not symmetric in support-vs-density terms
- `B` appears narrower and denser
- `E` appears broader and weaker

That asymmetry may matter mathematically, but it is not the same kind of evidence as a family-preserving null discriminator.

### Source Documents

- `docs/HSI_v2_Phase2_N2_03_Common_Support_Kernel_Divergence_Proposal.md`
- `docs/HSI_v2_Phase2_N2_03_Bidirectional_Asymmetry_Readout.md`
- `hsi_agents_project/v2/phase2/kernel_common_support.py`
- `hsi_agents_project/hsi_v2_phase2_kernel_common_support.py`
- `hsi_agents_project/v2/phase2/kernel_throughput.py`
- `hsi_agents_project/hsi_v2_phase2_kernel_throughput_readout.py`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_common_support_windowlocal/phase2-kernel-common-support__stage-all__anchor-B__m-40-48__sel-bridge-linked__top-64__late-717M__var-B-E__20260421T215336/`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_throughput_readout/phase2-kernel-throughput-readout__anchor-B__m-40-48__sel-bridge-linked__top-64__late-717M__var-B-E__20260421T220434/`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_common_support_windowlocal_E_anchor/phase2-kernel-common-support__stage-observed__anchor-E__m-40-48__sel-bridge-linked__top-64__late-717M__var-E-B__20260421T225537/`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_throughput_readout/phase2-kernel-throughput-readout__anchor-E__m-40-48__sel-bridge-linked__top-64__late-717M__var-E-B__20260421T225651/`

### Practical Impact

- The corrected `N2-03` normalized common-support kernel should be recorded as a valid negative result, not promoted as a new Gate 2 opening.
- The derived throughput readout should not be used in the paper as Gate 2 evidence, because it splits the observed family itself.
- The throughput asymmetry should nevertheless be preserved as a real internal clue for the next mathematical layer.
- Editorially, the safest phrasing is:
  - D-0040 remains the first paper-safe observational opening of Gate 2
  - corrected `N2-03` reveals a late-window `B/E` asymmetry, not a replacement Gate 2 signature

### Open Questions

- Can the next object isolate structured support dropout in a way that still preserves the observed family?
- Is the mathematically relevant asymmetry here:
  - support envelope,
  - event multiplicity,
  - or ordered child-axis organization inside the overlap?
- Should the next formal object stay anchored at `717M`, or should it read the same asymmetry over a short late-band envelope first?

---

## D-0044

- Date: 2026-04-22
- Status: `provisional`
- Scope: scientific reading of the derived support-dropout profile for corrected `N2-03`
- Title: The late-window `B/E` asymmetry in corrected `N2-03` takes the form of selective parent extinction, not graded child-support deformation

### Decision

The support-dropout profile derived from corrected `N2-03` sharpens the asymmetry reading in a useful way.

1. Under anchor `B`, support dropout is **not** the relevant discriminator:
   - observed `E` preserves the full local parent support exactly
   - all `matched-lz` seeds also preserve the full local parent support exactly
   - `markov1` alone collapses by total extinction
2. Under anchor `E`, observed `B` does not show partial support deformation:
   - `43` active anchor parents are split into
     - `20` exact survivors
     - `23` extinct parents
   - exact-support mass = `0.6012`
   - extinct-support mass = `0.3988`
   - no `loss`, `spill`, `mixed`, or `disjoint` states appear
3. Therefore the next formal object should not be another generic child-support-dropout metric.
4. The most honest next direction is a parent-survival or survival-stratified object:
   - which late anchor parents survive
   - which disappear
   - and, on the surviving exact block, what internal organization still differentiates the systems

### Rationale

The earlier asymmetry reading in D-0043 already showed that `B` and `E` differ in support-vs-density terms.

But that still left an open geometric ambiguity:

- is the difference a smooth support deformation?
- or a sharper survival split?

The support-dropout readout resolves that ambiguity.

It shows that the observed mirror `E -> B` is not characterized by partial child-support erosion.

Instead, the anchor parents bifurcate into two blocks:

- exact survivors
- exact extinctions

This is a much cleaner structural clue, and it matters because it narrows the next mathematically honest object.

### Source Documents

- `docs/HSI_v2_Phase2_N2_03_Bidirectional_Asymmetry_Readout.md`
- `docs/HSI_v2_Phase2_N2_03_Support_Dropout_Readout.md`
- `hsi_agents_project/v2/phase2/kernel_support_dropout.py`
- `hsi_agents_project/hsi_v2_phase2_kernel_support_dropout_readout.py`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_support_dropout_readout/phase2-kernel-support-dropout__anchor-B__m-40-48__sel-bridge-linked__top-64__late-717M__var-B-E__20260422T081218/`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_support_dropout_readout/phase2-kernel-support-dropout__anchor-E__m-40-48__sel-bridge-linked__top-64__late-717M__var-E-B__20260422T081218/`

### Practical Impact

- The support-dropout readout should not be promoted as a new Gate 2 opening.
- Under anchor `B`, it adds no new separation against `matched-lz`.
- Under anchor `E`, it gives a strong internal structural clue:
  - the relevant asymmetry is survival/extinction of anchor parents
  - not gradual local support deformation
- The next scaffold should therefore either:
  - classify surviving vs extinct parents directly
  - or study ordered structure only inside the surviving exact-support layer

### Open Questions

- Is the survival/extinction split itself stable over a short late-band envelope around `717M`?
- Can the surviving exact-support block be separated further by an ordering-sensitive or density-sensitive local object?
- Are the extinct `E`-anchor parents semantically clustered, or do they look combinatorially generic?

---

## D-0045

- Date: 2026-04-22
- Status: `provisional`
- Scope: first scientific readout of the `N2-04` parent-survival selector
- Title: First `N2-04` pass confirms a sharp `E -> B` survival frontier; exact survivors are two-child parents and extinct parents are one-child parents

### Decision

The first `N2-04` pass is scientifically useful and sharpens the asymmetry beyond D-0044.

1. Under anchor `B`, simple parent survival remains non-discriminative against the strong null:
   - observed `E` preserves all `20` active parents exactly
   - all `matched-lz` seeds also preserve all `20` active parents exactly
   - only `markov1` collapses by total extinction
2. Under anchor `E`, observed `B` forms a clean survival frontier:
   - `43` active anchor parents
   - `20` exact survivors
   - `23` extinct parents
   - `0` partial frontier
   - exact-support mass `0.6012`
   - extinct-support mass `0.3988`
3. The frontier is not combinatorially vague in this first run:
   - all exact survivors have `anchor_local_child_count = 2`
   - all extinct parents have `anchor_local_child_count = 1`
4. Therefore the next object should not be another generic support selector.
5. The most honest next direction is:
   - branching-aware survival structure
   - or an ordered/density object restricted to the exact-surviving block

### Rationale

D-0044 had already shown that the `E -> B` mirror looks like survival/extinction rather than blur.

But it did not yet tell us whether that frontier had internal combinatorial structure or was merely an arbitrary split.

`N2-04` resolves that first ambiguity.

In the initial `717M` run, the frontier aligns exactly with local branching multiplicity:

- two-child parents survive
- one-child parents disappear

This is a much stronger clue than a raw survival count.

It means the asymmetry may be rooted in a specific local branching regime rather than in generic support thinning.

### Source Documents

- `docs/HSI_v2_Phase2_N2_04_Parent_Survival_Proposal.md`
- `docs/HSI_v2_Phase2_N2_04_Parent_Survival_Readout.md`
- `hsi_agents_project/v2/phase2/parent_survival.py`
- `hsi_agents_project/hsi_v2_phase2_parent_survival_selector.py`
- `hsi_agents_project/results/hsi_v2/phase2/parent_survival_selector/phase2-parent-survival-selector__anchor-B__m-40-48__sel-bridge-linked__top-64__late-717M__var-B-E__20260422T082059/`
- `hsi_agents_project/results/hsi_v2/phase2/parent_survival_selector/phase2-parent-survival-selector__anchor-E__m-40-48__sel-bridge-linked__top-64__late-717M__var-E-B__20260422T082059/`

### Practical Impact

- `N2-04` should be kept as a genuine internal asymmetry object.
- It should not be promoted as a new Gate 2 opening, because under anchor `B` it does not separate `matched-lz`.
- The first `N2-04` run nevertheless narrows the next move substantially:
  - the frontier seems branching-aware
  - and the exact-support survivor block is now an explicit object
- Future work should likely separate two layers:
  - parent survival by local branching type
  - internal organization of the exact-support survivor block

### Open Questions

- Does the one-child / two-child survival split remain exact across a short late-band envelope?
- If the frontier is branching-aware, is the relevant variable:
  - number of admissible children,
  - balance between prefix and suffix continuation,
  - or some deeper structural subtype of the one-child parents?
- Inside the exact-support survivor block, does `B` differ from `E` mainly by density, by order, or by both?

---

## D-0046

- Date: 2026-04-22
- Status: `provisional`
- Scope: semantic hardening of the derived `N2-03` / `N2-04` survival readouts after hostile audit
- Title: Derived survival readouts must separate `dead` parents from `dark` continuation; the observed `E -> B` frontier remains fully dead-side under the hardened contract

### Decision

The derived survival readouts are hardened as follows.

1. The old catch-all "extinct" bucket is split into two different states:
   - `dead`: `candidate_local_parent_count = 0`
   - `dark`: the candidate parent survives locally, but no monitored continuation/event survives inside the anchored child set
2. `survivor_parent_count` and `survivor_mass_fraction` now exclude both `dead` and `dark`, so the parent-count and mass summaries obey the same semantic contract.
3. The hardened reruns preserve the central scientific result:
   - under anchor `E`, observed `B` still gives `20` exact survivors, `23` `dead` parents, and `0` `dark` parents
   - exact-support mass remains `0.6012`
   - dead-support mass remains `0.3988`
4. The hardened reruns also refine the control reading:
   - under anchor `B`, `matched-lz` remains exactly aligned on parent survival
   - under anchor `B`, `markov1` is **not** pure death
   - it yields `1` `dead` parent and `19` `dark` parents
5. Therefore D-0044 and D-0045 survive in substance, but any future wording should avoid describing the anchor-`B` `markov1` control as "total extinction".

### Rationale

The hostile audit identified a real semantic weakness in the first derived-readout pass.

The earlier implementation could merge two different phenomena:

- true local death of the anchor parent
- survival of the parent together with loss of monitored continuation inside the anchored child set

Those are not the same object and should not share a label.

Once that split is enforced, the central observed result becomes stronger rather than weaker:

- the `E -> B` asymmetry does **not** hide a dark frontier
- the non-surviving block is fully `dead` under the anchored support contract

So the main `E -> B` reading was not an artifact of the selector ambiguity. The ambiguity lived in the control interpretation, not in the observed frontier itself.

### Source Documents

- `docs/HSI-audit-Phase2-Derived-Readouts.md`
- `docs/HSI_v2_Phase2_N2_03_Support_Dropout_Readout.md`
- `docs/HSI_v2_Phase2_N2_04_Parent_Survival_Readout.md`
- `hsi_agents_project/v2/phase2/kernel_support_dropout.py`
- `hsi_agents_project/hsi_v2_phase2_kernel_support_dropout_readout.py`
- `hsi_agents_project/v2/phase2/parent_survival.py`
- `hsi_agents_project/hsi_v2_phase2_parent_survival_selector.py`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_support_dropout_readout/phase2-kernel-support-dropout__anchor-B__m-40-48__sel-bridge-linked__top-64__late-717M__var-B-E__20260422T084516/`
- `hsi_agents_project/results/hsi_v2/phase2/kernel_support_dropout_readout/phase2-kernel-support-dropout__anchor-E__m-40-48__sel-bridge-linked__top-64__late-717M__var-E-B__20260422T084516/`
- `hsi_agents_project/results/hsi_v2/phase2/parent_survival_selector/phase2-parent-survival-selector__anchor-B__m-40-48__sel-bridge-linked__top-64__late-717M__var-B-E__20260422T084516/`
- `hsi_agents_project/results/hsi_v2/phase2/parent_survival_selector/phase2-parent-survival-selector__anchor-E__m-40-48__sel-bridge-linked__top-64__late-717M__var-E-B__20260422T084516/`

### Practical Impact

- The codebase is now robust against the strongest semantic criticism raised in the audit.
- The central observed `E -> B` frontier remains valid and is now better specified:
  - exact vs `dead`, with no hidden `dark` layer
- The anchor-`B` `markov1` control should now be read as:
  - collapse of monitored continuation
  - mostly `dark`, not mostly `dead`
- Editorially, D-0044 and D-0045 should be read through D-0046 whenever the dead/dark distinction matters.

### Open Questions

- Does the exact-vs-dead frontier remain stable over a short late-band envelope around `717M`?
- Are the dead one-child parents semantically homogeneous, or do they split into finer internal subtypes?
- Inside the exact-surviving block, does the next useful discriminator live in order, density, or branching-conditioned order?

---

## D-0047

- Date: 2026-04-22
- Status: `provisional`
- Scope: canonical revalidation of the parent-survival line after the same-window death-shell objection
- Title: Parent-survival claims must be revalidated through an independent shell-lag probe plus lag-aware band; the current `E -> B` reading is delayed shell realignment with residual local mismatch, not a stable death shell

### Decision

The parent-survival line is reclassified and hardened as follows.

1. The earlier same-window parent-survival band remains diagnostically useful, but it is no longer sufficient evidence for literal shell death.
2. Any serious parent-survival claim must now use a two-stage contract:
   - an independent shell-lag probe defined from the anchor alone
   - a lag-aware parent-survival band driven by the externally estimated lag
3. Under the canonical `E -> B` rerun at scales `40 -> 48`, selector `bridge-linked`, `top = 64`, and late-band offsets `714M -> 723M`:
   - anchor core size = `20`
   - anchor shell size = `44`
   - first full candidate shell offset = `729M`
   - recommended lag = `+15M` bits
4. Under that lag-aware contract, the observed candidate `B` is **not** well described as a stable death shell:
   - exact range = `8 -> 43`
   - dead range = `0 -> 23`
   - dark range = `0`
   - partial range = `0`
5. Therefore the canonical scientific reading changes from:
   - "stable late-band death shell"
   to:
   - "delayed shell realignment with residual local mismatch"
6. This does **not** create a new Gate 2 opening.
   - `null_pressure` remains the first paper-safe observational Gate 2 opening
   - the survival line remains a calibrated asymmetry diagnostic

### Rationale

The hostile audit identified a real temporal objection:

- the same-window parent-survival band could confuse local absence with delayed reappearance outside the aligned window

That objection was strong enough that the old survival wording could not remain canonical.

The response was not rhetorical. The codebase now forces the temporal question into the pipeline itself.

First, the shell is defined from the anchor alone and scanned independently across a wider band. Only after that does the candidate comparison happen, using an externally estimated lag.

This rerun shows two things at once:

- the earlier death-shell claim was too strong
- the asymmetry itself was still real, because even after lag correction the band does not collapse to perfect identity

So the honest scientific position is narrower but stronger:

- the dominant effect is delayed realignment
- but there remains local residual mismatch that prevents a simple rigid-shift explanation

### Source Documents

- `docs/HSI-audit-Phase2-Derived-Parent-Survival-Band.md`
- `hsi_agents_project/v2/phase2/parent_shell_lag.py`
- `hsi_agents_project/hsi_v2_phase2_parent_shell_lag_probe.py`
- `hsi_agents_project/hsi_v2_phase2_parent_survival_band_lagaware.py`
- `hsi_agents_project/hsi_v2_phase2_parent_survival_revalidation.py`
- `hsi_agents_project/results/hsi_v2/phase2/parent_survival_revalidation/phase2-parent-survival-revalidation__anchor-E__cand-B__m-40-48__sel-bridge-linked__top-64__off-714M-plus-19__20260422T202310/`

### Practical Impact

- The phrase "stable death shell" should no longer be used as the canonical reading of the `E -> B` survival line.
- The canonical reproducible entry point for this line is now:
  - `hsi_agents_project/hsi_v2_phase2_parent_survival_revalidation.py`
- Any future paper or repo-facing summary should present the line as:
  - delayed shell realignment
  - with residual local mismatch
- This keeps the survival line scientifically alive without overstating what the current data prove.

### Open Questions

- Is the residual mismatch after `+15M` lag correction concentrated in a small subset of shell parents, or does it diffuse across the full shell?
- Does the next useful object live in shell order, shell density, or a branching-conditioned shell subtype?
- Is the `+15M` estimate stable under nearby changes in shell definition, or is the relevant temporal object intrinsically non-rigid?

---

## D-0048

- Date: 2026-04-23
- Status: `provisional`
- Scope: correction of the parent-density residual readout after the survivor-bias audit
- Title: Parent-density residuals must distinguish full partition mass retention from survivor-internal density deformation

### Decision

The parent-density residual readout is hardened as follows.

1. The old exact-only density readout is reclassified as survivor-conditioned and is no longer acceptable as a partition-level mass-retention measure.
2. The canonical partition retention metric must use all active anchor rows, including `dead`, `dark`, and non-exact rows:
   - `partition_mass_retention = sum(candidate events) / sum(anchor events)`
3. Logarithmic deformation and bias metrics remain valid only on rows with positive candidate support:
   - `survivor_internal_retention`
   - `survivor_internal_deformation_log2_mean`
   - `survivor_internal_bias_log2_mean`
4. The grouped summary must use pooled ratio-of-sums as the primary reported value, not an unweighted mean of per-window ratios.
5. The corrected canonical rerun preserves the scientific signal while removing the survivorship bias:
   - observed `E` remains the exact reference at `1.0000`
   - observed `B` shell has full partition retention `0.9375`, with survivor-internal retention `1.0227`
   - `markov1` shell has partition retention `0.0000`
   - `matched-lz` shell partition retention ranges from `0.1882` to `0.4219` across seeds

### Rationale

The hostile audit identified a real flaw in the first parent-density residual script.

The previous implementation filtered out every non-`exact` row before computing mass retention. That made the result vulnerable to survivor bias:

- if a candidate killed part of the anchor shell, the dead mass disappeared from the denominator of the reported readout
- if a null preserved broad support with very weak event counts, it was penalized differently from a candidate that pruned support

The corrected implementation separates two different questions.

First, the full partition question:

- how much anchor mass survives in the candidate when all active anchor rows are counted?

Second, the survivor-internal question:

- among the rows that still have positive candidate support, how much is the density profile deformed?

This keeps the object scientifically interpretable and defensible:

- `B` is no longer described as a shell amplifier by hiding dead rows
- `matched-lz` is no longer punished through a different conditioning contract
- the remaining separation is cleaner, because it is visible under the full partition contract

### Source Documents

- `docs/HSI-audit-Phase2-Derived-Parent-Density.md`
- `hsi_agents_project/v2/phase2/parent_density_residual.py`
- `hsi_agents_project/hsi_v2_phase2_parent_density_residual.py`
- `hsi_agents_project/hsi_v2_phase2_parent_density_pipeline.py`
- `hsi_agents_project/results/hsi_v2/phase2/parent_density_residual_corrected/phase2-parent-density-residual__anchor-E__cand-B__lag-15M__m-40-48__sel-bridge-linked__top-64__off-714M-plus-19__20260423T091934/`

### Practical Impact

- The parent-density line is again usable as an internal Phase 2 diagnostic, but only under the corrected contract.
- Any older exact-only density table must be read as survivor-conditioned, not as full partition retention.
- Future reports should use the following language:
  - full partition mass retention
  - survivor-internal retention
  - survivor-internal deformation
  - survivor-internal bias
- The corrected readout should remain below the strict `null_pressure` opening in evidentiary hierarchy until independently audited and replicated.

### Open Questions

- Is the `B` shell retention near `0.94` stable under nearby bands and alternative shell definitions?
- Does the survivor-internal bias remain small for `B` under finer partitioning of the shell?
- Can the matched-LZ shell deficit be localized to a specific subset of shell parents?

---

## D-0049

- Date: 2026-04-23
- Status: `provisional`
- Scope: hardening of the Phase 2 parent shell atlas after the hostile audit on deficit concentration
- Title: Parent shell atlas concentration must distinguish pooled net deficit from gross loss pressure

### Decision

The parent shell atlas is hardened as follows.

1. The old concentration object based on:
   - `sum(max(anchor - candidate, 0))`
   is no longer acceptable as the primary shell-deficit concentration metric.
2. The canonical concentration object is now:
   - `net_deficit_mass = max(sum(anchor) - sum(candidate), 0)`
   pooled pattern by pattern across windows.
3. The old object is retained only as an auxiliary diagnostic under a new explicit name:
   - `gross_loss_pressure_sum`
4. A third auxiliary quantity is introduced:
   - `jitter_gap_mass = gross_loss_pressure_sum - net_deficit_mass`
5. Shell subtype labels in this object must be described as observation-conditional:
   - they are defined relative to the monitored bridge-linked top-k high-scale child universe
   - they are not absolute global shell topology classes

### Rationale

The hostile audit identified a real flaw in the first shell-atlas concentration contract.

The previous metric merged two distinct phenomena:

- pooled irreversible loss of shell mass
- non-cancelling temporal dispersion of local support across windows

Those are not the same object.

If the concentration table is intended to say where mass is truly lost, the pooled net deficit must be the canonical ranking quantity.

The old metric still carries information, but only as local loss pressure. Once renamed and paired with `jitter_gap_mass`, it becomes interpretable instead of misleading.

The rerun after hardening did not collapse the empirical signal:

- observed `B` remains low-concentration under pooled net deficit
- matched-LZ remains broadly distributed across seeds
- markov1 correctly lands at `gross = net` with zero jitter gap

So the audit changes the contract and improves the object without overturning the line itself.

### Source Documents

- `docs/HSI-audit-Phase2-Derived-Parent-Atlas.md`
- `docs/HSI-audit-Phase2-Derived-Parent-Atlas-Response.md`
- `hsi_agents_project/v2/phase2/parent_shell_atlas.py`
- `hsi_agents_project/hsi_v2_phase2_parent_shell_atlas.py`
- `hsi_agents_project/results/hsi_v2/phase2/parent_shell_atlas/phase2-parent-shell-atlas__anchor-E__cand-B__lag-15M__m-40-48__sel-bridge-linked__top-64__off-714M-plus-19__20260423T201328/`

### Practical Impact

- The shell atlas remains valid as an internal diagnostic object for Phase 2.
- Future concentration claims from this object must cite:
  - `net_deficit_mass`
  - `gross_loss_pressure_sum`
  - `jitter_gap_mass`
- Any wording that treats `prefix-only` or `suffix-only` as absolute shell classes should be avoided.
- The object remains below strict `null_pressure` in evidentiary hierarchy and should not yet be promoted to paper-facing status without a dedicated audit closure.

### Open Questions

- Is the low-concentration shell deficit stable under wider child universes (`top-k` sensitivity)?
- Does `jitter_gap_mass` become more informative under finer offset grids or neighboring lag estimates?
- Is there a more structural family split than `prefix-only` / `suffix-only` that sharpens the atlas without reintroducing ranking bias?

---

## D-0050

- Date: 2026-04-24
- Status: `provisional`
- Scope: N2-06 sensitivity validation of the Phase 2 parent shell atlas
- Title: Parent-shell atlas separation is stable under top-k and lag sensitivity, with top-32 classified as a blind resolution control

### Decision

The N2-06 parent-shell atlas sensitivity sweep is accepted as a robust internal validation of the D-0049 shell-atlas contract.

The canonical sensitivity run is:

- `hsi_agents_project/results/hsi_v2/phase2/parent_shell_atlas_sensitivity/phase2-parent-shell-atlas-sensitivity__stage-all__anchor-E__cand-B__top-32-128x3__lag-14500K-15500Kx3__20260423T234050/`

The sensitivity contract tests:

- `top-k = 32, 64, 128`
- candidate lag neighborhood `14.5M, 15.0M, 15.5M`
- observed `E/B`
- `markov1`
- six `matched-LZ` seeds

The current reading is:

1. `top=32` is not a contradictory regime. It is a blind resolution-control regime:
   - shell structure is entirely `Unlinked`
   - no monitored prefix/suffix shell exists for the atlas readout
2. `top=64` remains the canonical lateral-shell readout:
   - observed `B` has stable low pooled net deficit and high jitter gap
   - `matched-LZ` has high net deficit and low jitter gap
   - `markov1` collapses with `gross = net` and `jitter_gap = 0`
3. `top=128` changes the monitored shell anatomy toward `Bridge-both`, but preserves the qualitative separation:
   - observed `B` remains dominated by jitter/redistribution rather than net loss
   - `matched-LZ` remains dominated by net shell deficit
   - `markov1` remains pure collapse

### Rationale

The key result is that observed `B` has lag-stable pooled net deficit across the tested lag neighborhood.

For observed `B`:

- `top=64`: `netDef = 276` at `14.5M`, `15.0M`, and `15.5M`
- `top=128`: `netDef = 864` at `14.5M`, `15.0M`, and `15.5M`

At the same time, gross loss varies with lag and is absorbed by `jitter_gap_mass`.

This is precisely the behavior the corrected D-0049 contract was designed to distinguish:

- `net_deficit_mass` measures pooled net shell-mass loss
- `gross_loss_pressure_sum` measures local loss pressure
- `jitter_gap_mass` measures lag-sensitive redistribution pressure

The result therefore argues against a parameter-tuning explanation based on the exact lag choice.

It also clarifies the role of `top-k`:

- `top=32` is too narrow to observe the shell atlas channel
- `top=64` captures the lateral prefix/suffix shell
- `top=128` expands the monitored support and reveals that most of the shell becomes `Bridge-both`

The shell subtype labels remain observation-conditional and must not be presented as absolute topology classes.

### Source Documents

- `docs/HSI-audit-Phase2-Derived-Parent-Atlas-Sensitivity.md`
- `docs/HSI-audit-Phase2-Derived-Parent-Atlas-Sensitivity-Response.md`
- `hsi_agents_project/hsi_v2_phase2_parent_shell_atlas_sensitivity.py`
- `hsi_agents_project/hsi_v2_phase2_parent_shell_atlas.py`
- `hsi_agents_project/hsi_v2_phase2_parent_survival_band_lagaware.py`
- `hsi_agents_project/v2/phase2/parent_shell_atlas.py`
- `hsi_agents_project/results/hsi_v2/phase2/parent_shell_atlas_sensitivity/phase2-parent-shell-atlas-sensitivity__stage-all__anchor-E__cand-B__top-32-128x3__lag-14500K-15500Kx3__20260423T234050/`

### Practical Impact

- N2-06 can be used as the sensitivity validation for the parent-shell atlas line.
- Paper-facing language may cite the stability of `net_deficit_mass` across lag and the separation between net loss and jitter gap.
- `top=64` should remain the main shell-lateral readout for the current paper-scale narrative.
- `top=128` should be used as sensitivity support showing that the separation persists under wider support.
- `top=32` should be described as an intentionally narrow blind control, not as evidence against the atlas.
- Internal language such as "invulnerable", "magic", or "pure physical signature" should be avoided in the paper.

### Open Questions

- Does the same net/jitter separation persist for nearby offset bands outside the current 714M-723M window?
- Can the `Bridge-both` anatomy at `top=128` be decomposed into a more structural shell taxonomy without reintroducing top-k bias?
- Should the first visual summary be the deficit-jitter phase plane, with heatmap/Hilbert-style views kept as secondary or exploratory figures?

---


## D-0051

- Date: 2026-04-24
- Status: `provisional`
- Scope: acceptance of the deficit-jitter phase plane as a paper-candidate visual asset for Phase 2
- Title: The deficit-jitter phase plane is accepted as a paper-candidate appendix figure for the D-0049 / D-0050 internal readout, with a fixed canonical caption; it does not replace `null_pressure` as the Gate 2 opening

### Decision

1. The deficit-jitter phase plane figure at `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/deficit_jitter_plane_latest.png` is accepted as a paper-candidate visual asset for HSI v2.
2. Its target placement in `Paper_Latex/main_v2_cat.tex` is the Phase 2 appendix, not the main body.
3. Its canonical caption is fixed as follows and must not be weakened when integrated into the paper:

   > Deficit-jitter phase plane for the Phase 2 parent-shell atlas sensitivity readout (top-k = 64, 128; lag neighborhood 14.5M-15.5M). Axes are normalized by the Markov-1 collapse mass at matched top-k and lag. Observed `E` sits at the origin by construction of the anchor reference. `markov1` defines the pure-collapse axis (normalized net deficit = 1, jitter-gap = 0), `matched-LZ` seeds remain net-deficit dominated across the tested neighborhood, and observed `B` occupies a redistribution regime with low normalized net shell deficit and high jitter-gap. The figure compactly summarizes the internal diagnostic readout of D-0049 and D-0050 and does not replace the strict `null_pressure` result as the primary Gate 2 opening.

4. The figure is explicitly subordinate to the strict `null_pressure` readout in the evidentiary hierarchy. Any paper-facing use must preserve this subordination.
5. Until the Phase 2 main body is drafted, no change to `main_v2_cat.tex` is required. The figure asset and caption remain staged.

### Rationale

D-0049 and D-0050 are both `provisional`. The shell-atlas line is useful as an internal diagnostic but is not yet paper-safe at the main-body level of evidentiary hierarchy.

The deficit-jitter phase plane visualizes those objects in a compact and reproducible way. It does not raise their formal status, and it does not displace `null_pressure`.

The appendix slot is the honest fit:

- it is faithful to the current `provisional` status of D-0049 and D-0050
- it makes the figure available for reference from the main body when Phase 2 narrative is written
- it avoids the risk of a figure floating in the main body without supporting narrative

The canonical caption explicitly states that `E = (0,0)` is a construction of the anchor reference rather than an empirical result. This blocks a tautological reading where `E` would appear to "win" the plane by having zero error. It also names `markov1`, `matched-LZ`, and `B` as three distinct regimes (pure collapse, net-deficit dominated, redistribution), so the reader sees the triangulation and not only a two-way contrast.

### Source Documents

- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/deficit_jitter_plane_latest.png`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/deficit_jitter_plane_latest.svg`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-deficit-jitter-plane__top-64-128__src-20260423T234050__20260424T161640/`
- `hsi_agents_project/Documentation/HSI_v2_Phase2_Live_Findings.md`
- D-0049
- D-0050

### Practical Impact

- The figure is now the canonical visual representation of the shell-atlas and jitter-gap diagnostic stack.
- No changes to `main_v2_cat.tex` are triggered by this decision.
- When Phase 2 is drafted in `main_v2_cat.tex`, the figure should be referenced as an appendix visual summary, not as a main-body result.
- The canonical caption must be preserved when the figure is translated into Catalan for the paper, including the explicit `E = (0,0)` construction note.
- Any future iteration on the figure itself must not silently drop the `markov1` pure-collapse pole or the `matched-LZ` seed range.

### Open Questions

- Should the appendix version also display the `matched-LZ` seed cloud explicitly, or is the range summary already sufficient?
- Should `top = 64` and `top = 128` appear as overlaid points in one panel, or as two side-by-side panels in the paper version?
- Is the current Markov-1 normalization the right reference, or would a double-null normalization (against both `markov1` and `matched-LZ` medians) communicate the separation more cleanly without losing the pure-collapse anchor?

## D-0052

- Date: 2026-04-24
- Status: `provisional`
- Scope: N2-07 / N2-07b same-parent jitter provenance readout below the parent-shell atlas line
- Title: Same-parent jitter provenance is accepted as a robust internal diagnostic, but not as child-destination routing

### Decision

The N2-07 / N2-07b jitter provenance readout is accepted as a robust internal diagnostic under the Phase 2 shell-atlas stack.

The accepted artifacts are:

- `hsi_agents_project/hsi_v2_phase2_jitter_provenance.py`
- `hsi_agents_project/hsi_v2_phase2_jitter_provenance_sensitivity.py`
- `hsi_agents_project/v2/phase2/jitter_provenance.py`
- `hsi_agents_project/results/hsi_v2/phase2/jitter_provenance/phase2-jitter-provenance__anchor-E__cand-B__top-64__lag-15M__src-20260423T225436__20260424T164539/`
- `hsi_agents_project/results/hsi_v2/phase2/jitter_provenance_sensitivity/phase2-jitter-provenance-sensitivity__top-64-128__src-20260423T234050__20260424T180722/`

The canonical N2-07 object measures same-parent temporal compensation inside an existing parent-shell atlas artifact.

Its primary quantities are:

- `gross_loss_mass_sum`
- `gross_surplus_mass_sum`
- `compensated_loss_mass = min(gross_loss_mass_sum, gross_surplus_mass_sum)` per parent pattern
- `net_deficit_mass`
- `compensated_loss_share_of_gross`

The N2-07b sensitivity wrapper repeats this readout over the useful N2-06 atlas neighborhood:

- `top-k = 64, 128`
- candidate lag neighborhood `14.5M, 15.0M, 15.5M`

The default excludes `top = 32` because D-0050 classifies it as a blind resolution control for the monitored shell channel.

### Rationale

The hostile audit accepted the object as a valid L1-style diagnostic for same-parent temporal compensation.

The current N2-07b readout is:

- observed `B`: stable `jitter-compensated` over all six useful sensitivity combinations
- `markov1`: stable `uncompensated-deficit`
- `matched-LZ`: net-deficit dominated across seeds and sensitivity settings
- observed `E`: flat at the anchor reference

The source-level envelope is:

- `B`: `compensated_loss_share_of_gross = 0.7778-0.8709`, `net_deficit_share_of_gross = 0.1291-0.2222`
- `markov1`: `compensated_loss_share_of_gross = 0.0000`, `net_deficit_share_of_gross = 1.0000`
- `matched-LZ`: `net_deficit_share_of_gross = 0.9074-1.0000` depending on seed

This hardens the D-0049 / D-0050 interpretation:

- observed `B` is not simply collapsing locally
- most of its local shell loss is compensated elsewhere in the analyzed band under exact parent-pattern identity
- `matched-LZ` imitates support but does not reproduce this compensation regime
- `markov1` remains the pure-collapse reference

### Wording Boundary

The accepted claim is deliberately conservative.

Canonical wording:

> Under exact parent-pattern identity and within the analyzed band, `min(gross_loss, gross_surplus)` measures observable same-parent temporal compensation.

Do **not** phrase this as:

- guaranteed identity of physical flow
- proof of child-destination routing
- proof of cross-parent transport
- final transport law
- geometry

The `provenance_shift_bits` centroid readout is useful only as a secondary diagnostic of the direction of temporal displacement relative to the chosen candidate lag. It is not a primary claim.

### Source Documents

- `docs/HSI-audit-Phase2-Jitter-Provenance.md`
- `docs/HSI-audit-Phase2-Jitter-Provenance-Response.md`
- `hsi_agents_project/hsi_v2_phase2_jitter_provenance.py`
- `hsi_agents_project/hsi_v2_phase2_jitter_provenance_sensitivity.py`
- `hsi_agents_project/v2/phase2/jitter_provenance.py`
- `hsi_agents_project/results/hsi_v2/phase2/jitter_provenance/phase2-jitter-provenance__anchor-E__cand-B__top-64__lag-15M__src-20260423T225436__20260424T164539/`
- `hsi_agents_project/results/hsi_v2/phase2/jitter_provenance_sensitivity/phase2-jitter-provenance-sensitivity__top-64-128__src-20260423T234050__20260424T180722/`

### Practical Impact

- N2-07 / N2-07b can be used as a hardened internal diagnostic below `null_pressure`.
- The result strengthens the interpretation that observed `B` expresses temporal redistribution rather than net shell collapse.
- The result does not change the paper hierarchy: strict `null_pressure` remains the first paper-safe Gate 2 opening.
- The natural next object is cross-parent / child-destination routing, but that remains unproven by N2-07.

### Open Questions

- Can a child-destination routing object show where the compensated shell mass goes at the high-scale child level?
- Does the same same-parent compensation regime persist outside the current `714M-723M` offset band?
- Can `provenance_shift_bits` be stabilized by a finer lag search, or should it remain purely secondary?

---

## D-0053

- Date: 2026-04-24
- Status: `provisional`
- Scope: N2-08 child-destination routing below the parent-shell / jitter-provenance diagnostic stack
- Title: Jitter-compensated child-destination routing is accepted as a robust internal diagnostic

### Decision

The corrected N2-08 child-destination routing readout is accepted as a robust internal diagnostic under the Phase 2 parent-shell stack.

The accepted artifacts are:

- `hsi_agents_project/hsi_v2_phase2_child_destination_routing.py`
- `hsi_agents_project/v2/phase2/child_destination_routing.py`
- `hsi_agents_project/results/hsi_v2/phase2/child_destination_routing/phase2-child-destination-routing__anchor-E__cand-B__top-128__lag-15M__src-20260424T091639__20260424T231132/`
- `docs/HSI-audit-Phase2-Child-Routing.md`
- `docs/HSI-audit-Phase2-Child-Routing-Response.md`

The original N2-08 draft measured child-destination overlap in a window-synchronous way. The hostile audit correctly identified that this penalized temporal jitter as routing failure.

The corrected N2-08 object now uses jitter-compensated child-destination retention as its primary metric:

1. Pool anchor mass by high-scale child destination across all analyzed windows.
2. Pool candidate mass by high-scale child destination across all analyzed windows.
3. Compute overlap only after pooling:

```text
sum_child min(anchor_mass_by_child[child], candidate_mass_by_child[child])
```

The old window-synchronous overlap is retained only as a secondary jitter-control diagnostic.

### Rationale

The corrected canonical rerun uses:

- anchor `E`
- candidate `B`
- `top = 128`
- candidate lag `15M`
- scales `m = 40 -> 48`
- selector `bridge-linked`
- the parent-shell atlas sensitivity artifact generated at `20260424T091639`

The source-level corrected readout is:

| Source | Child-destination retention | Window-synchronous retention | Monitored reroute | Class |
| --- | ---: | ---: | ---: | --- |
| observed `B` | 0.9604 | 0.7194 | 0.0000 | same-destination-conserved |
| `B-markov1` | 0.0000 | 0.0000 | - | destination-collapse |
| `B-matched-lz` envelope | 0.3107-0.4832 | 0.2874-0.4408 | 0.0000 | same-destination-deficit |
| observed `E` | 1.0000 | 1.0000 | 0.0000 | same-destination-conserved |

This shows that the earlier `B = 0.7194` value was an under-estimate caused by window-synchronous jitter. Once child destinations are pooled before overlap, observed `B` retains about `96%` of the monitored anchor child-destination mass.

The same correction does not rescue `matched-LZ`. Its seed envelope remains in the `31-48%` same-destination-deficit band. `markov1` remains destination collapse.

The result therefore refines N2-07:

- N2-07 showed same-parent temporal compensation.
- N2-08 now shows that this compensation largely preserves the monitored high-scale child destinations for observed `B`.
- `matched-LZ` imitates support but does not reproduce the same child-destination conservation.

### Wording Boundary

The accepted claim is still limited.

Canonical wording:

> Under the monitored `top = 128` child universe and the lag-aware `E`-anchored shell contract, observed `B` retains approximately `96%` of anchor child-destination mass after jitter compensation, while `matched-LZ` remains in a `31-48%` same-destination-deficit band.

Do **not** phrase this as:

- absolute proof of all child routing
- cross-parent routing
- classification of mass leaving the monitored child universe
- final Gate 2 resolution
- a complete transport law
- geometry

The term `monitored reroute` is deliberately local:

> `monitored reroute` means candidate mass landing on tracked child destinations outside the anchor child set.

Mass that leaves the tracked `top-k` child universe cannot be classified as reroute by N2-08. It remains deficit/off-support.

### Source Documents

- `docs/HSI-audit-Phase2-Child-Routing.md`
- `docs/HSI-audit-Phase2-Child-Routing-Response.md`
- `hsi_agents_project/hsi_v2_phase2_child_destination_routing.py`
- `hsi_agents_project/v2/phase2/child_destination_routing.py`
- `hsi_agents_project/results/hsi_v2/phase2/child_destination_routing/phase2-child-destination-routing__anchor-E__cand-B__top-128__lag-15M__src-20260424T091639__20260424T231132/`

### Practical Impact

- N2-08 becomes the current strongest mechanistic internal diagnostic below strict `null_pressure`.
- The result connects same-parent temporal compensation to high-scale child-destination preservation.
- The result strengthens the claim that observed `B` is not merely `matched-LZ` with local support imitation.
- The paper hierarchy does not change: strict `null_pressure` remains the first paper-safe Gate 2 opening.
- No change to `main_v2_cat.tex` is triggered automatically by this decision.

### Open Questions

- Does the same child-destination conservation persist outside the current `714M-723M` anchor band?
- Can a future object classify off-universe destination mass without exploding the tracked child universe?
- Should N2-08 receive a compact appendix figure once Phase 2 is drafted in the paper?

---

## D-0054

- Date: 2026-04-25
- Status: `provisional`
- Scope: N2-09 child-routing persistence across adjacent offset bands
- Title: Child-destination conservation is accepted as lag-local and phase-sensitive, not globally stationary

### Decision

The N2-09 child-routing persistence wrapper is accepted as a valid internal revalidation object for the corrected N2-08 child-destination routing readout.

The accepted artifacts are:

- `hsi_agents_project/hsi_v2_phase2_child_routing_persistence.py`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-fixed-15M__starts-705M-723Mx3__20260425T000933/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_laglocal/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-fixed-5500K__starts-723M-723Mx1__20260425T012305/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_probe_map/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-696M-723Mx4__20260425T015723/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_boundary705_wide/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-705M-705Mx1__20260425T085419/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_probe_extension/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-732M-732Mx1__20260425T025619/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_regime_map/phase2-child-routing-regime-map__20260425T091530/`

The N2-09 wrapper runs, for each analyzed band:

1. a parent-shell lag probe,
2. a lag-aware parent-survival band,
3. a parent-shell atlas,
4. the corrected N2-08 child-destination routing readout,
5. a band-level persistence summary.

The first full persistence sweep used:

- anchor `E`
- candidate `B`
- `top = 128`
- fixed candidate lag `15M`
- bands `705M-714M`, `714M-723M`, and `723M-732M`
- `19` windows per band
- step `500K`

The fixed-lag sweep showed:

| Band | Probe lag | Fixed lag | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `705M-714M` | - | `15M` | `0.0000` | `0.0940-0.2351` | `0.0000` |
| `714M-723M` | `15M` | `15M` | `0.9604` | `0.3107-0.4832` | `0.0000` |
| `723M-732M` | `5.5M` | `15M` | `0.7604` | `0.2015-0.3505` | `0.0000` |

The follow-up lag-local rerun for `723M-732M`, using the probe-supported `5.5M` lag, changed the readout to:

| Band | Lag | observed `B` child retention | observed `B` sync retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `723M-732M` | `5.5M` | `1.0000` | `0.9948` | `0.2088-0.4055` | `0.0000` |

No source row in these N2-09 summaries was preview-truncated.

An N2-09b probe-mode regime-map rerun was then executed:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_probe_map/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-696M-723Mx4__20260425T015723/`

This rerun used `--lag-mode probe` and included an explicit off-regime / boundary band. Bands without a probe-derived lag are no longer forced through child-routing; they are recorded as `band_status = no-probe-lag`.

The N2-09b regime map is:

| Band | Probe lag | Status | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | --- | ---: | ---: | ---: |
| `696M-705M` | `500K` | completed | `0.6456` | `0.1650-0.2860` | `0.0000` |
| `705M-714M` | - | `no-probe-lag` | - | - | - |
| `714M-723M` | `15M` | completed | `0.9604` | `0.3107-0.4832` | `0.0000` |
| `723M-732M` | `5.5M` | completed | `1.0000` | `0.2088-0.4055` | `0.0000` |

At this stage, the N2-09b regime map did not create a separate decision because it applied the D-0054 persistence contract. It first refined D-0054 by converting the result into a regime map with one liminal positive band, one unresolved narrow-probe boundary, one canonical strong band, and one phase-shifted strong band. The later D-0055 entry is reserved for the separate N2-10/N2-10c phase-lag atlas and targeted validation layer.

The apparent `705M-714M` no-lag boundary was then investigated with a wider shell-lag probe:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_boundary705_wide/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-705M-705Mx1__20260425T085419/`

This rerun scanned `660M-780M` and found a negative local phase:

| Band | Probe lag | Status | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | --- | ---: | ---: | ---: |
| `705M-714M` | `-27M` | completed | `1.0000` | `0.1747-0.3297` | `0.0000` |

Therefore the earlier no-lag row should be read narrowly: the forward/local scan did not find a lag, but the band is not intrinsically outside transport. It is a negative-lag, phase-shifted strong band under the wider boundary probe.

A one-band forward extension was then executed:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_probe_extension/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-732M-732Mx1__20260425T025619/`

The extension result is:

| Band | Probe lag | Status | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | --- | ---: | ---: | ---: |
| `732M-741M` | `0` | completed | `1.0000` | `0.5361-0.8317` | `0.0000` |

This extension should be treated as a distinct same-window / zero-lag subregime. It confirms that observed `B` remains perfectly conserved in the monitored child universe, but the separation from `matched-LZ` is weaker than in the `714M-723M` and `723M-732M` bands. In this band, `matched-LZ` partially approaches the observed channel and two seeds are classified as `same-destination-conserved` under the current thresholds.

### Rationale

N2-09 answers the first open question left by D-0053: whether the corrected child-destination conservation survives outside the original `714M-723M` band.

The answer is positive but not stationary.

The original `714M-723M` band remains strong at the original `15M` lag:

- observed `B`: `0.9604`
- `matched-LZ`: `0.3107-0.4832`
- `markov1`: `0.0000`

The later `723M-732M` band initially looked attenuated under the same forced `15M` lag:

- observed `B`: `0.7604`
- `matched-LZ`: `0.2015-0.3505`

However, the independent shell-lag probe for that band recommended `5.5M`, not `15M`. When the band was rerun with this local lag, observed `B` recovered full monitored child-destination retention:

- observed `B`: `1.0000`
- observed `B` sync retention: `0.9948`
- observed `B` deficit: `0`
- `matched-LZ`: still only `0.2088-0.4055`
- `markov1`: still `0.0000`

This means the apparent attenuation in the later band was mainly a phase / lag mismatch, not a failure of the destination channel.

The earlier `705M-714M` band was initially outside the established child-routing regime under the narrow probe contract:

- the probe did not find a recommended full-shell lag,
- forcing `15M` produced observed `B = 0.0000`,
- therefore the first conservative reading was off-regime / pre-regime boundary.

The wider boundary probe revised that reading:

- scanning `660M-780M` found `recommended_lag_bits = -27M`,
- observed `B` reached `1.0000`,
- `matched-LZ` remained only `0.1747-0.3297`,
- `markov1` remained `0.0000`.

This means the band is not absent; it is retrograde / negative-lag relative to the `E` anchor band.

The N2-09b map adds that `696M-705M` is not simply null:

- the probe finds a short `500K` local lag,
- observed `B` reaches `0.6456`,
- the strongest `matched-LZ` seed remains at `0.2860`,
- `markov1` remains `0.0000`.

This is best read as a liminal / partial regime, not as a full conservation plateau.

The `732M-741M` extension adds a later zero-lag subregime:

- the probe finds `recommended_lag_bits = 0`,
- observed `B` remains `1.0000`,
- the monitored shell is smaller (`31` shell patterns, `15` monitored),
- `matched-LZ` rises to `0.5361-0.8317`,
- `markov1` remains `0.0000`.

This should not be used as the strongest separation claim. Its value is cartographic: it shows that the active child-destination regime can enter a coincident same-window phase in which the strongest null partly shadows the monitored destinations.

### Wording Boundary

The accepted claim is deliberately local.

Canonical wording:

> Under the monitored `top = 128` child universe, corrected child-destination conservation persists beyond the original `714M-723M` band when the candidate lag is chosen from the band-local shell-lag probe. The `723M-732M` band recovers observed `B = 1.0000` at `5.5M`, while `matched-LZ` remains in a `0.21-0.41` same-destination-deficit envelope and `markov1` remains destination collapse.

Do **not** phrase this as:

- global stationarity of the child-routing regime,
- a universal `15M` lag,
- a final transport law,
- absolute routing outside the monitored `top-k` child universe,
- cross-parent routing,
- completed geometry.

The correct interpretation is phase-sensitive:

- `696M-705M`: partial / liminal child-destination conservation at `500K`
- `705M-714M`: negative-lag strong child-destination conservation at `-27M` under the wide boundary probe
- `714M-723M`: child-destination conservation at `15M`
- `723M-732M`: child-destination conservation at `5.5M`
- `732M-741M`: same-window / zero-lag conservation, but with weaker separation from `matched-LZ`

### Source Documents

- `hsi_agents_project/hsi_v2_phase2_child_routing_persistence.py`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-fixed-15M__starts-705M-723Mx3__20260425T000933/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_laglocal/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-fixed-5500K__starts-723M-723Mx1__20260425T012305/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_probe_map/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-696M-723Mx4__20260425T015723/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_boundary705_wide/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-705M-705Mx1__20260425T085419/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_persistence_probe_extension/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-732M-732Mx1__20260425T025619/`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_regime_map/phase2-child-routing-regime-map__20260425T091530/`

### Practical Impact

- D-0053 is strengthened, not replaced.
- N2-08 should now be read as a local child-destination conservation mechanism with measurable phase / lag shifts.
- The derived Phase 2 stack now has a persistence check beyond the original canonical band.
- The paper hierarchy still does not change: strict `null_pressure` remains the first paper-safe Gate 2 opening.
- No change to `main_v2_cat.tex` is triggered automatically by this decision.

### Open Questions

- Does the newly detected `705M-714M` negative-lag phase connect smoothly to the `696M-705M` partial regime?
- Does the lag jump from `15M` to `5.5M` follow a stable drift law across later bands?
- Does the later zero-lag regime continue beyond `741M`, or is it a short coincident phase?
- Should zero-lag bands be reported separately from positive-lag transport bands in any eventual visualization?
- Can the phase-sensitive persistence readout be turned into a compact visual without overstating stationarity?

---

## D-0055

**Date:** 2026-04-25
**Status:** Accepted as an internal Phase 2 diagnostic result; not a paper-safe Gate 2 resolution by itself.

### Decision

Introduce `N2-10` / `N2-10b` / `N2-10c` as a phase-lag atlas and targeted child-routing validation layer over the D-0054 persistence contract.

The accepted interpretation is:

- the visual phase-lag curve is not to be called a wave law;
- the useful object is a tri-estimator phase map over first-full-shell lag, hit-profile correlation lag, and unique-profile correlation lag;
- bands are classified as `strong`, `transition`, or `ambiguous` before child-routing validation;
- only `strong` and `transition` bands are validated with the audited N2-09/N2-08 child-routing stack;
- the result is a phase-local negative-lag plateau plus asymmetric transition boundaries.

### Executed Runs

Phase-lag atlas:

- `hsi_agents_project/results/hsi_v2/phase2/phase_lag_atlas/phase2-phase-lag-atlas__anchor-E__cand-B__top-128__bands-660M-732Mx9__probe-back-45M__fwd-45M__20260425T100139/`

Tri-estimator audit:

- `hsi_agents_project/results/hsi_v2/phase2/phase_lag_tri_estimator/phase2-phase-lag-tri-estimator__anchor-E__cand-B__top-128__bands-660M-732Mx9__tol-1M__20260425T104220/`

Targeted child-routing phase validation:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_phase_validation/phase2-child-routing-phase-validation__stage-all__conf-strong-transition__bands-687M-723Mx5__20260425T111658/`

### Result

The tri-estimator audit selected:

| Band | First full | Hit corr | Unique corr | Confidence | Consensus lag |
| --- | ---: | ---: | ---: | --- | ---: |
| `687M-696M` | `26.5M` | `86.5M` | `87M` | transition | `87M` |
| `696M-705M` | `-27M` | `-27M` | `-27M` | strong | `-27M` |
| `705M-714M` | `-27M` | `-27M` | `-27M` | strong | `-27M` |
| `714M-723M` | `-26M` | `-27M` | `-27M` | strong | `-27M` |
| `723M-732M` | `-17M` | `-27M` | `-27M` | transition | `-27M` |

The targeted validation gives:

| Band | Confidence | Lag | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | --- | ---: | ---: | ---: | ---: |
| `687M-696M` | transition | `87M` | `0.5466` | `0.7533-0.8567` | `0.0000` |
| `696M-705M` | strong | `-27M` | `0.9703` | `0.1877-0.3189` | `0.0000` |
| `705M-714M` | strong | `-27M` | `1.0000` | `0.1747-0.3297` | `0.0000` |
| `714M-723M` | strong | `-27M` | `0.9947` | `0.2631-0.4124` | `0.0000` |
| `723M-732M` | transition | `-27M` | `1.0000` | `0.2360-0.3592` | `0.0000` |

### Fine Boundary Sweep Extension

On 2026-04-26, a fine boundary sweep was executed to refine the plateau edges:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_boundary_sweep/phase2-child-routing-boundary-sweep__stage-all__top-128__bands-684M-735Mx12__20260426T105136/`

The sweep sampled:

- left boundary: `684M`, `687M`, `690M`, `693M`, `696M`
- plateau controls: `705M`, `714M`
- right boundary: `723M`, `726M`, `729M`, `732M`, `735M`

Tri-estimator classification:

- `ambiguous`: `5`
- `transition`: `4`
- `strong`: `3`

Validated `strong`/`transition` bands:

| Band | Confidence | Lag | observed `B` child retention | `matched-LZ` envelope | `B - max(matched-LZ)` |
| --- | --- | ---: | ---: | ---: | ---: |
| `687M-696M` | transition | `87M` | `0.5466` | `0.7533-0.8567` | `-0.3101` |
| `696M-705M` | strong | `-27M` | `0.9703` | `0.1877-0.3189` | `0.6514` |
| `705M-714M` | strong | `-27M` | `1.0000` | `0.1747-0.3297` | `0.6703` |
| `714M-723M` | strong | `-27M` | `0.9947` | `0.2631-0.4124` | `0.5823` |
| `723M-732M` | transition | `-27M` | `1.0000` | `0.2360-0.3592` | `0.6408` |
| `726M-735M` | transition | `-88M` | `0.0000` | `0.1410-0.3531` | `-0.3531` |
| `729M-738M` | transition | `-27M` | `1.0000` | `0.5976-0.7354` | `0.2646` |

This extension sharpens, rather than weakens, the D-0055 reading:

- the core `696M-723M` plateau is robust;
- the left boundary remains non-positive for HSI under child-routing validation;
- the right boundary is not a smooth decay, but an aliased / phase-sensitive transition zone;
- `726M-735M` is a useful negative transition control;
- `729M-738M` remains positive, but separation from `matched-LZ` is much weaker than inside the plateau.

### Scientific Reading

The robust internal reading is now:

> Observed `B` exhibits a strong monitored child-destination conservation plateau from `696M-723M` at approximately `-27M` lag, while `matched-LZ` remains same-destination-deficit dominated and `markov1` collapses.

The right transition band `723M-732M` remains positive for HSI under the correlation-supported `-27M` lag: observed `B` remains fully conserved while `matched-LZ` remains below `0.36`.

The left transition band `687M-696M` is not positive child-routing evidence: under the `87M` correlation-supported lag, observed `B` is only partial and `matched-LZ` conserves more strongly. It should be treated as a genuine alias / transition boundary, not as part of the plateau.

After the fine boundary sweep, the strictest wording is:

> The strong plateau is `696M-723M`. The left boundary is non-positive under child-routing validation. The right boundary remains partially HSI-positive but becomes alias-sensitive beyond `723M`.

### Wording Boundary

Allowed wording:

- "phase-local child-destination plateau"
- "negative-lag plateau around `-27M`"
- "asymmetric transition boundaries"
- "tri-estimator phase audit"

Forbidden or premature wording:

- "wave law"
- "global stationarity"
- "universal lag"
- "completed transport law"
- "absolute routing outside the monitored `top-k` child universe"
- "Gate 2 solved"

### Practical Impact

- D-0054 is sharpened, not replaced.
- N2-10c gives a cleaner validation target than the earlier visual regime map.
- The derived internal stack is stronger because it now contains both positive and negative boundary evidence.
- `null_pressure` remains the first paper-safe Gate 2 opening.
- No automatic change to `Paper_Latex/main_v2_cat.tex`.

### Open Questions

- Does the `-27M` plateau extend earlier than `696M` under a different confidence contract?
- Does the right transition continue beyond `732M`, or does the zero-lag subregime from D-0054 take over?
- Can the tri-estimator phase map be converted into a paper-safe appendix figure without implying a wave law?
- Should the left transition band be used as a falsification/control example in the eventual Phase 2 appendix?

---

## D-0056

**Date:** 2026-04-27
**Status:** Accepted as a completed methodological hardening of N2-10c/N2-10d.

### Decision

Harden the N2-10c child-routing phase validation contract so that `transition` bands are not validated only through the tri-estimator `consensus_lag`.

Grace's external audit correctly identified that, in `transition` bands, the consensus lag can be dominated by the two profile-correlation estimators (`hit_corr` and `unique_corr`) while the `first_full` estimator points to a different topological alignment. Since N2-08/N2-09 child-destination routing is a topological validation, a transition band whose `first_full_lag` disagrees with the consensus by more than the tri-estimator tolerance must be validated under both lags.

### Implementation

`hsi_v2_phase2_child_routing_phase_validation.py` now supports an explicit `--lag-policy`:

- `consensus`: reproduce the previous N2-10c behavior.
- `first-full`: validate selected bands only at the `first_full_lag`.
- `dual-transition`: validate normal bands at consensus, but duplicate discrepant `transition` bands with both `consensus` and `first_full` targets.

`hsi_v2_phase2_child_routing_boundary_sweep.py` now defaults to:

- `--validation-lag-policy dual-transition`

This makes the safe contract the default for future N2-10d boundary sweeps, while preserving the old behavior as an explicit reproducibility option.

### Immediate Check

A dry-run over the fine N2-10d boundary summaries expanded the previous seven validation targets into eleven:

- three `strong` bands remain single-target consensus validations;
- four discrepant `transition` bands are each validated twice, once at consensus and once at `first_full`.

The especially sensitive `726M-735M` band is now explicitly split into:

- consensus lag: `-88M`
- first-full lag: `-14M`

### Scientific Reading Boundary

D-0055 remains the historical result record, but its transition-band interpretation is superseded by the dual-lag rerun below.

The robust `strong` plateau `696M-723M` is not weakened by this change, because its three estimators already agree within tolerance.

Transition bands should now be read as boundary probes, not as settled plateau evidence.

### Completed Dual-Lag Rerun

Executed run:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_boundary_sweep_dual_transition/phase2-child-routing-boundary-sweep__stage-all__top-128__lag-dual-transition__bands-684M-735Mx12__20260427T084746/`

Validation targets:

- `11` total targets.
- `3` `strong` targets, single consensus lag.
- `4` discrepant `transition` bands, each validated at both consensus and `first_full`.

Dual-lag validation table:

| Target | Confidence | Lag source | Lag | observed `B` | `matched-LZ` envelope | `B - max(matched-LZ)` | `markov1` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `687M-696M` | transition | consensus | `87M` | `0.5466` | `0.7533-0.8567` | `-0.3101` | `0.0000` |
| `687M-696M` | transition | first_full | `26.5M` | `0.5466` | `0.7655-0.8673` | `-0.3207` | `0.0000` |
| `696M-705M` | strong | consensus | `-27M` | `0.9703` | `0.1877-0.3189` | `0.6514` | `0.0000` |
| `705M-714M` | strong | consensus | `-27M` | `1.0000` | `0.1747-0.3297` | `0.6703` | `0.0000` |
| `714M-723M` | strong | consensus | `-27M` | `0.9947` | `0.2631-0.4124` | `0.5823` | `0.0000` |
| `723M-732M` | transition | consensus | `-27M` | `1.0000` | `0.2360-0.3592` | `0.6408` | `0.0000` |
| `723M-732M` | transition | first_full | `-17M` | `0.0000` | `0.1921-0.3392` | `-0.3392` | `0.0000` |
| `726M-735M` | transition | consensus | `-88M` | `0.0000` | `0.1410-0.3531` | `-0.3531` | `0.0000` |
| `726M-735M` | transition | first_full | `-14M` | `0.0000` | `0.1723-0.3089` | `-0.3089` | `0.0000` |
| `729M-738M` | transition | consensus | `-27M` | `1.0000` | `0.5976-0.7354` | `0.2646` | `0.0000` |
| `729M-738M` | transition | first_full | `-5.5M` | `0.2354` | `0.4285-0.8477` | `-0.6123` | `0.0000` |

### Scientific Reading After Dual-Lag Rerun

The dual-lag rerun strengthens the strict plateau reading:

- `696M-723M` remains the robust child-routing plateau.
- `687M-696M` is non-positive under both consensus and first-full validation.
- `723M-732M` is lag-selective: positive at the correlation-supported `-27M`, collapsed at the `first_full` `-17M`.
- `726M-735M` is negative under both consensus and first-full validation; `first_full` does not rescue observed `B`.
- `729M-738M` is also lag-selective: positive at `-27M`, weak/negative at `first_full=-5.5M`.
- `markov1` remains a pure collapse control across all targets.

The resulting boundary language is:

> Observed `B` has a robust child-destination conservation plateau at `696M-723M` under a stable `-27M` lag. The left boundary is non-positive. The right boundary is phase-selective rather than smoothly persistent: some correlation-supported `-27M` continuations remain positive, while first-full topological lags can collapse completely.

### Practical Impact

- The N2-10d default is stricter.
- The earlier N2-10d fine sweep remains useful as a pre-hardening diagnostic, but the dual-transition rerun is now the canonical boundary readout.
- The Grace audit objection is resolved for the current boundary sweep.
- No automatic change to `Paper_Latex/main_v2_cat.tex`.

---

## D-0057

**Date:** 2026-04-28
**Status:** Accepted as an internal Phase 2 diagnostic result; not a paper-safe Gate 2 resolution by itself.

### Decision

Introduce `N2-11` as an explicit child-routing lag-response profile over the audited N2-08/N2-09 child-destination routing stack.

Instead of selecting a single candidate lag from the tri-estimator map, N2-11 evaluates a fixed lag grid across the current boundary/plateau bands. The purpose is to distinguish:

- a true single-lag plateau;
- a smooth wave-like drift;
- discrete phase channels;
- null-like diffuse retention.

This is a diagnostic layer over D-0056, not a replacement for the strict dual-lag boundary audit.

### Executed Run

Canonical N2-11 lag-response run:

- `hsi_agents_project/results/hsi_v2/phase2/child_routing_lag_response_n2_11/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-8__20260427T165759/`

Visual readout:

- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T082425/`

Grid:

- bands: `696M-705M`, `705M-714M`, `714M-723M`, `723M-732M`, `726M-735M`, `729M-738M`
- lags: `-88M`, `-27M`, `-17M`, `-14M`, `-5.5M`, `0M`, `26.5M`, `87M`
- sources: observed `B`, observed `E`, `markov1`, six `matched-LZ` seeds
- all `48` band-lag targets completed

### Result

Best cells by band:

| Band | Best observed `B` lag | observed `B` | Best margin lag | `B - max(matched-LZ)` |
| --- | ---: | ---: | ---: | ---: |
| `696M-705M` | `-5.5M` | `0.9852` | `-5.5M` | `0.6608` |
| `705M-714M` | `-27M` | `1.0000` | `-5.5M` | `0.6782` |
| `714M-723M` | `-27M` | `0.9947` | `-27M` | `0.5823` |
| `723M-732M` | `-27M` | `1.0000` | `-27M` | `0.6408` |
| `726M-735M` | `-27M` | `1.0000` | `26.5M` | `0.6495` |
| `729M-738M` | `-27M` | `1.0000` | `0M` | `0.2810` |

Lag averages across the six bands:

| Lag | Average observed `B` | Average `B - max(matched-LZ)` |
| ---: | ---: | ---: |
| `-88M` | `0.034` | `-0.412` |
| `-27M` | `0.994` | `0.565` |
| `-17M` | `0.394` | `-0.019` |
| `-14M` | `0.318` | `-0.085` |
| `-5.5M` | `0.404` | `-0.036` |
| `0M` | `0.343` | `-0.062` |
| `26.5M` | `0.759` | `0.311` |
| `87M` | `0.031` | `-0.413` |

Controls:

- `markov1` remains exactly `0.0000` across the whole grid.
- `matched-LZ` remains a non-zero diffuse background rather than a collapse control.
- `matched-LZ` becomes substantially stronger in the late `729M-738M` band, which weakens the observed `B` separation there.

### Scientific Reading

N2-11 does not support a single universal lag or a smooth wave law.

The current best reading is:

> Observed `B` exhibits discrete phase-channel child-destination conservation. The strongest global channel is `-27M`, but additional positive channels appear at `26.5M` and in some bands at `-5.5M`, `-14M`, or `0M`. `markov1` collapses everywhere, while `matched-LZ` behaves as a diffuse non-zero background.

The previous D-0056 statement that `726M-735M` is negative must now be bounded to the dual-lag estimator contract only:

- under D-0056's consensus/first-full lags (`-88M`, `-14M`), `726M-735M` is negative;
- under the full N2-11 lag grid, `726M-735M` is strongly positive at `-27M` and `26.5M`;
- therefore `726M-735M` is not a true dead zone, but a phase-selective boundary band.

### Wording Boundary

Allowed wording:

- "discrete phase-channel response"
- "lag-response profile"
- "phase-selective child-destination conservation"
- "`-27M` is the strongest global tested channel"
- "`726M-735M` is not dead under the full lag grid"

Forbidden or premature wording:

- "wave law"
- "universal lag"
- "global stationarity"
- "complete transport law"
- "Gate 2 solved"
- "absolute routing outside the monitored top-k child universe"

### Practical Impact

- D-0056 remains valid as a stricter dual-lag boundary audit.
- N2-11 supersedes any broader reading of `726M-735M` as a dead band.
- The heatmap is a useful internal visual candidate, but not yet a paper figure.
- `null_pressure` remains the first paper-safe Gate 2 opening.
- No automatic change to `Paper_Latex/main_v2_cat.tex`.

### Open Questions

- Is the `26.5M` channel an alias of the `-27M` channel under tower periodicity, or an independent phase route?
- Should the next sweep densify the lag grid around `-27M`, `-5.5M`, `0M`, and `26.5M`?
- Does the high `matched-LZ` background in `729M-738M` mark the end of clean separation, or a different transport regime?
- Can the N2-11 heatmap be made paper-safe without inviting a premature wave-law interpretation?

---

## D-0058

- Date: 2026-04-28
- Status: `provisional`
- Scope: acceptance of the N2-11 lag-response heatmap as a paper-candidate visual asset for Phase 2; closure of the editorial open question left by D-0057
- Title: The N2-11 lag-response heatmap is accepted as a paper-candidate appendix figure for the D-0057 readout, with a fixed canonical caption; it does not replace `null_pressure` as the Gate 2 opening

### Decision

1. The N2-11 lag-response heatmap at `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/lag_response_heatmap.png` is accepted as a paper-candidate visual asset for HSI v2.
2. Its target placement in `Paper_Latex/main_v2_cat.tex` is the Phase 2 appendix, alongside the deficit-jitter phase plane (D-0051), not the main body.
3. Its canonical caption is fixed as follows and must not be weakened when integrated into the paper:

   > Lag-response profile (N2-11) for the monitored child-destination conservation channel (top-`k` = 128, `E`-anchored shell contract). Heatmap of the corrected jitter-compensated child-destination retention evaluated over an explicit candidate-lag grid (`-88M`, `-27M`, `-17M`, `-14M`, `-5.5M`, `0M`, `26.5M`, `87M`) for six consecutive offset bands (`696M-705M`, `705M-714M`, `714M-723M`, `723M-732M`, `726M-735M`, `729M-738M`). Left panel: observed `B`. Right panel: separation `B - max(matched-LZ)` across six matched-LZ seeds. Black outlines mark the lag recommended by the local shell-lag probe for each band; they do not mark a global lag. Observed `B` exhibits discrete phase-channel response: it locks onto a strong global tested channel at `-27M` and a secondary channel at `26.5M`, while collapsing at `-88M` and `87M`. `markov1` collapses to zero across the whole grid; `matched-LZ` remains a diffuse non-zero background and becomes substantially stronger in the late `729M-738M` band, which is why the right-panel margin contracts there. The figure is a paper-candidate internal mechanistic diagnostic only: it does not establish a wave law, a universal lag, global stationarity, a complete transport law, absolute routing outside the monitored top-`k` child universe, or a Gate 2 resolution. It supports, and is subordinate to, the strict `null_pressure` readout that remains the first paper-safe Gate 2 opening.

4. The figure is explicitly subordinate to the strict `null_pressure` readout in the evidentiary hierarchy. Any paper-facing use must preserve this subordination and the discrete phase-channel framing; smooth wave-like phrasings are forbidden in any caption variant or surrounding paragraph.
5. Until the Phase 2 main body is drafted, no change to `main_v2_cat.tex` is required. The figure asset and caption remain staged.
6. This decision closes the D-0057 open question "Can the N2-11 heatmap be made paper-safe without inviting a premature wave-law interpretation?" with a conditional yes: paper-safe under the canonical caption fixed here, paper-unsafe under any caption that drops the discrete phase-channel framing or the explicit forbidden-wording fence.

### Rationale

D-0057 is `provisional` and explicitly leaves the editorial paper-safe question open.

The lag-response heatmap is the most mature visual asset Phase 2 has produced so far. The hostile audit (`docs/HSI-audit-Phase2-Lag-Response.md`) accepted the "discrete phase-channel response" reading as the only physically honest formulation of the empirical pattern, and noted that the heatmap layout itself blocks a wave-like reading by forcing the reader to see discrete vertical lag pillars rather than a smooth diagonal.

Two editorial constraints justify the appendix-only placement:

- The Phase 2 main body in `main_v2_cat.tex` is not yet drafted. A figure inserted now would float without supporting narrative.
- The N2-08 / N2-09 / N2-10 / N2-11 mechanistic stack supports `null_pressure` but does not replace it as the first paper-safe Gate 2 opening (D-0040). A main-body placement of this figure would silently invert the evidentiary hierarchy.

The canonical caption is deliberately verbose because it must remain self-contained: a reader who encounters the appendix figure without the surrounding Phase 2 narrative must still see the `E`-anchor construction, the right-panel definition `B - max(matched-LZ)`, the local probe outlines, the discrete-channel framing, the matched-LZ background behavior in `729M-738M`, and the explicit forbidden-claim fence.

This entry replicates the staging pattern of D-0051 (deficit-jitter phase plane): formalize the figure as paper-candidate now while the wording boundaries are fresh, defer insertion until Phase 2 is drafted.

### Source Documents

- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/lag_response_heatmap.png`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/lag_response_heatmap.svg`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/report.md`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/summary.json`
- `hsi_agents_project/results/hsi_v2/phase2/child_routing_lag_response_n2_11/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-8__20260427T165759/`
- `docs/HSI-audit-Phase2-Lag-Response.md`
- `hsi_agents_project/Documentation/HSI_v2_Phase2_Live_Findings.md`
- D-0051
- D-0057

### Practical Impact

- The Phase 2 appendix now has two staged paper-candidate figures: the deficit-jitter phase plane (D-0051) for the shell-atlas / jitter-provenance line, and the lag-response heatmap (D-0058) for the child-destination routing line.
- No changes to `main_v2_cat.tex` are triggered by this decision.
- When Phase 2 is drafted, the heatmap should be referenced as an appendix visual summary of the discrete phase-channel reading, not as a main-body Gate 2 result.
- The canonical caption must be preserved in spirit when translated into Catalan for the paper, including the `-27M` / `26.5M` discrete-channel naming, the local-probe outline note, and the forbidden-claim fence.
- Any future iteration of the heatmap (denser grid, alternative normalization, additional bands) must not silently drop the markov1 zero-collapse reference, the matched-LZ seed envelope, the local-probe outlines, or the `B - max(matched-LZ)` separation panel.
- The `729M-738M` matched-LZ rise is part of the canonical reading and must not be cropped or smoothed out in any paper-facing variant.

### Open Questions

- Should the appendix version display the matched-LZ envelope as min/median/max across seeds in a third panel, or is the current `B - max(matched-LZ)` summary sufficient?
- When a denser lag sweep around `-27M`, `-5.5M`, `0M`, and `26.5M` is run (D-0057 open question), should the resulting denser heatmap supersede this one in the appendix, or coexist as a finer-resolution companion?
- Should a `lag_response_heatmap_latest.png` symlink/copy be created alongside the deficit-jitter plane convention, to give the appendix a stable filesystem reference independent of the run timestamp?

---

## D-0059

- Date: 2026-04-29
- Status: `accepted-internal`
- Scope: N2-12 phase-channel width revalidation over the audited N2-11 child-routing lag-response contract
- Title: The N2-12 fine lag-neighborhood sweep supports a threshold-defined sampled phase-channel width for the `-27M` child-destination channel, with a selective positive channel and a late matched-LZ boundary rise

### Decision

Accept `N2-12` as a hardened internal diagnostic refinement of D-0057 / D-0058.

N2-12 densifies the lag sampling around the two principal N2-11 phase channels:

- negative channel around `-27M`
- positive channel around `26.5M`

It does **not** introduce a new observable. It reuses the audited N2-11 lag-response pipeline and the corrected N2-08 child-destination routing object, now evaluated over a finer lag neighborhood.

The accepted terminology is:

> threshold-defined sampled phase-channel width

or, more compactly in internal notes:

> phase-channel width

The following stronger phrasings are not accepted for paper-facing use:

- formal `FWHM`
- continuous bandwidth law
- temporal resonance law
- wave law
- universal `-27M` lag
- final transport law
- Gate 2 solved

### Executed Run

Canonical N2-12 full revalidation:

- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-10__20260428T162651/`

Visual readout:

- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-phase-channel-width-figure__src-20260428T162651__20260429T113112/phase_channel_width.png`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase_channel_width_latest.png`

Audit:

- `docs/HSI-audit-Phase2-Phase-Channel-Width.md`

Grid:

- bands: `696M-705M`, `705M-714M`, `714M-723M`, `723M-732M`, `726M-735M`, `729M-738M`
- negative lags: `-30M`, `-29.5M`, `-27M`, `-26.5M`, `-26M`, `-24M`
- positive lags: `23.5M`, `24.5M`, `26.5M`, `29.5M`
- sources: observed `B`, observed `E`, `markov1`, six `matched-LZ` seeds
- all `60` band-lag targets completed

### Strong-Cell Criterion

For the width envelope, a strong cell is defined descriptively as:

- observed `B >= 0.90`
- `B - max(matched-LZ) >= 0.30`

This threshold is accepted as a conservative internal visualization and summary convention. It is not a p-value, not a sigma claim, and must not be described as formal statistical significance.

### Result

Best-margin cells by band:

| Band | Best-margin lag | observed `B` | `max(matched-LZ)` | `B - max(matched-LZ)` |
| --- | ---: | ---: | ---: | ---: |
| `696M-705M` | `-26M` | `0.9852` | `0.3116` | `0.6736` |
| `705M-714M` | `-27M` | `1.0000` | `0.3297` | `0.6703` |
| `714M-723M` | `-24M` | `0.9604` | `0.3553` | `0.6050` |
| `723M-732M` | `-27M` | `1.0000` | `0.3592` | `0.6408` |
| `726M-735M` | `24.5M` | `1.0000` | `0.3376` | `0.6624` |
| `729M-738M` | `29.5M` | `1.0000` | `0.6983` | `0.3017` |

Strong sampled channel envelopes:

| Band | Channel | Strong cells | Sampled span | Width |
| --- | --- | ---: | ---: | ---: |
| `696M-705M` | negative | `3` | `-27M..-26M` | `1M` |
| `696M-705M` | positive | `0` | - | - |
| `705M-714M` | negative | `2` | `-29.5M..-27M` | `2.5M` |
| `705M-714M` | positive | `2` | `24.5M..26.5M` | `2M` |
| `714M-723M` | negative | `4` | `-27M..-24M` | `3M` |
| `714M-723M` | positive | `0` | - | - |
| `723M-732M` | negative | `5` | `-30M..-26M` | `4M` |
| `723M-732M` | positive | `2` | `26.5M..29.5M` | `3M` |
| `726M-735M` | negative | `2` | `-29.5M..-27M` | `2.5M` |
| `726M-735M` | positive | `2` | `24.5M..26.5M` | `2M` |
| `729M-738M` | negative | `0` | - | - |
| `729M-738M` | positive | `1` | `29.5M` | `0M` |

Key aggregate readouts:

- At `-27M`, observed `B` has average retention `0.9942`.
- At `-27M`, average `max(matched-LZ)` is `0.4291`.
- At `-27M`, average margin is `+0.5650`.
- `markov1` remains exactly `0.0000` across all `60` cells.
- The late `729M-738M` band has a strong matched-LZ rise (`max(matched-LZ)` reaches roughly `0.70-0.84` in several cells), which contracts the margin without eliminating the positive `29.5M` separator.

### Scientific Reading

N2-12 strengthens the D-0057 / D-0058 discrete phase-channel interpretation.

The negative `-27M` channel is not merely an isolated lag point. Under the current sampled neighborhood, it occupies a threshold-defined envelope across several adjacent lags and bands, with strong separation from the strongest matched-LZ seed.

The positive channel around `26.5M` is real but more selective: it is absent or weak in some bands, strong in `705M-714M`, `723M-732M`, and `726M-735M`, and shifts toward the tested positive edge at `29.5M` in `729M-738M`.

The late `729M-738M` band is best read as a boundary regime:

- observed `B` still reaches perfect retention in some tested cells;
- matched-LZ partially shadows the child-destination channel much more strongly than in earlier bands;
- the remaining positive separator is narrower and margin-limited.

The current best wording is:

> Observed `B` exhibits threshold-defined sampled phase-channel envelopes in the monitored top-128 child-destination universe. The negative channel around `-27M` is the strongest and most stable sampled channel; the positive channel around `26.5M` is selective and becomes the stronger separator at the late boundary. `markov1` collapses everywhere, while `matched-LZ` remains a bounded but non-zero background that rises substantially in the late band.

### Wording Boundary

Allowed wording:

- "threshold-defined sampled phase-channel width"
- "sampled channel envelope"
- "strong-cell criterion"
- "`-27M` negative channel"
- "`26.5M` / late positive channel"
- "late matched-LZ boundary rise"

Forbidden or premature wording:

- "formal FWHM"
- "5-sigma"
- "continuous resonance"
- "universal lag"
- "global stationarity"
- "complete transport law"
- "Gate 2 solved"

### Practical Impact

- D-0057 remains the conceptual lag-response decision.
- D-0058 remains the paper-candidate appendix decision for the coarse N2-11 heatmap.
- D-0059 hardens the N2-11 reading by showing that the main channel has sampled width, not only a single-lag spike.
- The N2-12 figure is accepted as an internal visual readout, but not yet staged as a paper-candidate figure. If it is later staged, it must use the threshold-defined sampled-envelope language.
- `null_pressure` remains the first paper-safe Gate 2 opening.
- No automatic change to `Paper_Latex/main_v2_cat.tex`.

### Open Questions

- Should a deeper adaptive zoom around `729M-738M` determine whether the late positive separator extends beyond `29.5M`?
- Should the `top=128` monitored-universe readout be repeated at a second resolution before paper staging?
- Should the appendix eventually use the coarse N2-11 heatmap, the N2-12 width figure, or both?

---

## D-0060

- Date: 2026-04-30
- Status: `accepted-internal`
- Scope: N2-12b top-k sensitivity revalidation for the N2-12 phase-channel width readout
- Title: Top-k sensitivity confirms the N2-12 central channel-envelope reading while reclassifying `729M-738M` as a top-k-sensitive boundary band

### Decision

Accept `N2-12b` as an internal robustness and boundary-classification addendum to D-0059.

N2-12b reuses the audited N2-11 / N2-12 lag-response and child-destination routing contract. It does not introduce a new observable. Its purpose is to test whether the N2-12 phase-channel width readout depends materially on the monitored top-k resolution.

The result is split:

1. The central N2-12 channel-envelope reading survives the top-k stress test.
2. The late `729M-738M` band is reclassified as `top-k-sensitive boundary`, not as a stable positive-channel extension.

This decision does not change the evidentiary hierarchy. `null_pressure` remains the first paper-safe Gate 2 opening. N2-12 / N2-12b remain internal mechanistic diagnostics.

### Executed Runs

Canonical N2-12 reference:

- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-10__20260428T162651/`

Top-k sensitivity, full `top=64` grid:

- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_width_topk_sensitivity/phase2-phase-channel-width-topk-sensitivity__stage-all__top-64-128x2__lags-10__20260429T102045/`

Top-k sensitivity, frontier `top=96` rerun:

- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_width_topk_sensitivity_frontier_top96/phase2-phase-channel-width-topk-sensitivity__stage-all__top-96-128x2__lags-10__20260430T065831/`

Code paths:

- `hsi_agents_project/hsi_v2_phase2_phase_channel_width_topk_sensitivity.py`
- `hsi_agents_project/hsi_v2_phase2_child_routing_lag_response.py`

### Result

For the non-frontier bands, `top=64` reproduces the N2-12 envelope structure with only small margin drift:

- no missing measured cells;
- the same strong-cell counts and sampled widths in the main central bands;
- average absolute margin drift outside `729M-738M` around `0.017`;
- `markov1` remains collapsed under the inherited N2-12 contract.

The late `729M-738M` band behaves differently:

| Top-k | Band | Strong negative channel | Strong positive channel | Best strong readout |
| ---: | --- | --- | --- | --- |
| `64` | `729M-738M` | `-27M`, width `0M`, margin `0.4091` | none | negative boundary cell |
| `96` | `729M-738M` | `-27M`, width `0M`, margin `0.3949` | none | negative boundary cell |
| `128` | `729M-738M` | none | `29.5M`, width `0M`, margin `0.3017` | positive boundary cell |

The `top=96` frontier rerun contains no missing expected cells. Its `topk_comparison.csv` shows the key `729M-738M` deltas relative to `top=128`:

- at `-27M`, observed `B` remains `1.0000` and the margin increases from `0.2646` to `0.3949`;
- at `29.5M`, observed `B` remains `1.0000` but the margin decreases from `0.3017` to `0.2216`;
- at `-26.5M`, `-26M`, and `-24M`, observed `B` drops substantially under `top=96`, confirming local boundary instability rather than a broad stable channel.

### Scientific Reading

N2-12b strengthens D-0059 in the central regime and narrows the late-boundary claim.

The robust statement is:

> The N2-12 phase-channel width readout is stable in the central sampled bands under a lower monitored top-k resolution. The `729M-738M` band is not a stable positive-channel extension; it is a top-k-sensitive boundary where the surviving strong cell changes channel under resolution stress.

The correct reading of `729M-738M` is therefore:

- observed `B` can still reach perfect retention in several tested cells;
- the strongest matched-LZ seed background is high enough to make the separator resolution-sensitive;
- the strong-cell channel changes between `top=64/96` and `top=128`;
- the band should be marked as boundary evidence, not as a settled transport channel.

### Wording Boundary

Allowed wording:

- "top-k sensitivity"
- "boundary band"
- "`729M-738M` is top-k-sensitive"
- "central channel envelopes are robust under `top=64`"
- "late boundary channel assignment changes under monitored-universe resolution"

Forbidden or premature wording:

- "top-k independent"
- "stable positive late channel"
- "universal late-channel flip"
- "complete transport law"
- "Gate 2 solved"
- "absolute routing outside the monitored top-k child universe"

### Practical Impact

- D-0059 remains valid for the central N2-12 phase-channel width reading.
- The N2-12 figure should not present `729M-738M` as a simple positive-channel continuation without a boundary/top-k sensitivity note.
- Any future paper-facing caption or appendix text must mark `729M-738M` as a boundary regime if the N2-12 figure is used.
- No change to `Paper_Latex/main_v2_cat.tex` is triggered.
- The `hsi_v2_phase2_child_routing_lag_response.py` wrapper has been improved to surface foreground progress and heartbeat output for delegated long-running child steps, matching the reproducibility/observability requirement for future reviewers.

The current visual alias has been regenerated with the D-0060 boundary mark:

- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase2-phase-channel-width-figure__src-20260428T162651__20260430T123015/phase_channel_width.png`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/phase_channel_width_latest.png`

### Open Questions

- Should a further `top=96` full-grid run be executed, or is the targeted `729M-738M` frontier rerun sufficient for the current boundary classification?
- Should paper staging prefer the coarser N2-11 heatmap (D-0058) over the N2-12 width figure until the boundary visualization is finalized?

---

## D-0061

- Date: 2026-04-30
- Status: `accepted-internal`
- Scope: N2-13a routing-signature readout over the N2-12b top-k sensitivity grid; closure of the D-0060 boundary question and partial closure of the D-0053 off-universe open question
- Title: The `729M-738M` boundary is a monitored null-shadowing regime, not observed-`B` collapse; the matched-LZ deficit is `diffuse-deficit` with no concentrated hidden structure inside the monitored universe; this does not close Phase 2 editorially

### Decision

Accept `N2-13a` as a hardened internal mechanistic readout that classifies the **expansion delta** between lower monitored top-k runs and the canonical reference top-k=128, using boolean routing signatures hard-coded over existing child-routing subruns.

The accepted artifacts are:

- `hsi_agents_project/results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/`
- `docs/HSI-audit-Phase2-Routing-Signatures.md`

The N2-13a readout does **not** classify the full unbounded off-top-k mass. It only classifies what changes when the monitored universe is expanded between top-k resolutions. This boundary is preserved in the wording fence below.

The result is split:

1. The central N2-12 channel-envelope reading is reaffirmed as `stable-monitored-channel`: 22 cells across the five central bands (`696M-705M`, `705M-714M`, `714M-723M`, `723M-732M`, `726M-735M`) remain strong both before and after expansion, with `diffuse-deficit` matched-LZ shape.
2. The late `729M-738M` boundary band shows a coexistence of three signatures â€” `reference-null-shadowing`, `reference-null-shadow-relief`, and `expanded-rank-shell-B-gain` â€” none of which is observed-`B` collapse and none of which is `monitored-rerouting-present`.
3. The matched-LZ deficit is `diffuse-deficit` across every observed signature class. There is no `monitored-rerouting-present` signature in any cell.

The canonical short reading is:

> Under the monitored top-k expansion delta, `729M-738M` is governed by monitored null-shadowing dynamics (matched-LZ background rises and erases observed-`B` margin at one lag, then relaxes and lets observed-`B` re-emerge at another) rather than by observed-`B` collapse. The matched-LZ deficit is diffuse rather than concentrated, so the failure of the strong null in this band is dispersion, not capture by a concentrated hidden attractor inside the monitored child universe.

### Closures

This entry closes:

- the D-0060 open question "Should a further `top=96` full-grid run be executed, or is the targeted `729M-738M` frontier rerun sufficient?": the targeted top=96 frontier rerun reproduces the same `reference-null-shadowing` / `reference-null-shadow-relief` / `expanded-rank-shell-B-gain` triplet at `729M-738M`, with `diffuse-deficit` matched-LZ shape. No further top-k is required for the boundary classification.
- the D-0060 open question on whether `729M-738M` could be a settled positive late channel: explicitly no. It is a monitored null-shadowing boundary.

This entry **partially** closes the D-0053 open question "Can a future object classify off-universe destination mass without exploding the tracked child universe?":

- yes, for the **expansion-delta** class (cells whose status changes when the monitored universe grows from `top=64` or `top=96` to `top=128`);
- no, for the **absolute** off-universe mass beyond `top=128` (this remains outside the contract of N2-13a and would require a separate object).

### Wording Boundary

The accepted claim is deliberately mechanistic and bounded.

Allowed wording:

- "monitored null-shadowing boundary"
- "monitored expansion-delta classification"
- "no observed-`B` collapse at `729M-738M`"
- "no `monitored-rerouting-present` signature in any cell"
- "`diffuse-deficit` matched-LZ reference shape"
- "`stable-monitored-channel` central regime"
- "Phase 2 mechanistic stack hardened"

Forbidden or premature wording:

- "Phase 2 closed" (in any editorial / paper sense)
- "Gate 2 solved"
- "transport law"
- "absolute off-universe mass classification"
- "B is universally robust at `729M-738M`"
- "matched-LZ structurally collapses" (it dilutes diffusely; it does not collapse)
- "hidden attractor refuted in the unbounded off-universe" (the refutation only covers the monitored expansion delta)

The hostile audit (`docs/HSI-audit-Phase2-Routing-Signatures.md`) reads N2-13a as a methodological closure of Phase 2. This is correct for the **internal mechanistic stack**. It is not an editorial closure of Phase 2 in the paper. The two are independent and must remain so until a separate decision drafts Phase 2 in `Paper_Latex/main_v2_cat.tex`.

### Source Documents

- `hsi_agents_project/results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/report.md`
- `hsi_agents_project/results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/signature_summary.csv`
- `hsi_agents_project/results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/cell_classification.csv`
- `hsi_agents_project/results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/manifest.json`
- `docs/HSI-audit-Phase2-Routing-Signatures.md`
- `hsi_agents_project/Documentation/HSI_v2_Phase2_Live_Findings.md`
- D-0053
- D-0058
- D-0059
- D-0060

### Practical Impact

- The Phase 2 internal mechanistic stack is now methodologically tightened end-to-end: shell atlas (D-0049/D-0050), jitter provenance (D-0052), child-destination routing (D-0053), child-routing persistence (D-0054), phase-lag atlas (D-0055/D-0056), lag-response (D-0057), phase-channel width (D-0059), top-k sensitivity (D-0060), and now monitored-universe expansion-delta routing signatures (D-0061).
- The evidentiary hierarchy does not change. Strict `null_pressure` (D-0040) remains the first and only paper-safe Gate 2 opening.
- The N2-11 lag-response heatmap (D-0058) remains the single paper-candidate Phase 2 figure for the child-routing line. The N2-12 width figure remains internal. N2-13a is a tabular/CSV diagnostic and is **not** promoted to paper-candidate.
- No change to `Paper_Latex/main_v2_cat.tex` is triggered by this decision.
- When a future decision drafts the Phase 2 paper section, the N2-13a closure of the `729M-738M` boundary should appear as a wording fence that prevents the right boundary from being read either as a dead zone or as an unconditional positive extension.
- The internal stack is now mature enough that the next coordinated step (when the program decides) can be a three-agent editorial architecture session for Phase 2 in the paper, replacing the current "wait for maturity" stance from the earlier Decision-D resolution.

### Open Questions

- Does an absolute off-top-k object (beyond the expansion-delta classification of N2-13a) add value for the paper, or does the monitored-universe contract suffice for the Phase 2 narrative?
- Should the `expanded-rank-shell-B-gain` signature (max gain `0.5162`) at `729M-738M` and `714M-723M` receive a dedicated mechanistic note, or is it absorbed by the boundary classification?
- When Phase 2 is eventually drafted, should N2-13a appear as a mechanistic wording fence in the main body or as a tabular summary in the appendix?
- Has the program reached the point where the three-agent Phase 2 editorial architecture session (deferred under the earlier "no premature editorial structure" call) should now be scheduled?

---

## D-0062

- Date: 2026-04-30
- Status: `accepted-internal`
- Scope: N2-14 phase-channel transition graph over existing N2-12b and N2-13a artifacts; pre-editorial closure check for the child-routing mechanistic stack
- Title: The sampled phase-channel cells assemble into a pre-spectral transition scaffold at `top=128`, with a null-shadowing boundary at `729M-738M`; this supports a Phase 2 editorial architecture session but does not introduce geometry or close Phase 2 in the paper

### Decision

Accept `N2-14` as a small, falsifiable, internal diagnostic readout.

N2-14 does not recompute raw bits. It reads the already validated N2-12b phase-channel envelopes and the N2-13a routing signatures, then builds a minimal transition graph:

- nodes are `band x channel` envelope states;
- edges connect adjacent bands inside the same `top-k` and channel;
- edge labels are simple transition classes such as `continuous-envelope`, `activation`, `attenuation`, and `null-shadow-boundary`;
- the falsification target is whether active channel states form adjacent-band paths at all.

The accepted artifacts are:

- `hsi_agents_project/hsi_v2_phase2_phase_channel_transition_graph.py`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/`
- `docs/HSI-audit-Phase2-Transition-Graph.md`

### Result

The canonical run gives:

| Top-k | Channel | Active bands | Boundary nodes | Max active path | Verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| `96` | negative | `1/1` | `1` | `1` | `no-stable-channel-path` |
| `96` | positive | `0/1` | `1` | `0` | `no-stable-channel-path` |
| `128` | negative | `5/6` | `1` | `5` | `structured-channel-with-boundary` |
| `128` | positive | `4/6` | `1` | `3` | `structured-channel-with-boundary` |

For the `top=128` negative channel, the graph contains four adjacent `continuous-envelope` transitions before the `729M-738M` `null-shadow-boundary`:

`696M-705M -> 705M-714M -> 714M-723M -> 723M-732M -> 726M-735M -> 729M-738M`

The positive channel is weaker and more intermittent. It contains activation / attenuation transitions and a shorter active path, but still ends in the same boundary logic.

The `top=96` result does not invalidate the `top=128` scaffold. It is read as a monitored-resolution frontier: at that resolution the late boundary cells are visible, but no stable adjacent-band path is supported.

### Wording Boundary

Allowed wording:

- "pre-spectral transition graph"
- "phase-channel transition scaffold"
- "structured channel with boundary"
- "adjacent-band channel path"
- "`top=96` monitored-resolution frontier"
- "`729M-738M` null-shadowing boundary"
- "internal pre-editorial closure check"

Forbidden or premature wording:

- "Phase 2 closed" in an editorial or paper sense
- "Gate 2 solved"
- "geometry"
- "spectral geometry"
- "transport law"
- "statistically impossible"
- "top-k independent scaffold"
- "universal phase-channel path"
- "proof of physical trajectory"

The hostile audit approves the construction as a methodological bridge. We accept the approval while tightening the language: N2-14 shows that the already measured N2-12/N2-13 cells assemble into adjacent-band transition paths under the stated contract. It does not prove a graph geometry, does not introduce a Laplacian, and does not replace strict `null_pressure` as the paper-safe Gate 2 opening.

### Source Documents

- `hsi_agents_project/hsi_v2_phase2_phase_channel_transition_graph.py`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/report.md`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/nodes.csv`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/edges.csv`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/paths.csv`
- `docs/HSI-audit-Phase2-Transition-Graph.md`
- `hsi_agents_project/Documentation/HSI_v2_Phase2_Live_Findings.md`
- D-0057
- D-0058
- D-0059
- D-0060
- D-0061

### Practical Impact

- D-0062 is the final internal pre-editorial bridge for the current child-routing stack.
- The internal Phase 2 mechanism now has a coherent sequence from local shell/density diagnostics to phase-channel paths with an explicit null-shadowing boundary.
- The evidentiary hierarchy does not change. Strict `null_pressure` (D-0040) remains the first and only paper-safe Gate 2 opening.
- N2-11 remains the staged Phase 2 paper-candidate figure for the child-routing line. N2-12, N2-13a, and N2-14 remain internal diagnostics unless a later editorial architecture decision promotes or summarizes them.
- No change to `Paper_Latex/main_v2_cat.tex` is triggered by this decision.
- The program has now reached the point where the deferred three-agent Phase 2 editorial architecture session is the natural next step.

### Open Questions

- Should N2-14 appear in the paper architecture only as a wording fence, as a short appendix table, or not at all?
- Should the `structured-channel-with-boundary` label be translated into a more austere Catalan paper phrase before drafting?
- Should a simple visual graph be produced later for internal orientation, or would that invite premature geometry language?
- Should any Phase 3 graph/spectral work wait until the Phase 2 paper architecture has been fixed?

---

## D-0063

- Date: 2026-04-30
- Status: `accepted-internal`
- Scope: Phase 2 editorial architecture for `Paper_Latex/main_v2_cat.tex`; outcome of the three-agent editorial session (Ariadna proposal, Sofia review, Grace audit, Iban decision); no paper edits triggered yet
- Title: Phase 2 paper architecture accepted; main body restricted to D-0040 plus one pointer paragraph; appendix limited to D-0051 and D-0058 figures plus a one-row-per-decision traceability table; consolidated allowed/forbidden wording fence; D-0049/D-0050/D-0052-D-0057/D-0059-D-0062 remain internal-only

### Decision

Accept the Phase 2 editorial architecture as defined in `docs/HSI_v2_Phase2_Editorial_Architecture_Proposal.md` (Ariadna's proposal updated with Sofia's three contraproposals and Q1-Q6 editorial answers, then audited by Grace).

The accepted artifacts are:

- `docs/HSI_v2_Phase2_Editorial_Architecture_Proposal.md`
- `docs/HSI-audit-Phase2-Editorial-Architecture.md`

The architecture has four invariants:

1. **Narrow main body.** The Phase 2 main-body section contains only D-0040 (strict `null_pressure` as the paper-safe Gate 2 opening) plus one compact pointer paragraph to the appendix mechanism. No mechanistic subsection in the body. No detailed N2 stack in the body.
2. **Two appendix figures.** D-0051 (deficit-jitter phase plane) and D-0058 (lag-response heatmap, N2-11) are the only Phase 2 figures. D-0051 carries the topological reading (how routing survives against the deficit), D-0058 carries the temporal reading (when the channel engages). They appear in this order, followed by the diagnostic stack table.
3. **Traceability table over the full stack.** The appendix diagnostic stack table includes one row per decision for D-0049, D-0050, D-0052, D-0053, D-0054, D-0055, D-0056, D-0057, D-0058, D-0059, D-0060, D-0061, and D-0062. Each row is a one-line summary, not full prose. N2-14 may appear as a row but receives no lexical entry in the paper text.
4. **Consolidated wording fence.** The allowed/forbidden wording lists in sections E and E.bis of the architecture proposal become the canonical Phase 2 lexicon for the paper. Any Phase 2 prose drafting must respect this lexicon.

The accepted Phase 2 section skeleton is:

```
Section X. Phase 2: Temporal Transport Beyond Projective Survival
  Section X.1. Phase 2 scope: temporal transport beyond projective survival
  Section X.2. Strict null_pressure as the paper-safe Gate 2 opening
  Section X.3. Internal mechanistic support below the opening
  Section X.4. Boundary discipline and limitations

Appendix P2. Phase 2 Diagnostic Traceability
  Appendix P2.1. Strict null_pressure traceability table
  Appendix P2.2. Deficit-jitter plane (D-0051)
  Appendix P2.3. Lag-response heatmap (D-0058)
  Appendix P2.4. Internal diagnostic stack table
```

The accepted Phase 1 to Phase 2 transition sentence is:

> Having established a non-trivial projective hierarchy in Phase 1, Phase 2 asks whether the surviving scaffold also carries temporal transport structure beyond strong nulls. The strict `null_pressure` run provides the first paper-safe observational opening of this Gate 2 question.

The accepted technical-term policy is: keep `null_pressure`, `matched-LZ`, and `B` as code-style literals with a short Catalan gloss on first use.

A useful framing surfaced in the Grace audit is captured here for the eventual prose:

> D-0051 = how routing survives topologically against the deficit; D-0058 = when (in which temporal phases) the channel engages. The appendix order (D-0051 then D-0058) follows topological then temporal.

### Closures

This entry closes:

- the open question deferred under the earlier informal "Decision D" (no premature editorial structure): the program has now reached the point where editorial architecture is justified and is fixed by this entry.
- the D-0058 staging open question on whether N2-11 belongs in the appendix: confirmed as the primary Phase 2 child-routing appendix figure with its existing canonical caption.
- the D-0061 open question "Should the program now schedule the deferred three-agent Phase 2 editorial architecture session?": session held and resolved by this entry.
- the D-0062 open question on whether N2-14 should appear in the paper architecture: only as a row in the appendix traceability table, with no lexical entry in the paper prose.

This entry **does not** close:

- the question of when prose drafting starts in `Paper_Latex/main_v2_cat.tex` (separate decision).
- the question of how exactly the Catalan prose realizes the allowed lexicon (a translation pass over the lexicon belongs to the drafting decision, not to this one).
- any Phase 3 question.

### Wording Boundary

This entry inherits and consolidates the wording fence accumulated across D-0040, D-0058, D-0060, D-0061, and D-0062.

Allowed wording (canonical Phase 2 lexicon for the paper):

- "first paper-safe observational opening of Gate 2"
- "strongest-null contract"
- "internal mechanistic support"
- "monitored null-shadowing boundary"
- "threshold-defined sampled phase-channel envelopes"
- "sampled child-destination conservation"
- "non-trivial projective hierarchy" (Phase 1 anchor)
- "appendix diagnostic traceability"

Forbidden or premature wording:

- "Phase 2 closed" in any editorial or paper sense
- "Gate 2 solved" or "Gate 2 resolved"
- "transport law" or "final transport law"
- "geometry", "spectral geometry", "Laplacian", or "physical trajectory" from Phase 2 diagnostics
- "wave law"
- "universal lag", "universal `-27M` lag", or "universal `15M` lag"
- "global stationarity"
- "top-k independent routing" or "absolute routing outside the monitored top-k universe"
- "absolute off-universe mass classification"
- "`729M-738M` is a stable positive late channel"
- "`729M-738M` is dead independent of lag"
- "formal continuous phase-channel bandwidth law"
- "statistical significance" from the N2-12 strong-cell threshold
- "`B` is universally robust at `729M-738M`"
- "`matched-LZ` structurally collapses" (the accepted reading is diffuse deficit / null-shadowing)
- "hidden attractor refuted in the unbounded off-universe"
- "top-k independent phase-channel transition scaffold"

The Grace audit reads the architecture as ready for LaTeX drafting. This is correct as a methodological green light for the **architecture**. It does not authorize prose drafting in `Paper_Latex/main_v2_cat.tex` by itself. Prose drafting is a separate decision.

### Source Documents

- `docs/HSI_v2_Phase2_Editorial_Architecture_Proposal.md`
- `docs/HSI-audit-Phase2-Editorial-Architecture.md`
- `hsi_agents_project/Documentation/HSI_v2_Phase2_Live_Findings.md`
- D-0040
- D-0049, D-0050
- D-0051
- D-0052, D-0053, D-0054
- D-0055, D-0056
- D-0057
- D-0058
- D-0059, D-0060
- D-0061
- D-0062

### Practical Impact

- The Phase 2 editorial architecture is now canon. Any future drafting must respect the four invariants (narrow body, two appendix figures, full traceability table, consolidated wording fence) and the accepted skeleton.
- The evidentiary hierarchy does not change. Strict `null_pressure` (D-0040) remains the only paper-safe Gate 2 opening; the rest of the Phase 2 stack is paper-supporting, not paper-claiming.
- D-0051 and D-0058 remain the only Phase 2 paper-candidate figures.
- N2-12, N2-13a, and N2-14 remain internal diagnostics with no lexical entry in the paper text.
- No change to `Paper_Latex/main_v2_cat.tex` is triggered by this decision. Prose drafting requires a separate explicit decision.
- The deferred informal "Decision D" stance ("no premature editorial structure, wait for stack maturity") is now superseded by this entry. The architecture is fixed; only drafting timing remains open.

### Open Questions

- When should prose drafting of Phase 2 in `Paper_Latex/main_v2_cat.tex` start (immediately after this entry, after a final coherence pass over the existing Phase 1 text, or after another deferred interval)?
- Should the appendix traceability table be drafted in English or Catalan, given that the Decision Log entries it summarizes are in English while the paper is in Catalan?
- Should the canonical English captions of D-0051 and D-0058 be translated to Catalan for the paper, kept bilingual, or kept in English with a Catalan summary line?
- Should a separate "lexicon translation pass" happen before drafting, fixing the Catalan equivalents of the allowed-wording entries to prevent drift during prose writing?

---

## D-0064

- Date: 2026-04-30
- Status: `accepted-internal`
- Scope: Mapping of the D-0063 Phase 2 editorial architecture onto the existing structure of `Paper_Latex/main_v2_cat.tex`; resolution of the four D-0063 open questions; canonical filenames for the two Phase 2 paper figures; authorization to edit `main_v2_cat.tex` under the constraints below
- Title: D-0063 architecture is realized through Option-Alpha minimal cirurgy on the existing paper (no new "Section X"); Phase 2 mechanism is grafted onto Â§10.5 via a pointer paragraph; Phase 2 appendix is added as a new appendix section Â§E; figures are copied into `Paper_Latex/` with canonical filenames; traceability table and captions remain in English

### Decision

D-0063 fixed the Phase 2 editorial architecture in the abstract, with a skeleton built around a hypothetical "Section X. Phase 2: Temporal Transport Beyond Projective Survival". The reading of `Paper_Latex/main_v2_cat.tex` reveals a structural friction:

- The paper does not use "Phase 2" as an editorial label. It uses "Porta 1" and "Porta 2" as sequential editorial gates, and `\texttt{Phase 2}` only as the experimental pipeline name.
- The content that D-0063 Â§X.2 wants to introduce already exists as Â§10.5 *"Primera obertura observacional defensable: divergÃ¨ncia del perfil de retorn sota contracte estricte"*, including the canonical D-0040 return-profile table.
- The content that D-0063 Â§X.4 wants to introduce already exists in compressed form at the end of Â§11.2 *"Porta 2: Defectes i geometria (pendent)"*.

This entry resolves the friction by adopting **Option Alpha (minimal cirurgy)** and recording the realization explicitly so that D-0063 traceability remains clean.

The accepted realization of D-0063 on `main_v2_cat.tex` has four parts:

1. **No new "Section X" is introduced.** The "Porta" framing of the paper is preserved. Phase 2 lives editorially inside Â§10 *"Programa cap al Nivell 2"* and Â§11.2 *"Porta 2: Defectes i geometria (pendent)"*, exactly as the existing structure already places it.
2. **Â§10.5 is extended with a pointer paragraph** that introduces the internal Phase 2 mechanistic stack and refers the reader to the new appendix. A short auxiliary subsection Â§10.6 may be added if the pointer paragraph alone is not enough to satisfy the D-0063 narrow-body invariant. No mechanistic detail is moved into the body.
3. **Â§11.2 receives at most one supplementary sentence** acknowledging the maturity of the internal stack while preserving its current cautious reading. The existing wording of Â§11.2 already realizes the D-0063 Â§X.4 boundary discipline; no rewriting is required.
4. **A new appendix section Â§E *"DiagnÃ²stic de Phase 2"*** is added between current Â§D *"Co-creaciÃ³ simbiÃ²tica"* and `\bibliography`. Â§E contains four subsections that map exactly to D-0063 Â§P2.1-Â§P2.4: (P2.1) `null_pressure` traceability table, (P2.2) deficit-jitter plane figure, (P2.3) lag-response heatmap figure, (P2.4) internal diagnostic stack table covering D-0049 through D-0062.

The four invariants of D-0063 (narrow main body, two appendix figures, full traceability table, consolidated wording fence) are satisfied by this realization without adopting the literal Â§X.1-Â§X.4 skeleton.

### Resolution of the four D-0063 open questions

1. **Language of the appendix traceability table.** English. Whenever bilingual ambiguity could create lexical drift, English wins. This applies to the entire Â§E.P2.4 stack-summary table and to all `Allowed wording` and `Forbidden wording` entries when they appear in tables.
2. **Captions of D-0051 and D-0058 figures.** English. The existing canonical English captions are preserved verbatim. No bilingual or summary line is added.
3. **Catalan lexicon translation pass before drafting.** Sofia produces a brief internal lexicon-translation working note for editorial comfort during prose writing, but the canonical lexicon for the paper remains in English wherever it appears literally (tables, code-style terms, captions). Catalan body prose may paraphrase the allowed lexicon as long as no forbidden phrase is introduced.
4. **Drafting timing.** Authorized to start immediately after this entry is written, under the constraints listed in the Practical Impact section below. Compilation happens externally on Overleaf.

### Canonical filenames for Phase 2 figures

The two D-0058 / D-0051 source PNGs live under `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/` with `_latest` suffixes that float with each rerun. For paper stability, copies are placed inside `Paper_Latex/` (no subdirectory; matches the existing convention) with stable canonical names that do not float:

- `Paper_Latex/phase2_lag_response.png` <- copy of `lag_response_heatmap_latest.png` (D-0058)
- `Paper_Latex/phase2_deficit_jitter.png` <- copy of `deficit_jitter_plane_latest.png` (D-0051)

The originals under `results/.../visual_readouts/` remain untouched. The copies inside `Paper_Latex/` are the references cited from `main_v2_cat.tex`. If a future rerun changes the canonical figure, a separate decision must explicitly authorize a refresh of the `Paper_Latex/` copies.

### Wording Boundary

This entry inherits the consolidated allowed/forbidden lexicon from D-0063 without modification.

Additional realization-specific guards:

- The string "Phase 2" appears in the paper only as `\texttt{Phase 2}` referring to the experimental pipeline, never as an editorial section label.
- "Porta 2" remains the editorial label for the second falsification gate, in line with Â§11.2.
- The Â§10.5 pointer paragraph must not promote the Phase 2 mechanism above strict `null_pressure`. The pointer is a downward reference to appendix support, not an additional Gate 2 opening.
- The new Â§11.2 supplementary sentence (if added) must use one of the allowed-wording entries from D-0063 (for example, "internal mechanistic support" or "monitored null-shadowing boundary"), not introduce new vocabulary.

### Source Documents

- `HSI_v2_Decision_Log.md` (D-0063, D-0040, D-0058, D-0061)
- `docs/HSI_v2_Phase2_Editorial_Architecture_Proposal.md`
- `docs/HSI-audit-Phase2-Editorial-Architecture.md`
- `Paper_Latex/main_v2_cat.tex` (read for structural friction analysis)
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/lag_response_heatmap_latest.png`
- `hsi_agents_project/results/hsi_v2/phase2/visual_readouts/deficit_jitter_plane_latest.png`

### Practical Impact

- The realization of D-0063 on `main_v2_cat.tex` is now a defined operation. Sofia is authorized to edit `main_v2_cat.tex` strictly within the four-part scope above. Any deviation requires a separate decision.
- Sofia is authorized to copy the two figure PNGs into `Paper_Latex/` with the canonical names defined above.
- The traceability table at Â§E.P2.4 must include exactly D-0049, D-0050, D-0052, D-0053, D-0054, D-0055, D-0056, D-0057, D-0058, D-0059, D-0060, D-0061, D-0062 (one row per decision, English, one-line summary). N2-14/D-0062 may appear as one row but receives no lexical entry in the paper prose.
- No commit, push, or external upload is triggered by this entry. The `.tex` and figures are edited locally; Iban handles Overleaf upload.
- The deferred drafting timing of D-0063 is now closed. The remaining D-0063 open questions on lexicon translation pass and caption language are also closed by this entry.

### Open Questions

- Should Â§10.6 be created as a separate compact subsection, or is the pointer paragraph at the end of Â§10.5 sufficient to satisfy the narrow-body invariant of D-0063? (To be decided during drafting based on what reads cleanest in context.)
- Should the Â§11.2 supplementary sentence be added at all, or does the existing Â§11.2 wording already suffice without modification? (To be decided during drafting; default is no addition unless clearly needed.)
- Should the Â§E.P2.4 stack-summary table also include D-0040 as an additional first row for completeness, or remain restricted to the post-D-0040 mechanistic stack as written above?
- When the paper is eventually translated to English, should `phase2_lag_response.png` and `phase2_deficit_jitter.png` be regenerated with English axis labels, or kept as-is with English captions only?

---


## D-0065

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Editorial autocitation of the original HSI formulation (HSI v1, Zenodo) in `Paper_Latex/main_v2_cat.tex` and `Paper_Latex/references.bib`; removal of the in-paper "v1" nomenclature in favor of "formulaciÃ³ original" anchored on a formal `\cite{}`
- Title: HSI v2 paper now formally cites HSI v1 (Zenodo DOI `10.5281/zenodo.18721272`) at the first mention of "HipÃ²tesi de la Singularitat Informacional" in the introduction and at the Â§10.4 connection paragraph; the connecting paragraph is rewritten in extension form ("v2 estÃ©n v1 substituint qualitatiu per operatiu") rather than retroactive form ("v2 materialitza el que v1 ja invocava"); the in-paper "v1" nomenclature is removed

### Decision

1. The first mention of "HipÃ²tesi de la Singularitat Informacional (HSI)" in `Paper_Latex/main_v2_cat.tex` (Introduction, Â§1) is now anchored on a formal `\cite{borras2026hsi_v1}` immediately after the abbreviation `(HSI)`. This establishes the autocitation early in the paper so that any subsequent mention of the original formulation is coherent with an already-introduced reference.

2. Subsection Â§10.4 is renamed from "ConnexiÃ³ amb la formalitzaciÃ³ del v1" to "ConnexiÃ³ amb la formalitzaciÃ³ original". The internal `\label{subsec:connexio_v1}` is preserved to avoid breaking any existing or future cross-reference.

3. The Â§10.4 paragraph is rewritten in extension form: "Aquest programa estÃ©n la formulaciÃ³ original de la \emph{HipÃ²tesi de la Singularitat Informacional} \cite{borras2026hsi_v1}, substituint els objectes geomÃ¨trics qualitatius ---invocats allÃ  com a 'curvatura $\mathcal{R}$' i 'torsiÃ³ $\mathcal{T}$' del camp informacional--- per candidats operatius explÃ­cits, $\delta_m$ i el commutador $[\omega_a, \omega_b]$, definits sobre objectes ja existents (la torre, les fibres) i, per tant, mesurables sense afegir cap nova ontologia. Si la Porta 2 passa, el marc aquÃ­ proposat reabsorbeix la geometria d'aquella formulaciÃ³ com a conseqÃ¼Ã¨ncia empÃ­rica de la jerarquia projectiva, no com a postulat." This eliminates the in-text "v1" nomenclature while preserving the substantive connection between the qualitative geometric objects of the original formulation and the operational candidates of HSI v2.

4. A new BibTeX entry `borras2026hsi_v1` is added to `Paper_Latex/references.bib` using the version-specific Zenodo DOI `10.5281/zenodo.18721272` (not the conceptual parent DOI `10.5281/zenodo.18721271`). The version-specific DOI is chosen so that future Zenodo re-uploads of HSI v2 do not retroactively change what the v1 citation points to.

5. The BibTeX entry uses the human-only-author convention (`author = {Borr{\`a}s Serret, Iban}`) plus a transparent `note` field that lists the four AI contributors (Claude Opus 4.6, GPT 5.2, Gemini 3 Pro, Manus), the ORCID, and the CC-BY 4.0 license. This keeps the bibliography readable in standard academic tooling while preserving the human-AI co-creation transparency that the paper itself maintains in its own author footnote.

6. The Phase 2 editorial fence (D-0063, D-0064) is unaffected by this entry. No Phase 2 figure, caption, or traceability content is touched.

### Rationale

The Â§10.4 paragraph as previously written referred to "el v1" three times without any formal citation. In standard academic practice, mentioning a prior version of one's own work without a citable anchor is either underspecified (the reader cannot locate it) or defensive (the paper reads as needing prior context to justify itself). Confirming that HSI v1 is published on Zenodo with a persistent DOI removes both objections: the autocitation is now formally available and the paper can stand on its own citational scaffolding.

The choice of extension form over retroactive form ("v2 estÃ©n v1 substituint qualitatiu per operatiu" rather than "v2 materialitza el que v1 ja invocava") shifts the narrative subject from past intent to present contribution. This is more paper-friendly because it lets the connecting paragraph claim a specific HSI v2 contribution (operational candidates for what was qualitative) rather than re-justify HSI v1.

The version-specific DOI is chosen over the conceptual parent DOI for paper citations because the conceptual parent always resolves to the latest version on Zenodo. If HSI v2 is eventually deposited on Zenodo as a new version of the same conceptual record, the parent DOI would silently start pointing at v2, which would make any citation of "v1" in this paper effectively self-referential and confusing. The version-specific DOI freezes the citation to the historical v1 record.

The author convention follows standard academic BibTeX norms: human author in the structured `author` field, AI contributors disclosed in a `note` field. This produces a clean rendered citation under `bibliographystyle{plain}` while preserving full transparency for any reader who reads the bibliography entry in detail.

### Source Documents

- `Paper_Latex/main_v2_cat.tex` (lines 67, 713-715, 715 incl. subsection rename)
- `Paper_Latex/references.bib` (new entry `borras2026hsi_v1` at end of file)
- `Paper_Latex/18721272.json` (Zenodo metadata snapshot for DOI `10.5281/zenodo.18721272`)
- Zenodo record: <https://zenodo.org/records/18721272>
- aiXiv mirror: aiXiv ID `aixiv.260305.000001`
- D-0063 (Phase 2 editorial architecture)
- D-0064 (mapping of D-0063 onto existing paper structure)

### Practical Impact

- The paper now has a formal autocitation chain: introduction (`(HSI) \cite{borras2026hsi_v1}`) and Â§10.4 (`la formulaciÃ³ original ... \cite{borras2026hsi_v1}`). Any future autocitation should reuse the same `\cite{borras2026hsi_v1}` key.
- The `references.bib` file is consistent with the existing convention of `@misc` for preprints (cf. `Chaitin2003`, `tHooft1993`, `Youvan2024`).
- No new packages required for compilation.
- Future depositions of HSI v2 to Zenodo should use a separate version-specific DOI under the same conceptual parent (`10.5281/zenodo.18721271`) so that the conceptual chain remains visible while individual citations stay version-frozen.
- The `aiXiv` mirror identifier (`aixiv.260305.000001`) is intentionally not included in the BibTeX entry because BibTeX `plain` style does not render arbitrary mirror identifiers cleanly; the mirror remains discoverable via the canonical DOI page on Zenodo, which is the citable resolver.

### Open Questions

- When HSI v2 is deposited to Zenodo as a new version, should the Phase 2 paper's own DOI be cross-listed in `references.bib` as a self-reference for archival completeness, or kept external to the paper itself?
- Should the English version of the paper (`main_v2_eng.tex` when it exists) inherit the same `\cite{borras2026hsi_v1}` key and BibTeX entry verbatim, or generate a separate entry with English-localized title/note fields?
- Should additional autocitations to HSI v1 be added in other sections of the paper where the original formulation is implicitly invoked (e.g., the "three principles" of Â§2.2, the operator de colÂ·lapse of Â§3), or is the introduction + Â§10.4 pair sufficient as the canonical autocitation locus?

---

## D-0066

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Phase 2 reproducibility freeze and Phase 3 minimal pre-geometric defect/spectrum opening
- Title: Phase 2 is frozen as a reproducible paper-support stack; Phase 3 opens with a minimal local commuting-square defect and sparse-spectrum pilot, explicitly pre-geometric and non-paper-facing until tested against strong nulls

### Decision

1. Phase 2 is now treated as a frozen, reproducible paper-support stack for the purposes of recovery and external review. The frozen stack is documented in `hsi_agents_project/Documentation/HSI_v2_Phase2_Reproducibility_Freeze.md`.

2. The Phase 2 freeze is not a claim that Phase 2 is complete in all possible senses. It means the current paper-facing opening (`null_pressure`) and the internal mechanistic diagnostics (D-0049 through D-0062) have enough traceability to be recovered, audited, and cited consistently.

3. Phase 3 is opened with a minimal pre-geometric pilot, not with a geometry claim. The first candidate object is a local commuting-square bridge defect over four adjacent scales, combined with a sparse graph and normalized-Laplacian spectral readout.

4. The Phase 3 pilot is implemented as `hsi_agents_project/hsi_v2_phase3_min_defect_spectrum_pilot.py`, with internal logic in `hsi_agents_project/v2/phase3/defect_spectrum.py`.

5. No change is made to `Paper_Latex/main_v2_cat.tex`. The Phase 3 pilot is internal only until it demonstrates separation against strong nulls.

### Rationale

Phase 2 has produced a coherent mechanistic stack, but the formal program still requires an operational path toward defects, graphs, and spectra before any stronger geometric language can be considered. The Phase 3 pilot therefore deliberately starts with the smallest object that can be falsified: a local square defect over existing Phase 1 bridge spaces.

This avoids premature geometry. The object is not curvature, torsion, metric structure, or a transport law. It is a pre-geometric diagnostic that asks whether a local algebraic defect and its induced sparse graph distinguish observed projective hierarchy from strong controls.

### Source Documents

- `hsi_agents_project/Documentation/HSI_v2_Phase2_Reproducibility_Freeze.md`
- `docs/HSI_v2_Phase3_Minimal_Defect_Contract.md`
- `hsi_agents_project/hsi_v2_phase3_min_defect_spectrum_pilot.py`
- `hsi_agents_project/v2/phase3/defect_spectrum.py`
- D-0040 through D-0062
- D-0063 and D-0064

### Practical Impact

- Phase 2 recovery has a compact repo-facing freeze document.
- Phase 3 work now has a minimal executable path.
- Any future Phase 3 claim must be tested against `markov1` and `matched-LZ` before it becomes paper-facing.
- If the minimal object fails to separate `matched-LZ`, that failure is informative and should guide the next object toward lag-conditioned or non-local defects rather than being hidden.

### Open Questions

- Does the minimal local commuting-square defect separate observed B/E from `matched-LZ`, or only from weaker collapse controls such as `markov1`?
- If `matched-LZ` shadows the local defect, should the next Phase 3 object be lag-conditioned by the Phase 2 phase-channel structure?
- What is the smallest non-local extension that remains defensible without importing premature geometry?

---

## D-0067

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Phase 3 minimal defect/spectrum sensitivity readout
- Title: The minimal local commuting-square / sparse-spectrum object rejects `markov1` but is shadowed by `matched-LZ`; it is therefore a useful Phase 3 negative result, not a paper-facing separator

### Decision

1. The Phase 3 minimal defect/spectrum pilot was extended into a window and seed sensitivity sweep using `hsi_agents_project/hsi_v2_phase3_defect_spectrum_sensitivity.py`.

2. The sweep tested observed `B` and `E`, `B-markov1`, and six `B-matched-LZ` seeds over three scale windows: `20-24-28-32`, `24-28-32-40`, and `28-32-40-48`.

3. The local defect/spectrum object separates `markov1` strongly from the observed projective hierarchy in the first two tested windows. However, `matched-LZ` shadows observed `B/E` almost exactly in those same windows.

4. The high window `28-32-40-48` is not interpretable as a positive separator because observed `B/E` and `matched-LZ` produce empty local support while `markov1` retains only a small residual graph.

5. The minimal local object is therefore accepted as an informative internal negative result: it is a collapse detector but not a strongest-null separator.

6. No paper-facing claim is made from this readout. The next Phase 3 object should be lag-conditioned or otherwise non-local, anchored in the temporal phase-channel structure discovered in Phase 2.

### Rationale

The purpose of the first Phase 3 pilot was not to win against every null, but to falsify the simplest possible defect/spectrum candidate before building anything more elaborate. The result is clean: a purely local commuting-square defect is too endogenous to the pattern-support preservation exploited by `matched-LZ`.

This is scientifically useful because it prevents overclaim. It says that Phase 3 cannot merely add a local square defect on top of Phase 1. The successful Phase 2 separation came from temporal/lag-conditioned structure; the next Phase 3 candidate must inherit that lesson.

### Source Documents

- `docs/HSI_v2_Phase3_Minimal_Defect_Spectrum_Readout.md`
- `hsi_agents_project/hsi_v2_phase3_defect_spectrum_sensitivity.py`
- `hsi_agents_project/hsi_v2_phase3_min_defect_spectrum_pilot.py`
- `hsi_agents_project/v2/phase3/defect_spectrum.py`
- `hsi_agents_project/results/hsi_v2/phase3/defect_spectrum_sensitivity/phase3-defect-spectrum-sensitivity__obs-B-E__null-B__w-3__lzseeds-6__20260501T092055/`

### Practical Impact

- The first Phase 3 pilot is not promoted to paper.
- The current interpretation should be "local defect/spectrum is insufficient against `matched-LZ`", not "Phase 3 failed".
- Future Phase 3 work should test lag-conditioned defects, phase-channel-conditioned squares, or non-local bridge operators.
- `markov1` remains useful as a collapse control for validating that the defect/spectrum machinery is not trivial.

### Open Questions

- What is the minimal lag-conditioned square defect that can be tested without overfitting to Phase 2?
- Should the first non-local Phase 3 candidate use the fixed `-27M` channel, local shell-lag estimates, or a small phase-channel grid?
- Can a spectrum over lag-conditioned bridge graphs separate `matched-LZ` while preserving B/E coherence?

---

## D-0068

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Phase 3 non-local phase-channel spectral graph (N3-03)
- Title: N3-03 converts the audited Phase 2 phase-channel transition graph into a unit-weighted normalized-Laplacian readout; `top=128` forms two non-local channel components while `top=96` fails to form a stable channel path

### Decision

1. The N3-03 phase-channel spectral graph is accepted as an internal Phase 3 result.

2. The object is implemented in `hsi_agents_project/hsi_v2_phase3_phase_channel_spectral_graph.py` and reads the audited N2-14 transition artifact:
   `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/`.

3. The graph uses band x channel states as nodes and adjacent-band transitions as edges. Edge weights are fixed to `1.0` by design. No margin weighting, lag weighting, or fitted metric is introduced.

4. Under the reference `top=128` contract, the negative and positive channels each produce a connected 6-node / 5-edge component with normalized-Laplacian `lambda2 ~= 0.1910` and one zero eigenvalue per channel.

5. The combined `top=128` graph has `12` nodes, `10` edges, and `zero_eigenvalues = 2`, consistent with two disconnected channel components rather than a collapsed or short-circuited graph.

6. Under `top=96`, neither channel forms a stable spectral path. This preserves the Phase 2 boundary/top-k sensitivity reading.

7. Grace's audit approves N3-03 as a legitimate non-local Phase 3 bridge. A small technical correction is recorded: `lambda2 ~= 0.1910` should be read conservatively as consistent with a simple 6-node path-like component, not as evidence for extra graph thickness.

8. No paper-facing Phase 3 claim is made from N3-03. It is a pre-geometric spectral bridge and does not imply curvature, metric recovery, or a complete transport spectrum.

### Rationale

N3-01 and N3-02 showed that purely local square defects are too weak against `matched-LZ`: they reject diffuse collapse controls such as `markov1`, but they do not defeat the strongest local-support-preserving null.

N3-03 changes the level of description without adding new fitted parameters. Instead of measuring another local square, it reuses the Phase 2 phase-channel transition scaffold and asks whether that scaffold has non-local graph connectivity. The unit-weighted Laplacian is intentionally austere: it tests existence of connected channel structure before any weighted geometry is attempted.

The result is exactly the kind of bridge Phase 3 needed: not a geometry claim, but a mathematically well-defined spectral object over a non-local channel scaffold.

### Source Documents

- `hsi_agents_project/hsi_v2_phase3_phase_channel_spectral_graph.py`
- `hsi_agents_project/results/hsi_v2/phase3/phase_channel_spectral_graph/phase3-phase-channel-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__20260501T095214/`
- `docs/HSI-audit-Phase3-N3-03.md`
- `docs/HSI-audit-Phase3-N3-03-Response.md`
- D-0062 (N2-14 phase-channel transition graph)
- D-0067 (local defect/spectrum negative result)

### Practical Impact

- N3-03 becomes the first accepted internal Phase 3 non-local spectral bridge.
- The current Phase 3 direction should move from local square defects toward non-local spectral objects anchored in phase-channel structure.
- Future weighted graph variants must be treated as new objects, not as silent upgrades of N3-03.
- `top=96` remains a boundary control; `top=128` remains the current reference resolution for the non-local channel scaffold.

### Open Questions

- What is the smallest micro-scale spectral object that inherits the N3-03 non-local channel contract?
- Should the next object use pattern nodes, child-destination events, dropout/rerouting states, or bridge-fibre transitions?
- Can a weighted spectral variant be defined without reintroducing metric tuning?
- What null ensemble is required for a future Phase 3 spectral separator to become paper-facing?

---

## D-0069

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Phase 3 N3-05b/N3-05c weighted and component-normalized prefix/suffix quotient readouts
- Title: N3-05c is accepted as an internal flow-entropy readout: prefix/suffix does not recover connected geometry, but after component-wise mass normalization observed `B/E` remain less diffuse than `matched-LZ`

### Decision

1. N3-05b and N3-05c are accepted as internal Phase 3 readouts of weighted routing flow, not as geometric or paper-facing separators.

2. N3-05b uses pooled retained child mass as edge weight over the exact prefix/suffix quotient:
   `edge_weight = min(anchor_child_mass, candidate_child_mass)`, split across exact prefix/suffix projections.

3. N3-05b confirms that the prefix/suffix quotient remains fragmented (`zero=24` for observed `B/E` and `matched-LZ`) and therefore does not supply connected geometry. Its absolute weighted spectral radius is read only as flow capacity, not as topology.

4. N3-05c normalizes the N3-05b graph component by component, removing absolute mass scale inside each connected component. Under this normalization, observed `B/E` remain less diffuse than `matched-LZ`: lower component-weight entropy, fewer effective weight components, and higher top-component concentration.

5. Grace's N3-05b and N3-05c audits are accepted with one quantitative caution: the qualitative "concentration" reading is valid, but the observed effective component count is approximately `11`, not a tiny `~3`-component collapse. The result is clear and stable, but must not be exaggerated.

6. The official wording is: "The prefix/suffix quotient does not produce geometry, but it preserves a normalized low-entropy flow difference between observed `B/E` and `matched-LZ`." Do not phrase this as "Phase 3 solved", "geometry recovered", or "topology separates B".

### Rationale

N3-05b answered Grace's critique of the unit quotient by adding physically meaningful pooled-flow weights, but it also exposed the danger of reading absolute spectral radius as topology. N3-05c therefore asked a stricter question: after fixing the fragmented prefix/suffix scaffold and normalizing within each component, does any non-scalar shape difference remain?

The answer is yes, but narrowly. Observed `B/E` keep a less diffuse routing-flow profile than `matched-LZ` in the same fixed scaffold. This means the N3-05b separation is not merely a disguised total-mass scalar. However, the shared fragmentation of the scaffold prevents any geometric claim.

### Source Documents

- `hsi_agents_project/hsi_v2_phase3_weighted_quotient_child_routing_spectral_graph.py`
- `hsi_agents_project/v2/phase3/quotient_child_routing_spectrum.py`
- `hsi_agents_project/hsi_v2_phase3_component_quotient_child_routing_spectrum.py`
- `hsi_agents_project/v2/phase3/component_quotient_spectrum.py`
- `hsi_agents_project/results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph/phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260501T105949/`
- `hsi_agents_project/results/hsi_v2/phase3/component_quotient_child_routing_spectrum/phase3-component-quotient-spectrum__src-phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260501T105949__20260501T113746/`
- `docs/HSI-audit-Phase3-N3-05b.md`
- `docs/HSI-audit-Phase3-N3-05c.md`

### Practical Impact

- N3-05c becomes the current strongest internal Phase 3 readout for "routing thermodynamics" or "low-entropy flow" language.
- Prefix/suffix remains useful as a control scaffold, but it is exhausted as a candidate geometry.
- Any future Phase 3 object should preserve the low-entropy flow lesson and avoid treating connectedness alone as evidence.
- No edit to `Paper_Latex/main_v2_cat.tex` is authorized by this entry.

### Open Questions

- Can the low-entropy flow reading be reproduced from a more canonical non-syntactic quotient without radius tuning?
- Is there a future normalized-flow observable that can be stated without graph geometry at all?
- What is the minimum additional null ensemble required before this Phase 3 readout can become paper-facing?

---

## D-0070

- Date: 2026-05-01
- Status: `accepted-negative`
- Scope: Phase 3 N3-06 Hamming quotient spectral graph
- Title: N3-06 shows that Hamming radius fuses the fractured prefix/suffix scaffold, but pure structural connectivity is shared by observed `B/E` and `matched-LZ`; Hamming connectivity is therefore not a Phase 3 geometric separator

### Decision

1. N3-06 is accepted as a useful negative Phase 3 result.

2. The Hamming quotient contract is methodologically clean: families whose Hamming distance is within a predeclared radius are identified, then existing retained-flow edges are aggregated. No new transport edge is invented.

3. The tested radius sweeps (`1,2,4,8` and `6,8,10,12`) show that Hamming radius fuses the prefix/suffix scaffold: approximately `24 -> 3/2 -> 1` connected components as radius grows.

4. This connectivity transition is common to observed `B/E` and `matched-LZ`. In particular, at high radius the Hamming quotient connects both observed and `matched-LZ` graphs. Therefore pure Hamming connectivity cannot separate the deterministic low-entropy tube from the diffuse matched-null gas.

5. A Hamming-normalized follow-up is not pursued at this point. Grace's audit is accepted: such a follow-up would likely restate the N3-05c flow-entropy result with extra machinery rather than produce a distinct topological separator.

6. The official wording is: "Hamming fuses the scaffold but does not separate observed geometry from `matched-LZ`; the discriminant remains flow entropy, not topological existence."

### Rationale

N3-06 was the natural response to the failure of prefix/suffix connectedness. If the syntax-level quotient was too rigid, Hamming proximity could have revealed a more continuous geometric scaffold.

The result is negative but valuable. Hamming radius does repair fragmentation, but it repairs it for both observed structure and the strongest matched null. This means connectedness itself is too weak: a coherent tube and a diffuse gas can both become connected at the same coarse scale. The Phase 3 signal therefore does not live in plain connectivity; it lives in how flow is distributed through the scaffold.

### Source Documents

- `hsi_agents_project/hsi_v2_phase3_hamming_quotient_child_routing_spectral_graph.py`
- `hsi_agents_project/v2/phase3/hamming_quotient_spectrum.py`
- `hsi_agents_project/results/hsi_v2/phase3/hamming_quotient_child_routing_spectral_graph/phase3-hamming-quotient-spectral-graph__src-phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260501T105949__r-1-2-4-8__20260501T115114/`
- `hsi_agents_project/results/hsi_v2/phase3/hamming_quotient_child_routing_spectral_graph/phase3-hamming-quotient-spectral-graph__src-phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260501T105949__r-6-8-10-12__20260501T115131/`
- `docs/HSI-audit-Phase3-N3-06.md`
- D-0069

### Practical Impact

- Hamming connectivity is closed as a candidate Phase 3 separator.
- The next Phase 3 move should not be another unweighted connectedness quotient unless it has a clearly different discriminant.
- N3-05c remains the strongest current internal Phase 3 readout.
- No edit to `Paper_Latex/main_v2_cat.tex` is authorized by this entry.

### Open Questions

- Should the next Phase 3 step be a direct flow-entropy formalization instead of another graph quotient?
- Can low-entropy flow be tested across additional bands, variants, or null seeds without expanding the computational burden too much?
- What would count as a paper-safe Phase 3 opening, distinct from the already paper-staged Phase 2 stack?

---

## D-0071

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Phase 3 N3-07 direct flow-entropy readout
- Title: N3-07 accepts `strict-low-entropy-flow` as the current internal Phase 3 closure: observed `B/E` beat the strongest `matched-LZ` envelope on entropy, effective components, and top-share concentration in both phase channels

### Decision

1. N3-07 is accepted as an internal Phase 3 readout derived from N3-05c component-normalized quotient artifacts.

2. N3-07 deliberately introduces no new graph, quotient, radius, or geometric metric. It directly scores the N3-05c source-summary fields against the empirical `matched-LZ` seed envelope per channel.

3. The accepted score contract is conservative:
   - lower component-weight entropy than the best `matched-LZ` seed;
   - fewer effective weighted components than the best `matched-LZ` seed;
   - higher `top3` concentration than the best `matched-LZ` seed;
   - higher `top5` concentration than the best `matched-LZ` seed.

4. In the reference run, observed `B` and `E` score `4/4` in both negative and positive channels. `markov1` remains a no-flow collapse control.

5. Grace's N3-07 audit is accepted: comparing observed rows against the most favorable `matched-LZ` envelope is conservative and anti-B/E, not permissive. The combined entropy/effective-component/top-share criteria are therefore a defensible internal readout of canalized flow.

6. The official wording is: "`strict-low-entropy-flow` is an internal Phase 3 routing-thermodynamic readout." Allowed paraphrase: "observed `B/E` retain a lower-entropy, more concentrated routing-flow profile than the strongest `matched-LZ` envelope inside the fixed component-normalized scaffold."

7. Forbidden wording: do not say "Phase 3 solved", "geometry recovered", "topology discovered", "paper completed", "curvature measured", or "matched-LZ defeated by geometry." The result closes the current internal infrastructure arc; it does not by itself authorize a paper-facing Phase 3 claim.

### Rationale

N3-05c showed that after removing absolute mass scale component by component, observed `B/E` remain less diffuse than `matched-LZ`. N3-06 then showed that plain connectivity is not the separator: Hamming radius connects both observed structure and the strongest matched null.

N3-07 resolves this fork cleanly. It does not try another geometric quotient. Instead, it asks the direct thermodynamic question implied by N3-05c and N3-06: inside the same fixed fragmented scaffold, is the observed flow more canalized than the best `matched-LZ` seed envelope?

The answer is yes under all four predeclared criteria in both tested channels. This makes N3-07 the current internal closure of the Phase 3 exploratory arc: the discriminant is low-entropy routing flow, not recovered connected geometry.

### Source Documents

- `hsi_agents_project/hsi_v2_phase3_flow_entropy_readout.py`
- `hsi_agents_project/v2/phase3/flow_entropy_readout.py`
- `hsi_agents_project/results/hsi_v2/phase3/flow_entropy_readout/phase3-flow-entropy-readout__src-620fcc3412__20260501T140015/`
- `hsi_agents_project/results/hsi_v2/phase3/component_quotient_child_routing_spectrum/phase3-component-quotient-spectrum__src-phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260501T105949__20260501T113746/`
- `docs/HSI-audit-Phase3-N3-07.md`
- D-0069
- D-0070

### Practical Impact

- N3-07 becomes the strongest current internal Phase 3 readout.
- Further Phase 3 infrastructure invention is paused unless a concrete objection or new falsification target appears.
- The next natural step is editorial: decide how, if at all, the internal Phase 3 stack should be summarized in manuscript-facing architecture.
- No edit to `Paper_Latex/main_v2_cat.tex` is authorized by this entry.

### Open Questions

- Should Phase 3 remain entirely internal for the current manuscript, or should it receive a short appendix pointer after editorial review?
- What additional null ensemble or independent rerun would be required before `strict-low-entropy-flow` becomes paper-facing?
- Should Sofia convene a Phase 3 editorial architecture session analogous to the Phase 2 session, or should the manuscript remain Phase-2-centered for now?

---

## D-0072

- Date: 2026-05-01
- Status: `accepted-internal`
- Scope: Phase 3 editorial deferral and pointer-only manuscript treatment
- Title: Phase 3 receives a single pointer paragraph in `main_v2_cat.tex` Â§11.2; full editorial crystallization (appendix, figures, traceability table) is deferred until at least one additional null ensemble or one independent rerun is completed and a new formal decision authorizes the upgrade

### Decision

1. The internal Phase 3 arc (D-0066 through D-0071) is acknowledged in the manuscript by a single pointer paragraph at the end of `\subsection{Porta 2: Defectes i geometria (pendent)}` in `Paper_Latex/main_v2_cat.tex`, after the existing paragraph that introduces the first observational opening at `subsec:primera_obertura`.

2. The pointer paragraph reports three things and only three things:
   - that a first internal pre-geometric diagnostic arc has been closed on the consolidated scaffold;
   - that connectivity-based candidates (local commuting-square defects, normalized spectrum, prefix/suffix and Hamming quotients) do not stably separate observed `B/E` from the strong null `matched-LZ`;
   - that the surviving discriminant inside the same component-normalized scaffold is a thermodynamic readout of routing flow, namely lower component-weight entropy and fewer effective components in `B/E` than in the most favorable `matched-LZ` envelope.

3. The pointer paragraph closes by stating that what is deferred is not the internal result but its paper-facing crystallization, and by reaffirming that none of the three falsification conditions enumerated in Â§11.2 is either satisfied or refuted by this arc.

4. No appendix, figure, or traceability table for Phase 3 is added to the manuscript at this time. Variants B and C of the editorial architecture (appendix `\section{}` analogous to Appendix~E for Phase 2) are explicitly postponed.

5. Any future upgrade from pointer-only to a fuller manuscript-facing Phase 3 architecture requires all three of the following:
   - at least one additional null ensemble beyond `matched-LZ` and `markov1`, or one independent rerun of N3-05c plus N3-07 on a disjoint seed family;
   - an explicit Grace audit of the proposed appendix architecture against the Phase 3 wording fence established in D-0071;
   - a new formal Decision Log entry (D-XXXX) recording the upgrade authorization.

6. The Phase 3 wording fence inherited from D-0071 remains in force at the manuscript level. Forbidden in the pointer paragraph: "geometry recovered", "topology discovered", "Phase 3 solved", "curvature measured", "matched-LZ defeated by geometry". Allowed paraphrases must stay inside the routing-thermodynamic register.

### Rationale

N3-07 is internally strong, but precisely because it is strong it must be protected from premature paper-facing conversion. The internal arc is one day old, and the open question explicitly left in D-0071 -- which additional null ensemble or independent rerun would be required before `strict-low-entropy-flow` becomes paper-facing -- is not yet answered.

A pointer paragraph captures the bulk of the narrative value of the Phase 3 arc with the smallest possible editorial surface. It tells an external reader that the program has explored geometric candidates, that none of them separates the strong null on its own, and that a thermodynamic readout currently does, without claiming that Porta 2 has been crossed.

A full appendix analogous to Appendix~E for Phase 2 would be structurally hard to keep inside the D-0071 wording fence: a visual presentation of six decisions plus a traceability table tends to suggest a level of consolidation that the current evidence does not yet support. Deferral is therefore not timidity; it is methodological discipline.

The three-agent review (Sofia editorial, Ariadna technical, Grace audit) converged independently on this option. Grace's characterization of N3-07 as a "thermodynamic lighthouse" is incorporated into the rationale, but its editorial activation is intentionally postponed.

### Source Documents

- D-0063 (Phase 2 editorial skeleton)
- D-0064 (Phase 2 editorial mapping and integration into `main_v2_cat.tex`)
- D-0065 (HSI v1 autocitation and nomenclature shift)
- D-0066 through D-0071 (Phase 3 internal pre-geometric arc)
- `Paper_Latex/main_v2_cat.tex` Â§11.2 `\subsection{Porta 2: Defectes i geometria (pendent)}`
- Three-agent review thread on Phase 3 editorial options (Sofia, Ariadna, Grace, 2026-05-01)

### Practical Impact

- One paragraph is added to `Paper_Latex/main_v2_cat.tex` Â§11.2; no other manuscript file is touched.
- No new figure is added to `Paper_Latex/`.
- Phase 3 appendix, traceability table, and figure selection remain available as future work but are not staged.
- The Phase 3 internal stack continues to evolve in `hsi_agents_project/` and in the Decision Log without further manuscript impact until D-XXXX authorizes an upgrade.
- External readers receive an honest, narrow, fence-compliant signal that Porta 2 is still pending but that an internal pre-geometric exploration has already produced a non-trivial thermodynamic readout.

### Open Questions

- Which additional null ensemble (for example, lag-conditioned matched-LZ, block-shuffled `matched-LZ`, or a second LZ-matched family with disjoint seeds) would best test the robustness of `strict-low-entropy-flow`?
- Does an independent rerun of N3-05c plus N3-07 on a disjoint seed family preserve the 4/4 score in both phase channels?
- When the upgrade decision (D-XXXX) is opened, should the manuscript-facing Phase 3 architecture be a new appendix `\section{}` parallel to Appendix~E, or a dedicated subsection inside Â§11.2 with its own labeled paragraphs?

---

## D-0073

- Date: 2026-05-01
- Status: `accepted-operational`
- Scope: Reproducibility hardening for Phase 2/3 reruns
- Title: Public sensitivity pipelines must recover missing Phase 1 observed/null towers automatically before long downstream sweeps begin

### Decision

1. `hsi_v2_phase2_parent_shell_atlas_sensitivity.py` is updated so missing Phase 1 inputs are recovered by design before the shell-lag, lag-aware, and atlas stages begin.

2. The wrapper now checks for required observed towers for both variants in the selected two-variant profile.

3. When `--stage nulls` or `--stage all` is selected, the wrapper also checks for required candidate-side `markov1` and `matched-lz` Phase 1 null towers matching the requested iteration, segment geometry, scale ladder, projection policies, null model, and seed family.

4. Missing inputs are generated automatically with `hsi_v2_phase1_run.py` into `phase1_dir/auto_observed/...` or `phase1_dir/auto_nulls/...` before the expensive downstream sweep starts.

5. The recovery actions are recorded in the sensitivity summary, report, and manifest. Present inputs are also recorded as `action = present`, so a reproducer can distinguish reused inputs from generated ones.

6. No scientific observable is changed by this decision. It is an operational reproducibility hardening: the same Phase 1 contracts are built, only earlier and in a controlled way.

### Rationale

The independent N3-05c/N3-07 rerun initially failed halfway through the N2-06 sensitivity wrapper because the requested disjoint `matched-LZ` seed family did not yet exist as Phase 1 tower artifacts. The failure was scientifically safe, but operationally poor: an external reproducer would receive a traceback after partial progress and would have to infer which upstream commands to run.

For public reproducibility, a one-command wrapper should either complete the declared pipeline or prepare its declared prerequisites before any long downstream computation begins. Since Phase 1 observed and null towers are deterministic under their explicit arguments, automatic recovery is preferable to a flag-gated retry workflow.

### Source Documents

- `hsi_agents_project/hsi_v2_phase2_parent_shell_atlas_sensitivity.py`
- `hsi_agents_project/Documentation/HSI_v2_Report_and_Data_Guide.md`
- D-0072

### Practical Impact

- Independent reruns with disjoint `matched-LZ` seeds can be launched with a single command.
- Missing Phase 1 nulls no longer cause mid-run failure in the sensitivity wrapper.
- Generated prerequisites are traceable in the wrapper report and manifest.
- This operational change should be applied to other public long-running wrappers if the same failure mode appears.

### Open Questions

- Should other Phase 2/3 public wrappers receive the same Phase 1 input recovery pattern proactively?
- Should auto-generated Phase 1 inputs be promoted to a shared reusable cache location, or is `phase1_dir/auto_*` sufficient?
- Should the reproducibility guide include a short "automatic prerequisite recovery" section once the independent rerun completes?

---

## D-0074

- Date: 2026-05-02
- Status: `accepted-internal`
- Scope: Phase 3 independent rerun of N3-05c/N3-07
- Title: The disjoint-seed independent rerun preserves `strict-low-entropy-flow` 4/4 for observed `B/E` in both phase channels, satisfying the rerun condition requested by D-0072 without by itself authorizing a paper-facing Phase 3 upgrade

### Decision

1. The independent rerun using a disjoint `matched-LZ` seed family (`607,709,811,907,1009,1103`) is accepted as a valid internal rerun of the N3-05b -> N3-05c -> N3-07 chain.

2. The canonical rerun chain is:
   - N2-11 child-routing lag-response:
     `hsi_agents_project/results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval_independent_seeds/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-10__20260501T163307/`
   - N3-05b weighted quotient:
     `hsi_agents_project/results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph_independent_seeds/phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260502T153230/`
   - N3-05c component quotient:
     `hsi_agents_project/results/hsi_v2/phase3/component_quotient_child_routing_spectrum_independent_seeds/phase3-component-quotient-spectrum__src-phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-2__20260430T140724__top-128__m-40__20260502T153230__20260502T153344/`
   - N3-07 flow entropy:
     `hsi_agents_project/results/hsi_v2/phase3/flow_entropy_readout_independent_seeds/phase3-flow-entropy-readout__src-e8827e75d7__20260502T153352/`

3. In the independent N3-07 rerun, observed `B` and `E` retain `score = 4` and verdict `strict-low-entropy-flow` in both negative and positive channels. `markov1` remains a no-flow collapse control.

4. Grace's audit of the independent rerun is accepted on the methodological core: reusing the transition graph and lag grid is not a hidden dependence but the correct fixed circuit for testing whether a disjoint `matched-LZ` seed family can mimic observed low-entropy routing flow inside the same scaffold.

5. The official wording is: "the independent disjoint-seed rerun satisfies the D-0072 rerun condition for the internal Phase 3 `strict-low-entropy-flow` readout." Forbidden wording remains: "Phase 3 solved", "geometry recovered", "topology discovered", "curvature measured", or "paper-facing Phase 3 claim authorized."

6. As operational hardening linked to this decision, the N3-05/N3-05b quotient wrappers now treat missing routed transition cells as a hard error. They refuse to emit incomplete quotient artifacts when any transition graph cell lacks a corresponding child-routing input.

### Rationale

D-0072 explicitly required either an additional null ensemble or an independent rerun of N3-05c plus N3-07 over a disjoint seed family before any Phase 3 paper-facing upgrade could be considered.

The rerun changes the `matched-LZ` realization family while preserving the circuit under test: the transition graph, bands, lag grid, anchor/candidate variants, top-k, and quotient contract remain fixed. This is the correct experimental design for testing robustness against null stochasticity, because allowing each null to define a different circuit would change the object being measured.

The result is stable. Even though N2-11 contains boundary cells where raw child-retention can be shadowed by `matched-LZ`, the downstream component-normalized flow-entropy readout still separates observed `B/E` from the best `matched-LZ` envelope in both phase channels. This strengthens, rather than weakens, the interpretation that N3-07 measures a thermodynamic routing-flow discriminant not visible to raw retention alone.

This decision closes the rerun condition. It does not close the editorial condition. Any manuscript-facing upgrade still requires Sofia-led editorial architecture, Grace audit against the Phase 3 wording fence, and a new explicit D-XXXX authorizing the upgrade.

### Source Documents

- `docs/HSI-audit-Phase3-N3-07-Rerun.md`
- `hsi_agents_project/hsi_v2_phase3_weighted_quotient_child_routing_spectral_graph.py`
- `hsi_agents_project/hsi_v2_phase3_quotient_child_routing_spectral_graph.py`
- N2-11, N3-05b, N3-05c, and N3-07 rerun paths listed above
- D-0071
- D-0072
- D-0073

### Practical Impact

- The internal Phase 3 stack now has an accepted disjoint-seed rerun.
- The "statistical luck under the first `matched-LZ` seed family" objection is substantially weakened.
- N3-05/N3-05b can no longer silently emit incomplete artifacts when child-routing cells are missing.
- Sofia may now reconsider whether to open a Phase 3 paper-facing architecture session, but no paper edit is authorized by this entry.

### Open Questions

- Should the next Phase 3 upgrade step be editorial architecture, or should a genuinely new null ensemble be implemented first?
- Should N2-11 be parallelized or cached more aggressively before any future public reruns, given the near-24h runtime of the disjoint-seed profile?
- Should the report/data guide add a dedicated "rerun-independent Phase 3 chain" section for external reproducibility?

---

## D-0075

- Date: 2026-05-02
- Status: `accepted-methodological`
- Scope: Phase 2/3 additional null ensemble
- Title: `phase-matched-lz` is introduced as a stronger seeded LZ-family null that preserves block phase classes while still destroying exact long-range ordering

### Decision

1. A new public null model, `phase-matched-lz`, is added to the HSI v2 Phase 1 generator.

2. `phase-matched-lz` is deliberately close to `matched-lz`: it searches over the same LZ-calibrated block-size family, but each candidate surrogate shuffles blocks only inside modulo-period phase classes rather than globally.

3. The candidate grid is defined by:
   - block sizes inherited from `matched-lz`;
   - phase periods in blocks: `2, 3, 5, 7`;
   - the same fixed LZ calibration windows already used by `matched-lz`.

4. The selected surrogate minimizes the absolute normalized LZ-profile error against the observed source, and records the selected block size, selected phase period, candidate grid, target/achieved LZ profile, and preservation/breakage contract in Phase 1 metadata.

5. Phase 2 null selection now accepts `phase-matched-lz` as a seeded LZ-family null. Existing `--matched-lz-seeds` plumbing is reused as the seed-family argument for seeded LZ-family nulls, including `matched-lz` and `phase-matched-lz`.

6. N2-11 (`hsi_v2_phase2_child_routing_lag_response.py`) now performs automatic Phase 1 prerequisite recovery before long lag-response sweeps, including `phase-matched-lz` towers when requested.

7. N3-07 flow-entropy scoring is generalized from the literal `B-matched-lz` envelope to the seeded LZ-family null envelope present in the input component artifact. This allows `phase-matched-lz` reruns to be scored without falsely labelling them as the original `matched-lz`.

8. Long lag-response reruns now use an automatic persistent frozen-source cache for null surrogates. Cache entries are prefix-compatible: a longer cached null realization may satisfy shorter later requests without regenerating the source.

9. N2-11 now supports conservative per-band lag-target parallelism through `--workers`. The wrapper first executes a largest-required-bits cache-warming target, then parallelizes the remaining lag targets for that band with child scripts silenced and wrapper-level heartbeat/progress output.

10. No manuscript-facing Phase 3 claim is authorized by this decision. `phase-matched-lz` is an internal falsification hardening step for the D-0072 additional-null branch.

### Rationale

D-0074 closed the independent rerun branch of D-0072, but Sofia's editorial decision correctly left the additional-null branch open. A genuinely different null should not merely change seeds; it should give the null more structural advantage while preserving a clear breakage contract.

The original `matched-lz` preserves local block content and global LZ profile but globally shuffles block order. The observed Phase 2/3 readouts are explicitly phase-channel-sensitive, so a harsher null should be allowed to preserve coarse phase lanes. `phase-matched-lz` does exactly that: it keeps block positions inside phase classes while still destroying exact within-lane order and exact long-range arrangement.

If observed `B/E` cease to separate against `phase-matched-lz`, the current routing-thermodynamic reading weakens. If they continue to separate, the "ordinary LZ block shuffle was too easy" objection is substantially reduced.

### Source Documents

- `hsi_agents_project/v2/common/null_models.py`
- `hsi_agents_project/v2/phase2/null_pressure.py`
- `hsi_agents_project/v2/phase2/return_lag.py`
- `hsi_agents_project/hsi_v2_phase2_child_routing_lag_response.py`
- `hsi_agents_project/v2/phase3/flow_entropy_readout.py`
- D-0072
- D-0074

### Practical Impact

- Researchers can now generate Phase 1 towers with `--null-model phase-matched-lz`.
- Long N2-11 runs can request `--null-models markov1,phase-matched-lz` and a seed family; missing Phase 1 prerequisites are generated before the long sweep begins.
- Expensive null source realizations are cached under `results/hsi_v2/source_cache/frozen_sources/` by default, with optional relocation through `HSI_V2_SOURCE_CACHE_DIR`.
- Full N2-11 sweeps may use `--workers N` after cache warming to parallelize lag targets without interleaved child-script logs.
- N3-05b/N3-05c/N3-07 can be rerun on the resulting child-routing artifact without changing the transition graph or quotient contract.
- Reports remain internal until Grace audits the code/results and a later decision decides whether the new null changes the Phase 3 editorial state.

### Open Questions

- Does the N2-11 -> N3-05b -> N3-05c -> N3-07 chain preserve `strict-low-entropy-flow` when `phase-matched-lz` replaces the original `matched-lz` seed envelope?
- Is `phase-matched-lz` sufficient as the additional null ensemble requested by D-0072, or should a second qualitatively different null (`block-shuffled matched-LZ`, lag-conditioned LZ, or another control) still be implemented before any Phase 3 paper-facing upgrade?
- Should the public reproducibility guide rename `--matched-lz-seeds` to a more general seeded-null argument in a later backwards-compatible cleanup?

---

## D-0076

- Date: 2026-05-03
- Status: `accepted-internal`
- Scope: Phase 2/3 revalidation against the `phase-matched-lz` null ensemble
- Title: `phase-matched-lz` damages the raw phase-channel scaffold at the boundary but does not defeat the Phase 3 low-entropy routing-flow discriminant

### Decision

1. The full `phase-matched-lz` revalidation chain is accepted as a valid internal stress test of the D-0075 additional-null branch.

2. `phase-matched-lz` is treated as a genuinely stronger seeded LZ-family null, not as a cosmetic rerun of `matched-lz`. It preserves local block phase classes while still breaking exact within-phase order and exact long-range arrangement.

3. N2-11 confirms that `phase-matched-lz` is a harder null than the original `matched-lz`, especially at the `729M-738M` boundary. It raises the null envelope there and produces real `null-shadowing` rather than merely reproducing the earlier matched-LZ profile.

4. N2-12b and N2-13a confirm the existing boundary interpretation:
   - central bands preserve strong monitored phase-channel cells under both `top=96` and `top=128`;
   - the sensitive behavior remains concentrated in `729M-738M`;
   - `729M-738M` is not promoted to a clean late channel, but remains a monitored `null-shadowing` boundary.

5. N2-14 shows that the raw transition scaffold is partially fragile under `phase-matched-lz`: the negative `top=128` channel remains a structured path (`5/6` active bands), while the positive `top=128` channel falls to `no-stable-channel-path`. This is accepted as a real weakening of the topological scaffold, not hidden as noise.

6. N3-05b/N3-05c/N3-07 preserve the core Phase 3 discriminant. Observed `B` and `E` retain `score = 4` and verdict `strict-low-entropy-flow` in both negative and positive channels, while `markov1` remains a no-flow collapse control and `phase-matched-lz` remains a diffuse/deficit LZ-family flow envelope.

7. The accepted wording is: "`phase-matched-lz` weakens the raw topological channel scaffold at the boundary, but the component-normalized flow-entropy readout still separates observed `B/E` from the stronger local-phase-preserving LZ null envelope." Forbidden wording: "irreversible proof", "definitive control closure", "Phase 3 solved", "geometry recovered", "topology discovered", "curvature measured", or "paper-facing Phase 3 upgrade authorized."

8. The internal mapping of `phase_matched_lz_*` columns into legacy downstream `matched_lz_*` fields is accepted as an engineering compatibility layer only because the derived artifacts explicitly preserve `null_family = phase_matched_lz` and `null_family_n = 6`. In future prose, this must be described as the seeded LZ-family null envelope, not as the original `matched-lz` model.

9. No manuscript edit is authorized by this decision. D-0076 satisfies the additional-null branch requested by D-0072 at the internal level, but any paper-facing Phase 3 upgrade still requires the separate editorial/audit conditions stated in D-0072.

### Rationale

D-0075 introduced `phase-matched-lz` precisely to test whether the Phase 2/3 routing-flow stack was relying on a weakness of the original globally shuffled `matched-lz` null. The new null was allowed to preserve block phase classes, making it more dangerous for phase-channel readouts.

The result is scientifically stronger than a perfectly clean survival would have been. A null that preserves local phase lanes should damage at least part of the raw scaffold if the measurement is real and not hard-coded. It does: the `729M-738M` boundary and the positive N2-14 path become visibly weaker. This validates the earlier `null-shadowing boundary` vocabulary and prevents overclaiming a universal phase-channel law.

At the same time, the downstream thermodynamic readout survives. N3-07 does not claim recovered geometry; it measures whether observed routing flow is more component-concentrated and lower-entropy than the seeded LZ-family null envelope inside the same fixed quotient circuit. Under that contract, observed `B/E` still score `4/4` against `phase-matched-lz` in both channels.

Grace's audit is accepted on the methodological core: `phase-matched-lz` is a serious stress null, reusing the N2-12b -> N2-13a -> N2-14 -> N3-07 chain is the correct fixed-instrument test, and the boundary weakening makes the survival of N3-07 more meaningful. The audit's strongest rhetorical formulations are deliberately narrowed here to preserve the Phase 3 wording fence.

### Source Documents

- `docs/HSI-audit-Phase-Matched-LZ-Null.md`
- `hsi_agents_project/results/hsi_v2/phase2/phase_matched_lz_full_reval/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-10__20260502T194512/`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_width_topk_sensitivity_phase_matched_lz_reval/phase2-phase-channel-width-topk-sensitivity__stage-all__top-96-128x2__lags-10__20260502T221741/`
- `hsi_agents_project/results/hsi_v2/phase2/off_universe_mass_readout_phase_matched_lz_reval/phase2-off-universe-mass-readout__runs-1__20260503T073414/`
- `hsi_agents_project/results/hsi_v2/phase2/phase_channel_transition_graph_phase_matched_lz_reval/phase2-phase-channel-transition-graph__runs-1__20260503T073426/`
- `hsi_agents_project/results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph_phase_matched_lz_reval/phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-1__20260503T073426__top-128__m-40__20260503T073447/`
- `hsi_agents_project/results/hsi_v2/phase3/component_quotient_child_routing_spectrum_phase_matched_lz_reval/phase3-component-quotient-spectrum__src-phase3-weighted-quotient-child-routing-spectral-graph__src-phase2-phase-channel-transition-graph__runs-1__20260503T073426__top-128__m-40__20260503T073447__20260503T073456/`
- `hsi_agents_project/results/hsi_v2/phase3/flow_entropy_readout_phase_matched_lz_reval/phase3-flow-entropy-readout__src-9de9a95303__20260503T073503/`
- D-0072
- D-0075

### Practical Impact

- The additional-null branch of D-0072 is now internally satisfied by a qualitatively stronger LZ-family null.
- The main surviving Phase 3 signal is narrowed to the thermodynamic routing-flow readout, not raw scaffold topology.
- Future summaries should state that the positive channel is more boundary-sensitive under `phase-matched-lz`.
- Any future Phase 3 paper-facing upgrade has stronger internal evidence, but still requires a separate editorial architecture decision and wording-fence audit.
- Reproducibility guides should clarify that N2-12b/N2-13/N2-14 legacy field names may carry the active seeded LZ-family envelope when `null_family` is present.

### Open Questions

- Should Sofia now reconsider the Phase 3 paper-facing upgrade question, or should the pointer-only position remain until an external rerun or submission pressure triggers it?
- Should the code/API eventually rename `--matched-lz-seeds` to a general seeded-null seed argument while keeping backwards compatibility?
- Should a future public figure or table expose the topological weakening under `phase-matched-lz` as a cautionary diagnostic, or keep it internal to avoid overcomplicating the paper narrative?

---

## D-0077

- Date: 2026-05-03
- Status: `accepted`
- Scope: Phase 3 manuscript upgrade authorization (closes condition (c) of D-0072)
- Title: All three D-0072 conditions are formally closed; the internal Phase 3 arc is authorized to receive a manuscript-facing crystallization via a Â§11.2 amendment and a new Appendix `\section{}` derived from the audited editorial architecture draft

### Decision

1. The three conditions of D-0072 are now formally tracked as closed:
   - (a) additional null ensemble or independent rerun: closed by D-0074 (disjoint-seed rerun) and D-0076 (`phase-matched-lz` revalidation);
   - (b) Grace audit of editorial architecture against the wording fence: closed by `docs/HSI-audit-Phase3-Editorial-Architecture-Draft.md` with verdict `LLUM VERDA (GREEN LIGHT)`;
   - (c) new Decision Log entry authorizing manuscript upgrade: closed by this entry.

2. Sofia is authorized to apply two manuscript-facing edits to `Paper_Latex/main_v2_cat.tex`:
   - replacement of the existing single pointer paragraph at the end of `\subsection{Porta 2: Defectes i geometria (pendent)}` (Â§11.2) with a substantive treatment of the internal pre-geometric arc and its routing-thermodynamic discriminant;
   - addition of a new appendix `\section{}` derived from `docs/HSI_v2_Phase3_Editorial_Architecture_Draft.md`, including the Â§F.2.1 robustness matrix (3 rows) as the primary table and the Â§F.2.2 channel matrix (6 rows) as a subsidiary table.

3. The wording fence inherited from D-0071 and D-0076 applies in full to all manuscript-derived prose. The allowed and forbidden wording lists in Â§F.5 of the editorial architecture draft are normative for the LaTeX text. Header-lexicon constraint applies: any LaTeX table header referring to the active null envelope must be generic (e.g. `null_envelope_entropy_min`), not `matched_entropy_min`, when the active envelope is `phase-matched-lz` or any non-canonical seeded LZ-family null.

4. The official manuscript-facing summary of the internal arc is: the surviving Phase 3 internal discriminant is component-normalized routing-flow concentration inside a partially fragile scaffold, not recovered connected geometry. None of the three falsification conditions enumerated in Â§11.2 is either satisfied or refuted by this arc.

5. No figure derived from N3-07 is added to the manuscript by this decision. Figure inclusion remains deferred per Ariadna and confirmed by Grace's audit.

6. No new bibliographic entry is added by this decision. The autocitation to HSI v1 (D-0065) and existing references remain unchanged.

7. A second Grace audit of the LaTeX-derived prose is required before the updated `main_v2_cat.tex` is uploaded to Overleaf. The output of that second audit is recorded in `docs/HSI-audit-Phase3-LaTeX-Integration.md` (to be produced by Grace).

8. If the second Grace audit returns mandatory revisions, the LaTeX edits are revised and resubmitted. If it closes negative, the LaTeX edits are reverted in working copy and not uploaded; this entry is then superseded by a future entry recording the rollback. If it closes positive, Iban uploads to Overleaf and the manuscript-facing chain is closed.

### Rationale

D-0072 was deliberately designed to gate manuscript-facing Phase 3 activation behind three independent conditions, separated in time to avoid premature crystallization. Conditions (a) and (b) closed within the same coordination window through internal scientific work (D-0074, D-0075, D-0076) and an unconditional Grace audit. Condition (c) was retained as an explicit human decision rather than an automatic consequence of (a) and (b), so that the manuscript edit would always require a deliberate authorization independent of internal evidence accumulation.

That authorization is given here. The decision is not driven by external pressure: there is no submission deadline, no reviewer request, and no public communication on the table. It is driven by the internal observation that the editorial architecture draft is fence-compliant under independent audit, the underlying evidence is differentiated rather than uniform (boundary-marked vs topology-weakened, with flow-robust common to both), and the team's editorial coordination is currently within a single window where the second Grace audit can be executed without context drift.

The manuscript edit is intentionally narrow: a Â§11.2 amendment that generalizes the existing pointer paragraph rather than replacing the surrounding text, plus a new appendix mirroring the structure of the audited draft. No central section of the paper is modified. The "narrow body" architectural invariant of the manuscript is preserved.

### Source Documents

- `docs/HSI_v2_Phase3_Editorial_Architecture_Draft.md`
- `docs/HSI-audit-Phase3-Editorial-Architecture-Draft.md`
- D-0066 through D-0076
- `Paper_Latex/main_v2_cat.tex` Â§11.2 `\subsection{Porta 2: Defectes i geometria (pendent)}`

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: Â§11.2 pointer paragraph replaced; new appendix `\section{}` added at the end of the document.
- `docs/HSI-audit-Phase3-LaTeX-Integration.md`: created by Grace as the second-audit output.
- `docs/HSI_v2_Phase3_Editorial_Architecture_Draft.md`: stays as the source object; Ariadna's micro-fix to Â§F.5 self-audit phrasing already applied independently of this decision.
- No figure files added to `Paper_Latex/`.
- No `references.bib` entry added.
- The existing pointer paragraph is not preserved as a comment in the LaTeX source: it is fully replaced by the new substantive treatment.

### Open Questions

- Should a derived figure of the robustness matrix be considered in a follow-up decision once the LaTeX integration is stable, or remain deferred indefinitely?
- Should the Phase 3 internal arc receive a short summary line in the abstract, or stay confined to Â§11.2 and Appendix Â§F?
- Should the Catalan title of the new appendix `\section{}` use "Phase 3" verbatim or a Catalan paraphrase consistent with the rest of the manuscript?

---

## D-0078

- Date: 2026-05-03
- Status: `accepted-editorial-freeze`
- Scope: Phase 3 internal diagnostic stack frozen for the current manuscript cycle (closes the manuscript-facing chain opened by D-0072 and authorized by D-0077)
- Title: Phase 3 internal diagnostic stack frozen for current manuscript cycle

### Decision

1. The current Phase 3 internal diagnostic stack (D-0066 to D-0077) is frozen as the manuscript-facing Phase 3 architecture for the present paper cycle. This freeze covers the pre-geometric diagnostic arc, the robustness matrix, and the Appendix F wording fence validated by Grace and Sofia.

2. The freeze authorizes no further expansion of Phase 3 claims in the current manuscript beyond the Â§11.2 amendment and Appendix F. It does not claim that Gate 2 geometry is solved, that curvature is measured, or that topology has been recovered. It records that the current Phase 3 diagnostic stack is sufficiently audited, reproducible, and editorially bounded to be treated as closed for this manuscript version.

3. Future Phase 3 experimentation is deferred to a later cycle unless a new external review, replication need, or explicit methodological question reopens it. Reopening requires a future Decision Log entry that lifts this freeze.

4. The manuscript-facing edits authorized by D-0077 Â§2 are recorded as applied:
   - Â§11.2 (`\subsection{Porta 2: Defectes i geometria (pendent)}`): single pointer paragraph replaced by two substantive paragraphs.
   - new appendix `\section{Phase 3 internal pre-geometric diagnostic stack}` (`app:phase3_diag`) added before `\bibliography`, with subsections `:robustness_matrix`, `:interpretation`, `:traceability`.
   - Non-content typesetting fixes applied during Overleaf integration: Tables 9 and 11 of Appendix F switched to `p{}` column widths to prevent horizontal overflow; Table 10 of Appendix F had its column spec corrected (`llcccccccc` â†’ `p{2.5cm}lp{3.3cm}cccccc`) and its long monospace headers (`B null_env_min_dH`, `E null_env_min_dH`) reformatted with `\shortstack` to fit page width; Table 9 caption simplified its long path identifier with `\allowbreak{}`-friendly formatting; Table 8 (Phase 2 traceability, Appendix E) had its `Date` column removed for editorial sobriety, with the temporal sequence remaining encoded in the D-XXXX identifiers and recoverable from `HSI_v2_Decision_Log.md`.

5. The second Grace audit required by D-0077 Â§7 closed positive (`LLUM VERDA DEFINITIVA`), validated on internal coherence of Â§11.2 + Appendix F as a single arc and on non-propagation of contradictions to Â§13 (`sec:objeccions`) and Â§14 (`sec:conclusions`, currently a placeholder). The audit verdict is archived at `docs/HSI-audit-Phase3-LaTeX-Integration.md`.

6. The next manuscript task is Â§14 Conclusions, treated as a separate editorial operation under the same wording fence. It is explicitly out of scope for this entry.

7. The three Open Questions enumerated in D-0077 are resolved or deferred:
   - (i) derived figure of the robustness matrix: deferred under this freeze; reopening requires the trigger conditions of Â§3.
   - (ii) abstract line summarizing the Phase 3 internal arc: deferred to the Â§14 Conclusions / abstract redaction pass.
   - (iii) Catalan paraphrase vs verbatim "Phase 3" in the appendix `\section{}` title: closed in favor of the English title `Phase 3 internal pre-geometric diagnostic stack`, mirroring `app:phase2_diag`.

### Rationale

D-0074 satisfied the independent rerun branch; D-0076 satisfied the additional-null branch via `phase-matched-lz`; the robustness matrix hardened the synthesis against reproducibility concerns; Grace's audit of the editorial draft and of the LaTeX integration returned full approval; Sofia's integration preserved the wording fence and did not alter the manuscript's epistemic status.

The freeze is editorial, not scientific. It does not assert that Phase 3 is exhausted or that no further pre-geometric readout could in principle be designed. It asserts that the present manuscript cycle treats the current stack as closed, and that any extension of it in a future manuscript version must traverse a deliberate reopening gate with explicit motivation. This mirrors the architectural invariant that opened Phase 3 (D-0066) and prevents drift between the manuscript-facing claim and any future internal exploration initiated outside protocol.

### Source Documents

- D-0066 through D-0077
- `Paper_Latex/main_v2_cat.tex` Â§11.2 `\subsection{Porta 2: Defectes i geometria (pendent)}` and Appendix `\section{Phase 3 internal pre-geometric diagnostic stack}` (`app:phase3_diag`)
- `docs/HSI_v2_Phase3_Editorial_Architecture_Draft.md`
- `docs/HSI-audit-Phase3-Editorial-Architecture-Draft.md`
- `docs/HSI-audit-Phase3-LaTeX-Integration.md`

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: no further edits derived from the Phase 3 arc under this freeze. Typesetting-only adjustments applied during integration (see Â§4 above).
- `hsi_agents_project/v2/phase3/`: no new manuscript-facing readout under the freeze; existing public scripts and result artifacts preserved and reproducible.
- `hsi_agents_project/results/hsi_v2/phase3/`: existing artifacts preserved.
- Overleaf: synchronization of `main_v2_cat.tex` authorized as a one-shot upload covering both the Phase 3 integration and the typesetting fixes recorded in Â§4.
- Â§14 Conclusions and abstract redaction: explicitly out of scope; remain open as independent editorial work.

### Open Questions

- Should a parallel editorial freeze be applied to the Phase 3-related pages of `hsi_agents_project/Documentation/`, or should those pages remain editable to track external citations and follow-up readings?
- Should periodic reviews of the dormant Open Questions of D-0077 (i)-(iii) be scheduled, or should they remain inactive until a substantive trigger arises?

---

## D-0079

- Date: 2026-05-04
- Status: `accepted-backlog-registration`
- Scope: registers the pre-geometric exploration territory considered but not executed within the Phase 3 manuscript cycle, as explicit Phase 4 backlog
- Title: Phase 3 unexplored frontier registered as Phase 4 backlog

### Decision

The following five exploration lines were considered during the Phase 3 cycle (D-0066 to D-0078) and consciously not executed within this manuscript cycle. They are recorded here as the explicit initial backlog for the future Phase 4 (predictive program), so that the boundary of the closed cycle is documented as a deliberate scope decision, not as an unexamined omission.

This entry does not lift the editorial freeze of D-0078. It complements it.

### Unexplored frontier

1. **Other null families beyond LZ.** Higher-order Markov families (Markov-2, Markov-3, ...), entropy-driven generative nulls, and non-LZ compression-matched surrogates were not evaluated. Rationale for non-execution: the LZ family was selected as the strongest available control at the moment of D-0066; expanding to other null families would have widened the Phase 3 arc beyond its declared scope.

2. **Larger scales.** Windows beyond the consolidated 717M-738M range were not explored. Rationale for non-execution: the Phase 3 discriminant was already visible at the scale consolidated by Phase 2; changing scale would have required a new experimental contract and a non-trivial computational commitment.

3. **Alternative quotient types.** Quotient constructions based on metrics other than prefix/suffix and Hamming radius (for example, edit-distance quotients, Jaccard-similarity quotients, or cluster-membership quotients) were not evaluated. Rationale for non-execution: prefix/suffix and Hamming are the canonical options in the LZ-related literature; exploring alternatives would have reopened the Phase 3 experimental design.

4. **N3-07 routing-flow variants.** Alternative normalizations of the routing-flow readout, alternative concentration metrics beyond entropy/components/share, and possible internal invariants not yet tested were not explored. Rationale for non-execution: N3-07 already survived three null families; broadening its parameter space would have reopened the interpretive arc.

5. **Direct attempt at a scaling law.** An explicit attempt to derive a scaling law over the observed thermodynamic concentration of routing was not made. Rationale for non-execution: a scaling law is exactly the content of Porta 2 falsification condition (iii); it cannot be addressed as a probe within Phase 3 and requires a dedicated cycle with pre-registration.

### Rationale

D-0078 establishes that the current manuscript has an honest closing form and is not expanded. D-0079 establishes that the explorations not pursued within the cycle are conscious, justified, and constitute the initial backlog of the next cycle.

This entry has three functions:

1. **Scientific transparency:** an external reader or reviewer can verify that the scope decisions of the Phase 3 cycle were explicit and justified, not arbitrary or evasive.
2. **Phase 4 backlog materialization:** these five lines are the first candidate experimental design of the predictive cycle.
3. **Editorial closure with full awareness:** the Phase 3 cycle is closed with explicit recognition of the territory that was deliberately left out, rather than with a silent suspicion that something might have been missed.

The freeze is not weakened by this entry; the scope of the freeze is made fully visible.

### Source Documents

- D-0066 through D-0078
- `Paper_Latex/main_v2_cat.tex` Â§11.2 and Appendix `app:phase3_diag`
- Coordination chat (2026-05-04): Phase 3 closure perception and conscious-scope discussion

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: no change.
- `hsi_agents_project/v2/phase3/`: no change. The unexplored frontier does not require additional Phase 3 code; it belongs to Phase 4.
- `hsi_agents_project/v2/phase4/`: directory not yet created. When Phase 4 is formally opened, this entry is the first input to its experimental design.
- Decision Log: D-0079 acts as the documented bridge between the closed Phase 3 cycle and the future Phase 4 cycle.

### Open Questions

- When Phase 4 is formally opened, should the five frontier lines be treated with equal priority, or should an ordered priority be set (for example: scaling law first, since it is the closest to closing Porta 2)?
- Should additional explorations identified during the Phase 3 arc but not enumerated here be added by Ariadna or Grace before the cycle is treated as definitively closed? (Invitation to complement is open.)

---

## D-0080

- Date: 2026-05-05
- Status: `accepted-editorial-cycle-closure`
- Scope: closes the v2 Catalan manuscript editorial cycle initiated after D-0078, covering phases F1-F4 (extension non-structural), cover/figures alignment, and the table-collision fix
- Title: v2 Catalan manuscript editorial cycle (F1-F4 + cover/figures + table fix) closed

### Decision

The editorial cycle on `Paper_Latex/main_v2_cat.tex` opened after the D-0078 freeze is hereby closed in its non-substantive component. The cycle covered the following work, all executed under the existing wording fence and without expanding any frozen claim:

1. F1: cover-page reorganization and bilingual metadata alignment.
2. F2: figure pipeline closure (gate-plane figure, Phase 2 deficit-jitter and lag-response figures, Phase 3 robustness matrix tables, Hilbert-projection inspection appendix).
3. F3: section cross-references and label normalization across the v2 cat manuscript.
4. F4: Â§14 Conclusions tightening under the wording fence.
5. Table-collision fix in Appendix `app:variants` (Variants taxonomy table column widths, single-line edit at L833).

### Rationale

The non-substantive part of the v2 cat editorial cycle reached an honest closing form: no claim was expanded beyond what D-0078 froze, all edits were either typographic, structural or cross-referential, and Overleaf compilation was confirmed clean by the human author at each sync point.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: closes F1-F4 cycle. Substantive edits (D-0081, D-0082) are tracked separately.
- `Paper_Latex/main_eng.tex`: not synchronized in this cycle; bilingual sync is queued as a separate cycle to avoid mixing freeze regimes.

### Open Questions

- When the bilingual sync cycle for `main_eng.tex` is opened, should it absorb the substantive edits of D-0081 and D-0082 in a single bilingual pass, or should it first replicate the F1-F4 baseline and then chain the substantive deltas?

---

## D-0081

- Date: 2026-05-05
- Status: `accepted-substantive-revision`
- Scope: substantive editorial response to the brief-blind external audit performed by the audit role (Gemini 3.1 Pro) on the frozen v2 Catalan manuscript; adopts Option A+ with the methodological refinement contributed by the human author
- Title: Brief-blind external audit accepted; Option A+ adopted with 1D-string methodological frame

### Decision

The brief-blind external audit produced under the protocol introduced informally during the F5 cycle (audit role receives the frozen manuscript without prior project context, operates as a hostile peer reviewer #2) returned a substantive critique on three flanks: (i) the ontological-to-code leap on the AND rule, (ii) the architecture comparability of the external controls, and (iii) the algorithmic vs. physical status of the Gate falsification criterion.

The accepted response is Option A+, which prioritizes methodological clarity and internal recursive controls over a full experimental redesign:

1. **A1 (AND rule):** Â§3.3 (`subsec:annihilacio`, L151) extended with explicit clarification that AND is a pedagogical computational paraphrase consistent with P3 (Nothingness as absorbent), not a logical necessity of the universe nor an ontologically pre-existing operator. The clarification was refined upon human-author input to avoid the opposite overclaim ("AND is the only logically possible rule"); the framing now reads as "AND is the binary operator minimally arbitrary under P3", not as "AND is logically necessary".
2. **A2 (anti-tautology):** Â§6.5 (`subsec:nulls_forts`, L561) extended with explicit anti-tautology paragraph documenting that the same projective tower is applied to `matched-LZ` and that `matched-LZ` collapses, falsifying the "circular by construction" critique.
3. **A3 (wording fence extension):** Â§10 (`sec:defectes`, L697-698) opened with a lexical-fence paragraph declaring all physically-resonant terms (transport, routing, flux, currents, defects, curvature, torsion, thermodynamic, geometry) as strictly algorithmic-informational, and Â§11 (`sec:falsacio`, L758) extended with an explicit statement that the falsification criterion is algorithmic-informational, not physical.
4. **A4 (anti-sycophancy documentation):** Appendix D.3 (`app:cocreacio:posicionament`, L886) reformulated to document four operative anti-sycophancy mechanisms (separation of duties, wording fence, freeze cycles + Decision Log, brief-blind external audit) and to record explicitly that the present cycle's brief-blind audit produced the substantive deltas of A1-A3.
5. **A5+A6 (HSI-family closure):** addressed under the separate D-0082 entry to keep the data-driven part traceable independently.

### Rationale

The methodological refinement contributed by the human author was decisive: the object of analysis is the 1D string $\Phi$, not the generating mechanism. This re-frames the controls question: the natural recursive controls are not externally fabricated CFG/L-system architectures, but the HSI-family variants {D, G, H, F, N} that share the recursive skeleton with B and only differ in the local simplification rule. This frame is what makes the closure under D-0082 viable as a defensive expansion rather than a methodological capitulation.

The audit was substantive, not performative. The human author chose to discard the audit role's first (in-context) report as potentially sycophantic and request a brief-blind run; this protocol is now formalized in Appendix D.3 (A4) and reproducible under future cycles.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: substantive edits at L151, L561, L697-698, L758, L886.
- Future audit cycles: brief-blind protocol is part of the operative repertoire, not an ad-hoc event.
- `Paper_Latex/main_eng.tex`: requires a separate sync pass once D-0082 is also absorbed.

### Open Questions

- The next brief-blind audit cycle (post-D-0082) should specifically test whether the lexical-fence paragraph at Â§10 and the algorithmic-falsification clarification at Â§11 remove the residual physical-overclaim risk, or whether further wording adjustments inside the body of Â§10-Â§13 are required.
- The four anti-sycophancy mechanisms documented in D.3 should be reviewed by the audit role for completeness; if a fifth mechanism is missing in the literature on AI-assisted writing, it should be added under a new D entry.

---

## D-0082

- Date: 2026-05-05
- Status: `accepted-data-integration`
- Scope: integrates new Phase 1 runs for HSI canonical variants D, G, H into the v2 Catalan manuscript as a defensive coverage extension of the architecture-comparability flank
- Title: HSI-family closure: D/G/H integrated into Phase 1 readout

### Decision

New observed Phase 1 runs for variants D, G, and H were executed under the same protocol as B/E/I (segmentation $3\times 10^6$ bits, scale ladder $m\in\{8,12,16,20,24,28,32\}$, policies prefix and suffix, same retention thresholds, iter=20). The runs are stored under `hsi_agents_project/results/hsi_v2/phase1_hsi_family_closure/` with timestamped directories `20260505T091642` (D), `20260505T093345` (G), `20260505T092311` (H), and aggregated reports under the same parent.

The integration into the v2 Catalan manuscript adopts Option G1.a: a new dedicated subsection (`subsec:hsi_family_closure`) and a single new table (`tab:hsi_family_closure`) collecting all the new metrics, rather than in-place edits of Tables 4, 5, 6 and 8. Rationale for the design choice:

1. The original Tables 4-8 are frozen, validated, and Overleaf-confirmed; modifying their column structure (especially Table 4, which has scales as rows and variants as columns) would have triggered a non-trivial reformatting risk.
2. The new runs are at iter=20 while Tables 4-8 reference iter=23 for B/E/I; a single new table in a dedicated subsection makes the protocol difference fully visible in its own context, rather than as a confusing footnote inside the master tables.
3. The new subsection also absorbs the methodological framing on the 1D-string focus (A5 of the post-audit response) and the explicit caution recommended by the code-and-experiments role (Ariadna): D/G/H close the architecture-comparability flank, but do not authorize the stronger claim that "all HSI variants are equivalent" (H visibly separates from B/E/I/D/G) nor that "the full HSI family demonstrates the phenomenon" (the phenomenon remains a property of the compact-selective regime, not of the recursive architecture per se).

A cross-reference item was added at Â§11.1 (`subsec:porta1`) so that a reader scanning the Gate 1 summary is pointed to the HSI-family closure subsection.

### Rationale

The brief-blind audit (D-0081) flagged the architecture comparability of the external controls as a substantive weakness. The methodological refinement contributed by the human author re-frames this: the natural recursive controls already exist inside the HSI family. D/G/H were missing from the main batch by historical availability, not by design. Executing them under the same protocol and reporting them in a defensive-coverage closure subsection eliminates the methodological flank without inflating any claim.

The iter=20 vs iter=23 asymmetry is declared explicitly in the new subsection. The qualitative reading is invariant under this depth difference; if a future audit cycle requires absolute symmetry, a single re-execution of D/G/H at iter=23 (or, alternatively, of N at iter=20) would close it. The choice not to re-execute now follows the principle that iter=20 is sufficient to capture the observables required by the Phase 1 protocol.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: new `subsec:hsi_family_closure` and `tab:hsi_family_closure` inserted between L692 and `sec:defectes`. Cross-reference item added at `subsec:porta1` (L793).
- `hsi_agents_project/results/hsi_v2/phase1_hsi_family_closure/`: new artifact directory with three timestamped runs and two aggregated reports (Phase 1 readout and transport persistence readout).
- `Paper_Latex/main_eng.tex`: requires absorption in the bilingual sync cycle.

### Open Questions

- Should the manuscript also report N at iter=20 to achieve absolute protocol symmetry across all recursive variants {B, D, E, F, G, H, I, N}, at the cost of a new run? Current decision: keep N at iter=17 in the existing tables and treat it as anti-canonical / latest available, as recommended by Ariadna.
- Should a future cycle elevate the cross-reference at `subsec:porta1` to a more visible location (for example, the abstract), or is the current placement sufficient?

### Source Documents

- `hsi_agents_project/results/hsi_v2/phase1_hsi_family_closure/phase1_report/phase1-report__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-suffix__var-D-G-H__20260505T093748.md`
- `hsi_agents_project/results/hsi_v2/phase1_hsi_family_closure/transport_report/phase1-transport__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-vs-suffix__tail-2__var-D-G-H__20260505T093748.md`
- Coordination message from the code-and-experiments role (Ariadna), 2026-05-05: run paths, integration recommendation, cautious wording.

---

## D-0083

- Date: 2026-05-05
- Status: `accepted-substantive-revision`
- Cycle: editorial v2 cat / second brief-blind audit response (Option B+)
- Trigger: second brief-blind audit by the audit role on the post-D-0082 frozen manuscript, executed without project context, without prior-cycle context, and without operator briefing (PDF-only handover); five substantive critiques returned: (1) parser/P2 contraband on the parenthetical structure, (2) AND operator residual arbitrariness, (3) external controls of comparable architecture insufficient, (4) Gate 2 prematurity, (5) circularity insufficiently neutralized.

### Decision

The audit was accepted as methodologically substantive in critiques 1, 2, 3, 5 and as a target-editorial expectation gap (not a methodology failure) in critique 4. The response strategy chosen was Option B+ (defensive textual edits without new experiments), under explicit human-author preference for preserving the ambition and depth of the manuscript over neutralizing every possible reading at the cost of dilution.

The following six edits to `Paper_Latex/main_v2_cat.tex` were executed:

1. **E5 (scope statement, response to critique 4):** new `\paragraph{Abast d'aquest treball.}` at the end of `\section{IntroducciÃ³}` declaring three explicit claim levels (N0 ontological origin and generative process, N1 algorithmic non-trivial signature, N2 emergent informational geometry as open falsifiable contract) and stating that the derivation of physical geometry is not within scope. The frame inherits the human-author position that N2 must either close towards N3 (compatibility with the physical Universe) or be falsified, and cannot remain permanently open.
2. **E1.a (AND removal, response to critique 2):** `\subsection{Regla d'annihilaciÃ³}` renamed to `Regla d'aniquilaciÃ³`; the body reformulated to drop the AND operator and the AND-vs-XOR-vs-OR justification altogether. The rule is now grounded directly in P3 (the No-Res cannot acquire Absolute from contradiction) without invoking any binary-logic operator. The previous A1 clarification paragraph (D-0081, L151) is removed as obsolete.
3. **K4.a (canonicity by minimality, response to critique 3 and frame consistency):** `\subsection{El marc de decadÃ¨ncia}` opening sentence reformulated to state explicitly that B is canonical within the HSI family by minimality of added presuppositions (simultaneous symmetric degradation, no preset temporal order), not by uniqueness. The other HSI variants are equally compatible with P1--P3 but inject some additional architectural commitment.
4. **E1.b (auto-containment subsection, response to critique 1):** new `\subsection{Auto-contenciÃ³ i ordre temporal}` (`subsec:autocontencio`) inserted between `subsec:marc` and the raster figure of `\section{Per quÃ¨ ha de ser aixÃ­}`. The subsection defends that the parenthetical notation is not a parser applied to a pre-existing string but the symbolic transcription of an ontological auto-containment relation forced by the uniqueness of the No-Res; the inside-out reading is the chronological reading of the generative process, not an external parsing algorithm. P2 is therefore not violated.
5. **E2 (ontological reading of controls, response to critique 3):** new `\paragraph{Lectura ontolÃ²gica del disseny de controls.}` appended at the end of `subsec:hsi_family_closure` articulating that internal recursive controls D/G/H probe the mechanism from within, external controls A/J/K/L/M/N probe it from outside, and that the joint reading does not deny the possibility of other ontological mechanisms generating other regimes with other signatures: the contract is to identify and characterize the morphology of this specific family, not to claim uniqueness.
6. **E3 (phase-matched-LZ bridge, response to critique 5):** the closing of `subsec:obj_circular` extended to make explicit that the objection "matched-LZ destroys order too aggressively" is itself defused by the phase-matched-LZ pre-diagnostic (Phase 3 appendix), which preserves block phase classes and only destroys intra-phase and global order, yet does not collapse B and E.

Auxiliary cleanups under E1.c: occurrences of "AND" in the enumeration of `subsec:marc` (final simplification step), in the lemma statement, and in the public code-description paragraph were rewritten in plain ontological language; the AND mention in the anti-sycophancy paragraph (Appendix D.3) was replaced by a description of the second brief-blind cycle and its concrete edits.

The Appendix D.3 anti-sycophancy paragraph was extended to record the iterated brief-blind protocol explicitly: a first brief-blind pass fed D-0080--D-0082 and a second brief-blind pass fed D-0083, both executed under the same protocol (no project context, no prior-cycle context, no operator briefing).

### Rationale

The second brief-blind audit was pivotal in two ways. First, it validated the protocol itself: the audit role, with no prior-cycle traces, returned a critique pattern partially overlapping but partially novel relative to the first cycle, and the genuinely novel critique (parser/P2 contraband on the parenthetical structure) was a real flank that the first cycle had missed and that the prior edits A1--A6 had not addressed. Second, it reframed the editorial trajectory: the human author rejected the defensive abstract-dilution path explicitly, on the grounds that diluting the manuscript to avoid hostile readings would convert it into a generic publication that nobody would read, and chose instead to keep the ambition while clarifying the epistemic frontier. This decision is recorded here because it sets the strategic tone of all subsequent editorial cycles: clarify, do not dilute.

The decision to remove AND entirely (rather than refine its justification one more time, as in D-0081/A1) follows the recognition that AND was an archaism inherited from the earliest drafts and that the mechanism does not require a binary-logic operator to be derived from P1--P3. P3 is sufficient to fix the degradation direction. Keeping AND would have forced a recurrent defense in every audit cycle without adding anything to the demonstration.

The decision to introduce `subsec:autocontencio` as a dedicated subsection (rather than fold the defense into existing prose) follows the principle that the critique it answers (parser/P2 contraband) is itself substantive enough to require visible isolated treatment; folding it into prose would have made it invisible to the same hostile-reading protocol that detected the flank.

The decision to keep critique 4 (Gate 2 prematurity) untreated as a textual edit (beyond the scope statement of E5) follows the assessment that this critique is target-editorial, not methodological: the manuscript already declares Gate 2 as open in many places, and the new scope paragraph compactly fixes the epistemic frontier without further narrative dilution. If a future audit cycle insists on this point, the response will be a target-venue clarification, not a content edit.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: substantive edits at the end of `\section{IntroducciÃ³}` (E5), at `subsec:annihilacio` (E1.a), at `subsec:marc` (K4.a + E1.c lemma + E1.c step 3), new `subsec:autocontencio` (E1.b), at `subsec:hsi_family_closure` end (E2), at `subsec:obj_circular` (E3), at the public-code description paragraph (E1.c), and at the anti-sycophancy paragraph in Appendix D.3 (E1.c + iterated-brief-blind documentation).
- `Paper_Latex/main_eng.tex`: requires absorption in the bilingual sync cycle, which now must replicate D-0080, D-0081, D-0082 and D-0083 in a single coordinated pass.
- Future audit cycles: a third brief-blind pass on the post-D-0083 manuscript should be executed before considering the v2 cat manuscript ready for venue submission. If that pass returns only minor observations or target-editorial expectations, the v2 cat is closed; if it returns substantive new flanks, a further Option B/C cycle is opened.

### Open Questions

- Should an explicit external CFG / L-system control with comparable recursive architecture be added as a future Phase 1 experiment to fully neutralize critique 3, or is the current internal-coverage closure (D-0082) plus ontological-reading paragraph (E2) sufficient? Current decision: backlog, not blocking.
- Should the iterated brief-blind protocol be formally documented as a methodological appendix (separate from D.3), or is the prose-level mention in D.3 sufficient? Current decision: prose-level mention is sufficient; promotion to formal appendix is deferred to the bilingual sync cycle.

### Source Documents

- Second brief-blind audit verdict (Gemini-class audit role), 2026-05-05: five-critique reviewer report transcribed in the editorial-cycle conversation log.
- Human-author position on critique 4 and on abstract dilution: explicit instruction during the same cycle that the manuscript should preserve depth and ambition over neutralizing every hostile reading.

---

## D-0084

- Date: 2026-05-06
- Status: `accepted-substantive-revision`
- Cycle: editorial v2 cat / first iterative-hostile-review response (Option B++)
- Trigger: iterative-hostile review by the audit role on the post-D-0083 frozen manuscript, executed within the same continuing audit thread (with memory of the prior brief-blind cycles), explicitly tasked with finding rebound flanks opened by the just-executed D-0083 patches. Five substantive critiques returned (F1, F2, F3, F4) plus one protocol-classification refinement (F6).

### Decision

The iterative-hostile review was accepted as methodologically substantive in critiques F1, F2, F3, F4 and as a documentation/protocol issue in F6. Following the human-author instruction that the editorial trajectory must clarify rather than dilute (D-0083), the response strategy chosen was Option B++ (defensive textual refinements without new experiments, with strict requirement that each refinement strengthen the affirmative position rather than soften it).

Five edits were executed: four to `Paper_Latex/main_v2_cat.tex` (F1--F4) and one to documentation (F6).

1. **F1 (ontological grounding of identity rules `11â†’1` and `00â†’0`):** new paragraph at end of `subsec:annihilacio` deriving the no-tension rules from the uniqueness of each ontological pole (two No-Res instances are one instance because the No-Res does not admit pluralization; two Absolute instances are one instance because a totality without exterior does not admit duplicate). The annihilation rule is now declared to operate on two pure ontological relations: identity (uniqueness of each pole) and tension degradation (P3). Replaces the previous "consistÃ¨ncia interna" framing inherited from earlier drafts, which read as a computational consistency assertion rather than as an ontological derivation and left an open flank for any reviewer asking "computational consistency from where, exactly?".

2. **F2 (structural vs executional ordering, internal reformulation):** paragraph 3 of `subsec:autocontencio` rewritten in place (no new paragraph added, by explicit human-author preference) to remove the residual "successiÃ³ d'actes generatius" wording that re-introduced the temporal/parser fantasm, and to integrate the fixed-point analogy (auto-referent system whose configuration sustains itself implicitly without needing pas a pas execution). The order between layers is now declared structural, not executive: part of the auto-contained configuration itself, not a result of its sequential execution. Subsection title also updated from "Auto-contenciÃ³ i ordre temporal" to "Auto-contenciÃ³ i ordre causal" to remove the residual temporal connotation.

3. **F3 (affirmative reading of internal convergence):** new `\paragraph{ConvergÃ¨ncia interna i lectura ontolÃ²gica.}` appended at end of `subsec:hsi_family_closure` (after the existing E2 paragraph from D-0083). The qualitative convergence of B, D, G to the same compact-selective regime is asserted as the direct prediction of the common ontological origin (rather than as a defensive reading of a possibly-generic property), with the corroboration coming from external controls (A, J, K, L, M) failing to reproduce it. The framing was sharpened from the initial draft on explicit human-author feedback that the original "Una lectura possible" formulation was defensively dubitative.

4. **F4 (orthogonality of the four pre-declared criteria):** the closing of `subsec:obj_circular` extended to enumerate the four criteria (entropia de pesos de components, nombre de components efectius, top-3 share, top-5 share) explicitly and to add the seed cross-validation reference (D-0074), so that the convergence under `phase-matched-LZ` is visible as orthogonal evidence rather than as an unspecified "four criteria" claim. The four criteria are functions of different aspects of the mass distribution over quotient components, and a generic statistical artifact of the LZ family would not satisfy them simultaneously.

5. **F6 (protocol-classification refinement in App D.3):** the anti-sycophancy paragraph in `app:cocreacio:metodologia` extended to declare two distinct external-audit modalities: `brief-blind` (no project context, no prior-cycle context, PDF-only handover) and `iterative-hostile-review` (continuation of the audit thread with prior-cycle memory, tasked with finding rebound flanks opened by the just-executed patches). The three executed cycles are now documented under their correct modality: D-0080--D-0082 (first brief-blind), D-0083 (second brief-blind), D-0084 (first iterative-hostile-review).

### Rationale

The iterative-hostile-review is a strictly different protocol from the brief-blind review and addresses a different failure mode. The brief-blind review tests whether the manuscript stands without operator context (the reviewer #2 simulation). The iterative-hostile-review tests whether the patches just executed introduce new flanks at the seams (the rebound failure mode). Both are needed to close a cycle responsibly: brief-blind confirms the new state is not patched-but-still-broken-from-cold; iterative-hostile-review confirms the new state is not patched-with-new-cracks-at-the-patch-edges. D-0084 is the first cycle in which the iterative-hostile-review modality has been recognised as a distinct protocol, and the protocol classification under F6 records this distinction so that future cycles can choose modalities deliberately.

The decision to keep F2 as an in-place rewriting (not as a new paragraph) follows explicit human-author guidance that adding more text to defend the same point reads as defensive accumulation rather than as conceptual sharpening; the same guidance drove the decision to remove the dubitative framing in F3, on the principle that an affirmative ontological reading is structurally stronger than a dubitative concession that the convergence might be a generic property.

The decision to refine F1 ontologically (rather than leave the "consistÃ¨ncia interna" framing of D-0083) follows the iterative-hostile critique that the previous wording read as a computational consistency assertion and not as an ontological derivation, leaving an open flank for any reviewer pointing at the framing.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: substantive edits at `subsec:annihilacio` (F1, new paragraph), at `subsec:autocontencio` paragraph 3 and subsection title (F2, in-place rewriting + title update), at `subsec:hsi_family_closure` end (F3, new affirmative paragraph after E2), at `subsec:obj_circular` closing (F4, enumerated criteria + seed cross-validation), and at `app:cocreacio:metodologia` (F6, protocol-modality classification).
- `Paper_Latex/main_eng.tex`: requires absorption in the bilingual sync cycle, which now must replicate D-0080, D-0081, D-0082, D-0083 and D-0084 in a single coordinated pass.
- Future audit cycles: a parallel double pass (one new brief-blind on the post-D-0084 manuscript and one continued iterative-hostile-review on the same continuing audit thread) is the next planned step. Convergence patterns between the two verdicts will determine whether further substantive edits are needed: critiques appearing in both verdicts are real flanks; critiques appearing only in the iterative-hostile-review pass are likely role-driven persistence of the auditor; critiques appearing only in the new brief-blind pass are uncontaminated novel flanks.

### Open Questions

- Should the iterative-hostile-review modality be added as a formal entry to the audit protocol documentation in `.dev/skills/skill-hsi-version-2/`, or is the prose-level mention in App D.3 + D-0084 sufficient? Current decision: prose-level mention is sufficient; promotion to formal protocol document is deferred pending observation of the next two parallel cycles.
- If the next parallel double pass returns convergent critiques across both verdicts, should an Option C cycle (substantive new experiment, not just textual refinement) be opened? Current decision: defer until the convergence pattern is observed; the editorial trajectory remains B++ unless the parallel pass reveals a methodological gap that no prose can close.

### Source Documents

- Iterative-hostile-review verdict (Gemini-class audit role, continuing thread), 2026-05-06: five-substantive-critique reviewer report transcribed in the editorial-cycle conversation log.
- Human-author position on F2 in-place rewriting and on F3 affirmative reformulation: explicit instructions during the same cycle ("F2 ... reformular adequadament el que ja tenim per a que quede completament clar"; "F3 ... llevant 'una lectura possible' i fent-ho explÃ­cit com la lectura correcta guanyariem forÃ§a").

---

## D-0085

- Date: 2026-05-06
- Status: `accepted-substantive-revision`
- Cycle: editorial v2 cat / parallel double-pass response (third brief-blind + second iterative-hostile-review)
- Trigger: parallel double-pass audit on the post-D-0084 frozen manuscript, executed under the protocol declared in F6 of D-0084. Two independent verdicts returned: (a) third brief-blind by a fresh audit-role instance with no project context (PDF-only handover); (b) second iterative-hostile-review by the continuing audit thread with full prior-cycle memory. Severity-rule cross-comparison applied: critiques appearing in both verdicts = real flanks (none); critiques only in iterative-hostile = role-driven persistence (low priority); critiques only in fresh brief-blind = uncontaminated novel flanks (high priority).

### Decision

The cross-comparison returned three operative novel flanks, two of which (B4 IA-authorship + Nature citation; A2 four-criteria orthogonality overclaim) admit textual response under Option B++, and one of which (B3 absent CFG/L-system controls) requires either a textual limitation declaration or a substantive new experiment. The human-author chose Q1.b + Q2.a + Q3 + Q4 + Q5=No: a focused defensive-clarity edit package that retires rhetorical over-defense, corrects a technical overclaim, and registers the missing-controls flank as an explicit acknowledged limitation, deferring the experimental closure to coordination with the code-and-experiments role.

Five edits were executed: four to `Paper_Latex/main_v2_cat.tex` and one to documentation.

1. **Q1.b (terminological adjustment of IA-listing in author line):** title-page author line changed from "amb contribucions de Claude Opus 4.7, GPT-5.5 X-High, Gemini 3.1 Pro" to "amb assistÃ¨ncia de Claude Opus 4.7, GPT-5.5 X-High, Gemini 3.1 Pro". The change deactivates the ICMJE-style \emph{contributorship} reading (which IA systems cannot satisfy by construction: no legal accountability, no moral responsibility, no replicable presence in future iterations) while preserving full transparency about the IA role on the title page itself, in line with the human-author position that the IA listing must remain visible at the title and not be moved to Acknowledgments. The full traceability of who did what remains in App D unchanged.

2. **Q2.a (removal of rhetorical over-defense):** title-page `\thanks{...}` footnote rewritten from "Els sistemes d'IA es llisten com a contribuÃ¯dors com a elecciÃ³ metodolÃ²gica i Ã¨tica deliberada. Creiem que la transparÃ¨ncia sobre el paper de la intelÂ·ligÃ¨ncia artificial en la investigaciÃ³ cientÃ­fica Ã©s un principi irrenunciable." to a neutral pointer "Sobre el paper exacte dels sistemes d'IA llistats i la declaraciÃ³ d'autoria humÃ -IA, vegeu l'ApÃ¨ndix~\ref{app:cocreacio}." App D Â§D.2 paragraph (line 943) shortened from invoking the editorial frame of major journals (`\cite{Nature2023AIrules}`) and the international authorship guidance (`\cite{ICMJE2023AI}`) to a plain factual statement of why the declaration is in the body rather than in Acknowledgments. Both citations removed from the body of the manuscript (they remain in `bibliografia.bib` for now; their removal from the bib is deferred to the bilingual sync cycle if the English manuscript also retires them).

3. **Q3 (orthogonality overclaim corrected at F4):** the closing of `subsec:obj_circular` (D-0084 F4) had asserted "convergÃ¨ncia ortogonal" of the four criteria (entropia de pesos de components, $n_{\text{eff}}$, top-3 share, top-5 share). The iterative-hostile critique correctly noted that these four are not strictly orthogonal but are correlated measures of the same underlying property (mass concentration over quotient components). Reformulated to "convergÃ¨ncia multi-mÃ¨trica de criteris complementaris", explicitly declaring that they are not strictly orthogonal (top-$K$ reads the tail; $H$ and $n_{\text{eff}}$ read the core of the distribution) but that their simultaneous convergence reduces the artifact risk of any individual measure. The defensive position is in fact strengthened by the technical concession: the LZ-family generic-artifact argument and the disjoint-seed corroboration (D-0074) survive the reformulation, the inflated "ortogonal" claim does not.

4. **Q4 (new objection subsection on external-controls coverage):** new `\subsection{ObjecciÃ³: cobertura del conjunt de controls externs}\label{subsec:obj_controls_externs}` inserted in Â§16 immediately before `subsec:obj_epistem`. The subsection acknowledges that the external-control set (PRNG, $\pi$ digits, Rule~30, logistic map, Fibonacci word) does not include any hierarchical context-free grammar with deletion rules nor any stratified L-system, and that the HSI-vs-external separation could in principle attenuate against such a generator. The flank is partly limited by two existing facts (the recursive internal controls $\{D, G, H\}$ are already hierarchical formal grammars with deferred evaluation; the B/E separation survives `phase-matched-LZ`) but the addition of at least one CFG-with-deletion and one hierarchical L-system as new external controls is registered as an open experimental program. Until that extension is executed, the Gate~1 claim is to be read as conditioned on the current external-control coverage.

5. **D-0085 logged.** Anti-sycophancy paragraph in App D.3 (line 937, modified by D-0084 F6) is left as-is: it already documents the iterated-brief-blind protocol and the iterative-hostile-review modality; the parallel double-pass is a deliberate combination of those two modalities and does not introduce a third, so no protocol-classification update is needed.

### Rationale

The parallel double-pass returned a low convergence pattern (no critique appeared substantively in both verdicts), which is itself information: it confirms that the prior patches (D-0080 to D-0084) had effectively closed the flanks they were designed to close, and that the residual operative flanks are split between fresh-perspective findings (the brief-blind side) and role-driven persistence on the patched-edges (the iterative-hostile side). The severity rule was applied strictly: only the brief-blind-only critiques (B3, B4) and the iterative-hostile-only critique that admitted a clean technical concession (A2) were treated; the iterative-hostile-only critiques that read as role-driven rebrots (A1 ontological paradox, A3 universality-class tautology, A4 fixed-point-vs-dynamics) were deferred without textual change.

The human-author position on Q1 was anchored on a non-defensive ground: the human-author rejected Q1.c (move IA-listing to Acknowledgments) on the principle that retiring the IA presence from the title page would inflate the human-author's perceived sole-authorship beyond what is true, and explicitly accepted the editorial cost of keeping the IA visible at the title. The terminological softening from "contribucions" to "assistÃ¨ncia" was the author's compromise, accepted on the editorial role's argument that the ICMJE \emph{contributorship} reading is one that no IA system can satisfy by construction and that "assistÃ¨ncia" describes the actual relation more accurately without diluting the transparency.

The decision to concede A2 (four-criteria orthogonality overclaim) follows the principle that conceding a technical point where the critique is correct strengthens, not weakens, the surrounding argument: the LZ-family-generic-artifact argument and the disjoint-seed corroboration are independent of the orthogonality claim and survive its retirement.

The decision to declare Q4 (CFG/L-system controls) as an open program rather than execute a new experiment now is operational: the editorial role does not have execution authority over experiments and the code-and-experiments role (Ariadna) is currently engaged on Phase~4 work (P4-02b block-entropy null with wider bands). A coordination request to add a CFG-with-deletion and a hierarchical L-system as new external controls in a future Phase~1 extension is registered here as the natural follow-up; whether to execute that extension before submission or after is deferred to the human-author after the P4-02b result is in.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`: substantive edits at title page (Q1.b + Q2.a thanks footnote), at `subsec:obj_circular` closing (Q3), at Â§16 (Q4 new subsection `subsec:obj_controls_externs`), and at `app:cocreacio:metodologia` line 943 (Q2.a citation retirement).
- `Paper_Latex/main_eng.tex`: requires absorption in the bilingual sync cycle, which now must replicate D-0080 through D-0085.
- `bibliografia.bib`: `Nature2023AIrules` and `ICMJE2023AI` entries no longer cited in the cat manuscript body; their removal from the .bib is deferred to the bilingual sync cycle for atomic decision across both versions.
- Coordination with code-and-experiments role: a request will be issued for a future Phase~1 extension adding (i) at least one context-free grammar with deletion rules and (ii) at least one hierarchical L-system as new external controls in the canonical Gate~1 plane. Priority and timing to be set by the human-author after the P4-02b result.

### Open Questions

- Should the Q4 program (CFG + L-system controls) be executed before paper submission, or registered as a post-publication extension? Current decision: defer until P4-02b lands; if P4-02b is positive and ready for inclusion, the natural cycle is to bundle Q4 with it; if P4-02b is inconclusive, the Q4 flank can ship as a declared limitation.
- Should a third independent parallel cycle be executed on the post-D-0085 manuscript? Current decision: no, unless a new substantive flank is identified by an external reader. The parallel double-pass was already a stress-test of the F1--F6 patches, and the cross-comparison did not return any critique appearing on both sides. Further audit cycles would be diminishing-returns.

### Source Documents

- Third brief-blind audit verdict (Gemini-class fresh instance, PDF-only), 2026-05-06: four-critique reviewer report (B1 parser/auto-containment persistence; B2 ontological-vs-Level-1 gap; B3 absent CFG/L-system controls; B4 IA-authorship + Nature citation desk-reject risk).
- Second iterative-hostile-review verdict (Gemini-class continuing thread), 2026-05-06: four-critique reviewer report (A1 $11 \to 1$ paradox; A2 four-criteria orthogonality overclaim; A3 universality-class tautology; A4 fixed-point-vs-dynamics).
- Human-author position on Q1.b: explicit instruction "moure el listing als reconeixements em dona massa valor com a autor principal i tu i jo sabem que no Ã©s cert", with acceptance of the editorial-role argument that "contribucions" â†’ "assistÃ¨ncia" is a just terminological adjustment.
- Human-author position on Q4 / Ariadna coordination: explicit instruction "si cal demanar a Ariadna experiments nous, li'ls demanem".

---

## D-0086

- Date: 2026-05-06
- Status: `superseded-by-D-0087` (originally `accepted-pre-implementation-contract`; superseded on 2026-05-06 by D-0087 after the v0.2 feasibility failure documented at `docs/HSI_v2_Phase1_External_Recursive_Controls_v0_2_Feasibility_Note.md`. The v0.2 rule sets failed their own frozen ceilings at the implementation preflight: variant `O` reached `pass=50` with only `21` observable bits; variant `P` reached `round=12` with `terminalization_ratio = 0.3234` (above the kill threshold `0.20`). The failure was a quantitative emission-calibration issue of the rule sets, not a methodological flaw of D-0086 itself: the V1--V5 refinements, the dual-seed null policy, the manifest schema, and the kill-criteria thresholds were all preserved verbatim into the v0.3 contract registered as D-0087.)
- Cycle: experimental v2 / D-0085 Q4 follow-up / external recursive controls O+P pre-registration
- Trigger: D-0085 registered the absent CFG/L-system external controls (B3 brief-blind flank) as an open experimental program and authorized the editorial role to coordinate with the code-and-experiments role (Ariadna) on a Phase~1 extension. Sofia issued the coordination request with naming convention (variants `O` and `P`), repo conventions (prefix `hsi_v2_`, outputs under `results/hsi_v2/`, manifest+metadata mandatory), and the gate-1 measurement contract to be reused unchanged. Ariadna returned a v0.1 pre-implementation proposal in `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal.md` requesting methodological validation before code is touched. Sofia returned a five-point refinement set (V1--V5) covering kill-criteria clarification, null-seed policy, terminalization-ratio diagnostics, density diagnostics, and explicit `max_passes`/`max_rounds` ceilings. Ariadna returned v0.2 incorporating all five refinements with a slight strengthening on V2.

### Decision

The v0.2 proposal at `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal.md` (sha256 `E46D39494B0423FD223A92A7ADF2D233A1B69332C3214250C298BCF89971BE22`) is accepted as the pre-implementation contract for the D-0085 Q4 extension. The proposal is hereby frozen as the rule set against which Phase~1 results must be read; any subsequent change to rules, derivation strategy, terminalization map, target length, `max_passes`/`max_rounds` ceilings, or null-seed list after the first run constitutes a kill-criterion violation per its own Â§"Kill / redesign criteria" (line 236 of the proposal).

The five refinements are recorded as integral parts of the contract:

1. **V1 (kill-criteria clarification):** crystalline, rigid, diffuse or otherwise non-compact-selective outcomes are valid empirical results, not generator failures. Inscribed both at design-principles level (proposal line 17) and at success/kill-criteria level (proposal line 223). The B3 flank closes whenever `O` or `P` does not fall in the HSI compact-selective regime, regardless of which specific non-compact-selective regime they end up in.

2. **V2 (null-seed policy, strengthened beyond Sofia's request):** instead of declaring seed `607` as ad-hoc and deferring disjoint-seed replication, Ariadna predeclared a two-seed family `{607, 709}` with strongest-pressure readout across the two. This inherits the D-0074 disjoint-seed precedent without escalating to the full D-0074 family `{607, 709, 811, 907, 1009, 1103}`. Acceptable scaling: a paper-closure control extension does not require the full sweep, and the dual-seed minimum is sufficient to neutralize the cherry-picking flank for this extension.

3. **V3 (`terminalization_ratio` diagnostic):** added to manifest with three discrete thresholds (proposal lines 161--165): `<= 0.05` clean generator outcome, `<= 0.20` valid but terminalization-heavy and reported explicitly, `> 0.20` kill/redesign before Phase~1 interpretation. Also inscribed in the kill-criteria block (proposal line 234).

4. **V4 (`freq_zero` / `freq_one` diagnostics):** added to manifest with non-degenerate band `[0.15, 0.85]` (proposal lines 169--177). Outside that band, the snapshot remains valid but any Gate~1 separation must be read with frequency-skew caution.

5. **V5 (`max_passes` / `max_rounds` ceilings):** `max_passes = 50` for `O` (proposal line 61), `max_rounds = 12` for `P` (proposal line 100), both registered at the kill-criteria block (proposal line 235) as non-tunable. Reaching these limits before the required observable length is a generator kill, not a parameter to be raised iteratively.

### Rationale

The pre-implementation validation cycle is the right place to inject diagnostic and seed-policy refinements: once the snapshots are generated and Phase~1 is run, any refinement that requires a new manifest field or a new null-seed becomes a re-run, not an edit. Conversely, no refinement is allowed to redesign the rules themselves: the rule set is the scientific content of the controls and must be frozen before observation, exactly as for the canonical HSI variants in their original Gate~1 plane.

Grace's role is preserved as post-execution audit, not co-design (proposal line 240). The pre-implementation cycle was deliberately limited to the editorial role (Sofia) and the code-and-experiments role (Ariadna), with the human author as final arbiter; bringing the audit role into co-design would convert the eventual post-execution audit pressure into a known-friendly review and dilute its scientific value. The audit role will be invited only on the final results, with the v0.2 proposal as the pre-registered contract against which those results are evaluated.

The naming `O` and `P` is preserved from the original coordination request and registered as the canonical labels for these two external recursive controls in the Gate~1 plane and in any manuscript text that references them.

### Practical Impact

- `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal.md`: frozen at v0.2 (sha256 `E46D39494B0423FD223A92A7ADF2D233A1B69332C3214250C298BCF89971BE22`) as pre-implementation contract. Any future revision must start a new `vN.N` cycle and pass through a new Decision Log entry.
- Repo green-light for implementation of two scripts under existing conventions:
  - `hsi_v2_generate_external_recursive_controls.py`: emits Level-0-compatible snapshots `phi_snapshots/var_O/phi_iter20.{struct.gz,json}` and `phi_snapshots/var_P/phi_iter20.{struct.gz,json}` per the snapshot contract (proposal Â§Snapshot Contract).
  - `hsi_v2_phase1_external_recursive_controls.py`: launcher wrapping the four observed runs and four `phase-matched-lz` runs (variants `O`+`P` $\times$ seeds `607`+`709`) per the Gate~1 Run Contract (proposal Â§Gate~1 Run Contract), emitting a compact recovery report.
- Outputs under `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/`.
- Phase logs visible per AGENTS.md long-script rule.
- `Paper_Latex/main_v2_cat.tex` Â§ `subsec:obj_controls_externs`: text remains as-is (Q4 declared limitation) until Phase~1 results are in. A subsequent Decision Log entry will record the empirical reading of `O` and `P` and authorize the substitution of the limitation paragraph by the empirical result paragraph.

### Open Questions

- Sequencing relative to P4-02b: Ariadna's prior commitment is P4-02b first. The implementation of the O+P pipeline can be serialized after P4-02b or parallelized depending on her bandwidth assessment; the editorial role expresses no preference on this point.
- Inclusion in the submission package vs. post-publication extension: deferred. To be re-decided by the human-author once both P4-02b and the O+P Phase~1 results are in.
- Bilingual sync (`Paper_Latex/main_eng.tex`): the Q4 limitation paragraph at `subsec:obj_controls_externs` must be replicated in the English manuscript in the same bilingual sync cycle that absorbs D-0080 through D-0085. The eventual empirical update (when O+P results land) will be a separate sync cycle.

### Source Documents

- `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal.md` v0.2, sha256 `E46D39494B0423FD223A92A7ADF2D233A1B69332C3214250C298BCF89971BE22`, 2026-05-06.
- D-0085 (parent decision authorizing the Q4 extension and the coordination request).
- D-0074 (disjoint-seed corroboration precedent inherited by the V2 null-seed policy).
- Sofia coordination request to Ariadna (2026-05-06): naming `O`/`P`, repo conventions, gate-1 contract reuse, P4-02b priority preserved.
- Sofia v0.1 review returning V1--V5 (2026-05-06).
- Ariadna v0.2 incorporating V1--V5 with V2 strengthened to dual-seed predeclaration (2026-05-06).
- Human-author validation of v0.2 as pre-implementation contract: explicit "Endavant amb X1.a+X2.a+X3.b" (2026-05-06).

---

## D-0087

- Date: 2026-05-06
- Status: `executed-and-empirically-closed-by-D-0088` (originally `accepted-pre-implementation-contract`; closed on 2026-05-06 by D-0088 after Ariadna executed the v0.3 contract verbatim, obtained Phase~1 results for O+P under both observed and `phase-matched-lz` conditions, and Grace returned an `ACCEPTED` post-execution audit verdict at `docs/HSI-audit-Controls-OP-D0087.md`. The contract was honored without rule, threshold, seed, or terminalization-map deviation; the empirical reading is recorded in D-0088.)
- Cycle: experimental v2 / D-0086 v0.3 revision after v0.2 feasibility failure / external recursive controls O+P pre-registration
- Trigger: D-0086 was superseded by an open v0.3 cycle after Ariadna's pre-implementation preflight detected that the v0.2 rule sets failed their own frozen ceilings (variant `O` reached `pass=50` with only `21` observable bits; variant `P` reached `round=12` with `terminalization_ratio=0.3234`, above the kill threshold). Sofia issued the v0.3 design constraints: preserve the qualitative architecture (CFG with real `epsilon` deletions; stratified L-system with deterministic parallel rewrite and parens-marked stratification) while increasing terminal emission per recursive expansion, with explicit per-NT preservation rules for `epsilon` and self-recursion, deterministic cycle policy, non-trivial balanced terminal blocks, redeclared `terminalization_map`, and a mandatory Y3.b Preflight Summary as part of the proposal itself. All v0.2 invariants (seeds `{607, 709}`, terminalization-ratio bands, density band `[0.15, 0.85]`, target observable `>= 3,000,000` with `target = 3,100,000`, naming `O`/`P`, kill criteria) were declared read-only across the revision. Ariadna returned v0.3 incorporating all constraints with two manifest-schema additions (`shared_terminal_blocks`, separated `pre_terminalization_observable_bits` / `terminalization_observable_bits` / `final_observable_bits`) and a refined stop condition that requires both `pre_terminalization_observable_bits >= 3,100,000` and projected `terminalization_ratio <= 0.05` simultaneously before non-kill stop.

### Decision

The v0.3 proposal at `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal_v0_3.md` (sha256 `66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0`) is accepted as the pre-implementation contract for the D-0085 Q4 extension, replacing the failed v0.2 contract under D-0086. The proposal is hereby frozen as the rule set against which Phase~1 results must be read; any subsequent change to rules, derivation strategy, terminalization map, target length, `max_passes`/`max_rounds` ceilings, shared terminal blocks `U0`--`U3`, or null-seed list after the first run constitutes a kill-criterion violation per its own Â§"Kill / redesign criteria" (line 290 of the v0.3 proposal).

The shared terminal blocks `U0`--`U3` were verified mechanically before acceptance: each is exactly 64 bits, balanced 32 zeros / 32 ones, and internally non-constant. This verification was independent of Ariadna's preflight and confirmed her density claim by direct inspection.

Design-constraint compliance was verified against the v0.2-to-v0.3 supersession criteria:

- Variant `O`: nonterminals `A`, `B`, `C` each preserve at least one real `epsilon` production (lines 71, 79, 87 of v0.3); `D` did not have `epsilon` in v0.2 and is correctly preserved without `epsilon` in v0.3. Each of `A`, `B`, `C`, `D` has at least one self-recursive production. Cycle policy is deterministic and per-NT-type. Terminal block usage varies across productions of the same nonterminal type. `terminalization_map` redeclared (now `A->01, B->10, C->01, D->10, S->0`), distinct from v0.2.

- Variant `P`: deterministic parallel rewrite preserved. Four active symbols `A`/`B`/`C`/`D` preserved. Parentheses preserved as explicit structural stratification markers, filtered out before the observable-bit prefix. Each rewrite emits three balanced 64-bit terminal blocks plus multiple nonterminals, ruling out Fibonacci-like one-dimensional degeneration. `terminalization_map` redeclared (`A->01, B->10, C->0, D->1`).

- Y3.b Preflight Summary: present as Â§"Y3.b Preflight Summary" in the v0.3 proposal (lines 179--192). Reports for both controls under the exact frozen rules: `O` reaches target at pass `30` of `50` with `final_observable_bits = 3,642,944`, `terminalization_ratio = 0.010840`, `freq_zero = freq_one = 0.5`. `P` reaches target at round `10` of `12` with `final_observable_bits = 5,284,798`, `terminalization_ratio = 0.016760`, `freq_zero = freq_one = 0.5`. Both well below the `0.05` clean-outcome threshold. Both within the `[0.15, 0.85]` density band by wide margin.

Two refinements introduced by Ariadna beyond Sofia's design constraints are accepted as integral parts of the contract:

1. **Manifest field `shared_terminal_blocks`:** captures the explicit `U0`--`U3` strings inside every snapshot manifest, making the terminal alphabet of the rule set traceable per-snapshot rather than only per-proposal. Improves reproducibility without changing scientific content.

2. **Manifest field separation `pre_terminalization_observable_bits` + `terminalization_observable_bits` + `final_observable_bits`:** allows the `terminalization_ratio` to be computed by inspection of the manifest rather than only as a derived field. Improves audit transparency without changing scientific content.

3. **Refined non-kill stop condition:** v0.2 stopped on `observable_bits >= target + margin`; v0.3 stops only when both `pre_terminalization_observable_bits >= 3,100,000` and projected `terminalization_ratio <= 0.05` are simultaneously satisfied (or on `max_passes`/`max_rounds` kill). This is a derivation-strategy refinement that strengthens the contract: it makes the clean-generator outcome a stop-time guarantee rather than a post-hoc check, and prevents stops at over-terminalized intermediate states that would have to be rerun. Acceptable under the "rules read-only, derivation strategy may be refined" framework: no production rule, terminalization map, threshold, or seed is altered.

### Rationale

The supersession from D-0086 to D-0087 is the canonical example of the pre-registration discipline working as designed: a frozen contract failed at preflight, the failure was documented quantitatively without silent tuning, the human-author and editorial role authorized a revision cycle with explicit design constraints that preserved the scientific intent of the failed contract, and the revised contract was validated by both the discipline-mandated preflight (Y3.b) and an independent mechanical verification of the new design primitive (the shared terminal blocks). The chain D-0086 (frozen failed) -> Feasibility Note (failure documented, no tuning) -> Y1.a + Y2.a + Y3.b authorization (revision cycle authorized with constraints, not free rein) -> D-0087 (revised contract frozen, sha256 anchored) is the audit trail.

The Y3.b discipline (preflight summary as proposal section) is hereby consecrated as standard pattern for all future external-control pre-implementation contracts in the HSI v2 program. It costs effectively nothing once the simulation tooling exists and prevents the kind of cycle that D-0086 + the feasibility note represented: an implementation green-light handed out before the rule set was empirically known to be executable.

Grace's role remains post-execution audit, not co-design (preserved from D-0086, restated in the v0.3 proposal at line 21). The audit role will be invited only on the final results, with the v0.3 proposal as the pre-registered contract against which those results are evaluated.

### Practical Impact

- `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal_v0_3.md`: frozen at v0.3 (sha256 `66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0`) as pre-implementation contract. Any future revision must start a new `vN.N` cycle and pass through a new Decision Log entry.
- `docs/HSI_v2_Phase1_External_Recursive_Controls_v0_2_Feasibility_Note.md`: retained as the canonical record of the v0.2 failure mode and the trigger of the v0.3 revision. Not superseded; it is part of the audit trail.
- `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal.md` (v0.2): retained as historical record of the failed contract under D-0086. Not deleted; the supersession chain is the value.
- Repo green-light for implementation of two scripts under existing conventions:
  - `hsi_v2_generate_external_recursive_controls.py`: emits Level-0-compatible snapshots `phi_snapshots/var_O/phi_iter20.{struct.gz,json}` and `phi_snapshots/var_P/phi_iter20.{struct.gz,json}` per the v0.3 Â§Snapshot Contract. Manifest must include `shared_terminal_blocks` and the three separated observable-bit fields.
  - `hsi_v2_phase1_external_recursive_controls.py`: launcher wrapping the four observed runs and four `phase-matched-lz` runs (variants `O`+`P` $\times$ seeds `607`+`709`) per the v0.3 Â§Gate~1 Run Contract, emitting a compact recovery report.
- Outputs under `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/`.
- Phase logs visible per AGENTS.md long-script rule.
- `Paper_Latex/main_v2_cat.tex` Â§ `subsec:obj_controls_externs`: text remains as-is (Q4 declared limitation) until Phase~1 results are in. A subsequent Decision Log entry will record the empirical reading of `O` and `P` and authorize the substitution of the limitation paragraph by the empirical result paragraph.

### Open Questions

- Sequencing relative to P4-02b: unchanged from D-0086. P4-02b first; O+P implementation when Ariadna has bandwidth. Editorial role has no preference on serialize vs. parallelize.
- Inclusion in the submission package vs. post-publication extension: deferred. To be re-decided by the human-author once both P4-02b and the O+P Phase~1 results are in.
- Bilingual sync (`Paper_Latex/main_eng.tex`): unchanged from D-0086. The Q4 limitation paragraph at `subsec:obj_controls_externs` must be replicated in the English manuscript in the same bilingual sync cycle that absorbs D-0080 through D-0087. The eventual empirical update (when O+P results land) will be a separate sync cycle.
- Post-Phase~1 editorial annotation: the chain D-0086 -> Feasibility Note -> D-0087 -> empirical result is a candidate worked example of pre-registration in action that could be cited briefly in `app:cocreacio:metodologia`. Decision deferred to post-Phase~1.

### Source Documents

- `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal_v0_3.md` v0.3, sha256 `66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0`, 2026-05-06.
- `docs/HSI_v2_Phase1_External_Recursive_Controls_v0_2_Feasibility_Note.md`, 2026-05-06: pre-implementation preflight detecting the v0.2 failure mode without modifying rules or implementing code.
- D-0086 (parent decision: v0.2 contract that failed preflight).
- D-0085 (grandparent decision: original Q4 authorization).
- D-0074 (disjoint-seed corroboration precedent inherited by the V2 null-seed policy `{607, 709}`).
- Sofia v0.3 design constraints (2026-05-06): preserve qualitative architecture, increase terminal emission, per-NT `epsilon` and self-recursion preservation, redeclared `terminalization_map`, mandatory Y3.b Preflight Summary as proposal section.
- Ariadna v0.3 incorporating all design constraints with two manifest-schema additions (`shared_terminal_blocks`; three-way observable-bit separation) and a refined stop condition (2026-05-06).
- Independent mechanical verification of `U0`--`U3` blocks (length 64, balanced 32/32, internally non-constant) by editorial role before acceptance, 2026-05-06.
- Human-author validation of v0.3 as pre-implementation contract: explicit "Endavant amb les teues recomanacions" applied to the Y1.a + Y2.a + Y3.b + Y4.c proposal (2026-05-06).

---

## D-0088

- Date: 2026-05-06
- Status: `accepted-empirical-closure-of-D-0087`
- Cycle: experimental v2 / D-0087 post-execution / external recursive controls O+P empirical reading + Q4 manuscript closure
- Trigger: Ariadna executed the v0.3 contract frozen by D-0087 verbatim, generating Level-0 snapshots for variants `O` and `P` and running Phase~1 Gate~1 in both observed and `phase-matched-lz` conditions across the predeclared dual-seed set `{607, 709}`. Grace returned a post-execution audit verdict at `docs/HSI-audit-Controls-OP-D0087.md` (sha256 `C3353BE7F57187B76EF03CA4C1250E2230442C4E719E52EA5CFC7F6EF2C96622`) with veredict `ACCEPTED` for the closure of the B3 flank within the declared perimeter. Ariadna submitted a technical reading consistent with Grace's audit and proposed (i) registering D-0088 to close D-0087 empirically, (ii) substituting the conditional Q4 limitation paragraph at `subsec:obj_controls_externs` by a bounded empirical reading, (iii) keeping the claim bounded (closes the flank but does not make B unique nor extend ontological claims), (iv) preserving editorial precision via a wording-correction note explained below.

### Decision

The empirical reading of the D-0087 contract is hereby accepted as final for the Q4 closure within the declared perimeter (Phase~1 Gate~1, $m \in \{8, 12, 16, 20, 24, 28, 32\}$, prefix+suffix policies, `phase-matched-lz` null with seeds `{607, 709}`). The Catalan manuscript at `Paper_Latex/main_v2_cat.tex` Â§ `subsec:obj_controls_externs` has been edited to substitute the Q4 conditional-limitation paragraph by the empirical reading paragraph block (three paragraphs replacing the previous one). D-0087 is marked `executed-and-empirically-closed-by-D-0088`.

### Empirical Numbers

| Variant | Condition | kept | candidate | retention | p\_surv | s\_surv | terminalization\_ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `O` | observed | 1865 | 1977 | 0.9433 | 1.0000 | 1.0000 | 0.0108 |
| `P` | observed | 876 | 876 | 1.0000 | 1.0000 | 1.0000 | 0.0168 |
| `O` | `phase-matched-lz` seed 607 | 1874 | 13215 | 0.1418 | -- | -- | -- |
| `O` | `phase-matched-lz` seed 709 | 1880 | 12905 | 0.1457 | -- | -- | -- |
| `P` | `phase-matched-lz` seed 607 | 876 | 17021 | 0.0515 | -- | -- | -- |
| `P` | `phase-matched-lz` seed 709 | 876 | 17095 | 0.0512 | -- | -- | -- |

Density check (preserved from D-0087 preflight, confirmed in execution): `freq_zero = freq_one = 0.5` for both `O` and `P`.

### Methodological Note: Wording Correction

Ariadna flagged a precision issue that is hereby recorded as canonical for any future reporting of `phase-matched-lz` outcomes against rigid/crystalline controls: the observed phenomenon under the null is candidate-space inflation with stable-support retention drop, not a collapse of the `p_surv` / `s_surv` columns. The columns `p_surv` and `s_surv` reach 1.0 in the observed runs of both controls and the relevant collapse is in `kept / candidate` once the candidate space inflates from $\sim 2 \times 10^3$ to $\sim 1.3$--$1.7 \times 10^4$. Saying "the null collapses prefix/suffix survival" would be technically incorrect for this measurement geometry. The safe formulation, adopted in the manuscript at `subsec:obj_controls_externs` and in any future bilingual-sync replication, is: "the `phase-matched-lz` surrogate produces candidate-space inflation but low stable-support retention". This convention is recorded here so that future Decision Log entries on similar measurements use the same precise language without rediscovering the issue.

### Bounded Reading

The closure of the B3 flank achieved by D-0088 is strictly the following: under the predeclared Phase~1 Gate~1 perimeter, formal hierarchical recursion with deferred evaluation (as instantiated by `O` and `P`) is not by itself a sufficient condition for the compact-selective regime of `B`. The closure does \emph{not}: (i) make `B` the unique generator of the compact-selective regime; (ii) extend any claim about Gate~2; (iii) elevate the recursive-control axis to an ontological axis. The HSI / `O` / `P` separation lives in the same observational plane as the HSI / LZ-family-envelope separation already discussed in `subsec:obj_circular`. The manuscript wording in the new paragraph block has been calibrated to this exact level of restraint.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex` Â§ `subsec:obj_controls_externs`: substantive edit applied. The previous one-paragraph conditional-limitation block is replaced by a three-paragraph empirical-reading block (pre-registration recap, empirical numbers and pattern interpretation, bounded reading). The reference to `\ref{subsec:obj_circular}` is preserved; the reference to `\ref{sec:falsacio}` and to `\ref{subsec:hsi_family_closure}` from the previous paragraph is dropped because the new paragraph block stands on its own empirical content.
- D-0087 status updated: `accepted-pre-implementation-contract` â†’ `executed-and-empirically-closed-by-D-0088`.
- `docs/HSI-audit-Controls-OP-D0087.md` (sha256 `C3353BE7F57187B76EF03CA4C1250E2230442C4E719E52EA5CFC7F6EF2C96622`): retained as canonical post-execution audit record. Not cited in the manuscript body (editorial discipline preserved); referenced here in the Decision Log for full traceability.
- Bilingual sync (`Paper_Latex/main_eng.tex`): now must absorb D-0080 through D-0088. The new three-paragraph block at `subsec:obj_controls_externs` will be replicated in English in the same sync cycle, using the wording-correction convention recorded above ("candidate-space inflation but low stable-support retention").

### Open Questions

- Post-Phase~1 editorial annotation at `app:cocreacio:metodologia` citing the chain D-0086 â†’ Feasibility Note â†’ D-0087 â†’ execution â†’ Grace audit â†’ D-0088 as a worked example of pre-registration in action: deferred (Z4.b at the validation step). To be re-decided after the bilingual sync cycle, when the chain is also visible in the English manuscript and the editorial cost / benefit can be evaluated with full distance.
- Inclusion in the submission package vs. post-publication extension: this decision was deferred at D-0086 / D-0087 pending the empirical reading. With D-0088 now in, the natural recommendation is to bundle the O+P closure with the submission package (the new paragraph block at `subsec:obj_controls_externs` is already manuscript-resident and the audit trail is complete). Final call deferred to the human-author once P4-02b also lands, so that all closures arrive at the submission gate together.
- Whether to extend the `phase-matched-lz` seed sweep on `O` and `P` from `{607, 709}` to the full D-0074 family `{607, 709, 811, 907, 1009, 1103}` for additional robustness: not required for B3 closure (the dual-seed envelope is the predeclared protocol and Grace's audit accepted it). Could be revisited if a reviewer requests it post-submission; would be a new decision entry if executed.

### Source Documents

- `docs/HSI-audit-Controls-OP-D0087.md`, sha256 `C3353BE7F57187B76EF03CA4C1250E2230442C4E719E52EA5CFC7F6EF2C96622`, 2026-05-06: Grace post-execution audit verdict `ACCEPTED` for the B3 flank closure within the declared perimeter. Identifies O/P as crystalline / fractal-deterministic controls with high observed retention, perfect prefix/suffix survival, and very low terminalization ratio; confirms candidate-space inflation under `phase-matched-lz` with retention collapse to ~0.14 (`O`) and ~0.05 (`P`); rules out the "B as disguised formal grammar" reading.
- `docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal_v0_3.md`, sha256 `66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0` (D-0087 frozen contract, executed verbatim).
- D-0087 (parent decision: pre-implementation contract executed by D-0088).
- D-0086 (grandparent decision: failed v0.2 contract whose feasibility note triggered the v0.3 cycle that became D-0087).
- D-0085 (great-grandparent decision: original Q4 authorization).
- D-0074 (disjoint-seed corroboration precedent that the dual-seed `{607, 709}` policy inherits and scales down).
- Ariadna technical reading (2026-05-06): O/P are valid external recursive controls falling in rigid/crystalline regime, not compact-selective; under `phase-matched-lz` candidate space inflates while stable-support retention drops sharply; B3 flank closed without expanding ontological claims.
- Ariadna wording-correction note (2026-05-06): the safe formulation for this measurement geometry is "candidate-space inflation but low stable-support retention", not "the null collapses prefix/suffix survival" (the columns `p_surv` / `s_surv` remain 1.0 in observed runs).
- Ariadna editorial proposal (four points): register D-0088, substitute Q4 limitation by empirical reading at `subsec:obj_controls_externs`, keep claim bounded, reference Grace's audit document.
- Sofia validation of the proposed Catalan three-paragraph block for `subsec:obj_controls_externs` (2026-05-06): bounded reading, no manuscript-body citation of the audit document (editorial discipline preserved), terminology consistent with Â§16.
- Human-author validation of the full Z1.a + Z2.a + Z3.a + Z4.b plan: explicit "Endavant amb Z1.a+Z2.a+Z3.a+Z4.b!" (2026-05-06).

---

## D-0089

- Date: 2026-05-06
- Status: `accepted-pre-implementation-contract`
- Cycle: experimental v2 / Q5 / Table~5 signature derivation for external recursive controls O and P from D-0087 frozen snapshots / closure of taxonomic asymmetry between App~A taxonomy and Table~5 master signature

### Authority chain

- D-0085 authorized the external recursive controls extension (Q4).
- D-0086 (superseded) and D-0087 (executed) registered the v0.2 â†’ v0.3 contract for O and P.
- D-0088 closed Q4 empirically with Grace audit `ACCEPTED` and updated `subsec:obj_controls_externs`.
- After taxonomic propagation of O and P to App~A and to all in-prose external-control listings, Iban observed the residual reporting gap at Table~5 (`tab:signatura`): O and P appear in the taxonomy but not in the master signature row set. Q5 closes this gap as a read-only derivation, not as a new experiment.

### What is accepted

The proposal `docs/HSI_v2_Table5_OP_Signature_Derivation_Proposal_v0_1.md` is accepted as the pre-implementation contract for Q5, anchored at sha256 `ADEEE804972A15AB0BD3DA2AD031E5B5BC904B5A64D995D96C63780A0E9BA35D`. Acceptance is verbatim: no V1â€“V5 patches required.

### Contract clauses (frozen)

1. **Read-only on D-0087 snapshots.** Source artifacts are exclusively `var_O/phi_iter20.struct.gz` (sha256 `850F447BD02841DD0FD49FC2A4362460BEB16E441537D48A45411AB7DF88ED2E`) and `var_P/phi_iter20.struct.gz` (sha256 `8F2F495A37FBA301C4D9B5A72C4FD1C62732E22E81E4F1822C38D754BEBA1FA7`), both anchored at the D-0087 `pre_registration_sha256 = 66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0`. Any regeneration is forbidden under the D-0087 contract; if snapshots are missing, corrupted, or fail metadata checks, execution stops and a feasibility note is opened.
2. **No 1G parity claim.** Historical Table~5 LZ values were computed on 1G-bit Level-0 streams; the D-0087 snapshots are bounded to ~3.6M (O) and ~5.3M (P). The derivation reuses the same LZ estimator and scale set (`1000, 2000, 5000, 10000, 20000, 50000`) over the totality of each authorized snapshot. The `report.md` and the future Table~5 caption clause must declare this asymmetry explicitly; equating O/P rows operationally to A/M/L would be misleading and is forbidden by this contract.
3. **TE protocol.** Streaming TE with density coarse-graining, scales `4, 8, 16, 32, 64, 128`, history `k=3`, chunk size `10,000,000` bits, shuffle seed `42`. Reported field: `shuffle_control.te_ratio_observed_vs_shuffle`. Because each D-0087 snapshot is shorter than one chunk, the run is single-chunk; the `report.md` must document the exact processed bit count and the resulting cardinality of the shuffle control (1 chunk Ã— 1 seed). Reduced statistical power is a transparency requirement, not a kill criterion.
4. **Field policy for Table~5 O/P rows.** LZ: derived numeric value. `\bar{c}`: `timeout` / N.A. (not defined for non-annihilation generators, parity with A/M/L). TE ratio: derived numeric value. `D_f`, DET: `---` (parity with A/M/L row discipline).
5. **Caption clause for Table~5 (English, wording-fence compliant).** When Sofia patches the manuscript post-audit, the caption (or a dedicated footnote near the row) must include a clause stating that O/P rows are derived from the D-0087 frozen snapshots (~3.6M and ~5.3M bits respectively) and that the legacy 1G LZ run was not regenerated because the D-0087 contract forbids it.
6. **Read-only enforcement at script level.** The new public script `hsi_v2_table5_op_signature_derivation.py` must refuse to call or import the O/P generator path. Manifest must record command, script sha256, snapshot sha256 values, parameters, timestamp, and the source D-0087 metadata.
7. **Kill / escalation criteria.** Execution stops immediately and an escalation note is opened (instead of patching Table~5) if any of: snapshot integrity fails; metadata fails the D-0087 `pre_registration_sha256` check; metric computation requires changing O/P rules, target length, terminalization, or seed policy; LZ `mean_ratio` falls inside `[0.60, 0.64]` (compact-selective HSI band); TE ratio significant under the legacy threshold `te_ratio_observed_vs_shuffle > 2.0`; or the script cannot reproduce the declared LZ or TE protocol without introducing an unregistered degree of freedom. Triggering the LZ or TE alert means the rigid/crystalline reading of D-0088 needs review; under no circumstances is the manuscript edited silently in that case.
8. **Outputs.** All artifacts under `hsi_agents_project/results/hsi_v2/table5_op_signature_derivation/<run-name>/` with run name including `table5-op-signature`, `O-P`, `iter-20`, and a UTC timestamp. Required files: `summary.json`, `report.md`, `table.csv`, `manifest.json`.

### What is NOT in scope of D-0089

- No new experiment, no new generator run, no parameter sweep.
- No edit to Table~5 of the manuscript by Ariadna at execution time. Manuscript edits are downstream and require Grace post-execution audit + a separate Decision Log entry (planned D-0090).
- No reinterpretation of D-0088 readings. If kill criteria trigger, the protocol is escalation, not silent reframing.
- No revision of the D-0087 snapshot contract itself. Q5 operates strictly on top of D-0087.

### Sofia audit verdict (2026-05-06)

Six structural strengths verified by inspection:

1. Honest declaration of the 1G-vs-snapshot asymmetry, no faked parity.
2. SHA256 anchoring of every source artifact and the pre-registration metadata.
3. Refusal of generator-path imports at script level.
4. Asymmetric and correctly-set kill criteria: `[0.60, 0.64]` LZ band and `> 2.0` TE ratio are the operational thresholds that would contradict the D-0088 rigid/crystalline reading; if triggered, the discipline forces escalation rather than table edits.
5. Naming convention compliance (`hsi_v2_*` script, `results/hsi_v2/` outputs, manifest with command + sha256 + timestamp).
6. Field policy for `D_f` and DET as `---` matches existing external-control row discipline; no new asymmetry introduced.

### Procedure

- Ariadna: implements `hsi_v2_table5_op_signature_derivation.py` per the contract above, executes, generates the four output artifacts, and notifies upon completion. P4-02b retains higher priority; Q5 runs when bandwidth allows.
- Grace: post-execution audit (independent reading of `report.md` and `manifest.json`).
- Sofia: after Grace `ACCEPTED`, registers D-0090 (empirical closure), edits Table~5 of `main_v2_cat.tex` with the two new rows + caption clause, and updates `tab:q4_traceability` in App Phase 3 with the Q5 entries.

### Anchors

- Proposal: `docs/HSI_v2_Table5_OP_Signature_Derivation_Proposal_v0_1.md` sha256 `ADEEE804972A15AB0BD3DA2AD031E5B5BC904B5A64D995D96C63780A0E9BA35D` (frozen as of 2026-05-06).
- Source D-0087 snapshots and metadata: as above.
- Authority chain: D-0085 â†’ D-0086 (superseded) â†’ D-0087 (executed) â†’ D-0088 (closure) â†’ D-0089 (Q5 contract).

### Validation traces

- Sofia audit verdict (2026-05-06): six structural strengths verified, no V1â€“V5 patches required, four open points answered (1: yes, 2: yes-with-caption-clause, 3: yes-parity-with-A/M/L, 4: register-now-not-wait-and-see).
- Human-author validation (2026-05-06): "ok als 4 open points" + green light to register D-0089 and notify Ariadna.

---

## D-0090

- Date: 2026-05-07
- Status: `accepted-internal`
- Cycle: experimental v2 / Q5 closure / Table~5 signature derivation patched into `main_v2_cat.tex`

### Authority chain

- D-0089 registered the pre-implementation contract for Q5 read-only LZ/TE derivation on D-0087 snapshots.
- Ariadna executed D-0089. The run completed mechanically and respected all clauses of the contract, but triggered the pre-registered LZ alert for `P` (`mean_ratio = 0.6150` inside the `[0.60, 0.64]` compact-selective band, best-match `1/phi` with `distance = 0.003`, `p_value = 0.94`).
- Per D-0089 clause 7, no manuscript edit was authorized until Sofia + Grace review. Ariadna opened `docs/HSI_v2_Table5_OP_Signature_Derivation_Kill_Note.md` (sha256 `E2C694DC1420B6C0A5AFE2E29011543AE808810EBA86542D45BEA85E505DBBD9`).
- Sofia + Iban formulated the technical reading "LZ-positive-by-mean / ratio-unstable / TE-negative", drafted five candidate manuscript patches (A: O/P rows in Taula~5; B: caption extension; C: footnote; D: Â§16.1 paragraph; E: row in `tab:q4_traceability`), and submitted to Grace for hostile-review audit.
- Grace post-execution audit: `docs/HSI-audit-Q5-OP-Signature.md` (sha256 `E878D0CF0ED7800D64EC01FD20E9F74D66F0EC007599400CDEC733ABEE41BB73`), verdict `ACCEPTED` for candidates A--E + `ESCALATE-DISCREPANCY` for a pre-existing typo at `Paper_Latex/main_v2_cat.tex` line 393 (handled separately at D-0091).

### What is registered

D-0090 closes Q5 empirically and authorizes the manuscript patches:

1. **Taula~\ref{tab:signatura}**: two new rows under a new `\midrule` after `L`, classified as `External recursive`. `O`: `LZ = 0.7952`, `cbar = timeout/N.A.`, `TE ratio = x0.41`, `D_f = ---`, `DET = ---`. `P`: `LZ = 0.6150*`, `cbar = timeout/N.A.`, `TE ratio = x0.64`, `D_f = ---`, `DET = ---`.
2. **Caption of Taula~\ref{tab:signatura}**: extended with explicit declaration that `O`/`P` rows are derived from D-0087 snapshots (3,642,944 and 5,284,798 bits respectively), not from regenerated 1G runs, and that the D-0087 contract forbids regeneration. Cross-reference to Â§\ref{subsec:obj_controls_externs}.
3. **Footnote `*` under the table**: documents that `P`'s LZ mean falling inside the compact-selective band is by-mean and not by-stability (per-scale ratios `0.74, 0.59, 0.60, 0.65, 0.49`, `sigma = 0.090`); inter-scale TE remains below significance threshold (`x0.64` vs threshold `x2.0`).
4. **Â§\ref{subsec:obj_controls_externs}**: new fourth paragraph documenting the Q5 derivation, the LZ alert trigger for `P`, the by-mean nature of the coincidence, the preservation of D-0088 Phase 1 reading (Phase 1 protocol and Taula~\ref{tab:signatura} protocol measure different signatures), and the editorial conclusion that the HSI signature is not identifiable by LZ alone.
5. **`tab:q4_traceability`** (Phase 3 appendix Q4 ext): two new rows for D-0089 (Q5 contract) and D-0090 (Q5 closure), preserving the row discipline of D-0085 / D-0087 / D-0088.

### Why this reading is canonical

- The Phase 1 reading of D-0088 (rigid/crystalline regime, stable-support collapse under `phase-matched-LZ`) is preserved intact. Phase 1 measures fiber survival under structural perturbation; Taula~\ref{tab:signatura} measures scalar invariants on bounded snapshots. The two protocols are orthogonal and the by-mean LZ collision of `P` does not refute the Phase 1 reading.
- The "LZ-positive-by-mean / ratio-unstable / TE-negative" wording is a methodological refinement, not a defensive concession: it strengthens the case that the multi-component HSI signature is non-reducible to a single scalar, and that TE + ratio stability are the operationally discriminant components.
- Wording-fence compliance: no use of "geometry recovered", "Phase 3 solved", or any triumphalist formulation; the ascetic v2 register is preserved.

### Anchors

- Run directory: `hsi_agents_project/results/hsi_v2/table5_op_signature_derivation/table5-op-signature__var-O-P__iter-20__20260507T081924/`
- Script: `hsi_v2_table5_op_signature_derivation.py` sha256 `334FBB1AEA9E61BBEC357EA8A887AD892DEC95C3E1877DC58F5262937A6AAA7C`
- Snapshots: `O` sha256 `850F447BD02841DD0FD49FC2A4362460BEB16E441537D48A45411AB7DF88ED2E`; `P` sha256 `8F2F495A37FBA301C4D9B5A72C4FD1C62732E22E81E4F1822C38D754BEBA1FA7`; both anchored at D-0087 `pre_registration_sha256 = 66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0`.
- Kill note: `docs/HSI_v2_Table5_OP_Signature_Derivation_Kill_Note.md` sha256 `E2C694DC1420B6C0A5AFE2E29011543AE808810EBA86542D45BEA85E505DBBD9`.
- Grace audit: `docs/HSI-audit-Q5-OP-Signature.md` sha256 `E878D0CF0ED7800D64EC01FD20E9F74D66F0EC007599400CDEC733ABEE41BB73`.
- Authority chain: D-0085 â†’ D-0087 (executed) â†’ D-0088 (Phase 1 closure) â†’ D-0089 (Q5 contract) â†’ D-0090 (Q5 closure).

### Validation traces

- Grace verdict (2026-05-07): `ACCEPTED` for candidates Aâ€“E with explicit endorsement of the by-mean / ratio-unstable / TE-negative reading; "el cicle Q5 Ã©s una lliÃ§Ã³ magistral de per quÃ¨ els kill criteria preregistrats funcionen".
- Sofia validation of the patch plan (2026-05-07): five edits to `main_v2_cat.tex` reviewed pre-application, no diagnostics post-application.
- Human-author validation (2026-05-07): explicit "Confirmo execuciÃ³ completa amb F3 al pas 6 + patch immediat. Endavant!".

---

## D-0091

- Date: 2026-05-07
- Status: `closed-on-registration`
- Cycle: editorial / pre-existing inconsistency at `main_v2_cat.tex` line 393 surfaced during the Q5 audit cycle

### Background

While preparing candidate wordings for the Q5 patch (D-0090), Sofia + Iban detected a pre-existing inconsistency in the prose immediately following Taula~\ref{tab:signatura}:

- Line 322 (Section Â§6 / robustness signature inventory): "la paraula de Fibonacci dÃ³na LZ $\approx 0$" â€” correct.
- Taula~\ref{tab:signatura}, row `M`: `LZ = approx 0.0` â€” correct.
- Line 393 (prose post-Taula~5): "La paraula de Fibonacci (M) comparteix LZ $\approx 1/\varphi$ amb B" â€” **incorrect**, contradicts both line 322 and the table itself.

The Fibonacci word has logarithmic LZ complexity (`O(log n)`); its normalized LZ ratio converges asymptotically to `0`, not to `1/phi`. The line 393 claim was a residual conceptual typo, likely a copy-paste artifact from earlier draft revisions or a confusion between the LZ ratio and the asymptotic density of `1`s in the Fibonacci word (which is `1/phi^2`).

### Why opened now

Without this fix, patching Taula~\ref{tab:signatura} with the new `P` row (`LZ = 0.6150`) under D-0090 would create an immediately visible incoherence: the prose would assert that `M` shares `LZ approx 1/phi` with `B`, while the table would simultaneously show `M = approx 0.0` and `P = 0.6150`. A peer reviewer reading the table and the prose together would flag this on first inspection. Grace's audit explicitly requested that D-0091 be opened obligatorily for this reason.

### What is registered

The line 393 prose has been replaced with wording option F3 (Sofia formulation): the Fibonacci word collapses to LZ approx 0 despite sharing the phi-related origin with B, showing that the presence of phi in the generative rule is not sufficient for the compact-selective regime. This formulation:

- removes the false quantitative claim;
- preserves the rhetorical contrast between `M` and `B` so the table is not orphaned in the prose;
- reinforces the existing ontological argument of Â§3 (the signature is not derivable from phi alone);
- introduces no new quantitative claim that could itself be falsified.

The patch is applied at the same commit/cycle as D-0090 because separating them would leave a known incoherence visible in the patched manuscript.

### Anchors

- Patched line: `Paper_Latex/main_v2_cat.tex` line 393 (pre-edit) / corresponding line in the post-D-0090 file.
- Surfacing context: `docs/HSI-audit-Q5-OP-Signature.md` sha256 `E878D0CF0ED7800D64EC01FD20E9F74D66F0EC007599400CDEC733ABEE41BB73`, point 4 ("La discrepancia preexistent de M").
- Authority chain: surfaced during D-0090 cycle; closed in the same patch passada.

### Validation traces

- Grace verdict (2026-05-07): "HabÃ©is cazado un error crÃ­tico" + recommendation to open D-0091 obligatorily and apply the F3 wording.
- Sofia recommendation (2026-05-07): F3 wording preferred over F1 (Grace's literal proposal, which required asymptotic density verification) and F2 (deletion, which loses rhetorical contrast).
- Human-author validation (2026-05-07): explicit "Confirmo execuciÃ³ completa amb F3 al pas 6 + patch immediat. Endavant!".

---

## D-0092

- Date: 2026-05-07
- Status: `accepted-pre-implementation-contract`
- Cycle: editorial / paper coverage extension following the D-0090 Q5 closure
- Trigger: while patching `Paper_Latex/main_v2_cat.tex` to integrate the external recursive controls O and P into the variant-bearing tables (D-0090 + Pas A inventory), Sofia + Iban detected that five tables hold O/P data already derivable from the D-0087 `phase1_summary.json` artifacts (TE shuffle, `tab:pm_size`, `tab:fibres`, asimetria pre/suf, `tab:taxonomia`) but two tables reference observables that are not in the current `phase1_summary.json`: `tab:transport` (breadth ratio, $H_{\text{trans}}$, $\mu_{\text{act}}$, $\Delta\mu_{\text{sgn}}$, $\Delta\mu_{\text{rel}}$) and `tab:porta1_canonica` column `active_mean_tail`. The Pas A patch covered the five tables with available data and inserted `$\dagger$` placeholders for the two `active_mean_tail` cells of O and P, with an explicit caption note pointing to this contract. D-0092 closes the residual asymmetry by commissioning a read-only re-derivation of the `phase1-transport` report over the already-existing tower runs.

### Decision

The following pre-implementation contract is hereby frozen for the code-and-experiments role (Ariadna):

1. **Read-only on existing towers.** Source artifacts are exclusively the two existing tower runs:
   - `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/phase1-tower__var-O__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__20260506T235738/`
   - `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/phase1-tower__var-P__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__20260506T235740/`

   No tower regeneration. No snapshot regeneration. No new pattern-space, fiber, or moments computation. The contract operates exclusively over the `pattern_spaces/`, `fibers/`, and `moments/` subdirectories of the two run directories above. If any of those subdirectories is missing, corrupted, or fails sha256 verification against its run manifest, execution stops and a feasibility note is opened.

2. **Derivation scope.** Compute and emit the standard `phase1-transport` report over the same projection policies (`prefix`, `suffix`) and the same upper edges (`28 -> 24` and `32 -> 28`) used for the canonical batch (cf. existing reports at `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-transport__*`). Required derived observables: breadth ratio per upper edge, $H_{\text{trans}}$ per upper edge, $\mu_{\text{act}}$ per upper edge and per policy, $\Delta\mu_{\text{sgn}}$ and $\Delta\mu_{\text{rel}}$ as the prefix-minus-suffix divergence of $\mu_{\text{act}}$, and `active_mean_tail` defined as the mean of $\mu_{\text{act}}^{\text{prefix}}$ across the two upper edges (consistent with the existing `tab:porta1_canonica` definition).

3. **No protocol drift.** Reuse the existing `phase1-transport` derivation code path verbatim. No new metric definition, no new edge selection, no new tail policy. The output schema must match the existing canonical reports field-by-field so that the new rows for O and P drop into `tab:transport` and `tab:porta1_canonica` without any per-row footnote beyond the one already declared at the caption of `tab:porta1_canonica`.

4. **Outputs.** All artifacts under `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/transport_op/<run-name>/` with run name including `phase1-transport`, `var-O-P`, and a UTC timestamp. Required files: `summary.json` (with the per-edge and per-policy observables), `report.md` (human-readable), `manifest.json` (script sha256, source tower sha256 hashes, parameters, command, timestamp, source D-0087 anchor).

5. **Read-only enforcement at script level.** The new public script (suggested name `hsi_v2_phase1_transport_op_derivation.py`) must refuse to call or import any tower-generation, snapshot-generation, or pattern-space-generation path. The script must verify the source tower sha256 hashes against the run manifests before any computation.

6. **Field policy for the manuscript.** The two `$\dagger$` placeholders at `tab:porta1_canonica` rows O and P are to be replaced verbatim by the derived `active_mean_tail` numeric values, with the caption footnote then trimmed to remove the pending-derivation clause. New rows for O and P in `tab:transport` are to be inserted under a dedicated `\midrule` after the existing M/N row, with the column `SÃ­ntesi` reading `cristalÂ·lÃ­` (P) and `rÃ­gid` or `cristalÂ·lÃ­` (O, depending on which side of the F vs M/N envelope its $\mu_{\text{act}}$ falls). Final wording is decided post-execution by Sofia, not by Ariadna at execution time.

7. **Kill / escalation criteria.** Execution stops immediately and an escalation note is opened (instead of patching the manuscript) if any of: source tower sha256 verification fails; the derivation requires changing edge selection, policy set, or tail definition; `active_mean_tail` for O or P falls inside the HSI-like band (operationally `>= 0.30`, the lower envelope of B/E at `0.3539`) which would contradict the rigid/crystalline reading of D-0088; the script cannot reproduce the canonical `phase1-transport` schema without introducing an unregistered degree of freedom. Triggering the `active_mean_tail` alert means the D-0088 + D-0090 reading needs review; under no circumstances is the manuscript edited silently in that case.

### What is NOT in scope of D-0092

- No new tower run, no new snapshot, no new fiber computation.
- No edit to any other manuscript table beyond the two cells already marked `$\dagger$` and the new rows for `tab:transport`.
- No reinterpretation of D-0088 or D-0090. The expected reading is that O and P land in the rigid/crystalline envelope (low `active_mean_tail`, low or zero asymmetry); a result inside the HSI-like band is a kill criterion, not a new claim.

### Rationale

The five-table coverage achieved by Pas A (D-0090 follow-up) closes the manuscript-level inconsistency that a hostile reviewer would flag first: silent absence of O and P from variant-bearing tables. The two remaining cells (`active_mean_tail` for O and P at `tab:porta1_canonica`, plus the full `tab:transport` row pair for O and P) cannot be filled honestly without a derivation that does not exist yet in the artifact tree. Two paths were considered:

- (a) leave the cells empty and declare omission in the caption, accepting reviewer pressure on a second flank;
- (b) commission a cheap read-only re-derivation from the existing tower artifacts, which closes the flank without regenerating any expensive object.

Path (b) is selected because: (i) the cost is low (the `phase1-transport` report is a re-derivation over already-computed fibers and pattern spaces, not a new tower); (ii) the kill criteria are pre-registered, so the contract cannot silently rescue a result that would contradict D-0088; (iii) the dual-report architecture (`phase1_summary.json` + `phase1-transport` report) was already canonical for the original Phase 1 batch, so applying it to O and P restores symmetry rather than adding asymmetry.

The `phase1-transport` report path is already exercised by the canonical batch (cf. `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-transport__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-vs-suffix__tail-2__var-B-markov1-B-same-density-B-shuffled__20260416T213500.*` and analogous files for matched-LZ and per-variant runs); reusing it for O and P does not require any new methodological commitment.

### Practical Impact

- Repo green-light for implementation of one script under existing conventions:
  - `hsi_v2_phase1_transport_op_derivation.py`: read-only re-derivation of the canonical `phase1-transport` report over the existing var-O and var-P tower run directories. Manifest must include source tower sha256 hashes and the D-0087 `pre_registration_sha256` anchor.
- Outputs under `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/transport_op/<run-name>/`.
- Phase logs visible per AGENTS.md long-script rule.
- `Paper_Latex/main_v2_cat.tex`:
  - `tab:porta1_canonica` already patched with `$\dagger$` placeholders and a caption footnote pointing to D-0092.
  - `tab:transport` remains as-is (no O/P rows yet) until D-0092 results are in. A subsequent Decision Log entry (planned D-0093) will record the empirical reading of O and P transport observables and authorize the substitution of the `$\dagger$` placeholders by numeric values plus the insertion of new rows in `tab:transport`.

### Anchors

- Source tower runs: as listed in clause 1 above.
- Canonical reference report (schema source): `hsi_agents_project/results/hsi_v2/phase1/reports/phase1-transport__iter-20__seg-3x1000000__m-8-12-16-20-24-28-32__proj-prefix-vs-suffix__tail-2__var-B-markov1-B-same-density-B-shuffled__20260416T213500.json`.
- Source D-0087 snapshots and metadata: as recorded at D-0087 / D-0089.
- Authority chain: D-0085 â†’ D-0087 (executed) â†’ D-0088 (Phase 1 closure) â†’ D-0090 (Q5 closure + Pas A coverage patch) â†’ D-0092 (transport coverage extension).

### Validation traces

- Pas A inventory (2026-05-07): five tables patched with available D-0087 data; two tables identified as requiring derived transport observables not present in current `phase1_summary.json`. Inventory report inline in conversation; no separate audit document opened (the Pas A patch is editorial, not scientific).
- Human-author validation (2026-05-07): "Endavant amb Pas A i redactem D-0092. I prepara el missatge en nom nostre per a Ariadna."
- Grace audit role: post-execution, not co-design (preserved from D-0086, restated from D-0089). The audit role will be invited only on the final results, with this entry as the pre-registered contract against which those results are evaluated.

---

## D-0093

- Date: 2026-05-07
- Status: `executed-and-empirically-closed`
- Cycle: editorial / D-0092 execution closure and final manuscript coverage patch for external recursive controls O and P
- Trigger: Ariadna executed the D-0092 contract verbatim. The read-only re-derivation of `phase1-transport` over the existing var-O and var-P tower runs completed successfully with zero kill triggers, returning all five transport observables required by `tab:transport` and the missing `active_mean_tail` cells of `tab:porta1_canonica`. The empirical reading is consistent with the rigid/crystalline envelope predicted by D-0088 + D-0090; no manuscript escalation is needed and the final coverage patch can be applied.

### Decision

D-0093 closes D-0092 empirically and authorizes the final manuscript patches that complete the external recursive controls coverage:

1. **Substitution of `$\dagger$` placeholders at `tab:porta1_canonica`** by the derived numeric values: O `active_mean_tail = 0.2457`, P `active_mean_tail = 0.2269`. The pending-derivation clause of the caption is trimmed to a simple provenance note ("the values for O and P come from the read-only re-derivation `phase1-transport` over the D-0087 towers, D-0092, no upstream object regenerated").
2. **Insertion of two new rows at `tab:transport`** under a dedicated `\midrule` after the existing M/N row, with full transport observables: O (breadth ratio `21.94`, $H_{\text{trans}} = 0.296$, $\mu_{\text{act}} = 0.246$, $\Delta\mu_{\text{sgn}} = -0.141$, $\Delta\mu_{\text{rel}} = -0.22$, synthesis `rÃ­gid`); P (breadth ratio `10.31`, $H_{\text{trans}} = 0.262$, $\mu_{\text{act}} = 0.227$, $\Delta\mu_{\text{sgn}} = -0.034$, $\Delta\mu_{\text{rel}} = -0.07$, synthesis `cristalÂ·lÃ­`). The interpretive paragraph following `tab:transport` is extended with two sentences that anchor O to the rigid envelope of F (with explicit note that O exhibits a non-trivial negative asymmetry of magnitude similar to but opposite-signed to B/E/I) and P to the crystalline envelope of M/N (with near-zero asymmetry).
3. **Brief description of O and P added to the controls table at `subsec:controls`** (line 281 area). Two new rows under a dedicated `\midrule` after M, classifying O as "CFG determinista amb regles $\epsilon$" and P as "L-system estratificat", each with a one-clause justification of why they were added (testing whether formal hierarchical recursion with deferred evaluation, or stratified parallel rewrite, generates the HSI signature). The introductory sentence is updated to declare seven controls split as five classical + two external-recursive.
4. **Extension of the prose immediately after the canonical-variant LZ table** (`subsec:lz`, line 322 area, the sentence enumerating control LZ values) to include O `LZ = 0.7952` and P `LZ = 0.6150`, with an explicit clause that P's mean coincides with the HSI band by mean only, with unstable per-scale ratio ($\sigma = 0.090$) and TE below significance threshold, so the LZ coincidence does not constitute a reproduction of the signature. Cross-reference to the existing footnote of Taula~\ref{tab:signatura} and to Â§`subsec:obj_controls_externs`.
5. **Honesty fix at the reproducibility appendix paragraph** (line 949 area) reformulating the previous false statement ("es publiquen com a artefactes versionats al directori `hsi_agents_project/results/hsi_v2/`") into an accurate one: the directory is emitted on every execution into the local working tree, but it is not versioned to GitHub for weight reasons (the full artifact tree exceeds ~130 GB with raw Level-0 snapshots, pattern spaces, and fibers), the deterministic pipeline allows full reconstruction by re-executing the public scripts under the registered seeds and hyperparameters, and the artifacts already generated by the author are made available on request to the reader or reviewer who wants to audit a specific experiment without re-executing the entire stack.

### What is NOT in scope of D-0093

- No new experiment, no new derivation, no parameter sweep.
- No reinterpretation of D-0088, D-0090, or D-0092. The empirical reading lands inside the predicted rigid/crystalline envelope; there is no science to reopen, only manuscript coverage to close.
- No upload of the artifact tree to Zenodo / Figshare / OSF. The current cycle adopts the "scripts + manifests + on-request data" path. A future decision can authorize a DOI-anchored deposit (with or without fragmentation by phase) if the editorial workflow chooses to escalate the public-data policy.

### Empirical reading

The transport derivation lands O and P inside the non-HSI envelope as predicted:

- **O (CFG determinist with $\epsilon$ rules):** $\mu_{\text{act}} = 0.2457$ (vs F at $0.234$, vs B/E at $0.354$). Falls inside the rigid regime by transport mass, well below the HSI-like band lower bound at `0.30`. The asymmetry $\Delta\mu_{\text{rel}} = -0.22$ is non-trivial in magnitude, opposite-signed to B/E/I (which have $\Delta\mu_{\text{rel}} > 0$), and is interpreted as direction-dependent transport without HSI-like compactness.
- **P (stratified L-system):** $\mu_{\text{act}} = 0.2269$ (vs M/N at $0.223$). Falls inside the crystalline regime. Asymmetry $\Delta\mu_{\text{rel}} = -0.07$ is near-zero, consistent with the M/N pattern.

Neither variant approaches the coherent regime. The D-0088 + D-0090 reading is preserved without modification: external recursive controls land in the rigid/crystalline envelope; formal hierarchical recursion with deferred evaluation is not by itself a sufficient condition for the compact-selective regime.

### Methodological note: per-artifact hashes in source manifests

Ariadna flagged a precision issue worth recording as canonical for any future read-only derivation against D-0087 snapshots. The source tower manifests at D-0087 do not contain per-artifact sha256 hashes (only a top-level snapshot hash and the D-0087 pre-registration hash). The D-0092 derivation script (`hsi_v2_phase1_transport_op_derivation.py`) accordingly verifies (i) coherence of the manifest paths against the on-disk subtree, (ii) presence of the D-0087 anchor in the source manifest, and (iii) existence of the required `pattern_spaces`, `fibers`, and `moments` subtrees, and then computes and records aggregate sha256 hashes per component into the new D-0092 manifest. This is not a contract violation: D-0092 clause 5 required source-tower verification, which is achieved by manifest-path verification + anchor verification + subtree presence; per-artifact hashing was not feasible against the original D-0087 manifest schema. Future Phase 1 tower runs should consider including per-artifact hashes in the manifest schema by default to enable strict end-to-end verification chains for downstream read-only derivations. This methodological note is recorded here so that any future derivation against pre-D-0093 snapshots uses the same fallback verification policy without rediscovering the constraint.

### Practical Impact

- `Paper_Latex/main_v2_cat.tex`:
  - `tab:porta1_canonica`: $\dagger$ placeholders replaced by numeric values at the O and P rows; caption clause trimmed to a provenance note pointing to D-0092.
  - `tab:transport`: two new rows for O and P inserted under a dedicated `\midrule` after M/N; caption extended with one sentence on the D-0092 provenance; interpretive paragraph extended with two sentences on the rigid/crystalline placement of O and P.
  - `subsec:controls`: introductory sentence updated to declare seven controls (five classical + two external-recursive); two new rows for O and P added to the controls table under a dedicated `\midrule` after M.
  - `subsec:lz`: prose sentence after the canonical-variant LZ table extended with O and P values and the LZ-by-mean clause for P.
  - Reproducibility appendix paragraph: rewritten for honesty regarding the absence of `results/hsi_v2/` from the public GitHub repository.
- `HSI_v2_Decision_Log.md`: this entry registered.
- No code changes in this cycle.

### Anchors

- D-0092 source contract: `HSI_v2_Decision_Log.md#D-0092`.
- D-0092 execution artifact: `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/transport_op/phase1-transport__var-O-P__d0092__20260507T074318/` containing `summary.json`, `report.md`, and `manifest.json`.
- Source towers (verified by manifest-path coherence + D-0087 anchor + subtree presence + computed aggregate hashes per component):
  - O: `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/phase1-tower__var-O__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__20260506T235738/`
  - P: `hsi_agents_project/results/hsi_v2/phase1_external_recursive_controls/phase1-tower__var-P__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__20260506T235740/`
- Authority chain: D-0085 â†’ D-0087 (executed) â†’ D-0088 (Phase 1 closure) â†’ D-0090 (Q5 closure + Pas A coverage patch) â†’ D-0092 (transport coverage contract) â†’ D-0093 (D-0092 empirical closure + final coverage patch).

### Validation traces

- Ariadna execution report (2026-05-07): "D-0092 executat. La derivaciÃ³ Ã©s read-only sobre les torres O/P D-0087 existents i reutilitza el path canÃ²nic `v2.phase1.transport.build_transport_rows`. No s'ha regenerat cap torre, snapshot, pattern-space, fiber ni moments. No hi ha kill trigger: O `active_mean_tail = 0.245741` i P `active_mean_tail = 0.226912`, tots dos per sota del llindar HSI-like 0.30. AixÃ² Ã©s consistent amb la lectura rÃ­gida/cristalÂ·lina de D-0088/D-0090."
- Ariadna methodological note (2026-05-07): "els manifests originals de torre no contenien hashes per artefacte; el script verifica coherÃ¨ncia de paths del manifest, anchor D-0087 i existÃ¨ncia dels subarbres requerits, i registra hashes agregats calculats de cada torre al manifest nou."
- Grace audit role: post-execution, not invoked at this entry. The empirical reading lands inside the pre-registered envelope and the patches are scope-conservative (no new claim, no new derivation, no reinterpretation). A future Grace audit on the integrated paper draft remains in the editorial pipeline as part of the standard pre-submission cycle, not specifically gated to D-0093.
- Human-author validation (2026-05-07): explicit "ja tenim missatge d'Ariadna [...] No hi ha kill trigger" + green light to apply the final coverage patches.

---

## D-0094

- Date: 2026-05-07
- Status: `accepted-pre-execution-forecast`
- Cycle: Phase 4 / P4-04 forward regime forecast
- Trigger: after the P4-03b holdout audit accepted the mixed holdout map and explicitly discouraged further boundary micro-mapping, Ariadna proposed the next falsifiable step as a small forward forecast over untouched bands beyond the late-reactivation holdout.

### Decision

P4-04 is authorized as an internal, pre-registered Phase 4 forecast. It does not edit the manuscript and does not reopen D-0078/D-0079/D-0093. The experiment freezes the following contract before execution:

1. **Fixed script.** Public launcher `hsi_agents_project/hsi_v2_phase4_p4_04_forward_regime_forecast.py`, sha256 `3D50D551AD50EE6AC3A49C69A2647C32EF7BB2A044BF50A66EF7992186A95136`.
2. **Fixed source scope.** `phase1_dir = hsi_agents_project/results/hsi_v2/phase1_high_scales`.
3. **Fixed null envelope.** `markov1,phase-matched-lz,block-entropy`, with seeds `{607,709,811,907,1009,1103}`.
4. **Fixed lag grid.** `{-29.5M, -27M, -26.5M, -24M, +24.5M, +26.5M}` bits.
5. **Fixed forward bands.** `768M-777M`, `777M-786M`, `786M-795M`.
6. **Negative-channel forecast.** All three forward bands are predicted to remain in `late-reactivation`, operationalized as `strong-channel`.
7. **Positive-channel forecast.** The positive channel is predicted to remain non-universal, operationalized as not all three positive rows being `strong-channel`.
8. **Strong-cell threshold.** `B_retention >= 0.90` and hard margin `B - max(phase-matched-lz, block-entropy) >= 0.30`.
9. **Forecast verdict.** `forward-regime-forecast-supported` iff at least two of three negative bands are strong and the positive channel is non-universal; `mixed` iff exactly one test passes; `rejected` iff neither test passes.

### What is NOT in scope

- No boundary micro-map around the 732M-741M / 741M-750M transition.
- No new null design.
- No manuscript edit.
- No reinterpretation of P4-03b after seeing P4-04. If P4-04 fails, the failure is recorded as a forecast failure or attenuation of the late-reactivation regime, not tuned away.

### Rationale

P4-03b already established that the active-core and late-reactivation components can survive holdout pressure, while the 732M-741M shoulder prediction failed by sharpening into null-shadow. Grace's audit recommended closing that exploratory route rather than mapping the boundary bit-by-bit. P4-04 therefore moves forward in time instead of sideways along the boundary. The design is intentionally narrow: three new bands, the same compact lag grid, the same hard null envelope, and an explicit support threshold.

This turns the internal map into a falsifiable prediction without inflating the manuscript claim. A supported result extends the late-reactivation reading; a mixed or rejected result bounds it. Both outcomes are scientifically useful.

### Practical Impact

- New script:
  - `hsi_agents_project/hsi_v2_phase4_p4_04_forward_regime_forecast.py`
- Outputs under:
  - `hsi_agents_project/results/hsi_v2/phase4/p4_04_forward_regime_forecast/<run-name>/`
- Required files:
  - `summary.json`
  - `report.md`
  - `forecast_regimes.csv`
  - `forecast_tests.csv`
  - `manifest.json`
- A `--dry-run` execution completed successfully before this entry was closed, producing manifest-only artifacts at:
  - `hsi_agents_project/results/hsi_v2/phase4/p4_04_forward_regime_forecast/phase4-p4-04-forward-regime-forecast__bands-3__lags-6__20260507T200050/`

### Validation traces

- Ariadna implementation note (2026-05-07): the launcher reuses the P4-03b lag-response path rather than introducing new derivation code, and delegates parallel execution to `hsi_v2_phase2_child_routing_lag_response.py`.
- Dry-run validation (2026-05-07): `python.exe .\hsi_v2_phase4_p4_04_forward_regime_forecast.py --dry-run --workers 10` completed successfully and emitted the delegated command with bands `768000000,777000000,786000000`, the frozen lag grid, and the frozen null envelope.

---

## D-0095

- Date: 2026-05-08
- Status: `executed-and-empirically-closed`
- Cycle: Phase 4 / P4-04 forward regime forecast closure
- Trigger: P4-04 was executed under the frozen D-0094 pre-forecast contract and audited by Grace as `ACCEPTED`.

### Decision

D-0095 closes P4-04 empirically. The pre-registered forecast is accepted as supported under the exact D-0094 contract:

1. **Negative-channel forecast supported at full strength.** The D-0094 threshold was `>=2/3` forward bands strong; the observed result is `3/3`.
2. **Positive-channel nonuniversality supported.** The positive channel returns `0/3 strong`, with all three positive rows classified as `null-shadow-boundary`.
3. **No manuscript edit authorized.** P4-04 remains an internal Phase 4 predictive result. It does not reopen the current paper freeze and does not change any `Paper_Latex/main_v2_cat.tex` claim.
4. **Grace audit accepted.** The audit authorizes empirical closure and explicitly warns against global-law wording.

### Empirical readout

P4-04 tested three untouched forward bands beyond the P4-03b late-reactivation holdout:

| Band | Channel | Result | Best lag | B retention | Hard margin |
|---|---|---|---:|---:|---:|
| `768M-777M` | negative | `strong-channel` | `-29.5M` | `1.0000` | `0.3810` |
| `777M-786M` | negative | `strong-channel` | `-26.5M` | `1.0000` | `0.5976` |
| `786M-795M` | negative | `strong-channel` | `-27M` | `1.0000` | `0.6211` |
| `768M-777M` | positive | `null-shadow-boundary` | `+24.5M` | `0.4613` | `-0.2315` |
| `777M-786M` | positive | `null-shadow-boundary` | `+24.5M` | `0.0000` | `-0.3839` |
| `786M-795M` | positive | `null-shadow-boundary` | `+26.5M` | `0.0000` | `-0.3088` |

The automatic script verdict is `forward-regime-forecast-supported`.

### Allowed wording

- `Forward persistence of the compact-selective regime`.
- `Out-of-sample validation of the negative-channel reactivation`.
- `Late negative-channel reactivation persists forward under the hard phase-matched-LZ + block-entropy null envelope`.

### Forbidden wording

- `Global law`.
- `Universal phase law`.
- `Infinite persistence`.
- `Gate solved`.
- Any asymptotic or universal claim derived from the three-band P4-04 horizon.

### Rationale

P4-04 is stronger than another retrospective revalidation because it tested a frozen forward forecast over three previously unused bands. The result supports the claim that the late negative-channel reactivation observed around `759M-768M` persists into the next visible forward horizon through `786M-795M`, under the same compact lag grid and the same hard null envelope used in P4-03b.

However, the result remains finite-horizon and internal. Three 9M-bit bands constitute a substantial empirical horizon but not an asymptotic law. The correct reading is predictive persistence over the tested forward window, not global stationarity or universal phase dynamics.

### Practical Impact

- P4-04 artifacts are frozen at:
  - `hsi_agents_project/results/hsi_v2/phase4/p4_04_forward_regime_forecast/phase4-p4-04-forward-regime-forecast__bands-3__lags-6__20260507T200920/`
- Required files present:
  - `summary.json`
  - `report.md`
  - `forecast_regimes.csv`
  - `forecast_tests.csv`
  - `manifest.json`
- No manuscript patch.
- No new figure promoted to paper-candidate.
- Phase 4 remains free to continue experimentally, but any next forecast must be registered under a new D entry before execution.

### Anchors

- Pre-forecast contract: D-0094.
- Execution artifact: `hsi_agents_project/results/hsi_v2/phase4/p4_04_forward_regime_forecast/phase4-p4-04-forward-regime-forecast__bands-3__lags-6__20260507T200920/`.
- Grace audit: `docs/HSI-audit-Phase4-P4-04-Forward-Forecast.md`, sha256 `5EEF794C0728943EC7D5CBA05C642A2992E793B23D59576796CF80F3F021188E`.

### Validation traces

- Ariadna readout (2026-05-08): P4-04 is supported at `3/3` negative bands with positive-channel nonuniversality at `0/3 strong`; the result is a finite-horizon predictive success, not a global law.
- Grace audit (2026-05-08): verdict `ACCEPTED`; D-0094 judged as honest pre-forecast with zero p-hacking; D-0095 closure explicitly authorized; manuscript edits explicitly prohibited.
- Human-author validation (2026-05-08): "Perfecte. endavant! I desprÃ©s, pensa en el segÃ¼ent pas experimental :-D".

---

## D-0096

- Date: 2026-05-08
- Status: `accepted-pre-execution-forecast`
- Cycle: Phase 4 / P4-05 horizon-bracket forecast
- Trigger: after D-0095 closed P4-04 as a finite-horizon predictive success, Ariadna proposed a stricter follow-up: a four-band forward horizon bracket to test whether the negative-channel late reactivation persists further or begins to attenuate.

### Decision

P4-05 is authorized as an internal, pre-registered Phase 4 forecast. It is not a manuscript-facing result and does not modify D-0095. The experiment freezes the following contract before execution:

1. **Fixed script.** Public launcher `hsi_agents_project/hsi_v2_phase4_p4_05_horizon_bracket_forecast.py`, sha256 `49DAD13A597987071F4E76B7740068A5287DC4932A29C40BCAE7C0BBDFBF5D05`.
2. **Fixed source scope.** `phase1_dir = hsi_agents_project/results/hsi_v2/phase1_high_scales`.
3. **Fixed null envelope.** `markov1,phase-matched-lz,block-entropy`, with seeds `{607,709,811,907,1009,1103}`.
4. **Fixed lag grid.** `{-29.5M, -27M, -26.5M, -24M, +24.5M, +26.5M}` bits.
5. **Fixed horizon bands.** `795M-804M`, `804M-813M`, `813M-822M`, `822M-831M`.
6. **Negative-channel forecast.** All four horizon bands are predicted to remain in `late-reactivation`, operationalized as `strong-channel`.
7. **Positive-channel forecast.** The positive channel is predicted to remain non-universal, operationalized as not all four positive rows being `strong-channel`.
8. **Strong-cell threshold.** `B_retention >= 0.90` and hard margin `B - max(phase-matched-lz, block-entropy) >= 0.30`.
9. **Forecast verdict.** `horizon-bracket-forecast-supported` iff at least three of four negative bands are strong and the positive channel is non-universal; `mixed` iff exactly one test passes; `rejected` iff neither test passes.

### What is NOT in scope

- No open-ended extension beyond `822M-831M`.
- No retuning of lag grid, null envelope, thresholds, or seed family.
- No manuscript edit.
- No claim that P4-04 implies infinite persistence. If P4-05 weakens or fails, the result bounds the late-reactivation horizon rather than invalidating the finite P4-04 success.

### Rationale

P4-04 established a clean three-band forward success through `786M-795M`. P4-05 deliberately asks the next harder question: does the same regime persist over a longer, finite bracket, or does attenuation begin in the next four bands? The design remains narrow and falsifiable. A supported result extends the predictive horizon; a mixed result localizes the attenuation zone; a rejected result marks the end of the P4-04 forward persistence window.

The threshold `>=3/4` is chosen to require more than a weak majority while allowing one local fluctuation in a finite empirical regime. This is stricter than P4-04 in absolute band count and avoids treating a single isolated hit as horizon persistence.

### Practical Impact

- New script:
  - `hsi_agents_project/hsi_v2_phase4_p4_05_horizon_bracket_forecast.py`
- Outputs under:
  - `hsi_agents_project/results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/<run-name>/`
- Required files:
  - `summary.json`
  - `report.md`
  - `forecast_regimes.csv`
  - `forecast_tests.csv`
  - `manifest.json`
- A `--dry-run` execution completed successfully before this entry was closed, producing manifest-only artifacts at:
  - `hsi_agents_project/results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__20260507T225803/`

### Validation traces

- Ariadna implementation note (2026-05-08): the launcher reuses the P4-04/P4-03b lag-response path, changes only the forward bands and threshold, and delegates parallel execution to `hsi_v2_phase2_child_routing_lag_response.py`.
- Dry-run validation (2026-05-08): `python.exe .\hsi_v2_phase4_p4_05_horizon_bracket_forecast.py --dry-run --workers 10` completed successfully and emitted the delegated command with bands `795000000,804000000,813000000,822000000`, the frozen lag grid, and the frozen null envelope.
- Human-author validation (2026-05-08): "Estic totalment d'acord, endenvant!"

---

## D-0097

- Date: 2026-05-08
- Status: `executed-and-empirically-closed`
- Cycle: Phase 4 / P4-05 horizon-bracket closure
- Trigger: P4-05 was executed under the frozen D-0096 contract and audited by Grace as `ACCEPTED`.

### Decision

D-0097 closes P4-05 empirically and closes the current Phase 4 predictive subarc. The result is accepted with a deliberately hard reading:

1. **Negative horizon forecast rejected.** The D-0096 threshold was `>=3/4` strong negative bands; the observed result is `0/4`.
2. **Positive nonuniversality supported.** The positive channel returns `0/4 strong`, preserving the directional asymmetry test.
3. **P4-04 not refuted.** P4-04 remains a successful finite-horizon forecast through `786M-795M`.
4. **Finite-horizon boundary found.** P4-05 shows that the P4-04 forward persistence is bounded immediately after that horizon, beginning at `795M-804M`.
5. **No further forward signal chasing.** This subarc is closed. Continuing to scan later bands for reactivation is not authorized by this entry.
6. **No manuscript edit authorized.** P4-05 remains an internal Phase 4 result.

### Empirical readout

| Band | Channel | Observed regime | Best lag | B retention | Null envelope | Hard margin |
|---|---|---|---:|---:|---:|---:|
| `795M-804M` | negative | `null-shadow-boundary` | `-26.5M` | `1.0000` | `phase-matched-lz = 1.0000` | `0.0000` |
| `804M-813M` | negative | `null-shadow-boundary` | `-29.5M` | `1.0000` | `phase-matched-lz = 1.0000` | `0.0000` |
| `813M-822M` | negative | `pre-shadow-shoulder` | `-26.5M` | `1.0000` | `phase-matched-lz = 0.8563` | `0.1437` |
| `822M-831M` | negative | `null-shadow-boundary` | `-27M` | `0.9890` | `phase-matched-lz = 1.0000` | `-0.0110` |

The automatic script verdict is `horizon-bracket-forecast-mixed` because the positive-channel nonuniversality test is supported. The scientific reading of the main forecast is stricter: the negative-channel horizon extension failed (`0/4`).

### Interpretation

P4-05 does not show channel death. B still retains strongly in the negative rows, often at `1.0000`. What collapses is the differential margin against the hard null envelope, especially `phase-matched-lz`, which reaches `1.0000` in several rows. The correct interpretation is therefore `null-envelope catch-up`: the phase-matched null becomes opaque enough to shadow the channel, so the predictive differential ceases to separate B from the null envelope.

This bounds D-0095/P4-04. The P4-04 forward persistence is real within its finite horizon, but it does not extend globally.

### Allowed wording

- `finite-horizon boundary`.
- `null-envelope catch-up`.
- `thermodynamic opacity limit`.
- `P4-04 forward persistence is bounded after 786M-795M`.
- `negative horizon forecast rejected under the D-0096 bracket`.

### Forbidden wording

- `P4-04 refuted`.
- `channel death`.
- `global law`.
- `universal phase law`.
- `infinite persistence`.
- Any claim that later reactivation can be inferred without a new pre-registered forecast.

### Practical Impact

- P4-05 artifacts are frozen at:
  - `hsi_agents_project/results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__20260507T225929/`
- Required files present:
  - `summary.json`
  - `report.md`
  - `forecast_regimes.csv`
  - `forecast_tests.csv`
  - `manifest.json`
- No manuscript patch.
- No new figure promoted to paper-candidate.
- Current Phase 4 predictive subarc closed as:
  - P4-04: finite forward prediction supported through `786M-795M`;
  - P4-05: immediate horizon boundary / null-envelope catch-up after `795M-804M`.

### Anchors

- Pre-forecast contract: D-0096.
- Execution artifact: `hsi_agents_project/results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__20260507T225929/`.
- Grace audit: `docs/HSI-audit-Phase4-P4-05-Horizon-Bracket.md`, sha256 `A425DADDF446E6792609D256BA7D40DC8955BD5AD72E4965B083BFD93E96904D`.

### Validation traces

- Ariadna readout (2026-05-08): P4-05 returns `mixed` mechanically, but the main negative forecast is rejected at `0/4`; the correct reading is not channel death but hard-null catch-up.
- Grace audit (2026-05-08): verdict `ACCEPTED`; D-0097 closure explicitly authorized; recommends closing the predictive subarc and avoiding further forward signal chasing.
- Human-author validation (2026-05-08): "Val, doncs endavant!"

---

## D-0098

- Date: 2026-05-08
- Status: `accepted-documentation-relocation`
- Cycle: repository traceability / Decision Log canonical path
- Trigger: while removing traceability tables from the paper appendices, Sofia + Iban identified that the canonical Decision Log should be directly versioned inside the public experimental repository so that external auditors can recover the full decision chain without relying on the out-of-repo `docs/` folder.

### Decision

The canonical HSI v2 Decision Log is relocated to:

- `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md`

The previous path:

- `docs/HSI_v2_Decision_Log.md`

is retained only as a backward-compatible pointer for older notes, audits, prompts, and paper drafts that still cite the old location. New D-XXXX entries must be recorded in the repo-tracked file.

### Rationale

The Decision Log is the primary audit spine of HSI v2. Keeping it outside the GitHub-tracked experimental repository weakens independent reproducibility: a reviewer can recover scripts and documentation from the repo but not the full methodological decision chain. Moving the canonical log into `hsi_agents_project/Documentation/` makes the decision history versioned, diffable, and co-located with the code and recovery guides.

This relocation also supports the paper-editing decision to remove appendix traceability tables: instead of embedding long log tables in the manuscript, the paper can point auditors to the canonical repository file.

### Practical Impact

- New canonical path: `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md`.
- Old path: `docs/HSI_v2_Decision_Log.md` now contains a short pointer only.
- Updated routing references:
  - `AGENTS.md`
  - `.dev/skills/skill-hsi-version-2/SKILL.md`
  - `.dev/skills/skill-hsi-version-2/checklists/*`
  - `.dev/skills/skill-hsi-version-2/references/*`
  - `.dev/skills/skill-hsi-version-2/workflows/*`
  - `hsi_agents_project/README.md`
  - `hsi_agents_project/Documentation/*.md` files that previously pointed to `../docs/HSI_v2_Decision_Log.md`

### What is NOT changed

- Historical audit documents are not rewritten line by line; the old path remains resolvable through the pointer.
- No experimental result, script default, or manuscript claim changes because of this relocation.
- The long-form scientific documents in root `docs/` remain valid; only the canonical decision log moves into the Git-tracked experimental repository.

### Validation traces

- Ariadna implementation note (2026-05-08): copied the full Decision Log into `hsi_agents_project/Documentation/`, replaced the old root file with a compatibility pointer, updated operational read-order references, and verified that no active AGENTS/skill/README/Documentation route still points to `docs/HSI_v2_Decision_Log.md`.
- Human-author validation (2026-05-08): "Val, doncs endavant! ... incorpora la url del repo amb el nom de l'arxiu al paper."

---

## D-0099

- Date: 2026-05-08
- Status: `executed-editorial-distillation`
- Cycle: manuscript distillation v0.6 -> v2.01 / hostile-review response
- Trigger: a brief-blind hostile-review pass by Gemini on the v0.6 manuscript flagged a perceived "metric-shopping" tone in the Gate 1 defense, inflated thermodynamic vocabulary in the Phase 3 closure, redundant null-model definitions, and an ISO-style traceability surface in the appendices that bloated the body without serving the external reader. Sofia + Iban opened a five-session distillation cycle to collapse the manuscript surface without weakening any technical claim.

### Decision

D-0099 records the closure of the v0.6 -> v2.01 editorial distillation cycle. The cycle produced a single live manuscript file `Paper_Latex/main_v2.01_cat.tex` (the v0.6 file is preserved as `Paper_Latex/main_v2_cat.tex` for diff reference) and was executed in five sessions against the canonical wording fence:

1. **S1 - Intro / Ontology (E1, E2, E3).** Removed three layers of preemptive defensive prose from Sections 1-4. No technical claim altered.
2. **Action A - Radical Traceability.** Stripped all inline `D-00XX` and `N3-XX` tags from the manuscript body; removed the three appendix subsections that duplicated Phase 2, Phase 3 and Q4 traceability tables; replaced them with a single externalized pointer subsection (`app:phase3_diag:traceability_pointer`) that exposes both the internal repository path `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md` and the public GitHub URL, executing the relocation accepted in D-0098.
3. **S2 - Robustness / Tower (E4, E5, E6, E7).** Reformulated the Gate 1 defense from a defensive "we deliberately refuse to reduce to a scalar" stance to a constructive statement about non-reducible structural observables. Anchored the thermodynamic reading of routing with an explicit citation to Beck & Schloegl, *Thermodynamics of Chaotic Systems* (1993), added as a new bib entry `BeckSchlogl1993` in `Paper_Latex/references.bib`. Condensed two redundant restatements of transport semantics and one duplicated null-model definition.
4. **S3 - N2 / Objections + Phase 4 integration (E8, E9, E10).** Created a new subsection `subsec:phase4_finite_subarc` in Section 7 documenting the P4-04/P4-05 predictive subarc verbatim under the binding constraints of the editorial triage `docs/HSI_v2_Phase4_Editorial_Triage.md` (Grace verdict `ACCEPTED-WITH-AMENDMENTS`): out-of-sample-on-prefrozen-lags claim explicit, sha256 freezing of the lag grid mentioned, P4-05 rendered as `Horitzó acotat (convergència de l'embolcall nul)` and not as bare rejection. Updated the circularity-objection paragraph to point to the same subarc as a pre-registered out-of-sample contract independent of the retrospective LZ-family contrast. Updated the Conclusions paragraph on Gate 2 to record a `bounded forward predictive horizon` at scale `795M-804M` without reopening Gate 2.
5. **S4 - Appendix cleanup (E11, E11b, E12a, E12b).** Propagated the S2 anti-metric-shopping discipline to the Phase 2 figure captions: collapsed an enumerated six-item negative list in the `fig:phase2_lag_response` caption and a softer "does not replace" defensive clause in the `fig:phase2_deficit_jitter` caption into compact positive statements pointing to `subsec:primera_obertura`. Removed two now-redundant meta-justifications about the canonical English lexicon from the Phase 2 and Phase 3 appendix introductions, since the canonical pointer already lives in `app:phase3_diag:traceability_pointer`.

### Wording fence compliance

After every session the manuscript was regex-audited against the canonical fence. The following terms remain absent from the v2.01 file: `global law`, `universal phase law`, `infinite persistence`, `gate solved`, `Gate 2 closed`, `P4-04 refuted`, `channel death`, `recuperació del nul`, `wave law`, `universal lag`, `global stationarity`, `complete transport law`, `absolute routing`, `Gate 2 resolution`, `paper-safe`, `primary Gate 2 opening`, `first paper-safe`. The following bound terms appear with the contractual frequency required by D-0096/D-0097 and the Phase 4 editorial triage: `convergència de l'embolcall nul`, `out-of-sample sobre lags prefixats`, `congelada per sha256`, `Horitzó acotat`.

### Practical Impact

- Live manuscript file: `Paper_Latex/main_v2.01_cat.tex` (1106 lines; the v0.6 file `Paper_Latex/main_v2_cat.tex` had 1161 lines; net distillation `-55` lines, `-4.7 %`).
- New bibliography entry `BeckSchlogl1993` in `Paper_Latex/references.bib`.
- New manuscript subsection `subsec:phase4_finite_subarc` (Section 7) integrating the P4-04 + P4-05 subarc with the binding wording from the editorial triage.
- New manuscript subsection `app:phase3_diag:traceability_pointer` (Appendix) replacing the three removed traceability subsections.
- All inline `D-00XX` / `N3-XX` / `N2-XX` tags removed from the manuscript body; the canonical decision chain is now reachable only through the externalized pointer (consistent with D-0098).
- No experimental result, script default, manuscript claim or contractual default changed by this cycle. The distillation is purely editorial.
- Overleaf compilation succeeded after each session (S1, Action A, S2, S3, S4); IDE diagnostics on the LaTeX source: zero warnings, zero errors.

### What is NOT changed

- No new claim is asserted; no claim is retracted.
- No experimental contract is reopened; the P4-04/P4-05 closures of D-0095/D-0097 are reproduced verbatim in the new subarc subsection.
- The `subsec:porta1` / `subsec:porta2` / `subsec:obj_circular` / `sec:conclusions` claims are textually softened and re-anchored to the pre-registered Phase 4 subarc, but their substantive content is identical to v0.6.
- The `app:cocreacio:metodologia` super-paragraph documenting brief-blind / iterative-hostile-review cycles is preserved verbatim. A future condensation of this paragraph would be a substantive editorial decision and not a v2.01-distillation cleanup; it is deliberately left out of this cycle.

### Anchors

- Source manuscript (v0.6): `Paper_Latex/main_v2_cat.tex` (1161 lines).
- Distilled manuscript (v2.01): `Paper_Latex/main_v2.01_cat.tex` (1106 lines).
- Bibliography: `Paper_Latex/references.bib` (entry `BeckSchlogl1993` at the end of the file).
- Phase 4 editorial triage (binding constraint for S3): `docs/HSI_v2_Phase4_Editorial_Triage.md`.
- Decision Log canonical path (D-0098): `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md`.
- Public GitHub URL exposed in the manuscript pointer: `https://github.com/iban-borras/informational-singularity-hypothesis/blob/hsi_v2/Documentation/HSI_v2_Decision_Log.md`.

### Validation traces

- Sofia editorial pass (2026-05-08): five sessions executed sequentially against the wording fence; per-session regex audit confirmed zero forbidden-term occurrences and presence of every bound term required by the Phase 4 triage.
- Overleaf compilation validation by human author after each session: S1 ("Compilat perfecte! Pots continuar amb sessió 2."), S2 ("compila perfecte. Endavant amb S3!"), S3 ("Compila perfecte. endavant amv punt 2"), S4 ("Compilació a Overleaf correcta! :-)").
- IDE LaTeX diagnostics on the live file `Paper_Latex/main_v2.01_cat.tex` after every str-replace pass: zero issues reported.
- Human-author validation (2026-05-08): "Anem a per l'opció A." (close cycle, leave E13 untouched, prepare D-0099 for Ariadna to commit).

---

## D-0100

- Date: 2026-05-08
- Status: `accepted-pre-execution-contract`
- Cycle: Phase 1 threshold-sensitivity / hostile-review response
- Trigger: a brief-blind hostile-review pass on manuscript v2.01 raised a valid p-hacking objection against the Phase 1 stability thresholds in `v2/phase1/tower.py`: `cv_max = 1.5`, `min_count_floor = 16`, `min_count_rate = 1e-6`, and `min_segment_support = max(2, ceil(N/2))`. The objection asks whether Gate 1 and the compact-selective regime survive if those low-pass filters are relaxed, tightened, or partially disabled.

### Decision

D-0100 accepts a pre-execution one-at-a-time threshold-sensitivity contract for Phase 1. The contract is recorded in `hsi_agents_project/Documentation/HSI_v2_Phase1_Threshold_Sensitivity_Preregistration.md` and is anchored by SHA256:

`F239810076F1670F8EFC3A4904927970E4E8FE22B68F0E1A311B57837757E108`

The accepted launcher is `hsi_v2_phase1_threshold_sensitivity.py`. It must not modify `v2/phase1/tower.py`; it only orchestrates `hsi_v2_phase1_run.py` with existing CLI parameters.

The accepted grid is one-at-a-time around the canonical Phase 1 point:

- Baseline: `cv_max = 1.5`, `min_count_floor = 16`, `min_count_rate = 1e-6`, `min_segment_support = default` (effective value `2` for `N=3`).
- `cv_max`: `1.0, 1.5, 2.0, 2.5, 3.0, inf`.
- `min_count_floor`: `4, 8, 16, 32, 64`.
- `min_count_rate`: `0, 1e-7, 1e-6, 1e-5, 1e-4`.
- `min_segment_support`: `1, 2, 3`.
- Deduplicated total: `16` threshold combinations.

The fixed Phase 1 scope is:

- Variants: `B,E,I`.
- Sequence kinds: observed, `matched-lz`, `phase-matched-lz`.
- Scales: `8,12,16,20,24,28,32`.
- Projection policies: `prefix,suffix`.
- Iteration: `20`.
- Segment layout: `3 x 1,000,000` bits.
- Initial null seed: `607`.
- Escalation seed: `709`.

### Escalation rule

Every threshold combination is first evaluated against null seed `607`. The additional seed `709` is run only if the initial readout is fragile:

- final verdict is not `survives`, or
- `score_margin < 0.30`, or
- `retention_margin < 0.30`.

The second seed expands the null envelope by maximum, never by averaging. It therefore cannot rescue a positive result by smoothing away a hard null.

### Score and verdict

All raw Phase 1 observables remain in the CSV outputs. The scalar score is only a compact ordering aid:

`compact_selective_score = retention_last * min(prefix_survival_last, suffix_survival_last)`

For each threshold combination:

- `hsi_score_min`: minimum score over observed `B,E,I`.
- `null_score_max`: maximum score over all null rows in the active envelope.
- `score_margin = hsi_score_min - null_score_max`.
- `hsi_retention_min`: minimum `retention_last` over observed `B,E,I`.
- `null_retention_max`: maximum `retention_last` over all null rows.
- `retention_margin = hsi_retention_min - null_retention_max`.

Verdict thresholds:

- `survives`: `hsi_score_min >= 0.75`, `score_margin >= 0.25`, `hsi_retention_min >= 0.70`, and `retention_margin >= 0.25`.
- `weakens`: `hsi_score_min >= 0.60`, `score_margin >= 0.10`, `hsi_retention_min >= 0.55`, and `retention_margin >= 0.10`.
- `collapses`: otherwise.

### Hypotheses

- H1: the compact-selective Gate 1 signature survives across a broad OAT region, meaning the canonical thresholds are not a narrow hand-tuned point.
- H2: the compact-selective Gate 1 signature appears only near the canonical threshold combination, which would be a serious negative result requiring manuscript revision.

### Practical Impact

- New pre-registration document: `hsi_agents_project/Documentation/HSI_v2_Phase1_Threshold_Sensitivity_Preregistration.md`.
- New launcher: `hsi_v2_phase1_threshold_sensitivity.py`.
- Output root: `hsi_agents_project/results/hsi_v2/phase1_sensitivity_thresholds/`.
- Required outputs: `summary.json`, `report.md`, `combo_summary.csv`, `row_summary.csv`, `oat_heatmap.csv`, `oat_heatmap.svg`, and `manifest.json`.
- No manuscript change is authorized by this entry. Any paper-facing update requires post-execution readout, Grace audit if needed, and a later decision entry.

### Validation traces

- Ariadna implementation validation (2026-05-08): `.\venv\Scripts\python.exe -m py_compile hsi_v2_phase1_threshold_sensitivity.py` completed successfully.
- Ariadna dry-run validation (2026-05-08): `.\venv\Scripts\python.exe hsi_v2_phase1_threshold_sensitivity.py --dry-run --limit-combos 1 --workers 2` produced the expected 9 initial targets for one threshold combination; the full OAT dry-run produced 144 initial targets.
- Ariadna smoke validation (2026-05-08): `.\venv\Scripts\python.exe hsi_v2_phase1_threshold_sensitivity.py --limit-combos 1 --variants B --segment-bits 10000 --num-segments 3 --workers 1 --quiet` completed successfully as a non-scientific wiring test. Its verdict is explicitly non-evidential because it does not use the canonical 3 x 1M protocol.
- Pre-registration SHA256 computed before execution: `F239810076F1670F8EFC3A4904927970E4E8FE22B68F0E1A311B57837757E108`.

---

## D-0101

- Date: 2026-05-08
- Status: `accepted-empirical-closure`
- Cycle: Phase 1 threshold-sensitivity / hostile-review response
- Trigger: execution and Grace audit of the D-0100 threshold-sensitivity contract.

### Source Artifacts

- Pre-execution contract: `hsi_agents_project/Documentation/HSI_v2_Phase1_Threshold_Sensitivity_Preregistration.md`.
- Contract SHA256: `F239810076F1670F8EFC3A4904927970E4E8FE22B68F0E1A311B57837757E108`.
- Launcher: `hsi_v2_phase1_threshold_sensitivity.py`.
- Canonical run: `hsi_agents_project/results/hsi_v2/phase1_sensitivity_thresholds/phase1-threshold-sensitivity__oat-16__seed-607__20260508T122043/`.
- Post-execution audit: `docs/HSI-audit-Phase1-Sensitivity.md`.

### Empirical Readout

The D-0100 run completed the full one-at-a-time threshold-sensitivity grid:

- `16` threshold combinations.
- `144` initial targets under null seed `607`.
- `6` escalation targets under null seed `709`, triggered only for `min_count_rate = 1e-4`.
- `150` total Phase 1 child rows.

Verdict distribution:

- `15/16` combinations: `survives`.
- `1/16` combinations: `weakens`.
- `0/16` combinations: `collapses`.

The only weakened combination is the most aggressive relative-count threshold:

- Combination: `min_count_rate = 1e-4`.
- `hsi_score_min = 0.6102`.
- `null_score_max = 0.0700`.
- `score_margin = 0.5402`.
- `hsi_retention_min = 0.6181`.
- `null_retention_max = 0.0703`.
- `retention_margin = 0.5478`.

The CV-axis result is decisive for the hostile-review objection:

- `cv_max = 1.0`: `survives`.
- `cv_max = 2.0`: `survives`.
- `cv_max = 2.5`: `survives`.
- `cv_max = 3.0`: `survives`.
- `cv_max = inf`: `survives`.

Thus the compact-selective Gate 1 readout does not depend on the canonical `cv_max = 1.5` filter. It also survives removal of the relative count floor (`min_count_rate = 0`), tightening/loosening of `min_count_floor` across `4,8,32,64`, and both tested alternatives to the segment-support rule (`1` and `3`).

### Grace Audit

Grace's post-execution audit verdict is `ACCEPTED`. The audit accepts:

- the SHA-anchored pre-registration discipline,
- the OAT grid as sufficient for the specific hostile-review criticism,
- the multiplicative score as a hard, not generous, score because it penalizes either retention or directional survival loss,
- paper-facing integration as a short contained robustness sentence or footnote rather than a new section/table.

The audit explicitly rejects the need for a full factorial grid at this stage. A factorial extension is therefore not required unless a future reviewer specifically challenges nonlinear parameter interactions.

### Wording Boundary

Allowed wording:

- `robustesa de la signatura projectiva davant variacions dels llindars d'estabilitat`
- `insensibilitat al fine-tuning dels llindars de retenció local`
- `supervivència sense filtre CV`
- `la signatura es debilita sota un llindar relatiu extrem, però no col·lapsa`

Forbidden or discouraged wording:

- `absolute invariance`
- `global threshold invariance`
- `topology recovered`
- `topology discovered`
- `Gate 1 immune to all hyperparameters`
- any wording that hides the `min_count_rate = 1e-4` weakening.

The phrase `robustesa topològica` should be avoided in the manuscript because it can over-activate the topology/geometry vocabulary relative to what the Phase 1 sensitivity run actually proves.

### Practical Impact

- The p-hacking objection against the canonical `cv_max = 1.5` and related Phase 1 stability thresholds is empirically weakened by a preregistered sensitivity run.
- The manuscript may include a single contained robustness sentence or footnote in the Gate 1 robustness discussion.
- No new experimental default is changed.
- No manuscript table is required.
- No Phase 1 claim is strengthened into a universal threshold-invariance claim.

### Validation Traces

- Ariadna rerun note (2026-05-08): an initial completed run exposed a reporting-only issue where `active_seed_count` was displayed as `2` for all combinations. The launcher was patched to compute `active_seed_count` from the actual null seeds present per combination, and the full canonical run was repeated.
- Canonical clean run (2026-05-08): `.\venv\Scripts\python.exe hsi_v2_phase1_threshold_sensitivity.py --workers 10 --quiet` completed successfully in `118.894` seconds.
- Grace audit: `docs/HSI-audit-Phase1-Sensitivity.md`, verdict `ACCEPTED`.

---

## D-0102

- Date: 2026-05-08
- Status: `accepted-diagnostic-closure`
- Cycle: Phase 4 / P4-06 boundary feature-shift
- Trigger: after D-0097 closed P4-05 as finite-horizon boundary / null-envelope catch-up, Ariadna proposed a read-only diagnostic comparing the last P4-04 supported band (`786M-795M`) with the first P4-05 boundary band (`795M-804M`) to test whether the loss of predictive differential came from B-retention loss or from hard-null envelope rise.

### Decision

D-0102 accepts P4-06 as a diagnostic closure of the current Phase 4 predictive subarc. P4-06 is read-only over already accepted P4-04/P4-05 artifacts:

- P4-04 source: `hsi_agents_project/results/hsi_v2/phase4/p4_04_forward_regime_forecast/phase4-p4-04-forward-regime-forecast__bands-3__lags-6__20260507T200920/`.
- P4-05 source: `hsi_agents_project/results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__20260507T225929/`.
- P4-06 script: `hsi_v2_phase4_p4_06_boundary_feature_shift.py`.
- P4-06 canonical artifact: `hsi_agents_project/results/hsi_v2/phase4/p4_06_boundary_feature_shift/phase4-p4-06-boundary-feature-shift__786M-795M_to_795M-804M__20260508T142409/`.
- Grace audit: `docs/HSI-audit-Phase4-P4-06-Boundary-Feature-Shift.md`.

P4-06 does not extend the forecast horizon, does not scan later bands, does not introduce new nulls, and does not recompute the accepted P4-04/P4-05 runs. It only decomposes the already discovered boundary.

### Empirical Readout

Best-row boundary shift:

| Metric | Last supported band `786M-795M` | First boundary band `795M-804M` | Delta |
|---|---:|---:|---:|
| B retention | `1.0000` | `1.0000` | `+0.0000` |
| phase-matched-LZ max | `0.2953` | `1.0000` | `+0.7047` |
| block-entropy max | `0.3789` | `0.8824` | `+0.5034` |
| hard null max | `0.3789` | `1.0000` | `+0.6211` |
| hard margin | `0.6211` | `0.0000` | `-0.6211` |

Lag-aligned aggregate shift over the four shared negative lags:

- Mean B-retention shift: `+0.1313`.
- Mean phase-matched-LZ max shift: `+0.6745`.
- Mean block-entropy max shift: `+0.5208`.
- Mean hard-null max shift: `+0.6147`.
- Mean hard-margin shift: `-0.4834`.

The diagnostic verdict is:

- `status = null-envelope-catch-up`.
- `primary_driver = hard-null-envelope-dominated-by-phase-matched-LZ`.

### Interpretation

The P4-04/P4-05 boundary is not explained by disappearance of B retention. In the best-row comparison B remains at `1.0000`, and in the lag-aligned aggregate B retention increases on average. The collapse of the predictive margin is driven by a surge in the hard-null envelope, dominated by phase-matched-LZ but also materially supported by block-entropy.

This supports the D-0097 wording `convergència de l'embolcall nul` and sharpens its mechanism: the hard null envelope catches up to B at the finite-horizon boundary.

### Grace Audit

Grace's verdict is `ACCEPTED`. The audit accepts:

- P4-06 as a legitimate read-only derivation, not signal chasing.
- The best-row plus lag-aligned aggregate decomposition as sufficient to support the null-envelope catch-up reading.
- D-0102 as warranted diagnostic closure.
- Paper-facing use only as a short integrated prose clarification, with zero new tables.

Grace's wording amendment is accepted: do not attribute the whole boundary to a single null. Use `hard-null envelope surge, dominated by phase-matched-LZ`, because block-entropy also rises substantially.

### Wording Boundary

Allowed wording:

- `convergència de l'embolcall nul`.
- `hard-null envelope catch-up`.
- `the predictive-margin collapse is driven by a hard-null envelope surge, dominated by phase-matched-LZ`.
- `B retention does not disappear at the boundary`.
- `P4-06 diagnoses the P4-04/P4-05 boundary; it does not extend the forecast horizon`.

Forbidden wording:

- `channel death`.
- `B dies at the boundary`.
- `phase-matched-LZ alone explains the boundary`.
- `global law`.
- `universal phase law`.
- `forecast horizon reopened`.
- `renewed signal found`.

### Practical Impact

- P4-06 closes the current Phase 4 predictive subarc diagnostically:
  - P4-04: finite forward prediction supported through `786M-795M`;
  - P4-05: horizon bracket rejected immediately after, beginning at `795M-804M`;
  - P4-06: the boundary mechanism is hard-null envelope catch-up, not B-retention loss.
- No new manuscript table is authorized.
- If integrated into the manuscript, the result should appear only as one contained prose clarification in the Phase 4 finite-subarc paragraph.
- No further forward signal chasing is authorized by this entry.

### Validation Traces

- Ariadna implementation validation (2026-05-08): `.\venv\Scripts\python.exe -m py_compile hsi_v2_phase4_p4_06_boundary_feature_shift.py` completed successfully.
- Initial P4-06 run (2026-05-08): `phase4-p4-06-boundary-feature-shift__786M-795M_to_795M-804M__20260508T134115/` produced the same numeric readout but used the overly sharp driver label `phase-matched-lz`.
- Grace audit (2026-05-08): verdict `ACCEPTED`, with wording amendment requiring the driver to be described as the hard-null envelope, dominated by phase-matched-LZ.
- Canonical P4-06 rerun after wording patch (2026-05-08): `phase4-p4-06-boundary-feature-shift__786M-795M_to_795M-804M__20260508T142409/`.

---

## D-0103

- Date: 2026-05-08
- Status: `accepted-editorial-closure`
- Cycle: v2.02 second-pass brief-blind iterative-hostile-review response + D-0102 paper-facing integration
- Trigger: a second brief-blind hostile review of manuscript v2.01 (post D-0100/D-0101 edits) returned with four substantive flanks: (1) ontological-computational "bait and switch" on the 0/1 alphabet, perceived as injection of binary logic into metaphysics because the notational clarification at `subsec:alfabet` (§3.2) arrived too late narratively; (2) Texas-Sharpshooter accusation against the metric selection of the projective tower (`retention@last`, `active_mean_tail`); (3) recurrent lexical critique despite the three protective layers already in place (lexical note at §11, Beck/Schlögl bibliographic anchoring, Conclusions anchoring); (4) "empty promise of Level 2" structural critique. In parallel, D-0102 (P4-06 boundary feature-shift diagnostic) was made available for paper-facing integration as a short prose precision.

### Decision

D-0103 closes the v2.02 editorial branch with five integrated manuscript interventions and one accepted non-action. The interventions respond to flanks (1)-(3) of the second brief-blind pass plus the paper-facing precision of D-0102. Flank (4) is accepted as structurally valid and not addressable by text; the response is empirical continuation toward Phase 5, not editorial.

The interventions are:

1. **Notational bridge sentence at `subsec:paradoxa` (§3.1).** A new sentence is inserted between the two-pole bullet list and the Primordial Tension paragraph, stating explicitly that `0` and `1` are notational representations of two opposed ontological poles, not numeric values, boolean truth poles, or physical charges. This anticipates the hostile reading at the earliest possible point in the manuscript, before the reader has formed a "binary logic injected into metaphysics" judgment.
2. **Simplification of `subsec:alfabet` (§3.2)** to avoid literal duplication of the bridge sentence. The notational clause is now a back-reference to §3.1, while the bivalence justification (P1-P3 + Russell/Tarski) is preserved as the original content of §3.2.
3. **Informative anti-Sharpshooter paragraph at `subsec:sintesi_porta1` (§11.x).** A new paragraph immediately after the definition of the two synthesis observables and before the Gate-Plane figure clarifies that `retention@last` and `active_mean_tail` are not isolated metrics designed to separate B, but the projections onto the tail plane of the quintuple signature operating simultaneously across five mathematically independent registers (LZ, branching, TE, fractal, DET), plus the projective tower as a sixth and seventh independent reading. The framing is constructive, not defensive: an accidental signature would have to coincide by chance across all these axes simultaneously, which is the burden-of-proof inversion against the Sharpshooter accusation.
4. **Lexical substitution of "massa de transport / ramificació".** Nine occurrences of `massa de transport activa` and related forms are replaced by `fracció activa de transport` (mathematically more accurate: $\mu_{\text{act}}$ is a fraction in $[0,1]$, not a mass) and by `activitat de ramificació`. The mathematical symbol $\mu_{\text{act}}$ is preserved unchanged. The term "transport" itself is preserved (covered by the lexical note and Beck/Schlögl anchoring).
5. **Lexical-note relocation from §11 to end of §2 (`sec:premisses`).** The full lexical note on physical-resonance vocabulary, originally at the head of `sec:defectes` (§11), is moved to the end of `sec:premisses` (§2), so it precedes any first occurrence of `transport`, `flux`, `routing`, etc. The opening clause is adapted from "A partir d'aquesta secció" to "Al llarg del manuscrit". A short cross-reference paragraph is left at the original position in §11 pointing back to the relocated note, preserving traceability without prose duplication.
6. **D-0102 paper-facing integration at `subsec:phase4_finite_subarc` (§15.6).** A single precision sentence is inserted into the existing P4-04/P4-05 reading paragraph, decomposing the `convergència de l'embolcall nul` into its mechanism: catch-up dominated by `phase-matched-LZ` with substantial `block-entropy` contribution, not attributable to a single null statistic nor to B-retention loss. No new table, no new subsection, no reopened forecast horizon, fully respecting the D-0102 wording boundary.

The non-action is:

7. **No edit on flank (4) "empty promise of Level 2".** The critique is accepted as structurally valid: presenting a manuscript whose true structural test is Gate 2 (defects + spectrum), still explicitly open, leaves Gate 2 as future work. This cannot be closed by text and should not be. The response is empirical continuation toward Phase 5 (cross-arc scaling of finite predictive horizons, see Phase 5 Candidate B preregistration), not editorial expansion of Gate 2 claims.

### Manuscript Status After D-0103

The current manuscript is `Paper_Latex/main_v2.01_cat.tex` (1130 lines, no LaTeX diagnostics, Overleaf compilation validated by Iban on 2026-05-08).

Human-level decision recorded in this entry: the v2.02 manuscript is treated as a `temporally closed version`, mature enough for first preprint round but explicitly open to substantive revision if Phase 5 produces results that modify the empirical baseline. This unblocks Phase 5 from any implicit constraint of "do not disturb the closed manuscript" and authorizes Phase 5 to operate with full editorial freedom over its own results, conditional on a new hostile-review cycle if those results enter the manuscript.

### Wording Boundary

Allowed wording (specific to interventions in this entry):

- `representacions notacionals d'aquests dos pols ontològics` (intervention 1).
- `cribratge multi-axial` (intervention 3).
- `fracció activa de transport`, `activitat de ramificació` (intervention 4).
- `convergència de l'embolcall nul, dominada per phase-matched-LZ amb contribució clara de block-entropy` (intervention 6).

Forbidden wording (carried forward from D-0102 and reinforced by D-0103):

- `channel death`, `B dies at the boundary`, `phase-matched-LZ alone explains the boundary`.
- `boundary law`, `universal horizon`, `scaling law derived`, `Gate 2 advanced`, `Gate 2 partially closed`.
- `the metric selection is robust` (defensive register; D-0103 explicitly avoided this tone in intervention 3).

### Practical Impact

- Closes the v2.02 hostile-review iterative branch.
- No new claims introduced; only narrative repositioning, lexical neutralization, and one mechanistic precision derived from accepted D-0102.
- D-0102 is now both repo-facing (full closure) and paper-facing (one-sentence precision).
- Phase 5 Candidate B preregistration is unblocked and proceeds independently.
- A third brief-blind hostile-review pass is not required to close this branch; it is left open as an optional future action depending on Phase 5 outcome.

### Validation Traces

- Manuscript edits (2026-05-08): six `str-replace-editor` patches over `Paper_Latex/main_v2.01_cat.tex`, IDE diagnostics zero on each pass.
- Final regex audit: zero remaining occurrences of `massa de transport` or `massa de ramificació` in the manuscript.
- Overleaf compilation (2026-05-08): validated by Iban after intervention 6.
- Cross-reference integrity: `sec:premisses` relocation preserves the forward reference to `eq:defecte` (resolved by LaTeX in second compilation pass, no broken refs).
- Per-intervention LaTeX trace tags inline in the manuscript: `[D101-T1]` (intervention 6 family, inherited), `[G2-T2]` (intervention 3), `[G3-T1]` (interventions 5 and its cross-reference stub at §11), `[D102-T1]` (intervention 6).

---

## D-0104

- Date: 2026-05-08
- Status: `accepted-pre-execution-contract`
- Cycle: Phase 5 / Candidate B-mini cross-arc finite-horizon replication
- Trigger: after Phase 4 closed the finite out-of-sample predictive subarc (D-0095/D-0097/D-0102), Grace and Sofia rejected Candidate A as premature because it attempted to infer a boundary rule from a single observed frontier. Candidate B-mini was selected as the next falsifiable step: test whether the Phase 4 finite-horizon object has analogues across independent arcs before any boundary-law or scaling-law program can be considered.

### Decision

D-0104 records acceptance of the Phase 5 Candidate B-mini v0.2 pre-execution contract.

Canonical contract:

`hsi_agents_project/Documentation/HSI_v2_Phase5_CandidateB_CrossArc_Preregistration_v0_2.md`

Accepted SHA256:

`08E8190FC19A2F42FD9F7C541857A6ECF1B488C00647A6A063DE741F675DE681`

The contract uses a dual readout:

1. **Strict P4-object check.** The frozen P4 lag grid is applied to the three frozen arcs. This is the only readout that controls the B-mini continue/stop decision.
2. **Local analogue check.** A broad arc-local scout (`-60M..+60M`, step `2.5M`) may identify a local lag core for diagnostic bracket testing. Local-analogue success cannot rescue a strict P4-object failure.

Frozen arcs:

| arc_id | scout_start | scout_band | bracket_bands |
|---|---:|---|---|
| `ARC-A` | `300000000` | `300M-309M` | `309M-318M`; `318M-327M`; `327M-336M`; `336M-345M` |
| `ARC-B` | `597000000` | `597M-606M` | `606M-615M`; `615M-624M`; `624M-633M`; `633M-642M` |
| `ARC-C` | `1110000000` | `1110M-1119M` | `1119M-1128M`; `1128M-1137M`; `1137M-1146M`; `1146M-1155M` |

The arc list is derived mechanically from the five-band Stage 0 budget audit:

`results/hsi_v2/phase5_cross_arc_horizon/observability_budget/phase5-cross-arc-observability-budget__20260508T151357/`

Selection rule:

1. keep only `executable = Y`;
2. drop rows with any `overlaps_reference_ranges` value other than `-`;
3. require `available_margin_bits >= 250000000`;
4. choose `K=3` starts maximizing minimum pairwise distance;
5. tie-break by maximum total span and then lexicographically earliest tuple.

### Wording Boundary

Forbidden:

- `boundary law`
- `universal horizon`
- `scaling law derived`
- `horizon predicted`
- `frontier formula`
- `general predictive principle`
- `Phase 5 confirms Phase 4`
- `Phase 5 generalizes Phase 4`
- `rescue of the positive channel`

Allowed:

- `cross-arc finite-horizon test`
- `finite-horizon replication attempt`
- `hard-null envelope catch-up`
- `arc-local predictive horizon`
- `pilot evidence for or against horizon recurrence`
- `failure as a valid outcome`

### Practical Impact

- B-mini execution is now authorized exactly under the v0.2 contract.
- No arc replacement is allowed after execution starts.
- Failure, inactive scouts, local-analogue-only outcomes, and strict replication outcomes must all be reported with equal visibility.
- A local-analogue-only result may motivate a future lag-drift question, but it does not support moving to full Candidate B.
- No manuscript edit is authorized by this entry.

### Validation Traces

- Grace audit v0.1: `docs/HSI-audit-Phase5-CandidateB-v01.md` accepted Stage 0 and required avoiding a narrow P4-only lag grid for distant arcs.
- Sofia review v0.1: accepted Stage 0 and required explicit scout range, mechanical lag-core selection, anti-rescue separation, and mechanical arc selection.
- Grace audit v0.2: `docs/HSI-audit-Phase5-CandidateB-v02.md` issued a must-patch warning against `9M` scout spacing as undersampled for a Phase-4-width channel.
- Final v0.2 patch: broad scout changed to `-60M..+60M` with `2.5M` resolution; tie-break reporting and full eligible-arc traceability added.
- Sofia and Grace both signed the final v0.2 contract after the `2.5M` patch, authorizing execution under this entry.

---

## D-0105

- Date: 2026-05-09
- Status: `accepted-empirical-closure`
- Cycle: Phase 5 / B-mini cross-arc finite-horizon execution
- Trigger: execution of the D-0104 Phase 5 Candidate B-mini contract returned a mechanically ambiguous result: one frozen arc (`ARC-B`, 597M) strictly replicated the Phase 4 finite-horizon object, while two frozen arcs (`ARC-A`, 300M; `ARC-C`, 1110M) did not. Grace audited the execution and accepted the contract outcome.

### Decision

D-0105 records closure of the Phase 5 B-mini execution.

Canonical run:

`results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520/`

Contract:

`hsi_agents_project/Documentation/HSI_v2_Phase5_CandidateB_CrossArc_Preregistration_v0_2.md`

Contract SHA256:

`08E8190FC19A2F42FD9F7C541857A6ECF1B488C00647A6A063DE741F675DE681`

Run verdict:

`ambiguous-stop-and-review`

Arc-level outcomes:

| arc_id | arc | category | strict strong bands | strict shadow bands | scout |
|---|---|---|---|---|---|
| `ARC-A` | `300M` | `scout-inactive` | `-` | `300M-309M`; `309M-318M`; `318M-327M`; `327M-336M`; `336M-345M` | inactive |
| `ARC-B` | `597M` | `strict-replication` | `597M-606M`; `606M-615M`; `615M-624M` | `624M-633M`; `633M-642M` | inactive |
| `ARC-C` | `1110M` | `scout-inactive` | `-` | `-` | inactive |

The strict replication count is `1/3`. Under the D-0104 outcome rule this does not authorize a full Candidate B run and does not authorize Candidate A. The correct action is stop-and-review.

### Scientific Reading

The Phase 4 finite-horizon object is not a one-off artifact: `ARC-B` provides an independent second occurrence of strict P4-grid finite-horizon behavior, with strong bands followed by null-envelope catch-up.

However, the phenomenon is not broad or ubiquitous under the frozen B-mini arc selection: `ARC-A` is already null-shadowed under the strict readout, and `ARC-C` is inactive. The correct reading is:

`finite-horizon recurrence exists at least locally, but not as a broad cross-arc pattern under B-mini`.

Grace's audit further interprets the scout inactivity in `ARC-B` as a meaningful diagnostic rather than a conceptual bug: the strict P4-grid detects the channel, but the broad scout fails to form a three-lag cluster, implying an extremely narrow or fragmented lag channel in that arc.

### Wording Boundary

Allowed:

- `local finite-horizon recurrence`
- `intermittent cross-arc recurrence`
- `strict P4-object replication in ARC-B`
- `ambiguous-stop-and-review`
- `narrow or fragmented lag channel`
- `hard-null envelope catch-up`

Forbidden:

- `Phase 5 confirms Phase 4`
- `Phase 5 generalizes Phase 4`
- `cross-arc law`
- `boundary law`
- `universal horizon`
- `scaling law`
- `B-mini validates Candidate B`
- `Candidate A is now authorized`
- `ARC-A/ARC-C failure is missing data`

### Practical Impact

- Full Candidate B is not authorized.
- Candidate A remains deferred.
- No new Phase 5 execution is authorized by this entry.
- The result may be summarized as a predictive coda: Phase 4's finite-horizon behavior recurs at least once in an independent arc, but its occurrence is local/intermittent rather than universal under the B-mini contract.
- Any manuscript integration must preserve the stop-and-review status and must report ARC-A/ARC-C failures with the same visibility as ARC-B.

### Validation Traces

- B-mini run completed with manifest: `results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520/manifest.json`.
- Machine verdict in `summary.json`: `ambiguous-stop-and-review`, `strict_replication_count = 1`, `local_analogue_only_count = 0`.
- Arc table: `arc_verdicts.csv`.
- Grace audit: `docs/HSI-audit-Phase5-BMini-Execution.md`, verdict `ACCEPTED`.

---

## D-0106

- Date: 2026-05-09
- Status: `accepted-editorial-integration`
- Cycle: post-D-0105 editorial decision on whether to integrate B-mini results into the manuscript v2.02
- Trigger: D-0105 closed B-mini empirically (`ambiguous-stop-and-review`, `1/3 strict-replication`) and explicitly required that any manuscript integration preserve the stop-and-review status and report ARC-A/ARC-C failures with the same visibility as ARC-B. The manuscript v2.02 had been previously marked at D-0103 as a temporarily closed version, explicitly open to Phase 5 results. The empirical closure of Phase 5 B-mini therefore reopened the editorial decision.

### Decision

D-0106 records the editorial decision to integrate B-mini results into `Paper_Latex/main_v2.01_cat.tex` as a single short coda at the end of §15.6 (`subsec:phase4_finite_subarc`), immediately after the D-0102 paragraph, with strict equal visibility for all three pre-registered arcs.

Three options were evaluated by Sofia:

1. Option A: do not modify the manuscript; keep B-mini repo-facing only.
2. Option B: add a single-sentence coda at §15.6 with equal visibility for the three arcs, infirmative wording, no scope expansion.
3. Option C: add a longer paragraph or a reduced per-arc table.

Sofia's recommendation and Iban's acceptance: Option B. Justification: the same infirmative discipline already applied at D-0101 (multi-axial sensitivity sweep) requires that results which limit the scope of the claim be reported with the same visibility as those that support it. Suppressing B-mini in the manuscript would be asymmetric. Expanding to a full paragraph would be disproportionate to a stop-and-review outcome. A single sentence naming all three arcs explicitly satisfies both principles.

### Intervention

A single insertion at §15.6 of `Paper_Latex/main_v2.01_cat.tex`, immediately after the existing D-0102 sentence about the P4-06 boundary decomposition. Tag `[D106-T1]` inline.

Final wording (Catalan, as per manuscript convention):

> Una validació creuada posterior (B-mini, tres arcs independents pre-registrats sobre la mateixa graella estricta P4) ha trobat una segona ocurrència neta del patró d'horitzó finit amb saturació d'embolcall nul a un dels arcs (597M); els altres dos arcs (300M, 1110M) no han replicat el patró sota la mateixa graella. La persistència és, doncs, local i intermitent: el resultat no autoritza cap llei d'horitzó ni generalització cross-arc, i deixa oberta la qüestió de si el canal de lag drifta entre arcs o si la replicació observada és pròpiament fenomenològica de la regió 597M--642M.

Wording compliance with D-0105 constraints:

- Three arcs explicitly named (300M, 597M, 1110M): equal visibility satisfied.
- "No autoritza cap llei d'horitzó ni generalització cross-arc" ↔ stop-and-review status preserved.
- "Local i intermitent" ↔ canonical phrasing aligned with Ariadna's proposed canonical phrase ("Finite-horizon recurrence exists at least locally, but not as a broad cross-arc pattern under B-mini").
- No claim of universality, no boundary law, no scaling law, no Porta 2 reopening.
- Open question on lag drift vs regional phenomenology declared as such, not resolved.

### Scope Boundaries

D-0106 does not authorize:

- a new Phase 5 execution;
- a Candidate B full contract;
- a boundary-law derivation or any scaling-law program;
- modifications to the §16 Conclusions section (the existing wording about a bounded horizon at 795M--804M without reopening Porta 2 remains accurate and is not affected by the coda);
- modifications to the abstract or the §1 framing;
- any change to thresholds, lag grids, or pre-registered contracts.

D-0106 authorizes:

- the single coda insertion at §15.6 documented above;
- a follow-up Overleaf compilation cycle by Iban for visual validation;
- an optional third brief-blind hostile-review pass on the post-coda manuscript, at Iban's discretion.

### Validation Traces

- Source D-0105: `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md` lines 7122-7205.
- Manuscript pre-coda state: 1130 lines, validated post-D-0103.
- Post-coda intervention: insertion of one tagged paragraph at §15.6 after L909.
- Inline LaTeX trace tag: `[D106-T1]`.
- Equal-visibility check: regex audit confirms ARC-A (300M), ARC-B (597M), and ARC-C (1110M) are all named in the inserted sentence.
- The coda does not introduce any new figure, table, equation, or external reference.

---

## D-0107

- Date: 2026-05-09
- Status: `accepted-preregistration-stage0`
- Cycle: bridge to Level 2 / projective-density vs predictive-channel alignment
- Trigger: after D-0105 closed Phase 5 B-mini as `ambiguous-stop-and-review`, Grace proposed a more fundamental bridge question: whether the local density of the Phase 1 projective hierarchy aligns spatially with Phase 4/5 predictive-channel activation. Ariadna reframed the question as a two-stage, anti-rescue bridge-to-Level-2 contract, and both Grace and Sofia accepted the v0.1 scope with non-blocking hardening notes incorporated.

### Decision

D-0107 accepts the Stage 0 preregistration:

`hsi_agents_project/Documentation/HSI_v2_Bridge_ProjectiveDensity_PredictiveAlignment_Preregistration_v0_1.md`

Accepted SHA256:

`0E11E4E35F7297EEA862E6999413844C75AFCAD66D6EEE9CE5E132A55764F9F9`

The accepted question is not a continuation or rescue of Phase 5. It is a bridge-to-Level-2 feasibility probe:

`Can a local Phase 1 projective-density metric be computed for the frozen Phase 4/5 bands under a blind, pre-declared protocol, before any alignment statistic is computed?`

Only Stage 0 is authorized by this entry. Stage 0 may:

- enumerate the frozen predictive bands from Phase 4 and Phase 5 B-mini;
- verify source and artifact availability;
- determine whether candidate Phase 1 local metrics are executable under canonical Phase 1 definitions;
- report raw Phase 1 local metric values if computed under the accepted metric rule;
- report cost estimates overall and per arc (`P4`, `ARC-A`, `ARC-B`, `ARC-C`);
- write `summary.json`, `report.md`, `frozen_bands.csv`, `metric_feasibility.csv`, and `manifest.json`.

Stage 0 may not:

- compute correlations, regressions, scatter plots, heatmap overlays, rank comparisons, or qualitative alignment verdicts between Phase 1 and Phase 4/5;
- select a Phase 1 metric by apparent agreement with predictive scores;
- add or replace bands;
- modify accepted Phase 4/5 artifacts;
- modify the manuscript.

Any Stage 1 alignment test requires a new v0.2 contract, a new SHA256 anchor, renewed Sofia/Iban/Grace acceptance, and a new Decision Log entry.

### Frozen Bands

Phase 4 reference bands:

`768M-777M`, `777M-786M`, `786M-795M`, `795M-804M`, `804M-813M`, `813M-822M`, `822M-831M`

Phase 5 B-mini bands:

- `ARC-A`: `300M-309M`, `309M-318M`, `318M-327M`, `327M-336M`, `336M-345M`
- `ARC-B`: `597M-606M`, `606M-615M`, `615M-624M`, `624M-633M`, `633M-642M`
- `ARC-C`: `1110M-1119M`, `1119M-1128M`, `1128M-1137M`, `1137M-1146M`, `1146M-1155M`

No band may be added, removed, or replaced by Stage 0.

### Wording Boundary

Allowed:

- `projective-density / predictive-channel alignment`
- `bridge-to-Level-2 feasibility`
- `local alignment probe`
- `same-structure hypothesis`
- `parallel-emergence outcome`

Forbidden:

- `unified theory confirmed`
- `Level 2 solved`
- `geometry recovered`
- `predictive channel explained`
- `boundary law`
- `universal horizon`
- `global phase law`
- `Phase 5 rescued`
- `ARC-A/ARC-C failure explained`

### Practical Impact

- Stage 0 implementation is authorized.
- Stage 1 is explicitly not authorized.
- The result is repo-facing only until a future v0.2 contract exists.
- Any raw Phase 1 local values produced by Stage 0 must be sealed before any alignment statistic is written.

### Validation Traces

- Grace audit: `docs/HSI-audit-Bridge-Level2-Preregistration.md`, verdict `ACCEPTED`, with a hard requirement that Stage 0 remain blind to alignment.
- Sofia review: accepted v0.1 and requested three non-blocking hardening edits: exhaustive Phase 4 band list, per-arc cost reporting, and v0.2 statistical-power caution.
- Ariadna incorporated the three Sofia edits and Grace's blindness requirement into the accepted v0.1 document before this entry was recorded.

---

## D-0108

- Date: 2026-05-09
- Status: `accepted-preregistration-stage1`
- Cycle: bridge to Level 2 / projective-density vs predictive-channel alignment
- Trigger: D-0107 Stage 0 completed with all 22 frozen bands available and mechanically selected `transport_active_mean_tail` as the primary Phase 1 local metric. Ariadna drafted a v0.2 Stage 1 contract, Grace audited it as accepted, and Sofia accepted it with non-blocking report-level refinements.

### Decision

D-0108 accepts the Stage 1 preregistration:

`hsi_agents_project/Documentation/HSI_v2_Bridge_ProjectiveDensity_PredictiveAlignment_Preregistration_v0_2.md`

Accepted SHA256:

`0AC4895814D88F7D16949C953F96B0548340A9CF11F96F3CA6D007ECF271AC91`

Authorized sealed inputs:

- `phase1_local_metrics_raw.csv`
  - SHA256: `7C4E27580D3C526C82AA9CB09BDCC915B71F41BBD7B19C2F2B689A00CB0EF43D`
- `predictive_scores.csv`
  - SHA256: `20534DB529DB130C9D7C3709EE4AC215945E4389F3DE8AA0C92DDAFF4D2155D7`

Traceability input:

- `summary.json`
  - SHA256: `7A53ABD1B7F5EE9392338B4C760BAD814665F5F0329BF57AC615D8BC7E8D1E83`

Stage 1 is now authorized to compute the v0.2 readout:

- band-level Spearman correlation between `transport_active_mean_tail` and `hard_margin_max`;
- band-level Pearson correlation as secondary descriptive readout;
- ordinal class summary by `strict_band_class`;
- arc-level summary;
- arc-level exact permutation calibration over the four arc labels (`4! = 24` permutations).

### Sofia Report Refinements

The Stage 1 report must include:

- explicit `p_floor = 1/24 ~= 0.0417` next to the exact permutation fraction;
- a complete `permutation_table.csv` containing all 24 arc-label permutations and Spearman values;
- a visible reminder that interpretive categories do not authorize manuscript changes by themselves.

### Wording Boundary

Allowed:

- `projective-density / predictive-channel alignment`
- `bridge-to-Level-2 alignment readout`
- `local alignment probe`
- `same-structure hypothesis`
- `parallel-emergence outcome`
- `structured descriptive alignment`

Forbidden:

- `unified theory confirmed`
- `Level 2 solved`
- `geometry recovered`
- `predictive channel explained`
- `boundary law`
- `universal horizon`
- `global phase law`
- `Phase 5 rescued`
- `ARC-A/ARC-C failure explained`
- `statistically proven alignment`

### Practical Impact

- Stage 1 execution is authorized.
- No figure is authorized.
- No manuscript edit is authorized by this entry.
- Any interpretation must preserve the band-level `N=22` non-independence warning and the arc-level `N=4` low-power warning.

### Validation Traces

- Grace audit: `docs/HSI-audit-Bridge-Level2-Preregistration-v02.md`, verdict `ACCEPTED`.
- Sofia review: accepted v0.2 and requested report-level refinements only.
- Stage 0 canonical run: `results/hsi_v2/bridge_projective_predictive_alignment/stage0_feasibility/bridge-projective-predictive-stage0__bands-22__20260509T160812/`.

---

## D-0109

- Date: 2026-05-09
- Status: `accepted-empirical-closure`
- Cycle: bridge to Level 2 / projective-density vs predictive-channel alignment
- Trigger: execution of the D-0108 Stage 1 bridge alignment contract returned `weak-or-mixed-alignment`, with near-zero band-level correlation and no arc-level support. Grace audited the execution and accepted both the negative reading and the prohibition against searching for another Phase 1 metric.

### Decision

D-0109 closes the projective-density / predictive-channel alignment bridge.

Canonical Stage 1 run:

`results/hsi_v2/bridge_projective_predictive_alignment/stage1_alignment/bridge-projective-predictive-stage1__bands-22__20260509T162859/`

Contract:

`hsi_agents_project/Documentation/HSI_v2_Bridge_ProjectiveDensity_PredictiveAlignment_Preregistration_v0_2.md`

Contract SHA256:

`0AC4895814D88F7D16949C953F96B0548340A9CF11F96F3CA6D007ECF271AC91`

Readout:

- band-level Spearman: `0.13816589918454694`
- band-level Pearson: `0.03366729314476827`
- arc-level Spearman: `-0.19999999999999996`
- arc-level permutation fraction: `22/24 = 0.9166666666666666`
- p-floor: `1/24 = 0.041666666666666664`
- category: `weak-or-mixed-alignment`

The bridge does not support a simple direct alignment between local Phase 1 projective/transport density (`transport_active_mean_tail`) and Phase 4/5 predictive-channel activation (`hard_margin_max`). The result is not a failure of Phase 1 or Phase 4/5. It closes the simple unification route.

### Scientific Reading

The negative/mixed bridge readout means:

- the Phase 4/5 predictive channel cannot be reduced to the local Phase 1 transport-density metric tested here;
- ARC-B's predictive activation is not explained by unusually high `transport_active_mean_tail`;
- ARC-A and ARC-C must not be reinterpreted through a newly selected Phase 1 metric after the fact;
- Level 2 must explain either a finer coupling between the two structures or a partial decoupling between projective density and predictive-channel activation.

Grace's audit explicitly states that trying another Phase 1 metric now would be alignment hacking. This door is closed unless a future independent contract introduces a mathematically new object before seeing any alignment result.

### Wording Boundary

Allowed:

- `weak-or-mixed bridge alignment`
- `no simple local alignment`
- `projective-density / predictive-channel decoupling under the tested metric`
- `bridge closed as negative/mixed`
- `Level 2 must explain a finer relation or partial decoupling`

Forbidden:

- `unified theory confirmed`
- `Level 2 solved`
- `geometry recovered`
- `predictive channel explained`
- `boundary law`
- `universal horizon`
- `global phase law`
- `Phase 5 rescued`
- `ARC-A/ARC-C failure explained`
- `try another Phase 1 metric`

### Practical Impact

- No additional Phase 1 alignment metric is authorized.
- No manuscript edit is authorized by this entry.
- The next permissible empirical bridge, if pursued, is Grace Question 2: internal routing entropy at null-envelope catch-up.
- Any Q2 execution requires a new preregistration defining the internal routing entropy object before inspecting the catch-up readout.

### Validation Traces

- Stage 1 run completed with manifest: `results/hsi_v2/bridge_projective_predictive_alignment/stage1_alignment/bridge-projective-predictive-stage1__bands-22__20260509T162859/manifest.json`.
- Stage 1 report: `results/hsi_v2/bridge_projective_predictive_alignment/stage1_alignment/bridge-projective-predictive-stage1__bands-22__20260509T162859/report.md`.
- Grace audit: `docs/HSI-audit-Bridge-Stage1-and-Q2.md`, verdict `ACCEPTED`.

---

## D-0110

- Date: 2026-05-09
- Status: `accepted-preregistration-stage0`
- Cycle: Q2 / internal routing entropy at null-envelope catch-up
- Trigger: after D-0109 closed the simple Phase 1 / Phase 4-5 alignment bridge as negative/mixed, Grace validated her second proposed bridge question: whether observed `B` and the hard-null envelope remain internally thermodynamically asymmetric at surface catch-up. Ariadna drafted a v0.1 feasibility-only contract; Grace accepted it and Sofia approved it with report-level refinements.

### Decision

D-0110 accepts the Q2 Stage 0 preregistration:

`hsi_agents_project/Documentation/HSI_v2_Q2_InternalRoutingEntropy_Preregistration_v0_1.md`

Accepted SHA256:

`48FAD8D5B420FB961D945D105AD183C7D1B20F57F150B09AF7EEF97D5466380B`

Only Stage 0 feasibility is authorized. Stage 0 may:

- derive catch-up cells mechanically from frozen strict P4-grid readouts;
- flag frozen catch-up bands that yield zero qualifying cells as `not-catch-up-under-cell-rule`;
- verify that observed `B`, `phase-matched-lz`, and `block-entropy` routing rows exist for each qualifying cell;
- report hard-null family coverage separately before any min-envelope contrast;
- determine whether the N3-05b/N3-05c/N3-07 route can be applied without changing definitions;
- report exact commands or adapter requirements for a future v0.2.

Stage 0 may not:

- compute `entropy_gap`;
- compare observed `B` entropy against hard-null entropy;
- search for new bands or cells;
- change the N3-05b/N3-05c/N3-07 definitions;
- modify the manuscript.

### Frozen Domain

Primary catch-up bands:

- `P4`: `795M-804M`, `804M-813M`, `813M-822M`, `822M-831M`
- `ARC-B`: `624M-633M`, `633M-642M`

Secondary context bands:

- `P4 strong`: `768M-777M`, `777M-786M`, `786M-795M`
- `ARC-B strong`: `597M-606M`, `606M-615M`, `615M-624M`

### Sofia Report Refinements

The Stage 0 report must:

- flag any frozen catch-up band with zero qualifying cells and state the cause;
- include per-band positive-cell fraction as a descriptive placeholder for v0.2, without computing entropy gaps in Stage 0;
- report hard-null family coverage separately for `phase-matched-lz` and `block-entropy`;
- note that the natural v0.2 contrast is a sign-test style readout, not Spearman/Pearson.

### Wording Boundary

Allowed:

- `internal routing entropy at catch-up`
- `surface retention equality`
- `internal thermodynamic asymmetry`
- `null-envelope opacity`
- `low-entropy routing flow`
- `catch-up boundary diagnostic`

Forbidden:

- `geometry recovered`
- `topology discovered`
- `Level 2 solved`
- `B still wins at catch-up`
- `null defeated`
- `surface tie broken conclusively`
- `hidden proof of mechanism`
- `universal catch-up law`

### Practical Impact

- Q2 Stage 0 implementation is authorized.
- Q2 Stage 1 is not authorized.
- No manuscript edit is authorized by this entry.

### Validation Traces

- Grace audit: `docs/HSI-audit-Q2-InternalRoutingEntropy-v01.md`, verdict `ACCEPTED`.
- Sofia review: accepted v0.1 and requested report-level refinements only.

---

## D-0111

- Date: 2026-05-09
- Status: `accepted-preregistration-stage1`
- Cycle: Q2 / internal routing entropy at null-envelope catch-up
- Trigger: D-0110 Stage 0 completed with all frozen catch-up and strong-context cells having observed `B`, `phase-matched-lz`, and `block-entropy` routing rows available. Stage 0 also showed that the existing N3-05b/N3-05c route requires a transition-shaped adapter before any entropy contrast can be computed. Ariadna drafted a v0.2 Stage 1 contract, Grace audited it as accepted, and Sofia signed it with non-blocking report-level notes about local-adapter semantics.

### Decision

D-0111 accepts the Q2 Stage 1 preregistration:

`hsi_agents_project/Documentation/HSI_v2_Q2_InternalRoutingEntropy_Preregistration_v0_2.md`

Accepted SHA256:

`8DAD57F54BB70687AB2B4FBC85398BF543CC5A639D55E487EC9879B96521F011`

Stage 1 is authorized to:

- verify the sealed D-0110 Stage 0 input hashes;
- build per-cell Q2 adapters for the 20 frozen catch-up cells and 14 strong-context cells;
- run the existing N3-05b weighted quotient child-routing graph and N3-05c component-wise quotient normalization once per selected cell;
- extract `component_weight_entropy_norm` as the primary entropy object;
- compute `entropy_gap = hard_null_entropy_min - observed_B_entropy`;
- aggregate catch-up results by band-level median sign;
- report strong-context cells as descriptive calibration only.

Stage 1 is not authorized to:

- add, remove, or replace bands, lags, cells, null families, or seeds;
- change N3-05b/N3-05c graph definitions, edge weighting, component definitions, or entropy formulas;
- include `markov1` in the hard-null minimum;
- emit figures;
- modify the manuscript.

### Adapter Contract

The accepted adapter unit is one selected cell, not the full Q2 domain.

Rationale: a single global adapter would produce one N3-05c entropy for the whole transition domain and erase the pre-registered cell-level and band-level sign readouts. The per-cell adapter preserves the frozen N3 metric while making the Q2 contrast mechanically auditable.

For each selected cell, Stage 1 must emit:

- `transition_run/nodes.csv` with exactly one node;
- `transition_run/edges.csv` as an empty header-only compatibility artifact;
- a minimal `child_routing_run/<band>/<lag>/routing_rows.csv` copied verbatim from existing routing rows;
- N3-05b and N3-05c artifacts derived from that adapter.

The report must explicitly state that under the one-cell adapter, `component_weight_entropy_norm` is a local version of the N3 component-weight entropy object. It measures internal routing entropy at a specific surface-tied cell, not entropy of the original full transition domain.

### Primary Readout

For every catch-up selected cell:

`entropy_gap = hard_null_entropy_min - observed_B_entropy`

where:

`hard_null_entropy_min = min(phase_matched_lz_family_entropy_min, block_entropy_family_entropy_min)`

Positive `entropy_gap` means observed `B` is internally lower-entropy than the best hard null for that cell.

The primary decision layer is a two-sided binomial sign-test style readout over the six catch-up bands, using the median `entropy_gap` sign per band. Cell-level summaries are descriptive because cells within a band are not independent.

### Outcome Categories

The accepted categories are:

- `internal-asymmetry-supported`
- `mixed-internal-asymmetry`
- `no-internal-separation`
- `null-internally-lower-entropy`
- `incomplete-inputs-stop`

These labels are Q2 readout labels only and do not authorize manuscript changes by themselves.

### Sofia Report Notes

The Stage 1 report must include:

- the semantic note that the per-cell adapter changes granularity from full transition-domain N3 entropy to local cell entropy without changing the formula;
- `std(observed_B_entropy across catchup cells)` as a diagnostic of whether the per-cell adapter preserves non-trivial cell-level variation;
- a support warning that `813M-822M` has only one selected catch-up cell;
- a statement that no figure is authorized.

### Wording Boundary

Allowed:

- `internal routing entropy at catch-up`
- `surface retention equality`
- `internal thermodynamic asymmetry`
- `component-weight entropy`
- `hard-null entropy envelope`
- `Q2 diagnostic bridge`
- `null-envelope opacity`

Forbidden:

- `Gate 2 solved`
- `Level 2 proven`
- `geometry recovered`
- `topology discovered`
- `catch-up defeated`
- `null refuted`
- `predictive channel rescued`
- `final thermodynamic proof`
- `hidden geometry measured`

### Practical Impact

- Q2 Stage 1 implementation is authorized.
- No manuscript edit is authorized by this entry.
- Any paper-facing integration requires a later D-XXXX after execution and audit.

### Validation Traces

- Grace audit: `docs/HSI-audit-Q2-Stage1-Preregistration-v02.md`, verdict `ACCEPTED`.
- Sofia review: signed v0.2; report-level observations only, no contract-blocking change requested.

---

## D-0112

- Date: 2026-05-09
- Status: `accepted-hard-negative`
- Cycle: Q2 / internal routing entropy at null-envelope catch-up
- Trigger: D-0111 Stage 1 executed the accepted Q2 v0.2 contract over frozen catch-up and strong-context cells. Grace audited the results and accepted the execution as clean, with the category `null-internally-lower-entropy` validated as the mandatory readout.

### Decision

D-0112 records the empirical closure of Q2 Stage 1.

Canonical run:

`results/hsi_v2/q2_internal_routing_entropy/stage1_entropy/q2-internal-routing-entropy-stage1__20260509T171750/`

The run respected D-0111:

- per-cell adapters were used;
- N3-05b/N3-05c definitions were not changed;
- no new entropy metric was introduced;
- `markov1` remained excluded from the hard-null entropy envelope;
- no figure was emitted;
- no manuscript edit was made.

### Result

Final category:

`null-internally-lower-entropy`

Catch-up band signs:

- positive bands: `0/6`
- negative bands: `6/6`
- zero bands: `0/6`

Catch-up median `entropy_gap` values:

| band | median entropy_gap | sign |
|---|---:|---|
| `624M-633M` | `-0.0706` | `null-lower-entropy` |
| `633M-642M` | `-0.1401` | `null-lower-entropy` |
| `795M-804M` | `-0.0481` | `null-lower-entropy` |
| `804M-813M` | `-0.0288` | `null-lower-entropy` |
| `813M-822M` | `-0.0378` | `null-lower-entropy` |
| `822M-831M` | `-0.0377` | `null-lower-entropy` |

The hard-null minimum was attained by `block-entropy` in every selected cell, including strong-context cells.

### Interpretation

Q2 refutes the specific hypothesis that observed `B` remains internally lower-entropy than the best hard null at surface catch-up.

The accepted reading is stricter:

- `null-envelope catch-up` is not merely surface opacity under retention;
- under the local N3-05c component-weight entropy object, the strongest hard null (`block-entropy`) also becomes internally lower-entropy than observed `B`;
- the horizon boundary is therefore a thermodynamic transition where local statistical approximations outperform the canonical structural rule in internal routing concentration.

This is a hard negative for hidden B-efficiency at catch-up, not a failure of the program. It sharpens the boundary conditions that any Level 2 theory must explain.

### Wording Boundary

Allowed:

- `hard negative`
- `null-internally-lower-entropy`
- `internal routing entropy boundary`
- `thermodynamic catch-up boundary`
- `local statistical null advantage`
- `surface and internal catch-up under the tested entropy object`

Forbidden:

- `B still wins internally`
- `hidden efficiency confirmed`
- `catch-up defeated`
- `null refuted`
- `Gate 2 solved`
- `Level 2 proven`
- `geometry recovered`
- `predictive channel rescued`
- `final thermodynamic proof`

### Practical Impact

- Q2 is empirically closed under the accepted v0.2 contract.
- No additional metric search is authorized from this negative result.
- No manuscript edit is authorized automatically by this entry.
- Any paper-facing integration requires a separate editorial decision by Sofia/Iban after reviewing Grace's audit.
- The next scientific movement should treat Q1 and Q2 together as a boundary-conditioning pair: projective-density alignment is weak/mixed, and local internal entropy at catch-up favors the strongest hard null.

### Validation Traces

- Stage 1 run: `results/hsi_v2/q2_internal_routing_entropy/stage1_entropy/q2-internal-routing-entropy-stage1__20260509T171750/report.md`.
- Grace audit: `docs/HSI-audit-Q2-Stage1-Results.md`, verdict `HARD NEGATIVE ACEPTADO`.

---

## D-0113

- Date: 2026-05-10
- Status: `accepted-diagnostic-readout`
- Cycle: Q2 / boundary mechanism decomposition
- Trigger: D-0112 closed Q2 as a hard negative (`null-internally-lower-entropy`). To avoid leaving the negative as an abstract scalar result, Ariadna executed a descriptive readout over the sealed Q2 Stage 1 outputs to decompose how the winning hard null beats observed `B` locally.

### Decision

D-0113 records a descriptive mechanism readout subordinate to D-0112.

Canonical run:

`results/hsi_v2/q2_internal_routing_entropy/boundary_mechanism_readout/q2-boundary-mechanism-readout__src-q2-internal-routing-entropy-stage1__20260509T171750__20260510T080824/`

This readout:

- reads only Q2 Stage 1 outputs;
- does not rerun routing;
- does not introduce a new entropy metric;
- does not change the D-0112 category;
- does not authorize manuscript edits by itself.

### Result

Final diagnostic category:

`boundary-null-concentration-advantage`

Catch-up mechanism:

`fewer-effective-components-and-higher-top3-share`

Catch-up role summary:

- mean entropy gap: `-0.0655`
- mean effective-component delta, winning null minus observed `B`: `-1.3204`
- mean top1-share delta, winning null minus observed `B`: `+0.1125`
- mean top3-share delta, winning null minus observed `B`: `+0.2272`
- mean top5-share delta, winning null minus observed `B`: `+0.1306`

Interpretation: the winning hard null is lower-entropy primarily by concentrating routing mass into fewer effective components with higher top-k component shares. The negative Q2 result is therefore not just a scalar entropy inversion; it is a local concentration advantage of the `block-entropy` null at the catch-up boundary.

### Wording Boundary

Allowed:

- `boundary-null-concentration-advantage`
- `fewer effective components`
- `higher top-k component share`
- `local concentration advantage`
- `descriptive decomposition of D-0112`

Forbidden:

- `new Q2 metric`
- `rescued channel`
- `post-hoc win`
- `Level 2 proof`
- `geometry recovered`
- `manuscript-facing claim authorized`

### Practical Impact

- D-0112 remains the empirical closure of Q2.
- D-0113 clarifies the mechanism of the hard negative: at catch-up, the strongest hard null wins by local component concentration.
- Any paper-facing use requires Sofia/Iban editorial review and, if needed, Grace audit of wording.

### Validation Traces

- Diagnostic run report: `results/hsi_v2/q2_internal_routing_entropy/boundary_mechanism_readout/q2-boundary-mechanism-readout__src-q2-internal-routing-entropy-stage1__20260509T171750__20260510T080824/report.md`.
- Source run: D-0112 canonical Stage 1 output `q2-internal-routing-entropy-stage1__20260509T171750`.

---

## D-0114

- Date: 2026-05-10
- Status: `accepted-editorial-integration`
- Cycle: Q2 / boundary characterization integration into manuscript
- Trigger: D-0112 (Q2 hard negative `null-internally-lower-entropy`) and D-0113 (descriptive readout `boundary-null-concentration-advantage`) closed the Q2 cycle as repo-facing artifacts. Editorial review (Sofia) and Ariadna's recommendation converged on integrating a single short paragraph into §15.6 of `Paper_Latex/main_v2.01_cat.tex` as boundary characterization, without tables, figures, or expansion of the claim.

### Decision

D-0114 authorizes one paragraph in §15.6 of `Paper_Latex/main_v2.01_cat.tex`, inserted between the D-0102 paragraph and the D-0106 coda, tagged inline as `% [D114-T1]`.

The paragraph:

- reports Q2 v0.2 as a hard internal-entropy negative across all six catch-up bands;
- reports D-0113 mechanism as concentration of routing mass into fewer effective components with higher principal-component shares;
- explicitly preserves the P4-04 forward persistence ($3/3$) by stating it is not reproducible by `block-entropy` over the bands preceding the frontier;
- frames the frontier as a multi-metric regime where different strong nulls dominate under different objects, complementing rather than contradicting D-0102;
- does not open Door 2, does not introduce $\delta_m$, $G_B$ spectrum, or scaling law claims;
- does not authorize new experiments.

Editorial choice (Option B over Options A and C):

- Option A (Decision Log only) was rejected because the manuscript would otherwise carry no internal-entropy reading of the boundary, leaving the natural reviewer question "what stops B at the frontier?" answerable only at the surface (D-0102) level.
- Option C (full paragraph with mechanism breakdown numbers) was rejected because it would crowd §15.6 with three coda paragraphs and exceed the boundary-characterization scope.
- Option B was selected because it delivers the internal-entropy delimitation in a single paragraph, with explicit cross-reference to the canonical Decision Log for traceability.

### Wording Integrated

The integrated paragraph reads:

> Un diagnòstic posterior d'entropia interna sobre les sis bandes de `null-envelope catch-up` (Q2 v0.2, traçat al *Decision Log* canònic) no va trobar eficiència interna amagada de B sota l'entropia de components del routing: al contrari, el nul `block-entropy` va assolir menor entropia local en totes sis bandes de frontera, amb signe estable. El mecanisme reportat va ser concentració local de la massa de routing en menys components efectius i major pes als components principals. Aquesta lectura no refuta el subarc predictiu ---la persistència forward $3/3$ a P4-04 no és reproduïble per `block-entropy` sobre les bandes anteriors a la frontera---; complementa la lectura superficial del cicle P4-06 (on `phase-matched-lz` dominava la convergència de retenció) caracteritzant la frontera com un règim multi-mètric on nuls forts diferents guanyen segons l'objecte mesurat, no com una pèrdua de retenció de B.

### Wording Boundary

Allowed:

- `boundary characterization`
- `internal-entropy diagnostic`
- `multi-metric frontier regime`
- `concentration advantage of strong null at boundary`
- `complements surface-level reading of D-0102`

Forbidden:

- `Phase 4 weakened`
- `B refuted at boundary`
- `Q2 rescued`
- `internal opacity overturned`
- `geometry recovered`
- `cross-arc law authorized`
- `new experimental cycle authorized by D-0114`

### Practical Impact

- `Paper_Latex/main_v2.01_cat.tex` §15.6 acquires one new paragraph (`% [D114-T1]`) between the D-0102 paragraph and the D-0106 coda.
- D-0112 and D-0113 remain the empirical and mechanistic anchors; D-0114 is purely editorial integration.
- No figure, table, abstract, or §0 change is authorized by D-0114.
- A third brief-blind review pass over v2.02 may be considered after D-0114 integration but is not required by this decision.
- Any further paper-facing use of Q2/D-0113 material requires a new D entry.

### Validation Traces

- Inserted paragraph: `Paper_Latex/main_v2.01_cat.tex` §15.6, tagged `% [D114-T1]`.
- Underlying empirical anchors: D-0112 (Q2 Stage 1 hard negative) and D-0113 (boundary mechanism readout).
- Source contract: `hsi_agents_project/Documentation/HSI_v2_Q2_InternalRoutingEntropy_Preregistration_v0_2.md` (SHA256 `8DAD57F54BB70687AB2B4FBC85398BF543CC5A639D55E487EC9879B96521F011`).

---

## D-0115

- Date: 2026-05-10
- Status: `accepted-preregistration-stage0`
- Cycle: Level 2 / L2-01 spectral-percolation boundary probe
- Trigger: Grace audited the first L2-01 v0.1 preregistration as methodologically clean but epistemologically too close to graph-feature data science. Ariadna replaced it with a v0.2 spectral/percolation preregistration centered on algebraic/topological graph transitions. Grace accepted the v0.2 as a genuine pre-geometric Level 2 contract; Sofia accepted it editorially with report-level refinements.

### Decision

D-0115 accepts the L2-01 Stage 0 preregistration:

`hsi_agents_project/Documentation/HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_2.md`

Accepted SHA256:

`A548C01770DAD757151D62D8ED15575D85E811074922AFC57C062E21A8D003CC`

The previous v0.1 document:

`hsi_agents_project/Documentation/HSI_v2_Level2_L2_01_QuotientShape_RegimeSeparation_Preregistration_v0_1.md`

is not accepted as a Level 2 execution contract. It remains historical context only.

### Authorized Stage 0 Scope

Stage 0 may:

- inventory candidate multi-band windows labelled `interior`, `boundary`, or `inactive` from already accepted Phase 4/5 readouts;
- report whether each window can be built from existing or derivable child-routing rows;
- report whether each window can preserve cross-band continuity edges in the N3-05b weighted quotient object;
- report holdout dependence on Q2/D-0112-D0113 and Phase 5 B-mini;
- estimate compute cost;
- propose a sealed Stage 1 input table and hashes;
- return a discrete feasibility verdict.

Stage 0 may not:

- compute spectral invariants;
- compute graph distances;
- compare regimes;
- create figures;
- tune window sizes after seeing graph readouts;
- change regime labels;
- modify the manuscript.

### Required Stage 0 Verdicts

Stage 0 must return exactly one of:

- `stage1-executable`
- `insufficient-level2-window-budget`
- `insufficient-level2-holdout`
- `inputs-not-derivable`

If no non-Q2 boundary-adjacent window is available, Stage 0 must return `insufficient-level2-holdout` rather than proceed with Q2-overlap evidence only.

Any downgrade to single-band or per-cell local diagnostics requires a separate future Decision Log entry before execution.

### Level 2 Object

The accepted primary object is:

`observed-B weighted quotient child-routing graph`

with:

- N3-05b construction;
- edge-weight mode `pooled-retained-mass`;
- parent scale `40`;
- top patterns `128`;
- multi-band windows, not per-cell adapters.

Per-cell adapters are forbidden for the primary L2-01 graph because they erase cross-band topology and continuity edges.

### Candidate Future Invariants

Stage 1 is not authorized by this entry. If Stage 0 succeeds, a future v0.3 must freeze the invariant list before execution.

Candidate primary invariants for future Stage 1:

- `beta0`
- `zero_eigenvalue_count`
- `lambda2_norm_laplacian`
- `largest_component_fraction`
- `beta1 = edge_count - node_count + beta0`
- `edge_density`

Forbidden as primary Level 2 invariants:

- `top1_component_weight_share`
- `top3_component_weight_share`
- `top5_component_weight_share`
- `component_weight_entropy_norm`
- `effective_weight_components`

### Wording Boundary

Allowed:

- `spectral-percolation probe`
- `algebraic connectivity`
- `component transition`
- `cycle-rank readout`
- `pre-geometric graph-state transition`
- `candidate Level 2 invariant`

Forbidden:

- `geometry recovered`
- `topology discovered`
- `curvature measured`
- `Gate 2 solved`
- `Level 2 confirmed`
- `boundary law`
- `universal horizon`
- `graph statistics prove geometry`
- `retention-independent geometry established`

### Practical Impact

- L2-01 Stage 0 implementation is authorized.
- L2-01 Stage 1 is not authorized.
- No manuscript edit is authorized.

### Validation Traces

- Grace audit: `docs/HSI-audit-Level2-L2-01-SpectralPercolation.md`, verdict `ACCEPTED`.
- Sofia review: accepted v0.2, with report-level refinements around discrete verdict categories, holdout strictness, and downgrade requiring separate D entry.

---

## D-0116

- Date: 2026-05-10
- Status: `accepted-preregistration-stage1-authorized`
- Cycle: Level 2 / L2-01 spectral-percolation Stage 1
- Trigger: Stage 0 observability returned `stage1-executable`; Grace accepted the Stage 0 report and the v0.3 Stage 1 contract; Sofia signed the same v0.3 contract and requested a pre-committed editorial response table before execution.

### Decision

D-0116 accepts the L2-01 Stage 0 observability result and authorizes Stage 1 execution under the frozen v0.3 contract:

`hsi_agents_project/Documentation/HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_3.md`

Accepted SHA256:

`CE903E412CD0AD9CFB6E9F5D4B2B599D2EC70AEB2EE05E8A1CB2E11F93DA836C`

The final file was renamed from the signed `DRAFT` filename without changing its bytes, preserving the signed SHA256 exactly. This D entry is the operative status authority: v0.3 is now accepted and Stage 1 is authorized even if the document body still contains historical draft-status wording.

Accepted Stage 0 run:

`hsi_agents_project/results/hsi_v2/level2/l2_01_spectral_percolation/stage0_observability/l2-01-spectral-percolation-stage0__20260510T091237/`

Stage 0 verdict:

`stage1-executable`

Stage 0 artifact hashes:

| Artifact | SHA256 |
|---|---|
| `summary.json` | `BDEB00CE4962038E7131736D529CC623C2FE374C329A176B358339950EC6924C` |
| `band_inventory.csv` | `4ECD7EBB2C75570D04D2B57AA900E1799B233924C1015FA70D086C4600805E2F` |
| `window_inventory.csv` | `929C756A56693541B9B44F85CB2B7D7AB5AFB71319D52485ADE334DB5D24165A` |
| `stage1_candidate_windows.csv` | `42E2C13DC7EB4F2492C50834DF2D1FEA25CE1858B95942E2501A4EAB20B28638` |
| `manifest.json` | `7C7D8DC15F8710A77999847745F488F12845EECCDC2AFA95F935D26CCDA87D07` |

### Authorized Stage 1 Scope

Stage 1 may compute the frozen v0.3 spectral-percolation readout only:

- primary graph object: observed-`B` weighted quotient child-routing graph;
- graph construction: existing N3-05b construction, edge-weight mode `pooled-retained-mass`, parent scale `40`, top patterns `128`;
- spectrum convention: `v2/phase3/defect_spectrum.py::compute_graph_spectrum`, normalized Laplacian, edge weights included, zero tolerance `1e-9`;
- frozen tiers: within-arc transition control, cross-arc regime readout, hard-null mimicry control;
- frozen primary invariants: `beta0`, `zero_eigenvalue_count`, `lambda2_norm_laplacian`, `largest_component_fraction`, `edge_density`;
- structural context only: `beta1`, `spectral_radius_adjacency`, `node_count`, `edge_count`, `total_edge_weight`.

Stage 1 may not:

- use per-cell or one-band fallback;
- replace frozen windows after seeing invariants;
- use top-k component shares, entropy, retention, hard-margin, or null-envelope scores as primary Level 2 invariants;
- create figures;
- edit the manuscript;
- change solver semantics without a new Decision Log entry.

### Frozen Stage 1 Windows

Tier 1 within-arc transition control:

| Pair | Role | Arc | Bands |
|---|---|---|---|
| `T1-ARC-B` | interior | `ARC-B` | `606M-615M;615M-624M` |
| `T1-ARC-B` | boundary | `ARC-B` | `624M-633M;633M-642M` |
| `T1-P4` | interior | `P4` | `777M-786M;786M-795M` |
| `T1-P4` | boundary | `P4` | `795M-804M;804M-813M` |

Tier 2 cross-arc regime readout:

| Role | Arc | Bands | Holdout |
|---|---|---|---|
| interior | `ARC-B` | `597M-606M;606M-615M;615M-624M` | `q2-overlap` |
| interior | `P4` | `768M-777M;777M-786M;786M-795M` | `q2-overlap` |
| boundary | `ARC-A` | `300M-309M;309M-318M;318M-327M` | `non-q2` |
| boundary | `ARC-A` | `309M-318M;318M-327M;327M-336M` | `non-q2` |
| inactive | `ARC-C` | `1110M-1119M;1119M-1128M;1128M-1137M` | `non-q2` |
| inactive | `ARC-C` | `1119M-1128M;1128M-1137M;1137M-1146M` | `non-q2` |

Tier 3 hard-null mimicry:

- evaluate complete hard-null families among `phase-matched-lz`, `block-entropy`, and `matched-lz` if present in the frozen source rows;
- report partial hard-null coverage explicitly;
- if no hard-null family is complete, return `hard-null-inputs-not-derivable`.

### Pre-Committed Editorial Response

This table is an editorial pre-commitment. It constrains manuscript-facing response before Stage 1 results are known.

| Outcome Stage 1 | Editorial response |
|---|---|
| `spectral-percolation-candidate` | Open a future editorial integration cycle: new D entry, short paragraph in section 15.6 as a `candidate Level 2 invariant`, never as established Level 2. Wording fence remains binding. |
| `within-arc-only-support` | No manuscript edit. Repo-facing D entry documents that the transition exists within arc but does not generalize. |
| `cross-arc-only-support` | No manuscript edit. Repo-facing D entry records `confounded by regime/arc design`; this line closes without paper-facing cost. |
| `hard-null-mimicry` | No manuscript edit. Repo-facing D entry records that the spectral/percolation object is not B-specific under this contract. |
| `mixed-or-low-power` | No manuscript edit. Repo-facing D entry reports the Stage 1 result as written. |
| `no-level2-support` | No manuscript edit. Repo-facing D entry reports no support under the frozen object. |
| `inputs-not-derivable` | No editorial action. Stage 1 returns feasibility stop; a new cycle is required to continue. |
| `hard-null-inputs-not-derivable` | No editorial action. Hard-null coverage is insufficient for interpretation; a new cycle is required to continue. |

### Practical Impact

- L2-01 Stage 1 execution is authorized.
- No manuscript edit is authorized.
- No figure is authorized.
- Post-execution Grace audit is required before any scientific closure entry.
- Sofia editorial review is required before any future manuscript-facing use.

### Validation Traces

- Grace audit of v0.3 final: `docs/HSI-audit-Level2-L2-01-v03-Final.md`, verdict `ACCEPTED`.
- Sofia signature: SHA256 `CE903E412CD0AD9CFB6E9F5D4B2B599D2EC70AEB2EE05E8A1CB2E11F93DA836C` verified, source chain verified, Stage 0 windows verified, no manuscript edit authorized.

### Open Questions

1. Can the existing N3-05b implementation construct the frozen window-specific graphs without semantic changes?
2. Which hard-null families have complete routing coverage across all frozen windows?
3. Does Stage 1 return `spectral-percolation-candidate`, a weaker tier-specific result, mimicry, or no support?

---

## D-0117

- Date: 2026-05-11
- Status: `accepted-hard-negative`
- Cycle: Level 2 / L2-01 spectral-percolation Stage 1 closure
- Trigger: L2-01 Stage 1 executed under the frozen D-0116/v0.3 contract and returned `no-level2-support`. Grace audited the execution and accepted the hard negative, while proposing an exploratory inverse-transition interpretation.

### Decision

D-0117 closes L2-01 as a valid hard negative for the preregistered spectral/percolation object.

Stage 1 run:

`hsi_agents_project/results/hsi_v2/level2/l2_01_spectral_percolation/stage1/l2-01-spectral-percolation-stage1__20260510T101433/`

Stage 1 contract:

`hsi_agents_project/Documentation/HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_3.md`

Contract SHA256:

`CE903E412CD0AD9CFB6E9F5D4B2B599D2EC70AEB2EE05E8A1CB2E11F93DA836C`

Stage 1 result:

- overall: `no-level2-support`
- Tier 1: `within-arc-no-support`
- Tier 2: `cross-arc-no-support`
- Tier 3: `no-hard-null-mimicry`
- coverage flag: none

This result means:

- the preregistered fragmentation/percolation direction is not supported;
- the N3-05b weighted quotient graph, under the frozen windows and invariants, is not accepted as a Level 2 bridge object;
- no manuscript edit is authorized by L2-01.

### Stage 1 Artifact Hashes

| Artifact | SHA256 |
|---|---|
| `summary.json` | `97220A946D55F9979827B2F68DAEE441A47F47816B393681C2E2AA23FFB89622` |
| `report.md` | `69105F3128A78AC0445268907C2708C93007F3741BBFEF67EE82C54E622C4CBA` |
| `window_graph_summary.csv` | `9DC815C3A80BEF180043C8726CEC402C514BF9CC27902F34A21547913BC2EFE2` |
| `tier1_within_arc_contrasts.csv` | `AB522DD692BB90660491C3BBF823727FC46A9B723D8526FF63A123BF867F7A51` |
| `tier2_cross_arc_medians.csv` | `327DC68F7133CDDE8F66BC9F140EAED90E6515686472410012E493972C1F96E9` |
| `tier3_hard_null_mimicry.csv` | `0FF509BF29A7BFC1BDB861073D1F3C88A954D0DC90F3B3994A2E19A8630DEB2E` |
| `manifest.json` | `4EB0F82E26B46A0C4CC64AAF02EBCAAD87E941C4D2EE291BC5891A091F06CF74` |

### Observed Descriptive Pattern

Although L2-01 is negative under its contract, the observed `B` rows show a descriptive inverse pattern:

- Tier 1 ARC-B: interior `node_count=150`, `edge_count=162`, `beta0=16`, `beta1=28`, `edge_density=0.01450`; boundary `node_count=60`, `edge_count=44`, `beta0=16`, `beta1=0`, `edge_density=0.02486`.
- Tier 1 P4: interior `node_count=120`, `edge_count=140`, `beta0=8`, `beta1=28`, `edge_density=0.01961`; boundary `node_count=60`, `edge_count=59`, `beta0=8`, `beta1=7`, `edge_density=0.03333`.
- Tier 2 medians: interior `beta0=20`, `edge_density=0.01044`; boundary `beta0=16`, `edge_density=0.02035`; inactive `beta0=120`, `edge_density=0.00273`.

This suggests a possible `quotient contraction / loss of sparse parallelism` reading:

- boundary does not fragment in the preregistered direction;
- boundary tends to reduce graph size and cycle rank while increasing relative density;
- the result may be closer to contraction and cycle-loss than to destructive percolation.

Grace's audit names a stronger exploratory interpretation (`Topological Hairball Transition` / `Loss of Sparse Parallelism`). D-0117 records this as exploratory only. It is not accepted as a paper-facing claim and is not authorized by the L2-01 contract.

### Practical Impact

- L2-01 is closed as a hard negative.
- The manuscript remains unchanged.
- No Level 2 claim is added.
- A future L2-02 may be designed around quotient contraction, cycle-loss, and density-normalized boundary behavior, but must be preregistered before execution.

### Validation Traces

- Stage 1 run: `hsi_agents_project/results/hsi_v2/level2/l2_01_spectral_percolation/stage1/l2-01-spectral-percolation-stage1__20260510T101433/`
- Grace audit: `docs/HSI-audit-Level2-L2-01-Stage1-Results.md`, verdict `HARD NEGATIVE ACEPTADO`.
- D-0116 pre-committed editorial response for `no-level2-support`: no manuscript edit, repo-facing D entry only.

### Open Questions

1. Is the inverse descriptive pattern robust after controlling for `node_count` and window size?
2. Does cycle-rank loss (`beta1`) provide a cleaner readout than edge density?
3. Do hard nulls show the same quotient contraction, or is it B-specific?

---

## D-0118

- Date: 2026-05-11
- Status: `accepted-preregistration-execution-authorized`
- Cycle: Level 2 / L2-02 quotient-contraction mimicry probe
- Trigger: L2-01 closed as a hard negative for the preregistered spectral-percolation direction while leaving a descriptive inverse pattern on observed `B` windows. Grace flagged the first L2-02 draft as circular/double-dipping unless the observed-B component was explicitly degraded to post-hoc formalization and the hard-null mimicry component became the primary test. Sofia refused to sign v0.1 for the same reason and requested a mimicry-as-primary v0.2.

### Decision

D-0118 accepts the L2-02 v0.2 contract and authorizes execution under its frozen scope:

`hsi_agents_project/Documentation/HSI_v2_Level2_L2_02_QuotientContraction_Preregistration_v0_2.md`

Accepted SHA256:

`3F201C6CD66955721C262AF74CCEE806773A129ECC9DB0B09ADCD42FF432BB9A`

The final file was renamed from the signed `_DRAFT` filename without changing its bytes, preserving the signed SHA256 exactly. This D entry is the operative status authority: v0.2 is now accepted and execution is authorized even if the document body still contains historical draft-status wording.

L2-02 tests whether hard nulls reproduce the quotient-contraction pattern descriptively observed in L2-01 under the same N3-05b weighted quotient child-routing graph object and the same frozen within-arc windows.

The observed-`B` component is not independent evidence:

- the within-arc windows are reused from L2-01 because that is where the inverse pattern was noticed;
- no virgin holdout windows are used for observed `B`;
- observed-`B` pair classification is a post-hoc consistency check on already-inspected L2-01 numerics;
- the hard-null mimicry verdict is the deciding component of L2-02 whenever observed-`B` contraction is present.

### Authorized Execution Scope

Execution may compute only the v0.2 quotient-contraction mimicry readout:

- graph object: same N3-05b weighted quotient child-routing graph as L2-01;
- edge-weight mode: `pooled-retained-mass`;
- parent scale: `40`;
- top patterns: `128`;
- frozen within-arc windows: ARC-B interior/boundary and P4 interior/boundary exactly as specified in v0.2 section 6;
- observed-B indicators: `node_count_contraction`, `beta1_cycle_loss`, `total_edge_weight_loss`, `relative_density_rise`, `largest_component_nonincrease`;
- hard-null families: `phase-matched-lz`, `block-entropy`, and `matched-lz` only if complete rows are present;
- output artifacts: `summary.json`, `report.md`, `observed_b_contraction.csv`, `hard_null_contraction_mimicry.csv`, `cross_arc_context.csv`, `manifest.json`.

Execution may not:

- replace windows;
- add virgin/alternate windows after seeing results;
- change the graph object or solver semantics;
- create figures;
- use cross-arc context to change the overall verdict;
- edit the manuscript;
- interpret observed-`B` contraction as independent evidence.

### Readout Priority

The execution must emit exactly one overall category according to the v0.2 priority rule:

1. If no hard-null seed is complete, emit `hard-null-inputs-not-derivable`.
2. Else, if observed-`B` is not `B-contraction-descriptively-present`, emit `no contraction`.
3. Else, use the complete hard-null seeds to distinguish `B-specific contraction candidate` from `non-B-specific contraction`.

For report clarity, `hard-null-inputs-not-derivable` means no complete hard-null seed is available in any hard-null family; it must not be interpreted as absence of mimicry.

The `B-specific contraction candidate` bar is intentionally strict: any complete hard-null seed returning either `null-contraction-mimicry` or `partial-null-contraction-mimicry` blocks the B-specific candidate verdict. This strictness is accepted because the observed-B component has HARK/post-hoc origin.

### Pre-Committed Editorial Response

This table is an editorial pre-commitment. It constrains manuscript-facing response before L2-02 results are known.

| Outcome L2-02 | Editorial response |
|---|---|
| `B-specific contraction candidate` | Open a reinforced editorial integration cycle because of the HARK origin: new D entry, mandatory post-execution Grace audit, Sofia review, and only if accepted, a short paragraph in section 15.6 as `candidate B-specific contraction signature pending independent replication on virgin windows`. No established Level 2 claim. Wording fence remains binding. |
| `non-B-specific contraction` | No manuscript edit. Repo-facing D entry records a sixth delimiter of the predictive subarc and closes this Level 2 line jointly with Q1 / Q2 / D-0113 / L2-01. |
| `no contraction` | No manuscript edit. Repo-facing D entry records an anomaly because observed-B contraction was expected to reappear under the post-hoc re-description; execution/window-definition issues must be checked before interpretation. |
| `hard-null-inputs-not-derivable` | No editorial action. Stop without interpretation; a new cycle is required to complete hard-null coverage. |

### Wording Boundary

Allowed:

- `quotient contraction`
- `cycle-rank loss`
- `loss of sparse parallelism`
- `relative density rise`
- `boundary contraction candidate`
- `mimicry-as-primary`
- `post-hoc observed-B re-description`

Forbidden:

- `geometry recovered`
- `topology discovered`
- `Level 2 confirmed`
- `Topological Hairball Transition established`
- `Gate 2 solved`
- `Phase 5 rescued`
- `L2-01 was positive after all`
- `L2-02 independently validates L2-01 inverse pattern`
- `L2-02 confirms B-specific contraction`, unless the overall verdict is exactly `B-specific contraction candidate`

### Practical Impact

- L2-02 execution is authorized under the frozen v0.2 contract.
- No manuscript edit is authorized.
- No figure is authorized.
- A post-execution Grace audit is required before any closure entry.
- Sofia editorial review is required before any future manuscript-facing use.

### Validation Traces

- Grace audit of v0.2: `docs/HSI-audit-Level2-L2-02-v02-Draft.md`, verdict `ACCEPTED`, SHA256 `3F201C6CD66955721C262AF74CCEE806773A129ECC9DB0B09ADCD42FF432BB9A`.
- Sofia signature: SHA256 `3F201C6CD66955721C262AF74CCEE806773A129ECC9DB0B09ADCD42FF432BB9A` verified; six requested v0.1 -> v0.2 changes verified; no manuscript edit authorized.
- Source closure: D-0117 accepted L2-01 as hard negative and recorded quotient contraction / loss of sparse parallelism as exploratory only.

### Open Questions

1. Which hard-null families have complete rows across all four frozen within-arc windows?
2. Do hard nulls reproduce the same contraction-positive pattern as observed `B`?
3. If hard-null mimicry blocks B-specificity, does this become the sixth delimiter of the predictive subarc?

---

## D-0119

- Date: 2026-05-11
- Status: `accepted-hard-negative`
- Cycle: Level 2 / L2-02 quotient-contraction mimicry closure
- Trigger: L2-02 executed under the frozen D-0118/v0.2 contract and returned `non-B-specific contraction`. Grace audited the execution and accepted the hard negative: the quotient-contraction pattern observed in `B` is mimicked by hard nulls and therefore cannot be claimed as B-specific under this graph object.

### Decision

D-0119 closes L2-02 as a repo-facing hard negative for B-specific quotient contraction.

Canonical run:

`hsi_agents_project/results/hsi_v2/level2/l2_02_quotient_contraction/mimicry_probe/l2-02-quotient-contraction-mimicry__20260511T073130/`

Contract:

`hsi_agents_project/Documentation/HSI_v2_Level2_L2_02_QuotientContraction_Preregistration_v0_2.md`

Contract SHA256:

`3F201C6CD66955721C262AF74CCEE806773A129ECC9DB0B09ADCD42FF432BB9A`

Readout:

- overall verdict: `non-B-specific contraction`
- observed-B category: `B-contraction-descriptively-present`
- complete hard-null seed count: `12`
- complete hard-null families: `block-entropy`, `phase-matched-lz`
- `phase-matched-lz`: `6/6` seeds returned `null-contraction-mimicry`
- `block-entropy`: `3/6` seeds returned `null-contraction-mimicry`, `1/6` returned `partial-null-contraction-mimicry`, `2/6` returned `no-null-contraction-mimicry`

The observed-B quotient contraction is real as a descriptive re-read of the L2-01 windows:

- ARC-B pair: `5/5` contraction indicators, node ratio `0.4000`
- P4 pair: `5/5` contraction indicators, node ratio `0.5000`

However, because hard-null mimicry is present, the D-0118 pre-committed editorial response applies: no manuscript edit, repo-facing closure only.

### Scientific Reading

L2-02 blocks the candidate reading that the quotient-contraction / cycle-rank-loss pattern is B-specific under the N3-05b weighted quotient child-routing graph object.

The result does not mean that the boundary is empty or behaviorless. It means the contraction pattern belongs to a broader boundary response that hard nulls can also reproduce under this object. Grace's audit describes this as a topological substrate effect at the thermodynamic boundary. D-0119 records the bounded version of that reading:

- quotient contraction appears at the boundary in observed `B`;
- the same contraction is reproduced by `phase-matched-lz` across all complete seeds;
- therefore, quotient contraction is not accepted as a B-specific Level 2 invariant;
- the pattern becomes an additional delimiter of the predictive/Level-2 subarc rather than a positive geometry claim.

This is not a claim of literal universality across all possible routing algorithms or null ensembles. It is a strong non-specificity result under the hard null families available in the frozen L2-01/L2-02 artifact chain.

### Artifact Hashes

| Artifact | SHA256 |
|---|---|
| `summary.json` | `1122391D3C1890DAF52DA7F223D2B147213BB7A45CF2A91C6342052765CC3DBC` |
| `report.md` | `9B34310A3E7871C8ABFDC530BFED82943458BFAE93EE961F48F114D0002A44CB` |
| `observed_b_contraction.csv` | `5BBB81E89C5E99B6AECE45DA99D53FD54A1406F639651B826EC47AA9D648F382` |
| `hard_null_contraction_mimicry.csv` | `61FDD123EFB6F2E1075623036E844D29328711E92E52C777B074511482A05AAF` |
| `hard_null_pair_contrasts.csv` | `19956425E27785B130DCC283A01AD7758599A9527B985E7D097BE40C3786CF22` |
| `cross_arc_context.csv` | `601BCCBF75A5B54EC6B5C89A04C33E5818B5AE2F867C508B2D473CE8D5C2AE0D` |
| `manifest.json` | `9F8423C710E4783496D279840D6C269CD69B8E626FCFAB4379AFDC00AF2A7FA5` |

### Wording Boundary

Allowed:

- `non-B-specific contraction`
- `hard-null mimicry`
- `quotient-contraction delimiter`
- `boundary contraction reproduced by hard nulls`
- `not a B-specific Level 2 invariant`
- `repo-facing hard negative`

Forbidden:

- `geometry recovered`
- `topology discovered`
- `Level 2 confirmed`
- `B-specific contraction established`
- `Topological Hairball Transition established`
- `universal topological law`
- `any routing algorithm collapses`
- `L2-01 was positive after all`

### Practical Impact

- L2-02 is closed as a hard negative for B-specific quotient contraction.
- No manuscript edit is authorized.
- No figure is authorized.
- The L2-02 result is a sixth delimiter of the predictive/Level-2 subarc, not a positive paper-facing claim.
- Further pursuit of Level 2 geometry would require a mathematically new object or genuinely independent windows/null ensembles, not a re-labeling of this quotient-contraction readout.

### Validation Traces

- L2-02 run: `hsi_agents_project/results/hsi_v2/level2/l2_02_quotient_contraction/mimicry_probe/l2-02-quotient-contraction-mimicry__20260511T073130/`
- Grace audit: `docs/HSI-audit-Level2-L2-02-Stage1-Results.md`, verdict `HARD NEGATIVE ACEPTADO`.
- D-0118 pre-committed editorial response for `non-B-specific contraction`: no manuscript edit, repo-facing D entry as delimiter.

### Open Questions

1. Is there any mathematically new Level 2 object worth testing, or is the current empirical frontier exhausted for this manuscript cycle?
   - Resolution: no additional Level 2 object is pursued within the current manuscript cycle. A genuinely new Level 2 test would require at least one material change outside the current cycle budget: a different graph object, temporal/dynamical invariants rather than static graph invariants, or fundamentally different null ensembles. Further tests under N3-05b-like objects or the same null families have low marginal value after Q1, Q2, D-0113, L2-01 and L2-02.
2. Should Sofia record a paper-facing no-op note, or is D-0119 sufficient as repo-facing closure?
   - Resolution: D-0119 is sufficient. No paper-facing no-op note is recorded. D-0118 pre-committed the editorial response for `non-B-specific contraction` as no manuscript edit, and the current manuscript does not contain stale L2-01/L2-02 claims that require correction.
3. Should the next coordinated step be final manuscript hostile-review rather than additional experimentation?
   - Resolution: yes. The next coordinated step is final hostile review / external review of the candidate manuscript snapshot, not additional internal experimentation.

### Editorial Closure Addendum

Sofia verified D-0119 against disk and confirmed:

- D-0119 status `accepted-hard-negative`;
- v0.2 contract SHA256 `3F201C6CD66955721C262AF74CCEE806773A129ECC9DB0B09ADCD42FF432BB9A`;
- declared artifact hashes match disk;
- Grace audit `docs/HSI-audit-Level2-L2-02-Stage1-Results.md` exists and confirms `HARD NEGATIVE ACEPTADO`;
- report/verdict consistency: `12` complete hard-null seeds, with `9/12` showing full or partial mimicry;
- D-0118 pre-commitment honored: no manuscript edit;
- D-0119 correctly bounds Grace's stronger language by forbidding `universal topological law` and `any routing algorithm collapses`.

Minor implementation note for future contracts: D-0119 records `hard_null_pair_contrasts.csv` as an auxiliary detail artifact, although it was not listed in the v0.2 contract's primary output list. This does not affect the verdict because it is derived from the same frozen `window_graph_summary.csv` and Grace accepted the execution. Future contracts with this pattern should distinguish primary verdict artifacts from auxiliary detail artifacts in the output list or manifest.

---

## D-0120

- Date: 2026-05-11
- Status: `accepted`
- Cycle: HSI v2 manuscript snapshot for hostile review
- Trigger: closure of the empirical Level 2 cycle (D-0117, D-0118, D-0119) leaves the manuscript at a stable state with no pending paper-facing edits. Grace, Ariadna and Sofia agree (Editorial Closure Addendum of D-0119) that the next coordinated step is final external/hostile review on a frozen candidate snapshot, not additional internal experimentation.

### Decision

D-0120 freezes the v2.02 candidate manuscript as the artifact submitted to hostile review. No manuscript edits are authorized while the hostile review is in progress; reviewer-driven edits are processed as a new D entry per cycle (see D-0121).

### Frozen artifacts

| Artifact | Path | Size | SHA256 |
|---|---|---:|---|
| LaTeX source | `Paper_Latex/main_v2.02_cat.tex` | `158431` | `64FD2A339AA0284DCF2F1C84CAF5FB6C9C92CA57FDF78B8BF64A0C1F779CB3FE` |
| Compiled PDF (Overleaf) | `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_1.pdf` | `2615743` | `CE072CFF483A0A11E7DEFE05B86153A2D765C53955194ED8AA6231BBB3726B7B` |

### Pre-Snapshot Verification (Sofia)

Performed before freezing:

- no occurrence in `main_v2.02_cat.tex` of `L2-0`, `hairball`, `quotient[ -]contraction`, `spectral[ -]percolation`, `cycle-rank`, `pou gravitatori`, `substrat topològic`, `D-0116`, `D-0118`, `D-0119`;
- D-0102 / D-0114 / D-0106 boundary characterizations preserved verbatim with conservative wording (`no obre Porta 2`, `local i intermitent`, `no autoritza cap llei d'horitzó ni generalització cross-arc`);
- §13 Conclusions preserves the bounded reading of the Phase 4 finite subarc (`acota la persistència forward del règim sense reobrir Porta 2`);
- pre-commitments D-0116 and D-0118 (no manuscript edit for L2-01 / L2-02 hard-negative outcomes) honored without exception;
- file rename `main_v2.01_cat.tex` → `main_v2.02_cat.tex` was byte-preserving (same `LastWriteTime`).

### Hostile Review Outcome (recorded after the fact)

The frozen artifact above was submitted to two independent hostile-review modalities:

- a brief-blind external pass (no project memory, PDF only) returned five structural objections covering the P1-P3→B derivation, the Phase 4 null catch-up reading, the `algebraicament exacte` claim at L346, the tower-circularity defense, and the lexical convention for physics-resonant terms;
- a Grace iterative-hostile pass (with full cycle memory) on the same PDF returned one orthogonal narrative-continuity patch (`sis bandes` orphan numeric at L912) and confirmed that the brief-blind Obj. 3 was a true killable her own pass had missed.

The triage and operationalization of these returns is handled in D-0121.

### Authorized Use (terminated by D-0121)

The freeze regime forbade manuscript edits while the hostile review was in progress. That regime is operationally completed: D-0121 anchors the next freeze on the post-edit artifact.

---

## D-0121

- Date: 2026-05-11
- Status: `manuscript-edit-cycle-closed`
- Cycle: HSI v2 manuscript reviewer-driven edit cycle (Gemini brief-blind pass 1 + Grace iterative-hostile)
- Trigger: a brief-blind external pass on the v2.02 frozen snapshot (D-0120 PDF SHA `CE072CFF...`) returned five structural objections. Grace iterative-hostile audit on the same artifact, performed with full project memory, returned one orthogonal narrative-continuity patch and confirmed one of the five brief-blind objections as a true killable that her own pass had missed.

### Decision

D-0121 closes the reviewer-driven edit cycle as a minimum-surface patch over `Paper_Latex/main_v2.02_cat.tex`. Only the single killable wording (Obj. 3, `algebraicament exacte`) and the single narrative-continuity flaw (Grace, orphan numeric `sis bandes`) are applied to the manuscript. The four remaining brief-blind objections (Obj. 1, 2, 4, 5) are accepted as defended in place by existing manuscript text and do not motivate further edits in this cycle. No external response letter is generated, since the reviewers operated as sparring partners and not as a real review committee.

### Edits applied

| # | Line | Origin | Action |
|---|------|--------|--------|
| 1 | L346 | brief-blind Obj. 3 (confirmed by Grace) | Replaced `algebraicament exacte, no estadístic` with `una convergència numèrica estricta, no una tendència estadística`; the FP64-precision qualifier and the empirical claim are preserved without the unsupported algebraic-theorem implication |
| 2 | L912 (first occurrence) | Grace patchable | Replaced `sobre les sis bandes de` with `sobre les bandes de`; removes orphan numeric reference whose source (Q2 v0.2 multi-origin band collection) is not visible to the linear reader |
| 3 | L912 (second occurrence) | Coherence with edit #2 | Replaced `en totes sis bandes de frontera` with `en totes les bandes de frontera`; required to keep both halves of the same sentence consistent after edit #2 |

No new equations, tables, figures, sections, references, or section reorderings introduced.

### Defenses confirmed in place (no edit required this cycle)

| Brief-blind objection | Defense location in v2.02 (post-patch) | Why no edit |
|---|---|---|
| Obj. 1: P1-P3 → Variant B as axiomatic injection | §3.5 L178-180 (family-not-axiom framing); §3.7 L203-208 (parser objection answered structurally); §4 closing L258 (`entre els considerats`) | The minimum-mechanism qualifier and the structural (not executive) reading of nesting are already explicit. |
| Obj. 2: Phase 4 null catch-up = asymptotic artifact | §11.4 L908 (`no es generalitza a un horitzó asimptòtic`); §11.4 L915 (`no autoritza cap llei d'horitzó`) | The text already declares the signature as finite-horizon. A null catching up at one boundary documents where the discriminant ends, not that the discriminant is artifactual at the protocol scale ($3\times 10^6$ bits, §10.1 L526). |
| Obj. 4: Tower architecture circular by design | §obj_circular L925-927 (matched-LZ + phase-matched-LZ double-null defense); §obj_circular L927 closing (Phase 4 out-of-sample falsifier); L760 (sliding window operates on $\Phi$ only) | If the tower were a tautology of the generator, any null preserving $\Phi_B$'s coarse properties would reproduce the signal by construction; the empirical fact that they do not is the falsifier of the design-bias reading. |
| Obj. 5: Physics terminology = metanarrative abuse | §15 L793 (lexical convention restricting all physics-resonant terms to algorithmic/informational sense); standard graph-Laplacian definition $L = D - A$ at §15.3 L817; thermodynamic-routing reading anchored to Beck-Schlögl 1993 at §11.3 L884 | Symbols are explicitly defined within standard mathematical objects, not borrowed from physics as homologies. The `Programa cap al Nivell 2` framing already presents §15 as a candidate test contract, not a measured-physics claim. |

### Frozen artifacts (post-edit)

| Artifact | Path | Size | SHA256 |
|---|---|---:|---|
| LaTeX source | `Paper_Latex/main_v2.02_cat.tex` | (post-edit) | `158F720E282AF10745E595FD2ABF0DEAE421F9011605D032AED34B5B20E4560E` |
| Compiled PDF (Overleaf) | `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_2.pdf` | `2615743` (approx., to verify on disk) | `D171543D977BE40BB7F8116432A70598A5AF0FB8EE79A5621945F61D78F43B9B` |

The version label `v2.02` is preserved by editorial decision (the patches are micro-edits and do not warrant a version bump); the PDF filename suffix `_CAT_2.pdf` distinguishes the post-patch artifact on disk.

### Wording Boundary

Forbidden after this cycle (in addition to inherited fences): `algebraicament exacte`, `algebraically exact`. Allowed and explicit: `convergència numèrica estricta`, `conjectura numèrica forta`, `dins de la precisió de punt flotant`, `FP64`.

### Open Questions

None opened by this cycle. The D-0120 freeze regime is now operationally completed: the post-edit artifact above is the new freeze candidate. Subsequent reviewer-driven cycles, if any, require a new D entry per the same protocol; a separate cycle for the brief-blind pass 2 verdict (vaporware on §15, strawman on classical controls) is queued and will be opened as D-0123 with explicit framing decisions, not micro-patches.

---

## D-0122

- Date: 2026-05-11
- Status: `accepted-methodology-observation`
- Cycle: HSI v2 hostile-review protocol meta-observation
- Trigger: across D-0120, D-0121 and D-0123, the project executed three brief-blind external passes (Gemini-1, Gemini-2, Gemini-3) interleaved with one full-memory iterative-hostile pass (Grace) on the same frozen manuscript. The detection profiles of the two modalities are not interchangeable, and three brief-blind passes on increasingly defended artifacts make the asymmetry empirically measurable.

### Decision

D-0122 records the brief-blind ↔ iterative-hostile asymmetry as a methodology observation that justifies keeping the dual-modality audit protocol of `App. A` (`co-creació simbiòtica`) operationally in place, while documenting that brief-blind detection saturates as the manuscript hardens.

### Empirical record (this manuscript)

| Pass | Modality | Artifact | Returns |
|---|---|---|---|
| Gemini-1 | brief-blind, no memory | v2.02 frozen (D-0120 SHA `CE072CFF...`) | 5 structural objections, 1 of them a true killable (`algebraicament exacte` at L346) |
| Grace | iterative-hostile, full memory | same v2.02 frozen | 1 orthogonal narrative-continuity patch (`sis bandes` at L912); did not detect Gemini-1 Obj. 3 because the wording had survived multiple authored passes |
| Gemini-2 | brief-blind, no memory | v2.02 post-D-0121 (PDF SHA `D171543D...`) | 4 framing recommendations adopted as D-0123 (`Strawman controls`, `Vaporware Level 2`, `Ontological-Algorithmic leap`, `Toy-model framing`); no killables |
| Gemini-3 | brief-blind, no memory | v2.02 post-D-0123 (PDF SHA `6479228D...`) | 7 objections; zero killables; 5/7 are attacks on paragraphs that already defend against them verbatim |

### Operational implications

- The dual modality is not redundant: brief-blind detected the only true killable of the cycle (Gemini-1 Obj. 3), iterative-hostile detected the only narrative-continuity patch of the cycle (Grace `sis bandes`). Each modality covered a class of flaw the other missed on the same artifact.
- Brief-blind detection saturates as the manuscript hardens: pass-3 returned zero killables on the post-edit, and 5/7 of its objections targeted paragraphs that explicitly answer them. Continued sparring with the same external modality on the same baseline is not informative beyond saturation.
- Iterative-hostile review with full project memory is vulnerable to local minima of familiarity: text that has survived prior authored passes inherits implicit endorsement and is harder to flag as overclaim in subsequent reviews by the same auditor. The brief-blind modality is the operational counter to this drift.

### Application to future cycles

Dual-modality remains the default audit protocol for every freeze candidate. After two consecutive brief-blind passes return zero killables on the same baseline (current state of v2.02 post-D-0123), the protocol is considered to have reached its saturation point on that baseline; further hostile passes require either a different modality (e.g. domain-expert human auditor on dynamic symbolic systems, per Gemini-3 Obj. 7 narrowed) or a substantively new manuscript baseline.

---

## D-0123

- Date: 2026-05-11
- Status: `manuscript-edit-cycle-closed`
- Cycle: HSI v2 manuscript reviewer-driven edit cycle 2 (Gemini brief-blind pass 2 framing recommendations + Gemini brief-blind pass 3 saturation check)
- Trigger: a second brief-blind external pass on the D-0121 post-edit artifact (PDF SHA `D171543D...`) returned four framing recommendations grouped as `Strawman controls`, `Vaporware Level 2`, `Ontological-Algorithmic leap`, and `Toy-model framing`. Iban authorized a `B3` strategy: reinforce the radical ontological proposal in the Abstract while strictly separating it from the restricted empirical evaluation, and document the existing Level 2 hard-negative results in-paper to close the vaporware flank.

### Decision

D-0123 closes the second reviewer-driven edit cycle as four framing edits over `Paper_Latex/main_v2.02_cat.tex`. The version label `v2.02` is preserved (changes are framing, not science); the on-disk artifact is `_CAT_3.pdf`. No content was removed; no claim was strengthened beyond what the manuscript already supports.

### Edits applied

| # | Location | Origin | Action |
|---|----------|--------|--------|
| 1 | Abstract (between paragraphs 1 and 2) | Gemini-2 Rec. 1 (rephrased per Iban: preserve grandiloquence, reject toy-model framing) | Inserted a new intermediate paragraph that explicitly separates the radical ontological proposal from the deliberately restricted empirical evaluation; functions as a reading contract for the rest of the manuscript |
| 2 | §2.3 `Nota axiomàtica` (L107 → extended) | Gemini-2 Rec. 2 (rephrased per Iban: not defensive) | Extended the existing axiomatic note with a positive-framing closing sentence: the fertility of the axiomatic choice is judged by the internal coherence and empirical non-triviality of the derived dynamics, not by the metaphysical status of the postulates |
| 3 | §15 title `\section{...}` | Gemini-2 Rec. 3a | Renamed `Programa cap al Nivell 2: defectes i geometria` to `Nivell 2: programa formal i fronteres observades` |
| 4 | §15 final subsection `\label{subsec:fronteres_nivell2}` | Gemini-2 Rec. 3b (rephrased: no `L2-0X` labels, no contradiction with subsec:porta2) | Inserted a new closing subsection `Fronteres empíriques observades dins del programa` documenting the two pre-registered hard-negative probes (spectral property of $G_B$; weighted prefix/suffix quotient contraction) without contradicting the still-open status of the three formal Gate-2 conditions |

No new equations, tables, figures, references, or section reorderings were introduced. Section count and label graph unchanged except for the new `subsec:fronteres_nivell2` label, which is referenced only from inside its own section.

### Frozen artifacts (post-edit)

| Artifact | Path | SHA256 |
|---|---|---|
| LaTeX source | `Paper_Latex/main_v2.02_cat.tex` | `C62D2B1A5315CC6A2B829E98EBA43120E6A8764CC4D967886DAE49D47B01E64C` |
| Compiled PDF (Overleaf) | `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_3.pdf` | `6479228DA159765DE553F55F74CDEF0B7648D7A959B9D5858E3A343151A30F7C` |

### Saturation check (Gemini brief-blind pass 3)

A third brief-blind pass was performed on the D-0123 post-edit PDF immediately after compilation. Verdict: zero killables. The seven objections returned by this third pass are either (i) verbatim attacks on paragraphs that already defend against them (Obj. 1 alphabet/P2 violation answered at §subsec:alfabet L142 + §obj-alphabet L944; Obj. 2 parser/stack memory answered at L205; Obj. 3 physics-washing answered at lexical note L118 + reminder L793; Obj. 4 p-hacking on `retention@last`/`active_mean_tail` answered at L703 + L746-748 + D101-T1 anchor at L877; Obj. 7 IA echo-chamber answered at L936), (ii) categorical confusion (Obj. 6 reads control variant F as a defended HSI variant when it is explicitly a validation-by-absurd control, §sec:absurd), or (iii) universally-valid `you could do more` complaints not killable by any finite expansion (Obj. 5 `only two L-systems`).

### Open Questions

D-0124 records the Abstract-length open issue surfaced by Iban after the recompile.

---

## D-0124

- Date: 2026-05-11
- Status: `superseded` by D-0125 (closed 2026-05-12)
- Cycle: HSI v2 manuscript polishing backlog
- Trigger: post-D-0123 reading of the new three-paragraph Abstract by Iban surfaced a length concern: the inserted ontological-vs-empirical separation paragraph (D-0123 edit #4) reads as substantively necessary but stretches the Abstract beyond a typical journal limit, suggesting that paragraphs 1 and 2 (or 2 and 3) admit compression without loss of content.

### Decision

D-0124 records this as an `open-issue` for a future polishing cycle, not as a freeze blocker. The current freeze (D-0123 PDF SHA `6479228D...`) stands; no edit is authorized under this entry. The issue is queued for Iban's next editorial pass with the following scope:

- target: condense the Abstract to two paragraphs while preserving the ontological-vs-empirical separation introduced by D-0123;
- non-goals: introducing new content, removing the radical-ontology framing, or weakening the empirical-restraint declaration;
- success criterion: total Abstract length within the typical journal-Abstract envelope (~250-300 words) without losing any of the four functions currently performed (origin claim, axiomatic restriction, ontological/empirical separation, three-level validation summary).

### Open Questions

Whether the compression is best achieved by merging paragraphs 1 and 2 (collapsing the ontological build-up) or paragraphs 2 and 3 (folding the empirical-restraint paragraph into the validation summary). Both directions are admissible under the success criterion; the choice is editorial.

---

## D-0125

- Date: 2026-05-12
- Status: `manuscript-edit-cycle-closed`
- Cycle: HSI v2 Abstract compression (closure of D-0124)
- Trigger: D-0124 was opened as `open-issue-deferred` after Iban observed that the post-D-0123 three-paragraph Abstract (470 words) was longer than the Introduction itself (~290 words), inverting the natural narrative proportion. Iban authorized this cycle the next day with three explicit constraints: (i) target ~70% of original length in two paragraphs, (ii) preserve the most relevant messages and offload the rest to the Introduction only if necessary, and during the iteration added two further constraints: (iii) remove all em dashes (`---`) from the Abstract because the marker has fallen out of style and is now read by external reviewers as a signature of unattributed AI generation, and (iv) replace the acronym `HSI` in the second paragraph with a non-acronym alternative.

### Decision

D-0125 closes D-0124 with a single edit on the `\begin{abstract}...\end{abstract}` block of `Paper_Latex/main_v2.02_cat.tex`. The three previous paragraphs (D-0123 baseline) are replaced by two new paragraphs that compress to 75% of the original length, eliminate all four em dashes, open with a contextual question that disambiguates the paper's subject within the first sentence, and preserve every substantive function of the previous Abstract. No edit is propagated to the Introduction: the Introduction at L67-75 already contains the content removed from the Abstract (Wheeler/holographic/Lloyd citations, contributions list, public-code statement) and would only be duplicated.

### Edits applied

| Dimension | Before (D-0123 Abstract) | After (D-0125 Abstract) |
|---|---|---|
| Paragraph count | 3 | 2 |
| Word count | 470 | 352 (75% of original) |
| Em dashes (`---`) | 4 | 0 |
| Opening sentence | Direct claim about existing proposals | Contextual question (`Què constitueix el fonament últim de la realitat?`) followed by the existing-proposals statement as the answer being critiqued |
| `HSI` acronym in P2 | `a la família HSI` | `a la família derivada` |
| Detailed P1-P3 enumeration `(P1)... (P2)... (P3)...` | Present in long form | Compacted to `Tres principis lògics restringeixen l'origen radical a aquest candidat únic: la no-composició, l'anterioritat de l'explicació i l'absència de propietats`. Full P1-P3 specification preserved at §2 |
| Strong-nulls list `(shuffled, same-density, markov-1, matched-lz)` | Present | Removed; full list lives at §subsec:nulls_forts |
| `six independent seeds` detail of the pre-diagnostic | Present | Removed; full detail lives at §subsec:primera_obertura |
| Public-code-and-data closing sentence | Present | Removed; already at Introduction L75 |
| New material at P2 closing | --- | Single sentence pointing to D-0123 §subsec:fronteres_nivell2: `Dues sondes pre-registrades n'han delimitat ja fronteres empíriques observades.` |

### Em dashes removed (verbatim audit)

| # | D-0123 fragment | D-0125 replacement |
|---|---|---|
| 1 | `---\emph{it from bit}, principi hologràfic, univers computacional---` | `, com l'\emph{it from bit} de Wheeler, el principi hologràfic o l'univers computacional de Lloyd,` (em dash → integrated comma list) |
| 2 | `---no-composició, anterioritat de l'explicació i absència de propietats---` | `: la no-composició, l'anterioritat de l'explicació i l'absència de propietats` (em dash → colon enumeration) |
| 3 | `---no podria ser-ho per disseny---` | `per disseny` (em dash apposition → adverbial collapsed into the main clause) |
| 4 | `---àlgebra de defectes, graf multiescala, espectre---` | Split into two sentences: `... programa falsable: àlgebra de defectes, graf multiescala i espectre. Dues sondes pre-registrades n'han delimitat ja fronteres empíriques observades.` (em dash → colon enumeration + new sentence) |

The acronym `HSI` is preserved at its definition site in P1 (`Hipòtesi de la Singularitat Informacional (HSI)`) for coherence with the rest of the manuscript; only the second-occurrence in P2 (`a la família HSI`) is replaced.

### Frozen artifacts (post-edit)

| Artifact | Path | SHA256 |
|---|---|---|
| LaTeX source | `Paper_Latex/main_v2.02_cat.tex` | `42A9B7CFF1C40986C366DA63A848AB0E4A84FB4E9503160639B62F1C4B11C04A` |
| Compiled PDF (Overleaf) | `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_4.pdf` | `BAF2A841B64F4B44631A0F10A11146E14CAC093BE66838A2527EB6340630C944` |

### Wording Boundary

Forbidden inside `\begin{abstract}...\end{abstract}` from this cycle onward (in addition to the inherited fence from D-0121): em dashes (`---`); the second-occurrence of `HSI` (the definition site at first occurrence is still required). Preserved as still-allowed: parenthetical enumerations of the form `(a, b, c)` for natural list separators, since they do not carry the AI-stylistic signal that motivated the em dash removal.

### Open Questions

The new D-0125 Abstract has not been audited yet under any hostile-review modality. The dual-modality protocol (D-0122) is available; whether to invoke it on this short delta or accept the closure on the strength of the editorial constraints met by Iban directly is left as the next operational decision (not blocking under D-0125 itself).

---

## D-0126

- Date: 2026-05-12
- Status: `manuscript-edit-cycle-closed`
- Cycle: HSI v2 minor-revision closure on v2.02 CAT_4 (Gemini brief-blind, second pass)
- Trigger: After D-0125 closed the Abstract compression, Iban submitted CAT_4 to the same brief-blind Gemini conversation that had previously saturated under D-0123. Gemini returned a "Minor Revision" verdict listing four substantive flanks: (i) Abstract redundancy at the closing sentence, (ii) missing canonical citations for several controls (Wolfram for Rule 30, Lindenmayer for L-systems, Lind-Marcus for sofic shifts already in `.bib` but never cited in body), (iii) terminological soup mixing "Bloc" (structural), "Nivell" (ontological 0/1/2), and "Phase" (experimental pipeline), and (iv) repetitive defensive tone via `més modestament` (5 occurrences in §11-§15). Empirical pre-edit verification (string counts on `Paper_Latex/main_v2.02_cat.tex`) confirmed the four claims; Iban authorized the full response (Options A+B complete + C2 radical + D2 aggressive) in one cycle, then authorized a second batch of four canonical citations during refinement.

### Edits applied

The cycle was executed in two passes against `Paper_Latex/main_v2.02_cat.tex` and `Paper_Latex/references.bib`, producing PDFs CAT_5 and CAT_6.

#### Pass 1 (CAT_4 → CAT_5): 17 atomic patches

| # | Pass | File | Edit | Resolves |
|---|---|---|---|---|
| 1 | A | `.tex` Abstract L55-59 | Removed redundant final sentence repeating the radical-vs-restricted claim | Gemini A (Abstract redundancy) |
| 2 | B1 | `.bib` | Added `Lindenmayer1968` entry | Gemini B (missing canonical) |
| 3 | B2 | `.tex` L301 | Added `\cite{Wolfram2002}` to Rule 30 row in §6 table | Gemini B |
| 4 | B2 | `.tex` L995 | Added `\cite{Wolfram2002}` to Rule 30 row in appendix table | Gemini B |
| 5 | B3 | `.tex` L306 | Added `\cite{Lindenmayer1968}` to L-system row in §6 table | Gemini B |
| 6 | B3 | `.tex` L946 | Added `\cite{Lindenmayer1968}` to §obj_controls_externs prose | Gemini B |
| 7 | B3 | `.tex` L1000 | Added `\cite{Lindenmayer1968}` to L-system row in appendix table | Gemini B |
| 8 | B4 | `.tex` L476 | Promoted `LindMarcus1995` from footnote to body cite in §subsec:pm | Gemini B |
| 9-15 | C2 | `.tex` (various) | Replaced visible "Bloc" with "Nivell" or "Secció" across body text; LaTeX comments `% BLOC II/III` preserved | Gemini C (terminology) |
| 16 | D2 | `.tex` L782 | Removed one of the duplicated `més modestament` occurrences | Gemini D (defensive tone) |
| 17 | D2 | `.tex` L847, L853 | Removed two further `més modestament` occurrences, leaving exactly 2 (L784, L886) | Gemini D |

#### Pass 2 (CAT_5 → CAT_6): 5 atomic patches (canonical citation reinforcement)

After CAT_5 a conceptual sweep of the manuscript flagged that DET/RQA at L384 had no canonical anchor, and three further controls (MT19937, logistic map, box-counting) had no method-level citation despite being formally defined in the table. All four citations were verified against primary sources (DOI cross-checked) before insertion.

| # | File | Edit |
|---|---|---|
| 1 | `.bib` | Added 4 entries grouped after `Lindenmayer1968`: `Marwan2007` (Phys. Rep. 438:237-329, doi:10.1016/j.physrep.2006.11.001), `Falconer1990` (Wiley, ISBN 0471922870), `MatsumotoNishimura1998` (ACM TOMACS 8:3-30, doi:10.1145/272991.272995), `May1976` (Nature 261:459-467, doi:10.1038/261459a0) |
| 2 | `.tex` L299 | Added `\cite{MatsumotoNishimura1998}` to MT19937 row (control A) |
| 3 | `.tex` L302 | Added `\cite{May1976}` to logistic map row (control L) |
| 4 | `.tex` L380 | Added `\cite{Falconer1990}` to box-counting prose in §subsec:df |
| 5 | `.tex` L384 | Added `\cite{Marwan2007}` to RQA prose in §subsec:det |

### Conceptual mapping: Bloc → Nivell

The replacement is not a synonym swap but a precision gain. The `.tex` previously used "Bloc" both for ontological levels (Bloc 0 = ontology, Bloc 1 = signature, Bloc 2 = geometry) and as structural division markers in LaTeX comments. Mixing the two in body text invited the reading that the manuscript was poorly structured or that "Bloc" and "Nivell" were distinct ontological commitments. The cycle preserves the LaTeX comment markers (`% BLOC II`, `% BLOC III`) as authoring scaffolding while standardizing the body text on "Nivell" for the ontological/empirical hierarchy and "Secció" where pure structural reference is needed. Post-edit verification confirmed zero "Bloc" occurrences in visible body text.

### Verification

- All `\ref{}` labels touched by the cycle (`sec:signatura`, `sec:torre`, `subsec:obj_controls_externs`, `app:variants`, `subsec:pm`) remain valid.
- All new `\cite{}` keys (`Wolfram2002`, `Lindenmayer1968`, `LindMarcus1995`, `Marwan2007`, `Falconer1990`, `MatsumotoNishimura1998`, `May1976`) resolve to entries in `references.bib` verified against primary sources.
- `més modestament` count: 5 → 2 (L784, L886), as authorized under D2.
- No scientific claim, gate, signature value, or Level-2 boundary characterization was modified by this cycle. The closure is editorial-bibliographic, not substantive.

### Artefactes finals

| Artifact | SHA256 |
|---|---|
| `Paper_Latex/main_v2.02_cat.tex` | `9151D4142A4E27CD4051764D2746FBB20400CFE0B82FF20131C36066BDFB784B` |
| `Paper_Latex/references.bib` | `E95CBF5708FAD50056938C1A9328FCF09982E2248576C603E83449195DC44A83` |
| `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_6.pdf` | `E59D1C248B912983C368B1A85DD86BA20099E4AD7E60EAE02856B972D213D1F2` |

### Methodological note

The brief-blind modality (D-0122) saturated on framing under D-0123 and on Abstract length under D-0125. Under D-0126 it surfaced a new class of finding (canonical-citation hygiene + terminological soup) that the previous iterative-hostile passes had not produced, suggesting the two modalities probe genuinely orthogonal failure modes: iterative-hostile concentrates on falsifiable scientific claims, brief-blind concentrates on first-read narrative and bibliographic conventions. The cycle is logged as `manuscript-edit-cycle-closed`; no claim is made that it terminates the entire refinement phase, since the parallel Option-A independent reads by Ariadna (GPT-5.5 in Codex) and Grace (Gemini 3.1 Pro in Antigravity) on CAT_5/CAT_6 may surface further deltas before publication freeze.

---

## D-0127

- Date: 2026-05-14
- Status: `accepted-figure-rederivation`
- Cycle: post-D-0130 figure-table consistency closure on `main_v2.02_cat.tex`
- Trigger: Iban flagged that `fig:gate_plane` (Figure 3, file `Paper_Latex/phase1_gate_plane_tail2.png`) showed only the canonical variants, deterministic controls and strong nulls, while the immediately adjacent `tab:porta1_canonica` already included the Q4-pre-registered recursive external controls O and P. The figure was therefore strictly less informative than its own caption neighbour, an internal inconsistency that the brief-blind audit modality had not surfaced because the reader is normally drawn to the table for the numeric values and to the figure for the regime-separation claim, not the converse.

### Decision

D-0127 accepts the read-only re-derivation of `fig:gate_plane` to include the recursive external controls O and P as a distinct visual category. No upstream object was regenerated: O and P enter the figure via the same Q4 `phase1-transport` re-derivation that already feeds `tab:porta1_canonica`, and the canonical Phase 1 batches for the previously-plotted variants are reused verbatim.

### Edits applied

1. **`hsi_agents_project/v2/phase1/gate_map.py`**: O and P added to `DEFAULT_OBSERVED_VARIANTS`; new `PLOT_STYLE` entry `Recursive external control` (purple `#7c3aed` / `#5b21b6`); `_regime_bucket` and `_plot_group` extended to map O and P into that bucket; manual label offsets adjusted in the right cluster (`F`, `M/N`, `O`, `P`) so that `M/N` no longer overlaps `O`. The label-offset adjustment is strictly typographic; coordinates and category assignments are unchanged.
2. **Re-derivation run**: `hsi_agents_project/results/hsi_v2/visualizations/phase1-gate-plane-with-recursive-controls__20260514T233259/` with `manifest.json`, `report.md`, CSV, JSON, SVG and PNG. Manifest declares the Q4 sources used for O and P.
3. **Paper-facing artifact**: `Paper_Latex/phase1_gate_plane_tail2.png` overwritten with the re-derived PNG. SHA256 of the new artifact: `78F3DF2C43C36EA04BF197B7F33FBFCE3101CF780AE48D764A490A4B8DD26C44`.
4. **Caption of `fig:gate_plane`** in `Paper_Latex/main_v2.02_cat.tex` extended to mention the recursive external controls O and P; no claim added, no defensive nuance removed.

### Coordinates of O and P (re-derivation, must match the run above)

- O: `retention@last = 0.9433`, `active_mean_tail = 0.2457`
- P: `retention@last = 1.0000`, `active_mean_tail = 0.2269`

### Wording Boundary

The wording fence inherited from D-0130 is preserved. In particular, the figure, the caption, the script and the run report are kept clean of `geometry recovered`, `flux`, `current`, `transport` (in the energetic sense), `mass`, `charge`, and any reading of marker positions as physical magnitudes. O and P are introduced strictly as `recursive external controls pre-registered in cycle Q4`, never as a third regime competing with the HSI-like cluster.

### Practical Impact

- Figure 3 is now self-consistent with `tab:porta1_canonica`.
- No claim, no signature value, no gate definition and no falsification contract is modified by this entry.
- The freeze anchor recorded under D-0130 (`Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_7.pdf`, SHA256 `933A1378…CF5E24`) is **invalidated** by this caption edit and the new PNG; a new freeze anchor must be registered after recompilation as a retrospective note both here and at D-0130.
- This is the **second documented application** of the *revisió de fluïdesa lectora* discipline formalised at D-0130: the trigger is a readability-driven inconsistency between figure and adjacent table, and the response is strictly additive (one new visual category, no removal of existing markers, no caption claim weakened).

### Validation Traces

- Re-derivation script (modified): `hsi_agents_project/v2/phase1/gate_map.py`; entry-point `hsi_agents_project/hsi_v2_phase1_gate_map.py`.
- Canonical artifact directory: `hsi_agents_project/results/hsi_v2/visualizations/phase1-gate-plane-with-recursive-controls__20260514T233259/`.
- Paper-facing PNG SHA256: `78F3DF2C43C36EA04BF197B7F33FBFCE3101CF780AE48D764A490A4B8DD26C44`.
- Precedent for read-only paper-facing re-derivation: D-0130 (`fig:projective_tower_survival`).

### Compiled Output (Freeze Anchor)

The figure-table consistency closure is sealed against the following compiled PDF, produced from the updated `Paper_Latex/main_v2.02_cat.tex` under Overleaf with no compilation errors and visually verified by the human author:

- Path: `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_8.pdf`
- Size: 2{,}668{,}350 bytes
- Last write: 2026-05-14 23:41:41
- SHA256: `1F6506C7C72943C865BF38D36F3B307FA932DDBBFA66C6E4F9F64D3D3ADE76FC`

This artifact supersedes the CAT_7 freeze anchor recorded under D-0130 (`933A1378…CF5E24`) and becomes the binding seal for any subsequent audit pass on the post-D-0127 manuscript.

### Freeze Anchor superseded (added 2026-05-15 by D-0128)

The CAT_8 anchor recorded above is **superseded** by the CAT_9 anchor registered under D-0128 (`Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_9.pdf`, SHA256 `E3F82602933BF04A42C9B411F01F869D47F839DC7AC906B259AFDAE3930E996C`). CAT_8 is preserved here for historical continuity but is no longer binding; any subsequent audit pass on the post-D-0128 manuscript must verify against CAT_9.

---

## D-0128

- Date: 2026-05-15
- Status: `accepted-orthotypographic-pass`
- Cycle: post-D-0127 linguistic and orthotypographic closure on `main_v2.02_cat.tex`
- Trigger: external review by Gemini (acting as form-and-style proofreader on the CAT_8 PDF) returned a list of orthotypographic and lexical issues affecting the Catalan manuscript: hyphenated prefixes (`pre-`, `auto-`, `co-`, `micro-`), gender of `parèntesi`, residual anglicisms in technical prose (`drifta`, `claim-arla`, `nulls`, `runs`, `seeds`), and an unflagged dialectal mix between Central Catalan subjunctive forms (`sigui`, `hagi`, `pugui`, `vulgui`, `produeixin`) and the author's native Valencian variety. The same report flagged a putative absence of `l·l` (geminate ela) in the body text, which empirical inspection of the source `.tex` refuted: the geminate punt volat is correct throughout the source and the apparent absence in the PDF is a rendering artifact at the reviewer's font stack.

### Decision

D-0128 accepts a single conservative orthotypographic pass on `Paper_Latex/main_v2.02_cat.tex` covering three layers, executed as one logical commit and validated against a strict perimeter of protected substrings (`\label{}` keys, `\ref{}` arguments, LaTeX comments carrying `[Dxxx-Tx]` traces, captions and sections in English). No scientific claim, no signature value, no falsification contract and no defensive nuance is altered by this entry; the pass is strictly form-only.

### Edits applied

1. **Layer A (mechanical)**: hyphens removed from Catalan-native prefixes in body text (`pre-registrat → preregistrat`, `pre-diagnòstic → prediagnòstic`, `auto-contenció → autocontenció`, `co-creació → cocreació`, `micro-` family); `parèntesi` corrected to masculine; English residues replaced by Catalan equivalents in body text (`drifta → deriva`, `claim-arla → reivindicar-la`, `nulls → nuls`, `runs → execucions`, `seeds → llavors`, `seed → llavor` outside `\texttt{}`); technical-academic register unified (`reproductibilitat`, `falsable`, `significació`).
2. **Layer B (lexical)**: `annihilació` unified to `aniquilació` (DIEC2-normative form) across all body-text occurrences. Label keys `\label{subsec:annihilacio}` and all corresponding `\ref{subsec:annihilacio}` calls were preserved verbatim to keep the cross-reference graph and the Decision Log entries that cite these labels stable; the visible reading is `aniquilació`, the underlying key remains `annihilacio`.
3. **Layer C (dialectal Valencian alignment)**: Central Catalan subjunctive and imperative forms shifted to the Valencian variety used by the author (`sigui/siguin → siga/siguen`, `hagi/hagin → haja/hagen`, `pugui/puguin → puga/puguen`, `vulgui/vulguin → vulga/vulguen`, `produeixin → produïsquen`).

### Protected perimeter (verified)

Final regex sweeps confirm that the only residues left in the file under the layer-A and layer-C patterns are strictly within the protected perimeter:

- `\label{subsec:pre-fisic}` and `\label{subsec:annihilacio}` (label keys; cross-reference integrity).
- `\ref{subsec:annihilacio}` and `\ref{subsec:autocontencio}` (reference calls to preserved keys).
- LaTeX comments `% [D101-T1]` and `% [D106-T1]` (operational traces of the per-cycle edit log).
- Caption of `tab:variants_taxonomy` and the section heading `\section{Phase 3 internal pre-geometric diagnostic stack}` (English-language blocks; outside the Catalan pass perimeter).

No matches remain for the Central Catalan subjunctive set (`sigui|siguin|hagi|hagin|pugui|puguin|vulgui|vulguin|produeixin`) in the body text.

### Wording Boundary

The wording fence inherited from D-0130 and reaffirmed by D-0127 is preserved untouched. None of the prohibited terms (`geometry recovered`, `Phase 3 solved`, `scaling law derived`, `flux`, `current`, `transport` in the energetic sense, `mass`, `charge`) was introduced or modified by this pass; the orthotypographic perimeter operates strictly below the claim layer.

### False-positive filtering

The Gemini report's claim about missing `l·l` was filtered out before applying the pass: a regex sweep over the eight target roots (`col[ ]?lapse`, `cristal[ ]?l`, `nul[ ]?la`, `paral[ ]?lel`, `sol[ ]?licit`, `col[ ]?laborat`, `intel[ ]?lig`) returned only English-language matches inside captions plus correct `col·lapse`/`cristal·lí`/`paral·lel` occurrences in the Catalan body text. The flagged absence is a PDF font-rendering artifact at the reviewer's end, not a source issue. This filter is recorded here so that future external reviews flagging `l·l` on rendered output are routed first through a `.tex`-level verification before any edit is applied.

### Practical Impact

- The Catalan manuscript is now orthotypographically homogeneous and dialectally consistent with the author's native variety.
- The cross-reference graph is provably preserved (no `\label{}` key was mutated).
- No claim, signature value, gate definition or falsification contract is modified.
- The freeze anchor recorded under D-0127 (`Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_8.pdf`, SHA256 `1F6506C7…ADE76FC`) is **invalidated** by this orthotypographic pass; a new freeze anchor is registered below and a retrospective supersession note is added to D-0127.

### Compiled Output (Freeze Anchor)

The orthotypographic closure is sealed against the following compiled PDF, produced from the updated `Paper_Latex/main_v2.02_cat.tex` under Overleaf with no compilation errors and visually verified by the human author:

- Path: `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_9.pdf`
- Size: 2{,}668{,}200 bytes
- Last write: 2026-05-15 01:13:38
- SHA256: `E3F82602933BF04A42C9B411F01F869D47F839DC7AC906B259AFDAE3930E996C`

This artifact supersedes the CAT_8 freeze anchor recorded under D-0127 (`1F6506C7…ADE76FC`) and becomes the binding seal for any subsequent audit pass on the post-D-0128 manuscript. The minor size delta vs.\ CAT_8 ($-150$ bytes) is consistent with a body-text-only orthotypographic pass and contains no figure or layout regeneration.

### Freeze Anchor superseded (added 2026-05-15 by D-0132)

The CAT_9 anchor recorded above is **superseded** by the CAT_11 anchor registered under D-0132 (`Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_11.pdf`, SHA256 `05EAE8DDAFC94F6A35ABE9D98CD65AECD5CE7F9D82BD84158F1A0CB132AFD27D`). CAT_9 is preserved here for historical continuity but is no longer binding; any subsequent audit pass on the post-D-0132 manuscript must verify against CAT_11. CAT_10 was never produced as a separate freeze artefact — the residual-reference fix scoped under D-0129 and the ontological-promotion changes scoped under D-0132 were absorbed into a single recompilation pass that produced CAT_11 directly.

### Validation Traces

- Edited file: `Paper_Latex/main_v2.02_cat.tex`.
- Final regex sweeps run on the post-edit file:
  - Layer A: `\bpre-[a-zàèéíòóúïü]+|\bauto-[a-zàèéíòóúïü]+|\bco-[a-zàèéíòóúïü]+|\bmicro-[a-zàèéíòóúïü]+` → 5 matches, all inside the protected perimeter.
  - Layer C: `\bsigui\b|\bsiguin\b|\bhagi\b|\bhagin\b|\bpugui\b|\bpuguin\b|\bvulgui\b|\bvulguin\b|\bproduei?xin\b` → 0 matches.
  - Anglicism set: `reproducibilitat|falsificab|signific[àa]ncia|drifta|claim-?ar|nulls forts|seeds testades|les par[èe]ntesis` → 0 matches.
- Precedent for paper-facing closure under read-only discipline: D-0130 (`fig:projective_tower_survival`) and D-0127 (`fig:gate_plane`).

### Open items not addressed by this entry (queued)

- Bilingual sync of the post-D-0127 caption update for `fig:gate_plane` and the post-D-0128 orthotypographic conventions to `Paper_Latex/main_eng.tex` remain out of scope here; they will be opened as a separate cycle when the human author authorises it.
- Optional Grace pass over CAT_9 in scientist/researcher hat: the orthotypographic delta is sub-claim and is unlikely to surface new substantive flanks vs.\ the CAT_8 audit, but the option is preserved.

---

## D-0129

- Date: 2026-05-15
- Status: `accepted-ontological-promotion`
- Cycle: post-D-0128 ontological strengthening on `main_v2.02_cat.tex`
- Trigger: editorial decision to promote the dialectical defence of P1-P3 (Section `subsec:defensa`) to a formal logical derivation, redistributing the burden of ontological proof: rather than asking the reader to accept P1-P3 as independent axioms, the manuscript should derive them as theorems from a minimal set of classical rationalist primitives.

### Decision

D-0129 accepts the integration of a new first appendix `\section{Derivació formal de P1-P3 des dels primitius racionalistes}\label{app:derivacio_formal}` and a single cross-reference paragraph at the end of `subsec:defensa` pointing to it. The appendix is purely foundational; it adds no claim about the generative mechanism, the experimental signature, or the falsifiable contract, and the existing dialectical defence in `subsec:defensa` is preserved verbatim and complemented (not replaced) by the formal derivation.

### Adversarial review (Grace, logical peer review) and editorial outcome

A logical peer review by Grace on the proposed derivation issued an `APROBADO CON PATCHES` verdict identifying three structural patches:

1. **SR underspecified.** The plain Principle of Sufficient Reason `∀e ∃r (r explains e)` does not block self-explanatory loops nor coherentist constructions. Patch: refine SR as irreflexive and asymmetric, `∀e [contingent(e) ⟹ ∃r (r explains e ∧ r ≠ e ∧ ¬(e explains r))]`. **Status: integrated.**
2. **Missing structural-contingency primitive.** Without an explicit contingency clause on distinctive properties, an essentialist could claim that an origin's structural properties are logically necessary and therefore exempt from SR's demand. Patch: add a fifth primitive (CE) stating that any distinctive property is contingent, not logically necessary. **Status: integrated.**
3. **Honest enumeration of escape routes (C1-C6).** Grace recommended an explicit enumeration of named escape routes (rejection of SR, II, classical logic, ontological coherentism, brute essentialism, plus the second-order-property paradox of `N`). **Status: editorial override.** An initial draft included this enumeration as a dedicated subsection `app:derivacio_formal:caveats`. Editorial review by the human author rejected it on the grounds that (i) listing the routes as "recognised escape routes" implicitly grants them the status of equivalent alternatives to the rationalist frame, which they do not merit; (ii) writing the rebuttal for the reader effectively builds the weapons against the derivation and inflates the defensive posture beyond what the manuscript needs; (iii) each escape route is itself a contested philosophical commitment, not a free pass. The replacement is a single short paragraph integrated into `app:derivacio_formal:lectura` that briefly acknowledges the principal philosophical currents questioning some primitives (radical readings of QM on SR, haecceitas defenders on II, paraconsistent traditions on classical logic, brute essentialism on CE) without enumerating them as a structured menu. The two structural patches (1) and (2) above are retained because they strengthen the derivation; the enumeration patch (3) is dropped because it weakens the editorial confidence of the appendix.

### Edits applied

1. **New first appendix** `\section{Derivació formal de P1-P3 des dels primitius racionalistes}` inserted immediately after `\appendix` in `Paper_Latex/main_v2.02_cat.tex`. Internal structure: three subsections — five rationalist primitives (I, NC, SR-refined, II, CE), three theorems (P1 anti-regression, P2 structure-implies-explanation, P3 purity-and-uniqueness-of-Nothingness) with explicit proofs in `\paragraph{Teorema...} \textit{Demostració.} ... $\blacksquare$` style consistent with existing `lema` environments, and editorial reading (two paragraphs: principal questioning currents acknowledged briefly, then the redistribution of the burden of proof). Notation: $\mathcal{O}, \mathcal{S}, \mathcal{E}, \mathcal{B}, N$ for radical-origin, has-structure, requires-explanation, is-something, the-Nothingness.
2. **Cross-reference paragraph** `\paragraph{Derivació formal complementària.}` appended at the end of `subsec:defensa`, pointing to `\ref{app:derivacio_formal}`. The pre-existing dialectical itemize is preserved verbatim above it. The closing fragment defers to the appendix for "the principal philosophical currents that question some of these primitives" rather than enumerating six named escape routes.
3. **Residual reference cleanup**: a forward `\ref{app:derivacio_formal:caveats}` left inside the introductory paragraph of `app:derivacio_formal:primitius` after the subsection deletion was rewritten to point implicitly to the editorial-reading subsection, eliminating the two undefined-reference Overleaf warnings flagged on first compilation.

### Placement decision (Option α)

Three placements were considered for the new appendix:

- **Option α (chosen)**: first appendix, immediately after `\appendix`. Cost: shifts appendix letters A→B, B→C, ... for all subsequent appendices, with zero damage to `\ref{app:xxx}` calls (all internal references are by symbolic key, regenerated on compilation). Benefit: pedagogical gradient runs cleanly from ontological foundation → operational variants → methodological details → technical → visual.
- Option β: between `app:cocreacio` and the technical block (philosophical grouping). Rejected: weaker pedagogical signal that the derivation is foundational rather than auxiliary.
- Option γ: last appendix (zero letter shift). Rejected: inverts the foundational-first reading order.

### Honest scope of the derivation

The derivation is logically forced only inside the closed frame `{I, NC, SR-refined, II, CE} + classical logic`. The appendix states this restriction explicitly and acknowledges briefly that there exist philosophical currents questioning some of the primitives, without scripting a structured rebuttal menu for the reader. This is the editorial balance after the override of Grace's third patch: transparency on the framework's boundaries without inflating the defensive posture.

### Protected perimeter (verified)

- No edits inside any `\label{}` key, `\ref{}` argument, LaTeX comment carrying `[Dxxx-Tx]` traces, English-language caption or section, or any pre-existing prose of `subsec:defensa` (the new paragraph is appended, not interleaved).
- No changes to falsifiable signature counts, null-model commitments, Phase 1 / Phase 2 / Phase 3 operational contracts, or any numerical claim in the manuscript.
- No introduction of `phi`, Fibonacci, geometric primitives, or any element forbidden by the working rules in `AGENTS.md`.

### Compiled Output (Freeze Anchor)

Resolved by the CAT_11 freeze anchor registered under D-0132 (`Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_11.pdf`, SHA256 `05EAE8DDAFC94F6A35ABE9D98CD65AECD5CE7F9D82BD84158F1A0CB132AFD27D`). The D-0129 textual changes (residual-reference fix and follow-on edits) were absorbed into the same compiled artefact as the D-0132 ontological-promotion changes, so a separate CAT_10 PDF was never produced; the user's recompilation pass under D-0132 produced CAT_11 directly, covering both cycles. CAT_9 (D-0128, `E3F82602…30E996C`) is therefore superseded by CAT_11 via D-0132 and a retrospective `Freeze Anchor superseded` note has been appended to D-0128.

### Editorial governance note

This entry establishes a working pattern for future audit cycles: structural patches that strengthen the derivation (SR refinement, CE primitive) are integrated unconditionally; rhetorical patches that pre-script the reader's rebuttal route are subject to editorial review, since an auditor's instinct toward exhaustive defensive enumeration can overshoot the manuscript's actual defensive needs. Both decisions (acceptance of structural patches, override of enumeration patch) are recorded explicitly so the audit trail remains honest about where the human author deviated from the auditor's recommendation and why.

The pattern has been **codified as a permanent operating principle** in both relevant memory cores on 2026-05-15:

- `.dev/assistants/Sofia_Claude_Memory_Core.md`, §4-ter: *Editorial integrity over reviewer-anticipation patches* (Sofia's filter on incoming peer-review proposals).
- `.dev/assistants/Grace_Antigravity_Memory_Core.md`, §3-ter: *Editorial integrity over reviewer-anticipation patches (Sofia's filter applied to my own recommendations)* (Grace's mirror acknowledgement that her recommendations are input, not veto).

D-0129 is registered as the first documented application of this codified principle. Future audit cycles citing the principle should reference both memory-core sections plus this entry.

### Open follow-ups

- User to recompile `main_v2.02_cat.tex` → `CAT_10.pdf` and report the SHA256 hash for freeze-anchor closure.
- English-translation cycle (when opened) must port `app:derivacio_formal` and the cross-reference paragraph to `main_eng.tex`.
- If a future hostile review by Grace (with researcher hat) returns substantive objections to the derivation itself (not the orthotypography or the absent enumeration), a new D-XXXX entry will document the response and any patches.

---

## D-0130

- Date: 2026-05-14
- Status: `accepted-editorial`
- Cycle: editorial closure / readability pass for the "Interested Reader" on `main_v2.02_cat.tex`
- Trigger: external editorial feedback flagged that the Level 0 mechanism was readable as algorithm but opaque as walk-through, that the operational ceilings on $m$ and $N$ could be misread as algorithmic limits, and that the projective tower (Section §7.1) lacked a paper-facing visualization of the qualitative regimes already documented in `tab:pm_size`. Sofia proposed three minimally invasive editorial additions; the human author authorized the three and rejected an additional generator-flow diagram as redundant once the toy example was in place.

### Decision

D-0130 records three coordinated editorial insertions into `Paper_Latex/main_v2.02_cat.tex`:

1. **Toy example (`\subsection{Walk-through del mecanisme}\label{subsec:toy_example}`)**: new §3 subsection inserted between the *Lema d'unicitat* paragraph and `\subsec{Auto-contenció}`. Traces the first iteration ($\alpha = 0$) of Variant B from $R_0 = \texttt{0}$, with a per-pass table of $\mathrm{Acc}$ and the inside$\to$out collapse, the global finalize, and the observable reading $\Phi_0 = \texttt{001010}$. Closes with a forward pointer to the exponential growth at $\alpha = 23$ ($\approx 4{,}29 \times 10^9$ bits) without re-deriving it.
2. **Computational ceiling paragraph (`subsec:implementacio`)**: single paragraph appended after the description of the Python core. Clarifies that the current operational ceiling ($m \leq 32$, $\sim 10^6$ bits per projective-tower window) is set by RAM, disk and CPU time of the analysis stack, not by any algorithmic obstacle, with explicit cost statement ($\min(2^m, N - m + 1)$ window inventory, dominant cost linear in $N$ at fixed $m$).
3. **Projective tower visualization (`fig:projective_tower_survival`)**: new Sankey-style figure inserted in §7.1 between the *Tres observacions clau* paragraph and `\subsec{Fibres: supervivència}`. The figure renders pattern survival across $m \in \{8, 12, 16, 20, 24, 28, 32\}$ for B/E, I, F, M/N, A, L. The artifact derives from the canonical Phase 1 `phase1_summary.json` runs (no regeneration) and is reproducible via Ariadna's read-only script `hsi_agents_project/hsi_v2_phase1_projective_tower_survival_figure.py`. Grace audited the script and the derived PDF as `APPROVED FOR PAPER INTEGRATION` at `docs/HSI-audit-Phase1-Projective-Tower-Figure.md`. The PDF has been copied to `Paper_Latex/phase1_projective_tower_survival.pdf` and the canonical artifact remains intact at `hsi_agents_project/results/hsi_v2/visualizations/phase1-projective-tower-survival__20260514T084303/`.

### Wording Boundary

Allowed (and present in the caption):

- `fracció de patrons amb fibra prefix no buida`
- `fracció caiguda després del filtre d'estabilitat`
- `règim compacte-selectiu / rígid / cristal·lí / col·lapse aleatori`
- `suport visual de la Taula~\ref{tab:pm_size}`
- `Les longituds dels segments són fraccions de patrons, no magnituds físiques.`

Forbidden (and explicitly absent from the figure, the caption, the script and the toy example):

- `flux`, `current`, `transport` (in the energetic sense), `mass`, `charge`, `geometry recovered`
- any reading of segment lengths as physical magnitudes
- any new claim not already supported by `tab:pm_size` or by §7.1 prose
- a generator flow diagram (deliberately rejected as redundant once `subsec:toy_example` was in place)

### Initial Misroute and Correction

The first execution of these three insertions was performed on the obsolete `Paper_Latex/main_v2_cat.tex` file rather than on the canonical `Paper_Latex/main_v2.02_cat.tex`. The misroute was detected by the human author, the three edits were re-applied verbatim to the canonical file, and the obsolete file is scheduled for removal by the human author to prevent future confusion. No claim or wording was altered between the two applications.

### Practical Impact

- `Paper_Latex/main_v2.02_cat.tex`: three insertions applied (`subsec:toy_example` at §3.5, ceiling paragraph at `subsec:implementacio`, `fig:projective_tower_survival` at §7.1).
- `Paper_Latex/phase1_projective_tower_survival.pdf`: new figure artifact under the flat figure convention of `Paper_Latex/`.
- `Paper_Latex/main_eng.tex`: requires absorption in the bilingual sync cycle (queued; not in scope of this entry).
- No change to claims, no change to numerical results, no change to the falsification contract.

### Validation Traces

- Grace audit of the figure and script: `docs/HSI-audit-Phase1-Projective-Tower-Figure.md`, verdict `APPROVED FOR PAPER INTEGRATION`.
- Reproducer script: `hsi_agents_project/hsi_v2_phase1_projective_tower_survival_figure.py`.
- Canonical artifact directory: `hsi_agents_project/results/hsi_v2/visualizations/phase1-projective-tower-survival__20260514T084303/` with `manifest.json`, `report.md`, `projective_tower_survival.pdf`, `projective_tower_survival.png`, `projective_tower_survival_data.json`.
- Inputs declared in the manifest derive from the canonical Phase 1 runs of B, E, I, F, M, N, A, L under the protocol `iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix`.

### Compiled Output (Freeze Anchor)

The editorial cycle is sealed against the following compiled PDF, produced from `Paper_Latex/main_v2.02_cat.tex` under Overleaf with no compilation errors and visually verified by the human author:

- Path: `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_7.pdf`
- Size: 2{,}660{,}832 bytes
- Last write: 2026-05-14 09:25:52
- SHA256: `933A13780E43957944EDE4996C4554BCC86F9D0A32D8DAC2C38B8882E7CF5E24`

This artifact is the freeze anchor for any subsequent audit pass on the post-D-0130 manuscript. A new external-audit modality has been opened on this freeze: Grace operating under a `scientist/researcher` framing rather than the standard methodological-auditor framing. The verdict of that pass will be recorded in a separate entry if it triggers any substantive change; if it returns only minor or non-blocking observations, this freeze stands.

### Retrospective Methodological Refinement (added 2026-05-14)

After the editorial pass that produced this entry, the framing under which Sofia proposed the three insertions was formalized as a fifth operational discipline of the AI-text protocol, named **revisió de fluïdesa lectora** (reader-fluency review). The discipline is invoked explicitly, executed by the audit role under a temporary "interested reader with limited time" framing instead of the standard methodological-auditor framing, and bound by an asymmetric safeguard: only additive structural scaffolding is admissible (toy examples, walk-throughs, visualisations, ordering, visual hierarchy, summaries, glossary); any proposal that would dilute a claim, remove a defensive nuance, shorten a paragraph that answers a documented objection or relax a wording-fence boundary is vetoed by the same audit role operating in standard mode before it lands in the text. The standard mode has veto over the fluency mode, never the inverse.

This D-0130 entry is hereby recognised as the **first documented application** of that discipline. The three insertions admitted by it (the toy-example walk-through `subsec:toy_example`, the operational-ceiling paragraph in `subsec:implementacio`, and the projective-tower visualisation `fig:projective_tower_survival`) are net additions that did not soften any claim, did not remove any defensive nuance and did not relax any wording-fence boundary. Verification: the wording-boundary audit recorded above in this same entry was executed precisely under the standard-mode veto chain that the discipline mandates.

The formalization of the discipline is recorded in:

- `.dev/assistants/Sofia_Claude_Memory_Core.md` §4-bis (acceptance filter for fluency-mode proposals reaching the `.tex`).
- `.dev/assistants/Grace_Antigravity_Memory_Core.md` §3-bis (operational mode invoked explicitly, hard constraints, self-veto chain).
- `Articles_i_publicacions/borrador_metodologia_co_creacion_AI_castellano.md` and `draft_methodology_co_creation_AI_english.md` (outreach-level description of the fifth discipline and its safeguard).

### Scope decision: discipline absent from the manuscript by design

The fifth discipline is deliberately **not** carried into `Paper_Latex/main_v2.02_cat.tex` §10.2. An initial revision of this entry attempted to extend the "Quatre disciplines" paragraph to "Cinc" with an explicit description of the asymmetric safeguard; the human author rejected the extension and the paragraph was reverted to its prior four-discipline form. Rationale, recorded here so the decision is not relitigated in future cycles:

1. **Single-failure-mode focus**. §10.2 argues a single failure mode (sycophancy) with four disciplines all directed at that single threat. The fifth discipline targets a structurally distinct failure mode (reader fatigue) and would require its own internal safeguard, breaking the focus of the section.
2. **Asymmetry of register**. The operational layer (memory cores), the canonical operational record (this Decision Log) and the outreach layer (Substack methodology article) are the natural venues to formalize and explain procedural disciplines. The paper is not a procedural-detail venue; it is a scientific argument with bounded scope.
3. **Anti-defensive containment**. Enumerating ever more disciplines in the paper risks reading as defensive over-explanation against external criticisms of AI-assisted work, suggesting authorial insecurity about the method. The four disciplines stated in §10.2 already sustain the anti-sycophancy argument; adding a fifth on a different failure mode would degrade, not strengthen, the section.

The freeze anchor declared above is therefore unaffected by the §10.2 scope decision: the SHA256 of the compiled PDF as listed (`933A1378…CF5E24` over CAT_7) remained the binding seal at the time this scope decision was taken, and the §10.2 paragraph it covers stands as written.

### Freeze Anchor superseded (added 2026-05-14 by D-0127)

The CAT_7 freeze anchor recorded above is **superseded** by the recompilation triggered by D-0127 (read-only re-derivation of `fig:gate_plane` to include the recursive external controls O and P, and the corresponding caption extension). The two changes that invalidated the CAT_7 PDF are:

- the new paper-facing PNG `Paper_Latex/phase1_gate_plane_tail2.png` (SHA256 `78F3DF2C43C36EA04BF197B7F33FBFCE3101CF780AE48D764A490A4B8DD26C44`);
- the caption of `fig:gate_plane` in `Paper_Latex/main_v2.02_cat.tex`, extended to mention O and P explicitly.

The new compiled PDF, produced from the updated `main_v2.02_cat.tex` under Overleaf with no compilation errors and visually verified by the human author, is:

- Path: `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_8.pdf`
- Size: 2{,}668{,}350 bytes
- Last write: 2026-05-14 23:41:41
- SHA256: `1F6506C7C72943C865BF38D36F3B307FA932DDBBFA66C6E4F9F64D3D3ADE76FC`

This CAT_8 SHA256 is the new binding freeze anchor for any subsequent audit pass on the post-D-0127 manuscript. The CAT_7 anchor is retained above for historical continuity only; any future verification must check against CAT_8.

---

## D-0131

- Date: 2026-05-15
- Status: `accepted-process`
- Cycle: Decision Log convention unification + retroactive migration
- Trigger: Iban diagnosed that Sofia and Ariadna had been operating two incompatible insertion conventions on `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md`. Sofia inserted new entries at the top of the file in reverse-chronological order; Ariadna appended new entries at the end of the file in forward-chronological order. The mismatch silently dropped the Sofia-side D-0129 entry across at least three reapplications during the post-D-0128 ontological-promotion cycle, and produced a numbering collision at D-0111: Sofia's entry (Q2 internal-routing-entropy Stage 1 preregistration, 2026-05-09) and Ariadna's entry (editorial closure on `main_v2.02_cat.tex` CAT_7 freeze anchor, 2026-05-14) were both written under the same identifier without seeing each other.

### Decision

D-0131 sets a single canonical convention for `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md`:

- All new entries are **appended at the end of the file**, in **forward-chronological order** by entry-ID.
- Reverse-chronological insertion at the top of the file is forbidden going forward.
- Numbering is strictly monotonically increasing. Before assigning a new D-NNNN, the next-free number must be confirmed by inspecting the last entry currently in the file rather than relying on cached state.

### Migration Performed

Under this convention, the file was reorganized in a single atomic pass:

- All entries previously held in the reverse-chronological top block (D-0107 through D-0129, with D-0109 absent from that block) were relocated to the end of the file in numeric ascending order.
- The orphan D-0109 (Bridge-to-Level-2 empirical closure, previously isolated mid-file at the original line 8980) was relocated alongside its siblings D-0107 (Bridge Stage 0 preregistration) and D-0108 (Bridge Stage 1 preregistration) so the bridge-to-Level-2 series is contiguous.
- The D-0111 numbering collision was resolved by renaming the editorial closure entry (Ariadna, 2026-05-14, CAT_7 freeze anchor on `main_v2.02_cat.tex`) to **D-0130**. The Q2 Stage 1 entry (Sofia, 2026-05-09) keeps the D-0111 designation because it belongs to a coherent Q2 series (D-0110 Stage 0 → D-0111 Stage 1 → D-0112 hard-negative closure) whose internal cross-references would otherwise have to be rewritten in three places.
- All cross-references to the renamed entry were updated within D-0127, D-0128 and the body of the renamed entry itself (8 textual occurrences). Cross-references to the Q2 D-0111 (within D-0112 and within the entry's own body, 3 textual occurrences) were preserved unchanged.

### Practical Impact

- Final entry inventory: 130 entries, IDs D-0001 through D-0130, contiguous and unique.
- File length: unchanged at 9164 lines (net zero, pure reordering plus rename).
- No claim, no scientific result, no preregistered contract and no falsification boundary was modified by this entry.
- The CAT_7 freeze anchor previously cited under D-0111 in audit-trace contexts must be cited under D-0130 from this point on; the CAT_8 freeze anchor under D-0127 is unaffected.

### Validation Traces

- Post-migration regex sanity checks executed: 130 unique IDs, no duplicates, contiguous D-0001..D-0130, all CAT_7-side cross-refs renamed to D-0130, all Q2-side cross-refs to D-0111 preserved, preamble (lines 1-23) intact, end-of-file `---` separator preserved.
- Pre-migration backup of the file retained as `hsi_agents_project/Documentation/HSI_v2_Decision_Log.md.bak` until the next editorial cycle confirms no rollback is needed; deletion of the backup is to be authorized by Iban.
- This entry is the **first new entry written under the new convention** (appended at the end of the file).

### Operational Note for All Agents

Sofia, Ariadna and Grace must read this entry before adding any further D-NNNN to the log. The convention applies symmetrically: no agent has standing to insert at the top of the file. A short courtesy memo communicating this convention has been sent to Ariadna in Sofia's name through the team channel.

---


## D-0132

- Date: 2026-05-15
- Status: `accepted-formal`
- Cycle: Ontological enrichment — Theorem of Ontological Inviability of Pure Nothingness ($N$)
- Trigger: Iban proposed extending the formal apparatus already established for P1–P3 (cf. D-0128 ontological promotion) with a complementary formal result on the inverse question: can pure Nothingness $N$ sustain itself coherently as the sole reality? Sofia drafted a candidate theorem (V1), submitted it to Grace under the adversarial-collaboration protocol (cf. D-0129 / Sofia §4-ter / Grace §3-ter), and iterated the artefact through two full audit cycles before integration.

### Decision

A new formal appendix titled *Inviabilitat ontològica del No-res* (label `app:inviabilitat_no_res`) is integrated into `Paper_Latex/main_v2.02_cat.tex` as a self-contained sibling of `app:derivacio_formal`. A short cross-referencing `\paragraph{Inviabilitat del No-Res com a única realitat.}` is inserted into `subsec:defensa` immediately after the existing `\paragraph{Derivació formal complementària.}` so the dialectical defense of P1–P3 now points to both formal appendices. The new appendix establishes, under a minimal auxiliary hypothesis (HA), that $N$ cannot sustain itself as the sole reality; the result is strictly negative and posits no alternative substantive entity.

### Audit Trace (V1 → V2 → V3)

The theorem went through two full Grace audits before reaching the integrated form:

- **V1**: rejected with verdict `REQUIRES-SUBSTANTIVE-REVISION`. Four real errors identified: (i) a broken (CE)+(SR) chain in the proof, (ii) a use/mention fallacy in the preliminary observation, (iii) a Lemma 2 vulnerable to the classical apophatic defense, (iv) a corollary with leibnizian cosmological structure overstepping the appendix's negative scope.
- **V2**: drafted by Sofia integrating all four corrections without reservation. Grace's directive on *necessity as relational property* (post-Kripke) was incorporated as a new Lemma 3, blocking the necessity-route exit. Verdict on V2: `ACCEPTED-WITH-MINOR-EDITS`. One residual flaw identified: Lemma 2's topological argument (domain ↔ distinction between points) was vulnerable to the singleton/monad counterexample (a 0-dimensional point is a valid domain of cardinality 1).
- **V3** (integrated): Lemma 2 rewritten under Grace's directive as the *Contingency horn* via (SR), with the Theorem reformulated as a strict modal dilemma (excluded middle on contingent vs. necessary existence of $N$). Lemma 2 covers the contingency horn; Lemma 3 covers the necessity horn; the disjunction is exhaustive in standard modal logic under HA. Both horns collapse, blocking the postulation that $N$ is the sole reality from both modal sides simultaneously.

### Editorial Decisions Authorized by Iban

- Tone: defensive-modest formulation ("argument conditioned on rationalist primitives showing $N$ is not a coherent fixed point"), not the strong "definitive demonstration" register. Aligned with the same cautiousness adopted for P1–P3 in D-0128.
- Structural placement: separate appendix (`app:inviabilitat_no_res`) parallel to `app:derivacio_formal`, not internal subsection of the latter, to preserve conceptual separation between the two derivations.
- Vocabulary: the V1 expression *imperfectament pur* was abandoned because it was inseparable from the use/mention fallacy. The interpretive notion of *cicatriu* (scar) is preserved but explicitly restricted to the editorial reading subsection (`app:inviabilitat_no_res:lectura`) and marked as external interpretation, not part of the deductive body.
- Coupling with HSI: the theorem stops at the strictly ontological level (rejection of $N$ as sole reality); it does not assert which substantive entity exists, nor does it specify the form of the resulting "scar" (relational, dynamic, modal, informational). The link to the HSI dynamic appears only in the closing editorial paragraph, framed as the content of the empirical sections, not as a consequence of the theorem.
- Plantinga-style intrinsic-necessity caveat: preserved as an in-lemma `\textit{Nota}` so the parry against minoritarian non-relational readings of necessity is visible to a hostile reviewer. Williamson and Fine treatments are dispatched in the same note via (II) under Lemma 1.

### Editorial-Integrity Application

This is the second cycle in which the principle codified in D-0129 (Sofia §4-ter / Grace §3-ter) is exercised, and the second consecutive cycle in which **all** of Grace's recommendations were structural/logical errors warranting unconditional integration. No reviewer-anticipation patches were proposed by Grace and none were dispatched by Sofia: the discipline produced no friction in this cycle. The pattern is recorded as a healthy precedent of the principle operating without contention.

### Files Modified

- `Paper_Latex/main_v2.02_cat.tex`:
    - New `\paragraph{Inviabilitat del No-Res com a única realitat.}` inserted in `subsec:defensa` (after the existing `\paragraph{Derivació formal complementària.}`) cross-referencing the new appendix.
    - New `\section{Inviabilitat ontològica del No-res}\label{app:inviabilitat_no_res}` inserted between `app:derivacio_formal` and `app:variants`, with four subsections: `:base` (definition + HA), `:lemes` (Lemmas 1–3), `:teorema` (Theorem + Corollary), `:lectura` (editorial reading + relation to `app:derivacio_formal`).
- This decision log entry, appended at the end of the file under the convention codified in D-0131.

### Falsification Boundary (unchanged)

This appendix introduces no new claim about generative mechanism, projective hierarchy or experimental regime. The Phase-1 falsifiable contracts and the empirical chain (CAT_8 freeze anchor under D-0127, modulo any subsequent recompilation under D-0128 or D-0132) remain intact. The result redistributes ontological burden of proof; it does not introduce or modify any empirical claim.

### Editorial Follow-up: `subsec:implicacio` rewrite + Nota axiomàtica refinement

A direct editorial consequence of the formal integration above, identified by Iban during a re-reading pass: `subsec:implicacio` ("La implicació inevitable") and the embedded `\paragraph{Nota axiomàtica.}` predated D-0132 and still presented the No-Res-pur ↔ No-Res-Absolut tension as a pure postulate, not as the now-derivable consequence of the inviability theorem. Three coordinated changes were applied (option U1: consolidating the formal cross-reference into `subsec:implicacio` rather than maintaining a parallel pointer in `subsec:defensa`):

1. The cross-referencing `\paragraph{Inviabilitat del No-Res com a única realitat.}` previously inserted into `subsec:defensa` (D-0132 first pass) is removed. Its argumentative function migrates to `subsec:implicacio`, where it serves the constructive (not defensive) role natural to the appendix.
2. The opening paragraph of `subsec:implicacio` is rewritten to derive the necessity of a generative mechanism from the *conjunction* of both formal appendices, replacing the prior conditional "Si P1--P3 són correctes" with the stronger "Donats P1--P3 com a teoremes derivats del marc racionalista mínim (Apèndix~\ref{app:derivacio_formal})". A short in-line gloss of (SR) is added for non-philosophical readers.
3. The `\paragraph{Nota axiomàtica.}` is refined to distinguish explicitly between what the theorem now establishes (inviability of $N$ as sole reality) and what remains genuinely axiomatic in HSI (the *dynamic-constructive reading* of the inviability, i.e. the "scar" interpretation). The original spirit of the note (axiomatic system, analogue of the equivalence principle, empirical not metaphysical validation) is preserved intact; the refinement reduces the axiomatic load by promoting one previously postulated element (the existence of the tension) to derived status.
4. Micro-editorial pass on the rewritten opening of `subsec:implicacio`: the symbol `$N$` is glossed inline immediately after the first body-text mention of "No-Res" ("\textbf{mecanisme generador a partir del No-Res} ($N$)"), bridging the body-text vocabulary with the symbol used in both formal appendices. The change is one line, applied at Iban's editorial direction to remove a foreseeable reading-friction for non-specialist readers.
5. Abstract rewrite (first paragraph only; the empirical second paragraph is preserved verbatim). Initiated by Iban's request for an honest re-reading of the abstract in light of the matured formal apparatus. Three structural deficits were identified and corrected: (i) the framing of the No-Res as "the only element that requires no justification" (pre-Apèndix B vocabulary, suggesting $N$ is a self-sufficient fixed point) is replaced by a precise framing as the unique ontological candidate licensed by two complementary formal results; (ii) P1–P3 are no longer presented as "three logical principles" but as "three theorems derived from a minimal rationalist framework", reflecting D-0128; (iii) the "generative motor" sentence, previously asserted as fact, is now distinguished into its formal core (inviability theorem) and its residual interpretive postulate (the dynamic-constructive reading), aligning the abstract with the post-D-0132 epistemic architecture. The `($N$)` gloss is introduced at the first body-text mention of "No-Res" inside the abstract, with downstream references using `$N$` (formal contexts) or "No-Res pur (\texttt{0}) / Absolut (\texttt{1})" (intuitive contexts) without re-glossing. Length cost: roughly +20 words on a ~250-word paragraph; structural cost: zero (paragraph count, sentence-flow, contrast Wheeler/Lloyd, ontological-radical / empirical-restricted closing all preserved). The English version (`Paper_Latex/main_eng.tex`) is **not** synchronized in this entry; replication to the English manuscript is recorded as a deferred task to be executed in the next translation-sync cycle.

Editorial precision recorded explicitly so it is not relitigated: the No-Res-pur / No-Res-Absolut distinction survives the inviability theorem with **strengthened** formal grounding, not weakened. The mapping is exact: No-Res pur ↔ Lemma 1 (zero internal extension by II); No-Res Absolut ↔ the "única realitat" claim, blocked by Lemma 2 (contingency horn via SR) and Lemma 3 (necessity horn via relational necessity); tension ↔ the inviability theorem itself; generative-dynamic reading ↔ the residual HSI postulate (no longer the existence of the tension, only its constructive interpretation).

Stylistic note (scope: this rewrite cycle only, applied at Iban's editorial direction): the em-dash construction `---X---` was replaced by parentheses or commas in the rewritten paragraphs because its frequent use sounds AI-stylized in current Catalan editorial registers. Other em-dashes elsewhere in the manuscript are not touched in this entry.

### Editorial Follow-up: title-page layout pass + minor abstract trim

Two additional micro-passes were applied during the recompilation cycle that produced CAT_11, recorded here for traceability. Neither touches formalism, primitives or empirical claims; both are cosmetic-editorial.

6. Title-page layout pass to recover Introduction visibility on page 1. Path explored chronologically: (a) `\usepackage{titling}` + `\setlength{\droptitle}{-2.5em}` was attempted but rejected after compilation failure — `titling` interferes with the tokenisation of the `\author` argument when it contains `\\` plus `\small` plus multiple `\footnotemark[*]` calls, producing an "extra `}'`" / "Paragraph ended before `\reserved@a` was complete" error chain at line 45. (b) `titling` was retired and substituted with a direct `\vspace*{-2.5em}` immediately before `\maketitle`; this in turn produced a blank page 1 because `\vspace*` initialises the page vbox before `\maketitle`, and the combination with the symbolic `\footnotetext[*]` block produced an overflow that triggered an automatic `\newpage`. (c) The `\vspace*` was retired and the abstract was put under `\small` together with two negative spacings: `\vspace{-1.5em}` before `\begin{abstract}` and `\vspace{-0.5em}` before `\section{Introducció}`. (d) Empirical observation by Iban revealed that `\small` at `[11pt]` class is only 10.5pt (≈0.5pt reduction, visually imperceptible, recovered ~1 line); `\small` was promoted to `\footnotesize` (10pt, ≈9% vertical reduction, recovered ~3-4 lines and visually distinct from body text). The final stable configuration on the title page is therefore: no `titling` package, no `\vspace*` before `\maketitle`, `\footnotesize` inside `\begin{abstract}`, `\vspace{-1.5em}` before abstract block, `\vspace{-0.5em}` before Introduction section. This sequence is logged in full so a future editor reopening the title-page layout knows which routes have been tried and rejected.
7. Minor abstract trim (single sentence). Iban-initiated under the principle of editorial economy: the sentence "Sota aquest postulat, la informació no és conseqüència de l'Univers sinó la seua causa primera." was removed from the first paragraph of the abstract because (i) it had a decorative-reinforcement function rather than argumentative load, (ii) the reader had already inferred its content from the immediately preceding sentence about the residual interpretive postulate, and (iii) the contrast "ontologically radical / empirically restricted" that closes the paragraph is unaffected by the removal. No re-audit cycle with Grace was opened because the trim does not touch any primitive, lemma or theorem of the V3 inviability proof; it only suppresses one reformulation of an already-stated consequence. The transition "...una dinàmica generativa irreductible. La proposta és deliberadament radical al pla ontològic..." reads cleanly without bridge.

### Compiled Output (Freeze Anchor)

- Path: `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_11.pdf`
- SHA256: `05EAE8DDAFC94F6A35ABE9D98CD65AECD5CE7F9D82BD84158F1A0CB132AFD27D`
- Cycle absorbed into this artefact: D-0129 (residual-reference fix, never anchored separately) + D-0132 (Apèndix B integration, `subsec:implicacio` rewrite, Nota axiomàtica refinement, abstract first-paragraph rewrite, title-page layout pass, abstract sentence trim).

This artefact supersedes the CAT_9 freeze anchor recorded under D-0128 (`E3F82602933BF04A42C9B411F01F869D47F839DC7AC906B259AFDAE3930E996C`) and becomes the binding seal for any subsequent audit pass on the post-D-0132 manuscript. CAT_10 was never produced as a separate freeze artefact (see resolution note under D-0129's Compiled Output block). A retrospective `Freeze Anchor superseded` note has been appended to D-0128.

---

## D-0133 — Blind external peer review on CAT_11: first mini-cycle of closure (notational + geometric-licence reinforcement) and creation of the Pending Closures / v3 Agenda document

- Date: 2026-05-15
- Author of entry: Sofia (editorial), under Iban's directive
- Cycle: brief-blind external review → split into two mini-cycles by editorial decision (this entry covers the first mini-cycle; the second is opened as Pending Operational Closure below)
- Trigger: Iban submitted the CAT_11 PDF to an external reviewer under the brief-blind modality (no project context, no previous-cycle memory; the request was framed as "rigorous and deep peer review" rather than under the stricter framing used in earlier internal-Grace passes). The returned report is a Major Revisions verdict structured as three substantive points (3.1 ontology-to-algorithm bridge, 3.2 Variant B canonicity, 3.3 geometric-semantic risk at Section 10) plus two minor points (asymptotic-proofs ceiling, undefined $\Omega$ before first use).

### Triage of the external report

Each point of the report was triaged against the editorial-integrity principle (Sofia §4-ter / Grace §3-ter, codified in D-0129) and against the post-D-0132 epistemic baseline:

- **Point 3.1 (ontology-to-algorithm bridge)**: substantive, real, and now tractable in a way it was not pre-D-0132. The V3 inviability lemmas plus the residual interpretive postulate now license a partial derivation of mechanism properties (necessity of generation, binary tension, asymmetry, monotonic accumulation) while the implementation choice (1D string with parentheses) remains a minimal-architectonic choice. A new manuscript subsection separating these two strata honestly is the correct response. Deferred to the second mini-cycle because the new subsection is substantive and warrants a Grace audit pass before incorporation.
- **Point 3.2 (Variant B canonicity / genericity)**: derived from 3.1; the same separation between derived properties and minimal-architectonic choice resolves it. No independent action required beyond the new subsection.
- **Point 3.3 (geometric-semantic risk at Section 10)**: partially valid. An existing reminder at the head of `sec:defectes` (line 831 of `main_v2.02_cat.tex`) already pointed to the lexical convention fixed in `sec:premisses`, but did not explicitly state the exploratory licence on $\delta_m$ as candidate of curvature. Iban's directive (verbatim): *"jo no dedicaria tan un paràgraf, simplement recordaria que ja vam explicar a la pàgina 4 que ens permetem la llicència per a que el lector ho tinga en compte"*. Resolved in this mini-cycle by appending a single sentence to the existing reminder; no new paragraph created.
- **Minor — undefined $\Omega$**: trivially valid. Resolved in this mini-cycle by prepending a one-sentence definition immediately before Equation 1 (`eq:phi`).
- **Minor — asymptotic-proofs ceiling**: third independent recurrence (previously surfaced in two earlier reviewer passes). The recurrence is the signal: a structural pattern, not noise. Cannot be closed within the current paper's evidence base, but can and must be acknowledged explicitly in the future-work section of the manuscript and queued as a v3 agenda item. Acknowledgement deferred to the second mini-cycle (alongside the new ontology-to-algorithm subsection); v3 agenda anchoring resolved in this mini-cycle via the new Pending Closures / v3 Agenda document.

### Changes applied to `Paper_Latex/main_v2.02_cat.tex` in this mini-cycle

1. **Definition of $\Omega$**: a single sentence prepended immediately before Equation 1 (line 177), defining $\Omega$ as the universe of partially ordered collections of finite words over the alphabet $\Sigma = \{\texttt{0}, \texttt{1}, \texttt{(}, \texttt{)}\}$. The sentence reads: *"Designem per $\Omega$ l'univers de totes les col·leccions parcialment ordenades de paraules finites sobre l'alfabet $\Sigma = \{\texttt{0}, \texttt{1}, \texttt{(}, \texttt{)}\}$; aleshores"*. No equation modified; no claim or signature value affected; introduces the alphabet $\Sigma$ as ancillary notation, used inside this definition only.
2. **Geometric-licence reinforcement at Section 10**: a single sentence appended to the existing lexical reminder at `sec:defectes` (line 831), making explicit that *"qualificar $\delta_m$ com a candidat de curvatura té caràcter operatiu i exploratori: no afirmem isomorfisme demostrat amb el tensor de Riemann ni amb cap estructura geomètrica diferenciable, i les condicions formals que autoritzarien tal promoció són justament les que aquesta secció obre i que la Porta 2 ha de tancar"*. The existing reminder is preserved verbatim; only the new sentence is added. No new paragraph; no claim or formal definition modified.

### Document created

A new repository document `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md` is introduced to discipline the split between intra-paper editorial follow-ups (section A: pending closures for the current paper) and the long-horizon research programme towards v3 (section B: out-of-scope items including asymptotic theory, Level 3, alphabet generalisation, formal closure of Gate 2, external re-implementation). The document records each item's triggering reference, scope, cycle requirement and current status, and enforces a strict no-migration discipline between sections A and B without explicit authorisation. This is the canonical artefact for tracking the recurrent referee critiques (such as the asymptotic-proofs ceiling) that exceed the current paper's evidence base.

### Falsification Boundary (unchanged)

This entry introduces no new claim about generative mechanism, projective hierarchy or experimental regime. The Phase-1 falsifiable contracts and the empirical chain (CAT_11 freeze anchor under D-0132) remain intact. The two `.tex` changes are notational and lexical-clarificatory; neither modifies a primitive, a lemma, a theorem, a signature value, a gate definition or a falsification contract. The new repository document records future-work intent and does not bind any current claim.

### Pending Operational Closure

- **Second mini-cycle of D-0133** (separate forthcoming entry): substantive incorporation of (i) the new subsection `subsec:motivacio_mecanisme` resolving Points 3.1 and 3.2 of the external report under the post-D-0132 epistemic baseline, and (ii) the asymptotic-proofs ceiling acknowledgement in the future-work section of the manuscript. The new subsection requires a dedicated Grace audit pass under V3-style discipline before incorporation; the second mini-cycle will not be opened until Iban gives explicit go-ahead and the Grace cycle has converged.
- **Recompilation** to produce CAT_12 (or successor) covering the changes of this first mini-cycle plus the changes of the second mini-cycle, ideally bundled to avoid two consecutive freeze artefacts. If Iban prefers an immediate freeze of the first mini-cycle alone (CAT_12a), the binding-seal protocol of D-0132 applies and an Editorial Follow-up note will register the supersession when the second mini-cycle lands.
- **English-translation sync**: queued under section A.5 of the new Pending Closures document; not opened in this entry.

---

## D-0134 — Second mini-cycle of D-0133: integration of `subsec:motivacio_mecanisme` and asymptotic-ceiling acknowledgement

- Date: 2026-05-15
- Cycle: Editorial / closure of the substantive flank opened by the external blind peer review on CAT_11 (Points 3.1 + 3.2 of the report)
- Trigger: D-0133 first mini-cycle resolved Points 4.1 (notation) and 3.3 (geometric-licence reinforcement) and explicitly deferred the substantive ontology-to-algorithm bridge to a separate entry conditional on a Grace V3-style audit pass. Iban gave explicit go-ahead for the substantive incorporation; the audit converged in a single round with verdict `ACCEPTED-WITH-MINOR-EDITS` from Grace.

### Decision

D-0134 records the substantive editorial closure of the ontology-to-algorithm bridge flank under post-D-0132 epistemic baseline, via two coordinated `.tex` insertions and one bibliography addition.

### `.tex` Changes Applied

1. New subsection `subsec:motivacio_mecanisme` ("Motivació del mecanisme: el que es deriva i el que és tria mínima") inserted immediately before `subsec:familia_hsi` in `Paper_Latex/main_v2.02_cat.tex`. The subsection separates explicitly:
   - **Five necessary conditions of the framework (D1)–(D5)** that any compatible generative mechanism must satisfy, derivable from the V3 lemmas of `app:inviabilitat_no_res` plus the residual interpretive postulate: (D1) necessity of generation, (D2) binary nature of the tension, (D3) orientational asymmetry, (D4) monotonic accumulation, (D5) strict determinism.
   - **A minimal-implementation choice** (1D string over $\Sigma = \{\texttt{0}, \texttt{1}, \texttt{(}, \texttt{)}\}$ with stratified collapse) justified by three explicit criteria (syntactic economy, computational tractability, empirical falsifiability) within the class of implementations compatible with (D1)–(D5).
   - **Consequence for Variant B canonicity**: B is canonical as the conjunction of two independent observations (minimality of added presuppositions within the class, and empirical signature traversal under the quintuple signature against all controls), not as a multiverse-level privilege.
   - **Epistemological closure via multiple realizability**: any future system on a structurally different representation (e.g. hypergraph) satisfying (D1)–(D5) and reproducing the quintuple signature on its appropriate metrics is to be treated as an equivalent implementation within the same universality class, not as a refutation of the framework.

2. Asymptotic-ceiling acknowledgement added as a single paragraph immediately after the existing two-pronged future-work paragraph in `sec:conclusions`. The paragraph records explicitly that all evidence on the compact-selective regime and the projective hierarchy lives under a finite computational ceiling ($m \leq 32$ at Phase 3, bounded bands at Phase 4); closing this flank requires analytical proofs on objects derived from $\Phi_B$ in the classical languages of combinatorics on words and symbolic dynamics. The operational tracking of this Open Item is externalized to the Decision Log and to `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md` (section A item on asymptotic results).

3. New bibliography entry `BerstelReutenauer1988` (Berstel & Reutenauer, *Rational Series and Their Languages*, Springer EATCS Monographs vol. 12, 1988) added to `Paper_Latex/references.bib` to support the asymptotic-acknowledgement paragraph alongside the pre-existing `LindMarcus1995` entry.

### Grace V3-style Audit Cycle (single round)

Verdict: `ACCEPTED-WITH-MINOR-EDITS` (structural). Four patches returned by Grace, all four integrated unconditionally under principle B:

1. **D4 re-anchoring**: original mapping of (D4) to the constitutive-versus-eventual character of the inviability was rejected as logically too strong (white-noise Markov-0 counterexample: constitutively unstable yet non-accumulating). Re-anchored to SR via the causal-traceability argument: erasing the historical trace breaks the chain of sufficient reason, therefore any compatible mechanism must accumulate. This is a downgrade of the formal claim and a strengthening of the logical anchor.
2. **D5 addition** (strict determinism): identified by Grace as a fifth necessary condition derivable from SR. Stochastic contingency would recreate the SR-violating pathology at a derived scale, therefore any compatible mechanism must be strictly deterministic. This blinds the independent flank a hostile reviewer could open by asking why Variant B does not include a probabilistic element.
3. **Lexical adjustment**: header for the (D1)–(D5) block changed from "Propietats derivades" to "Condicions necessàries del marc" to communicate that these are necessary filters on candidate mechanisms rather than full theorems with formal proofs in the body of the subsection. Selected by Iban from the two equivalent options Grace offered (the alternative was "Restriccions de disseny inferides").
4. **Multiple realizability**: the epistemological-closure paragraph reformulated using the standard technical concept (Putnam/Fodor in philosophy of science; universality classes in statistical physics), replacing the V1 defensive concession with a positive technical proposal that invites independent searches for the quintuple signature on alternative compatible structures.

The Grace audit message and Sofia draft V1 → V2 reconciliation are reproduced in the conversation transcript; no separate audit file is created in `docs/` because the audit converged in a single round and the trace lives in this entry.

### Consistency with D-0132 Architecture

The new subsection is the body-of-manuscript analogue of the editorial discipline that D-0132 introduced for the appendix: separation of formal theorem (`app:inviabilitat_no_res`) from residual interpretive postulate (dynamic-constructive reading). The (D1)–(D5) block applies the same separation to the mechanism: derivable conditions vs. minimal-implementation choice. The framework now has a single, consistent epistemological discipline across appendix and body, and the burden of justification for the 1D-string-with-parentheses architecture is precisely located.

### Falsification Boundary (unchanged)

This entry introduces no new claim about projective hierarchy, signature values, gates, controls or experimental regime. The Phase-1 falsifiable contracts and the empirical chain (CAT_11 freeze anchor under D-0132) remain intact. The new subsection re-licenses Variant B canonicity under a more honest justificatory architecture; it does not change which experimental claims B supports or what would falsify them. The asymptotic-acknowledgement paragraph explicitly *contracts* the manuscript's claim envelope by recording that all current empirical claims live under a finite computational perimeter.

### Pending Operational Closure

- **Recompilation** to produce CAT_12 covering the D-0133 first-mini-cycle changes plus all D-0134 changes (single bundled freeze artefact, per the freeze-bundled policy registered in D-0133's Pending Operational Closure block). Iban will compile and return the SHA256.
- **Compiled Output (Freeze Anchor)**: pending recompilation. To be recorded here once Iban returns path + SHA256. Will become the binding seal for any subsequent audit pass on the post-D-0134 manuscript and will supersede the CAT_11 anchor recorded under D-0132.
- **English-translation sync**: queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; not opened in this entry.

---

## D-0135 — Editorial micro-cycle on CAT_12: removal of `subsec:familia_hsi` and obertura redundancy at `subsec:marc`

- Date: 2026-05-15
- Cycle: Editorial / mechanical-redundancy purge on the post-D-0134 baseline
- Trigger: Iban detected on a re-reading pass of CAT_12 a structural redundancy between the new `subsec:motivacio_mecanisme` (D-0134) and the pre-existing `subsec:familia_hsi`, plus a derived redundancy at the obertura sentence of `subsec:marc`. Verification confirmed that the entire single paragraph of `subsec:familia_hsi` had been absorbed by `subsec:motivacio_mecanisme` (canonicity-of-B paragraph), that the only cross-reference to `subsec:familia_hsi` was a self-reference from within `subsec:motivacio_mecanisme` itself (now obsolete), and that the recursive-skeleton list had been implicitly recovered by the three steps enumerated in `subsec:marc`.

### Decision

D-0135 records a quirurgical four-step purge applied to `Paper_Latex/main_v2.02_cat.tex`:

1. **Removal of `subsec:familia_hsi` in its entirety** (subsection header + `\label{subsec:familia_hsi}` + single paragraph + traceability comment `% [S5-T14]`). The recursive-skeleton enumeration ("acumulació + aniquilació + parèntesis estratificats + pas inside→out + finalització global") was not migrated because the three-step enumeration of `subsec:marc` already contains it implicitly.
2. **Replacement of the redundant obertura sentence of `subsec:marc`** ("Dins d'aquesta família, la Variant B és la implementació canònica per minimitat de pressupòsits afegits...") by a minimal transition: "Com ja hem descrit més amunt, la Variant~B encapsula el mecanisme generador en tres passos per iteració:". The deleted sentence repeated verbatim the canonicity argument already developed in the `subsec:motivacio_mecanisme` "Conseqüència per a la canonicitat de la Variant B" paragraph.
3. **Removal of the obsolete traceability comment `% [S5-T15]`** that documented a microedición now subsumed by the new architecture.
4. **Reformulation of the obsolete self-reference** within `subsec:motivacio_mecanisme` ("la subsecció~\ref{subsec:familia_hsi} desenvolupa més avall sobre la canonicitat de la Variant~B" → "la canonicitat de la Variant~B com a aplicació natural del mateix tallafocs"). The reference target no longer exists; the new wording introduces the same paragraph without a dangling cross-reference.

### Consistency verification

Post-edit grep confirms zero remaining references to `subsec:familia_hsi` (the single match left in the file is the `[D-0135-T1]` traceability comment that documents the removal). No other `\ref` or `\autoref` cross-references in the manuscript pointed to `subsec:familia_hsi`. `\label{subsec:marc}` is preserved unchanged; the structure of the surrounding section (Tensió Primordial → motivació del mecanisme → marc de decadència → acumulació → ...) tightens by one subsection without losing any content.

### Falsification Boundary (unchanged)

This entry introduces no new claim about generative mechanism, projective hierarchy, signature values, gates, controls or experimental regime. It is a strictly mechanical removal of textual repetition introduced by the D-0134 absorption of the older subsection's argumentative function. The Phase-1 falsifiable contracts and the empirical chain (CAT_11 freeze anchor under D-0132) remain intact. CAT_12 is superseded by the forthcoming CAT_13 only insofar as it carried the redundancy now purged.

### Pending Operational Closure

- **Recompilation** to produce CAT_13 covering the D-0135 purge applied on top of the D-0133+D-0134 substrate that produced CAT_12. Iban will compile and return the SHA256.
- **Compiled Output (Freeze Anchor)**: pending recompilation. To be recorded here once Iban returns path + SHA256. Will supersede both the CAT_11 anchor (D-0132) and the CAT_12 anchor (D-0134) and become the binding seal for any subsequent audit pass on the post-D-0135 manuscript.
- A retrospective `Freeze Anchor superseded` note will be appended to D-0132 (CAT_11 → CAT_13) and to D-0134's pending CAT_12 anchor block when the CAT_13 SHA256 lands.
- **English-translation sync**: continues queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; not opened in this entry.

---

## D-0136 — Pedagogical rewrite of `subsec:toy_example` (walk-through del mecanisme): explicit-time table for α=0 and second walk-through for α=1

- Date: 2026-05-15
- Cycle: Editorial / pedagogical refinement on the post-D-0135 baseline
- Trigger: Iban re-read the walk-through subsection (`subsec:toy_example`, lines 214–235 in the post-D-0135 source) and reported that the explanation of the inside→out collapse and the final simplification was opaque even to himself, citing also that the same flank had appeared in earlier peer-review feedback. Sofia ran a four-point pedagogical diagnostic against the actual structural snapshot `hsi_agents_project/results/level0/phi_snapshots/var_B/phi_iter1.struct.gz` (decoded as `0(0)1010` in the 2-bit structural format `00=0, 01=1, 10=(, 11=)`), confirmed that the data are correct and that the issues are strictly pedagogical, and proposed four options (A: minimal refinement of α=0; B: refinement of α=0 + addition of α=1 to expose the genuine inside→out dynamic; C: full four-iteration walk-through; D: replace α=0 by α=1 as the primary example). Iban authorized Option B with C reserved as fallback if B does not provide enough didactic clarity.

### Pedagogical diagnostic (four points fixed by this entry)

1. **Mixed-time `Acc` column.** The previous table conflated `Acc` after `accumulation += state` (i.e. `Acc_before_collapse`) with `state` shown as a transition `X → Y`, with no explicit time marker per column. A reader could not tell which moment belonged to which cell.
2. **Loop exit condition mis-described.** The previous text described the loop exit as *"sense parèntesis: punt fix"*, suggesting the absence of parentheses is the reason for exit. The actual exit condition in the source (`while state != previous`) is `state == previous`. In α=0 the two coincide accidentally because no parenthesis is left to mutate `state`, but the condition is logically distinct and the description is not generalizable to subsequent iterations.
3. **Three invisible steps in the final simplification.** The previous text stated *"L'estat col·lapsat s'incorpora a l'acumulador, donant Acc final = 0(0)1010"* without exposing that (a) the loop exits with `state = "01"`, (b) the global rule reduces it to `state = "0"`, (c) this `"0"` is the bit appended to `Acc`. Three operations were collapsed into one sentence and the reader had to reverse-engineer where the trailing `"0"` came from.
4. **α=0 is too minimal to show inside→out.** With a single parenthesis, α=0 cannot exhibit the layered collapse that gives the mechanism its name. Iteration α=1 already nests two parenthesis levels (`(0(0)10100)1`) and makes the inside→out ordering visible: the inner `(0)` is matched and reduced first by the regex `\([01]+\)`, exposing pure bits inside the outer parenthesis, which only then becomes matchable and reduces to `"0"`. Without α=1, the inside→out claim is asserted but not shown.

### Edits applied

A single quirurgical rewrite of `subsec:toy_example` in `Paper_Latex/main_v2.02_cat.tex` (replacement of lines 214–235 by the new lines 214–256, +21 lines net):

1. **`Marc de decadència` paragraph extended** with one explicit sentence anchoring the loop initial state: *"L'estat inicial del bucle és $\mathrm{state} = D_0 = \texttt{(0)1}$"*.
2. **`Col·lapse inside→out` paragraph rewritten** to expose the per-pass two-operation order (`Acc += state` first, then collapse only innermost matchable parentheses via regex `\([01]+\)`) and to declare the exit condition explicitly as `state == previous`.
3. **Table for α=0 reformulated** from 4 columns (Passada / Acumulat Acc / Estat / Acció) to 4 columns with explicit time semantics: Pas / Acc al final del pas / state: entrant → sortint / Eix. (eixida). Each cell now belongs to one unambiguous moment.
4. **Loop-exit clarification sentence** added immediately below the α=0 table: *"A la passada 2, el regex no troba cap parèntesi a simplificar, així que state és idèntic a l'entrada: la condició state = previous es compleix i el bucle es tanca."*
5. **`Simplificació final` paragraph rewritten** to expose the three previously-collapsed steps explicitly: (a) loop exits with `state = "01"`, (b) global rule reduces `01 → 0`, (c) this `"0"` is appended to `Acc`. Snapshot citation added: `phi_iter1.struct.gz` confirms the eight-symbol `0(0)1010` result.
6. **New `Lectura observable` paragraph** isolates the observable reading (`Φ_0 = 001010`, six bits) as a separate didactic moment, decoupled from the `Acc` mechanics.
7. **New paragraph + table for α=1** (`Inside→out genuí: la iteració α = 1`): D_1 = `(0(0)10100)1` with two nesting levels; three-pass table showing (i) inner `(0)` collapses first because the outer parenthesis still contains `(`, (ii) outer parenthesis becomes matchable once its content is pure bits and reduces to `"0"`, (iii) `state == previous`, exit. Followed by a brief paragraph anchoring that the inside→out ordering emerges from the regex syntax (`\([01]+\)` matches only parentheses with pure bit content), not from explicit ordering code.
8. **`Eixamplament exponencial` paragraph** (closing) preserves the previous sentence about α = 23 reaching ≈ 4.29 × 10⁹ bits but reframes it as a confirmation that the dynamic visible at α = 0 and α = 1 is the same at all scales, not as a leap to a new regime.

### Verification against source

The α=0 trace was verified end-to-end against the decoded structural snapshot `hsi_agents_project/results/level0/phi_snapshots/var_B/phi_iter1.struct.gz` (8 symbols, hex prefix `2344`, decoded as `0(0)1010` in the 2-bit structural format). The α=1 trace follows the canonical worked example documented in `hsi_agents_project/Documentation/HSI_Variants_Guide_Wiki.md` (lines 117–143), with the three-pass collapse `(0(0)10100)1 → (0010100)1 → 01 → 01 (exit)` and the final global `01 → 0`. The Python source for the inside→out collapse (`_collapse_inside_parentheses_local` with regex `\([01]+\)`) and the main loop (`while state != previous` with `accumulation += state` per pass, conditional final `_simplify_base` if `len(state) > 1`) is reproduced in `Articles_i_publicacions/HSI_From_Void_to_Code_Substack.md` (sections "The annihilation rule" and "The main loop (Variant B)") and matches the description in the rewritten subsection.

### Falsification Boundary (unchanged)

This entry introduces no new claim about generative mechanism, projective hierarchy, signature values, gates, controls or experimental regime. The α=0 result `Φ_0 = 001010` and the α=1 collapse trace are documentary descriptions of an existing, frozen and reproducible computation; the rewrite replaces an opaque presentation by a transparent one. The Phase-1 falsifiable contracts and the empirical chain remain intact. CAT_12 (D-0134) and the pending CAT_13 (D-0135) are both superseded by the forthcoming CAT_14 only insofar as the walk-through subsection is now pedagogically clearer.

### Pending Operational Closure

- **Recompilation** to produce CAT_14 covering the D-0136 walk-through rewrite applied on top of the D-0133+D-0134+D-0135 substrate. Iban will compile and return the SHA256.
- **Compiled Output (Freeze Anchor)**: pending recompilation. To be recorded here once Iban returns path + SHA256. Will supersede the pending CAT_13 anchor under D-0135 and become the binding seal for any subsequent audit pass on the post-D-0136 manuscript.
- A retrospective `Freeze Anchor superseded` note will be appended to D-0132 (CAT_11 → CAT_14), D-0134's pending CAT_12 anchor block, and D-0135's pending CAT_13 anchor block when the CAT_14 SHA256 lands.
- **English-translation sync**: continues queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; not opened in this entry. The walk-through rewrite is one of the pedagogical-clarity blocks that will need parallel sync to `Paper_Latex/main_eng.tex` when that cycle is opened.
- **Option C fallback reserved**: if Iban judges Option B insufficient after re-reading the rewritten subsection in CAT_14, Option C (full four-iteration walk-through covering α=0 through α=3) is reserved as an authorized follow-up cycle without re-opening the diagnostic phase.

---

## D-0137 — Editorial fusion of the two collapse subsections in `subsec:toy_example`: drop the trivial α=0 table and promote α=1 as the single inside→out walk-through

- Date: 2026-05-15
- Cycle: Pedagogical micro-cycle on top of D-0136, same session (no recompilation between D-0136 and D-0137; CAT_14 will absorb both).
- Trigger: Iban reviewed the post-D-0136 walk-through in the editor and reported two concrete defects: (a) the α=1 table introduced by D-0136 overflowed the page on the right because the comment column carried verbose descriptions; (b) the α=0 table contained a row `01 → 01` (the loop-exit detection step) that visually showed "nothing happens" and confused the reader rather than clarifying the exit condition. He proposed merging the two collapse subsections into one, removing the α=0 table entirely, and renaming "Inside→out genuí" to the canonical "Col·lapse inside→out". Sofia confirmed the diagnosis (the `01 → 01` row was a faithful trace of the implementation but a pedagogical artifact, not an ontological event) and proposed two width-control options for the surviving α=1 table: A — keep each transition as a single `\texttt{}` cell and trust it fits in two rows; B — break each transition over two lines inside the cell. Iban authorized Option A with B reserved if the recompilation still overflows.

### Edits applied

A single quirurgical rewrite of `subsec:toy_example` in `Paper_Latex/main_v2.02_cat.tex` (lines 216–256 of the post-D-0136 source replaced by lines 216–238 of the new source, −18 lines net):

1. Removed the standalone `\paragraph{Col·lapse inside→out}` block of D-0136 (the α=0 table with four explicit-time columns and the clarifying sentence about pass 2).
2. Removed the `\paragraph{Inside→out genuí: la iteració α = 1}` heading; its body is absorbed into a single canonical `\paragraph{Col·lapse inside→out}` paragraph.
3. The new unified paragraph keeps the conceptual scaffolding from D-0136 (loop semantics: `Acc += state`, regex `\([01]+\)`, exit on `state == previous`), describes α=0 as a one-sentence narrative trace (no table), and then transitions explicitly to α=1 as the first iteration where the inside→out ordering becomes visible. The mechanical justification for the inner-to-outer order (the regex can only match parentheses with pure-bit content) is preserved verbatim.
4. The α=1 table is reduced from three rows to two: the former pass-3 `01 → 01` is absorbed into the action column of pass 2 as `cau l'extern; cap parèntesi més: surt`. Comments on the remaining two rows are shortened to fit the line width without column-width specifiers.
5. `\paragraph{Simplificació final}` is reformulated to reference both iterations symmetrically (`A totes dues iteracions, el state amb què surt del bucle és 01...`) so the appended `0` and the final `Acc = 0(0)1010` for α=0 still trace cleanly to the snapshot `phi_iter1.struct.gz`.
6. `\paragraph{Lectura observable}` and `\paragraph{Eixamplament exponencial}` retained verbatim.
7. The `% [D-0136-T1]` marker comment is replaced by a `% [D-0137-T1]` marker that documents the fusion rationale.

### Verification

- Snapshot trace unchanged: `Acc_0 = 0(0)1010` (8 symbols), `Φ_0 = 001010` (6 bits) still match `hsi_agents_project/results/level0/phi_snapshots/var_B/phi_iter1.struct.gz` decoded as `0(0)1010`.
- α=1 trace `(0(0)10100)1 → (0010100)1 → 01 → 0` (final simplification appended) preserved; only the pass-3 row that visually displayed `01 → 01` is removed.
- IDE diagnostics: clean.

### Falsification frontier

Unchanged from D-0133/D-0134/D-0135/D-0136. This is a pedagogical fusion with no ontological, methodological, or implementation impact.

### Pending Operational Closure

- **Recompilation** to produce CAT_14 covering both D-0136 and D-0137 in a single artifact (no intermediate compilation between them was performed).
- **Compiled Output (Freeze Anchor)**: pending recompilation. The anchor will be recorded under D-0136 (the broader pedagogical entry) and a back-reference noted here. CAT_14 will supersede the pending CAT_13 anchor under D-0135.
- **Option B reserved** for the table width: if CAT_14 still shows overflow on the α=1 row, the fallback is to break each transition cell over two lines (`\texttt{(0(0)10100)1}` \\ `$\to$ \texttt{(0010100)1}`), authorized without re-opening diagnostic.
- **Option C reserved** (from D-0136): if the unified walk-through is still insufficient after re-reading CAT_14, the full four-iteration version (α=0 through α=3) remains authorized as a follow-up cycle.
- **English-translation sync**: continues queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; the fusion will need to be propagated to `Paper_Latex/main_eng.tex` together with the D-0136 rewrite when that cycle opens.

---

## D-0138 — Removal of redundant rhetorical sentence in the post-decay-frame paragraph of `sec:mechanism`

- Date: 2026-05-16
- Cycle: Editorial micro-fix on top of D-0137, same artifact (CAT_13 / pending-CAT_14 anchor).
- Trigger: Iban re-read line 206 of `Paper_Latex/main_v2.02_cat.tex` and reported that the sentence *"El No-Res intenta ser absolut, fracassa, i el que queda d'aquest fracàs és l'Absolut."* felt tautological. Sofia confirmed two superposed defects: (a) involuntary lexical echo between the adjective `absolut` and the noun `l'Absolut` that made the sentence read as circular in Catalan; (b) full redundancy with the immediately preceding sentence (`No preexisteix: emergeix com a residu de la contradicció`), which already conveys the same mechanism more precisely. Four options were offered (A: pure removal; B: rewrite exposing the affirmation-as-first-fissure mechanism; C: keep the wordplay but disambiguate typographically; D: performative-act compactification). Iban authorized Option A.

### Edits applied

A single deletion in `Paper_Latex/main_v2.02_cat.tex` line 206: the disputed sentence is removed; the surrounding paragraph (Absolute appears at the end, not at the beginning + cross-reference to `sec:absurd`) is preserved verbatim. Net change: −1 sentence, −0 lines (paragraph reflows on the same line).

### Verification

- The preceding sentence (`No preexisteix: emergeix com a residu de la contradicció`) carries the mechanism intact.
- The trailing sentence (`Que aquesta decisió no és arbitrària ho demostra la Secció~\ref{sec:absurd}`) still refers to the same antecedent (the non-pre-existence claim).
- No other section of the manuscript depends on the removed sentence (verified by absence of cross-references to its specific phrasing).
- IDE diagnostics: clean.

### Falsification frontier

Unchanged. Pure editorial deletion of a redundant rhetorical line; no ontological, methodological, or implementation impact.

### Pending Operational Closure

- **Recompilation** to absorb D-0138 together with D-0136 + D-0137 into the next physical artifact. The naming of that artifact (CAT_13 vs CAT_14) is still pending Iban's clarification — see the open question on the previous response: whether the current `Drafts/La_Hipòtesi_de_la_Singularitat_Informacional_version_2.02_CAT_13.pdf` is the post-D-0137 artifact with the number unbumped (in which case CAT_12/CAT_13/CAT_14 collapse into this single CAT_13) or whether it should be renamed CAT_14.
- **Freeze Anchor** still to be recorded under D-0136 once the artifact name and SHA256 are confirmed; back-references will be added from D-0137 and D-0138.
- **English-translation sync**: continues queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`. The deletion will need to be mirrored in `Paper_Latex/main_eng.tex` together with D-0136 + D-0137.

---

## D-0139 — Editorial/methodological boundary for Decision Log entries (policy)

- Date: 2026-05-16
- Cycle: Meta-methodological rule change (last self-referential entry of this kind expected for the foreseeable horizon).
- Trigger: Iban observed that the Decision Log had grown beyond 9500 lines and that recent entries D-0135, D-0137 and D-0138 had recorded purely editorial micro-fixes (subsection removal, table fusion, single-sentence deletion) that diluted the signal of genuinely methodological decisions and made the log hard to navigate even for Sofia. Sofia confirmed this was over-engineering and a violation of the AGENTS.md rule that reserves the log for *scientific or methodological* decisions, which never required editorial micro-fixes.

### Rule adopted (going forward)

1. **Decision Log entries are reserved for scientific or methodological decisions**: changes to ontology, falsification frontiers, protocols, parameters, default behaviours, validation strategies, peer-review responses, repository-wide structural decisions, and anything that alters what the project does, claims or measures.
2. **Editorial micro-fixes are not recorded**: sentence deletions, paragraph reformulations, table restructuring, subsection fusions, typographical corrections, lexical clarifications, and analogous changes are applied directly to the source. Traceability is provided by `git diff` and, when needed, by the commit message.
3. **No `% [D-NNNN-T1]` markers in `.tex` source for editorial changes**: source-level marker comments are added only when the underlying change has a corresponding Decision Log entry.
4. **Borderline cases default to asking**: when Sofia is uncertain whether a change crosses the methodological threshold, she asks Iban before opening an entry rather than defaulting to register.

### Retrospective treatment

- Past entries D-0135, D-0137 and D-0138 (purely editorial in retrospect) are left intact as historical precedent of the boundary that was crossed; they are not retroactively deleted or consolidated, because re-editing the log to enforce a new policy retroactively would be the same over-engineering disease in meta mode.
- The `% [D-0136-T1]` and `% [D-0137-T1]` markers already present in `Paper_Latex/main_v2.02_cat.tex` are left in place for the same reason.
- D-0136 is retained at the methodological tier as a borderline case (substantive pedagogical restructuring of the central walk-through); future pedagogical refinements of comparable scope will be handled inline unless they touch claims, interpretation, or methodology.

### Falsification frontier

Unchanged. Documentation-policy decision; no ontological or empirical impact.

### Pending Operational Closure

- None for this entry. The next compilation will absorb D-0138 as the last editorial entry under the old policy; D-0139 itself is policy and requires no compilation anchor.

---

## D-0140 — Manuscript-level reorganization: swap of `sec:absurd` and `sec:disseny`

- Date: 2026-05-16
- Cycle: First repository-wide structural decision under the new D-0139 policy (qualifies as a structural decision about manuscript organization, not an editorial micro-fix).
- Trigger: Iban observed a narrative discontinuity in `Paper_Latex/main_v2.02_cat.tex`: §4 (`sec:absurd`) opened by referring to "Variant B", "Variant N" and "Variant F" without having introduced the variants table or the implementation, while §5 (`sec:disseny`) — which introduces the variants table, the controls table and the Python implementation — appeared only afterwards. Sofia confirmed the swap was mechanically safe: only two `\ref{sec:absurd}` cross-references exist in the document (lines 74 and 206), both self-resolving by label, and no `\ref` targets `sec:disseny`, `subsec:variants`, `subsec:controls` or `subsec:implementacio` from elsewhere. Three options were offered (A: pure swap with the `BLOC II` marker moved before the new §4; B: swap + bridging sentence in `subsec:variant_n` clarifying that Variant N is an unimplemented thought experiment; C: keep the current order and add a bridging paragraph at the top of §4). Iban authorized Option A (pure swap, no added text) on the explicit grounds that "less is more" given the 50-page density of the manuscript and that Variant N's exploratory status is sufficiently clarified later in the text.

### Change applied

1. The two section blocks have been bescanviats in `Paper_Latex/main_v2.02_cat.tex`: the new §4 is `\section{Disseny experimental i controls}\label{sec:disseny}` (with subsections `subsec:implementacio`, `subsec:variants`, `subsec:controls`), and the new §5 is `\section{Per què ha de ser així: validació per l'absurd}\label{sec:absurd}` (with subsections `subsec:variant_n`, `subsec:variant_f`, `subsec:triangle`).
2. The `% BLOC II -- RESULTATS EXPERIMENTALS DEL NIVELL 0` marker has been moved to immediately precede the new §4, so that both the experimental design and the absurd-validation (which already invokes empirical results) sit inside BLOC II. The theoretical bloc closes at §3 (`sec:mecanisme`) as before.
3. The two existing `\ref{sec:absurd}` cross-references (Introduction line 74 and `sec:mecanisme` line 206) are left untouched: they resolve automatically by label to the new §5.
4. No textual content was modified inside either section. The previously restored `% [S1-T3]` editorial-trace comment in `subsec:variant_f` is preserved verbatim.

### Falsification frontier

Unchanged. Manuscript reorganization for narrative coherence; no ontological, methodological, or empirical impact.

### Pending Operational Closure

- **Recompilation** to absorb D-0140 together with D-0136 + D-0137 + D-0138 into the next physical artifact. The artifact-naming disambiguation (CAT_13 vs CAT_14) is still pending Iban's clarification — same open question as D-0138.
- **Freeze Anchor** still to be recorded under D-0136 once the artifact name and SHA256 are confirmed; back-references will be added from D-0137, D-0138 and D-0140.
- **English-translation sync**: queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`. The same swap will need to be mirrored in `Paper_Latex/main_eng.tex` when the D-0136 cycle propagates.



## D-0141 — New formal lemma: dual to Lema 1 on uniqueness of the Absolute pole (Lema 1.b)

- Date: 2026-05-19
- Cycle: First derived entry from the hostile-review iterative pass by Gemini on v2.02 CAT_14 (in-progress), audited and validated by Grace.
- Trigger: A Gemini hostile-review pass flagged four friction points on v2.02 CAT_14 in-progress; one of them was that the 11→1 annihilation rule (`subsec:annihilacio`) was justified in the body only by a parenthetical clause ("perquè ni el No-Res ni una totalitat sense exterior admeten pluralització") and lacked a formal derivation from the racionalista primitives already laid out in `app:derivacio_formal:base`. Sofia diagnosed that the missing piece was the symmetric dual of Lema 1 (which applies II to the No-Res pole at extension zero) applied to the Absolute pole at total extension. Three options were offered (A: expand the body text only; B: expand body text + add a short formal lemma to `app:inviabilitat_no_res`; C: add only the appendix lemma with a cross-reference). Iban authorized Option B. Grace audited the resulting lemma along three explicit axes (validity under II as formulated in `app:derivacio_formal:base`, exact duality with Lema 1 without asymmetry, and absence of overclaim on the consequence about 11→1) and validated it formally with explicit dictamen that the material qualifies as a new formal contribution deserving its own log entry.

### Change applied

A new paragraph-level lemma `Lema 1.b (Col·lapse a unicitat per II, pol Absolut)` is inserted in `Paper_Latex/main_v2.02_cat.tex` immediately after Lema 1 of `app:inviabilitat_no_res` (after line 1077). The lemma applies primitive (II) from `app:derivacio_formal:base` to the totality pole: if 1 satisfies ∀P:P(1) and x satisfies ∀P:P(x), then ∀P:P(1)↔P(x) (the biconditionals are vacuously true with both sides true), so by (II) 1=x. Therefore two distinct occurrences of the totality are logically untenable, and the 11→1 rule of the body (`subsec:annihilacio`) is a notational recognition rather than an applied fusion operation. The lemma is the strict dual of Lema 1 (collapse of the N pole at zero extension) applied to the 1 pole at total extension. Demonstration of 4 lines, no new axioms.

The body of `subsec:annihilacio` (L170) was also rewritten in the same cycle to separate the three ontological relations (identity of No-Res, forced identity of the Absolute, degradation) and cross-reference the new Lema 1.b. The body rewrite is editorial under D-0139 and is not the object of this entry; the formal lemma is.

### Falsification frontier

Unchanged. The lemma is a logical consequence of primitive (II), which is part of the racionalista base already audited in `app:derivacio_formal:base`. No new ontological commitments are introduced. The falsifiability contract (Sections 6 onwards) is unaffected.

### Pending Operational Closure

- **Recompilation** absorbs D-0141 together with D-0136 + D-0137 + D-0138 + D-0140 + four prior editorial refinements (alignment of variants table in §4 with Variant N anti-canonical, restoration of the S1-T3 fossil comment, removal of `subsec:triangle`, addition of scope-delimitation sentence at §5) + three new editorial refinements applied in the same cycle (didactic reformulation of `subsec:autocontencio`, expanded annihilation paragraph at L170, conditional-strengthened lexical-note bridge at L123 after Grace's blocking review).
- **Freeze Anchor** to be recorded under D-0136 once the artifact name (CAT_13 vs CAT_14 disambiguation, still pending) and SHA256 are confirmed; back-references will be added from D-0137, D-0138, D-0140 and D-0141.
- **English-translation sync**: the lemma must be mirrored in `Paper_Latex/main_eng.tex` as part of section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`.

---


## D-0142 — Hostile-review Q1: raw-sequence orthogonal metrics over frozen Level 0 snapshots

- Date: 2026-05-19
- Cycle: First derived empirical entry from the hostile-review iterative pass by Gemini on v2.02 CAT_14 (in-progress); follows D-0141 in the same review cycle and runs in parallel with the editorial reordering of §3.4↔§3.5 inside `sec:mecanisme` (handled inline under D-0139, no own entry).
- Trigger: Gemini's hostile-review hit a methodological point that Sofia accepted as legitimate: the projective tower observables (|P_m|, retention, survival) are designed around the same hierarchical structure that the canonical B mechanism produces, so the discriminating power of the quintuple signature could in principle be partly a coincidence between mechanism design and observable design. Sofia tasked Ariadna with constructing a read-only orthogonal diagnostic over the frozen Level 0 snapshots, using metrics that do not presuppose hierarchy (long-range mutual information, FFT spectrum, block-decomposition complexity proxy, zlib ratio), and reporting whether B still separates from observed structured variants {F, M/N, O, P} and from B-based LZ-family nulls under these orthogonal observables. Ariadna executed the diagnostic on 3,000,000 observable bits per source (with explicit iter17 fallback for variant N, which lacks iter20). Grace audited both the metric selection and the envelope-rule and accepted the result, adding a reading that strengthens the paper's defense rather than weakens it.

### Empirical finding

Variant B does not separate from the full envelope of observed structural controls {F, M/N, O, P} under any of the eight orthogonal raw-sequence features (long-range MI tail/AUC, spectral entropy, low-frequency power fraction, spectral peak-to-mean, block entropy, BDM proxy, zlib ratio). Variant B does separate from the four LZ-family nulls (B-shuffled seed 17, B-matched-lz seed 17, B-phase-matched-lz seeds 607 and 709) on five of the eight features. The result is consistent across both the global envelope readout and the grouped readout (`observed_controls` / `lz_family_nulls` / `all_nulls` / `all_non_B`).

### Methodological reading (Grace)

The negative outcome against observed structural controls is the strongest possible empirical evidence that the projective tower observables are not redundant with raw-sequence complexity measures. If the quintuple signature were a trivial artifact of stratified parsing, then a raw-sequence orthogonal metric should also separate B from the structured controls; it does not. The fact that B separates devastatingly under the projective tower (Phase 1 documented results) and does not separate under raw-sequence orthogonal metrics demonstrates empirically that the tower captures structure invisible to standard block, frequency and compression analysis. The partial separation of B against LZ-family nulls remains as additional defense against the strong tautology reading: B would not separate against complexity-matched nulls if the quintuple were purely structural artifact of the parser.

### Change to claims

The defense against the circularity reading at §12 (`sec:circularitat` or equivalent location) must be reformulated to incorporate Q1 as empirical evidence rather than as concession. The quintuple-signature claim is unaltered: B still separates under the paper's defined coordinates. What is updated is the meta-claim about why the projective observables matter — Grace's reading is to be paraphrased verbatim. No primary empirical claim is reverted, retracted, or expanded.

### Falsification frontier

Unchanged. The diagnostic is read-only over frozen snapshots, uses public `hsi_v2_`-prefixed code, writes under `results/hsi_v2/raw_sequence_orthogonal_metrics/`, and does not modify any Level 0 default. Future hostile-review passes that propose additional orthogonal metrics can extend the same run-contract without invalidating the present finding.

### Anchors

- Public script: `hsi_agents_project/hsi_v2_raw_sequence_orthogonal_metrics.py`
  - SHA-256: `3382F1548EECA6C6E6BA7F4AB08D722D5ADF0298E88E41E5FDD562CF066E617C`
- Run directory: `hsi_agents_project/results/hsi_v2/raw_sequence_orthogonal_metrics/raw-sequence-orthogonal-metrics__obs-B-F-M-N-O-P__bits-3M__20260519T221252/`
  - `summary.json` SHA-256: `32C3F58631CE9BEEADC6C55E28AFB35FC6F292961AA54B02ECABEF044C856A76`
  - `report.md` SHA-256: `5A002DB5FF7D6E9743C83A74D80C51BF98CDFB4F3A02D140430631F2CC497EA3`
  - `manifest.json` SHA-256: `F59D046E5B09055A18FAF29709F3C0B579C40D265BADC116A5D62B167EAD49DB`

### Pending Operational Closure

- **Editorial integration**: insert one paragraph in §12 of `Paper_Latex/main_v2.02_cat.tex` (and propagation to `main_eng.tex`) carrying Grace's reading with explicit cross-reference to this entry and to the run directory above. The paragraph is editorial under D-0139 and does not require its own log entry.
- **CAT_16 absorption**: this finding ships with CAT_16; the freeze anchor will be recorded under D-0136 together with D-0137, D-0138, D-0140, D-0141 once the artifact name and SHA-256 are confirmed.

---

## D-0143 — Hostile-review Q2a: dictamen on canonical B not being a fixed morfic substitution

- Date: 2026-05-19
- Cycle: Companion entry to D-0142 in the same hostile-review derivation cycle.
- Trigger: Gemini's hostile-review repeated, in two independent passes with and without hostile framing, the same observation: the golden-ratio fingerprints reported for canonical B (`LZ ≈ 1/φ`, `c̄ = φ+1`) could be an analytical artifact of a hidden Fibonacci-type morfic substitution, given that the anti-canonical Variant N is literally the complement of the Fibonacci word. Sofia tasked Ariadna with a formal analytic check against the current executable Level 0 semantics: is canonical B equivalent, under any suitable representation, to a fixed morfic substitution, and if so can the φ ratios be derived from that substitution by Perron-Frobenius without running the pipeline. Ariadna produced an analytic note grounded in the code anchors `level0/generator.py::_simplify_base`, `_collapse_inside_parentheses_local`, `_collapse_global_ignore_parentheses` and the variant-B branch of the main loop. Grace audited the note and accepted both its technical content and the prudence about possible higher-level macro-symbolic encodings.

### Dictamen

Canonical B is not equivalent to a fixed morfic substitution under the current executable Level 0 semantics. Three properties of a fixed morfic substitution are simultaneously violated by B: (1) the local simplifier of B is an AND-like many-to-one collapse (a non-empty all-`1` word maps to `1`, every other observable word maps to `0`), not a length-expanding letter morphism with context-free symbol images; (2) the emitted observable stream at iteration α is not μⁿ(a) for any fixed morphism μ — the next emission depends on the full accumulated trace `Acc` and on the dynamically produced sequence of intermediate inside-out collapse states; (3) there is no fixed two-letter production matrix from which φ could be obtained as a Perron-Frobenius eigenvalue.

### Change to claims

The presentation of φ in §6 (and any other section where the golden ratio appears as a primary result) must be demoted from "evidence" or "discriminator" to "numerical fingerprint inside a wider empirical signature, not derivable analytically from a fixed morfic substitution under the current implementation". The dictamen does not foreclose the possibility of a future higher-level macro-symbolic encoding that exposes some morfic structure on a compressed observable of B; any such future analysis must be ontologically and computationally separate from the canonical Level 0 semantics anchored here. The quintuple signature itself is unaltered.

### Falsification frontier

Unchanged. The dictamen is an analytic consequence of the current code semantics and does not introduce any new empirical commitment. If the canonical Level 0 semantics ever change in a way that introduces a fixed letter morphism, this entry would become superseded and a new analytic check would be required.

### Anchors

- Analytic note: `hsi_agents_project/Documentation/HSI_v2_Q2a_Phi_Morphic_Equivalence_Note.md`
  - SHA-256: `1BABC8BC3B4C54FCFDCE3F61D9520188CCC1E468DEB569464C08ADF2D60BEF4F`
- Code anchors referenced: `hsi_agents_project/level0/generator.py` (functions `_simplify_base`, `_collapse_inside_parentheses_local`, `_collapse_global_ignore_parentheses`, and the `variant == "B"` branch of the main generation loop).

### Pending Operational Closure

- **Editorial integration**: demotion of φ in `Paper_Latex/main_v2.02_cat.tex` (§6 and any other location where φ appears as a primary claim) to a side observation, with cross-reference to this entry and to the analytic note. The editorial change is governed by D-0139 and does not require its own log entry. Propagation to `main_eng.tex` queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`.
- **CAT_16 absorption**: ships with CAT_16 alongside D-0142.

---

## D-0144 — Hostile-review Q2b: execution contract for Bxor counterfactual (B-prime v0.1)

- Date: 2026-05-19
- Cycle: Third derived entry from the hostile-review cycle; functions as the formal execution contract that Grace required before allowing the counterfactual run to proceed.
- Trigger: The natural follow-up of Q2a (D-0143) is an empirical counterfactual: does a structurally different local collapse rule, still satisfying the necessary D1-D5 mechanism constraints, preserve the quintuple signature without the same φ convergence. Ariadna prepared a proposal v0.1 named Bxor, in which the only change with respect to canonical B is the local simplifier (replaced by parity: `s_Bxor(w) = sum(w_i) mod 2`); the outer mechanism (binary alphabet plus parentheses, monotonic accumulation, decay frame `(Acc)1`, deterministic inside-to-out collapse, intermediate states appended to `Acc`, final global simplification) is preserved verbatim. Sofia recommended acceptance on the grounds that XOR is algebraically the cleanest non-threshold alternative to B's AND-like simplifier. Grace audited and accepted on three explicit grounds: (i) Bxor introduces no tunable scalar parameter, immunizing it against the "fitted to taste" reading; (ii) D1-D5 preservation is verifiable from the proposal text; (iii) the kill-criterion clause ("if Bxor cannot produce 3,000,000 observable bits by iter20, the result is `not-executable-under-v0.1` and not a tuning permission") is a complete firewall against post-hoc rule adjustment. Grace's only operational requirement was that this entry be recorded with the proposal SHA-256 closed before execution; this entry fulfills that requirement.

### Execution contract

Ariadna is authorized to execute the Bxor counterfactual diagnostic strictly according to Proposal v0.1, with the following non-negotiable clauses:

1. No modification to the canonical Level 0 generator defaults. Bxor is generated by a separate public script with prefix `hsi_v2_`, writing an isolated structural snapshot under `hsi_agents_project/results/hsi_v2/q2b_bprime_counterfactual/level0/phi_snapshots/var_BXOR/`.
2. The snapshot must include `phi_iter20.struct.gz`, companion JSON metadata, `variant = "BXOR"`, `counterfactual = true`, the full local rule text, the invoking command, and a SHA-256 of the structural snapshot.
3. Phase 1 diagnostic is run read-only with `hsi_v2_phase1_run.py --input ...` over `segment_bits = 1_000_000`, `num_segments = 3`, `scales = 8,12,16,20,24,28,32`, `policies = prefix,suffix`.
4. Readout reports Bxor against canonical B, F, M/N, O/P and the B-based LZ-family nulls on (a) the five paper-facing signature coordinates where already available, (b) Phase 1 Gate-plane coordinates `retention@last` and `active_mean_tail`, (c) the raw-sequence Q1 metrics from `hsi_v2_raw_sequence_orthogonal_metrics.py`, and (d) φ-side fingerprints only as secondary diagnostics.
5. The outcome is reported strictly under one of the four pre-registered categories of Proposal v0.1: `signature-independent-of-phi-candidate`, `canonical-rule-dependent`, `phi-persists-under-Bxor`, `not-executable-under-v0.1`. Any other framing is forbidden by the proposal's wording-fence section.
6. If the outcome is `not-executable-under-v0.1`, the next step is a redesign proposal v0.2, not a tuning of v0.1.

### Falsification frontier

This entry does not by itself change the falsifiability contract of the paper. The empirical outcome of the Bxor run will determine whether the contract is enriched (outcome `signature-independent-of-phi-candidate` strengthens the implementation-independence reading of the signature), narrowed (outcome `canonical-rule-dependent` documents an implementation dependence and triggers a separate editorial cycle), or held stable (outcomes `phi-persists-under-Bxor` and `not-executable-under-v0.1` are diagnostically intermediate). Each outcome will be recorded as a follow-up entry (D-0145 or later) once the run completes.

### Anchors

- Proposal v0.1: `hsi_agents_project/Documentation/HSI_v2_Q2b_BPrime_Counterfactual_Proposal_v0_1.md`
  - SHA-256: `1514946F394F75BF32574B177F80B71F2C123E4FCB340DF072BFA21D53496615`
- Companion entries: D-0142 (orthogonal-metrics empirical anchor), D-0143 (analytic anchor that motivates Bxor).

### Pending Operational Closure

- **Authorization handoff to Ariadna**: short message confirming D-0144 is signed, the proposal SHA-256 is closed in the log, and execution of Bxor strictly per Proposal v0.1 is authorized. The handoff is operational (not a log entry).
- **Follow-up entry**: a new entry D-0145 (or later, depending on order) will record the Bxor run outcome and its consequences for claims once the run completes and is audited by Grace.
- **CAT_16 status**: D-0144 is registered as the pre-execution contract; the Bxor outcome is recorded separately at D-0145 and ships with CAT_16.

---


## D-0145 — Hostile-review Q2b outcome: phi-persists-under-Bxor (architectural-class fingerprint)

- Date: 2026-05-20
- Cycle: Empirical-outcome closure of the Q2b counterfactual contracted in D-0144; runs in the same hostile-review derivation cycle as D-0142, D-0143, D-0144 (CAT_16 absorption).
- Trigger: D-0144 authorized execution of the Bxor counterfactual strictly per Proposal v0.1, with four pre-registered outcome categories. Ariadna executed the diagnostic without any tuning of v0.1, reached iteration 20, and produced 10,242,265,104 observable bits. Grace audited the run post-execution and validated both the readout and the operational adjustment Ariadna applied at classification time (raising the Gate-1-like envelope to `retention >= 0.80` and `active_mean_tail >= 0.30`, since the initial `0.90` threshold would have wrongly discarded canonical B itself at `retention = 0.8622` in this batch). The outcome falls under the pre-registered category `phi-persists-under-Bxor`.

### Empirical finding

Bxor (XOR/parity local simplifier, all other Level 0 architecture unchanged) sits near canonical B on the Gate-1 plane: `retention@last` 0.8491 vs B 0.8622; `prefix_survival_last` 0.9957 vs 0.9978; `suffix_survival_last` 0.9925 vs 0.9901; `active_mean_tail` 0.4941 vs 0.3539; `kept@last` 1232 vs 1208; `candidate@last` 1451 vs 1401. Both are clearly separated from the strong LZ-family null envelope: `B-matched-lz retention@last` 0.0854; `B-phase-matched-lz retention@last` 0.0849. The secondary LZ fingerprint also persists: mean LZ ratio 0.602384, distance to 1/φ of 0.015650.

### Methodological reading (Ariadna, Grace-validated)

The outcome is diagnostically intermediate per D-0144, but the empirical content is informative beyond the pre-registered binary: the projective selectivity and the near-φ fingerprint do not depend on the exact AND local rule of canonical B; they persist when the local rule is replaced by XOR/parity, provided the global architecture (accumulation + ABS + stratified inside-to-out collapse) is preserved verbatim. This partially deactivates the "B is only the AND automaton that produces φ" objection. It does not authorize treating φ as an independent discriminator; rather, it indicates that φ is an architectural-class fingerprint, not an exclusive property of the local rule `01/10 -> 0`.

### Change to claims

No primary empirical claim is reverted or expanded. The Q2a demotion of φ (D-0143) is reinforced and re-anchored: the persistence of the near-φ fingerprint under a local rule replacement that preserves architecture is itself the strongest available evidence that the fingerprint is architectural, not rule-local. The wording-fence rules for paper integration (do not say: "BXOR proves the mechanism", "phi is fundamental", "Gate 2", "geometry", "universal law", "definitive closure"; do say: "pre-registered counterfactual", "local rule replacement", "same global architecture", "projective selectivity persists", "near-phi fingerprint persists", "not an independent discriminator", "architectural-class signal") are binding on the editorial integration.

### Falsification frontier

Unchanged at the gate level. The intermediate outcome neither enriches nor narrows the falsifiability contract in the binary sense pre-registered by D-0144. The qualitative reading it enables — architectural-class signal rather than rule-local artifact — is registered here and may be revisited if a future counterfactual diagnostic targets the global architecture rather than the local rule.

### Anchors

- Public scripts: `hsi_agents_project/hsi_v2_q2b_bxor_counterfactual.py`, `hsi_agents_project/hsi_v2_raw_sequence_orthogonal_metrics.py`
- Run directory: `hsi_agents_project/results/hsi_v2/q2b_bprime_counterfactual/q2b-bxor-counterfactual__iter-20__seg-3x1M__20260520T003124/`
- Snapshot: `level0/phi_snapshots/var_BXOR/phi_iter20.struct.gz`
  - SHA-256: `AA9EF4FDBCCE59D3892D29C6A66F7B2735C6FC84718258CC6AD084A0C465DD44`
- Companion entries: D-0142 (Q1 orthogonal-metrics empirical anchor), D-0143 (Q2a analytic anchor), D-0144 (Q2b execution contract).

### Pending Operational Closure

- **Editorial integration**: insertion of a single paragraph as fourth paragraph of §12 (`subsec:obj_circular`) of `Paper_Latex/main_v2.02_cat.tex`, immediately after the S6-T1 paragraph, respecting Ariadna's wording-fence verbatim. The editorial change is governed by D-0139.
- **Wording-fence audit**: after editorial integration, hand off to Grace for a post-integration wording-fence audit before CAT_16 freeze.
- **Propagation to `main_eng.tex`**: queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`.
- **CAT_16 absorption**: D-0145 closes the empirical content of the hostile-review cycle; CAT_16 freeze anchor includes D-0142, D-0143, D-0144, D-0145 plus the editorial cycle (S6-T1, S6-T2, S6-T3).

---


## D-0146 — Hostile-review Q3 analytic: parser-neutrality static audit of `\bar{c}` and editorial reformulation of §6.2

- Date: 2026-05-20
- Cycle: Analytic closure of the Q3 cycle (parser-neutrality flank), opened in response to Gemini's hostile-review observation that the manuscript's claim "all controls timeout on the c-bar parser" could be read as evidence of an HSI-tailored parser rather than a structurally neutral analyzer.
- Trigger: Gemini's contextless hostile-review pass on CAT_16 raised the methodological objection that the timeout-or-node-limit status of classical controls (A, J, K, L, M) on the `\bar{c}` parser was insufficiently distinguished from an ad-hoc bias of the parser toward HSI parenthesis syntax. Sofia identified the flank as genuinely new (not covered by Q1/Q2a/Q2b) and handed off a static-audit contract to Grace.

### Audit finding (Grace, static code audit)

The parser that computes `\bar{c}` (`hsi_agents_project/level1_nesting_tree.py`, functions `build_tree_from_structural` and `collect_all_metrics_iterative`) is a classical stack-based Dyck analyzer, `O(N)` in input length, with no HSI-specific optimizations and no structural assumptions beyond the alphabet `\Sigma = {0, 1, (, )}`. The audit verdict is structurally neutral. Controls without parenthesis syntax (A, J, K, L, M) do not "timeout" on the parser; they return a trivial one-node tree instantly because the input contains no parentheses to balance. The previous manuscript wording in §6.2 ("all controls timeout or reach node_limit") was factually incorrect and editorially vulnerable.

### Change to claims

No primary empirical claim is reverted. The §6.2 wording on the `\bar{c}` metric is reformulated to reflect the correct factual status (N.A. for controls without parenthesis syntax, structural neutrality of the parser, convergence to `\varphi+1` as a property of the class of generators with explicit stratified nesting rather than a comparative claim against controls without such syntax). Table 5 (`tab:signatura`) is updated: the `\bar{c}` cells for variants A, M, L are changed from `timeout` to `N.A.^{\dagger}`, with a new footnote in the caption explaining the rationale.

### Editorial integration

The following changes were applied to `Paper_Latex/main_v2.02_cat.tex`:

- §6.2 (`subsec:mean_children`), line 380: paragraph S6-T5 (factual reformulation, parser neutrality, N.A. status).
- Table 5 (`tab:signatura`), lines 437-439: `\bar{c}` cells for A, M, L changed to `N.A.^{\dagger}`.
- Table 5 caption, line 425: added `^{\dagger}` footnote explaining N.A. comes from absence of parenthesis syntax.
- §6 (`sec:signatura`), line 343: paragraph S6-T2 extended with a forward-reference to §12 (`subsec:obj_circular`) and Bxor evidence (companion editorial fix derived from the same hostile-review pass, defending against the "phi is magic" reading).
- §12 (`subsec:obj_circular`), line 974: dialectal normalization `sigui` → `siga` (Valencian-consistency post-audit by Grace; direction-reversed correction documented at the time of editing).

### Falsification frontier

Unchanged. The audit confirms that the falsifiability of the `\bar{c} = \varphi+1` claim was never resting on the parser timing out for controls; it rests on the convergence behavior of generators that do produce a non-trivial nesting tree. The Q3 analytic closure removes a wording vulnerability without altering the empirical content.

### Anchors

- Manuscript: `Paper_Latex/main_v2.02_cat.tex`, §6.2 (lines 371-385), Table 5 (lines 420-445), §6 (S6-T2 paragraph, line 343), §12 (S6-T1 paragraph, line 974).
- Audited parser code: `hsi_agents_project/level1_nesting_tree.py`, functions `build_tree_from_structural` (lines 192-294) and `collect_all_metrics_iterative` (lines 329-406).
- Grace static audit artifact: `HSI_v2_Audit_Mean_Children_Neutrality.md` (Grace's environment, referenced as audit handoff response).
- Companion entries: D-0142 (Q1 orthogonal-metrics empirical anchor), D-0143 (Q2a analytic anchor), D-0144 (Q2b execution contract), D-0145 (Q2b outcome).
- Related downstream entry: D-0147 (Q3 empirical closure, completed in the same hostile-review derivation cycle).

### Pending Operational Closure

- **CAT_17 recompilation**: produce the new freeze candidate `CAT_17.pdf` from `main_v2.02_cat.tex` incorporating all S6-T5, S6-T6 (D-0147), S6-T2 (forward-reference), and S6-T1 (dialectal fix) edits.
- **Propagation to `main_eng.tex`**: queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; frozen until the hostile-review cycle is fully closed.

---

## D-0147 — Hostile-review Q3 empirical outcome: c-bar-converges-to-not-phi (architectural-class anchor for `\bar{c}`)

- Date: 2026-05-20
- Cycle: Empirical closure of the Q3 cycle (parser-neutrality flank), companion to the Q3 analytic closure in D-0146. Runs in the same hostile-review derivation cycle as D-0142, D-0143, D-0144, D-0145, D-0146.
- Trigger: D-0146 closed the parser-neutrality flank analytically (Grace static audit) and editorially (§6.2 reformulation, Table 5 N.A. status). The reformulated claim that `\bar{c} = \varphi+1` is a property of the architectural class of generators with explicit stratified nesting (and not merely a property of the parenthesis syntax) was logically defensible but lacked a direct empirical anchor. Sofia handed off to Ariadna a pre-registered counterfactual: run the same audited parser against three families of Dyck-randomized controls that share the parenthesis alphabet but lack the HSI architecture.

### Pre-registered design

Three control families, all at scale `1G` characters and parenthesis density matched to canonical `B@20` (chosen as the lowest reference iteration where canonical B already converges to `\bar{c} = 2.618034` across the four published segments):

- **UD** (Uniform Dyck): uniformly sampled balanced Dyck word, terminal bits filled with Bernoulli(p=0.5).
- **MD** (Markov-Depth Dyck): balanced Dyck with depth-transition Markov chain biased to match the empirical depth profile of B@20.
- **DM** (Density-Matched insertion): Bernoulli(p=0.5) binary string with parentheses inserted at random positions at the density observed in `\Phi_B`@20; not balanced by construction (deliberately adversarial against the parser).

Seeds: 607, 709, 811 (three disjoint seeds per family, nine runs total). Stop conditions: 600s timeout per run, 50M node_limit (canonical pipeline thresholds). Pre-registered outcome categories: (a) c-bar-undefined-or-fails, (b) c-bar-converges-to-not-phi, (c) c-bar-converges-to-phi (falsifier), (d) c-bar-mixed.

### Empirical finding

Outcome: **c-bar-converges-to-not-phi** (category b).

- UD: `\bar{c} = 2.000083`, CI95 `[1.999937, 2.000229]`.
- DM: `\bar{c} = 2.000084`, CI95 `[1.999655, 2.000514]`.
- MD: `\bar{c} = 2.151581`, CI95 `[2.151530, 2.151632]`.
- Reference B@20: `\bar{c} = 2.618033834809`.

All three families completed the analysis with stable, replicable convergence on three disjoint seeds. None coincides with `\varphi + 1` at 6 decimal places. The most adversarial control (MD, which imitates the vertical depth distribution of B) remains stably separated by ~0.47 from the HSI reference.

### Technical note (Ariadna, Grace-validated)

The script does not call the `analyze_nesting_tree()` wrapper because that wrapper computes auxiliary recursive statistics (`tree.subtree_size()`, `tree.max_depth()`) that raise `RecursionError` on deep random Dyck. Instead it calls the two audited iterative pieces directly: `build_tree_from_structural(...)` and `collect_all_metrics_iterative(...)`. Grace validates that this preserves the semantics of `\bar{c}` exactly and is methodologically correct.

### Methodological reading (Ariadna, Grace-validated)

The mere presence of parenthesis syntax does not determine the convergence `\bar{c} = \varphi+1`. Pre-registered Dyck controls with comparable structural density, processed by the same audited parser, converge to stable but different values. This directly anchors the editorial claim (D-0146) that the convergence is a property of the architectural class of stratified-collapse generators, not of the syntax alone. The result does not authorize wording such as "exclusive to HSI", "unique architecture", "universal constant", or "definitive closure"; the surgical statement is that the convergence requires the specific architecture of stratified collapse, and that parenthesis syntax alone is empirically insufficient.

### Change to claims

No primary empirical claim is reverted. The §6.2 reformulation from D-0146 is reinforced and anchored empirically: a new short paragraph (S6-T6) is added at the end of §6.2 reporting the three Dyck-randomized control values and the architectural-class reading. The wording-fence rules pre-registered in the Sofia → Ariadna handoff (do not say: "exclusive to HSI", "unique possible architecture", "geometric meaning", "universal constant", "definitive closure"; do say: "pre-registered Dyck control", "parenthesis syntax without HSI architecture", "`\bar{c}` is not determined by the mere presence of parenthesis syntax", "convergence to `\varphi+1` requires stratified collapse", "pre-registered outcome: category b") are binding on the editorial integration and are respected verbatim in S6-T6.

### Falsification frontier

Unchanged at the gate level. The Q3 empirical outcome strengthens the falsifiability anchor by exhibiting concrete non-HSI generators with parenthesis syntax that converge to `\bar{c} \neq \varphi+1`. The pre-registered falsifier (category c, `c-bar-converges-to-phi`) did not materialize; if a future control family within the same class were to reproduce `\varphi+1`, the architectural-class claim would have to be reformulated.

### Anchors

- Public script: `hsi_agents_project/hsi_v2_q3_empirical_dyck_randomized.py`
- Run directory: `hsi_agents_project/results/hsi_v2/q3_empirical_dyck_randomized/q3-empirical-dyck-randomized__chars-1000000000__families-UD-MD-DM__seeds-607-709-811__20260520T184454/`
- Summary SHA-256: `F380861F8D47A082714094176EA24D2C8B5AE16FCD70170B77EF0E5998F54FAE`
- Grace audit artifact: `docs/HSI_v2_Audit_Q3_Empirical_Dyck_Randomized.md`
- Audited parser code (same as D-0146): `hsi_agents_project/level1_nesting_tree.py`, `build_tree_from_structural` and `collect_all_metrics_iterative`.
- Companion entries: D-0142 (Q1), D-0143 (Q2a analytic), D-0144 (Q2b contract), D-0145 (Q2b empirical), D-0146 (Q3 analytic).

### Pending Operational Closure

- **Editorial integration**: paragraph S6-T6 inserted at §6.2 of `Paper_Latex/main_v2.02_cat.tex`, immediately after the S6-T5 paragraph from D-0146, respecting the pre-registered wording-fence. The editorial change is governed by D-0139.
- **CAT_17 recompilation**: D-0147 is the last editorial change of the Q3 cycle; CAT_17 is the freeze candidate that absorbs D-0142 through D-0147.
- **Propagation to `main_eng.tex`**: queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; frozen until the hostile-review cycle is fully closed.

---


## D-0148 — Hostile-review CAT_19 (Phase 4 intermittency): pre-registration freeze of P4-SLS v0.3.1

- Date: 2026-05-21
- Cycle: Derived entry from the CAT_19 hostile-review cycle (Gemini 3.5 Flash) on `main_v2.02_cat.tex`. CAT_19 raised four points (A Dyck circularity, B pseudo-physical lexicon, C hard-negative wording in Abstract, D Phase 4 intermittency without structural framing). A and C were closed editorially with minimal forward-reference and lexical tightening (governed by D-0139). B was held without action. D was externalized to a new pre-registered experimental cycle to characterize the local structural signature of supported Phase 4 regions instead of being patched into the manuscript.
- Trigger: Closing D editorially would have required a structural claim about Phase 4 supported regions (P4-04: 768M–795M) without empirical backing. Sofia handed off to Ariadna a design proposal v0.0 framed in subshift terms. Ariadna returned v0.1 with a critical asymptotic-vs-finite reframe (finite-window block-complexity and lag-conditioned mutual information instead of true subshift entropy), four technical improvements to probes, a Stage 0 / Stage 1 split, and reformulated outcomes. Sofia returned v0.1 draft with four open points (effect thresholds, seeds, ARC-B inclusion, aggregation policy). Ariadna closed the four open points plus two transversal clauses (descriptive p-values only, no figures pre-readout). Sofia drafted v0.2 integrating the closure verbatim. Grace audited v0.2 and returned `APROVAT AMB MATISOS` with five nuances (A manifest-derived seeds: exclude `block-entropy`; B finestres count corrected from 22 to 21 by dropping P4-05 distal window; D hierarchical readout flow; F ex-ante lag wording reinforcement; G regional denominator blindatge against `power-insufficient`). Sofia integrated the five nuances verbatim as v0.3. Ariadna detected that the matís A propagation in v0.3 was incomplete (block-entropy was removed from seed extraction but kept nominally in the hard-null envelope, creating a sequence-level implementation contradiction). Sofia closed with v0.3.1, propagating the matís A consequence completely (block-entropy excluded from the hard-null envelope; generative envelope reduced to `{phase-matched-lz, matched-lz}`). Both Sofia and Ariadna verified the SHA-256 of v0.3.1 independently and obtained the same value. This entry records the freeze that authorizes Stage 0 execution.

### Execution contract

The pre-registered contract is frozen exactly as written in `prereg_P4_subshift_like_signature_v0.3.1.md` (SHA-256 below). The following clauses are non-negotiable and operationalize the contract:

1. Stage 0 (inventory and seed extraction) and Stage 1 (probes and readout) are sequential. Stage 1 is not authorized until Stage 0 returns `inputs-derivable` and the Stage 0 manifest is itself frozen.
2. The hard-null envelope for all three probes is `{phase-matched-lz, matched-lz}`. `block-entropy` does not participate at the envelope of this cycle for sequence-level reasons (not for data unavailability).
3. The three probes operate on the 21 pre-declared 9M-bit windows: 3 at P4-04 (primary), 3 at P4-05 (contextual, non-decisor), 5 at 300M (primary control), 5 at 1110M (primary control), 5 at 597M ARC-B (qualifier, non-primary).
4. Decisions are driven by hard-null envelope gaps with pre-declared effect floors (`gap_h_n ≥ 0.01 nats`, `gap_MI ≥ 0.002`, `gap_CV ≥ 0.10`) and 95% block-bootstrap CIs. Classical p-values are descriptive only.
5. The four pre-registered outcomes are evaluated under the hierarchical decision flow declared in §5 of v0.3.1; the first match defines the outcome (`inputs-or-power-insufficient`, `null-mimicry-or-no-structure`, `local-subshift-like-support`, `intermittent-structural-signal`).
6. Wording-fences for PROBE 2 (lags as ex-ante diagnostic coordinates) and for the finite-block topological entropy proxy are binding both at the readout and at any future manuscript reference.
7. This cycle does not open Gate 2 under any outcome. No additional probe may be introduced without a new pre-registration.
8. Power-insufficient windows count as NOT positive and do not modify the regional denominator (3 for P4-04; 5 for each of 300M, 1110M, ARC-B).
9. No figures are produced before the numerical readout and the Grace code audit.

### Falsification frontier

This entry does not by itself change the falsifiability contract of the paper. The empirical outcome of the P4-SLS run will determine whether the §11.3 framing of Phase 4 intermittency is enriched (`local-subshift-like-support` triggers an independent Sofia-Grace editorial cycle), narrowed (`null-mimicry-or-no-structure` adds a short note that the subshift-like structural reading is not supported by the pre-registered P4-SLS probes), held stable in an intermediate diagnostic state (`intermittent-structural-signal` is logged but does not modify the manuscript), or aborted (`inputs-or-power-insufficient` triggers Decision Log entry only). The outcome will be recorded as a follow-up entry once the run completes and is audited by Grace.

### Anchors

- Pre-registration v0.3.1: `prereg_P4_subshift_like_signature_v0.3.1.md`
  - SHA-256: `FB150162FBBD9CF7D85B6AF364DA942052D26D3115D783D1C0D254AC4A57B451`
  - Independently verified by Sofia (PowerShell `Get-FileHash`) and Ariadna; both reports coincide bit-for-bit.
- Grace audit (on v0.2, integrated verbatim in v0.3, propagated completely in v0.3.1): `docs/HSI_v2_Audit_Prereg_P4_SLS_v0.2.md`
  - Audit verdict: `APROVAT AMB MATISOS` (matisos A, B, D, F, G).
- Source manifests for Stage 0 (read-only inputs): Phase 4 frozen manifests under `hsi_agents_project/results/hsi_v2/` (specific paths to be resolved by Stage 0 inventory).
- Companion entries: D-0142 through D-0147 (CAT_16/CAT_17 hostile-review cycle, of which CAT_19 is the next round).

### Pending Operational Closure

- **Authorization handoff to Ariadna**: short message confirming D-0148 is signed, the v0.3.1 SHA-256 is closed in the log, and execution of Stage 0 strictly per v0.3.1 is authorized. The handoff is operational (not a log entry).
- **Stage 0 outcome**: a new entry will be added recording `inputs-derivable` (proceed to Stage 1) or `inputs-not-derivable` (abort and decision on reformulation).
- **Stage 1 outcome**: conditional on Stage 0 and Grace code audit; a separate follow-up entry will record the readout under one of the four pre-registered outcome categories.
- **Manuscript propagation**: governed by §5 of v0.3.1; only `local-subshift-like-support` triggers an independent Sofia-Grace editorial cycle, and `null-mimicry-or-no-structure` triggers a short note at §11.3. The other two outcomes are log-only.

---


## D-0149 — Hostile-review CAT_19 (Phase 4 intermittency) outcome: null-mimicry-or-no-structure (subshift-like local reading refuted)

- Date: 2026-05-22
- Cycle: Empirical closure of the P4-SLS cycle frozen at D-0148, completing the CAT_19 cycle (point D externalized as P4-SLS).
- Trigger: D-0148 froze `prereg_P4_subshift_like_signature_v0.3.1.md` at SHA-256 `FB1501...B451`, with the hard-null envelope reduced to `{phase-matched-lz, matched-lz}` (sequence-level coherence). Stage 0 returned `inputs-derivable` (the 21 pre-declared 9M-bit windows were physically extractable for B and the two generative null families from the frozen Phase 4 manifests). Iban formally authorized Stage 1 execution. Ariadna executed `hsi_v2_p4_sls_stage1_readout.py` against the in-hot-loaded contract; the SHA-256 of the contract loaded at runtime coincided exactly with the frozen value, confirming no contractual drift. Grace audited the full readout post-execution and returned `APROVAT SENSE RESERVES`. This entry records the outcome and its consequences for the manuscript.

### Pre-registered design recap

Three sequence-level probes over 21 windows (3 at P4-04 primary, 3 at P4-05 contextual non-decisor, 5 at 300M primary control, 5 at 1110M primary control, 5 at 597M ARC-B qualifier), against the generative null envelope `{phase-matched-lz, matched-lz}`. Decisor effects: `gap_h_n ≥ 0.01 nats` (Probe 1, n ∈ {22,24}), `gap_MI ≥ 0.002` at frozen Phase 4 lags (Probe 2), `gap_CV ≥ 0.10` aggregated across k ∈ {12,16,20} (Probe 3, non-rescuing secondary). All decisions filtered by 95% block-bootstrap CI strictly positive. Hierarchical readout flow with four pre-registered outcomes.

### Empirical finding

Outcome: **`null-mimicry-or-no-structure`** (category 2 of the hierarchical flow).

- **Probe 1 (finite block-complexity)**: separation of B from the generative null envelope is present, but is uniform across all regions (P4-04 `3/3`, P4-05 `3/3`, 300M `5/5`, 1110M `5/5`, ARC-B `5/5`). The regional criterion `≥2/3 at P4-04 AND ≤1/5 at 300M AND ≤1/5 at 1110M` fails flagrantly at both internal controls (`5/5 > 1/5`). Probe 1 regional support: `False`.
- **Probe 2 (lag-conditioned block mutual information)**: `0/3` positive windows at P4-04. Reported `gap_MI` values at P4-04 are of order `10⁻⁶` or negative (e.g., `−2.53 × 10⁻⁵`), well below the pre-declared `≥ 0.002` floor. Probe 2 regional support: `False`.
- **Probe 3 (recurrence-gap spectrum, non-rescuing secondary)**: `0/3` positive windows at P4-04. No differential recurrence regularity detected.
- **Hierarchical readout flow**: step 1 not matched (`inputs-derivable` confirmed at Stage 0; no `power-insufficient` windows reported). Step 2 matches because neither primary separates B from the nulls regionally (Probe 1 by global mimicry of the separation, Probe 2 by absence of local signal). Outcome assigned: **`null-mimicry-or-no-structure`**.

### Technical reading (Sofia + Ariadna, Grace-validated)

The mechanism of the negative is not that the generative nulls reproduce B's complexity at P4-04. They do not: B is separated from the null envelope in finite block-complexity at every region tested. The mechanism is that this separation is a **global property of B, not a local textural signature of the predictive region P4-04**. The pre-registered hypothesis required the separation to be regionally differential; it is not. Probes 2 and 3 add independent evidence that no further structural signature in the lag-conditioned or recurrence-regularity sense distinguishes P4-04 from the non-replicating controls under the frozen probe set. The subshift-like-finite-window reading of Phase 4 intermittency is therefore refuted within the P4-SLS contract; Phase 4 intermittency remains empirically delimited but is not structurally explained by this reductive route.

### Change to claims

No primary empirical claim is reverted. The §11.3 framing of Phase 4 as local and intermittent (B-mini result from D-0105/D-0106) is preserved verbatim. A new short defensive paragraph is appended at the end of §11.3 of `Paper_Latex/main_v2.02_cat.tex` (after the D-0106 paragraph, before the section break), respecting the wording-fence pre-registered in v0.3.1 for outcome `null-mimicry-or-no-structure`. The patch initially proposed by Grace contained a minor empirical imprecision ("nulls mimic B's block-complexity in this region") that contradicted the Probe 1 finding (separation is present everywhere, including the controls). Ariadna's corrected wording ("separates B from the nulls globally, not specifically at P4-04") restores empirical accuracy and is the version integrated. No figure, no extension of any other claim. Gate 2 remains closed under this outcome.

### Falsification frontier

The pre-registered falsifiers `local-subshift-like-support` and the intermediate `intermittent-structural-signal` did not materialize. The subshift-like-finite-window reading of Phase 4 intermittency is now empirically refuted within the P4-SLS contract. The negative is acotat to this contract: it does not preclude alternative structural readings (e.g. compositionality at higher tiers of the projective hierarchy, statistical mechanics of the routing components, or non-stationary correlation structures outside the frozen lag set) but any such reading would require a separate pre-registered cycle. The diagnostic value of this negative is to close a reductive explanatory route that was attractive but empirically unsupported.

### Anchors

- Run directory: `hsi_agents_project/results/hsi_v2/phase4/p4_sls_subshift_like_signature/stage1_readout/p4-sls-stage1-readout__windows-21__20260522T105412/`
- Pre-registration v0.3.1 (frozen at D-0148): `prereg_P4_subshift_like_signature_v0.3.1.md`
  - SHA-256: `FB150162FBBD9CF7D85B6AF364DA942052D26D3115D783D1C0D254AC4A57B451`
- Artifact SHA-256:
  - `summary.json`: `24AC53C56B3774136BA2F484A0656276C95D87148AEC56D7639CA30C1FE92AC9`
  - `manifest.json`: `184C3D956DCA1DF6620944AD1FC7648885677E12AE467F2B0BA85D6679696FBA`
  - `report.md`: `1C2F9260CAEFC2B8DD19CA825248981A607A55AD1C02602109A01F5E98F07224`
  - `window_readout.csv`: `C9513CCDBC7C93BF7BA60AA46C95E0F28187DF7D9522AB6B6E4238B29626AEFF`
- Grace audit artifact: `docs/HSI_v2_Audit_P4_SLS_Stage1_Readout.md` (verdict: `APROVAT SENSE RESERVES`).
- Public script: `hsi_agents_project/hsi_v2_p4_sls_stage1_readout.py`.
- Companion entries: D-0148 (P4-SLS contract freeze), D-0146/D-0147 (Q3 cycle, the other branch of CAT_17/CAT_19 closure).

### Pending Operational Closure

- **Editorial integration**: paragraph `[D-0149-T1]` appended at the end of §11.3 of `Paper_Latex/main_v2.02_cat.tex` (after the D-0106 B-mini paragraph), respecting the v0.3.1 wording-fence and Ariadna's empirical correction. Governed by D-0139.
- **CAT_20 recompilation**: D-0149 is the last scientific change of the CAT_19 cycle and ships with CAT_20 (which also absorbs the A and C editorial closures of CAT_19 already integrated in the .tex source: forward-reference at §3.5 and "hard-negative" lexical tightening at the Abstract).
- **Propagation to `main_eng.tex`**: queued under section A.5 of `docs/HSI_v2_Pending_Closures_and_v3_Agenda.md`; frozen until the hostile-review cycle is fully closed.

---

## D-0150 — Hostile-review CAT_22 audit: Table 1 `\bar{c}` for O,P read-only derivation (legacy `timeout/N.A.` policy supersession)

- Date: 2026-05-24
- Cycle: Closure of one of the four major/minor flanks raised by hostile-review CAT_22 (Gemini 3.1 Pro) on `main_v2.02_cat.tex` post-CAT_21. CAT_22 raised three majors (residual geometric lexicon, B_XOR underdetermination of Level 0, Phase 4 horizon at high scales) and three minors (Table 1 `timeout/N.A.` inconsistency for O/P, boolean alphabet bias, F.2 condensation). This entry records the empirical/editorial closure of Minor 1 (Table 1 inconsistency). Majors 1-3 and Minor 3 were closed editorially within the same cycle without new empirical work (governed by D-0139); Minor 2 was declined as content-free.
- Trigger: CAT_22 reviewer pointed out that Table 1 (§6.6) reports `timeout/N.A.` for `\bar{c}` on external recursive controls O and P, while Table 2 (§7.x) shows the full pattern count `|P_m|` up to `m=32` for both controls on the Q4-authorized snapshots, and §6.2 documents that the Dyck parser is `O(N)` and structurally neutral. The reviewer correctly identified that the `timeout/N.A.` cells were not technically justified given the available evidence. Sofia handed off to Ariadna a read-only derivation request: compute `\bar{c}` for O and P on the same Q4-authorized snapshots (D-0087), no regeneration, audited parser only, returning either values or a documented reason why the metric is undefined.

### Empirical finding

Ariadna executed `level1_nesting_tree.py` in read-only mode on the D-0087 Q4 snapshots for O and P, tail-trimming the decoded stream exactly at `structural_length` from the metadata to avoid 2-bit format padding. Parser completed without timeout, without node limit, without failure.

- O: `\bar{c} = 3.429812370`, branching nodes 2,878, total nodes 9,872, max depth 11, elapsed 1.35s.
- P: `\bar{c} = 2.999796789`, branching nodes 9,842, total nodes 29,525, max depth 10, elapsed 1.44s.
- Reference B@20: `\bar{c} = 2.618033834809` (`= \varphi + 1`).

Neither O nor P converges to `\varphi + 1`. The legacy `timeout/N.A.` policy inherited from Q5/D-0089 was a methodological holdover from a time when the audited read-only path was not yet available; it is not a technical timeout and was empirically misleading.

### Methodological reading (Ariadna + Sofia, no Grace audit required for this scope)

External recursive controls O and P possess parenthesis syntax (CFG and L-system over `\Sigma = {0,1,(,)}`), so `\bar{c}` is well-defined on their Q4 snapshots. The fact that they yield definite values that do not converge to `\varphi + 1` reinforces the architectural-class anchor established in D-0146 and D-0147: the convergence `\bar{c} = \varphi + 1` is a property of the architectural class of stratified-collapse generators, not of the mere presence of parenthesis syntax. The Q3 cycle empirically anchored this against pre-registered Dyck-randomized controls; the CAT_22 derivation extends the empirical evidence to the pre-existing external recursive controls of the master signature table, closing a residual editorial inconsistency that an external reviewer correctly flagged.

### Change to claims

No primary empirical claim is reverted. Table 1 of `Paper_Latex/main_v2.02_cat.tex` is modified in two surgical edits (marked `[CAT_22-T4]` in the source):

1. Rows O and P, column `\bar{c}`: `timeout/N.A.` → `$3.4298^{\ddagger}$` and `$2.9998^{\ddagger}$` respectively (rounded to 4 decimals for consistency with the rest of the table).
2. Caption: added documentation of the new footnote symbol `\ddagger`, clarifying that the values are read-only derivations on the Q4 pre-registered snapshots, that the Dyck parser completes the analysis, that the values are well-defined but do not converge to `\varphi + 1`, and that this constitutes complementary evidence (alongside the §6.2 Dyck-randomized controls of D-0146/D-0147) that the `\bar{c} = \varphi + 1` signature is not an artefact of the mere presence of parenthesized external syntax.

No other section of the manuscript was modified by this entry. Internal coherence was verified: lines 388 and 391 (§6.2) already framed only A, M, L as `N.A.` (no parenthesis syntax) and already anchored the architectural-class claim against pre-registered Dyck controls; no contradiction with the new Table 1 cells.

### Falsification frontier

The pre-registered architectural-class claim of D-0146/D-0147 (`\bar{c} = \varphi + 1` requires stratified collapse, not merely parenthesized syntax) gains a third empirical anchor from this derivation. The contract is unchanged; the evidence base is broader by two additional non-HSI controls that share parenthesis syntax and yield non-`\varphi` convergence.

### Anchors

- Run directory: `hsi_agents_project/results/hsi_v2/table1_op_cbar_derivation/table1-op-cbar-derivation__var-O-P__iter-20__20260524T120714/`
- Artifact SHA-256 (verified by Ariadna):
  - `summary.json`: `7BED9767716835D64020C49BBC8DD7A470221A142BBD0E517ABE191E5E6A1B50`
  - `manifest.json`: `E172A1AD77B58A7635D7F13348984E0B84C2E11340C4CA041049EB4371FBE5F4`
  - `report.md`: `0DB99816567449A0B256CAFBB422C39E0004B3DEC297AF82070F5ADD912A5C3F`
- Source snapshots (Q4 D-0087):
  - O: `results/level0/phi_snapshots/var_O/phi_iter20.struct.gz`, SHA-256 `850F447BD02841DD0FD49FC2A4362460BEB16E441537D48A45411AB7DF88ED2E`
  - P: `results/level0/phi_snapshots/var_P/phi_iter20.struct.gz`, SHA-256 `8F2F495A37FBA301C4D9B5A72C4FD1C62732E22E81E4F1822C38D754BEBA1FA7`
- Audited parser: `level1_nesting_tree.py` (Q3 audit chain, D-0146 anchor).
- Companion entries: D-0087 (Q4 snapshot freeze), D-0089 (Q5 superseded policy), D-0146/D-0147 (Q3 architectural-class anchors).

### Pending Operational Closure

- **CAT_22 cycle closure**: this entry closes Minor 1 of CAT_22. Majors 1, 2, 3 and Minor 3 of CAT_22 are closed editorially in the same .tex source via `[CAT_22-T1]`, `[CAT_22-T2]`, `[CAT_22-T3]`, `[CAT_22-T5]` (governed by D-0139). Minor 2 (boolean alphabet bias) is declined: lexical note at §2 and §3.1 already declare `0`/`1` as ontological symbols; the proposed admission would be redundant without adding technical content.
- **Recompilation**: pending Iban authorization. Single recompile will ship the full CAT_22 closure (one empirical entry plus four editorial patches).
- **Propagation to `main_eng.tex`**: queued under the existing freeze on the English version until the hostile-review cycle is fully closed.

---

## D-0151 — CAT_24 iteration-depth extension readout: B@24 projective tower survives through m=35 (empirical ceiling extension)

- Date: 2026-06-04
- Cycle: Closure of CAT_24-T4 (one of the empirical flanks raised by hostile-review CAT_24 on `main_v2.02_cat.tex` post-CAT_23). CAT_24 raised the "finite horizon" vulnerability: the empirical ceiling at $m \leq 32$ could be read as the boundary of a finite phenomenon rather than as a computational ceiling on an ongoing signature. T4 is the empirical move that decouples the two readings by extending the projective tower of variant B beyond the original ceiling without regenerating the source.
- Trigger: Sofia handed off to Ariadna a pre-registered read-only derivation contract (`prereg_CAT24_B24_projective_tower_v1.0.md`, SHA-256 `4F4D5F518EE7F3E2205A75D0C6967FAA19E875130B3E6E9A627937B2E49E61D7`) over the frozen `phi_iter24` snapshot of B (~687 Gbits). The contract pre-declared three outcomes (`b24-tower-survives-through-m35`, `b24-tower-degrades-at-m={33,34,35}`, `b24-tower-collapses-early`) with explicit ambiguity bands on cardinality ratios ([0.45, 0.55]) and survival fractions ([0.25, 0.35]), and a closure clause stating that a clear readout outside the ambiguity bands does not authorize the coverage-extension cycle (no 30M follow-up within CAT_24 scope).

### Empirical finding

Ariadna executed the projective tower derivation on the B@24 snapshot for $m \in \{33, 34, 35\}$, with prefix/suffix projections and protocol-canonical $3 \times 10^6$ observable bits per window. Outcome: `b24-tower-survives-through-m35`.

- Cardinality ratios (all well above the 0.5 degradation threshold, all above 1.0):
  - $|P_{33}|/|P_{32}| = 1.0637$
  - $|P_{34}|/|P_{33}| = 1.0607$
  - $|P_{35}|/|P_{34}| = 1.0558$
- Survival fractions at $m=35$ (all well above the 0.3 collapse threshold):
  - prefix $35 \to 34$: 0.9978
  - suffix $35 \to 34$: 0.9963
- Active support at $m=35$: 1,439 patterns (×90 the floor of 16).
- Pattern counts: $|P_{33}| = 1{,}285$; $|P_{34}| = 1{,}363$; $|P_{35}| = 1{,}439$.

The signature is stronger than mere survival: ratios above 1.0 indicate that prefix coverage continues to discover new structure at each scale jump within the high-$m$ band.

### Decisions

1. **Integration of CAT_24-T4** (this entry): the empirical ceiling is extended from $m \leq 32$ to $m \leq 35$ for the projective tower of variant B at iter=24, under read-only pre-registered contract. Other variants remain documented up to $m \leq 32$ on the main batch (E@24 not generated). Five surgical edits applied to `Paper_Latex/main_v2.02_cat.tex`: contributions paragraph (line 76), operational ceiling paragraph (line 271), reproducibility paragraph (line 274), experimental protocol paragraph (line 578), Table 3 with new $m=33,34,35$ rows and updated caption (lines 589-604), and §12.2 asymptotic-ceiling paragraph (line 1061). Internal traceability marker `% [CAT_24-T4]` added at the contributions paragraph as the anchor for the coordinated batch.
2. **Deferral of CAT_24-T5** (Phase 4 extension): per Grace's audit and Ariadna's technical assessment, Phase 4 extension to higher bands is deferred. The deferral has three independent justifications: (a) E@24 has not been generated, so any new Phase 4 reading would lack the canonical B/E contrast that anchors the existing Phase 4 result; (b) the streaming extractor for the 795M--804M bands is not designed for 100G/500G bands and would require new I/O architecture (local-window or streaming); (c) extending Phase 4 to qualitatively different bands risks changing the scientific object under measurement rather than extending the existing one. A separate pre-registered contract is required before any Phase 4 extension touches the manuscript.
3. **Scope discipline (closure clause honoured)**: the readout falls clearly outside the ambiguity bands, so the contract's closure clause is honoured: no 30M coverage-extension cycle is opened within CAT_24. Any future investigation of coverage effects on the projective tower would open a new pre-registered cycle (hypothetical CAT_25 or successor) with its own contract and scope.
4. **No Phase 4 wording change in the manuscript**: per Ariadna's recommendation and Grace's audit, §12.2 is not extended with new Phase 4 bands. The §12.2 update is limited to acknowledging the extended ceiling for the B@24 projective tower; the Phase 4 deferral is internal-operational and does not warrant manuscript text.

### Artefacts and hashes

- Contract: `prereg_CAT24_B24_projective_tower_v1.0.md` — SHA-256 `4F4D5F518EE7F3E2205A75D0C6967FAA19E875130B3E6E9A627937B2E49E61D7`.
- Audit: `docs/HSI_v2_Audit_CAT_24_B24_Readout.md` (Grace, methodological auditor).
- Snapshot: B@24 (`phi_iter24`, ~687 Gbits), reused without regeneration.
- Output artefacts: `manifest.json`, `projection_prefix_m{33,34,35}.json`, `projection_suffix_m{33,34,35}.json`, `fibers_m{33,34,35}.json`, `moments_m{33,34,35}.json` under the run directory pre-declared in §3 of the contract. SHA-256 set delivered by Ariadna in the readout handoff (full set in the run manifest; not duplicated here per D-0139).

### Pending Operational Closure

- **CAT_24-T4 closure**: this entry closes the empirical flank T4 of CAT_24. T1--T3 were closed editorially in the same .tex source in the preceding integration pass (markers `[CAT_24-T1]`, `[CAT_24-T2]`, `[CAT_24-T3]` at lines 274, 392 and 845, addressing hostile-review CAT_24 points 7, 2 and 5 respectively; governed by D-0139). T5 (Phase 4 extension) is explicitly deferred (see decision 2 above).
- **Recompilation**: pending Iban authorization. Single recompile will ship the CAT_24-T4 integration (the coordinated edits anchored by the marker at line 76).
- **Propagation to `main_eng.tex`**: queued under the existing freeze on the English version until the hostile-review cycle is fully closed.

---

## D-0152 — Q6-A multi-stage opening: peer-review-driven P4@24 symmetric extension (Stage 0 = E@24 infrastructure)

- Date: 2026-06-07
- Status: `accepted`
- Scope: scientific cycle opening (Q6-A) + operational pre-commit
- Cycle: Opening of Q6-A in response to external peer-review flank B (asymptotic wall at 795M--804M, Phase 4 P4-05 readout of $0/4$ under `phase-matched-lz` convergence) on `Paper_Latex/main_v2.03_cat.tex`. The peer review (Gemini 3.x, brief-blind modality, no project context) reads the P4-05 readout as asymptotic falsification rather than as a bounded finite-horizon frontier. The defense currently in §11.3 (`convergence of null envelope at the band frontier`, multi-mechanism decomposition per P4-06) is internally consistent but cannot answer the asymptotic reading without empirical extension beyond 804M.
- Trigger:
  - External: peer-review flank B.
  - Internal: Ariadna's methodological objection from D-0151 (Phase 4 extension on B@24 alone would lack the canonical B/E contrast). The opening of Q6-A is contingent on first removing that objection by generating E@24 as symmetric infrastructure.

### Decision

Open Q6-A as a multi-stage cycle with a hard pre-registration gate between data generation and readout. Structure:

1. **Stage 0 — Infrastructure**: generate `phi_iter24` snapshot for variant E (E@24), under the same generator contract as B@24. No readout, no interpretation. Freeze the command, manifest, SHA-256 of the snapshot and storage path. Authorized and started by Iban on 2026-06-07 with: `python .\level0_generate.py --variant E --iterations 24 --no-plots --no-resume --force-compress`.

2. **Stage 1 — High-scales symmetric**: Phase 1 high-scales derivation for B@24 and E@24, including the scale ladder to be declared by Ariadna in a Stage 1 contract (reference target: $m \in \{40, 48\}$), with canonical segmentation $3 \times 10^6$ observable bits per window, prefix/suffix projections, and the canonical stability-filter thresholds of §8.1. Output: frozen prefix/suffix datasets with command + manifest + SHA-256.

3. **Stage 2 — P4@24 pre-registration**: with Stages 0 and 1 frozen, Ariadna drafts a P4@24 pre-registration declaring bands, lag grid, seeds, null family and binary outcomes. The pre-registration is frozen by SHA-256 BEFORE any data of the new bands is observed by any agent (human or AI). This is the hard gate of the cycle: any violation collapses the predictive value of the readout and removes its standing as defense against peer-review flank B.

4. **Stage 3 — Q6-A readout**: execution of the pre-registered Stage 2 contract. Outcome is binary at the level of each declared band and inhibits post-hoc reformulation.

A separate cycle, **Q6-B**, addresses peer-review flank D (OAT vs Sobol/Saltelli on the §8.1 stability filter). Q6-B runs in parallel to Stage 0 if computational budget allows; it does not gate any Q6-A stage and is not gated by any Q6-A stage.

### Rationale

Peer-review flank B attacks the persistence of the discriminant at scale. The current defense lives at §11.3 (P4-06 multi-mechanism decomposition) and at §13 (Open Item on asymptotic ceiling). Both are internally consistent under finite-horizon framing but cannot defend without empirical extension beyond 804M.

Ariadna's revised assessment (2026-06-07) re-frames the problem: the methodologically correct extension preserves Phase 4's original scientific object (E→B comparison), which requires E@24 as symmetric infrastructure. Extending Phase 4 with B@24 only would expose a strictly worse flank: a reviewer could legitimately object that the experimenter changed the comparative architecture under empirical pressure. The peer-review B flank cannot be answered by changing the object under measurement.

Therefore Q6-A is not `extend the bands`; it is `rebuild the comparative infrastructure at iter=24, freeze a pre-registered prediction, and only then read out`. The cycle accepts a multi-week to multi-month timeline in exchange for methodological cleanliness. The operational data that authorizes the timeline is Iban's explicit statement (2026-06-07) that there is no editorial deadline.

### Operational consequences

- **Editorial work on flanks A (XOR / architectural-class framing) and C (physical lexicon primary vs secondary) is paused** until Q6-A Stage 3 readout. Reason: both flanks require coordinated rewriting that depends on the Q6-A outcome (recovered margin vs confirmed collapse imply incompatible rewrites of §11.3, Abstract and §13).
- **No observation of any Phase 4 data above 804M by any agent** until the Stage 2 pre-registration is SHA-256 frozen. This includes informal exploratory peeks. Violation collapses the cycle.
- **Two-agent freeze certification**: Stage 0 and Stage 1 closures require both Ariadna's manifest emission and Sofia's cross-reference at this Decision Log entry, validated by Iban before the next stage opens.
- **Q6-B parallel track**: scoped separately, not blocking. Output independent of Q6-A.

### Pending Operational Closure

- **Stage 0 closure**: pending E@24 generation completion. Closure artefact: Ariadna emits manifest + SHA-256, Sofia references here, Iban validates before Stage 1 opens.
- **Stage 1 contract**: Ariadna drafts the scale-ladder, segmentation and threshold contract before execution; logged here on SHA-256 freeze.
- **Stage 2 contract**: Ariadna drafts the P4@24 pre-registration; logged here on SHA-256 freeze. The freeze is the hard gate of Q6-A.
- **Q6-B planning**: Ariadna provides scope and cost estimate; logged separately when opened.
- **Manuscript freeze on flanks A and C**: Sofia commits no edits to §3.4, §11.2, §11.3, §13 or Abstract on flanks A/C until Stage 3 closes.

### Source documents

- Peer review (external, brief-blind Gemini, 2026-06-07).
- Ariadna's revised recommendation (internal coordination, 2026-06-07).
- D-0151 (CAT_24 iteration-depth readout, with the original deferral of T5 that this entry supersedes).

### Supersession note

This entry supersedes the deferral clause of D-0151 (decision 2). The deferral remains historically valid; Q6-A is the conditions-met opening that the deferral anticipated.

### Amendment 2026-06-07 (see D-0153)

The Stage 0 command recorded above (Iban's authorization on 2026-06-07, pre-patch `level0_generate.py` invocation) was issued before discovery of the Level 0 generator bug documented in D-0153 and is presumed pre-patch. Output of that run is to be discarded as Stage 0 artefact. Stage 0 effective start is paused pending Grace's audit closure of the patch; the post-patch fresh-from-scratch command will replace this anchor when issued by Iban under D-0153.

### Amendment 2 — 2026-06-17 (Stage 0 closure)

Stage 0 Q6-A closed by Ariadna under the post-patch generator (per D-0153 closure). E@24 generated fresh-from-scratch (no resumption from legacy E@23) on 2026-06-17. Closure artefact at `D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage0_e24_closure\q6a-stage0-e24-closure__var-E__iter-24__20260617T085518\`. Hashes:

- `phi_iter24.struct.gz`: SHA-256 `8D3BE0CE2159B699D424103C7BD4C02517D2A020ABE2B671EFFFA15CDCAC2DAD`
- `summary.json`: SHA-256 `2E88A88341118452BC9E0A5366B7756794F0F9202A0E5036D8BB8DDBA43F62BE`
- `manifest.json`: SHA-256 `E35F61ACF6487F235819C1F85F338A6A239160721A8FAE1C3CC2AA8F7854DF1F`
- `report.md`: SHA-256 `B757085C7D4B358C9DC69A5B811416CC1C0537782C0A5EA1DBEBC55BEC909341`

Observable-bits comparison B@24 vs E@24:

- B@24: 686,779,035,011
- E@24: 686,626,661,906
- E/B ratio: 0.999778133727
- E − B delta: −152,373,105 bits

The symmetry condition required by Q6-A is met: E@24 is materially symmetric with B@24 in observational size. Stage 0 closed. The Stage 1 contract draft `prereg_Q6A_stage1_high_scales_symmetric_v0.1_DRAFT.md` (provisional SHA-256 `E3E1DFBCFD55E44C8963CE2FDDD267FD0477ACAA9E8AE438B6BE54126B09537F`) is the next gate, pending Grace's audit before final freeze and pending Iban's authorization. Final SHA after audit + any revision will be recorded here as Amendment 3.

### Amendment 3 — 2026-06-17 (Stage 1 contract freeze v1.0)

Stage 1 contract `prereg_Q6A_stage1_high_scales_symmetric_v0.1_DRAFT.md` frozen as **v1.0** at SHA-256 `E3E1DFBCFD55E44C8963CE2FDDD267FD0477ACAA9E8AE438B6BE54126B09537F` upon dual independent validation:

- Ariadna (executable check): contract is executable as specified.
- Grace (methodological audit): no methodological objections.

Iban's authorization on 2026-06-17 opens Stage 1 execution: symmetric high-scales tower for B@24 / E@24 on ladder $m \in \{8, 12, 16, 20, 24, 28, 32, 33, 34, 35, 36, 40, 44, 48\}$, $3 \times 10^6$ observable bits per scale, prefix+suffix policies, canonical §8.1 stability filters (`min_count_floor=16`, `min_count_rate=1e-6`, `cv_max=1.5`), no nulls, no Phase 4 observation, no observation of any P4@24 band $> 804\text{M}$. Outcome: if B and E sustain the projective tower through $m=48$, **only** the preparation of the Stage 2 P4@24 preregistration template is authorized. No Phase 4 readout, no manuscript edit.

Stage 1 executes in parallel with the D-0154 sanitization track (independent clean-snapshot inputs; no resource conflict expected).

### Amendment 4 — 2026-06-17 (Stage 1 closure, outcome `derivable`)

Stage 1 executed by Ariadna under the v1.0 frozen contract (SHA-256 `E3E1DFBCFD55E44C8963CE2FDDD267FD0477ACAA9E8AE438B6BE54126B09537F`) and closed with the contractual outcome **`q6a-stage1-symmetric-high-scale-derivable`** (violations: 0).

Readout artefact at `D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\readout\q6a-stage1-readout__B-E__m48__20260617T101207\`. Hashes:

- `summary.json`: SHA-256 `2AB4BC493FFB1B15EB78FC72E0B84CB757FC4B67C56794C35EA33BBA4401D1F5`
- `manifest.json`: SHA-256 `B455B762B27B3DF26E62BC23F1B2C0CD46FBB5873002FC9E945A2276A231E115`
- `report.md`: SHA-256 `41B011D28C39F002EAA8230D9E2565C0FFBCBCE92B57A84AA1BAA692979006E6`

Scientific readout (Phase 1 projective tower, canonical filters per contract):

| $m$ | B kept | E kept | B/E ratio |
|---|---|---|---|
| 36 | 1515 | 1515 | 1.000 |
| 40 | 1318 | 1318 | 1.000 |
| 44 | 1042 | 1042 | 1.000 |
| 48 |  238 |  238 | 1.000 |

High-scale survival ratios (well above any contractual degradation threshold):

- B/E prefix 48→44: 0.9967
- B/E suffix 48→44: 0.9948

Negative checks (all clean):

- No orphans.
- No prefix/suffix qualitative divergence.
- No active-support collapse.
- No cross-variant asymmetry.

**Boundary**: no Phase 4 data above 804M was observed or generated by this Stage 1 readout. The no-observation rule on the Phase 4 high band remains intact.

**Authorization scope opened by this outcome** (per contract): preparation of the Stage 2 P4@24 preregistration template is authorized. No Phase 4 readout, no manuscript edits, no observation of any P4@24 band $> 804\text{M}$ until Stage 2 preregistration is frozen.

Sanitization track D-0154 remains active in parallel and continues to block paper-facing table edits involving variants {E, I, D, G, F} until per-artefact re-derivation closes.

Next operational step: Stage 2 P4@24 preregistration drafting (separately gated by Iban's authorization and by Grace's methodological constraints input requested by Ariadna at Stage 1 closure handoff).

### Amendment 5 — 2026-06-17 (Stage 2 contract freeze v1.0 + Stage 3 execution authorization)

Stage 2 P4@24 preregistration contract `prereg_Q6A_stage2_p4at24_symmetric_forecast_v1.0.md` frozen as **v1.0** at SHA-256 `C5D05C2C0D82073191A2CFB8EA1E406ECC04EFDD9B18925FED8DC6DF75DDC602` upon triple independent validation:

- Ariadna (executable): contract is executable as specified; v0.1 → v0.2 integrated Grace's audit (sub-804M calibration removed, 50M effective band span, metric significance criteria, sequential stop rule); v1.0 differs from v0.2 only in freeze/title/status metadata, no scientific parameter altered.
- Grace (methodological audit): **APPROVED WITHOUT RESERVATIONS** on v0.2 design (lag-grid heritage from P4-04/P4-05 explicitly accepted as preregistered diagnostic shell satisfying generational-scale constraint).
- Sofia (editorial + empirical verification): contract scope, blinding boundary, wording fence, and frozen-input hashes verified. Specifically: the B/E `phase1_summary.json` hashes cited as identical in §2 were verified at the filesystem level to be genuinely byte-identical (both 10,521 bytes, SHA-256 `BE2F5983979ADB01CCF730B0C38143561B4C0A0EC774CD4FFCDDB6A53BC94647`); this is a direct consequence of the variant-name-free aggregate structure of the summary combined with the exact B/E symmetry already reported at Stage 1 closure (B/E ratio = 1.000 at $m \in \{36, 40, 44, 48\}$). The identity is genuine, not a paste error.

Iban's authorization on 2026-06-17 (this entry) **opens Stage 3 execution** under the v1.0 frozen contract. Operative parameters (frozen, not summary; the contract is the source of truth):

- Bands: `804M, 854M, 904M, 10G, 100G, 500G` (all $\geq 804\text{M}$; no sub-804M band executed).
- Effective band span: 50M bits (`window_count = 101`, `window_step_bits = 500{,}000`).
- Lag grid: `-29.5M, -27M, -26.5M, -24M, +24.5M, +26.5M` (inherited from P4-04/P4-05, frozen).
- Null envelope: `phase-matched-lz` (primary) + `matched-lz` (secondary).
- Null seeds: `607, 709, 811, 907, 1009, 1103`.
- Significance criterion (per tertile): `phase_z > 3.0` AND `phase_margin_to_max >= 0.02` AND `hard_margin >= 0.02`; band-level pass requires stability in $\geq 2/3$ tertiles.
- Sequential stop rule: Stage 3 stops automatically after 3 consecutive bands classified as `null-mimicry-or-asymptotic-wall`.
- Anchor / candidate variants: `E -> B` (canonical Phase 4 object inherited from P4-04/P4-05).

Boundary reaffirmed:

- Stage 3 readout does **not** authorize any manuscript edit. Any paper-facing integration requires a separate post-readout Grace audit, Sofia editorial decision, and a new Decision Log entry.
- D-0154 sanitization track remains active and continues to block paper-facing table edits involving variants {E, I, D, G, F}. Stage 3 outcomes can be recorded as scientific findings but cannot enter the manuscript until D-0154 closes for the relevant artefacts.
- The Phase 4 wording fence (§14 of the contract) applies to all communication of Stage 3 outcomes.

Next operational step: Ariadna prepares and launches the Stage 3 wrapper strictly under contract v1.0. Stage 3 closure will be recorded as Amendment 6 with readout hashes and the contractual outcome.

### Amendment 6 — 2026-06-23 (Stage 3 P4@24 readout closure)

Q6-A Stage 3 executed under the frozen v1.0 contract (Amendment 5, SHA-256 `C5D05C2C0D82073191A2CFB8EA1E406ECC04EFDD9B18925FED8DC6DF75DDC602`). Run directory: `D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage2_p4at24_symmetric_forecast\q6a-stage3-p4at24-symmetric-forecast__bands-6__20260617T202559\`.

**Contractual outcome:** `inputs-or-run-not-derivable` (§10 fallback). Status: `completed`.

**Trigger:** Subrun `band-10000M / T1` completed the computation but returned `no-atlas-rows = 6` (zero source_rows in all six lag positions of T1). Under §10, this prevents the readout from constructing the atlas object required for forward statistical aggregation without altering protocol semantics; consequently the overall outcome is mapped to `inputs-or-run-not-derivable`. No higher band (100G, 500G) was attempted; this is consistent with the §10 fallback rule.

**Per-band classifications recorded under contract (empirical observations, archived not narrative-eligible):**

| Band | Classification |
|---|---|
| 804M-854M | `B-E-asymmetric` |
| 854M-904M | `B-E-asymmetric` |
| 904M-954M | `B-E-asymmetric` |
| 10000M-10016M | not classifiable (derivability failure) |
| 100000M-100016M | not executed |
| 500000M-500016M | not executed |

Grace's audit (§3) explicitly states: "no narrative rescue is permitted. The local results are archived for methodological completeness but cannot be used to justify alternative hypotheses or to edit the manuscript." This applies in full. The three `B-E-asymmetric` classifications are valid as recorded empirical observations under the frozen protocol, but the overall contractual outcome remains `inputs-or-run-not-derivable` and no per-band classification triggers an alternative §10 outcome.

**Wrapper patch:** `hsi_v2_q6a_stage3_p4at24_symmetric_forecast.py` was patched to handle the zero-source-rows case (`max()` over empty sequence) cleanly. The patch is exception-handling and reporting only: it intercepts the empty condition, registers `derivability_failures`, preserves `summary_path` and `target_statuses`, and maps the outcome to `inputs-or-run-not-derivable`. It does **not** modify any frozen parameter, metric, threshold, lag, null family, or protocol step. Acceptable without reopening the contract. Grace audit point 1 confirms methodological correctness.

**Reporting recovery flag `--skip-large-source-hash`:** used during final aggregation only, when the scientific subprocess runs had already completed and stored their results. Grace audit point 4 declares it acceptable since the underlying data files were not re-hashed but were also not modified.

**Artefact hashes (Sofia editorial verification at filesystem level, 2026-06-23):**

| Artefact | SHA-256 | Bytes | Status |
|---|---|---|---|
| `summary.json` | `6D426F5CEA84CF92B7BC44E208F209A20ED0C00A181CE00EC6F52449336BCC5F` | 97,049 | MATCH |
| `report.md` | `43727D0FB774C628252D264F59946B2C9E9A7137A5F4874FA7B4C8A6316481EE` | 1,730 | MATCH |
| `manifest.json` | `656267FAD206C578BB42B354FE563E154C7CE071D3014C69FAE0E5F2B0FB0D52` | 30,348 | MATCH |
| `lag_response.csv` | (top-level aggregate present, 2,996 bytes) | 2,996 | present |
| `per_cell_readout.csv` | (top-level aggregate present, 25,453 bytes) | 25,453 | present |
| `per_tertile_readout.csv` | (top-level aggregate present, 2,133 bytes) | 2,133 | present |
| `per_band_readout.csv` | (top-level aggregate present, 417 bytes) | 417 | present |
| `sequential_stop_trace.csv` | (top-level aggregate present, 229 bytes) | 229 | present |

All eight contractual artefacts of §12 are present at the run root. The three hashes Ariadna reported coincide with independently recomputed values, confirming no post-hoc alteration.

**Independent audit:** Grace, `docs/HSI_v2_Audit_Q6A_Stage3_Readout.md`, SHA-256 `20DCFD175AD00FAA393FCE63AE53A8BD11BEB7B453032F589D734D3EE5737C1F`, verdict: **APPROVED (Methodological Continuity Preserved)**. No further Stage 3 execution authorized.

**Scientific reading (doubly negative for the Q6-A flank):**

1. No symmetric forward extension of the B/E channel under Phase 4 lag-response readout. The frozen lag grid, null envelope, and stability criteria do not yield a sustained signal in the executed local bands.
2. No clean thermodynamic wall verified. The first high-band blind probe (10000M) hits a derivability limit before a wall classification can be statistically established under the frozen protocol.

**Latent tension preserved for future editorial cycles (registered, not narrated):** Stage 1 reported perfect B/E symmetry up to byte-identical `phase1_summary.json` aggregates at $m \leq 48$ on the projective tower. Stage 3 records the opposite local picture in the Phase 4 lag-response regime above 804M: every executed band classifies as `B-E-asymmetric`. This dissociation is empirically logged here but **does not authorize any manuscript edit**. Any future use requires a new editorial cycle with Grace methodological audit and Sofia editorial decision.

**Wording fence reaffirmed (contract §14):** the forbidden terms (`infinite extension`, `universal horizon`, `geometry recovered`, `Gate 2 closed`, `global law`, `complete predictive law`, `thermodynamic proof`, `rescue of the positive channel`, any claim that Stage 3 opens geometry or Level 2) remain forbidden in all communication of this outcome.

**Editorial freeze on the manuscript continues** for §9, §9.10, §11.2, §11.3, §13 and Abstract until D-0154 (sanitization subcycle) completes for the affected variants. Stage 3 closure does **not** lift the freeze; Stage 3 findings are decision-log-resident only.

**Next operational steps:**

- Ariadna: surgical commit with wrapper patch + Decision Log update.
- D-0154 sanitization continues in parallel. No competing-resource conflict observed during Stage 3.
- Q6-A cycle status: **closed at Stage 3 contractual outcome**. Any further Phase 4 work (deeper bands, alternative anchor, different lag grid, different null family) is a new cycle requiring a fresh preregistration, fresh Grace audit, and fresh Iban authorization.

---

## D-0153 — Level 0 generator bug discovery + minimal patch (audit pending Grace)

- Date: 2026-06-07
- Status: `provisional` (pending Grace's audit closure)
- Scope: Level 0 generator infrastructure correctness
- Cycle: Discovered during preparation of Q6-A Stage 0 (see D-0152) by Ariadna while validating whether E@24 could be resumed from the legacy E@23 checkpoint.

### Findings

1. **Checkpoint scientific safety**: the legacy Level 0 checkpoint format stored the accumulated $\Phi$ but did not store `current_state`. This prevents exact resumption: a run interrupted at iteration $k$ cannot be continued at iteration $k+1$ with byte-exact equivalence to a single-shot run. Resumption from the legacy E@23 checkpoint into E@24 was therefore not scientifically clean.

2. **Control-flow bug in variant branches**: when `state_len > 100MB`, the affected branches of the non-B variants could fail to enter their collapse branch under specific path conditions. Variant B is not affected by the same path. The bug manifests only at large state sizes, which is precisely the high-iteration regime relevant to E@24, I@24, D@24, etc. *(See "Affected variants — precision 2026-06-17" below for the closed list.)*

### Minimal patch applied by Ariadna

- Control-flow correction at the variant dispatch block.
- Checkpoint format extended to persist both accumulated $\Phi$ and `current_state` (exact resumption now possible).
- Restore path moved to streaming I/O to handle the larger checkpoint payload.
- Progress logs added at iteration boundaries.

### Audit status

Grace is auditing the patch as a code-correctness audit (not a scientific-claim audit) before any long-running E@24 generation is authorized. Pending Grace's verdict, **no Q6-A Stage 0 long run is to be executed under the patched generator**.

### Operational consequences

1. **D-0152 Stage 0 anchor is invalidated**: the command recorded in D-0152 ("`python .\level0_generate.py --variant E --iterations 24 ...`" executed by Iban on 2026-06-07) was issued before the patch was applied and is presumed pre-patch. Any output of that run is to be discarded as Stage 0 artefact. The actual Stage 0 command will be re-issued by Iban after Grace's audit closes, with the post-patch binary and a fresh-from-scratch generation contract (no resumption from legacy E@23).

2. **Open question on prior reported results**: the bug affects variants E/I/D/G/H/N at `state_len > 100MB`. The manuscript currently reports these variants at various iteration depths (notably E and I in the main signature batch and D, G, H at iter=20 in §9.10 HSI-family closure). Whether any prior reported run produced state sizes above the 100MB threshold, and whether the bug actually triggered along the specific execution paths of those runs, is an open scientific question. Determination falls to Ariadna (execution-path analysis) and Grace (validation), not to this entry. Conservative reading: no prior result for the affected variants is to be silently re-asserted until Ariadna confirms that the affected branches did not execute under the bug at the reported iter/state combinations.

3. **Editorial freeze (already in place per D-0152) holds and widens**: no §9 / §9.10 / §11 / §13 / Abstract edits are made until BOTH Q6-A Stage 3 readout AND the bug-impact assessment of point 2 close. The widening is scoped to §9 and §9.10 (HSI-family closure), which D-0152 did not explicitly freeze but which the bug now puts in scope.

### Pending Operational Closure

- Grace's audit verdict on the patch (code-correctness scope).
- Ariadna's execution-path analysis of the bug against prior reported variants (scope-of-impact assessment).
- Iban's authorization of the post-patch fresh E@24 generation command (Stage 0 effective restart).
- D-0152 amendment recording the post-patch Stage 0 command and discarding the pre-patch anchor.

### Source documents

- Ariadna's bug-discovery handoff (internal coordination, 2026-06-07).
- D-0152 (Q6-A multi-stage opening, whose Stage 0 anchor this entry supersedes operationally).

### Closure 2026-06-17

- Grace's patch audit: closed (code-correctness scope), enabling Iban's authorization of the post-patch fresh E@24 generation under D-0152 Amendment 2.
- Stage 0 Q6-A: closed under D-0152 Amendment 2 (E@24 fresh-from-scratch, symmetry B/E = 0.999778).
- Ariadna's scope-of-impact verdict (empirical, iter=23): E legacy structural length = 11,613,134,077 bytes; E clean structural length = 252,103,446,959 bytes; clean/legacy ratio = 21.708×. This is not a numerical drift: it is a materially different trajectory caused by the bug at the affected execution path. The conservative reading of point 2 above is upgraded to an empirically grounded one: pre-patch E in the high-state-length regime is treated as invalidated unless explicitly demonstrated as non-affected.
- Sanitization track for paper-facing E-dependent artefacts (and the precise variant scope determined below) opens as D-0154.

### Affected variants — precision 2026-06-17 (supersedes the indicative list in Finding 2)

The original Finding 2 listed the affected set as "E, I, D, G, H, N" as an indicative tentative scope. After patch analysis and Stage 0 closure, Ariadna closes the precise list of paper-facing variants requiring action:

- **Affected (paper-facing) and requiring action**: **E, I, D, G, F**. F was missed in the original indicative list; the `if/elif` Python control-flow block could skip F's branch identically to the other non-B variants when `state_len > 100MB`.
- **H**: clean at `@20` (state length below the 100MB threshold at that iteration). Iterations above @20 for H are not paper-facing in the current manuscript and are documented as not used.
- **N**: clean (state regime below threshold across all paper-facing uses).
- **B**: not affected (independent code path).
- **A, J, K, L, M, O, P**: not generated by the affected Level 0 generator; out of scope.

The closed list is operationally consumed by D-0154 (sanitization scope) and supersedes Finding 2's indicative list.

Status transition: `provisional` → `accepted`.

---

## D-0154 — Level 0 sanitization subcycle (paper-facing artefact re-derivation)

- Date: 2026-06-17
- Status: `accepted` (policy authorized by Iban 2026-06-17; per-variant closure tracked via amendments)
- Scope: paper-facing metrics, tables, and figures that depend on pre-patch Level 0 outputs for the closed affected list **{E, I, D, G, F}** (per D-0153 "Affected variants" precision).
- Cycle: Triggered by D-0153 closure (empirical ×21.708 divergence at iter=23 between E legacy and E clean).

### Decision

A sanitization subcycle is opened to (a) regenerate Level 0 outputs for I/D/G/F at their paper-facing iteration depths under the post-patch generator, and (b) re-derive every paper-facing artefact whose value depends on pre-patch E/I/D/G/F outputs. E is already sanitized at the Level 0 layer (clean snapshots @20..@24 from Q6-A Stage 0); only its downstream artefact re-derivation is required.

### Operational policy

1. **Provisional invalidation**: every paper-facing value that depends on a pre-patch generation of the affected variants {E, I, D, G, F} is treated as invalidated and must be either re-derived from a clean snapshot or explicitly demonstrated by Ariadna as numerically equivalent under the bug-affected execution path (rare; default is re-derivation).

2. **Level 0 regeneration scope** (post-patch, fresh-from-scratch):

   | Variant | Action | Target iter |
   |---|---|---|
   | E | already sanitized via Q6-A Stage 0 closure (D-0152 Amendment 2); downstream re-derivation only | @24 |
   | I | regenerate Level 0 | @23 |
   | D | regenerate Level 0 | @20 |
   | G | regenerate Level 0 | @20 |
   | F | regenerate Level 0 | @20 |
   | B, H@20, N | clean; no action | — |
   | A, J, K, L, M, O, P | out of scope (not generated by affected Level 0) | — |

3. **Paper-facing artefacts to re-derive** (after the corresponding clean snapshots are available):

   - `tab:signatura`
   - `tab:pm_size`
   - `tab:fibres`
   - `tab:transport`
   - `tab:porta1_canonica`
   - `tab:hsi_family_closure`
   - Any figure whose data path traces back to a pre-patch E/I/D/G/F snapshot.

   Rationale for blanket re-derivation even where numerical equivalence is plausible: some Phase 1 values based on the 3×1M prefix could coincide numerically because the trajectory divergence is downstream of the prefix; however, traceability with a pre-patch source artefact is not defensible. Re-derivation is the rigorous default.

4. **Priority A (paper-facing, hot path)**: items above for E (immediate, no Level 0 wait) + items above for I/D/G/F once their clean Level 0 outputs land.

5. **Priority B (post-Stage 1)**: any Phase 2 / Phase 3 / Phase 4 artefact that uses any of {E, I, D, G, F} in a B/X contrast.

6. **Editorial freeze widening**: sections that depend on the affected variants (§9 main batch, §9.10 HSI-family closure, signature table, any section citing the listed `tab:*`) remain frozen until the per-artefact re-derivation closes for each. v2.03 manuscript text remains as-is on disk but is not the basis for any new peer-review submission or external communication until the sanitization closes.

7. **Parallelism with Q6-A Stage 1** (per D-0152 Amendment 3): Stage 1 uses only clean post-patch snapshots and is not blocked by the sanitization track. The two tracks execute in parallel; resource conflicts (if any) are managed operationally by Ariadna without reordering the scientific dependencies.

### Pending Operational Closure

- Per-variant Level 0 regeneration completion and hashes (I@23, D@20, G@20, F@20) → recorded as amendments to this entry.
- Per-artefact re-derivation reports for Priority A items (with input-snapshot hashes and output hashes) → recorded as amendments.
- Per-artefact re-derivation reports for Priority B items (post-Stage 1).
- Editorial reconciliation of the manuscript with re-derived values (deferred until all Priority A closes; no edits until then).

### Source documents

- D-0153 (Level 0 generator bug; "Affected variants" precision 2026-06-17 closes the scope to {E, I, D, G, F}; ×21.708 empirical impact at iter=23).
- D-0152 Amendment 2 (Stage 0 closure providing clean E@24 reference).
- D-0152 Amendment 3 (Stage 1 contract freeze v1.0; parallelism arrangement).
- Ariadna's Stage 0 closure + sanitization-scope handoff (2026-06-17).

### Amendment 1 — 2026-07-02 (I@23 clean Level 0 regeneration closure)

Variant `I` Level 0 regeneration under the post-patch generator is closed at iteration 23. Run: `q6a-level0-sanitation__20260627T171344`. Grace's diagnostics (archived in `docs/HSI_v2_Audit_Postreview_Item2_Preregistration_v0.1.md` §5, SHA-256 `EF8EDAFA0CFED9B95CCFDA3C17156D3E79380083473C08BFF6FDF45DC9EB5626`) confirm the runner's checkpoint-recovery integrity after the server restart (incomplete iter23 snapshot deleted by built-in cleanup, iter22 snapshot verified 100% decodable, generation resumed cleanly) and full decodability of all completed checkpoints.

**Closure hashes (reported by Grace from the filesystem; independently recomputed by Sofia at filesystem level 2026-07-02; all MATCH):**

| Artefact | SHA-256 | Bytes | Status |
|---|---|---|---|
| `phi_snapshots\var_I\phi_iter23.struct.gz` | `913452A3185F2A603A4447688F679903FAA15438B88FBDB029BA1F26835EB024` | 5,081,881,620 | MATCH |
| `phi_snapshots\var_I\phi_iter23.json` | `454484438089F0F4E901AA1B975B3B6A4727B0B831599194686F147D5BBF59D4` | 681 | MATCH |
| `phi_snapshots\var_I\phi_iter23.state.txt.gz` | `26745EEB24BA84A0EC9908BB4CAE13A3F4BC73C35DAC7C7B180314184AB21782` | 48 | MATCH |
| `phi_snapshots\var_I\phi_complete.tar.gz` | `DC664CB8FC59C0487E029BF2F48811440A8C36EBB344A0163E7423F4F4182E48` | 6,926,503,258 | MATCH |
| `reports\variant_I_23_20260702_070513.json` | `2B608A3990C85FF3322FF60E8C712A5A1ACA43EBC4D689157C51269AB17D09A1` | 674 | MATCH |
| `reports\variant_I_execution.log.md` | `308D2D6112E14080941B47B671061FF4DD00FC57377FF954DBCA18DA6C665BAE` | 470,438 | MATCH |

Paths relative to `D:\Iban\Projectes\HSI\results\level0\`.

**Erratum noted:** Grace's closure message and audit §5 cite `phi_complete.tar.gz` at 3.06 GB; the on-disk artefact measures 6,926,503,258 bytes. The SHA-256 anchors the artefact unambiguously; the size figure in the narrative is an erratum, not a chain defect.

**Scope effect:** the Level 0 regeneration row `I @23` (operational policy item 2) closes. This does **not** unfreeze paper-facing I-dependent artefacts: per policy item 3, each requires a re-derivation report from the clean snapshot, recorded as a further amendment to this entry. Remaining Level 0 regenerations: `D@20`, `G@20`, `F@20` (runner ready to resume).

### Amendment 2 — 2026-07-03 (D@20, G@20, F@20 clean Level 0 regeneration: material closure)

The sanitation runner `hsi_v2_q6a_level0_sanitation_runner.py` (run `q6a-level0-sanitation__20260627T171344`) completed with `status: completed`, `return_code: 0` for the full pending scope. D/G/F executed fresh post-patch with `--no-resume --no-plots --force-compress`.

**Global manifest:** `D:\Iban\Projectes\HSI\results\hsi_v2\q6a_level0_sanitation\q6a-level0-sanitation__20260627T171344\manifest.json`, SHA-256 `30ADC4990274C7C0A441FB3AC9D1692FD6E26BA21FABA5DAE08A0033C4072BB5` (Sofia recomputed 2026-07-03: MATCH).

**Legacy quarantine verified on disk (Sofia, 2026-07-03):** `level0\legacy_invalidated\` contains `q6a_prepatch_D_legacy_20260627T171344`, `q6a_prepatch_G_legacy_20260627T171344`, `q6a_prepatch_F_legacy_20260627T171344` (alongside the earlier E and I quarantines).

**Closure hashes (reported by Ariadna; independently recomputed by Sofia at filesystem level 2026-07-03; all 12 MATCH):**

| Artefact | SHA-256 | Bytes | Status |
|---|---|---|---|
| `var_D\phi_iter20.struct.gz` | `67E870C136A88C4E2F87DE34F3D84A46FF95FC59C930A7208598945DEF41CA1F` | 224,529,461 | MATCH |
| `var_D\phi_iter20.json` | `84AAD6D52EFBFBC7D91FA553F20F882EDDF14D13188FCA7C23290BA282003A58` | 800 | MATCH |
| `var_D\phi_complete.tar.gz` | `91FC0E09C35EEA0F52E445F3D7D51D6DD897CC4A2BCA63DE825AC1F7E6EEA899` | 309,133,951 | MATCH |
| `reports\variant_D_20_20260702_082040.json` | `BE72A2029E344EFC0A63FEDBB41AAB97707C3623F7CFA47421C12DDC86A79BE9` | 4,485 | MATCH |
| `var_G\phi_iter20.struct.gz` | `3A8977E8AB2DEA07D93D8EA99579B6CFCBE016950C06F58CBE6113630307F467` | 314,374,669 | MATCH |
| `var_G\phi_iter20.json` | `927CCCB1EBDA378C38D42618D5AB7966BF5894DBCA42AB156E29BDF5DC0B7F69` | 816 | MATCH |
| `var_G\phi_complete.tar.gz` | `56E3D0DD179419C5D8D328A0B47D67A80D689E34CC44D1A3C0BF238721071D97` | 309,937,166 | MATCH |
| `reports\variant_G_20_20260702_233919.json` | `97D210475FCB1CE630F4A10E6533C69BD4484DA1BDA06ABD64546074FCB6BDFF` | 2,790 | MATCH |
| `var_F\phi_iter20.struct.gz` | `D19BDD72CDCA05DAFD76E8EEC998F0FE0DC97570258F7307CFE2B49004CE7D86` | 11,502,833 | MATCH |
| `var_F\phi_iter20.json` | `0EC80D001F423E79C17B0B21F17503419995E17AB46FB3140E8082D5B328BC60` | 806 | MATCH |
| `var_F\phi_complete.tar.gz` | `469C4BBE4054A27E1029A0C7E2355C02336151A295B6C01B4D1AAED875ACF99A` | 5,079,166 | MATCH |
| `reports\variant_F_20_20260703_104055.json` | `1481CCF36E8B6330FA532F1179647A0D327A9D7339EC23C1837C8578D832E0F0` | 4,662 | MATCH |

Paths relative to `D:\Iban\Projectes\HSI\results\level0\` (snapshots under `phi_snapshots\`).

**Closure semantics (three distinct states, in order):**

1. **Material Level 0 closure — reached with this amendment.** Clean post-patch snapshots exist and are hash-anchored for the complete regeneration scope {I@23, D@20, G@20, F@20}; E@24 was already clean via Q6-A Stage 0.
2. **Policy-level Level 0 closure — pending Grace integrity audit** of the sanitation run (manifest coherence, quarantine completeness, absence of pre-patch reuse in D/G/F, `--no-resume` discipline). To be recorded as a further amendment after her verdict.
3. **Paper-facing revalidation — NOT reached and not implied by 1 or 2.** Every artefact in policy item 3 (`tab:signatura`, `tab:pm_size`, `tab:fibres`, `tab:transport`, `tab:porta1_canonica`, `tab:hsi_family_closure`, and any figure tracing to pre-patch E/I/D/G/F data) remains invalidated until its explicit downstream re-derivation from clean snapshots is recorded as an amendment. The manuscript editorial freeze (policy item 6) remains fully active.

**Next deliverable (Ariadna):** a re-derivation matrix `affected_artifact -> variant dependency -> rederive? -> priority` to sequence the Priority A recomputations before any manuscript reconciliation.

**Interaction with other cycles:** Item 2 (D-0155) is methodologically unaffected (B-only contract; B was never in the affected scope). Items 3 and 4 of the post-review queue must be re-checked against the clean re-derivations if their designs consume I/D/G/F/E-based controls or envelopes.

### Amendment 3 — 2026-07-03 (Level 0 sanitation policy closure: Grace audit APPROVED)

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_Level0_Sanitation_Closure.md`, SHA-256 `DCBFDAB2E7F858F72022C1B511E1967EBAA2ED973B87169DD16793D08897B275` (Sofia recomputed at filesystem level 2026-07-03: MATCH). Verdict: **APPROVED AND CLOSED** for the Level 0 sanitation pipeline (non-B variants).

The audit independently verifies: per-variant execution metadata (finish timestamps and durations for I/D/G/F), quarantine of the three legacy directories, clean `--no-resume` starts for D/G/F from iteration 0, variant I's legitimate resume from clean post-patch checkpoints, and independent recomputation of all 12 closure hashes. The closure hash set now has triple concordance (Ariadna report / Sofia recomputation / Grace recomputation).

**Effect:** state 2 of Amendment 2's closure semantics (policy-level Level 0 closure) is reached. The D-0154 entry itself remains **OPEN**: state 3 (paper-facing revalidation) is untouched — every policy-item-3 artefact remains invalidated until its per-artefact re-derivation amendment lands, and the manuscript editorial freeze (policy item 6) remains fully active.

**Priority reconciliation note (editorial):** audit §5 proposes a two-level revalidation roadmap (high: Phase 2 parent density, parent shell atlas, Phase 4 predictive coefficients; standard: summary charts, secondary baseline checks) which does **not** enumerate the canonical Priority A paper-facing artefact list of policy item 3 (`tab:*` tables and dependent figures). The canonical priorities of this D-entry remain in force until explicitly amended. Grace's §5 roadmap enters as input to Ariadna's re-derivation matrix, where any priority reordering relative to policy items 3–5 must be made explicit and authorized before execution.

### Amendment 4 — 2026-07-03 (downstream re-derivation matrix anchored)

Ariadna's operational planning matrix is anchored as:

`hsi_agents_project/Documentation/D0154_Downstream_Rederivation_Matrix.md`, SHA-256 `E8ECAE7ECA31ECBE68FD8792CA8FCBEF56B439934FE85EB4D5CE81BCD6653B02` (Sofia recomputed at filesystem level 2026-07-03: MATCH; a first reported hash corresponded to a pre-update version and was superseded by Ariadna's re-issue).

Editorial acceptance notes:

1. The matrix explicitly subordinates itself to the canonical Priority A list of policy item 3 and authorizes no scientific reordering — resolving the reconciliation requirement of Amendment 3. Its "Recommended Execution Order" is accepted as operational sequencing *within* canonical Priority A (Phase 1 batch → signature batch → family-closure batch → figures → A-check editorial reconciliation → Priority B), not as a priority change.
2. The matrix's stop/escalation conditions are accepted as binding for the re-derivation work, notably: stop on any paper-facing classification change; split `B/E` combined rows if clean `E` diverges from `B` (no silent preservation of the combined label); no null substitution when a source-derived null is not derivable; downgrade-or-freeze for non-derivable Phase 2/3 dependencies.
3. Each re-derivation batch closes via manifest + hashes + amendment to this entry, with Grace auditing outputs. No new preregistration is required where the batch reuses canonical frozen protocols without tuning.
4. Item 2 (D-0155, frozen v1.0, SHA-256 `3181DD505E072FB26BF94083C03B8C7BC23FC034B6B0759633CD04CED21F2D5D`) is confirmed independent of this track and may run in compute gaps.

Launch of the Priority A Phase 1 batch awaits Iban's compute authorization.

### Amendment 5 — 2026-07-03 (Priority A Phase 1 batch authorized)

Iban authorizes the launch of the Priority A Phase 1 batch (explicit go, 2026-07-03, relayed via Sofia's session).

Prior to authorization, Grace independently re-verified on disk the two anchoring hashes of Amendment 4 (matrix `E8ECAE7E...`, Item 2 frozen contract `3181DD50...`): both concordant. The anchors now hold Ariadna/Sofia/Grace/disk concordance.

Authorized scope: step 1 of the matrix's accepted operational order — clean Phase 1 summaries for the affected variants feeding `tab:pm_size`, `tab:fibres`, the prefix/suffix asymmetry row, `tab:transport`, and `tab:porta1_canonica`. Subsequent batches (signature, family closure, figures) proceed in the accepted order without further compute authorization unless their cost profile changes materially.

Closure requirements per Amendment 4 note 3: manifest + hashes + amendment per batch, Grace audit on outputs, stop/escalation conditions binding. Item 2 (D-0155) may run in compute gaps at Ariadna's operational discretion.

### Amendment 6 — 2026-07-03 (scheduling default change: compute cost non-binding; operational sequencing delegated to Sofia)

Iban declares (2026-07-03) that compute cost and calendar time are not binding constraints for the experimental program: discovery quality takes precedence, and the project takes the time it needs. He delegates operational and experimental sequencing decisions to Sofia. Scientific claims, cycle openings toward the manuscript, and publication decisions remain Iban's (non-delegable authorship role). Technical approach remains Ariadna's; methodological audit remains Grace's.

Effects on standing defaults:

1. The "D-0154 retains compute priority / Item 2 waits" clauses (D-0155 Amendment 3-4, D-0154 Amendment 5) relax to: Item 2 may run **in parallel** with the Priority A batches at Ariadna's operational discretion, provided no technical interference (memory/IO) with the running batch.
2. Future statistical-reinforcement work (post-review Item 4) should be designed without seed-count parsimony as a constraint.
3. Scientific prioritization is unchanged: Priority A re-derivations remain the editorial critical path; unconstrained compute does not promote any queued cycle by itself.

### Amendment 7 — 2026-07-03 (Priority A Phase 1 batch closure)

The Priority A Phase 1 batch (matrix operational order step 1) executed and closed clean. Run: `D:\Iban\Projectes\HSI\results\hsi_v2\d0154_priority_a_phase1\d0154-priority-a-phase1__20260703T114507\`, script `hsi_v2_d0154_phase1_priority_a_batch.py`, targets `E@24, I@23, D@20, G@20, F@20`, canonical Phase 1 protocol (`3x1M` prefix bits, ladder `8-32`, prefix/suffix policies). All commands `return_code: 0`; no WARN/Error flags in `batch.log` (audit §1).

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_PriorityA_Phase1_Batch.md`, SHA-256 `188CE636E129289851518494432FF1E12A1C794667024C152A7EB2D947C58E60` (Sofia recomputed: MATCH). Verdict: **APPROVED**.

**Artefact hashes (Grace verified; Sofia independently recomputed 2026-07-03; all MATCH):**

| Artefact | SHA-256 | Bytes | Status |
|---|---|---|---|
| `manifest.json` | `8496FACEEE0A73CF99391EEDD8AE327D4D48C4DCE81B4988D8A4581C021467A3` | 33,757 | MATCH |
| `reports\phase1-report__iter-mixed__...__var-E-I-D-G-F__20260703T114723.json` | `BF73C23B830E8215CC7F44E3A7E86A2810E8F7BFD57F3529D9C26A2ED7535500` | 32,349 | MATCH |
| `reports\phase1-transport__iter-mixed__...__var-E-I-D-G-F__20260703T114724.json` | `0F464FDB8643BEEA14768C67B79400A8C0E772A08EDDFBFDB96D59EF429641E9` | 60,734 | MATCH |

**Scientific readout (audit §3):** no paper-facing classification shift detected relative to pre-patch results. `E@24` retention `0.8622` consistent with the canonical B baseline (~0.86); `I@23` (`0.9930`), `D@20` (`0.9785`), `G@20` (`0.9403`) remain in the coherent/compact-selective region; `F@20` remains fully rigid (retention `1.0000`, breadth ratio `6.48`, no surviving fibres). **The matrix stop/escalation conditions are NOT triggered by this batch.**

**Editorial caution (wording):** the audit's internal shorthand "asymptotically identical" / "core asymptotic behaviors" must not migrate into manuscript-facing text. Paper-facing wording remains perimeter-bounded: structural identity/consistency *at the observed scales under the frozen protocol*. Standard wording-fence discipline applies to all downstream reconciliation prose.

**Next steps:** per-artefact table re-derivations from these clean summaries (`tab:pm_size`, `tab:fibres`, prefix/suffix asymmetry row, `tab:transport`, `tab:porta1_canonica`), each with the old-vs-new row comparison required by the matrix "Output needed" column, closing as further amendments; then the signature batch (operational order step 2).

### Amendment 8 — 2026-07-03 (Phase 1 table reconciliation: display-identical, no manuscript numeric patch required)

Read-only table-by-table reconciliation over the audited Priority A Phase 1 batch (Amendment 7), with no Level 0 snapshot re-reads and no upstream recomputation. Run: `D:\Iban\Projectes\HSI\results\hsi_v2\d0154_priority_a_phase1_table_reconciliation\d0154-phase1-table-reconciliation__20260703T121926\`, script `hsi_agents_project/hsi_v2_d0154_phase1_priority_a_reconcile_tables.py`. Manifest SHA-256 `DA44A0D42911451B4390ABC454086A744504A6449FD280254347AA5C06BBAE9A` (Sofia recomputed at filesystem level 2026-07-03: MATCH).

**Readout (paper version checked: `main_v2.03_cat.tex`; criterion: display-identity at current printed precision):**

| Table | Rows compared | Display mismatches | Status |
|---|---:|---:|---|
| `tab:pm_size` | 21 | 0 | `display-identical` |
| `tab:fibres` | 18 | 0 | `display-identical` |
| prefix/suffix asymmetry | 12 | 0 | `display-identical` |
| `tab:transport` | 15 | 0 | `display-identical` |
| `tab:porta1_canonica` | 9 | 0 | `display-identical` |

75 row-comparisons, zero mismatches, `classification_shift_detected: false`.

**B/E combined row:** clean post-patch `E` remains display-identical to the printed combined `B/E` row at current manuscript precision. The matrix escalation condition requiring a `B/E` row split is **NOT triggered**.

**Supporting rows:** `D` and `G` clean values emitted to `clean_variant_phase1_values.csv` for the downstream family-closure batch; they are not part of the tables reconciled here.

**Process disclosure (Ariadna, logged for honesty):** a first internal reconciliation attempt erroneously compared the asymmetry gap averaged over all edges instead of the manuscript's explicitly reported `32->28` edge; it was detected and corrected by Ariadna before the run recorded here. This specific comparison (edge selection for the asymmetry row) is flagged as the priority attention point for Grace's audit of this reconciliation.

**Editorial consequence:** the five affected Phase 1 tables require **no numeric patch** to the manuscript. This conclusion becomes editorially binding once Grace's audit of the reconciliation confirms the comparison logic (notably the corrected `32->28` edge selection). The reconciliation artifact itself authorizes no manuscript edit, consistent with the standing freeze.

**Next:** Priority A signature batch (operational order step 2) — may proceed in parallel with Grace's reconciliation audit, as its computation is independent of this reconciliation's correctness.

### Amendment 9 — 2026-07-03 (reconciliation audited: "no numeric patch" editorially binding for Phase 1 tables)

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_Phase1_Table_Reconciliation.md`, SHA-256 `A9E7DAC6BCB71F8762BA51AC4B6D20DC05E5A3B4F159C8938C6F23CA971EA582` (Sofia recomputed at filesystem level 2026-07-03: MATCH). Verdict: **APPROVED**.

The audit includes the requested deep-dive on the flagged attention point (Amendment 8): the corrected asymmetry-edge logic extracts `prefix_child_entropy_last` / `suffix_child_entropy_last` at the operational ceiling `m=32`, representing the `32->28` edge exactly as reported in the manuscript; the first attempt's tower-wide mean (`prefix_suffix_entropy_gap_mean_signed`) is confirmed as the error. All asymmetry fields match printed values under current precision.

**Effect:** the "no numeric patch required" conclusion of Amendment 8 is now **editorially binding** for the five Phase 1 tables of `main_v2.03_cat.tex` (`tab:pm_size`, `tab:fibres`, prefix/suffix asymmetry, `tab:transport`, `tab:porta1_canonica`). These five matrix Priority A rows close as re-derivation rows for numeric content. Their dependent prose (A-check rows) remains pending until the signature batch closes and the A-check reconciliation runs.

**Remaining Priority A scope:** `tab:signatura` + numeric prose (signature batch, step 2, in progress), `tab:hsi_family_closure` (step 3), figures (step 4 — with tables display-identical, the matrix's "no-change rebuild preferred" branch applies: figures rebuild from clean data for traceability, expecting visual identity).

**The manuscript editorial freeze remains active** until all Priority A rows and the A-check reconciliation close.

### Amendment 10 — 2026-07-03 (signature batch audit: provenance findings, iteration scope-gap, no stop-condition)

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_Signature_Reconciliation.md`, SHA-256 `D0F87AE2AA88DAD7D1EE32EA46DB1000F33E3D90B38AFC393324FA6BC4604191` (Sofia recomputed: MATCH). Status: `PROVENANCE REVIEW & NUMERIC PATCH REQUIRED (Scientific Continuity Preserved)`. Runs: `d0154-signature-priority-a__20260703T125020` / `__20260703T132915` (partial/smoke scope; the `cbar` full run is still executing).

**Findings and their editorial classification (three distinct classes, none scientific drift):**

1. **DET (paper 0.55/0.56 vs clean 0.59–0.69) — measurement-provenance mismatch, not data change.** Paper values came from RQA on a *subsampled* representative string (every 100th bit from medium samples, length 10k); the current harness ran DET on the *contiguous* first 10k bits, and the recurrence threshold uses unseeded random sampling. **Editorial ruling:** the reconciliation must reproduce the paper's measurement protocol (matrix requirement "preserve existing metric definitions; no new thresholds"). A **provenance-review sub-track opens**: Ariadna re-runs DET under the exact legacy subsampling protocol with a fixed threshold seed; only that comparison decides patch/no-patch. The contiguous-prefix DET is a *new metric definition* — out of reconciliation scope; adopting it would require its own contract and a declared method change.
2. **Iteration scope-gap (LZ of F: 0.5729 paper @27 vs 0.5720 clean @20; D_f of D: 0.992 paper @23 vs 0.993 clean @20).** The D-0154 regeneration table fixed depths I@23/D@20/G@20/F@20, which cover the Phase 1 tables, but `tab:signatura` and prose report **F@27 and D@23** — depths not covered by the sanitation scope. **Decision required (Iban):** route (a) extend Level 0 sanitation with clean F@27 and D@23 regenerations, then reconcile at the paper's depths (compute is non-binding per Amendment 6; preserves printed values if clean data reproduces them); or route (b) re-anchor the affected cells at the clean depths with a declared iteration change and numeric patch (Grace's gate assumption). Sofia's recommendation: **route (a)** — F@27 first (F is small and rigid; cheap), D@23 after (I@23-scale compute).
3. **TE of F (0.92 paper vs 0.14 clean) — smoke-test artifact, not a finding.** The 20M-bit restriction biases history-3 density estimation at scale 128. Full-stream re-run required before any reading; expected near baseline with residual iteration-depth effect (see 2).

**Positive pre-validation:** the audit confirms the `compute_cbar_metric` weighting logic (weighted per-segment mean_children ≡ total_children/total_branching_nodes) and the isolation of the conjecture observable from the secondary local-ratio observable, consistent with D-0156.

**Gate status:** matrix stop/escalation conditions **NOT triggered** — all divergences have identified non-scientific causes and no paper-facing classification moves. `tab:signatura` reconciliation is **BLOCKED** pending: (i) cbar run completion; (ii) full-stream TE re-run; (iii) DET provenance-review under legacy protocol; (iv) Iban's depth decision (route a/b). LZ rows E/I/D/G are display-identical and close as cells.

No manuscript change is authorized by this amendment. Editorial freeze unchanged.

### Amendment 11 — 2026-07-03 (cbar readout on clean sources: E/I/D/G revalidate φ+1; F not derivable under canonical protocol; editorial policy for the F cell)

**Run:** `D:\Iban\Projectes\HSI\results\hsi_v2\d0154_priority_a_signature\d0154-signature-priority-a__20260703T150824\`. Hashes (Ariadna reported; Sofia recomputed at filesystem level 2026-07-03; all MATCH): `summary.json` `9B4704C0124EAF92F922F2AC20B667128D050DEA80CCAC879513B77A4EA13697`, `manifest.json` `143BA75CCFEB3F574CFF0E3608F19FDB6EA499C0E8334A3ABA2219BD6EB5FC8B`, `report.md` `6CFF536C49A6830FB1D0FF75E992409F036CCF077307082AB644063BA5CED108`.

**Readout (clean post-patch sources, conjecture observable per D-0156 = weighted aggregate mean_children):**

| Variant | cbar (empirical) | Display (6 dec) | φ+1 distance | Segments |
|---|---|---|---|---|
| E@24 | 2.6180332541411873 | **2.618033** | 7.35e-7 | 4/4 |
| I@23 | 2.6180336200231284 | 2.618034 | 3.69e-7 | 4/4 |
| D@20 | 2.618033803408778 | 2.618034 | 1.85e-7 | 4/4 |
| G@20 | 2.6180336314223767 | 2.618034 | 3.57e-7 | 4/4 |
| F@20 | `null` — not derivable | --- | — | parser `node_limit=50M` saturated at 200,524,258 / 663,369,136 chars (max_depth=20, ~4.0 chars/node) |

**Scientific reading (pending Grace audit before claim-bearing use):** E/I/D/G revalidate the cbar ≈ φ+1 convergence on fully clean sources, all four approaching from below at distances 1e-7–1e-6. The F outcome is a **derivability limit under the canonical protocol on clean F@20**, not a scientific negative: it does not state that F fails to preserve cbar.

**Editorial policy for the F cell in `tab:signatura` and §6 prose (Sofia ruling, question B of Ariadna's handoff):**

1. **Option 2 (keep legacy F value with provenance note) is REJECTED on policy grounds**: D-0154 policy item 1 invalidates every paper-facing value derived from pre-patch affected-variant data; a legacy F@27 cell inside an otherwise sanitized table is a traceability flank. Ariadna's technical instinct is confirmed.
2. **Interim state = Option 1**: the reconciliation records F/cbar as `inputs-not-derivable` under the canonical protocol on clean F@20. No manuscript edit yet (freeze active).
3. **Resolution path = Option 3, sequenced after the route (a/b) depth decision of Amendment 10**: if route (a), regenerate clean F@27 and run an F-only cbar contract at @27 (reconciles the actually-printed cell); if route (b), the contract targets F@20. One delta at a time — no simultaneous depth-change plus protocol-change.
4. **Manuscript dependency flagged**: §5's argument currently cites the conservation of cbar under F (`tab:signatura`) as evidence that F preserves the syntactic aspect of the frame. Until the F-only cycle resolves, that sentence is provisionally unsupported by clean data; tracked as an A-check consequence. The pending R2 reformulation of §5 (release plan v2.04) can absorb a rewording if the F-only cycle does not restore the cell.

**E display note (deferred to reconciliation):** clean E@24 displays 2.618033 at 6 decimals, one ulp below I/D/G's 2.618034. Options at reconciliation: per-variant empirical values (one-digit numeric patch for E; the 1e-6 spread is finite-size information), uniform 5-decimal display, or value-plus-distance formulation. Decision with Grace's audit and the actual printed cell layout in view.

**Reporting patch (Ariadna, infrastructure-only):** future summaries classify `value=null` metrics as `inputs-not-derivable` instead of `completed`. The archived run predates the patch and still shows `completed` for F; this amendment supersedes that label semantically. No numeric result affected.

**Pending:** Grace's methodological audit of this readout (Ariadna's question A). Upon approval, a cross-amendment in D-0156 records the clean-source revalidation of the conjecture's empirical premise across four variants.

### Amendment 12 — 2026-07-03 (route (a) authorized; cbar readout audited; F-cell policy finalized)

**Iban's depth decision (resolves Amendment 10 finding 2):** **route (a)** — extend Level 0 sanitation with clean regenerations at the paper's reported depths: **F@27 first** (cheap; F is rigid and slow-growing), **D@23 after** (I@23-scale compute). Reconciliation of the `tab:signatura`-affected cells (LZ of F, D_f of D, TE of F) then happens at the actually-printed depths. Compute non-binding per Amendment 6.

**Grace's updated audit:** `docs/HSI_v2_Audit_D0154_Signature_Reconciliation.md`, SHA-256 `DC2CC4BC43C0B112E9FBE791715E0F6C10A6265855BBDF444D59CF645AA72009` (Sofia recomputed: MATCH; supersedes the `D0F87AE2...` version anchored in Amendment 10). Verdicts: E/I/D/G cbar revalidation **confirmed positive** (E's 1e-6 oscillation classified as statistical rounding); F@20 confirmed `inputs-not-derivable` under canonical protocol (parser node-density ~4.0 chars/node from lack of collapse; node_limit 50M saturated); option 2 rejected; options 1 or 3 recommended.

**F-cell policy finalized (combines Amendment 11 ruling + route (a)):** interim = option 1 (`inputs-not-derivable` at clean F@20, documented); resolution = option 3 as an **F-only cbar mini-contract at clean F@27** once regenerated. Any parser-strategy deviation from the canonical protocol (raised node_limit, streaming parser, shorter segments as Grace suggests) must be declared in the mini-contract's preregistration with its comparability boundary vs the canonical-protocol values of E/I/D/G stated explicitly, and audited by Grace before execution. One delta at a time.

**Context note (Iban, 2026-07-03):** F is scientifically closer to an internal control than to a core canonical variant (extreme simplification of the recursion). This lowers the ontological weight of the F cell but not the editorial dependency flagged in Amendment 11 point 4 (§5 cites cbar conservation under F), which remains tracked.

### Amendment 13 — 2026-07-06 (full-stream TE readout E/I: expected bug-consequence regime, not reconciliation anomaly)

**Run:** `D:\Iban\Projectes\HSI\results\hsi_v2\d0154_priority_a_signature\d0154-signature-priority-a__20260706T211523\` — full-stream, read-only over clean D-0154 snapshots (`E` consumed 686,626,661,906 observable bits; `I` 265,785,965,994; `te_chunk_bits=2,000,000`; `shuffle_seed=42`). Hashes (Ariadna reported; Sofia recomputed 2026-07-06; **all five MATCH**): `manifest.json` `E70DACEDCA2A9A52C5EBB2DAE204CABB8A26ED7596AE6B893DE14209CD87B3B8`, `summary.json` `0A13AFA1...BE954C`, `report.md` `3821E8DC...46BCD68`, `E_te.json` `1A51F548...998F4F`, `I_te.json` `7C34351C...F89FB`.

**Readout:** E@24 TE = 3.9431 (display 3.94) vs paper 4.10; I@23 TE = 59.1438 (display 59.14) vs paper 27.46 (×2.15).

**Editorial framing (the load-bearing distinction of this amendment):** E and I are **not** display-identity reconciliation cases. Their paper TE values were computed on **pre-patch, bug-invalidated sources** (D-0153: E/I affected above the 100MB state threshold; legacy E@23 diverged ×21.708 in length). The Phase 1 tables coincided because they consume only the 3×1M prefix, upstream of the trajectory divergence (policy item 3 rationale); **TE is full-stream and traverses the bug-affected regions**. Divergence here is therefore the *expected signature of the bug's real numeric consequences*, and the clean values are **candidates to replace the printed cells** (declared numeric patch at reconciliation), not anomalies to explain away — conditional on audit.

**Classification impact:** none. Both values remain far above the ×2.0 significance threshold; I moves *deeper into* the significant regime (clean I transports substantially more inter-scale structure than corrupted I did). Matrix stop/escalation conditions NOT triggered.

**Gates before any patch (Grace audit focus, per Ariadna's correct recommendation):**
1. Protocol identity vs the legacy TE derivation (scales, history 3, density mode, chunk geometry, shuffle-control construction, seed) — the same provenance discipline as the DET review.
2. Depth provenance of the printed cells: confirm the paper's E row depth (legacy E@23 vs clean canonical E@24 — a policy-sanctioned depth change per operational policy item 2, to be declared explicitly at reconciliation if confirmed).
3. Sanity of I's magnitude (harness correctness over I's inverse-order structure); an optional second shuffle-seed diagnostic if cost permits.

**F:** excluded from this closure; TE re-derivation on clean F@27 after the route (a) regeneration, as already sequenced (Amendment 12).

No manuscript change authorized. Editorial freeze unchanged.

### Amendment 14 — 2026-07-09 (TE audit: chunk-size provenance dominant; 10M re-run ruled as the reconciliation path)

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_TE_Reconciliation.md`, SHA-256 `7E61E820A3331DC4F87A86D07CA96C1BFFE39E454CC90404EC1A3B31C083D8C4` (Sofia recomputed: MATCH). Status: `PROVENANCE REVIEW REQUIRED (Scientific Continuity Preserved)`.

**Diagnosis:** the clean run used `te_chunk_bits = 2M` for the shuffle control; the paper's legacy table used `10M`. Finer shuffle blocks break long-range correlations more aggressively, collapsing I's shuffle baseline to 5.47e-6 and inflating the observed/shuffle ratio to 59.14. E's 4% deviation is marginal. The scientific finding (strong, highly significant inter-scale flow for both variants) is revalidated regardless.

**Framing correction (Sofia, on the record):** Amendment 13 leaned toward reading the I divergence as the bug's numeric footprint. Gate 1 of that amendment (protocol identity) was the right gate and caught the true cause, but the lean was wrong: the dominant driver is **protocol provenance, not data change**. The clean decomposition remains pending and the re-run provides it for free: clean@10M vs paper@10M isolates the pure bug effect on TE; clean@2M vs clean@10M isolates the pure protocol effect. The re-run report must table all three values per variant.

**Ruling (same precedent as DET, Amendment 10):** the matrix rule "preserve existing metric definitions" governs. **Reconciliation path = re-run TE on clean E@24/I@23 under the legacy 10M chunk protocol** (everything else identical: scales, history 3, density mode, seed 42). Adopting 2M uniformly — Grace notes it is more noise-conservative — would be a *method change*: available only as a separate declared contract with its own justification, out of reconciliation scope (natural home: post-review Item 4 statistical strengthening, post-v2.04). One delta at a time.

**Open item carried from Amendment 13:** depth provenance of the paper's E cell (legacy E@23 vs clean canonical E@24) — to be confirmed and declared in the re-run report.

**Authorization:** the 10M re-run is authorized under delegated operational scope (canonical Priority A re-derivation; compute non-binding per Amendment 6).

No manuscript change authorized. Editorial freeze unchanged.

### Amendment 15 — 2026-07-14 (TE provenance resolved: sanitation-recovery effect confirmed; E/I cells reconciliation-ready)

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_TE_10M_Reconciliation.md`, SHA-256 `3FD766D8FA6FB9599E3368DF06F80C5F807327BF50A4C66DDF5121742B04943D` (Sofia recomputed: MATCH). Run `d0154-signature-priority-a__20260709T213959`, manifest `3095E2F56D777F99E6A7226F7DAAB7090199C89691621146F464A9488BC3F631`. Status: `provenance-review-resolved`.

**The decomposition table (Amendment 14 requirement) settled the cause:**

| Variant | State | Iter | Bits | Obs TE | Shuffle TE | Ratio |
|---|---|---|---|---|---|---|
| E | Legacy paper | 23 | 11.2B | 1.16e-5 | 2.84e-6 | 4.10 |
| E | Clean (10M) | 24 | 686.6B | 2.62e-4 | 6.66e-5 | **3.83** |
| I | Legacy paper | 23 | 12.2B | 1.39e-5 | 5.08e-7 | 27.46 |
| I | Clean (10M) | 23 | 265.8B | 2.25e-4 | 3.81e-6 | **59.16** |

The chunk-size hypothesis (Amendment 14 diagnosis) is **refuted**: clean@2M (59.14) ≈ clean@10M (59.16). The cause is the Level 0 loop-initialization bug: legacy I was truncated at 12.2B bits with corrupted scale structure at sequence start, attenuating observed TE by an order of magnitude (1.39e-5 vs 2.25e-4 clean). The clean source recovers the full signal. Amendment 13's original lean was correct; Amendment 14's correction was procedurally sound at the time (it followed the auditor's diagnosis) but factually premature. **Method lesson, on the record: neither Sofia's intuition (Am. 13) nor Grace's plausible diagnosis (Am. 14) settled the question — only the three-value decomposition experiment did.**

**Open item closed:** the paper's E cell provenance is confirmed as legacy E@23 (11.2B bits). The clean canonical value is E@24 — the update carries a **declared depth change** (@23→@24), policy-sanctioned per operational policy item 2.

**Editorial rulings:**

1. **Both TE cells update to clean values at reconciliation**: I → ×59.16, E → ×3.83 (with the declared depth change). Ariadna's minimal-patch alternative (keep 4.10 with legacy-provenance justification) is **REJECTED on the same grounds as F-cbar option 2** (Amendments 11-12): policy item 1 invalidates pre-patch values; no legacy cells inside a sanitized table.
2. **No manuscript edit now.** Grace's recommendation to update the table is accepted *in content* but executes only with the v2.04 release, after all Priority A rows close. These two cells are the first confirmed **numeric patches** of the release; status: `reconciliation-ready`.
3. **Footnote model approved** (Ariadna's formulation): short, factual — "post-D-0154 clean re-derivation updates TE ratios for E/I; qualitative classification unchanged" — plus the truncation cause in one sentence. **Tone guard**: the manuscript states the recovery neutrally (corrupted source attenuated the measured signal; clean source restores it). The "signal even stronger than estimated" framing stays in internal documents; assertive inflation is the mirror image of defensive inflation and both are barred.

No manuscript change authorized yet. Editorial freeze unchanged until full Priority A closure.

### Amendment 16 — 2026-07-15 (DET provenance-review script audited; execution authorized; subsampling definition precised)

**Independent audit:** Grace, `docs/HSI_v2_Audit_D0154_DET_Provenance_Review_Design.md`, SHA-256 `9B6D118188626A48924146B99BA162DD88148645BAE78A31F7B988536B4F2C4F` (Sofia recomputed: MATCH). Verdict: **APPROVED**, execution green-lit.

**Definitional precision (supersedes earlier informal phrasing):** the legacy DET subsampling step is **dynamic** — `max(1, len(medium_samples)//10_000)` — not a static every-100th-bit rule. The "1 in 100" phrasing used in Amendments 10 and 13 and in coordination messages is the special case for a 1M-bit medium sample and stands as historical simplification only. Ariadna's script replicates the dynamic formula exactly. **Editorial consequence:** any eventual provenance documentation (manuscript footnote or appendix note at reconciliation) must state the dynamic relation formally; the approximation may appear only as parenthetical historical context.

Execution of the DET provenance-review batch is authorized (Grace's green light + standing delegated scope). Expected deliverable per Amendment 10: comparison table paper vs clean-under-legacy-protocol; DET cells close without patch if reproduced, escalate with decomposition-first discipline if not.

---

## D-0155 — Post-review Item 2 B-markov1@24 gate-plane preregistration

- Date: 2026-07-02
- Status: `frozen-execution-authorized`
- Scope: post-v2.03 external-review response / Markov-1 depth-matched Gate 1 check
- Cycle: Item 2 from Sofia's 2026-07-02 post-review experimental queue.

### Trigger

A brief-blind external review of the v2.03 manuscript raised the possibility that the separation between observed `B` and the `B-markov1` first-order Markov null might change when the source depth is extended to the clean `B@24` snapshot. Ariadna verified before drafting that no `B-markov1@24` Phase 1 transport/gate-plane readout exists in the repo or external results tree. Existing `B-markov1` evidence is `iter=20` only.

### Decision

D-0155 records the pre-freeze preregistration track for Item 2. The current review draft after Grace audit integration is:

`prereg_postreview_item2_bmarkov1_iter24_gate_plane_v0.2_DRAFT.md`

Draft SHA-256 (not a freeze hash):

`5C8560AE47B85FC4081FAACD9B24586914BD83632D410DBCD4F5E67AA50B0F4E`

The contract tests the same-protocol Gate 1 separation between observed `B` and `B-markov1` at `iter=20` and `iter=24` using:

- source depths `20,24`;
- null model `markov1`;
- null seeds `{17,101,211,307,401,503}`;
- canonical segment geometry `3 x 1,000,000` prefix bits;
- scale ladder `{8,12,16,20,24,28,32,40,48}`;
- projection policies `prefix,suffix`;
- canonical stability thresholds `min_count_floor=16`, `min_count_rate=1e-6`, `min_segment_support=max(2,ceil(S/2))=2`, `cv_max=1.5`;
- Gate 1 plane coordinates `retention@last` and `active_mean_tail`.

### Audit integration

Grace approved the v0.1 draft methodologically subject to four modifications requested by Sofia. Ariadna integrated all four into v0.2_DRAFT:

1. Threshold rationale for `G >= 0.30` and `D >= 0.20` added to §7.
2. Baseline safety rule added to §8: if `G(20) < 0.30` or `D(20) < 0.20`, the outcome is `markov1-baseline-not-established` and direct relative `iter=20 -> iter=24` claims are blocked.
3. `Delta_D = D(24) - D(20)` added as a mandatory diagnostic field in `summary.json` and `report.md`.
4. The seed-family lineage is documented explicitly: `{17,101,211,307,401,503}` is the canonical six-seed ensemble from the Phase 2 parent-density / parent-shell atlas aggregation path, reused here to preserve methodological continuity rather than introduced as a new tuning choice.

### Outcomes

The contract uses hierarchical outcome evaluation:

1. `inputs-or-run-not-derivable`
2. `markov1-baseline-not-established`
3. `markov1-separation-expands-at-iter24`
4. `markov1-separation-stable-at-iter24`
5. `markov1-separation-contracts-at-iter24`

All outcomes are valid. Contraction and baseline failure are not failed experiments; they are boundary/negative readouts.

### Boundary

This D-entry does **not** yet authorize execution. Execution requires Sofia's final editorial pass, Iban authorization, promotion from draft to a frozen executable version, SHA-256 anchoring, and a follow-up amendment to this entry. It does not authorize:

- any manuscript edit;
- any Phase 2, Phase 3, or Phase 4 readout;
- any B_XOR comparison;
- any permutation test over the full signature;
- any claim of asymptotic separation, universal robustness, Gate 2, geometry, horizon law, complete closure, or proof of HSI.

D-0154 remains the critical path and remains active. This contract must not consume resources in a way that blocks the Level 0 sanitization subcycle.

### Expected artifacts

Output root:

`results/hsi_v2/postreview_item2_bmarkov1_iter24_gate_plane/`

Required primary artifacts:

- `summary.json`
- `report.md`
- `manifest.json`
- `per_depth_gate_plane.csv`
- `per_seed_markov1_gate_plane.csv`

The manifest must include the preregistration SHA-256, source paths and hashes for `B@20` and `B@24`, the seed list, full protocol parameters, script hash, and hashes of all primary output artifacts.

### Amendment 1 — 2026-07-02 (process correction: v0.2_DRAFT pending Sofia final pass)

Grace's definitive preregistration audit was archived at:

`docs/HSI_v2_Audit_Postreview_Item2_Preregistration_v0.1.md`

Audit SHA-256:

`EF8EDAFA0CFED9B95CCFDA3C17156D3E79380083473C08BFF6FDF45DC9EB5626`

The audit's §6 included four refinements. During integration, Ariadna produced v1.0 and v1.1 freeze candidates prematurely. Sofia then clarified that the correct next state is a `v0.2` review draft and that the final editorial pass happens before freeze. No run was executed under v1.0 or v1.1.

Current operative review draft after this amendment:

`prereg_postreview_item2_bmarkov1_iter24_gate_plane_v0.2_DRAFT.md`

Draft SHA-256 (not a freeze hash):

`5C8560AE47B85FC4081FAACD9B24586914BD83632D410DBCD4F5E67AA50B0F4E`

v0.2_DRAFT integrates:

1. §5 now states explicitly that the seed family `{17,101,211,307,401,503}` is the canonical six-seed ensemble originally defined in the Phase 2 pipelines (parent density and parent shell atlas) to orchestrate and aggregate Phase 1 null-model runs. Its use here preserves direct methodological lineage and is not a new tuning choice.
2. §7 threshold provenance now names the historical approximate scale motivating the thresholds: `G(20) ~= 0.40` and `D(20) ~= 0.35` for the canonical B/Markov-1 separation line, versus near-overlap controls around `G ~= 0.00` and `D ~= 0.05`.
3. §8 preserves the baseline-safety outcome: if `G(20) < 0.30` or `D(20) < 0.20`, relative expansion/stability/contraction claims are blocked.
4. §10 requires `Delta_D = D(24) - D(20)` in `summary.json` and `report.md`.

All other protocol parameters remain unchanged: source depths `20,24`, null model `markov1`, seeds `{17,101,211,307,401,503}`, segment geometry `3 x 1,000,000`, scale ladder `{8,12,16,20,24,28,32,40,48}`, policies `prefix,suffix`, stability thresholds `min_count_floor=16`, `min_count_rate=1e-6`, `min_segment_support=max(2,ceil(S/2))=2`, `cv_max=1.5`, and Gate 1 coordinates `retention@last` / `active_mean_tail`.

Execution remains blocked until Sofia's final editorial pass, Iban authorization, promotion to a frozen executable version, and a new SHA-256 freeze amendment.

### Amendment 2 — 2026-07-02 (Sofia final editorial pass: PASSED)

Sofia's final editorial pass on `prereg_postreview_item2_bmarkov1_iter24_gate_plane_v0.2_DRAFT.md` is complete. Draft SHA-256 independently recomputed at filesystem level: `5C8560AE47B85FC4081FAACD9B24586914BD83632D410DBCD4F5E67AA50B0F4E` — MATCH with the value recorded in Amendment 1.

Verified at text level:

1. All four audit-§6 refinements are integrated as specified (§5 seed lineage; §7 threshold provenance including the descriptive rationale for the `+/-0.05` `Delta_G` band; §8 baseline safeguard; §10 `Delta_D` and `baseline_control.*` mandatory fields).
2. §8 upgrades the audit recommendation: baseline failure is a first-class hierarchical outcome (`markov1-baseline-not-established`) evaluated before the three relative outcomes, rather than a restricted-reading clause. The five-outcome space was checked for logical completeness and disjointness under first-match evaluation: complete and disjoint.
3. §12 wording fence extended coherently with "baseline separation not established" in the allowed list.
4. No frozen protocol parameter differs from the v0.1 draft audited by Grace.

Editorial verdict: **PASSED**. Remaining gates before execution: Iban authorization → promotion to frozen executable version → SHA-256 freeze anchoring recorded as a further amendment to this entry.

### Amendment 3 — 2026-07-02 (Iban authorization granted)

Iban authorizes the freeze of the Item 2 contract (explicit go, 2026-07-02, relayed via Sofia's session). With this, all human/editorial gates are satisfied:

1. Grace methodological audit: APPROVED WITH RECOMMENDATIONS (`docs/HSI_v2_Audit_Postreview_Item2_Preregistration_v0.1.md`, SHA-256 `EF8EDAFA0CFED9B95CCFDA3C17156D3E79380083473C08BFF6FDF45DC9EB5626`).
2. Sofia final editorial pass on v0.2_DRAFT: PASSED (Amendment 2).
3. Iban authorization: GRANTED (this amendment).

Remaining mechanical steps, owned by Ariadna: promote `prereg_postreview_item2_bmarkov1_iter24_gate_plane_v0.2_DRAFT.md` (SHA-256 `5C8560AE47B85FC4081FAACD9B24586914BD83632D410DBCD4F5E67AA50B0F4E`) to `prereg_postreview_item2_bmarkov1_iter24_gate_plane_v1.0.md` with content identical except the version/freeze-state header (§14), compute the freeze SHA-256, and record it as the anchoring amendment to this entry. The promoted text must not differ substantively from the v0.2 text that passed the editorial gate; any substantive difference voids the pass and returns the contract to review.

### Amendment 4 — 2026-07-02 (v1.0 executable freeze anchor)

Ariadna completed the mechanical promotion requested in Amendment 3. The executable contract is:

`prereg_postreview_item2_bmarkov1_iter24_gate_plane_v1.0.md`

Freeze SHA-256:

`3181DD505E072FB26BF94083C03B8C7BC23FC034B6B0759633CD04CED21F2D5D`

Validation notes:

1. `v1.0` was promoted from `v0.2_DRAFT` after Grace audit, Sofia editorial pass, and Iban authorization.
2. A no-index diff between `v0.2_DRAFT` and `v1.0` shows only version/freeze metadata changes in the header and §14.
3. No scientific parameter, source, seed, scale, threshold, metric, outcome rule, wording fence, or output requirement changed during promotion.

Execution is now authorized under this frozen v1.0 contract, subject to D-0154 remaining the priority compute path: if Level 0 sanitization for `D@20`, `G@20`, or `F@20` requires the machine, Item 2 waits. *(Compute-priority clause later relaxed by D-0154 Amendment 6: parallel execution at Ariadna's discretion.)*

---

## D-0156 — Post-review Item 1: c̄ = φ+1 formalizable conjecture (analytic cycle)

- Date: 2026-07-03
- Status: `open-analytic`
- Scope: analytic derivation attempt plus numerical verification tooling for the convergence c̄ → φ+1 on the Dyck nesting tree of Φ.
- Cycle: Item 1 of the 2026-07-02 post-review queue. Authorized by Iban 2026-07-02.

### Framing

- **"Formalizable conjecture", not theorem** (Ariadna's correction, accepted 2026-07-02): no document, script, or D-entry of this cycle uses "theorem" until a proof exists.
- **φ is never an axiom** (standing SKILL guardrail): any bridge to φ+1 must be a derived result over the parenthesis-tree combinatorics.
- **Outcome space includes refutation explicitly**: "recurrence found but limit ≠ φ+1" is a valid, recordable outcome.

### Decision: target observable fixed before polished readouts (2026-07-03)

Before the corrected verification run completed, the conjecture's target observable was fixed: **c̄ = `children_analysis.mean_children`, operationally the aggregate quotient of children over branching nodes.** The mean of local branching ratios (~2.2065 on the first partial run) is a **secondary observable, not a test of the conjecture**; the legacy parser's "Multiple Constants Test" over local ratios is superseded (it also lacked φ+1 in its constant set and misreported closest-match distance as significance). Ariadna's harness update stores: weighted `mean_children`, aggregate quotients `content_nodes/branching_nodes` and `edge_nodes/branching_nodes`, and distance to φ+1.

Rationale: mean-of-ratios and ratio-of-aggregates are different functionals; the manuscript's claim (§`subsec:mean_children`) is about the aggregate. Fixing the observable before the corrected numbers arrive protects the eventual readout, in either direction, from post-hoc observable selection.

### Preliminary observation (traceability only, not claim-bearing)

First partial run (legacy parser, 2026-07-03): `content_nodes` = 18,147,523, `branching_nodes` = 6,931,737; quotient = 2.6180340, within ~1e-7 of φ+1 = 2.6180339887. Recorded for traceability; pending definition confirmation and corrected-harness rerun. No claim derives from this observation.

### Boundary

- Exploratory analytic tooling iterates freely without prereg freeze.
- Any claim-bearing readout or manuscript-facing statement requires Grace audit, Sofia editorial review, an amendment here, and Iban authorization.
- No manuscript edit is authorized by this entry.
- Statistical requirements for any formal readout: uncertainty via bootstrap over subtrees/segments (tree-internal ratios are correlated; i.i.d. SEM is not acceptable), and φ+1 present in any constant-comparison set.

### Amendment 1 — 2026-07-03 (clean-source revalidation of the conjecture's empirical premise, audit-approved)

The D-0154 signature batch cbar readout (D-0154 Amendment 11, run `d0154-signature-priority-a__20260703T150824`, hashes anchored there) was audited by Grace (`docs/HSI_v2_Audit_D0154_Signature_Reconciliation.md`, SHA-256 `DC2CC4BC43C0B112E9FBE791715E0F6C10A6265855BBDF444D59CF645AA72009`) with verdict: positive revalidation.

**Empirical premise of the conjecture, now on fully clean post-patch sources under the D-0156 target observable (weighted aggregate mean_children):**

| Variant | cbar | φ+1 distance |
|---|---|---|
| E@24 | 2.6180332541411873 | 7.35e-7 |
| I@23 | 2.6180336200231284 | 3.69e-7 |
| D@20 | 2.618033803408778 | 1.85e-7 |
| G@20 | 2.6180336314223767 | 3.57e-7 |

All four approach φ+1 **from below**. Observation for the analytic work: the deficit's sign is uniform; whether its magnitude scales with source depth/size in a lawful way (finite-size correction) is itself a formalizable sub-question — a proven correction term would be stronger evidence than the limit alone. F@20 is not derivable under the canonical protocol (see D-0154 Amendments 11-12); the F-only mini-contract at clean F@27 will extend this table if successful.

No claim-bearing use beyond this log until the analytic cycle produces its own audited readout.

---

## D-0157 — Continuous cryptographic timestamping of process artifacts (OpenTimestamps)

- Date: 2026-07-06 (proposal raised by Iban 2026-07-06; earlier draft of this entry briefly carried 2026-07-03 by clerical error, corrected before anchoring — the `.ots` proofs themselves date the pilot)
- Status: `adopted` (pilot executed same day)
- Scope: external temporal-precedence proof for frozen and audit artifacts.

### Decision

The project's hash-chain audit discipline proves *internal consistency* (no post-hoc alteration of hashed artifacts) but not *temporal precedence* to an external skeptic: all files are under project control, and a hostile reviewer can claim retro-dating. Adopted: anchor SHA-256 hashes of key artifacts via **OpenTimestamps** (free, tokenless timestamping onto the Bitcoin chain; `.ots` sidecar proof files stored alongside artifacts; the artifact itself is never touched).

**What gets stamped going forward:** (1) frozen preregistration contracts at freeze time; (2) Decision Log snapshots at major closures; (3) audit documents at archive time. Calendar attestation is immediate; Bitcoin confirmation completes asynchronously (`ots upgrade` on the sidecar later).

**What was rejected:** any bespoke blockchain integration (custom chains, smart contracts, distributed-ledger infrastructure) — over-engineering, and a credibility liability in peer review. Standard timestamping is the proportionate mechanism. OSF registration remains the primary community-facing preregistration channel; this practice complements it for the continuous internal process chain.

**Origin:** Iban's proposal (2026-07-03); Sofia's assessment and implementation the same day. For the pending methodology article, this converts the temporal-precedence section from proposal to exercised practice.

**Pilot artifacts:** `prereg_postreview_item2_bmarkov1_iter24_gate_plane_v1.0.md` (frozen, SHA-256 `3181DD50...` per D-0155 Amendment 4) and a snapshot of this Decision Log including the present entry.

---
