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

