# HSI v2 Level 2 L2-01: Spectral-Percolation Boundary Probe v0.2

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Supersedes for Level 2 purposes, if accepted: `HSI_v2_Level2_L2_01_QuotientShape_RegimeSeparation_Preregistration_v0_1.md`.

Source correction: Grace audit `docs/HSI-audit-Level2-L2-01-Preregistration.md` accepted v0.1 only with severe warning: the v0.1 object was methodologically clean, but epistemologically closer to graph-feature data science than to Level 2 geometry. This v0.2 redirects L2-01 toward a genuinely algebraic/spectral boundary question.

Purpose: test whether the transition from predictive-channel interior to null-envelope boundary is accompanied by a topological/spectral transition in the quotient-routing graph.

This is not a geometry claim. It is a pre-geometric spectral/percolation probe.

## 1. Scientific Question

When moving from channel interior to null-envelope boundary, does the observed-`B` weighted quotient-routing graph undergo a measurable spectral/topological transition?

Candidate signatures:

- fragmentation or percolation shift in connected components;
- collapse of algebraic connectivity;
- change in cycle-rank / first Betti number of the quotient graph;
- change in largest-component dominance;
- divergence from hard-null spectral/percolation context.

If yes:

- Level 2 has a candidate algebraic object to formalize: the boundary is not only a retention/entropy frontier, but a graph-state transition.

If no:

- the Phase 4/5 boundary remains real, but this quotient graph object does not carry a Level 2 invariant for it.

Either result is valid.

## 2. Non-Goals

- Do not recover geometry.
- Do not claim topology discovered.
- Do not claim curvature, Gate 2 closure, or Level 2 confirmation.
- Do not fit a boundary law.
- Do not classify regimes with a multivariate feature vector.
- Do not rescue Phase 5.
- Do not use top-k component shares as primary evidence.
- Do not use `B_retention`, `hard_margin`, `entropy_gap`, or null-envelope scores as graph features.
- Do not modify the manuscript.

## 3. Why v0.1 Is Not Enough

The v0.1 proposal was clean but weakly geometric. It asked whether generic quotient-shape features separate regimes.

This v0.2 changes the question.

It does not ask:

> Can a feature vector classify interior/boundary/inactive regimes?

It asks:

> Does a concrete graph-theoretic object suffer a spectral/percolation transition at the boundary?

The primary object is therefore the quotient-routing graph itself, not a downstream graph-feature classifier.

## 4. Primary Graph Object

Primary object:

`observed-B weighted quotient child-routing graph`

Source path:

- existing N3-05b construction;
- edge-weight mode: `pooled-retained-mass`;
- parent scale: `40`;
- top patterns: `128`;
- graph built over local multi-band windows, not per-cell adapters.

Important constraint:

Per-cell adapters are forbidden for the primary L2-01 graph because they erase cross-band topology and continuity edges. The primary Stage 1 object must use windows with at least two adjacent bands whenever the observability budget permits.

Reason:

Level 2 asks about graph-state transitions. A one-cell graph can diagnose local concentration, as in Q2, but cannot diagnose percolation across a transition domain.

## 5. Regime Labels

Regime labels must be inherited from accepted Phase 4/5 readouts only.

Allowed labels:

- `interior`: bands classified as strong predictive-channel support under accepted strict-grid readouts;
- `boundary`: bands classified as null-envelope catch-up under accepted strict-grid readouts;
- `inactive`: bands classified as strict failure or scout-inactive under accepted B-mini readouts.

Label provenance may use:

- `B_retention`;
- `hard_margin`;
- hard-null envelope readouts;
- Q2 catch-up labels.

But these quantities may not be graph features or scoring variables.

## 6. Stage 0: Observability Budget

v0.2 authorizes no spectral statistic yet.

Stage 0, once accepted, may only:

1. inventory candidate multi-band windows for `interior`, `boundary`, and `inactive`;
2. report whether each window can be built from existing or derivable child-routing rows;
3. report whether each window preserves cross-band continuity edges in N3-05b;
4. report holdout dependence on Q2 cells and Phase 5 B-mini cells;
5. estimate compute cost;
6. propose a sealed Stage 1 input table and hashes;
7. recommend whether Stage 1 is executable or whether the budget is insufficient.

Stage 0 may not:

- compute spectral invariants;
- compute graph distances;
- compare regimes;
- create figures;
- tune window sizes after seeing invariants;
- change regime labels.

## 7. Window Policy

Preferred window size:

`3 adjacent 9M bands`

Minimum window size:

`2 adjacent 9M bands`

Single-cell or single-band windows are forbidden for primary Stage 1 unless a later decision explicitly downgrades the experiment to local diagnostics rather than Level 2.

Stage 0 must report candidate windows separately by source arc.

The preferred Stage 1 design, if feasible, should include at least:

- two `interior` windows;
- two `boundary` windows;
- two `inactive` windows.

If this cannot be met without reusing Q2/D-0112 cells as the dominant source of evidence, Stage 0 must return:

`insufficient-level2-window-budget`

## 8. Holdout Policy

Primary preference:

Use windows not used in Q2/D-0112-D0113.

Allowed fallback:

If boundary windows necessarily overlap Q2 because Q2 defines the accepted boundary, overlap must be declared as a dependency and Stage 1 must include at least one non-Q2 boundary-adjacent window if available.

Forbidden:

- presenting Q2-derived cells as independent Level 2 evidence;
- choosing windows after inspecting spectral invariants;
- replacing failed windows after seeing graph readouts.

## 9. Candidate Invariants For Future Stage 1

Stage 1 is not authorized by this v0.2.

If Stage 0 succeeds, a future v0.3 must freeze the exact invariant list before execution.

Candidate primary invariants:

- `beta0`: connected component count;
- `zero_eigenvalue_count`: Laplacian zero multiplicity, expected to match `beta0` for the graph support;
- `lambda2_norm_laplacian`: algebraic connectivity;
- `largest_component_fraction`: node fraction in the largest connected component;
- `beta1`: cycle rank, computed as `edge_count - node_count + beta0` on the undirected support graph;
- `edge_density`: `2E / (N(N-1))`, reported only when node count is comparable or normalized by window.

Candidate secondary weighted invariants:

- `spectral_radius_adjacency`;
- weighted `lambda2` if the existing graph spectrum path already exposes it without definition changes;
- total projected-flow weight and family-continuity weight as metadata, not regime scores.

Forbidden as primary Level 2 invariants:

- `top1_component_weight_share`;
- `top3_component_weight_share`;
- `top5_component_weight_share`;
- `component_weight_entropy_norm`;
- `effective_weight_components`.

Reason:

Those quantities remain valid diagnostics, but after Q2/D-0113 they are too close to concentration statistics. L2-01 must test graph topology/spectrum, not concentration advantage.

## 10. Future Stage 1 Hypotheses

Stage 1 is not authorized here, but the hypotheses should be shaped as follows:

H1 spectral/percolation transition:

- boundary windows show higher `beta0` / zero multiplicity and lower `lambda2_norm_laplacian` than interior windows;
- inactive windows may either fragment further or collapse to low-support graphs, and must be interpreted separately.

H0 no Level 2 transition under this object:

- the spectral/percolation invariants do not separate interior from boundary beyond the preregistered calibration.

H2 hard-null mimicry:

- hard nulls reproduce the same spectral/percolation transition, implying the object is not B-specific and should not be promoted as a Level 2 invariant.

Exact thresholds and calibration are not authorized by v0.2.

## 11. Future Calibration

A future v0.3 Stage 1 contract should prefer:

- invariant-by-invariant tests, not a multivariate classifier;
- within-arc permutation or sign tests when sample size permits;
- explicit low-power boundary if `N` is small;
- no scatterplot or heatmap before numeric readout is sealed.

Multivariate shape classifiers are explicitly out of scope for L2-01.

## 12. Wording Fence

Allowed:

- `spectral-percolation probe`;
- `algebraic connectivity`;
- `component transition`;
- `cycle-rank readout`;
- `pre-geometric graph-state transition`;
- `candidate Level 2 invariant`.

Forbidden:

- `geometry recovered`;
- `topology discovered`;
- `curvature measured`;
- `Gate 2 solved`;
- `Level 2 confirmed`;
- `boundary law`;
- `universal horizon`;
- `graph statistics prove geometry`;
- `retention-independent geometry established`.

## 13. Manuscript Boundary

No manuscript edit is authorized by this draft.

Even Stage 0 success does not authorize manuscript integration.

Any paper-facing use requires:

1. accepted Stage 0 result;
2. accepted v0.3 Stage 1 contract;
3. execution;
4. Grace audit;
5. Sofia/Iban editorial decision;
6. new Decision Log entry.

## 14. Proposed Stage 0 Command

Proposed executable after acceptance:

```powershell
python.exe .\hsi_v2_level2_l2_01_spectral_percolation_stage0.py `
  --output-dir results\hsi_v2\level2\l2_01_spectral_percolation\stage0_observability
```

This command is not authorized until v0.2 is accepted, SHA256-anchored, and recorded in the Decision Log.

