# HSI v2 Level 2 L2-02: Quotient-Contraction Mimicry Probe v0.2 DRAFT

Status: draft for Sofia/Iban/Grace review. Not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Purpose: test whether hard nulls reproduce the quotient-contraction pattern descriptively observed in L2-01.

This document does not authorize execution or manuscript edits.

## 1. Source Chain

Parent closure:

- D-0117: L2-01 accepted hard negative.
- Stage 1 run: `results/hsi_v2/level2/l2_01_spectral_percolation/stage1/l2-01-spectral-percolation-stage1__20260510T101433/`

Key Stage 1 hashes:

| Artifact | SHA256 |
|---|---|
| `summary.json` | `97220A946D55F9979827B2F68DAEE441A47F47816B393681C2E2AA23FFB89622` |
| `window_graph_summary.csv` | `9DC815C3A80BEF180043C8726CEC402C514BF9CC27902F34A21547913BC2EFE2` |
| `tier1_within_arc_contrasts.csv` | `AB522DD692BB90660491C3BBF823727FC46A9B723D8526FF63A123BF867F7A51` |
| `tier2_cross_arc_medians.csv` | `327DC68F7133CDDE8F66BC9F140EAED90E6515686472410012E493972C1F96E9` |
| `manifest.json` | `4EB0F82E26B46A0C4CC64AAF02EBCAAD87E941C4D2EE291BC5891A091F06CF74` |

## 2. Scientific Question

L2-01 leaves a descriptive inverse pattern on the observed-B windows: boundary windows contract in graph extent and cycle rank while relative density rises.

This observed-B pattern is expected to repeat under any re-description of the same already-inspected data. Therefore it is not independent evidence.

The genuinely new question in L2-02 is:

> Do hard nulls reproduce the same quotient-contraction pattern under the same N3-05b graph object and the same frozen windows?

If hard nulls reproduce it:

- the contraction pattern is not B-specific under this object;
- it becomes another delimiting result for the predictive subarc.

If hard nulls do not reproduce it:

- the observed-B contraction remains a B-specific candidate pattern, but still requires caution because the observed-B direction was defined after L2-01 results were known.

## 3. Data-Reuse Disclosure

The within-arc windows are reused from L2-01 because they are exactly where the inverse pattern was noticed.

This is post-hoc with respect to observed `B`.

No virgin holdout windows are used for the observed-B component of L2-02. Therefore the observed-B readout is an exploratory formalization of already-inspected windows, not an independent test.

Consequences:

- observed-B pair classification is descriptive confirmation of already-seen L2-01 numerics;
- observed-B pair classification is not independent evidence;
- the hard-null mimicry verdict is the deciding component of L2-02;
- no out-of-sample or predictive validity claim is allowed.

## 4. Non-Goals

- Do not claim geometry recovered.
- Do not claim topology discovered.
- Do not claim Level 2 confirmed.
- Do not claim `Topological Hairball Transition` as established.
- Do not claim L2-02 independently validates the L2-01 inverse pattern.
- Do not reclassify L2-01 as positive.
- Do not edit the manuscript.
- Do not create a figure before numeric readout is sealed.

## 5. Primary Object

Same graph object as L2-01:

- N3-05b weighted quotient child-routing graph;
- edge-weight mode: `pooled-retained-mass`;
- parent scale: `40`;
- top patterns: `128`;
- normalized Laplacian convention inherited from `v2/phase3/defect_spectrum.py::compute_graph_spectrum`;
- zero tolerance `1e-9`.

Reason:

L2-02 tests whether the same object rejected for fragmentation nevertheless carries a null-sensitive contraction pattern. Changing the object would confound the mimicry question.

## 6. Frozen Windows

Stage 1 must reuse the same frozen within-arc windows as L2-01:

| Pair | Role | Arc | Bands |
|---|---|---|---|
| `T1-ARC-B` | interior | `ARC-B` | `606M-615M;615M-624M` |
| `T1-ARC-B` | boundary | `ARC-B` | `624M-633M;633M-642M` |
| `T1-P4` | interior | `P4` | `777M-786M;786M-795M` |
| `T1-P4` | boundary | `P4` | `795M-804M;804M-813M` |

Cross-arc rows may be reported only as descriptive context:

| Role | Arc | Bands |
|---|---|---|
| interior | `ARC-B` | `597M-606M;606M-615M;615M-624M` |
| interior | `P4` | `768M-777M;777M-786M;786M-795M` |
| boundary | `ARC-A` | `300M-309M;309M-318M;318M-327M` |
| boundary | `ARC-A` | `309M-318M;318M-327M;327M-336M` |
| inactive | `ARC-C` | `1110M-1119M;1119M-1128M;1128M-1137M` |
| inactive | `ARC-C` | `1119M-1128M;1128M-1137M;1137M-1146M` |

Cross-arc context cannot change the overall L2-02 verdict.

## 7. Observed-B Descriptive Confirmation

Primary indicators for observed-B re-description:

1. `node_count_contraction`: `boundary_node_count < interior_node_count`.
2. `beta1_cycle_loss`: `boundary_beta1 < interior_beta1`.
3. `total_edge_weight_loss`: `boundary_total_edge_weight < interior_total_edge_weight`.
4. `relative_density_rise`: `boundary_edge_density > interior_edge_density`.
5. `largest_component_nonincrease`: `boundary_largest_component_fraction <= interior_largest_component_fraction`.

Observed-B pair classification:

- `B-contraction-descriptively-present`: both within-arc pairs pass at least 4/5 indicators.
- `B-contraction-descriptively-partial`: exactly one within-arc pair passes at least 4/5 indicators.
- `B-contraction-not-recovered`: no within-arc pair passes at least 4/5 indicators.

This classification is a consistency check on the L2-01 numerics, not the primary test.

## 8. Node-Count Guard

Because edge density is size-dependent, L2-02 must report node-count ratios explicitly.

Rules:

- If `boundary_node_count / interior_node_count < 0.25`, the pair is flagged `severe-size-contraction`.
- If `boundary_node_count / interior_node_count` is in `[0.25, 0.75)`, the pair is flagged `moderate-size-contraction`.
- If `boundary_node_count / interior_node_count >= 0.75`, the pair is flagged `weak-or-no-size-contraction`.

The density-rise indicator cannot be interpreted without this flag.

## 9. Hard-Null Mimicry Primary Test

Hard-null families:

- `phase-matched-lz`;
- `block-entropy`;
- `matched-lz`, only if complete rows are present.

For each complete family and seed, repeat the observed-B pair classification.

Hard-null pair classification uses the same 5 indicators and the same 4/5 threshold.

Family/seed categories:

- `null-contraction-mimicry`: both within-arc pairs are contraction-positive.
- `partial-null-contraction-mimicry`: exactly one within-arc pair is contraction-positive.
- `no-null-contraction-mimicry`: no within-arc pair is contraction-positive.

Coverage categories:

- `complete`: all four within-arc windows are available for that null seed;
- `partial`: at least one within-arc window is missing;
- `absent`: no rows available.

If no hard-null seed is complete across all families, the overall result is `hard-null-inputs-not-derivable`.

## 10. Overall Readout

The report must emit exactly one overall category:

- `B-specific contraction candidate`: observed-B is `B-contraction-descriptively-present` and no complete hard-null seed returns `null-contraction-mimicry` or `partial-null-contraction-mimicry`.
- `non-B-specific contraction`: observed-B is `B-contraction-descriptively-present` and at least one complete hard-null seed returns `null-contraction-mimicry` or `partial-null-contraction-mimicry`.
- `no contraction`: observed-B is not `B-contraction-descriptively-present`.
- `hard-null-inputs-not-derivable`: no hard-null seed is complete.

Evaluation priority:

1. If no hard-null seed is complete, emit `hard-null-inputs-not-derivable`.
2. Else, if observed-B is not `B-contraction-descriptively-present`, emit `no contraction`.
3. Else, use the complete hard-null seeds to distinguish `B-specific contraction candidate` from `non-B-specific contraction`.

The mimicry verdict is the deciding component whenever observed-B contraction is present.

## 11. Output Artifacts

Required outputs:

- `summary.json`;
- `report.md`;
- `observed_b_contraction.csv`;
- `hard_null_contraction_mimicry.csv`;
- `cross_arc_context.csv`;
- `manifest.json`.

No figure is authorized by v0.2.

The report must state explicitly:

- L2-02 is a post-hoc formalization for observed-B;
- no virgin holdout windows are used for the observed-B component;
- the primary new test is hard-null mimicry;
- L2-01 remains a hard negative for the fragmentation/percolation direction;
- any `hairball` wording remains exploratory unless accepted after execution and audit.

## 12. Stop Conditions

Stop before interpretation if:

- the L2-01 Stage 1 artifact hashes do not verify;
- any frozen observed-B within-arc window is missing;
- `beta1`, `edge_density`, `node_count`, `largest_component_fraction`, or `total_edge_weight` cannot be recovered without changing the graph object;
- a window replacement is needed;
- the implementation would need a figure or visual clustering to interpret the result.

## 13. Wording Fence

Allowed:

- `quotient contraction`;
- `cycle-rank loss`;
- `loss of sparse parallelism`;
- `relative density rise`;
- `boundary contraction candidate`;
- `mimicry-as-primary`;
- `post-hoc observed-B re-description`.

Forbidden:

- `geometry recovered`;
- `topology discovered`;
- `Level 2 confirmed`;
- `Topological Hairball Transition established`;
- `Gate 2 solved`;
- `Phase 5 rescued`;
- `L2-01 was positive after all`;
- `L2-02 confirms B-specific contraction`, unless the overall verdict is exactly `B-specific contraction candidate`;
- `L2-02 independently validates L2-01 inverse pattern`.

## 14. Manuscript Boundary

No manuscript edit is authorized by this draft or by L2-02 execution.

Even `B-specific contraction candidate` requires a separate D entry before any manuscript edit. The data-reuse origin makes editorial caution mandatory even on a positive outcome.

A future paper-facing use requires:

1. accepted execution result;
2. post-execution Grace audit;
3. Sofia editorial review;
4. new Decision Log entry authorizing manuscript integration.
