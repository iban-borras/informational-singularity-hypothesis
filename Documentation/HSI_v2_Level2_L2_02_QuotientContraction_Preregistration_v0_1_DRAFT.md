# HSI v2 Level 2 L2-02: Quotient-Contraction Boundary Probe v0.1 DRAFT

Status: draft for Sofia/Iban/Grace review. Not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Purpose: test the inverse descriptive pattern discovered by L2-01 without reusing the failed fragmentation/percolation claim.

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

Does the boundary regime show quotient contraction and loss of sparse parallelism rather than fragmentation?

More precisely:

> When moving from interior to boundary inside the same arc, does the N3-05b weighted quotient graph lose cycle rank and graph extent while increasing relative density?

This is not a rescue of L2-01. It is a new hypothesis motivated by the failed direction of L2-01.

## 3. Non-Goals

- Do not claim geometry recovered.
- Do not claim topology discovered.
- Do not claim Level 2 confirmed.
- Do not claim `Topological Hairball Transition` as established.
- Do not reclassify L2-01 as positive.
- Do not edit the manuscript.
- Do not create a figure before numeric readout is sealed.

## 4. Primary Object

Same graph object as L2-01:

- N3-05b weighted quotient child-routing graph;
- edge-weight mode: `pooled-retained-mass`;
- parent scale: `40`;
- top patterns: `128`;
- normalized Laplacian convention inherited from `v2/phase3/defect_spectrum.py::compute_graph_spectrum`;
- zero tolerance `1e-9`.

Reason:

L2-02 tests whether the same object rejected for fragmentation nevertheless carries a different boundary signature. Changing the object would confound the interpretation.

## 5. Frozen Windows

Stage 1 must reuse the same frozen observed-B windows as L2-01:

### Primary Within-Arc Pairs

| Pair | Role | Arc | Bands |
|---|---|---|---|
| `T1-ARC-B` | interior | `ARC-B` | `606M-615M;615M-624M` |
| `T1-ARC-B` | boundary | `ARC-B` | `624M-633M;633M-642M` |
| `T1-P4` | interior | `P4` | `777M-786M;786M-795M` |
| `T1-P4` | boundary | `P4` | `795M-804M;804M-813M` |

### Secondary Cross-Arc Context

| Role | Arc | Bands |
|---|---|---|
| interior | `ARC-B` | `597M-606M;606M-615M;615M-624M` |
| interior | `P4` | `768M-777M;777M-786M;786M-795M` |
| boundary | `ARC-A` | `300M-309M;309M-318M;318M-327M` |
| boundary | `ARC-A` | `309M-318M;318M-327M;327M-336M` |
| inactive | `ARC-C` | `1110M-1119M;1119M-1128M;1128M-1137M` |
| inactive | `ARC-C` | `1119M-1128M;1128M-1137M;1137M-1146M` |

The within-arc pairs are primary because they control the regime/arc confound.

## 6. Primary Readout

Primary L2-02 indicators:

1. `node_count_contraction`: `boundary_node_count < interior_node_count`.
2. `beta1_cycle_loss`: `boundary_beta1 < interior_beta1`.
3. `total_edge_weight_loss`: `boundary_total_edge_weight < interior_total_edge_weight`.
4. `relative_density_rise`: `boundary_edge_density > interior_edge_density`.
5. `largest_component_nonincrease`: `boundary_largest_component_fraction <= interior_largest_component_fraction`.

L2-02 is intentionally not a lambda2 test. L2-01 showed `lambda2_norm_laplacian = 0` across the relevant observed-B windows; reusing lambda2 as a primary L2-02 indicator would be non-informative.

## 7. Node-Count Guard

Because edge density is strongly size-dependent, L2-02 must report node-count ratios explicitly.

Rules:

- If `boundary_node_count / interior_node_count < 0.25`, the pair is flagged `severe-size-contraction`.
- If `boundary_node_count / interior_node_count` is in `[0.25, 0.75)`, the pair is flagged `moderate-size-contraction`.
- If `boundary_node_count / interior_node_count >= 0.75`, the pair is flagged `weak-or-no-size-contraction`.

The density-rise indicator cannot be interpreted without this flag.

## 8. Pair-Level Classification

A within-arc pair is `quotient-contraction-positive` if at least 4 of the 5 primary indicators hold.

A within-arc pair is `quotient-contraction-negative` if 0 or 1 of the 5 primary indicators hold.

All other cases are `quotient-contraction-mixed`.

## 9. Overall Observed-B Readout

Primary observed-B result:

- `quotient-contraction-candidate`: both within-arc pairs are `quotient-contraction-positive`.
- `single-arc-contraction`: exactly one within-arc pair is `quotient-contraction-positive`.
- `mixed-or-low-power`: at least one pair is mixed and neither rule above applies.
- `no-quotient-contraction-support`: both pairs are negative.

Cross-arc medians may be reported as context but cannot override the within-arc result.

## 10. Hard-Null Control

Hard-null families:

- `phase-matched-lz`;
- `block-entropy`;
- `matched-lz`, only if complete rows are present.

For each complete family and seed, repeat the pair-level classification.

Hard-null categories:

- `null-contraction-mimicry`: any hard-null family has at least one seed with both within-arc pairs positive.
- `partial-null-contraction-mimicry`: any hard-null seed has exactly one within-arc pair positive.
- `no-null-contraction-mimicry`: no complete hard-null seed has a positive pair.
- `hard-null-inputs-not-derivable`: no hard-null family is complete.

If `null-contraction-mimicry` occurs, L2-02 cannot be read as B-specific.

## 11. Output Artifacts

Required outputs:

- `summary.json`;
- `report.md`;
- `pair_contraction.csv`;
- `cross_arc_context.csv`;
- `hard_null_contraction_mimicry.csv`;
- `manifest.json`.

No figure is authorized by v0.1.

The report must state explicitly:

- L2-02 is descriptive and low-N;
- L2-01 remains a hard negative for the fragmentation/percolation direction;
- any `hairball` wording is exploratory unless the L2-02 contract is executed and audited.

## 12. Stop Conditions

Stop before interpretation if:

- the L2-01 Stage 1 artifact hashes do not verify;
- any frozen observed-B within-arc window is missing;
- `beta1`, `edge_density`, `node_count`, or `total_edge_weight` cannot be recovered without changing the graph object;
- a window replacement is needed;
- the implementation would need a figure or visual clustering to interpret the result.

## 13. Wording Fence

Allowed:

- `quotient contraction`;
- `cycle-rank loss`;
- `loss of sparse parallelism`;
- `relative density rise`;
- `boundary contraction candidate`;
- `inverse descriptive pattern`.

Forbidden:

- `geometry recovered`;
- `topology discovered`;
- `Level 2 confirmed`;
- `Topological Hairball Transition established`;
- `Gate 2 solved`;
- `Phase 5 rescued`;
- `L2-01 was positive after all`.

## 14. Manuscript Boundary

No manuscript edit is authorized by this draft or by L2-02 execution.

A future paper-facing use requires:

1. accepted execution result;
2. post-execution Grace audit;
3. Sofia editorial review;
4. new Decision Log entry authorizing manuscript integration.
