# HSI v2 Level 2 L2-01: Spectral-Percolation Stage 1 v0.3 DRAFT

Status: draft for Sofia/Iban/Grace review. Not executable until accepted, renamed without `DRAFT`, SHA256-anchored, and recorded in the Decision Log.

Purpose: freeze a Stage 1 spectral-percolation readout after the accepted Stage 0 observability inventory.

This contract does not authorize manuscript edits.

Revision note: this draft absorbs the non-blocking Sofia review comments after Grace accepted SHA256 `0980A7FE6006541ED664B23101C7C2CB7948EBCDC84E18836586DC0E3FDEB7D0`.

## 1. Source Chain

Accepted parent contract:

- `Documentation/HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_2.md`
- SHA256: `A548C01770DAD757151D62D8ED15575D85E811074922AFC57C062E21A8D003CC`

Accepted Stage 0 run:

- `results/hsi_v2/level2/l2_01_spectral_percolation/stage0_observability/l2-01-spectral-percolation-stage0__20260510T091237/`

Stage 0 artifact hashes:

| Artifact | SHA256 |
|---|---|
| `summary.json` | `BDEB00CE4962038E7131736D529CC623C2FE374C329A176B358339950EC6924C` |
| `band_inventory.csv` | `4ECD7EBB2C75570D04D2B57AA900E1799B233924C1015FA70D086C4600805E2F` |
| `window_inventory.csv` | `929C756A56693541B9B44F85CB2B7D7AB5AFB71319D52485ADE334DB5D24165A` |
| `stage1_candidate_windows.csv` | `42E2C13DC7EB4F2492C50834DF2D1FEA25CE1858B95942E2501A4EAB20B28638` |
| `manifest.json` | `7C7D8DC15F8710A77999847745F488F12845EECCDC2AFA95F935D26CCDA87D07` |

Stage 0 verdict:

- `stage1-executable`

## 2. Scientific Question

Does the observed-`B` weighted quotient child-routing graph show a spectral/percolation transition between predictive-channel interior, null-envelope boundary, and inactive regimes?

The primary threat detected after Stage 0 is regime/arc confounding:

- `interior` lives in `ARC-B` and `P4`;
- `boundary` holdout lives mostly in `ARC-A`;
- `inactive` lives in `ARC-C`.

Therefore Stage 1 must report three independent tiers:

1. within-arc transition control;
2. cross-arc regime readout;
3. hard-null mimicry control.

No single tier may be silently substituted for another.

## 3. Primary Graph Object

Primary object:

`observed-B weighted quotient child-routing graph`

Construction:

- existing N3-05b construction;
- edge-weight mode: `pooled-retained-mass`;
- parent scale: `40`;
- top patterns: `128`;
- graph built over multi-band windows;
- no per-cell adapter;
- no one-band primary graph;
- no fitted graph embedding;
- no learned clustering.

Spectrum convention:

- normalized Laplacian as implemented by `v2/phase3/defect_spectrum.py::compute_graph_spectrum`;
- edge weights included in the adjacency matrix;
- zero eigenvalue tolerance fixed at `1e-9`;
- no alternative solver semantics without a new Decision Log entry.

If the existing N3-05b implementation cannot construct these exact window-specific objects without changing graph semantics, execution must stop with a feasibility note. The script may not silently redefine the graph.

## 4. Frozen Stage 1 Windows

### Tier 1: Within-Arc Transition Control

Purpose: test the regime transition while holding the source arc fixed.

These windows are Q2-overlap by design. This is accepted here because the tier controls the regime/arc confound, not the holdout question.

| Pair | Role | Arc | Window ID | Bands |
|---|---|---|---|---|
| `T1-ARC-B` | interior | `ARC-B` | `ARC-B__interior__606M-615M_to_615M-624M__n2` | `606M-615M;615M-624M` |
| `T1-ARC-B` | boundary | `ARC-B` | `ARC-B__boundary__624M-633M_to_633M-642M__n2` | `624M-633M;633M-642M` |
| `T1-P4` | interior | `P4` | `P4__interior__777M-786M_to_786M-795M__n2` | `777M-786M;786M-795M` |
| `T1-P4` | boundary | `P4` | `P4__boundary__795M-804M_to_804M-813M__n2` | `795M-804M;804M-813M` |

### Tier 2: Cross-Arc Regime Readout

Purpose: test whether the same spectral/percolation pattern separates the Stage 0 candidate regimes in the broader cross-arc design.

| Role | Arc | Window ID | Bands | Holdout |
|---|---|---|---|---|
| interior | `ARC-B` | `ARC-B__interior__597M-606M_to_615M-624M__n3` | `597M-606M;606M-615M;615M-624M` | `q2-overlap` |
| interior | `P4` | `P4__interior__768M-777M_to_786M-795M__n3` | `768M-777M;777M-786M;786M-795M` | `q2-overlap` |
| boundary | `ARC-A` | `ARC-A__boundary__300M-309M_to_318M-327M__n3` | `300M-309M;309M-318M;318M-327M` | `non-q2` |
| boundary | `ARC-A` | `ARC-A__boundary__309M-318M_to_327M-336M__n3` | `309M-318M;318M-327M;327M-336M` | `non-q2` |
| inactive | `ARC-C` | `ARC-C__inactive__1110M-1119M_to_1128M-1137M__n3` | `1110M-1119M;1119M-1128M;1128M-1137M` | `non-q2` |
| inactive | `ARC-C` | `ARC-C__inactive__1119M-1128M_to_1137M-1146M__n3` | `1119M-1128M;1128M-1137M;1137M-1146M` | `non-q2` |

### Tier 3: Hard-Null Mimicry Control

Purpose: test whether hard nulls reproduce the same spectral/percolation pattern.

Required families, if complete routing rows are available:

- `phase-matched-lz`;
- `block-entropy`;
- `matched-lz`, only if present in the frozen source rows.

`markov1` is diagnostic collapse context only. It is not a hard-null mimicry family for this tier.

Coverage rule:

- a hard-null family is `complete` only if all frozen Tier 1 and Tier 2 windows can be built for that family;
- a hard-null family is `partial` if at least one frozen window is missing;
- Tier 3 must report coverage by family and by window before reporting mimicry.

If no hard-null family is complete, Stage 1 must return `hard-null-inputs-not-derivable` and stop before any positive Level 2 reading.

If at least one family is complete and at least one family is partial, Stage 1 may continue on the complete families, but the report must emit the `hard-null-inputs-partial` coverage flag.

## 5. Frozen Invariant List

Primary invariants:

1. `beta0`: connected component count of the undirected support graph.
2. `zero_eigenvalue_count`: multiplicity of zero in the normalized Laplacian spectrum.
3. `lambda2_norm_laplacian`: algebraic connectivity.
4. `largest_component_fraction`: node fraction in the largest connected component.
5. `edge_density`: undirected support density, `2E / (N(N-1))`, with `0` if `N < 2`.

Structural context invariants:

6. `beta1`: cycle rank, `edge_count - node_count + beta0`.
7. `spectral_radius_adjacency`: adjacency spectral radius.
8. `node_count`.
9. `edge_count`.
10. `total_edge_weight`.

Only the five primary invariants may drive the tier verdicts.

The structural context invariants must be reported, but they may not rescue a negative primary readout.

Sanity checks:

- `beta0` and `zero_eigenvalue_count` are theoretically redundant for the support graph. Any mismatch must be flagged as a numerical anomaly, not as transition evidence.
- `node_count` must be reported for every window. If the max/min node-count ratio exceeds `2.0` inside any Tier 1 pair or Tier 2 regime contrast, the report must flag `node_count-imbalance`; this flag contextualizes `edge_density` but does not change the pre-registered verdict rules.

Forbidden as primary Level 2 invariants:

- `top1_component_weight_share`;
- `top3_component_weight_share`;
- `top5_component_weight_share`;
- `component_weight_entropy_norm`;
- `effective_weight_components`;
- `B_retention`;
- `hard_margin`;
- `entropy_gap`;
- null-envelope surface scores.

## 6. Directional Tests

For an interior-to-boundary comparison, the spectral-percolation direction is:

| Invariant | Boundary support condition |
|---|---|
| `beta0` | `boundary > interior` |
| `zero_eigenvalue_count` | `boundary > interior` |
| `lambda2_norm_laplacian` | `boundary < interior` |
| `largest_component_fraction` | `boundary < interior` |
| `edge_density` | `boundary < interior` |

A window pair is `transition-positive` if at least 4 of the 5 primary conditions hold.

A window pair is `transition-negative` if 0 or 1 of the 5 primary conditions hold.

All other cases are `transition-mixed`.

The fixed spectral tolerance is `1e-9`, inherited from the existing spectrum path. No adaptive tolerance is allowed.

## 7. Tier Verdicts

### Tier 1 Verdict

Inputs:

- `T1-ARC-B`;
- `T1-P4`.

Rules:

- `within-arc-transition-support`: both pairs are `transition-positive`.
- `within-arc-mixed`: exactly one pair is `transition-positive`, or at least one pair is `transition-mixed`.
- `within-arc-no-support`: both pairs are `transition-negative`.

Tier 1 is the primary confound-control tier.

### Tier 2 Verdict

Inputs:

- the six cross-arc windows in Section 4.

Rules:

1. Compute the median of each primary invariant for `interior`, `boundary`, and `inactive`.
2. Apply the five directional tests to `median(boundary)` vs `median(interior)`.
3. Report `inactive` separately as calibration context.

Categories:

- `cross-arc-transition-support`: at least 4 of 5 median directional tests hold.
- `cross-arc-no-support`: 0 or 1 of 5 median directional tests hold.
- `cross-arc-mixed`: any other result.

Tier 2 cannot overrule Tier 1. It only reports generalization under the Stage 0 cross-arc design.

### Tier 3 Verdict

For each hard-null family, repeat Tier 1 and Tier 2 with that family replacing observed `B`.

Categories:

- `hard-null-mimicry`: at least one hard-null family reaches both `within-arc-transition-support` and `cross-arc-transition-support`.
- `partial-hard-null-mimicry`: at least one hard-null family reaches either Tier 1 or Tier 2 support, but not both.
- `no-hard-null-mimicry`: no hard-null family reaches support in either Tier 1 or Tier 2.
- `hard-null-inputs-not-derivable`: no required hard-null family is complete.
- `hard-null-inputs-partial`: at least one hard-null family is complete and at least one hard-null family is partial; this is a coverage flag reported alongside the mimicry category, not a positive or negative mimicry result.

If Tier 3 returns `hard-null-mimicry`, no positive Level 2 interpretation is allowed even if observed `B` is positive.

## 8. Overall Readout Categories

The report must emit exactly one overall category:

- `spectral-percolation-candidate`: Tier 1 support, Tier 2 support, and no hard-null mimicry.
- `within-arc-only-support`: Tier 1 support, Tier 2 not support, and no hard-null mimicry.
- `cross-arc-only-support`: Tier 2 support, Tier 1 is specifically `within-arc-no-support`, and no hard-null mimicry. This is explicitly confounded and cannot be promoted.
- `hard-null-mimicry`: Tier 3 mimicry.
- `mixed-or-low-power`: no mimicry, and at least one observed tier is mixed. If Tier 1 is `within-arc-mixed`, this category applies even when Tier 2 supports the transition.
- `no-level2-support`: observed tiers do not support the transition.
- `inputs-not-derivable`: graph construction or required routing rows fail.
- `hard-null-inputs-not-derivable`: observed-B graphs are buildable, but no required hard-null family is complete.

Only `spectral-percolation-candidate` would justify a later editorial discussion. It still would not authorize manuscript changes by itself.

The Stage 1 report must state explicitly that these are descriptive directional readouts under low-N pre-registered constraints, not inferential tests.

## 9. Output Artifacts

Required outputs:

- `summary.json`;
- `report.md`;
- `window_graph_summary.csv`;
- `tier1_within_arc_contrasts.csv`;
- `tier2_cross_arc_medians.csv`;
- `tier3_hard_null_mimicry.csv`;
- `graph_nodes.csv`;
- `graph_edges.csv`;
- `manifest.json`.

Required report notes:

- low-N descriptive status, not p-value inference;
- `beta0` / `zero_eigenvalue_count` sanity check;
- node-count imbalance flags using the `2.0` max/min threshold;
- hard-null coverage by family and window;
- any `hard-null-inputs-partial` coverage flag.

The manifest must include:

- v0.3 contract path and SHA256;
- Stage 0 run path and artifact hashes;
- exact command;
- source routing-row paths;
- graph construction parameters;
- numerical tolerance, if any;
- elapsed time.

No figure is authorized by this contract.

## 10. Stop Conditions

Stop and emit `inputs-not-derivable` if:

- any frozen observed-B window cannot be built exactly;
- any required graph object needs a semantic change from N3-05b;
- the script needs per-cell or one-band fallback;
- the script needs a non-frozen replacement window;
- the output would omit a primary invariant.

Stop and emit `hard-null-inputs-not-derivable` if:

- observed-B graphs are buildable, but no required hard-null family is complete for the frozen windows.

Do not replace windows after seeing any invariant.

## 11. Wording Fence

Allowed:

- `spectral-percolation probe`;
- `algebraic connectivity`;
- `component transition`;
- `cycle-rank readout`;
- `pre-geometric graph-state transition`;
- `candidate Level 2 invariant`;
- `within-arc confound control`;
- `hard-null mimicry control`.

Forbidden:

- `geometry recovered`;
- `topology discovered`;
- `curvature measured`;
- `Gate 2 solved`;
- `Level 2 confirmed`;
- `boundary law`;
- `universal horizon`;
- `graph statistics prove geometry`;
- `retention-independent geometry established`;
- `Phase 5 rescued`;
- `Q2 confirmed by L2`.

## 12. Manuscript Boundary

No manuscript edit is authorized by this draft or by execution of Stage 1.

A future paper-facing use would require:

1. post-execution Grace audit;
2. Sofia editorial review;
3. a new Decision Log entry authorizing any manuscript change.

If the result is `spectral-percolation-candidate`, the correct reading is:

> a pre-geometric graph-state transition candidate has been observed under a frozen spectral-percolation contract.

It is not:

> any of the forbidden recovery claims listed in Section 11.

## 13. Execution Status

This v0.3 is not executable while it has `DRAFT` in the filename.

Execution requires:

1. Grace signature;
2. Sofia/Iban signature;
3. rename to `HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_3.md`;
4. SHA256 anchor;
5. Decision Log entry accepting the contract and authorizing Stage 1.
