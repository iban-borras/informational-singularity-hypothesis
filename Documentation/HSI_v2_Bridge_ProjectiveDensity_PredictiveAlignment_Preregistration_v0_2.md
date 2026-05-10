# HSI v2 Bridge: Projective Density vs Predictive Channel Alignment v0.2

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Supersedes for Stage 1 purposes: `HSI_v2_Bridge_ProjectiveDensity_PredictiveAlignment_Preregistration_v0_1.md`.

Purpose: authorize the first alignment readout between the sealed Stage 0 Phase 1 local metric and the sealed Phase 4/5 predictive-channel score, while preserving the blindness discipline established by D-0107.

This is not a Phase 5 rescue path. It is a bridge-to-Level-2 alignment readout.

## 1. Stage 0 Closure

Accepted Stage 0 run:

`results/hsi_v2/bridge_projective_predictive_alignment/stage0_feasibility/bridge-projective-predictive-stage0__bands-22__20260509T160812/`

Stage 0 verdict:

`stage1-executable`

Recommended metric:

`transport_active_mean_tail`

Stage 0 blindness remained intact:

- `alignment_statistics_computed = false`
- `scatter_or_heatmap_created = false`
- `qualitative_alignment_verdict = false`

## 2. Sealed Inputs

Stage 1 may read only the following sealed CSV inputs from the Stage 0 run:

| input | SHA256 |
|---|---|
| `phase1_local_metrics_raw.csv` | `7C4E27580D3C526C82AA9CB09BDCC915B71F41BBD7B19C2F2B689A00CB0EF43D` |
| `predictive_scores.csv` | `20534DB529DB130C9D7C3709EE4AC215945E4389F3DE8AA0C92DDAFF4D2155D7` |

The Stage 0 summary is retained for traceability:

| input | SHA256 |
|---|---|
| `summary.json` | `7A53ABD1B7F5EE9392338B4C760BAD814665F5F0329BF57AC615D8BC7E8D1E83` |

If any hash fails, Stage 1 must stop.

## 3. Frozen Bands

Stage 1 uses exactly the 22 frozen bands from D-0107:

| arc_id | bands |
|---|---|
| `P4` | `768M-777M`; `777M-786M`; `786M-795M`; `795M-804M`; `804M-813M`; `813M-822M`; `822M-831M` |
| `ARC-A` | `300M-309M`; `309M-318M`; `318M-327M`; `327M-336M`; `336M-345M` |
| `ARC-B` | `597M-606M`; `606M-615M`; `615M-624M`; `624M-633M`; `633M-642M` |
| `ARC-C` | `1110M-1119M`; `1119M-1128M`; `1128M-1137M`; `1137M-1146M`; `1146M-1155M` |

No band may be added, removed, or replaced.

## 4. Frozen Phase 1 Metric

Primary Phase 1 local metric:

`transport_active_mean_tail`

Justification:

- It was the highest-priority metric in v0.1.
- It is available for all 22 frozen bands.
- It is derived from the canonical Phase 1 transport path.
- It was selected by the mechanical Stage 0 rule before any alignment statistic was computed.

Secondary metrics:

- `projective_retention_tail`
- `pattern_density_tail`

Secondary metrics may be reported only as robustness/context rows. They may not change the primary verdict.

## 5. Frozen Predictive Score

Primary predictive score:

`hard_margin_max`

Secondary predictive descriptor:

`strict_band_class` with values:

- `strong`
- `shadow`
- `inactive`

Primary alignment is between:

`transport_active_mean_tail` vs `hard_margin_max`

No broad-scout readout is used.

## 6. Stage 1 Readouts

Stage 1 must report the following, in this order:

1. **Band-level Spearman rank correlation** between `transport_active_mean_tail` and `hard_margin_max` across the 22 frozen bands.
2. **Band-level Pearson correlation** between the same two columns, reported as secondary and descriptive only.
3. **Ordinal class summary**: mean and median `transport_active_mean_tail` by `strict_band_class`.
4. **Arc-level summary**: per-arc mean `transport_active_mean_tail`, max `hard_margin_max`, and count of `strong`, `shadow`, and `inactive` bands.
5. **Arc-level permutation calibration** over the four arc labels (`P4`, `ARC-A`, `ARC-B`, `ARC-C`), using the arc-level mean Phase 1 metric and arc-level max predictive score.

## 7. Statistical-Power Boundary

The interpretation must preserve two levels:

- Band-level has `N=22`, but adjacent bands are correlated by construction. It is a structured descriptive readout, not 22 independent samples.
- Arc-level has only four units. It is a calibration against obvious over-reading, not a high-power statistical test.

No p-value may be presented as decisive evidence.

## 8. Permutation Calibration

Arc-level permutation calibration:

- Units: the four arc labels `P4`, `ARC-A`, `ARC-B`, `ARC-C`.
- Phase 1 arc value: mean `transport_active_mean_tail` across bands in the arc.
- Predictive arc value: maximum `hard_margin_max` across bands in the arc.
- Statistic: Spearman correlation across the four arcs.
- Null: all `4! = 24` permutations of predictive arc labels against fixed Phase 1 arc values.
- Report:
  - observed arc-level Spearman;
  - number of permutations with absolute statistic >= observed absolute statistic;
  - exact permutation fraction.

This calibration is descriptive because `N=4`.

## 9. Outcome Categories

The report may classify the result into one of the following categories:

- `aligned-bridge-support`: band-level Spearman is positive and large (`rho >= 0.50`), and arc-level ordering is directionally compatible with the same reading.
- `weak-or-mixed-alignment`: band-level Spearman is positive but below `0.50`, or band-level and arc-level readings disagree.
- `parallel-emergence-outcome`: band-level Spearman is near zero or negative (`rho <= 0.10`), indicating no simple local alignment under this metric.
- `inconclusive`: missing data, hash failure, or any contract violation.

These categories are interpretive labels for this bridge readout only. They do not authorize manuscript changes by themselves.

## 10. Wording Fence

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

## 11. Required Artifacts

Output root:

`hsi_agents_project/results/hsi_v2/bridge_projective_predictive_alignment/stage1_alignment/`

Required artifacts:

- `summary.json`
- `report.md`
- `joined_band_table.csv`
- `arc_summary.csv`
- `permutation_table.csv`
- `manifest.json`

No figure is authorized by this v0.2 contract.

## 12. Stop Criteria

Stop and escalate if:

- any sealed input hash fails;
- any of the 22 frozen bands is missing from either input;
- duplicate band keys produce ambiguous joins;
- any Stage 1 script attempts to read broad-scout rows;
- any output computes an unlisted metric and uses it for the verdict;
- any correlation is computed before this v0.2 is accepted and logged.

## 13. Acceptance Before Execution

Before execution:

1. Sofia/Iban must accept or revise this v0.2 scope.
2. Grace must audit the Stage 1 alignment protocol.
3. The final v0.2 must be SHA256-anchored.
4. A Decision Log entry must record acceptance.

Only after these four steps may a Stage 1 script be executed.
