# HSI v2 Bridge: Projective Density vs Predictive Channel Alignment v0.1

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Purpose: test Grace Question 1 in the narrowest falsifiable form: whether local strength of the Phase 1 projective hierarchy aligns spatially with Phase 4/5 predictive-channel activation.

This is not a Phase 5 rescue path. It is a bridge-to-Level-2 feasibility and alignment probe.

## 1. Scientific Question

Does the local projective-hierarchy density measured from Phase 1 artifacts correlate with the predictive-channel activation measured in the already frozen Phase 4/5 bands?

Interpretation boundaries:

- If aligned, the long-range predictive channel can be read as a horizontal manifestation of the same multiscale projective structure isolated in Phase 1.
- If not aligned, Phase 1 projective hierarchy and Phase 4/5 predictive channel must be treated as distinct emergent structures that any Level 2 geometry must explain separately.
- Either result is valid. Failure to align is not a failed experiment.

## 2. Non-Goals

- Do not re-open Candidate B full execution.
- Do not rescue ARC-A or ARC-C.
- Do not infer a boundary law, scaling law, universal horizon, or global phase law.
- Do not select new arcs after seeing alignment results.
- Do not tune a Phase 1 metric to match Phase 5 hard margins.
- Do not modify `Paper_Latex/main_v2.01_cat.tex`.

## 3. Frozen Predictive Bands

The predictive side is fixed before this contract by Phase 4 and B-mini:

### Phase 4 reference arc

Use the already reported Phase 4 finite-horizon / catch-up bands from the canonical P4-04/P4-05 artifacts:

| arc_id | bands |
|---|---|
| `P4` | `768M-777M`; `777M-786M`; `786M-795M`; `795M-804M`; `804M-813M`; `813M-822M`; `822M-831M` |

### Phase 5 B-mini arcs

Use all five bands from each D-0104/D-0105 frozen B-mini arc:

| arc_id | bands |
|---|---|
| `ARC-A` | `300M-309M`; `309M-318M`; `318M-327M`; `327M-336M`; `336M-345M` |
| `ARC-B` | `597M-606M`; `606M-615M`; `615M-624M`; `624M-633M`; `633M-642M` |
| `ARC-C` | `1110M-1119M`; `1119M-1128M`; `1128M-1137M`; `1137M-1146M`; `1146M-1155M` |

No band may be added, removed, or replaced after acceptance.

## 4. Predictive-Channel Score

The predictive score is derived only from existing strict P4-grid readouts.

Per band:

- `hard_margin_max`: maximum hard-null margin across the strict P4 lag grid and negative channel.
- `B_retention_max`: maximum observed B retention across the same cells.
- `strict_band_class`:
  - `strong` if any cell has `B_retention >= 0.90` and `hard_margin >= 0.30`;
  - `shadow` if no strong cell and any cell has `B_retention >= 0.50` and `hard_margin <= 0.05`;
  - `inactive` otherwise.

The broad scout readout from B-mini is not used in the predictive score.

## 5. Candidate Phase 1 Local Density Metrics

The core methodological risk is that Phase 1 was originally computed on fixed segment snapshots, not necessarily on every Phase 4/5 band. Therefore v0.1 does not yet authorize the alignment test. It authorizes only Stage 0 feasibility.

Stage 0 must determine whether a local Phase 1 score can be computed for every frozen band without violating existing contracts.

Candidate metrics, in priority order:

1. `transport_active_mean_tail`: tail transport active mean from the canonical Phase 1 transport path, computed locally on the same band.
2. `projective_retention_tail`: mean of prefix/suffix survival fractions over the top two Phase 1 edges available in the local tower.
3. `pattern_density_tail`: normalized kept-pattern density at the highest available scale, reported only as a secondary diagnostic.

Metric selection rule:

- Prefer `transport_active_mean_tail` if the canonical transport path can run locally for all frozen bands without changing definitions.
- If transport cannot be computed locally for all bands, use `projective_retention_tail`.
- If neither can be computed for all bands, Stage 0 returns `not-executable` and no alignment test is authorized.

No metric may be chosen by inspecting its correlation with the predictive score.

## 6. Stage 0 Feasibility Audit

Stage 0 is read-only except for writing its own report.

Stage 0 blindness requirement:

- Stage 0 may extract frozen predictive-score values only to document availability and source integrity.
- Stage 0 must not compute any correlation, regression, rank comparison, scatter plot, heatmap overlay, or qualitative alignment verdict between Phase 1 local metrics and predictive scores.
- Stage 0 must output raw Phase 1 local metric values separately from the frozen predictive-score table.
- The Stage 0 report may recommend a Phase 1 metric only by the mechanical rule in section 5, never by its apparent agreement with predictive-channel scores.

Inputs:

- existing Phase 4/P4-04/P4-05 artifacts;
- existing D-0105 B-mini artifacts;
- existing source streams and Phase 1 pipeline code;
- no regenerated Level 0 snapshots unless explicitly allowed by a later accepted contract.

Outputs:

- list of frozen bands;
- available predictive score per band;
- for each candidate Phase 1 metric:
  - whether it can be computed locally for every frozen band;
  - exact command needed;
  - expected cost estimate;
  - expected cost estimate per arc (`P4`, `ARC-A`, `ARC-B`, `ARC-C`);
  - whether any existing artifact already contains the metric;
- recommended Stage 1 metric by the rule in section 5;
- verdict:
  - `stage1-executable`;
  - `not-executable-with-current-artifacts`;
  - `requires-new-contract-for-local-phase1-runs`.

Stage 0 does not compute correlations.

## 7. Stage 1 Alignment Test

Stage 1 is not authorized by this v0.1 draft. If Stage 0 returns `stage1-executable`, a v0.2 contract must freeze:

- the final Phase 1 metric;
- exact local Phase 1 command(s);
- exact predictive-score source files;
- the alignment statistic;
- null/permutation procedure if used;
- kill criteria;
- output paths.

Recommended alignment statistic for v0.2:

- Spearman rank correlation between local Phase 1 score and `hard_margin_max` over frozen bands;
- ordinal contrast between Phase 1 score and `strict_band_class`;
- arc-level aggregation reported separately to avoid pretending that adjacent bands are independent samples.

Statistical-power caution for v0.2:

- band-level reporting would have `N=22` frozen bands, but adjacent bands are correlated by construction and must not be presented as independent samples;
- arc-level reporting would have only four units (`P4`, `ARC-A`, `ARC-B`, `ARC-C`) and therefore has structurally weak statistical power;
- v0.2 should consider an arc-level permutation calibration if the Stage 0 output supports a clean alignment test.

## 8. Kill / Stop Criteria

Stop and escalate if:

- computing the Phase 1 local metric requires changing `v2/phase1/tower.py` defaults;
- any frozen band lacks source coverage needed for the selected local Phase 1 metric;
- the selected metric differs between bands only because of unequal observable length;
- any script would need to regenerate or alter accepted Phase 4/5 artifacts;
- Stage 0 discovers that the available Phase 1 metric was already influenced by Phase 4/5 arc selection.

## 9. Wording Fence

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

## 10. Required Artifacts If Implemented

Stage 0 output root:

`hsi_agents_project/results/hsi_v2/bridge_projective_predictive_alignment/stage0_feasibility/`

Required Stage 0 artifacts:

- `summary.json`
- `report.md`
- `frozen_bands.csv`
- `metric_feasibility.csv`
- `manifest.json`

Any Stage 1 artifacts require a v0.2 contract and are intentionally not specified here.

## 11. Acceptance Before Execution

Before Stage 0 execution:

1. Sofia/Iban must accept or revise this v0.1 scope.
2. Grace must audit the anti-rescue and anti-post-hoc boundaries.
3. The final v0.1 must be SHA256-anchored.
4. A Decision Log entry must record acceptance.

Before Stage 1 execution:

1. Stage 0 must complete.
2. A v0.2 contract must be written from the Stage 0 result.
3. Sofia/Iban/Grace must accept v0.2.
4. A new Decision Log entry must authorize Stage 1.
