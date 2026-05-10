# HSI v2 Q2: Internal Routing Entropy at Null-Envelope Catch-Up v0.2

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Supersedes for Stage 1 purposes: `HSI_v2_Q2_InternalRoutingEntropy_Preregistration_v0_1.md`.

Purpose: authorize the first internal-routing entropy contrast at already frozen null-envelope catch-up cells, using the existing N3-05b/N3-05c component-quotient path without changing its metric definitions.

This is not a Phase 5 rescue path, not a predictive extension, and not a Level 2 geometry claim. It is a narrow thermodynamic diagnostic at surface-retention equality.

## 1. Stage 0 Closure

Accepted Stage 0 run:

`results/hsi_v2/q2_internal_routing_entropy/stage0_feasibility/q2-internal-routing-entropy-stage0__20260509T165423/`

Stage 0 verdict:

`stage1-contract-needed-with-transition-adapter`

Stage 0 found that all required routing rows are present, but the existing N3-05b/N3-05c path requires a transition-shaped adapter before any entropy contrast can be computed.

| role | selected cells | observed rows | phase-matched-LZ rows | block-entropy rows | status |
|---|---:|---:|---:|---:|---|
| `catchup` | 20 | 20 | 20 | 20 | `n3-route-inputs-present` |
| `strong-context` | 14 | 14 | 14 | 14 | `n3-route-inputs-present` |

Stage 0 did not compute `component_weight_entropy_norm`, `entropy_gap`, or any observed-vs-null entropy comparison.

## 2. Sealed Stage 0 Inputs

Stage 1 may read only the following sealed Stage 0 artifacts:

| input | SHA256 |
|---|---|
| `catchup_cells.csv` | `25174BAB9BE0469061D827C1F36FFBB7D83AEF7CB4B20E816D078458AFE994E4` |
| `context_cells.csv` | `85177E96E7674D7B4D3BE637EE85427FEBF90B9838D02BC68F6DCEB5A08D57FF` |
| `routing_coverage.csv` | `080AD771D826533763D37D42DE1E1A4DF1337BB583B7C824D890311AEA5D9605` |
| `summary.json` | `05A0A05A2E768AB61591938A0C7735E4FD8754253B4355F6E0C0B112ED71A8B0` |

If any hash fails, Stage 1 must stop.

No cell may be added, removed, or replaced after acceptance.

## 3. Frozen Cell Domain

Primary catch-up cells are exactly the 20 rows in `catchup_cells.csv`.

Secondary strong-context cells are exactly the 14 rows in `context_cells.csv`.

Only `channel = negative` cells are in scope.

The allowed lag values are inherited from the frozen strict P4-grid readouts:

`-29500000,-27000000,-26500000,-24000000,24500000,26500000`

The selected catch-up bands are:

| source | band | selected cells |
|---|---|---:|
| `P4` | `795M-804M` | 4 |
| `P4` | `804M-813M` | 3 |
| `P4` | `813M-822M` | 1 |
| `P4` | `822M-831M` | 4 |
| `ARC-B` | `624M-633M` | 4 |
| `ARC-B` | `633M-642M` | 4 |

The selected strong-context bands are:

| source | band | role |
|---|---|---|
| `P4 strong` | `768M-777M` | report-only calibration |
| `P4 strong` | `777M-786M` | report-only calibration |
| `P4 strong` | `786M-795M` | report-only calibration |
| `ARC-B strong` | `597M-606M` | report-only calibration |
| `ARC-B strong` | `606M-615M` | report-only calibration |
| `ARC-B strong` | `615M-624M` | report-only calibration |

Strong-context cells calibrate sign and scale. They cannot change the catch-up verdict.

## 4. Q2 Transition Adapter

Stage 1 must first build Q2 cell adapters that make each frozen Stage 0 cell readable by the existing N3-05b/N3-05c code path.

The adapter is a format bridge only. It must not change routing rows, graph definitions, edge weights, component definitions, entropy formulas, or verdict thresholds inside N3-05b/N3-05c.

The adapter unit is one selected cell, not the full Q2 domain.

Rationale: a single global adapter would produce one N3-05c entropy for the whole transition domain, which would erase the pre-registered cell-level and band-level sign readouts. One-cell adapters preserve the frozen N3 metric while making the Q2 contrast mechanically auditable.

### 4.1 Adapter output root

Adapter outputs must live under:

`results/hsi_v2/q2_internal_routing_entropy/stage1_entropy/<run-name>/adapter/`

For every selected cell, the adapter must emit:

- `cells/<cell_id>/transition_run/nodes.csv`
- `cells/<cell_id>/transition_run/edges.csv`
- `cells/<cell_id>/transition_run/manifest.json`
- `cells/<cell_id>/child_routing_run/summary.json`
- `cells/<cell_id>/child_routing_run/<band>/<lag>/routing_rows.csv`
- `cells/<cell_id>/child_routing_run/manifest.json`

`cell_id` must be deterministic:

`<role>__<source_slug>__<band>__lag-<lag_bits>`

where `role` is `catchup` or `strong-context`, `source_slug` is the sealed source label normalized to ASCII lowercase with non-alphanumerics replaced by `-`, and `lag_bits` is the signed integer lag.

### 4.2 `transition_run/nodes.csv`

Each cell adapter must emit exactly one transition node.

Required fields:

| field | value |
|---|---|
| `top_patterns` | `128` |
| `band` | copied from that sealed cell CSV row |
| `band_start_bits` | parsed from `band` lower endpoint |
| `band_stop_bits` | parsed from `band` upper endpoint |
| `channel` | `negative` |
| `node_class` | `q2-catchup` or `q2-strong-context` |
| `strong_count` | `1` |
| `best_margin_lag_bits` | copied from that sealed cell `lag_bits` |
| `best_margin` | copied from that sealed cell `hard_margin` |
| `cell_role` | `catchup` or `strong-context` |
| `source` | copied from sealed cell CSV |

Only the fields consumed by `load_transition_cells()` may affect N3-05b/N3-05c. Additional fields are traceability metadata only.

### 4.3 `transition_run/edges.csv`

Each cell adapter must emit an empty `edges.csv` with header only.

Rationale: N3-05b/N3-05c load transition cells from `nodes.csv`; the transition `edges.csv` is required only by wrapper-level run validation. It must not introduce extra structure.

### 4.4 `child_routing_run`

Each cell adapter must build a minimal child-routing run from existing `routing_rows.csv` inputs located by Stage 0.

For each sealed cell `(band, lag_bits)`:

1. Find the exact existing routing rows indexed by Stage 0 for that `(band, lag_bits)`.
2. Copy rows for observed `B`, `phase-matched-lz`, and `block-entropy`.
3. Do not include `markov1` in the hard-null contrast, but it may be copied to the adapter only if kept in a clearly marked `collapse-control` report section. It must not enter `hard_null_entropy_min`.
4. Preserve every original routing-row field verbatim.
5. Write the combined rows to:

`cells/<cell_id>/child_routing_run/<band>/<lag>/routing_rows.csv`

where `<lag>` is the integer `lag_bits` value as text.

`summary.json` in the minimal child-routing run is adapter metadata only. It must not redefine any source metric.

If any selected cell lacks observed `B`, `phase-matched-lz`, or `block-entropy` rows, Stage 1 must stop before N3-05b.

## 5. Frozen Entropy Path

Stage 1 must run the existing N3 path once per selected cell, in this order:

1. N3-05b weighted quotient child-routing spectral graph:
   - `hsi_v2_phase3_weighted_quotient_child_routing_spectral_graph.py`
   - weighted mode must remain the script's built-in `pooled-retained-mass` mode.
   - `--top-patterns 128`
   - `--parent-scale 40`
   - `--transition-run <adapter/cells/<cell_id>/transition_run>`
   - `--child-routing-run <adapter/cells/<cell_id>/child_routing_run>`
2. N3-05c component-wise quotient normalization:
   - `hsi_v2_phase3_component_quotient_child_routing_spectrum.py`
   - `--weighted-quotient-run <cell N3-05b-output>`

For every selected cell, Stage 1 must extract one `component_weight_entropy_norm` value for observed `B` and one distribution per hard-null family from that cell's N3-05c `source_summary.csv`.

Primary entropy object:

`component_weight_entropy_norm`

Secondary descriptors:

- `effective_weight_components`
- `top1_component_weight_share`
- `top3_component_weight_share`
- `top5_component_weight_share`

No alternate entropy metric may be introduced.

## 6. Hard-Null Envelope

For each selected cell and hard-null family:

`family_entropy_min = min(component_weight_entropy_norm over that family's seeds)`

The hard-null envelope for each cell is:

`hard_null_entropy_min = min(phase_matched_lz_family_entropy_min, block_entropy_family_entropy_min)`

`markov1` is excluded from this minimum.

The report must show:

- observed `B` entropy;
- `phase-matched-lz` entropy distribution across seeds separately;
- `block-entropy` entropy distribution across seeds separately;
- each family minimum;
- the resulting hard-null minimum;
- which null family and seed attained the minimum.

If the minimum is always attained by the same family, that dependence must be stated explicitly.

## 7. Primary Contrast

For every catch-up selected cell:

`entropy_gap = hard_null_entropy_min - observed_B_entropy`

Interpretation:

- `entropy_gap > 0`: observed `B` is internally lower-entropy than the best hard null.
- `entropy_gap = 0`: no internal entropy separation under this object.
- `entropy_gap < 0`: the hard null is internally lower-entropy than observed `B`.

The sign is symmetric by construction. Positive values are not privileged by formula.

## 8. Aggregation and Statistical Boundary

Primary decision layer:

Band-level sign readout over the six catch-up bands.

For each catch-up band:

- compute the median `entropy_gap` across its selected cells;
- compute `positive_cell_fraction = count(entropy_gap > 0) / selected_cells`;
- classify the band as:
  - `B-lower-entropy` if median `entropy_gap > 0`;
  - `no-separation` if median `entropy_gap = 0`;
  - `null-lower-entropy` if median `entropy_gap < 0`.

Primary contrast:

Two-sided sign-test style readout over the six band-level median signs.

The report must declare the discrete power boundary:

- maximum catch-up bands: `N = 6`;
- if all six signs are positive, two-sided binomial p-floor is approximately `0.03125`;
- this is diagnostic evidence only, not a final proof.

Cell-level summaries are descriptive only because cells within a band are not independent.

Strong-context bands must be reported in the same descriptive format, but cannot affect the primary catch-up verdict.

## 9. Outcome Categories

Stage 1 must assign exactly one category:

| category | rule |
|---|---|
| `internal-asymmetry-supported` | at least 5/6 catch-up bands have positive median `entropy_gap`, and no catch-up band has median `entropy_gap < 0` |
| `mixed-internal-asymmetry` | signs are mixed or only 3-4/6 catch-up bands are positive |
| `no-internal-separation` | 0-2/6 catch-up bands are positive and no negative dominance is present |
| `null-internally-lower-entropy` | at least 4/6 catch-up bands have negative median `entropy_gap` |
| `incomplete-inputs-stop` | any sealed catch-up cell lacks required observed or hard-null routing rows |

These labels are Q2 readout labels only. They do not authorize manuscript changes by themselves.

## 10. Required Outputs

Stage 1 must emit:

- `summary.json`
- `report.md`
- `cell_entropy.csv`
- `band_entropy_summary.csv`
- `family_entropy_summary.csv`
- `adapter_manifest.json`
- `manifest.json`

`cell_entropy.csv` must include one row per selected cell and source family before envelope aggregation.

`band_entropy_summary.csv` must include median `entropy_gap`, `positive_cell_fraction`, selected-cell count, and category per band.

`family_entropy_summary.csv` must report observed `B`, `phase-matched-lz`, and `block-entropy` distributions separately.

The report must include:

- a boundary statement that Stage 1 reuses N3-05b/N3-05c and introduces no new entropy metric;
- the adapter contract and the fact that `transition_run/edges.csv` is empty by design;
- zero-cell handling inherited from Stage 0: none occurred, but future reruns must flag them explicitly;
- hard-null family dependence;
- strong-context descriptive calibration;
- the sign-test p-floor boundary;
- the final Q2 category.

## 11. Kill Criteria

Stage 1 must stop before computing the entropy contrast if:

- any sealed Stage 0 hash fails;
- any selected catch-up cell is missing observed `B`, `phase-matched-lz`, or `block-entropy` routing rows;
- the adapter would require changing N3-05b/N3-05c code definitions;
- `component_weight_entropy_norm` is absent from the N3-05c output;
- the adapter cannot preserve original routing-row fields verbatim;
- any extra cell, band, lag, or null family is needed to make the contrast computable.

If a kill criterion triggers, no manuscript edit is authorized.

## 12. Wording Fence

Allowed wording:

- `internal routing entropy at catch-up`
- `surface retention equality`
- `internal thermodynamic asymmetry`
- `component-weight entropy`
- `hard-null entropy envelope`
- `Q2 diagnostic bridge`
- `null-envelope opacity`

Forbidden wording:

- `Gate 2 solved`
- `Level 2 proven`
- `geometry recovered`
- `topology discovered`
- `catch-up defeated`
- `null refuted`
- `predictive channel rescued`
- `final thermodynamic proof`
- `hidden geometry measured`

## 13. Manuscript Boundary

No manuscript edit is authorized by this contract.

If Stage 1 is accepted by Grace and Sofia, a separate D-XXXX decision is required before any paper-facing integration.

## 14. Execution Command

Proposed executable after acceptance:

```powershell
python.exe .\hsi_v2_q2_internal_routing_entropy_stage1.py `
  --stage0-run results\hsi_v2\q2_internal_routing_entropy\stage0_feasibility\q2-internal-routing-entropy-stage0__20260509T165423 `
  --output-dir results\hsi_v2\q2_internal_routing_entropy\stage1_entropy
```

This command is not authorized until the v0.2 contract is accepted, SHA256-anchored, and recorded in the Decision Log.
