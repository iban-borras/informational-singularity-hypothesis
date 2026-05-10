# HSI v2 Q2: Internal Routing Entropy at Null-Envelope Catch-Up v0.1

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Purpose: test Grace Question 2 in the narrowest falsifiable form: when observed `B` and hard nulls are surface-indistinguishable by retention at null-envelope catch-up, do they still differ internally in routing entropy?

This is not a rescue of Phase 5 and not a geometry claim. It is a final thermodynamic diagnostic bridge before Level 2.

## 1. Scientific Question

At already discovered `null-envelope catch-up` bands/cells, does observed `B` achieve surface retention through lower internal routing entropy than the hard-null envelope?

If yes:

- surface retention equality does not imply internal physical equivalence;
- the null may be matching the surface by high-entropy memorization while observed `B` remains more canalized.

If no:

- the catch-up boundary is a genuine internal opacity boundary under this entropy object;
- the Level 2 theory must not assume hidden low-entropy separation at catch-up.

Either result is valid.

## 2. Non-Goals

- Do not reopen Candidate B.
- Do not explain ARC-A/ARC-C failure.
- Do not search for new catch-up bands.
- Do not introduce a new graph metric if the existing N3-07 path is executable.
- Do not claim geometry, topology, curvature, or Level 2 closure.
- Do not modify the manuscript.

## 3. Frozen Catch-Up Domain

Q2 uses only bands already frozen by Phase 4 and Phase 5 B-mini.

Primary catch-up bands:

| source | bands |
|---|---|
| `P4` | `795M-804M`; `804M-813M`; `813M-822M`; `822M-831M` |
| `ARC-B` | `624M-633M`; `633M-642M` |

Secondary context bands, report-only:

| source | bands |
|---|---|
| `P4 strong` | `768M-777M`; `777M-786M`; `786M-795M` |
| `ARC-B strong` | `597M-606M`; `606M-615M`; `615M-624M` |

No other band may be added after acceptance.

## 4. Catch-Up Cell Selection

For each frozen catch-up band, cells are selected mechanically from the existing strict P4-grid readout:

- channel: negative only;
- lag grid: `-29500000,-27000000,-26500000,-24000000,24500000,26500000`;
- include a cell if `B_retention >= 0.50` and `hard_margin <= 0.05`;
- if multiple cells qualify in the same band, include all qualifying cells and report band-level aggregates separately.

This selection is based on already accepted surface readouts. It does not inspect routing entropy.

## 5. Internal Routing Entropy Object

Preferred entropy object:

`component_weight_entropy_norm`

Source:

- existing N3-05c / N3-07 route:
  - weighted quotient child-routing graph;
  - component-wise quotient normalization;
  - direct flow-entropy readout.

Interpretation:

- lower `component_weight_entropy_norm` means routing mass is more concentrated into fewer normalized quotient components;
- lower `effective_weight_components` supports the same reading;
- higher `top3_component_weight_share` and `top5_component_weight_share` support stronger concentration.

Primary contrast:

`entropy_gap = hard_null_entropy_min - observed_B_entropy`

Positive `entropy_gap` means observed `B` is internally lower-entropy than the best hard null in that catch-up cell.

## 6. Hard-Null Envelope

Preferred hard-null envelope:

- `phase-matched-lz`
- `block-entropy`

Seeds:

`607,709,811,907,1009,1103`

`markov1` remains a collapse control and is not part of the hard-null entropy minimum.

If both hard-null families cannot be represented in the existing N3-07 path for the frozen cells, Stage 0 must report the limitation and stop before Stage 1.

## 7. Stage 0 Feasibility

This v0.1 authorizes only Stage 0 feasibility, not the entropy contrast itself.

Stage 0 must determine:

- whether the N3-05b/N3-05c/N3-07 pipeline can be applied to the frozen catch-up cells without changing definitions;
- whether all required observed `B`, `phase-matched-lz`, and `block-entropy` source rows exist or can be generated under existing accepted scripts;
- exact commands required;
- expected cost per source band/cell;
- whether the entropy object would be cell-level, band-level, or arc-level;
- whether existing Phase 3 artifacts already contain the required readout for any frozen cell.

Stage 0 must not compute the entropy contrast.

## 8. Stage 1 Candidate Readout

Stage 1 is not authorized by this v0.1 draft.

If Stage 0 returns executable, v0.2 must freeze:

- exact catch-up cells;
- exact source artifacts;
- exact entropy fields;
- aggregation rule from cell to band and arc;
- hard-null envelope rule;
- stop criteria;
- output paths.

Recommended Stage 1 readouts:

1. per-cell `entropy_gap`;
2. per-band median `entropy_gap`;
3. count of catch-up bands where observed `B` is lower-entropy than every hard-null family;
4. secondary gaps for `effective_weight_components`, `top3_component_weight_share`, and `top5_component_weight_share`.

## 9. Stop Criteria

Stop and escalate if:

- applying the entropy object requires changing N3-05b/N3-05c/N3-07 definitions;
- catch-up cells cannot be derived mechanically from frozen strict P4-grid data;
- any hard-null family is missing in a way that would make the envelope asymmetric;
- the script needs to search for new bands/cells;
- the result would depend on choosing among multiple entropy metrics after seeing values.

## 10. Wording Fence

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

## 11. Required Stage 0 Artifacts If Implemented

Output root:

`hsi_agents_project/results/hsi_v2/q2_internal_routing_entropy/stage0_feasibility/`

Required artifacts:

- `summary.json`
- `report.md`
- `catchup_cells.csv`
- `entropy_path_feasibility.csv`
- `manifest.json`

Any Stage 1 artifacts require a v0.2 contract and are intentionally not specified here.

## 12. Acceptance Before Execution

Before Stage 0 execution:

1. Sofia/Iban must accept or revise this v0.1 scope.
2. Grace must audit the catch-up-domain freeze and entropy-object definition.
3. The final v0.1 must be SHA256-anchored.
4. A Decision Log entry must record acceptance.

Before Stage 1 execution:

1. Stage 0 must complete.
2. A v0.2 contract must be written from the Stage 0 result.
3. Sofia/Iban/Grace must accept v0.2.
4. A new Decision Log entry must authorize Stage 1.
