# HSI v2 Level 2 L2-01: Quotient-Shape Regime Separation v0.1

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and recorded in the Decision Log.

Purpose: define the first Level 2 probe in the narrowest falsifiable form: test whether routing-quotient shape invariants, excluding retention and hard-margin as features, separate pre-existing channel regimes.

This is not a geometry claim. It is a candidate-invariant screening test.

## 1. Scientific Question

Do quotient-shape features of local child-routing graphs distinguish three already defined regimes:

- `interior`: strong predictive-channel bands;
- `boundary`: null-envelope catch-up bands;
- `inactive`: scout-inactive or strict-failure bands;

without using retention, hard-margin, or entropy-gap as explanatory variables?

If yes, Level 2 has a candidate shape invariant worth formalizing.

If no, the Phase 4/5 boundary remains empirically real but is not captured by this first quotient-shape invariant family.

Either result is valid.

## 2. Non-Goals

- Do not recover geometry.
- Do not claim topology, curvature, or Gate 2 closure.
- Do not fit a boundary law.
- Do not rescue Phase 5.
- Do not search for better labels after seeing shape results.
- Do not use `B_retention`, `hard_margin`, or `entropy_gap` as shape features.
- Do not modify the manuscript.

## 3. Input Discipline

L2-01 must use regime labels inherited from accepted Phase 4/5 readouts.

Allowed label sources:

- Phase 4 P4-04/P4-05/P4-06 accepted strict-grid readouts;
- Phase 5 B-mini accepted strict-grid readouts;
- Q2 accepted catch-up cell list for boundary labels only.

Forbidden as features:

- `B_retention`;
- `hard_margin`;
- `phase_matched_lz_max`;
- `block_entropy_max`;
- `entropy_gap`;
- `hard_null_min_family`;
- any label-derived score.

These quantities may appear in metadata and label provenance only.

## 4. Stage 0: Observability Budget

v0.1 authorizes no shape-separation statistic yet.

Stage 0, once accepted, may only:

1. inventory candidate bands/cells with existing or derivable child-routing rows;
2. identify which bands can be labelled `interior`, `boundary`, or `inactive` from already accepted readouts;
3. verify whether N3-05b/N3-05c local per-cell adapters can be applied without definition changes;
4. report counts per regime, arc, source, lag, and null family;
5. estimate compute cost;
6. propose the sealed Stage 1 input table and hashes.

Stage 0 may not:

- compute shape distances;
- compute class separation;
- run permutations;
- create figures;
- choose features based on observed separation;
- change regime labels.

## 5. Preferred Holdout Policy

Primary preference:

Use bands/arcs not used in Q2/D-0112-D0113 whenever the observability budget permits.

Reason:

Q2 already showed that `block-entropy` has a concentration advantage at catch-up. L2-01 should avoid simply rediscovering that same fact on the same cells and calling it geometry.

If the budget is insufficient for a holdout-only readout, Stage 0 must stop with verdict:

`insufficient-holdout-budget`

and no Stage 1 contract may be written without a new decision.

## 6. Candidate Shape Features

Stage 0 may verify availability of the following fixed feature vector from local N3-05c component summaries:

Primary observed-`B` shape vector:

- `component_weight_entropy_norm`
- `effective_weight_components`
- `top1_component_weight_share`
- `top3_component_weight_share`
- `top5_component_weight_share`
- `weighted_degree_entropy_norm`
- `weighted_mass_normalized_rho`

Secondary hard-null context:

the same vector for `phase-matched-lz` and `block-entropy`, reported separately.

No feature may be added after Stage 0 if its addition depends on observed class separation.

## 7. Stage 1 Candidate Statistic

Stage 1 is not authorized by this v0.1.

If Stage 0 succeeds, the expected Stage 1 contract should freeze:

- the exact input rows;
- the exact feature vector;
- the exact normalization rule;
- the exact class labels;
- the exact permutation scheme.

Proposed primary statistic for Stage 1:

`shape_separation_ratio = median_between_regime_distance / median_within_regime_distance`

using observed-`B` shape vectors only.

Proposed distance:

Euclidean distance after feature-wise rank normalization over the sealed Stage 1 table.

Proposed calibration:

Permutation of regime labels within arc blocks, preserving per-arc cell counts.

These are proposals only. They become binding only in a future v0.2 contract.

## 8. Outcome Categories For Future Stage 1

Future Stage 1 should use mutually exclusive categories similar to:

| category | meaning |
|---|---|
| `shape-invariant-supported` | quotient-shape separation is stronger than the preregistered permutation boundary |
| `weak-or-mixed-shape-separation` | some descriptive separation, but not enough for the frozen criterion |
| `no-shape-separation` | quotient-shape vectors do not distinguish regimes |
| `incomplete-inputs-stop` | sealed rows cannot support the test |

Exact thresholds are not authorized by v0.1 and must be frozen in v0.2.

## 9. Wording Fence

Allowed:

- `candidate quotient-shape invariant`
- `regime separation probe`
- `shape readout`
- `pre-geometric invariant screening`
- `Level 2 probe`

Forbidden:

- `geometry recovered`
- `topology discovered`
- `curvature measured`
- `Gate 2 solved`
- `Level 2 confirmed`
- `boundary law`
- `universal horizon`
- `shape proves mechanism`
- `retention-independent geometry established`

## 10. Manuscript Boundary

No manuscript edit is authorized by this draft.

Even if a future Stage 1 is positive, paper-facing integration requires:

1. accepted v0.2 contract;
2. execution;
3. Grace audit;
4. Sofia/Iban editorial decision;
5. new Decision Log entry.

## 11. Proposed Stage 0 Command

Proposed executable after acceptance:

```powershell
python.exe .\hsi_v2_level2_l2_01_quotient_shape_stage0.py `
  --output-dir results\hsi_v2\level2\l2_01_quotient_shape_regime_separation\stage0_observability
```

This command is not authorized until v0.1 is accepted, SHA256-anchored, and recorded in the Decision Log.

