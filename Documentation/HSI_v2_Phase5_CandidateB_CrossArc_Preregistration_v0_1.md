# HSI v2 Phase 5 Candidate B-mini Preregistration v0.1

Status: draft for Sofia/Iban/Grace review; not executable until signed and hashed.

Purpose: define a first, bounded Phase 5 contract for testing whether the finite predictive horizon observed in Phase 4 recurs across independent arcs, without promoting any boundary law or universal scaling claim.

## 1. Scientific Object

Phase 4 established one finite out-of-sample predictive horizon followed by hard-null envelope catch-up. Phase 5 Candidate B asks whether that structure is local to the Phase 4 arc or whether analogous finite horizons appear in other independent arcs.

This v0.1 chooses a pilot form, `B-mini`, before any full cross-arc program.

## 2. Wording Fence

Forbidden wording:

- `boundary law`
- `universal horizon`
- `scaling law derived`
- `horizon predicted`
- `frontier formula`
- `general predictive principle`
- `Phase 5 confirms Phase 4`
- `Phase 5 generalizes Phase 4`
- `rescue of the positive channel`

Allowed wording:

- `cross-arc finite-horizon test`
- `finite-horizon replication attempt`
- `hard-null envelope catch-up`
- `arc-local predictive horizon`
- `pilot evidence for or against horizon recurrence`
- `failure as a valid outcome`

## 3. Non-Goals

This contract does not authorize:

- scanning later Phase 4 bands for renewed signal;
- fitting any boundary-risk formula;
- deriving or claiming a boundary law;
- training on one set of arcs and validating on the same arcs;
- replacing failed arcs after observing results;
- any manuscript edit.

## 4. B-mini Design

B-mini is a small pilot with `2-3` frozen arcs, selected only after a read-only observability budget audit.

The pilot has two possible outcomes:

1. If finite horizons replicate in enough frozen arcs, a full Candidate B contract may be drafted later.
2. If the pattern does not replicate, Candidate B stops or is redesigned explicitly; failure is logged as a valid result.

This design deliberately prevents Candidate A from using the single Phase 4 boundary as an implicit training set.

## 5. Stage 0: Observability Budget Audit

Before any arc list is frozen, run a read-only inventory of available data.

The audit must report:

- available contiguous observable ranges by variant and source;
- ranges already used by Phase 2, Phase 3, and Phase 4 horizon work;
- candidate ranges far enough from the Phase 4 arc to count as independent;
- whether each candidate range supports the required band count and lag offsets;
- exclusions caused by missing snapshots, missing null runs, or insufficient forward room.

No lag-response experiment is authorized during Stage 0.

If fewer than two executable independent arcs exist, B-mini is not executable and the result is a design failure, not a scientific negative.

## 6. Arc Freeze Rule

After Stage 0, the arc list must be frozen in a new signed contract version before execution.

Rules:

- the frozen list must contain `2-3` arcs;
- arcs cannot overlap the Phase 4 P4-04/P4-05 horizon-bracket region;
- arcs cannot be replaced after any result is seen;
- failed arcs remain in the report;
- arc selection must not use hard-null envelope trajectories from the tested arcs.

If deterministic spacing over the executable budget is possible, prefer it over manual selection.

## 7. Lag Grid Policy

For B-mini, the default proposal is to reuse the P4-04/P4-05 lag grid for comparability.

This reuse is a non-trivial dependency and must be stated in every report.

Rationale: B-mini does not claim an independently discovered lag law. It tests whether the already defined Phase 4 channel object has finite-horizon analogues elsewhere.

If Sofia/Grace prefer a new lag grid, this v0.1 must be revised before execution. The grid cannot be changed after arc freeze.

## 8. Arc-Level Readout

For each frozen arc, run a compact P4-style forward test over consecutive bands.

The readout must include:

- B retention by band and lag;
- strongest hard-null envelope by band and lag;
- hard margin `B - max(null_envelope)`;
- source of the hard-null maximum;
- positive-channel diagnostic retained as non-universality control.

The hard-null envelope must include the accepted strong nulls available for the arc. If a null family is unavailable, the arc is not executable unless the contract is revised before execution.

## 9. Horizon Replication Criterion

An arc is counted as finite-horizon-like only if all conditions hold:

1. At least one early band has a strong negative-channel cell.
2. A later tested band shows hard-null envelope catch-up under the same frozen lag grid.
3. The catch-up is caused by null rise, B collapse, or a documented mixture; the mechanism must be reported, not inferred silently.
4. The positive channel does not become the primary explanatory object.

Provisional thresholds:

- strong cell: `B_retention >= 0.90` and `hard_margin >= 0.30`;
- null-shadow boundary: `hard_margin <= 0.05` with `B_retention >= 0.50`, or an explicitly reported B-collapse case;
- band tolerance: catch-up may occur within the tested forward band set only; no extra band may be added after seeing results.

These thresholds are draft values. They must be accepted or revised before the signed execution contract.

## 10. B-mini Outcome Rules

For `K=3` frozen arcs:

- `>=2/3` finite-horizon-like arcs: B-mini supports moving to a full Candidate B contract.
- `1/3`: ambiguous; stop and review before any full run.
- `0/3`: B-mini negative; do not run full Candidate B without a new rationale.

For `K=2` frozen arcs:

- `2/2`: B-mini supports moving to full Candidate B.
- `0/2` or `1/2`: B-mini does not support full Candidate B.

These are decision rules for whether to continue, not manuscript claims.

## 11. Feature-Tautology Guard

B-mini does not fit a feature model. However, arc selection and interpretation can still become tautological if they depend on the same null trajectories used to score success.

Therefore:

- Stage 0 may use observability, coverage, spacing, and existing observed-stream availability.
- Stage 0 may not select arcs because their hard-null envelope already looks favorable.
- Any optional descriptive features must be separated into:
  - observed-stream features;
  - null-envelope response features;
  - metadata/coverage features.

No fitted risk score is allowed in Phase 5 B-mini.

## 12. Failure Handling

Failure modes are first-class outcomes:

- no executable budget: design failure;
- frozen arcs inactive: negative pilot result;
- null envelope already high from the first band: negative or boundary-dominated pilot result;
- B collapses before null catch-up: finite horizon not established unless explicitly covered by the signed criterion;
- inconsistent arc results: ambiguous pilot result.

All failures must be recorded with the same visibility as successes.

## 13. Required Artifacts

Stage 0 audit artifacts:

- `summary.json`
- `report.md`
- `candidate_arcs.csv`
- `manifest.json`

If later authorized, B-mini execution artifacts:

- `summary.json`
- `report.md`
- `arc_results.csv`
- `source_response.csv`
- `manifest.json`

All artifacts must live under `hsi_agents_project/results/hsi_v2/phase5_cross_arc_horizon/`.

## 14. Acceptance Before Execution

Before any B-mini execution:

1. This draft must be revised into an execution contract.
2. The final contract must include the frozen arc list.
3. The final contract must include the frozen lag grid.
4. The final contract must include the accepted thresholds.
5. The final contract must be SHA256-anchored.
6. Grace must sign before execution.

Until then, only Stage 0 observability-budget audit is methodologically safe.

