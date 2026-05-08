# HSI v2 Phase 1 Threshold Sensitivity Preregistration

Status: pre-execution contract draft accepted by Ariadna after the v2.01 hostile-review pass.

Purpose: test whether the Gate 1 compact-selective Phase 1 readout depends on a narrow hand-tuned stability filter choice in `v2/phase1/tower.py`.

## Trigger

A brief-blind hostile review of manuscript v2.01 asked whether the Phase 1 result could be an artifact of:

- `cv_max = 1.5`
- `min_count_floor = 16`
- `min_count_rate = 1e-6`
- `min_segment_support = max(2, ceil(N/2))`

This preregistration defines a read-only sensitivity analysis over those hyperparameters before running the experiment.

## Fixed Experimental Grid

The run uses a one-at-a-time grid around the canonical Phase 1 point.

Baseline:

- `cv_max = 1.5`
- `min_count_floor = 16`
- `min_count_rate = 1e-6`
- `min_segment_support = default` (`max(2, ceil(N/2))`; effective value `2` for `N=3`)

OAT values:

- `cv_max`: `1.0, 1.5, 2.0, 2.5, 3.0, inf`
- `min_count_floor`: `4, 8, 16, 32, 64`
- `min_count_rate`: `0, 1e-7, 1e-6, 1e-5, 1e-4`
- `min_segment_support`: `1, 2, 3`

Deduplicated total: `16` combinations.

## Fixed Phase 1 Scope

- Variants: `B,E,I`
- Sequence kinds: observed, `matched-lz`, `phase-matched-lz`
- Initial null seed: `607`
- Escalation null seed: `709`
- Scales: `8,12,16,20,24,28,32`
- Policies: `prefix,suffix`
- Iteration: `20`
- Segment layout: `3 x 1,000,000` bits
- Output root: `results/hsi_v2/phase1_sensitivity_thresholds/`

The driver must not modify `v2/phase1/tower.py`; it only passes existing CLI parameters to `hsi_v2_phase1_run.py`.

## Escalation Rule

For every combination, run observed + two null families with seed `607`.

Run the additional seed `709` only if the initial null-envelope readout is fragile:

- final verdict is not `survives`, or
- `score_margin < 0.30`, or
- `retention_margin < 0.30`.

The additional seed expands the null envelope; it cannot rescue a positive result by averaging down a hard null.

## Score and Verdict

All raw Phase 1 observables remain in the CSV. The scalar score is only an ordering aid:

`compact_selective_score = retention_last * min(prefix_survival_last, suffix_survival_last)`

For each threshold combination:

- `hsi_score_min`: minimum score over observed `B,E,I`
- `null_score_max`: maximum score over all null rows in the active envelope
- `score_margin = hsi_score_min - null_score_max`
- `hsi_retention_min`: minimum `retention_last` over observed `B,E,I`
- `null_retention_max`: maximum `retention_last` over all null rows
- `retention_margin = hsi_retention_min - null_retention_max`

Verdict:

- `survives`: `hsi_score_min >= 0.75`, `score_margin >= 0.25`, `hsi_retention_min >= 0.70`, and `retention_margin >= 0.25`.
- `weakens`: `hsi_score_min >= 0.60`, `score_margin >= 0.10`, `hsi_retention_min >= 0.55`, and `retention_margin >= 0.10`.
- `collapses`: otherwise.

## Hypotheses

H1: the compact-selective Gate 1 signature survives across a broad OAT region, meaning the canonical thresholds are not a narrow hand-tuned point.

H2: the compact-selective Gate 1 signature appears only near the canonical threshold combination, which would be a serious negative result requiring manuscript revision.

## Required Outputs

The run must emit:

- `summary.json`
- `report.md`
- `combo_summary.csv`
- `row_summary.csv`
- `oat_heatmap.csv`
- `oat_heatmap.svg`
- `manifest.json`

The manifest must include the preregistration SHA256, commands, parameters, child run paths, and threshold-combination metadata.

