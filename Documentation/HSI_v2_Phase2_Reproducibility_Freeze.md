# HSI v2 Phase 2 Reproducibility Freeze

**Date:** 2026-05-01  
**Status:** frozen recovery guide for the current Phase 2 paper-support stack  
**Scope:** reproducibility and artifact navigation only; no new scientific claim

## Purpose

This document freezes the current Phase 2 recovery path after the paper integration of the strict `null_pressure` opening and the internal mechanistic appendix stack.

It is not the scientific source of truth. The canonical decisions remain in:

- `HSI_v2_Decision_Log.md`
- `../docs/HSI_v2_Phase2_Editorial_Architecture_Proposal.md`
- `Documentation/HSI_v2_Phase2_Live_Findings.md`

The purpose here is narrower: a future researcher should be able to identify which Phase 2 artifacts are canonical for the current paper draft, which scripts reproduce them, and which families are internal diagnostics rather than independent openings.

## Frozen Evidentiary Hierarchy

1. **Paper-safe opening:** strict `null_pressure` under D-0040 / D-0046.
2. **Paper appendix support:** D-0051 deficit-jitter plane and D-0058 lag-response heatmap.
3. **Internal diagnostic stack:** D-0049 through D-0062, summarized in the paper appendix traceability table.
4. **Exploratory or deprecated material:** older Phase 2 pilots remain useful for recovery, but they are not paper-facing unless explicitly promoted by the Decision Log.

This hierarchy is part of the freeze. Do not promote a lower layer without a new Decision Log entry.

## Canonical Paper-Support Artifacts

### Strict `null_pressure`

- Artifact family: `results/hsi_v2/phase2/window_sweep_deep/null_pressure_strict_d0039_reval/phase2-return-lag-null-pressure__obs-B__m-48__sel-bridge-linked__top-64__w-5__off-714M__step-3M__20260421T181545/`
- Role: first paper-safe observational opening of Gate 2.
- Read first: `report.md`, then `summary.json`, then `dataset.json`, then `manifest.json`.

### Deficit-jitter plane

- Stable figure alias: `results/hsi_v2/phase2/visual_readouts/deficit_jitter_plane_latest.png`
- Role: appendix diagnostic figure D-0051.
- Reading: topological deficit/redistribution support below `null_pressure`.

### Lag-response heatmap

- Stable figure alias: `results/hsi_v2/phase2/visual_readouts/lag_response_heatmap_latest.png`
- Role: appendix diagnostic figure D-0058.
- Reading: temporal phase-channel engagement below `null_pressure`.

## Public Script Families To Preserve

The following public scripts form the current reproducible Phase 2 stack:

- `hsi_v2_phase2_null_pressure_window_sweep.py`
- `hsi_v2_phase2_parent_survival_revalidation.py`
- `hsi_v2_phase2_parent_density_pipeline.py`
- `hsi_v2_phase2_parent_shell_atlas.py`
- `hsi_v2_phase2_parent_shell_atlas_pipeline.py`
- `hsi_v2_phase2_parent_shell_atlas_sensitivity.py`
- `hsi_v2_phase2_jitter_provenance.py`
- `hsi_v2_phase2_jitter_provenance_sensitivity.py`
- `hsi_v2_phase2_child_destination_routing.py`
- `hsi_v2_phase2_child_routing_persistence.py`
- `hsi_v2_phase2_child_routing_lag_response.py`
- `hsi_v2_phase2_child_routing_phase_validation.py`
- `hsi_v2_phase2_phase_lag_atlas.py`
- `hsi_v2_phase2_phase_lag_tri_estimator.py`
- `hsi_v2_phase2_lag_response_heatmap.py`
- `hsi_v2_phase2_phase_channel_width.py`
- `hsi_v2_phase2_phase_channel_width_figure.py`
- `hsi_v2_phase2_phase_channel_width_topk_sensitivity.py`
- `hsi_v2_phase2_off_universe_mass_readout.py`
- `hsi_v2_phase2_phase_channel_transition_graph.py`
- `hsi_v2_phase2_deficit_jitter_plane.py`

Each serious run must keep writing `summary.json`, `report.md`, `manifest.json`, and any relevant CSV/PNG/SVG side artifacts.

## Minimal Recovery Order

For a clean Phase 2 audit, read in this order:

1. `HSI_v2_Decision_Log.md` entries D-0040 through D-0064.
2. `hsi_agents_project/Documentation/HSI_v2_Phase2_Live_Findings.md`.
3. This freeze document.
4. The strict `null_pressure` artifact family listed above.
5. The two stable visual aliases listed above.
6. The paper appendix traceability table in `Paper_Latex/main_v2_cat.tex`.

## Guardrails

- Do not treat the internal diagnostic stack as a second Gate 2 opening.
- Do not claim a transport law, geometry, spectral geometry, wave law, or Gate 2 resolution from Phase 2 artifacts.
- Do not use timestamped figure paths in paper-facing LaTeX when a stable alias exists.
- Do not regenerate expensive artifacts unless the manifest or Decision Log says a contract changed.
- Do not hide a changed default in code only; record it in `HSI_v2_Decision_Log.md`.

## Current Transition

Phase 2 is now reproducibly frozen enough to serve as the substrate for the next formal step: a minimal Phase 3 defect/spectrum contract. That next step must define falsifiable defects before any geometry language is introduced.
