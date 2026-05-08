# HSI v2 Phase 3 Reproducibility Freeze

## Purpose

This document is the repo-facing recovery note for the frozen Phase 3 internal diagnostic stack. It does not replace the canonical scientific record in `HSI_v2_Decision_Log.md`; it gives an external reviewer or future agent the shortest safe path from code to the Phase 3 artifacts used by the current manuscript cycle.

Phase 3 is frozen for the current manuscript cycle by D-0078. The freeze means:

- allowed: bug fixes, reproducibility hardening, clearer logs, manifests, and documentation;
- not allowed without a new Decision Log entry: new Phase 3 claims, new manuscript-facing figures, reinterpretation of D-0066 through D-0078, or wording that implies geometry recovered / topology discovered / curvature measured / Phase 3 solved.

The local code marker is `v2/phase3/FROZEN.md`.

## Current Reading

The accepted internal Phase 3 reading is narrow:

- local geometric candidates tested in Phase 3 do not separate observed `B/E` from the LZ-family null envelope;
- the surviving discriminator is thermodynamic routing concentration, measured through component-normalized low-entropy flow;
- this is not a Gate 2 geometry solution and does not satisfy or refute the three Porta 2 falsification conditions in the paper.

The strongest hardening sequence is:

1. canonical `matched-lz` envelope;
2. disjoint-seed `matched-lz` rerun;
3. stronger `phase-matched-lz` null family.

The important pattern is not uniform raw scaffold survival. Under `phase-matched-lz`, the raw phase-channel scaffold weakens at the boundary, while the N3-07 flow-entropy discriminator still preserves the observed `B/E` separation.

## Canonical Phase 3 Code

Run commands from `hsi_agents_project/` with the project virtual environment activated.

| Script | Role |
|--------|------|
| `hsi_v2_phase3_component_quotient_child_routing_spectrum.py` | N3-05c component-normalized quotient readout |
| `hsi_v2_phase3_hamming_quotient_child_routing_spectral_graph.py` | N3-06 Hamming quotient falsification check |
| `hsi_v2_phase3_flow_entropy_readout.py` | N3-07 direct low-entropy flow readout |
| `hsi_v2_phase3_robustness_matrix.py` | Frozen Phase 3 robustness matrix over canonical, disjoint-seed, and phase-matched LZ-family runs |

The robustness matrix defaults are pinned to exact timestamped artifacts, not "latest by modification time". Parent-directory auto-selection is disabled unless `--allow-latest` is passed for exploratory local work.

```powershell
python.exe .\hsi_v2_phase3_robustness_matrix.py
```

## Canonical Artifact Families

Generated results are intentionally excluded from git by `.gitignore` because they can be large. Reproducibility therefore depends on pinned paths, `manifest.json`, and report/CSV artifacts in the local or archived result bundle.

Primary Phase 3 artifacts:

| Artifact | Purpose |
|----------|---------|
| `results/hsi_v2/phase3/flow_entropy_readout/phase3-flow-entropy-readout__src-620fcc3412__20260501T140015/` | Canonical N3-07 matched-LZ envelope run |
| `results/hsi_v2/phase3/flow_entropy_readout_independent_seeds/phase3-flow-entropy-readout__src-e8827e75d7__20260502T153352/` | Disjoint-seed matched-LZ rerun for D-0074 |
| `results/hsi_v2/phase3/flow_entropy_readout_phase_matched_lz_reval/phase3-flow-entropy-readout__src-9de9a95303__20260503T073503/` | Phase-matched-LZ hard-null rerun for D-0076 |
| `results/hsi_v2/phase3/robustness_matrix/phase3-robustness-matrix__cases-3__top-128__20260503T082138/` | Frozen robustness matrix synthesis |

Required Phase 2 source artifacts for the Phase 3 robustness chain:

| Artifact | Purpose |
|----------|---------|
| `results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/` | Canonical N2-14 transition graph source |
| `results/hsi_v2/phase2/phase_channel_transition_graph_phase_matched_lz_reval/phase2-phase-channel-transition-graph__runs-1__20260503T073426/` | Phase-matched-LZ N2-14 revalidation source |

Each artifact family should contain at minimum:

- `summary.json`;
- `report.md`;
- `manifest.json`;
- the script-specific CSV files listed in the report.

The robustness matrix artifact also emits `robustness_matrix.csv` and `case_summary.csv`.

## Audit Trail

Canonical decisions and audits live outside this repo-facing folder:

- `HSI_v2_Decision_Log.md`, D-0066 through D-0079;
- `../docs/HSI-audit-Phase3-N3-07.md`;
- `../docs/HSI-audit-Phase3-N3-07-Rerun.md`;
- `../docs/HSI-audit-Phase-Matched-LZ-Null.md`;
- `../docs/HSI-audit-Phase3-Robustness-Matrix.md`;
- `../docs/HSI-audit-Phase3-Editorial-Architecture-Draft.md`;
- `../docs/HSI-audit-Phase3-LaTeX-Integration.md`.

## Boundary With Phase 4

Phase 4 starts from D-0079. Phase 4 is the predictive/pre-registered continuation of the program and must not be folded back into the current manuscript without a new explicit editorial decision.

Use this Phase 3 freeze document as a baseline, not as an invitation to expand Phase 3 claims.
