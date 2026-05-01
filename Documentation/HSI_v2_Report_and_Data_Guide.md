# HSI v2 Report and Data Guide

## Purpose

This document explains how to navigate the HSI v2 outputs now living under `results/hsi_v2/`, which scripts produce them, and how to reconstruct the scientific reading from stored artifacts.

It complements, rather than replaces:

- `README.md` for repo-wide orientation
- `Documentation/Results_Data_Guide.md` for the legacy v1 result stack
- `Documentation/HSI_v2_Phase1_Live_Findings.md` for the current scientific interpretation
- `Documentation/HSI_v2_Phase2_Live_Findings.md` for the current post-gate temporal readout
- `Documentation/HSI_v2_Phase2_Visual_Readouts_Plan.md` for planned Phase 2 figures after sensitivity checks
- `Documentation/HSI_v2_Phase2_Reproducibility_Freeze.md` for the frozen Phase 2 paper-support recovery path
- `../docs/HSI_v2_Decision_Log.md` for methodological defaults and decision history

---

## Current HSI v2 Scope

HSI v2 currently covers three disciplined layers:

1. **Preflight**
   Factor complexity, branching, and early projective-hierarchy discrimination against strong nulls.

2. **Phase 1**
   Local pattern spaces `P_m`, adjacent-scale projections, fibers, moments, and derived readouts of coherence and transport.

3. **Phase 2**
   Post-gate temporal transport prototypes. The current strict objects are:
   - observed-anchored `null_pressure`
   - strict-band transport-defect pilots

4. **Phase 3**
   Pre-geometric defect and spectrum entry point. Current objects include:
   - minimal commuting-square defect plus sparse bridge-graph spectrum pilot
   - lag-conditioned and phase-channel spectral readouts
   - exploratory child-routing quotient graphs, including the pooled-mass weighted quotient
   - direct flow-entropy readouts derived from component-normalized quotient artifacts

The current goal is intentionally narrow:

> demonstrate or refute a non-trivial projective hierarchy before attempting defects, graphs, spectrum, or geometry.

Phase 2 and Phase 3 should therefore be read with care:

- it is downstream of the Phase 1 gate
- it is still selective and disciplined
- not every exploratory Phase 2/3 pilot is yet canonical

---

## Script Map

### Preflight layer

| Script | Output family | Role |
|--------|---------------|------|
| `hsi_v2_preflight_factor_complexity.py` | `results/hsi_v2/preflight/preflight-fc__*/` | Runs factor-complexity and branching over one variant/iteration plus nulls |
| `hsi_v2_preflight_report.py` | `results/hsi_v2/preflight/reports/preflight-report__*/` | Consolidates a coherent preflight batch |

### Phase 1 layer

| Script | Output family | Role |
|--------|---------------|------|
| `hsi_v2_phase1_run.py` | `results/hsi_v2/phase1/phase1-tower__*/` and `results/hsi_v2/phase1/nulls/phase1-tower__*/` | Builds one full Phase 1 tower run or one strong-null surrogate run |
| `hsi_v2_phase1_report.py` | `results/hsi_v2/phase1/reports/phase1-report__*/` | Summarizes tower width, retention, survival, entropy, and moments |
| `hsi_v2_phase1_coherence_report.py` | `results/hsi_v2/phase1/reports/phase1-coherence__*/` | Summarizes tail coherence over the last edges of the tower |
| `hsi_v2_phase1_transport_report.py` | `results/hsi_v2/phase1/reports/phase1-transport__*/` | Summarizes transport persistence across consecutive edges |
| `hsi_v2_phase1_gate_map.py` | `results/hsi_v2/phase1/reports/phase1-gate-map__*/` | Builds a compact Porta 1 table plus a 2D gate-plane visual (`retention@last` vs `active_mean_tail`) as `SVG` and optional `PNG` |

### Phase 2 layer

| Script | Output family | Role |
|--------|---------------|------|
| `hsi_v2_phase2_null_pressure_window_sweep.py` | `results/hsi_v2/phase2/window_sweep*/phase2-return-lag-null-pressure__*/` | Runs the strict observed-anchored return-profile opening against nulls over a deep window band |
| `hsi_v2_phase2_transport_defect_strict.py` | `results/hsi_v2/phase2/transport_defect_strict*/phase2-transport-defect-strict__*/` | Runs the strict-band transport-defect pilot on observed-anchored support |
| `hsi_v2_phase2_parent_survival_revalidation.py` | `results/hsi_v2/phase2/parent_survival_revalidation/phase2-parent-survival-revalidation__*/` | Runs the canonical two-stage shell-lag revalidation pipeline: independent lag probe plus lag-aware parent-survival band |
| `hsi_v2_phase2_parent_density_pipeline.py` | `results/hsi_v2/phase2/parent_density_pipeline/phase2-parent-density-pipeline__*/` | Runs the canonical one-command density pipeline: shell-lag revalidation plus partition-level mass retention and survivor-internal density deformation |
| `hsi_v2_phase2_parent_shell_atlas.py` | `results/hsi_v2/phase2/parent_shell_atlas/phase2-parent-shell-atlas__*/` | Builds the lag-aware shell atlas from a canonical revalidation artifact |
| `hsi_v2_phase2_parent_shell_atlas_pipeline.py` | `results/hsi_v2/phase2/parent_shell_atlas_pipeline/phase2-parent-shell-atlas-pipeline__*/` | Runs the canonical one-command shell-atlas pipeline: revalidation plus lag-aware shell atlas |
| `hsi_v2_phase2_parent_shell_atlas_sensitivity.py` | `results/hsi_v2/phase2/parent_shell_atlas_sensitivity/phase2-parent-shell-atlas-sensitivity__*/` | Runs the N2-06 shell-atlas sensitivity sweep over top-k and lag neighborhoods |
| `hsi_v2_phase2_jitter_provenance.py` | `results/hsi_v2/phase2/jitter_provenance/phase2-jitter-provenance__*/` | Builds the N2-07 same-parent temporal provenance readout from a shell-atlas artifact |
| `hsi_v2_phase2_jitter_provenance_sensitivity.py` | `results/hsi_v2/phase2/jitter_provenance_sensitivity/phase2-jitter-provenance-sensitivity__*/` | Builds the N2-07b jitter-provenance sensitivity envelope over N2-06 atlas artifacts |
| `hsi_v2_phase2_child_destination_routing.py` | `results/hsi_v2/phase2/child_destination_routing/phase2-child-destination-routing__*/` | Builds the N2-08 child-destination routing readout over monitored shell parents |
| `hsi_v2_phase2_deficit_jitter_plane.py` | `results/hsi_v2/phase2/visual_readouts/phase2-deficit-jitter-plane__*/` | Renders a deterministic visual plane from audited shell-atlas sensitivity artifacts |

### Phase 3 layer

| Script | Output family | Role |
|--------|---------------|------|
| `hsi_v2_phase3_min_defect_spectrum_pilot.py` | `results/hsi_v2/phase3/min_defect_spectrum/phase3-min-defect-spectrum__*/` | Runs the pre-geometric minimal defect/spectrum pilot: commuting-square defect plus sparse bridge-graph normalized-Laplacian readout |
| `hsi_v2_phase3_defect_spectrum_sensitivity.py` | `results/hsi_v2/phase3/defect_spectrum_sensitivity/phase3-defect-spectrum-sensitivity__*/` | Runs the seed/window sensitivity sweep for the minimal defect/spectrum pilot |
| `hsi_v2_phase3_lag_conditioned_defect_spectrum_pilot.py` | `results/hsi_v2/phase3/lag_conditioned_defect_spectrum/phase3-lag-conditioned-defect-spectrum__*/` | Runs the N3-02 lag-conditioned follow-up: anchor-defined local square evaluated in a lag-shifted candidate window |
| `hsi_v2_phase3_phase_channel_spectral_graph.py` | `results/hsi_v2/phase3/phase_channel_spectral_graph/phase3-phase-channel-spectral-graph__*/` | Runs the N3-03 non-local spectral readout over the audited N2-14 phase-channel transition graph |
| `hsi_v2_phase3_micro_child_routing_spectral_graph.py` | `results/hsi_v2/phase3/micro_child_routing_spectral_graph/phase3-micro-child-routing-spectral-graph__*/` | Runs the N3-04 exploratory micro-spectral graph over child-routing events inside the N3-03 channel scaffold |
| `hsi_v2_phase3_quotient_child_routing_spectral_graph.py` | `results/hsi_v2/phase3/quotient_child_routing_spectral_graph/phase3-quotient-child-routing-spectral-graph__*/` | Runs the N3-05 exploratory prefix/suffix quotient graph over retained child-routing flow |
| `hsi_v2_phase3_weighted_quotient_child_routing_spectral_graph.py` | `results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph/phase3-weighted-quotient-child-routing-spectral-graph__*/` | Runs the N3-05b exploratory pooled-mass weighted quotient graph over retained child-routing flow |
| `hsi_v2_phase3_component_quotient_child_routing_spectrum.py` | `results/hsi_v2/phase3/component_quotient_child_routing_spectrum/phase3-component-quotient-spectrum__*/` | Runs the N3-05c component-wise normalized readout over an N3-05b weighted quotient artifact |
| `hsi_v2_phase3_hamming_quotient_child_routing_spectral_graph.py` | `results/hsi_v2/phase3/hamming_quotient_child_routing_spectral_graph/phase3-hamming-quotient-spectral-graph__*/` | Runs the N3-06 exploratory Hamming quotient radius sweep over an N3-05b weighted quotient artifact |
| `hsi_v2_phase3_flow_entropy_readout.py` | `results/hsi_v2/phase3/flow_entropy_readout/phase3-flow-entropy-readout__*/` | Runs the N3-07 direct low-entropy flow readout over an N3-05c component-normalized quotient artifact |

Other Phase 2 public scripts exist in the repo, but the current scientific reading should prioritize the strict objects above unless the decision log says otherwise.

---

## Output Layout

### Preflight

`results/hsi_v2/preflight/`

- `preflight-fc__.../factor_complexity.json`
  Raw factor-complexity payload for one variant and its controls.
- `preflight-fc__.../manifest.json`
  Inputs, arguments, paths, and timestamps for the run.
- `reports/preflight-report__...json`
  Machine-readable batch summary.
- `reports/preflight-report__...md`
  Human-readable synthesis.

### Phase 1

`results/hsi_v2/phase1/`

- `phase1-tower__.../dataset.json`
  Dataset metadata and run parameters.
- `phase1-tower__.../pattern_spaces/`
  Stored `P_m` summaries per scale.
- `phase1-tower__.../projection_towers/`
  Projection summaries per policy and edge.
- `phase1-tower__.../fibers/`
  Fiber summaries and per-edge distributions.
- `phase1-tower__.../moments/`
  Moment payloads for `q = 0,1,2,3`.
- `phase1-tower__.../phase1_summary.json`
  Compact single-run summary.
- `phase1-tower__.../manifest.json`
  Inputs, arguments, outputs, and generation metadata.

`results/hsi_v2/phase1/nulls/`

- `phase1-tower__.../dataset.json`
  Same artifact family as an observed Phase 1 run, but explicitly marked as a surrogate through:
  - `config.sequence_kind = "null_surrogate"`
  - `config.null_model`
  - `config.null_seed`
  - `config.source_variant`
- `phase1-tower__.../manifest.json`
  Same manifest structure, with the source snapshot and null-generation arguments preserved.
- `phase1-tower__.../(pattern_spaces|projection_towers|fibers|moments)/`
  The full Phase 1 tower outputs for that surrogate sequence.

Strong-null Phase 1 runs live under `phase1/nulls/` so they do not get mixed into the default observed-batch reports.

`results/hsi_v2/phase1/reports/`

- `phase1-report__...`
  Comparative tower report.
- `phase1-coherence__...`
  Comparative tail-coherence report.
- `phase1-transport__...`
  Comparative transport-persistence report.
- `phase1-gate-map__...`
  Compact Porta 1 table plus `CSV`, `JSON`, `Markdown`, and visual summary artifacts (`SVG`, plus `PNG` when matplotlib is available).

### Phase 2

`results/hsi_v2/phase2/`

- `window_sweep*/phase2-return-lag-null-pressure__.../`
  Strict return-profile comparison runs, with dataset, summary, report, and manifest.
- `transport_defect_strict*/phase2-transport-defect-strict__.../`
  Strict-band transport-defect runs, again with dataset, summary, report, and manifest.
- `parent_survival_revalidation/phase2-parent-survival-revalidation__.../`
  Canonical wrapper runs. Each pipeline run contains:
  - a top-level `summary.json`, `report.md`, and `manifest.json`
  - a nested `probe/phase2-parent-shell-lag-probe__.../` artifact family
  - a nested `lagaware/phase2-parent-survival-band-lagaware__.../` artifact family
- `parent_density_pipeline/phase2-parent-density-pipeline__.../`
  Canonical density wrapper runs. Each pipeline run contains:
  - a top-level `summary.json`, `report.md`, and `manifest.json`
  - a nested `revalidation/phase2-parent-survival-revalidation__.../` artifact family
  - a nested `density/phase2-parent-density-residual__.../` artifact family
  The density readout separates full partition mass retention from survivor-internal deformation. Older exact-only density artifacts should not be read as full partition retention.
- `parent_shell_atlas/phase2-parent-shell-atlas__.../`
  Lag-aware shell atlas artifacts derived from a canonical revalidation run. Each atlas run contains:
  - `structure.csv`, `subtypes.csv`, `patterns.csv`, `concentration.csv`, and `atlas.csv`
  - pooled `net_deficit_mass`, `gross_loss_pressure_sum`, and `jitter_gap_mass`
  - observation-conditional shell subtype summaries for the monitored child universe
- `parent_shell_atlas_pipeline/phase2-parent-shell-atlas-pipeline__.../`
  Canonical shell-atlas wrapper runs. Each pipeline run contains:
  - a top-level `summary.json`, `report.md`, and `manifest.json`
  - a nested `revalidation/phase2-parent-survival-revalidation__.../` artifact family
  - a nested `atlas/phase2-parent-shell-atlas__.../` artifact family
- `parent_shell_atlas_sensitivity/phase2-parent-shell-atlas-sensitivity__.../`
  N2-06 sensitivity wrapper runs. Each sweep contains:
  - a top-level `summary.json`, `report.md`, `manifest.json`, `structure_sensitivity.csv`, `concentration_sensitivity.csv`, and `subtype_sensitivity.csv`
  - nested top-k and lag-specific probe, lag-aware, and atlas artifact families
  - the same `net_deficit_mass`, `gross_loss_pressure_sum`, and `jitter_gap_mass` contract as the atlas
- `jitter_provenance/phase2-jitter-provenance__.../`
  N2-07 same-parent temporal provenance readouts derived from shell-atlas artifacts. Each run contains:
  - `source_provenance.csv`, `pattern_provenance.csv`, and `window_deltas.csv`
  - `summary.json`, `report.md`, `dataset.json`, and `manifest.json`
  - compensated-loss and provenance-shift fields that separate temporal jitter from uncompensated net deficit
- `jitter_provenance_sensitivity/phase2-jitter-provenance-sensitivity__.../`
  N2-07b sensitivity wrappers across N2-06 atlas artifacts. Each run contains:
  - `source_envelope.csv`, `source_provenance_sensitivity.csv`, and `pattern_provenance_sensitivity.csv`
  - `summary.json`, `report.md`, `dataset.json`, and `manifest.json`
  - stability ranges for compensated-loss share, net-deficit share, and provenance class across top-k and lag
- `child_destination_routing/phase2-child-destination-routing__.../`
  N2-08 child-destination routing readouts over monitored shell parents. Each run contains:
  - `source_routing.csv`, `pattern_routing.csv`, and `routing_rows.csv`
  - `summary.json`, `report.md`, `dataset.json`, and `manifest.json`
  - child-destination retention, candidate reroute share, top-child match mass, and preview-truncation flags
- `visual_readouts/phase2-deficit-jitter-plane__.../`
  Deterministic Phase 2 visual readouts. The deficit-jitter plane contains:
  - `deficit_jitter_plane.svg`
  - `deficit_jitter_plane.png`
  - `deficit_jitter_points.csv`
  - `summary.json`, `report.md`, and `manifest.json`
- exploratory families such as `phase2-return-lag__...`, `phase2-sparse-graph__...`, `phase2-square-cocycle__...`, and `phase2-defect-pilot__...`
  Useful for search-space recovery, but not all of them are current canonical evidence.

### Phase 3

`results/hsi_v2/phase3/`

- `min_defect_spectrum/phase3-min-defect-spectrum__.../`
  Minimal pre-geometric defect/spectrum pilots. Each run contains:
  - `dataset.json`
  - `summary.json`
  - `report.md`
  - `compact_rows.csv`
  - `manifest.json`
  The readout combines `delta_square`-style commuting-square defects with sparse bridge graph metrics and a normalized-Laplacian spectrum. It is not a geometry claim.
- `defect_spectrum_sensitivity/phase3-defect-spectrum-sensitivity__.../`
  Seed/window sensitivity sweeps over the minimal pilot. Each run contains:
  - `summary.json`
  - `report.md`
  - `compact_rows.csv`
  - `manifest.json`
  The current accepted internal readout is that the minimal object rejects `markov1` but is not a `matched-lz` separator.
- `lag_conditioned_defect_spectrum/phase3-lag-conditioned-defect-spectrum__.../`
  N3-02 lag-conditioned pilots. Each run contains:
  - `dataset.json`
  - `summary.json`
  - `report.md`
  - `compact_rows.csv`
  - `source_summary.csv`
  - `manifest.json`
  The current smoke readout is that a single-lag local square keeps B/E coherent and rejects `markov1`, but is still shadowed by `matched-lz`.
- `phase_channel_spectral_graph/phase3-phase-channel-spectral-graph__.../`
  N3-03 non-local spectral readouts over N2-14 transition-graph artifacts. Each run contains:
  - `summary.json`
  - `report.md`
  - `series_summary.csv`
  - `transition_counts.csv`
  - `graph_nodes.csv`
  - `graph_edges.csv`
  - `manifest.json`
  The current readout is accepted-internal after Grace's N3-03 audit and D-0068.

- `micro_child_routing_spectral_graph/phase3-micro-child-routing-spectral-graph__.../`
  N3-04 exploratory micro-spectral child-routing graphs. Each run contains:
  - `summary.json`
  - `report.md`
  - `graph_summary.csv`
  - `graph_nodes.csv`
  - `graph_edges.csv`
  - `manifest.json`
  The current smoke readout is unaudited and should be treated as a design probe, not as a Phase 3 separator.

- `quotient_child_routing_spectral_graph/phase3-quotient-child-routing-spectral-graph__.../`
  N3-05 exploratory prefix/suffix quotient graphs. Each run contains:
  - `summary.json`
  - `report.md`
  - `graph_summary.csv`
  - `graph_nodes.csv`
  - `graph_edges.csv`
  - `manifest.json`
  The current smoke readout is unaudited and indicates that unit edge presence is too coarse; pooled mass weights are now tested separately by N3-05b.

- `weighted_quotient_child_routing_spectral_graph/phase3-weighted-quotient-child-routing-spectral-graph__.../`
  N3-05b exploratory pooled-mass quotient graphs. Each run contains:
  - `summary.json`
  - `report.md`
  - `graph_summary.csv`
  - `graph_nodes.csv`
  - `graph_edges.csv`
  - `manifest.json`
  The accepted internal D-0069 reading is weighted flow-capacity over a fixed prefix/suffix quotient: observed `B/E` retain high flow, `matched-lz` is reduced, and `markov1` has no quotient flow. It is not a connected-geometry claim.

- `component_quotient_child_routing_spectrum/phase3-component-quotient-spectrum__.../`
  N3-05c component-wise normalized quotient readouts derived from N3-05b artifacts. Each run contains:
  - `summary.json`
  - `report.md`
  - `source_summary.csv`
  - `component_summary.csv`
  - `manifest.json`
  The accepted internal D-0069 reading is that observed `B/E` remain lower-entropy / less diffuse than `matched-lz` after removing absolute mass scale inside the fixed 24-component prefix/suffix quotient.

- `hamming_quotient_child_routing_spectral_graph/phase3-hamming-quotient-spectral-graph__.../`
  N3-06 exploratory Hamming quotient radius sweeps derived from N3-05b artifacts. Each run contains:
  - `summary.json`
  - `report.md`
  - `graph_summary.csv`
  - `graph_nodes.csv`
  - `graph_edges.csv`
  - `manifest.json`
  The accepted-negative D-0070 reading is that Hamming radius fuses the prefix/suffix scaffold, but the connectivity transition is shared by observed B/E and `matched-lz`; Hamming connectivity is not a geometric separator.

- `flow_entropy_readout/phase3-flow-entropy-readout__.../`
  N3-07 direct flow-entropy readouts derived from N3-05c artifacts. Each run contains:
  - `summary.json`
  - `report.md`
  - `flow_entropy_summary.csv`
  - `manifest.json`
  The accepted internal D-0071 reading is that observed `B/E` score `4/4` as `strict-low-entropy-flow` against the strongest `matched-LZ` seed envelope in both negative and positive channels. This is an internal routing-thermodynamic closure, not a geometry or paper-facing Phase 3 claim.

---

## How to Reconstruct a Result

### If you want the scientific interpretation first

Read:

1. `Documentation/HSI_v2_Phase1_Live_Findings.md`
2. `Documentation/HSI_v2_Phase2_Live_Findings.md`
3. the latest `phase1-report__...md`
4. the latest `phase1-coherence__...md`
5. the latest `phase1-transport__...md`

### If you want to audit one run end to end

Read:

1. `manifest.json`
2. `phase1_summary.json`
3. `fibers/fibers_prefix.json`
4. `fibers/fibers_suffix.json`
5. `moments/moments_prefix.json`
6. `moments/moments_suffix.json`

### If you want to know whether a conclusion is canonical or provisional

Check:

- `../docs/HSI_v2_Decision_Log.md` for accepted defaults and methodological decisions
- `Documentation/HSI_v2_Phase1_Live_Findings.md` for current interpretation
- `Documentation/HSI_v2_Phase2_Live_Findings.md` for the current strict Gate 2 readout

---

## Current Phase 1 Taxonomy

Based on the current `3 x 1,000,000` bit batch with scales `8,12,16,20,24,28,32`:

- `B/E/I` -> coherent
- `F` -> rigid
- `M/N` -> crystalline
- `A/J/L` -> collapsed
- `K` -> trivial

This taxonomy is supported by three stacked report layers:

1. tower richness
2. tail coherence
3. transport persistence

The key point is not only that some towers persist. It is that **different kinds of towers persist for different mechanisms**.

---

## Interpretation Boundaries

What HSI v2 currently supports:

- a non-trivial projective hierarchy exists in the HSI family
- not all persistent hierarchies are equivalent
- `B/E/I` are richer than rigid or crystalline controls in the current observable stack

What HSI v2 does **not** yet support:

- a final geometric theory
- a complete causal-flow theory replacing v1 transfer entropy
- the claim that persistence alone is HSI-specific

---

## Minimal Command Reference

```powershell
# Preflight on one run
python .\hsi_v2_preflight_factor_complexity.py --variant B --iteration 20 --segment-bits 1000000 --num-segments 3 --m-min 8 --m-max 32

# Preflight summary
python .\hsi_v2_preflight_report.py --variants B,E,I,F,A,J,K,L,M

# Phase 1 on one run
python .\hsi_v2_phase1_run.py --variant B --iteration 20 --segment-bits 1000000 --num-segments 3

# Phase 1 on one strong-null surrogate
python .\hsi_v2_phase1_run.py --variant B --iteration 20 --segment-bits 1000000 --num-segments 3 --null-model shuffled --null-seed 17

# Phase 1 comparative summaries
python .\hsi_v2_phase1_report.py
python .\hsi_v2_phase1_coherence_report.py
python .\hsi_v2_phase1_transport_report.py
python .\hsi_v2_phase1_gate_map.py

# Phase 2 shell atlas pipeline
python .\hsi_v2_phase2_parent_shell_atlas_pipeline.py --profile canonical-all

# Phase 2 shell atlas sensitivity sweep
python .\hsi_v2_phase2_parent_shell_atlas_sensitivity.py --profile canonical-sensitivity

# Phase 2 same-parent jitter provenance
python .\hsi_v2_phase2_jitter_provenance.py

# Phase 2 jitter provenance sensitivity
python .\hsi_v2_phase2_jitter_provenance_sensitivity.py

# Phase 2 child-destination routing
python .\hsi_v2_phase2_child_destination_routing.py

# Phase 2 deficit-jitter visual plane
python .\hsi_v2_phase2_deficit_jitter_plane.py
```

---

## Recommended Reading Order

For experimental recovery:

1. `README.md`
2. `Documentation/HSI_v2_Report_and_Data_Guide.md`
3. `Documentation/HSI_v2_Phase1_Live_Findings.md`
4. `Documentation/HSI_v2_Phase2_Live_Findings.md`
5. `../docs/HSI_v2_Decision_Log.md`

For paper writing:

1. `Documentation/HSI_v2_Phase1_Live_Findings.md`
2. `Documentation/HSI_v2_Phase2_Live_Findings.md`
3. latest `phase1-report__...md`
4. latest `phase1-coherence__...md`
5. latest `phase1-transport__...md`
6. `../docs/HSI_v2_Decision_Log.md`

For external review:

1. `manifest.json` of the target run
2. report markdown and matching JSON
3. `Documentation/HSI_v2_Phase1_Live_Findings.md`
4. `Documentation/HSI_v2_Phase2_Live_Findings.md`
5. `../docs/HSI_v2_Decision_Log.md`

For visual readout planning:

1. `Documentation/HSI_v2_Phase2_Visual_Readouts_Plan.md`
2. latest `parent_shell_atlas_sensitivity/.../summary.json`
3. latest `parent_shell_atlas_sensitivity/.../structure_sensitivity.csv`
4. latest `parent_shell_atlas_sensitivity/.../concentration_sensitivity.csv`
5. latest `parent_shell_atlas_sensitivity/.../subtype_sensitivity.csv`
