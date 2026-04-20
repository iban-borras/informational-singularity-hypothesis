# HSI v2 Report and Data Guide

## Purpose

This document explains how to navigate the HSI v2 outputs now living under `results/hsi_v2/`, which scripts produce them, and how to reconstruct the scientific reading from stored artifacts.

It complements, rather than replaces:

- `README.md` for repo-wide orientation
- `Documentation/Results_Data_Guide.md` for the legacy v1 result stack
- `Documentation/HSI_v2_Phase1_Live_Findings.md` for the current scientific interpretation
- `../docs/HSI_v2_Decision_Log.md` for methodological defaults and decision history

---

## Current HSI v2 Scope

HSI v2 currently covers two disciplined stages:

1. **Preflight**
   Factor complexity, branching, and early projective-hierarchy discrimination against strong nulls.

2. **Phase 1**
   Local pattern spaces `P_m`, adjacent-scale projections, fibers, moments, and derived readouts of coherence and transport.

The current goal is intentionally narrow:

> demonstrate or refute a non-trivial projective hierarchy before attempting defects, graphs, spectrum, or geometry.

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

---

## How to Reconstruct a Result

### If you want the scientific interpretation first

Read:

1. `Documentation/HSI_v2_Phase1_Live_Findings.md`
2. the latest `phase1-report__...md`
3. the latest `phase1-coherence__...md`
4. the latest `phase1-transport__...md`

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
```

---

## Recommended Reading Order

For experimental recovery:

1. `README.md`
2. `Documentation/HSI_v2_Report_and_Data_Guide.md`
3. `Documentation/HSI_v2_Phase1_Live_Findings.md`
4. `../docs/HSI_v2_Decision_Log.md`

For paper writing:

1. `Documentation/HSI_v2_Phase1_Live_Findings.md`
2. latest `phase1-report__...md`
3. latest `phase1-coherence__...md`
4. latest `phase1-transport__...md`
5. `../docs/HSI_v2_Decision_Log.md`

For external review:

1. `manifest.json` of the target run
2. report markdown and matching JSON
3. `Documentation/HSI_v2_Phase1_Live_Findings.md`
4. `../docs/HSI_v2_Decision_Log.md`
