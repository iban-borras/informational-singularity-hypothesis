# HSI v2 Phase 2 Visual Readouts Plan

## Purpose

This note records the planned visual readouts for Phase 2 once the shell-atlas sensitivity sweep has stabilized.

The goal is to create figures that help readers see the empirical structure without smuggling in geometry that has not yet been proven.

## Timing

Do this after the N2-06 shell-atlas sensitivity sweep has been read and, ideally, after hostile/audit review of the resulting numerical interpretation.

Do not promote any visual to paper-facing status until it is reproducible from stored artifacts and robust across the relevant top-k and lag sensitivity checks.

## Figure Priority

### 1. Deficit-Jitter Phase Plane

Candidate role: primary paper figure.

Implemented entry point:

- `hsi_v2_phase2_deficit_jitter_plane.py`

Data source:

- `parent_shell_atlas_sensitivity/.../concentration_sensitivity.csv`
- or one canonical `parent_shell_atlas/.../concentration.csv`

Core axes:

- X: pooled net deficit mass normalized by the Markov-1 collapse mass at the same top-k and lag
- Y: jitter-gap mass normalized by the Markov-1 collapse mass at the same top-k and lag

Expected reading:

- `E`: origin or near-origin reference
- `B`: moderate/low net deficit with high jitter gap
- `matched-LZ`: high net deficit with low jitter gap
- `markov1`: near-pure collapse, high net deficit and zero jitter gap

Scientific role:

This is the most defensible visual summary of the current Phase 2 claim: observed B does not merely collapse like the nulls; a substantial part of its apparent local loss is redistribution/jitter rather than true shell mass loss.

### 2. Shell Transport Heatmap

Candidate role: secondary paper figure or appendix figure.

Data source:

- `parent_shell_atlas/.../atlas.csv`
- `parent_shell_atlas/.../patterns.csv`
- `parent_shell_atlas_sensitivity/.../subtype_sensitivity.csv`

Rows:

- shell patterns or grouped shell subtypes

Columns:

- `E`, `B`, `markov1`, and `matched-LZ` seeds

Possible color fields:

- partition mass retention
- net deficit mass
- gross loss pressure
- jitter gap mass
- survivor-internal deformation

Scientific role:

This figure should show where the shell transport changes pattern-by-pattern, and whether the null failures are diffuse, concentrated, or structurally different from observed B.

### 3. Optional Hilbert/Gray Binary Field

Candidate role: exploratory supplement only, unless later justified by robustness.

Data source:

- pattern-level rows from `atlas.csv` or `patterns.csv`
- binary pattern codes at the monitored scale

Possible layout:

- deterministic Hilbert ordering
- deterministic Gray-code ordering
- another fixed binary-space ordering with no tuned layout parameters

Required caption discipline:

> This is a deterministic visual ordering of binary patterns, not an assumed physical geometry.

Scientific role:

Use this only as a visual lens over the binary pattern field. It may reveal texture in the distribution of shell deficit or jitter, but it must not be treated as evidence for an underlying Hilbert geometry.

### 4. Child-Routing Regime Map

Candidate role: internal diagnostic figure, possible appendix candidate after Phase 2 is editorially organized.

Implemented entry point:

- `hsi_v2_phase2_child_routing_regime_map.py`

Data source:

- `child_routing_persistence_probe_map/.../band_persistence.csv`
- `child_routing_persistence_probe_map/.../source_persistence.csv`
- `child_routing_persistence_boundary705_wide/.../band_persistence.csv`
- `child_routing_persistence_boundary705_wide/.../source_persistence.csv`
- `child_routing_persistence_probe_extension/.../band_persistence.csv`
- `child_routing_persistence_probe_extension/.../source_persistence.csv`

Core axes:

- X: adjacent offset bands
- Y: corrected child-destination retention

Visual elements:

- observed `B` as the primary point
- `matched-LZ` seed envelope as a vertical range
- `matched-LZ` seed average as a secondary point
- explicit lag labels, including negative-lag and zero-lag phases

Scientific role:

This figure summarizes the N2-09 / N2-09b / N2-09c regime map:

- partial / liminal conservation at `696M-705M`
- negative-lag strong conservation at `705M-714M` after wide boundary probing
- strong positive phase at `714M-723M`
- phase-shifted strong positive phase at `723M-732M`
- same-window / zero-lag coincident phase at `732M-741M`

Caption discipline:

> This is a regime map of the monitored `top = 128` child-destination readout. It shows lag-local conservation and boundary structure, not a globally stationary transport law.

## Later Visual Families

These are potentially useful after the two core figures above:

- Markov-tower transport graph: bipartite parent-child flow from `m=40` to `m=48`.
- Multidimensional embedding: PCA/MDS over derived readout vectors only, never over arbitrary layout choices.
- Persistence barcode: survival/death of shell patterns across top-k, lag, source, and variant.

## Promotion Criteria

A visual readout becomes paper-safe only if:

- it is generated by a deterministic script
- it is reproducible from stored artifacts
- it has a manifest or enough metadata to reconstruct inputs
- its caption states what is data-derived and what is only a visual ordering
- the main interpretation remains stable under N2-06 sensitivity checks
- it does not replace the numerical tables; it summarizes them
