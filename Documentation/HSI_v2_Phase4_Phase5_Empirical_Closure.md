# HSI v2 Phase 4/5 Empirical Closure

Status: repo-facing closure note, derived from accepted Decision Log entries through D-0114.

Purpose: compactly record what the Phase 4/5 predictive arc established, what it falsified, and what must not be reopened by inertia before Level 2 work begins.

## 1. Closure Summary

Phase 4/5 is closed as a finite-horizon predictive arc, not as a universal predictive law.

The accepted empirical reading is:

- a finite predictive channel exists in the tested regime;
- the channel has a boundary;
- the boundary is not a soft visual artifact;
- at the boundary, strong nulls catch up under surface retention;
- Q2 shows no hidden internal-efficiency advantage for observed `B` under local `component_weight_entropy_norm`;
- D-0113 shows the strongest hard null's boundary advantage is a local concentration advantage: fewer effective components and higher top-k component shares.

This closure strengthens the program because it constrains the phenomenon instead of expanding it rhetorically.

## 2. Empirical Chain

| Decision | Role | Reading |
|---|---|---|
| D-0095 / D-0097 | Phase 4 forward predictive subarc | finite forward predictive horizon; no universal persistence |
| D-0102 | P4-06 boundary feature shift | `null-envelope catch-up` is a real boundary, not a missed positive channel |
| D-0105 / D-0106 | Phase 5 B-mini | cross-arc replication is mixed; ARC-B replicates, ARC-A/ARC-C do not |
| D-0109 | Q1 projective-density bridge | local projective density does not simply explain predictive-channel activation |
| D-0112 | Q2 internal routing entropy | hard negative: `block-entropy` is internally lower-entropy at catch-up |
| D-0113 | Q2 boundary mechanism | boundary null advantage is local concentration into fewer effective components and higher top-k shares |
| D-0114 | manuscript integration | one bounded paragraph integrated into §15.6; no new experimental cycle authorized |

## 3. What Is Closed

- Do not search for another Phase 1 local metric to rescue Q1.
- Do not search for another Q2 entropy metric to rescue observed `B` at catch-up.
- Do not reinterpret B-mini as broad cross-arc confirmation.
- Do not claim a boundary law, scaling law, universal horizon, or Gate 2 closure.
- Do not add figures or tables for Q2 unless a later editorial decision explicitly authorizes them.

## 4. What Remains Open

The open question is no longer whether the Phase 4/5 channel can be extended by more empirical retuning.

The open question is Level 2:

> Is there a geometry-like invariant of the routing quotient shape that explains the transition between channel interior, boundary catch-up, and inactive regimes without using retention itself as the explanatory variable?

The next experiment should therefore test shape invariants, not more retention sweeps.

## 5. Working Rule For The Next Cycle

The next cycle must be preregistered and audited before execution.

It must not use:

- `B_retention`;
- `hard_margin`;
- `entropy_gap`;
- post-hoc selected lags;
- labels fitted after seeing shape readouts.

It may use those quantities only to define already frozen regime labels inherited from Phase 4/5.

## 6. Practical Status

Phase 4/5 is suitable for commit and freeze as a completed empirical arc.

Level 2 should begin from a new preregistration document, not from ad-hoc extension of existing Phase 4/5 scripts.

