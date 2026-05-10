# HSI v2 Phase 5 Candidate B-mini Preregistration v0.2 DRAFT

Status: draft for Sofia/Iban/Grace review; not executable until accepted, SHA256-anchored, and signed.

Supersedes: `HSI_v2_Phase5_CandidateB_CrossArc_Preregistration_v0_1.md` if accepted.

Purpose: incorporate Grace's v0.1 audit warning that a narrow P4 lag-grid would assume translational invariance while preserving Sofia's requirement that B-mini still test the literal Phase 4 object. This v0.2 draft therefore uses a dual readout:

1. a strict P4-object check on the fixed P4 grid;
2. a local analogue check using an arc-local broad lag scout followed by a post-scout bracket.

## 1. Audit Inputs

Grace audit: `docs/HSI-audit-Phase5-CandidateB-v01.md`

Grace v0.2 audit: `docs/HSI-audit-Phase5-CandidateB-v02.md`

Accepted points:

- Candidate A remains deferred.
- Stage 0 observability-budget audit is methodologically valid.
- Failure remains a valid result.

Required correction:

- Do not reuse the narrow P4 lag-grid as the only grid for distant arcs.
- Add an arc-local broad lag scout before testing each horizon bracket.
- Use scout spacing fine enough to sample a Phase-4-width channel. The v0.2 audit rejected `9M` spacing as guaranteed undersampling; this draft uses `2.5M`.

## 2. Stage 0 Budget Result

Read-only budget script:

`hsi_v2_phase5_cross_arc_observability_budget.py`

Five-band rerun used for this v0.2 draft:

`results/hsi_v2/phase5_cross_arc_horizon/observability_budget/phase5-cross-arc-observability-budget__20260508T151357/`

Budget result:

- candidate arcs: `39`
- executable candidates: `34`
- greedy non-overlapping executable arcs: `12`
- verdict: `budget-supports-b-mini-3-arc-freeze`
- execution authorized by Stage 0: `false`

The five-band geometry is required because each arc now contains:

1. one scout band;
2. four post-scout horizon-bracket bands.

## 3. Proposed Frozen B-mini Arcs

The following three arcs are proposed for freezing, subject to Sofia/Iban/Grace acceptance.

| arc_id | scout_start | scout_band | bracket_bands | rationale |
|---|---:|---|---|---|
| `ARC-A` | `300000000` | `300M-309M` | `309M-318M`; `318M-327M`; `327M-336M`; `336M-345M` | selected by the mechanical spread rule below |
| `ARC-B` | `597000000` | `597M-606M` | `606M-615M`; `615M-624M`; `624M-633M`; `633M-642M` | selected by the mechanical spread rule below |
| `ARC-C` | `1110000000` | `1110M-1119M` | `1119M-1128M`; `1128M-1137M`; `1137M-1146M`; `1146M-1155M` | selected by the mechanical spread rule below |

Selection rule:

- start from the five-band Stage 0 `candidate_arcs.csv`;
- keep only rows with `executable = Y`;
- drop rows with any `overlaps_reference_ranges` value other than `-`;
- require `available_margin_bits >= 250000000`;
- choose `K=3` arcs that maximize the minimum pairwise distance between `scout_start` values;
- tie-breaker 1: maximize total span (`max(start) - min(start)`);
- tie-breaker 2: choose the lexicographically earliest start tuple.

Applying this rule to `phase5-cross-arc-observability-budget__20260508T151357/candidate_arcs.csv` yields exactly:

`300000000,597000000,1110000000`.

Eligible arc starts under the rule:

`300000000,327000000,354000000,381000000,408000000,435000000,462000000,489000000,516000000,543000000,570000000,597000000,840000000,867000000,894000000,921000000,948000000,975000000,1002000000,1029000000,1056000000,1083000000,1110000000`.

The selected triple has:

- minimum pairwise distance: `297000000`;
- total span: `810000000`;
- lexicographic start tuple: `(300000000,597000000,1110000000)`.

No arc may be replaced after acceptance.

## 4. Strict P4-Object Check

Each frozen arc is first evaluated as a strict Phase 4 object using the fixed P4-04/P4-05 lag grid:

`-29500000,-27000000,-26500000,-24000000,24500000,26500000`

This readout is explicitly marked as a non-trivial dependency on Phase 4. It asks only:

`Does the same lag object tested in Phase 4 have a finite-horizon analogue in this frozen arc?`

The strict P4-object check is run over all five frozen bands for the arc. It does not use the scout grid.

The B-mini continue/stop outcome rules in section 10 are based exclusively on this strict P4-object check.

## 5. Broad Scout Lag Grid

Each arc also has a single-band broad lag scout on the scout band declared in section 3.

Scout grid:

`-60000000,-57500000,-55000000,-52500000,-50000000,-47500000,-45000000,-42500000,-40000000,-37500000,-35000000,-32500000,-30000000,-27500000,-25000000,-22500000,-20000000,-17500000,-15000000,-12500000,-10000000,-7500000,-5000000,-2500000,0,2500000,5000000,7500000,10000000,12500000,15000000,17500000,20000000,22500000,25000000,27500000,30000000,32500000,35000000,37500000,40000000,42500000,45000000,47500000,50000000,52500000,55000000,57500000,60000000`

Rationale:

- broad enough to avoid assuming perfect translational invariance;
- includes the known P4 neighborhood only as one point in a wider grid;
- symmetric enough to detect positive or negative active cores;
- fixed `2.5M` resolution across the whole range;
- fine enough to sample a channel with Phase-4-like width of roughly `5.5M`;
- fixed before execution.

The broad scout is diagnostic. It cannot by itself support moving to full Candidate B.

## 6. Scout Decision Rule

The scout is used only to choose a local lag core or to classify the arc as inactive.

For each arc:

1. Run the scout band over the full scout grid.
2. Consider negative and positive channels separately.
3. For the negative channel, mark a scout lag as eligible if:
   - `B_retention >= 0.70`
   - `hard_margin >= 0.10`
4. Partition eligible negative lags into connected clusters under the fixed `2.5M` scout-grid adjacency.
5. A cluster is active only if it contains at least `3` eligible lags.
6. If multiple active clusters exist, choose the cluster with:
   - largest number of eligible lags;
   - then highest mean hard margin;
   - then smallest absolute distance between its median lag and `-27000000`;
   - then most negative median lag.
7. Within the selected cluster, choose the consecutive three-lag subcluster with the highest sum of hard margins. This is the local lag core.
8. If no active cluster exists, the arc is `negative-scout-inactive`.

If the P4-anchored tie-break (`-27000000`) is used to choose between equally eligible clusters, `report.md` must state this explicitly for the affected arc. This tie-break is a reporting obligation, not a hidden selection degree of freedom.

The positive channel is reported as a diagnostic, not as the primary object.

## 7. Local Bracket Lag Core

For a `negative-scout-active` arc, define the bracket lag core mechanically:

- use the three-lag local core selected by section 6;
- do not add any lag outside that core;
- do not change the core after seeing post-scout bands.

No hand-picked lag may be added after the scout.

## 8. Local Analogue Horizon Bracket Test

For each `negative-scout-active` arc, run the four post-scout bands using only the frozen local bracket lag core.

Arc-level local-analogue finite-horizon-like result requires:

1. at least one early post-scout bracket band has a strong negative-channel cell;
2. a later post-scout bracket band shows hard-null envelope catch-up;
3. the mechanism is reported as null rise, B collapse, or mixed.

Thresholds:

- strong cell: `B_retention >= 0.90` and `hard_margin >= 0.30`;
- null-shadow boundary: `hard_margin <= 0.05` with `B_retention >= 0.50`, or a separately reported B-collapse case;
- no extra band may be added after seeing results.

This local-analogue readout is diagnostic. It may open a future question about lag drift across arcs, but it cannot by itself authorize moving to full Candidate B.

## 9. Arc Result Categories

Each arc is classified into one of the following categories:

- `strict-replication`: strict P4-object check satisfies the finite-horizon-like criterion.
- `local-analogue-only`: strict P4-object check fails, but the local scout/bracket readout satisfies the finite-horizon-like criterion.
- `no-replication`: neither strict nor local-analogue readout satisfies the finite-horizon-like criterion.
- `scout-inactive`: the broad scout finds no active negative lag cluster; this is a valid non-replicating outcome.

Only `strict-replication` counts toward the B-mini continue/stop decision.

## 10. B-mini Outcome Rules

For three frozen arcs:

- `>=2/3` `strict-replication` arcs: B-mini supports drafting a full Candidate B contract.
- `1/3`: ambiguous; stop and review before any full Candidate B.
- `0/3`: negative B-mini; do not run full Candidate B without a new rationale.

Inactive scout arcs count as non-replicating arcs, not as missing data.

If `0/3` strict arcs but `>=2/3` local-analogue arcs occur, this does not support full Candidate B. It may justify a new, separately pre-registered question about phase-channel lag drift across arcs.

## 11. Null Envelope

Use the hard-null envelope available in the Phase 4 predictive stack:

- `markov1`
- `phase-matched-lz`
- `block-entropy`

Seeds:

`607,709,811,907,1009,1103`

The report must state whether any null family was generated during execution or read from cache.

## 12. Contamination Guards

- Stage 0 reads only metadata, source lengths, and geometric executability.
- Stage 0 does not read retention, hard-margin, or null-envelope performance.
- Scout bands may select local lag cores, but they cannot be counted as post-scout horizon-bracket evidence.
- Local-analogue success cannot rescue a strict P4-object failure for B-mini outcome purposes.
- Arc list is frozen before scout execution.
- Failed, inactive, and ambiguous arcs remain in the report.

## 13. Required Artifacts

Execution artifacts, if this draft is accepted:

- `summary.json`
- `report.md`
- `strict_p4_results.csv`
- `scout_results.csv`
- `bracket_results.csv`
- `arc_verdicts.csv`
- `source_response.csv`
- `manifest.json`

Output root:

`hsi_agents_project/results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/`

## 14. Acceptance Before Execution

This draft is not executable as-is.

Before execution:

1. Sofia/Iban must accept or revise the three proposed arcs.
2. Grace must accept the broad scout + post-scout bracket protocol.
3. The final contract must be SHA256-anchored.
4. A Decision Log entry must record the accepted contract.
5. No script may run the scout until the accepted contract exists.
