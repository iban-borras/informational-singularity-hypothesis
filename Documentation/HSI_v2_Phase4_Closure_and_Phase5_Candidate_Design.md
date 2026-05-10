# HSI v2 Phase 4 Closure and Phase 5 Candidate Design

Status: working design note, not a manuscript patch and not an experimental contract.

Purpose: summarize the current Phase 4 endpoint and identify which next scientific question is different enough to justify a new phase rather than continuing the closed forward-forecast subarc.

## Phase 4 Predictive Subarc Closure

The current Phase 4 predictive subarc is closed by D-0095, D-0097, and D-0102.

### What Phase 4 Established

1. P4-04 converted the retrospective phase-channel map into a pre-registered forward forecast.
2. The forecast was supported over three untouched forward bands:
   - `768M-777M`
   - `777M-786M`
   - `786M-795M`
3. The negative channel remained strong in `3/3` bands under the hard null envelope.
4. The positive channel remained non-universal.

This is the positive part of the result: HSI produced a finite out-of-sample predictive horizon.

### What Phase 4 Bounded

P4-05 tested the next four bands:

- `795M-804M`
- `804M-813M`
- `813M-822M`
- `822M-831M`

The negative horizon-bracket forecast was rejected at `0/4` strong bands. This does not refute P4-04. It bounds it.

The correct reading is finite-horizon predictive success followed by null-envelope convergence, not global phase persistence.

### What Phase 4 Diagnosed

P4-06 compared the last supported band (`786M-795M`) with the first boundary band (`795M-804M`) and found:

- B retention does not disappear at the boundary.
- The predictive margin collapses because the hard-null envelope rises sharply.
- The rise is dominated by phase-matched-LZ, with a substantial block-entropy contribution.

Best-row shift:

| Metric | `786M-795M` | `795M-804M` | Delta |
|---|---:|---:|---:|
| B retention | `1.0000` | `1.0000` | `+0.0000` |
| phase-matched-LZ max | `0.2953` | `1.0000` | `+0.7047` |
| block-entropy max | `0.3789` | `0.8824` | `+0.5034` |
| hard margin | `0.6211` | `0.0000` | `-0.6211` |

Aggregate shift across four aligned negative lags:

- mean B-retention shift: `+0.1313`
- mean phase-matched-LZ max shift: `+0.6745`
- mean block-entropy max shift: `+0.5208`
- mean hard-margin shift: `-0.4834`

## Hard Stop

Do not continue this subarc by scanning later bands for renewed signal.

D-0097 and D-0102 explicitly close that route. A later-band scan would be signal chasing unless it is part of a new, independently pre-registered question with a different object.

## Phase 5 Candidate Questions

Phase 5 should not mean "more Phase 4 bands." It should ask a new question.

### Candidate A: Finite-Horizon Boundary Law

Question: can we predict the null-envelope catch-up boundary before running the full lag-response readout?

Possible object:

- estimate a boundary risk score from pre-boundary observables;
- forecast whether the next band will remain separated or be null-shadowed;
- evaluate on non-overlapping future or held-out arcs.

Why this is different from P4:

- P4 asked whether a channel persists.
- Candidate A asks whether the loss of differential can itself be predicted.

Risk:

- easy to overfit because the boundary was discovered retrospectively.

Minimum discipline:

- pre-register features;
- freeze training arcs and test arcs;
- no per-band tuning;
- measure against hard-null envelope, not only B retention.

### Candidate B: Cross-Arc Scaling of Predictive Horizons

Question: do finite predictive horizons appear at comparable relative positions across independent arcs, or was the P4 horizon a local accident?

Possible object:

- choose multiple anchor arcs before execution;
- for each arc, run a compact P4-style forward forecast;
- measure supported horizon length before null-envelope catch-up.

Why this is different from P4:

- P4 tested one local horizon.
- Candidate B tests whether horizon length is a repeatable phenomenon across arcs.

Risk:

- computationally expensive;
- requires very careful pre-registration of arcs to avoid cherry-picking.

Minimum discipline:

- arc list frozen before execution;
- no replacing failed arcs;
- hard-null envelope unchanged;
- report failures as failures, not as boundary discoveries unless the contract allows it.

### Candidate C: Null-Envelope Mechanism Decomposition

Question: what property lets phase-matched-LZ and block-entropy catch up at the boundary?

Possible object:

- compare pre-boundary and boundary bands using null-family internal features;
- look for which null feature predicts envelope opacity;
- avoid new forecast claims until the mechanism is stable.

Why this is different from P4:

- P4 used nulls as controls.
- Candidate C makes null opacity itself the object of study.

Risk:

- may drift into post-hoc explanation unless tightly constrained.

Minimum discipline:

- use P4-06 as the only seed observation;
- pre-register a small fixed feature set;
- test on another boundary or held-out arc.

## Post-Audit Update

Grace's independent review and Sofia's editorial response supersede the initial recommendation below.

Decision after audit: Candidate B is mandatory before Candidate A can become legitimate.

Reason:

- Candidate A tries to infer a boundary rule from a single observed boundary (`795M-804M`), which is pre-empirical for any law-like claim.
- Candidate B asks first whether finite horizons recur across independent arcs.
- If Candidate B yields at least several independently documented horizons, Candidate A can later become a Phase 6 question with a real empirical base.

Recommended Phase 5 form: Candidate B-mini.

Minimal first pass:

1. Audit the remaining observability budget before freezing any arcs.
2. Freeze a small pilot list of `2-3` independent arcs.
3. For each arc, run a compact P4-style finite-horizon test.
4. Report failed arcs with the same visibility as successful arcs.
5. Use the pilot only to decide whether a full Candidate B run is warranted.

Candidate A is now explicitly deferred. It should not be reopened until Candidate B has produced a plural set of horizon observations or a documented negative result.

## Paper Boundary

This note does not authorize any manuscript edit.

The current manuscript can safely report:

- finite forward predictive success;
- bounded horizon;
- null-envelope catch-up at the boundary if Sofia chooses to integrate D-0102 wording.

It should not report:

- a boundary law;
- a universal horizon;
- a general scaling claim;
- any Phase 5 result before a new pre-registered experiment exists.
