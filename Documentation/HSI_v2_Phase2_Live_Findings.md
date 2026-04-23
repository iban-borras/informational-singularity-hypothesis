# HSI v2 Phase 2 Live Findings

## Status

**Document type:** living findings log  
**Language:** English  
**Scope:** HSI v2 Phase 2 temporal transport prototypes, strict null-pressure readout, strict `N2-01`, strict `N2-02`, derived survival readouts, and post-D-0037 / D-0039 / D-0047 interpretation boundaries  
**Last updated:** 2026-04-22  
**Current status:** strict `null_pressure` remains the first paper-safe observational opening of Gate 2; the parent-survival line has now been revalidated under an independent shell-lag probe plus lag-aware band and should be read as delayed shell realignment with residual local mismatch, not as a stable death shell

This document records the live scientific reading of the current HSI v2 Phase 2 outputs stored in the repository.

It is not a paper draft and not a decision log.

Its role is narrower:

- preserve the current scientific readout in a repo-facing form
- help experimental recovery from stored artifacts
- separate what already looks defensible from what is still only diagnostic

Canonical methodological decisions still live in:

- `../docs/HSI_v2_Decision_Log.md`

---

## Why Phase 2 Exists

Phase 1 established that a non-trivial projective hierarchy survives in the HSI family and that not all persistent towers are equivalent.

Phase 2 starts only after that gate.

Its job is not yet to build full geometry. Its job is to ask a stricter question:

> do strong nulls reproduce the temporal organization of the observed hierarchy, or only some local shadow of it?

The current active Phase 2 line focuses on return-profile and transport objects anchored to observed `B`.

---

## Current Phase 2 Scope

The current repo-facing Phase 2 readout should be understood in four layers:

1. **Exploratory pilots**
   Early graph, cocycle, defect, and return-lag objects that helped map the search space.

2. **Strict null-pressure**
   Observed-anchored return-profile comparison under the corrected contract:
   - fixed pattern list from observed `B`
   - strip-global return-lag evaluation
   - multi-window readout
   - strong-null seed envelope

3. **Strict `N2-01`**
   Endogenous transport-defect pilot over the deep band `m = 40 -> 48`, again anchored to observed `B`.

4. **Strict `N2-02`**
   Counterfactual transport-defect pilot with:
   - observed low/high support frozen from `B`
   - observed transport kernel frozen from `B`
   - candidate profiles transported through that fixed law

At the moment, only the strict objects should be treated as scientifically current.

---

## Main Findings

### 1. D-0037 and D-0039 imposed a revalidation boundary, and the strict `null_pressure` opening survived it

The shared `return_lag` layer now resolves observed-anchored overrides against:

1. `config.variant`
2. `config.source_variant`
3. the undecorated base variant prefix

This matters because null-surrogate runs may be stored as decorated names such as:

- `B-markov1`
- `B-matched-lz`

So any strict Phase 2 null comparison that relied on observed-anchored `return_lag` selection should now be treated as pending explicit revalidation under D-0037.

D-0039 then tightens the sweep contract further:

- source realizations must be frozen once at the full sweep-global required length and only sliced across offsets
- boundary-spanning pattern starts must be preserved with an `m-1` overlap across segments
- `null_pressure` summaries must show both the strongest opening and the closest-imitation envelope

Practical consequence:

- the older strict `null_pressure` opening had to be rerun under the corrected contract
- that rerun has now been completed successfully
- so the `null_pressure` line is no longer merely "best candidate"
- it is again the current paper-safe Gate 2 opening

### 2. Strict `N2-01` remains informative, but not decisive

The strict-band `N2-01` transport defect is methodologically valid, but it does not yet separate `matched-lz`.

Compact readout:

| Source | N windows | `d_JS` avg | `d_flow` avg | `d_mean` avg |
| --- | ---: | ---: | ---: | ---: |
| observed calibration (`B/E` smoke) | 2 | 0.0000 | 0.0000 | 0.0000 |
| `markov1` | 4 | 0.3936 | 3.2383 | 0.8239 |
| `matched-lz` | 24 | 0.0018 | 0.0248 | 0.0123 |

Interpretation:

- `markov1` is decisively outside the observed baseline
- `matched-lz` remains very close to the observed zero-defect regime

So the current strict `N2-01` object acts as:

- a weak-null calibration
- an endogenous self-consistency probe

It does **not** yet act as the first decisive strong-null transport defect.

### 3. Strict `N2-02` preserves `B/E` exactly under the frozen observed law

The full strict `N2-02` run with anchor `B`, scales `40 -> 48`, selector `bridge-linked`, `top = 64`, and offsets `714M, 717M, 720M, 723M` preserves the observed pair perfectly:

| Source | Offsets | Coverage avg | `d_JS` avg | `d_flow` avg | `d_mean` avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B` observed | 4 | 0.9607 | 0.0000 | 0.0000 | 0.0000 |
| `E` observed | 4 | 0.9833 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

- freezing the observed transport law of `B` does **not** break the `B/E` family
- this is strong internal evidence that the counterfactual object is not hallucinating mismatch where the family should remain coherent
- it also reinforces the older v1/v2 structural intuition that `E` remains inside the same effective family as `B`

### 4. Strict `N2-02` rejects `markov1` strongly

Under the same fixed observed law:

| Source | Offsets | Coverage avg | `d_JS` avg | `d_flow` avg | `d_mean` avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B-markov1` | 4 | 0.1821 | 0.4376 | 1.6043 | 0.6970 |

Interpretation:

- `markov1` fails both structurally and dynamically
- it covers only a small fraction of the observed child support
- and where it does cover, it still departs strongly from the observed counterfactual transport fit

So `N2-02` is a genuine strong weak-null rejection, not merely a cosmetic rewrite of `N2-01`.

### 5. Strict `N2-02` still does not materially open `matched-lz`

This is the central scientific result of the completed `N2-02` batch.

Across 24 `matched-lz` rows (`6 seeds x 4 offsets`):

| Source | Rows | Coverage avg | `d_JS` avg | `d_wJ` avg | `d_flow` avg | `d_mean` avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B-matched-lz` envelope | 24 | 0.9944 | 0.0045 | 0.0224 | 0.0266 | 0.0124 |

Envelope maxima:

- `d_JS max = 0.0123`
- `d_wJ max = 0.0637`
- `d_flow max = 0.0731`
- `d_mean max = 0.0495`

Interpretation:

- `matched-lz` remains much farther than observed zero only in a very weak sense
- the absolute defect scale is still too small to support a formal Gate 2 opening
- support coverage remains almost perfect, so the fixed observed law does not force any broad support collapse in the strong null

This means the counterfactual move was mathematically the right question, but this specific law-level object is still not the one that breaks the hardest null.

### 6. `terminal_fraction` is not a primary Gate 2 discriminator here

In the current strict `N2-02` run:

- `B` sits at `0.3281`
- `E` sits at `0.3281`
- `markov1` sits at `0.3281`
- `matched-lz` also sits at `0.3281`

So `terminal_fraction` is completely neutralized in this counterfactual object.

It should not be used as a primary Gate 2 axis here.

### 7. The current honest Phase 2 position is now sharper than it looked before the D-0039 rerun

Before D-0037 and D-0039, it was tempting to phrase the state as:

- strict `null_pressure` gives the first defensible opening
- `N2-01` fails only because it is endogenous

The current stricter reading is now:

- the strict `null_pressure` result has now survived revalidation under D-0037 and D-0039
- it remains the first paper-safe observational opening of Gate 2
- `N2-01` is valid but not decisive
- `N2-02` is also valid, preserves `B/E`, rejects `markov1`, but still does **not** materially open `matched-lz`

So the current picture is two-layered:

- `null_pressure` is the first paper-safe Gate 2 opening
- no transport-defect object has yet superseded it as the stronger formal object

### 8. The first same-window parent-survival band was diagnostically useful, but scientifically too strong

The hostile audit on the fine-band parent-survival readout landed on a real point.

The same-window band:

- did identify a sharp late-band asymmetry under anchor `E`
- but it could not distinguish absolute parent death from delayed reappearance outside the aligned window

So the old shorthand reading:

- "`23`-pattern dead shell"

must now be downgraded.

What survives from that first band is only the diagnostic clue:

- there is a short late-band region where `E` and `B` fall badly out of synchrony under the anchored support contract

What does **not** survive is the literal extinction reading.

### 9. The canonical parent-shell revalidation replaces the death-shell claim by delayed shell realignment with residual local mismatch

The new canonical pipeline is:

1. independent shell-lag probe on the anchor-defined low-scale shell
2. lag-aware parent-survival band using the externally estimated lag

Canonical result for anchor `E`, candidate `B`, scales `40 -> 48`, selector `bridge-linked`, `top = 64`, and the full `714M -> 723M` half-megabit band:

- anchor core size = `20`
- anchor shell size = `44`
- first full candidate shell offset = `729M`
- recommended lag = `+15M` bits

After applying that external lag, the lag-aware band reads:

- `B`: exact range `8 -> 43`, dead range `0 -> 23`, dark range `0`, partial range `0`
- `E`: exact range `20 -> 43`, dead range `0`, dark range `0`, partial range `0`

Interpretation:

- the earlier "stable death shell" wording is no longer scientifically acceptable
- the dominant phenomenon is delayed shell realignment
- but the lag is **not** a perfectly rigid translation, because the lag-aware band still shows local residual mismatch
- so the honest summary is:
  - `E -> B` exhibits delayed shell realignment with residual local mismatch

This is scientifically better than the earlier, cleaner story because it keeps the temporal objection open where the data demand it.

### 10. The parent-density residual readout is now corrected for survivor bias

The hostile audit of the first density readout found a real conditioning problem.

The old readout filtered to `exact` rows before computing mass retention. That was legitimate only as a survivor-conditioned quantity, not as full partition retention.

The corrected readout now separates two objects:

- full partition mass retention, computed over all active anchor rows including `dead` and `dark`
- survivor-internal density deformation, computed only where the candidate has positive support and `log2` is defined

The grouped summary now reports pooled ratio-of-sums as the primary value, not an unweighted mean of window ratios.

Corrected canonical rerun for anchor `E`, candidate `B`, lag `+15M`, scales `40 -> 48`, selector `bridge-linked`, `top = 64`, and the `714M -> 723M` half-megabit band:

| Source | Partition | Part ret | Surv ret | Surv def | Surv bias | Dead m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `E` observed | Shell | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `B` observed | Shell | 0.9375 | 1.0227 | 0.8630 | -0.3669 | 0.0833 |
| `B-markov1` | Shell | 0.0000 | - | - | - | 0.5607 |
| `B-matched-lz` envelope | Shell | 0.1882-0.4219 | 0.2118-0.4219 | 1.6775-2.6091 | -2.6072 to -1.6194 | 0.0000-0.1114 |

Interpretation:

- `B` no longer looks like an artificial shell amplifier once dead mass is counted
- `B` still retains far more shell mass than `matched-lz`
- `matched-lz` preserves broad support but its shell density is weak and strongly negatively biased
- `markov1` remains a collapse control in this shell-density object

This keeps the density line scientifically useful, but still as an internal diagnostic below the strict `null_pressure` opening in evidentiary priority.

---

## Current Repo-Facing Scientific Position

What HSI v2 currently supports in Phase 2:

- the strict `null_pressure` line is revalidated under the strongest current contract
- `matched-lz` still fails to preserve the observed return-profile geometry of `B`
- the strongest average opening sits at `717M`, with attenuation toward `726M`
- `B/E` remain coherent under both endogenous and counterfactual transport tests
- `markov1` is strongly rejected by the strict transport-defect family
- `matched-lz` remains the strongest and most informative null
- the counterfactual move to a frozen observed law was the correct methodological escalation
- the parent-survival line now supports delayed shell realignment plus local residual mismatch in `E -> B`
- the corrected parent-density residual line separates full mass retention from survivor-internal deformation and avoids survivor bias

What HSI v2 does **not** yet support in Phase 2:

- a paper-final Gate 2 claim based on `N2-01`
- a paper-final Gate 2 claim based on `N2-02`
- a paper-final claim of a stable `E -> B` death shell
- the claim that `E -> B` is explained by a single rigid lag with no residual local distortion
- the claim that the frozen observed law `K_B` materially separates `matched-lz`
- a final transport law
- a completed defect theory
- a geometry claim derived from Phase 2
- a paper-final claim based only on the parent-density residual readout

---

## Key Artifacts To Read First

### Older strict null-pressure opening candidate

- `results/hsi_v2/phase2/window_sweep_deep/null_pressure_strict/phase2-return-lag-null-pressure__obs-B__m-48__sel-bridge-linked__top-64__w-5__off-714M__step-3M__20260419T102749/`

### Strict `N2-01` observed calibration smoke

- `results/hsi_v2/phase2/transport_defect_strict_smoke/phase2-transport-defect-strict__stage-observed__m-40-48__sel-bridge-linked__top-64__off-717M__var-B-E__20260420T095713/`

### Strict `N2-01` weak-null comparison

- `results/hsi_v2/phase2/transport_defect_strict_B_markov1/phase2-transport-defect-strict__stage-nulls__m-40-48__sel-bridge-linked__top-64__off-714M-plus-4__var-B__20260420T193214/`

### Strict `N2-01` strong-null seed envelope

- `results/hsi_v2/phase2/transport_defect_strict_B_matchedlz/phase2-transport-defect-strict__stage-nulls__m-40-48__sel-bridge-linked__top-64__off-714M-plus-4__var-B__20260420T212430/`

### Strict `N2-02` complete counterfactual batch

- `results/hsi_v2/phase2/transport_defect_counterfactual/phase2-transport-defect-counterfactual__stage-all__anchor-B__m-40-48__sel-bridge-linked__top-64__off-714M-plus-4__var-B-E__20260421T012228/`

### Canonical parent-shell revalidation

- `results/hsi_v2/phase2/parent_survival_revalidation/phase2-parent-survival-revalidation__anchor-E__cand-B__m-40-48__sel-bridge-linked__top-64__off-714M-plus-19__20260422T202310/`

### Corrected parent-density residual readout

- `results/hsi_v2/phase2/parent_density_residual_corrected/phase2-parent-density-residual__anchor-E__cand-B__lag-15M__m-40-48__sel-bridge-linked__top-64__off-714M-plus-19__20260423T091934/`

---

## Reading Order For Recovery

If you want the current scientific state quickly, read:

1. `Documentation/HSI_v2_Phase1_Live_Findings.md`
2. `Documentation/HSI_v2_Phase2_Live_Findings.md`
3. `../docs/HSI_v2_Decision_Log.md`

If you want the current counterfactual result specifically, read:

1. the strict `N2-02` report under `results/hsi_v2/phase2/transport_defect_counterfactual/...`
2. the matching `summary.json`
3. `Documentation/HSI_v2_Phase2_Live_Findings.md`
4. `../docs/HSI_v2_Decision_Log.md`

If you want to know what still counts as the best Gate 2 candidate, read:

1. the strict `null_pressure` report under `results/hsi_v2/phase2/window_sweep_deep/null_pressure_strict/...`
2. D-0037 in `../docs/HSI_v2_Decision_Log.md`
3. this live-findings document

If you want the current survival-line reading specifically, read:

1. the canonical parent-shell revalidation report under `results/hsi_v2/phase2/parent_survival_revalidation/...`
2. the nested shell-lag probe summary
3. the nested lag-aware band summary
4. D-0047 in `../docs/HSI_v2_Decision_Log.md`

If you want the current parent-density reading specifically, read:

1. the corrected parent-density residual report under `results/hsi_v2/phase2/parent_density_residual_corrected/...`
2. the matching `summary.json`
3. D-0048 in `../docs/HSI_v2_Decision_Log.md`

---

## Bottom Line

Phase 2 is still scientifically alive, but the current honest reading is stricter than before.

The full strict `N2-02` batch confirms that:

- `B/E` remain coherent under the frozen observed transport law
- `markov1` fails strongly
- `matched-lz` still remains too close

The canonical parent-survival revalidation now adds:

- the old same-window "death shell" wording was too strong
- the anchor-defined shell reappears in `B` after an external lag of about `+15M` bits
- after lag correction, the main phenomenon is delayed realignment, not literal extinction
- but the realignment is not perfectly rigid and still leaves local residual mismatch
- the corrected density readout shows that `B` retains shell mass much more strongly than `matched-lz`, without hiding dead rows

So the next serious move is not to overstate a transport-defect victory.

It is to:

- revalidate the older strict `null_pressure` opening under D-0037
- preserve `N2-02` as a valuable negative result
- preserve the survival line as a calibrated asymmetry diagnostic, not as a solved shell law
- and design the next law-level object only after that boundary is clean
