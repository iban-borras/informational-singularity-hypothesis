# HSI v2 Phase 2 Live Findings

## Status

**Document type:** living findings log  
**Language:** English  
**Scope:** HSI v2 Phase 2 temporal transport prototypes, strict null-pressure readout, strict `N2-01`, strict `N2-02`, derived survival readouts, parent-density residuals, shell-atlas diagnostics, jitter-provenance diagnostics, child-destination routing diagnostics, child-routing persistence diagnostics, phase-lag atlas diagnostics, lag-response diagnostics, phase-channel width diagnostics, top-k sensitivity diagnostics, monitored-universe routing-signature diagnostics, phase-channel transition-graph diagnostics, and post-D-0037 / D-0039 / D-0047 / D-0048 / D-0049 / D-0050 / D-0051 / D-0052 / D-0053 / D-0054 / D-0055 / D-0056 / D-0057 / D-0058 / D-0059 / D-0060 / D-0061 / D-0062 interpretation boundaries  
**Last updated:** 2026-04-30  
**Current status:** strict `null_pressure` remains the first paper-safe observational opening of Gate 2; the parent-survival, parent-density, shell-atlas, jitter-provenance, child-destination routing, child-routing persistence, phase-lag atlas, lag-response, phase-channel width, top-k sensitivity, monitored routing-signature, and transition-graph lines now form a hardened internal diagnostic stack below it, with `B` read as shell-mass preservation, observable same-parent temporal compensation, monitored high-scale child-destination conservation, threshold-defined sampled phase-channel envelopes, adjacent-band phase-channel paths, and a `729M-738M` monitored null-shadowing boundary rather than observed-`B` collapse or `matched-lz`-like net shell collapse

This document records the live scientific reading of the current HSI v2 Phase 2 outputs stored in the repository.

It is not a paper draft and not a decision log.

Its role is narrower:

- preserve the current scientific readout in a repo-facing form
- help experimental recovery from stored artifacts
- separate what already looks defensible from what is still only diagnostic

Canonical methodological decisions still live in:

- `HSI_v2_Decision_Log.md`

---

## Why Phase 2 Exists

Phase 1 established that a non-trivial projective hierarchy survives in the HSI family and that not all persistent towers are equivalent.

Phase 2 starts only after that gate.

Its job is not yet to build full geometry. Its job is to ask a stricter question:

> do strong nulls reproduce the temporal organization of the observed hierarchy, or only some local shadow of it?

The current active Phase 2 line combines observed-anchored return-profile tests around `B` with parent-shell diagnostics across the delayed `E -> B` support contract.

---

## Current Phase 2 Scope

The current repo-facing Phase 2 readout should be understood in five layers:

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

5. **Derived parent-shell diagnostics**
   Lag-aware parent survival, corrected parent density, shell atlas, same-parent jitter provenance, child-destination routing, child-routing persistence, and sensitivity sweeps over `top-k` and lag.

At the moment, the strict objects remain the evidentiary top layer. The derived parent-shell diagnostics are scientifically current as internal readouts, not as replacements for the strict `null_pressure` opening.

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

### 11. The parent-density and shell-atlas lines now form a hardened internal readout below `null_pressure`

D-0048, D-0049, and D-0050 now give the parent-shell line a cleaner role.

They do not replace the strict `null_pressure` opening. The evidentiary hierarchy remains:

- `null_pressure` first, because it is the current paper-safe opening against the strongest null envelope
- parent-density and shell-atlas below it, because they diagnose where the shell mass goes once the delayed `E -> B` realignment has been established

Within that lower diagnostic layer, the picture is now much sharper.

The corrected parent-density readout shows that `B` preserves most of the anchor shell mass after lag correction, while `matched-lz` preserves broad support but loses much more shell mass. This is not merely a statement that the null "has less"; it is a statement about the mode of failure.

The shell atlas then splits that failure into two different objects:

- net shell deficit, where mass genuinely disappears from the monitored shell
- jitter-gap, where gross mismatch is mostly absorbed as temporal redistribution rather than net loss

This distinction is the important scientific content.

For observed `B`, the canonical `top = 64` atlas has low net deficit and high jitter-gap. The shell is damaged locally, but much of the mismatch looks like displaced or jittered transport rather than collapse.

For `matched-lz`, the pattern is reversed. The net deficit stays high and the jitter-gap remains low. The null can imitate local support, but it does not reproduce the same redistribution regime.

For `markov1`, the reading is different again: `jitter_gap = 0` because gross loss and net loss coincide. That makes `markov1` a positive control for the meaning of jitter-gap. The quantity is not measuring empty noise or decorative slack; it is measuring the part of gross mismatch that is not simple collapse.

D-0050 then hardens this reading under sensitivity.

The `top = 32` slice is now treated as a blind resolution control. At that resolution the monitored lateral shell is not actually visible, so its empty atlas is not evidence against the parent-shell readout. It tells us that the readout needs enough observed support to resolve the shell.

At `top = 64`, the canonical separation remains the cleanest expression of the diagnostic: `B` sits in the redistribution regime, while `matched-lz` remains net-deficit dominated.

At `top = 128`, the observed support anatomy changes and Bridge-both structure becomes dominant, but the qualitative separation still survives. That is useful because it says the readout is not a single lucky `top-k` artifact.

One boundary must stay explicit: labels such as Prefix-only, Suffix-only, and Bridge-both are observation-conditional labels inside the selected anchor support. They are not absolute topological classes of the underlying hierarchy.

So the stabilized reading is:

- `B` preserves shell mass unusually well and expresses much of the residual mismatch as temporal redistribution
- `matched-lz` keeps superficial support but loses shell mass in a net-deficit dominated way
- `markov1` collapses without jitter-gap, confirming what the jitter-gap axis means

This is a hardened internal readout below `null_pressure`, not a new independent claim that Gate 2 is solved.

### 12. N2-07 shows that the `B` jitter-gap is mostly same-parent temporal compensation

D-0052 adds one mechanistic refinement below the shell atlas.

The question is no longer only:

- how much shell mass is lost?

It is also:

- when a parent pattern loses mass in one local window, does the same parent pattern regain mass elsewhere in the analyzed band?

N2-07 answers this at the same-parent level. It does not prove cross-parent routing and it does not prove child-destination flow. It only measures observable compensation under exact parent-pattern identity.

The canonical N2-07 readout over the `top = 64`, lag `15M` shell atlas gives:

- observed `B`: `compensated_loss_share_of_gross = 0.8125`
- `markov1`: `0.0000`
- `matched-LZ`: low compensation across seeds, with net deficit dominating

N2-07b then repeats the readout across the useful N2-06 sensitivity neighborhood:

- `top = 64, 128`
- lag `14.5M, 15.0M, 15.5M`

The stabilized envelope is:

- observed `B`: stable `jitter-compensated`, `compensated_loss_share = 0.7778-0.8709`
- `markov1`: stable `uncompensated-deficit`, `compensated_loss_share = 0.0000`
- `matched-LZ`: net-deficit dominated, `net_deficit_share = 0.9074-1.0000`
- observed `E`: flat by construction of the anchor reference

This is exactly the prudent interpretation of `jitter_gap_mass`.

For `B`, most local shell loss is compensated elsewhere in the analyzed band under the same parent-pattern identity. For `matched-LZ`, local loss mostly remains net deficit. For `markov1`, loss is pure collapse.

The canonical wording boundary is important:

- allowed: observable same-parent temporal compensation
- not allowed: guaranteed identity of physical flow
- not allowed: child-destination routing
- not allowed: final transport law

So N2-07 strengthens the internal mechanism under the shell-atlas line, but it still remains below strict `null_pressure` in evidentiary hierarchy.

### 13. N2-08 shows monitored child-destination conservation after jitter compensation

D-0053 adds the next mechanistic refinement below N2-07.

N2-07 established that most of the `B` jitter-gap is observable same-parent temporal compensation. N2-08 asks a stricter question:

- when mass is temporally displaced, does it still land on the same high-scale child destinations?

The first N2-08 draft was too strict because it compared destination overlap window by window. Grace's hostile audit correctly identified that this penalized temporal jitter as if it were routing failure.

The corrected N2-08 object now pools mass by child destination across the full analyzed band before computing overlap. The older window-synchronous value is retained only as a jitter-control diagnostic.

Canonical corrected run:

- anchor `E`
- candidate `B`
- `top = 128`
- lag `15M`
- scales `m = 40 -> 48`
- selector `bridge-linked`

Corrected source-level readout:

| Source | Child retention | Sync retention | Monitored reroute | Class |
| --- | ---: | ---: | ---: | --- |
| observed `B` | 0.9604 | 0.7194 | 0.0000 | same-destination-conserved |
| `B-markov1` | 0.0000 | 0.0000 | - | destination-collapse |
| `B-matched-lz` envelope | 0.3107-0.4832 | 0.2874-0.4408 | 0.0000 | same-destination-deficit |
| observed `E` | 1.0000 | 1.0000 | 0.0000 | same-destination-conserved |

The correction is scientifically important because it does not simply inflate every source.

Observed `B` rises from a window-synchronous `0.7194` to a jitter-compensated `0.9604`. This means that most of the apparent routing loss was temporal displacement, not loss of destination channel.

`matched-LZ` rises only modestly and remains in a `31-48%` same-destination-deficit band. `markov1` remains destination collapse.

The current internal reading is therefore:

- `B` preserves the monitored high-scale child-destination channels after jitter compensation
- `matched-LZ` imitates local support but does not preserve those child destinations at comparable mass
- `markov1` provides the pure-collapse pole

This moves the derived diagnostic stack one level deeper:

- parent-density says `B` preserves much more shell mass than `matched-LZ`
- shell atlas says `B` expresses residual mismatch mainly as jitter-gap rather than net deficit
- N2-07 says much of that jitter-gap is same-parent temporal compensation
- N2-08 says that, after jitter compensation, the compensated `B` mass mostly remains on the same monitored child destinations

The boundary remains explicit.

N2-08 does not prove:

- cross-parent routing
- absolute routing over all possible children
- classification of mass leaving the monitored `top-k` child universe
- final Gate 2 resolution
- a complete transport law

The phrase `monitored reroute` is deliberately local. It only means candidate mass landing on tracked child destinations outside the anchor child set. Candidate mass that leaves the tracked child universe remains deficit/off-support in this object.

### 14. N2-09 shows child-routing persistence as lag-local and phase-sensitive

D-0054 revalidates N2-08 outside the original `714M-723M` band.

The key change is conceptual.

N2-08 showed that observed `B` preserves monitored high-scale child destinations after jitter compensation in the canonical band. N2-09 asks whether this remains true across neighboring temporal bands.

The first persistence sweep used:

- anchor `E`
- candidate `B`
- `top = 128`
- fixed lag `15M`
- bands `705M-714M`, `714M-723M`, and `723M-732M`

The fixed-lag readout was:

| Band | Probe lag | Fixed lag | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `705M-714M` | - | `15M` | `0.0000` | `0.0940-0.2351` | `0.0000` |
| `714M-723M` | `15M` | `15M` | `0.9604` | `0.3107-0.4832` | `0.0000` |
| `723M-732M` | `5.5M` | `15M` | `0.7604` | `0.2015-0.3505` | `0.0000` |

This could have been read as attenuation in the later band, but the shell-lag probe warned us that the correct local lag there was not `15M`; it was `5.5M`.

The lag-local rerun for `723M-732M` gives:

| Band | Lag | observed `B` child retention | observed `B` sync retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `723M-732M` | `5.5M` | `1.0000` | `0.9948` | `0.2088-0.4055` | `0.0000` |

This is the important scientific reading:

- the child-destination conservation is not a flat stationary property under one universal lag
- it persists into the next band when the lag is measured locally
- the apparent loss at fixed `15M` was mainly phase mismatch
- `matched-LZ` still fails to reproduce the destination channel, even when the same local lag is used
- `markov1` remains pure destination collapse

The earlier `705M-714M` band initially required conservative language. A narrow forward probe did not find a full-shell lag, and forcing `15M` collapsed observed `B`. That was a valid boundary reading under the first probe contract, but it was not the final story.

A wider boundary probe over `660M-780M` resolved this band as a negative-lag phase. It found `lag = -27M`; under that lag, observed `B` returns to `1.0000` child-destination retention, while `matched-LZ` remains only `0.1747-0.3297` and `markov1` remains `0.0000`.

The boundary therefore remains explicit, but its meaning changed: it is not an intrinsic no-transport gap. It is a retrograde / negative-lag phase relative to the `E` anchor band.

N2-09 does not prove:

- global stationarity
- a universal `15M` transport lag
- a final transport law
- absolute routing outside the monitored `top-k` child universe
- cross-parent routing
- geometry

The current internal wording should be:

> corrected child-destination conservation persists as a lag-local, phase-sensitive regime.

N2-09b then converted this persistence check into an explicit regime map by running `--lag-mode probe` over:

- `696M-705M`
- `705M-714M`
- `714M-723M`
- `723M-732M`

The result is:

| Band | Probe lag | Status | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | --- | ---: | ---: | ---: |
| `696M-705M` | `500K` | completed | `0.6456` | `0.1650-0.2860` | `0.0000` |
| `705M-714M` | `-27M` | completed under wide boundary probe | `1.0000` | `0.1747-0.3297` | `0.0000` |
| `714M-723M` | `15M` | completed | `0.9604` | `0.3107-0.4832` | `0.0000` |
| `723M-732M` | `5.5M` | completed | `1.0000` | `0.2088-0.4055` | `0.0000` |

This is better than a cherry-picked positive-only confirmation.

It gives:

- a partial / liminal positive band at `696M-705M`
- a negative-lag strong band at `705M-714M`
- a canonical strong band at `714M-723M`
- a phase-shifted strong band at `723M-732M`

The `705M-714M` boundary is especially useful because it exposed the difference between a narrow forward probe and a wider phase search. The first map correctly recorded `no-probe-lag` instead of forcing an invalid lag; the follow-up then showed that the missing relation was located behind the anchor band, at `-27M`.

A one-band forward extension was then run over `732M-741M`.

The probe returned `lag = 0`, i.e. the shell is already fully present in the same window. The readout is:

| Band | Probe lag | Status | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | ---: | --- | ---: | ---: | ---: |
| `732M-741M` | `0` | completed | `1.0000` | `0.5361-0.8317` | `0.0000` |

This should be read as a same-window / zero-lag subregime, not as the strongest child-routing separation.

It still supports observed `B` conservation, but `matched-LZ` is much closer here than in the `714M-723M` and `723M-732M` bands. The value of this band is therefore cartographic: it shows the active child-destination regime entering a coincident phase where the strongest null partly shadows the monitored destinations.

### 15. N2-10/N2-10c turns the child-routing map into a phase-local plateau

D-0055 adds a phase-lag atlas and a targeted child-routing validation over the atlas.

The important methodological change is that we no longer treat the apparent visual curve as evidence by itself. N2-10 first maps three lag estimators per band:

- first full shell lag
- hit-profile correlation lag
- unique-profile correlation lag

N2-10b then classifies each band by estimator agreement:

| Band | First full | Hit corr | Unique corr | Confidence |
| --- | ---: | ---: | ---: | --- |
| `687M-696M` | `26.5M` | `86.5M` | `87M` | transition |
| `696M-705M` | `-27M` | `-27M` | `-27M` | strong |
| `705M-714M` | `-27M` | `-27M` | `-27M` | strong |
| `714M-723M` | `-26M` | `-27M` | `-27M` | strong |
| `723M-732M` | `-17M` | `-27M` | `-27M` | transition |

The first targeted validation reran the audited child-routing stack only on the strong and transition bands, using the consensus lag from the tri-estimator audit.

The result is:

| Band | Confidence | Lag | observed `B` child retention | `matched-LZ` envelope | `markov1` |
| --- | --- | ---: | ---: | ---: | ---: |
| `687M-696M` | transition | `87M` | `0.5466` | `0.7533-0.8567` | `0.0000` |
| `696M-705M` | strong | `-27M` | `0.9703` | `0.1877-0.3189` | `0.0000` |
| `705M-714M` | strong | `-27M` | `1.0000` | `0.1747-0.3297` | `0.0000` |
| `714M-723M` | strong | `-27M` | `0.9947` | `0.2631-0.4124` | `0.0000` |
| `723M-732M` | transition | `-27M` | `1.0000` | `0.2360-0.3592` | `0.0000` |

This sharpens the previous regime-map reading.

The robust internal claim is now:

> A strong negative-lag child-destination plateau exists from `696M-723M`, centered at about `-27M`, where observed `B` preserves monitored child destinations at `0.9703-1.0000`, while `matched-LZ` remains in a deficit envelope and `markov1` collapses.

The right transition band `723M-732M` remains positive for HSI under the `-27M` correlation-supported lag: it preserves observed `B = 1.0000` and still separates from `matched-LZ`. This was later hardened by D-0056, which showed that the same band collapses at the discrepant `first_full = -17M` lag. It is therefore a phase-selective boundary, not an unconditional extension of the plateau.

The left transition band `687M-696M` is not positive evidence under child-routing. There, the correlation-supported `87M` lag gives observed `B = 0.5466`, while `matched-LZ` reaches `0.7533-0.8567`. That band is therefore a genuine transition/alias boundary, not part of the strong plateau.

This is scientifically useful. It prevents us from over-reading the visual "wave" impression and replaces it with a measured phase-local plateau plus asymmetric transition zones.

N2-10d then refined the plateau edges with a fine boundary sweep over:

- `684M`, `687M`, `690M`, `693M`, `696M`
- `705M`, `714M`
- `723M`, `726M`, `729M`, `732M`, `735M`

The first fine sweep produced `5` ambiguous bands, `4` transition bands, and `3` strong bands. D-0056 then hardened the contract: `strong` bands remain single-lag validations, while discrepant `transition` bands are validated twice, once at consensus and once at `first_full`.

The dual-lag validated readout is:

| Band | Confidence | Lag source | Lag | observed `B` child retention | `matched-LZ` envelope | Reading |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `687M-696M` | transition | consensus | `87M` | `0.5466` | `0.7533-0.8567` | left boundary, non-positive |
| `687M-696M` | transition | first_full | `26.5M` | `0.5466` | `0.7655-0.8673` | left boundary, non-positive |
| `696M-705M` | strong | consensus | `-27M` | `0.9703` | `0.1877-0.3189` | plateau |
| `705M-714M` | strong | consensus | `-27M` | `1.0000` | `0.1747-0.3297` | plateau |
| `714M-723M` | strong | consensus | `-27M` | `0.9947` | `0.2631-0.4124` | plateau |
| `723M-732M` | transition | consensus | `-27M` | `1.0000` | `0.2360-0.3592` | right transition, positive but lag-selective |
| `723M-732M` | transition | first_full | `-17M` | `0.0000` | `0.1921-0.3392` | right transition, collapse at first_full |
| `726M-735M` | transition | consensus | `-88M` | `0.0000` | `0.1410-0.3531` | alias transition, negative |
| `726M-735M` | transition | first_full | `-14M` | `0.0000` | `0.1723-0.3089` | alias transition, negative |
| `729M-738M` | transition | consensus | `-27M` | `1.0000` | `0.5976-0.7354` | right transition, weak separation |
| `729M-738M` | transition | first_full | `-5.5M` | `0.2354` | `0.4285-0.8477` | right transition, non-positive at first_full |

This changes the visual intuition.

The phase map should not be read as a smooth wave. It is better read as a robust `-27M` plateau with asymmetric, aliased transition zones. The left boundary is genuinely non-positive for HSI; the right boundary remains phase-selective rather than smoothly persistent. Consensus-supported `-27M` continuations can remain positive, but discrepant `first_full` lags can collapse completely.

N2-10/N2-10c does not prove:

- a wave law
- global stationarity
- a universal lag
- absolute routing outside the monitored `top-k` child universe
- final Gate 2 resolution

### 16. N2-11 turns the boundary question into a lag-response profile

D-0057 adds a direct lag-response readout over the audited child-destination routing stack.

The methodological move is simple but important: instead of choosing one lag from the tri-estimator map and asking whether the band succeeds or fails, N2-11 evaluates the same child-destination object over an explicit lag grid.

Canonical run:

- `results/hsi_v2/phase2/child_routing_lag_response_n2_11/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-8__20260427T165759/`

Visual readout:

- `results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/`
- stable figure aliases: `results/hsi_v2/phase2/visual_readouts/lag_response_heatmap_latest.png` and `results/hsi_v2/phase2/visual_readouts/lag_response_heatmap_latest.svg`

The tested grid covers six bands:

- `696M-705M`
- `705M-714M`
- `714M-723M`
- `723M-732M`
- `726M-735M`
- `729M-738M`

and eight lags:

- `-88M`, `-27M`, `-17M`, `-14M`, `-5.5M`, `0M`, `26.5M`, `87M`

The compact result is:

| Band | Best observed `B` lag | observed `B` | Best margin lag | `B - max(matched-LZ)` |
| --- | ---: | ---: | ---: | ---: |
| `696M-705M` | `-5.5M` | `0.9852` | `-5.5M` | `0.6608` |
| `705M-714M` | `-27M` | `1.0000` | `-5.5M` | `0.6782` |
| `714M-723M` | `-27M` | `0.9947` | `-27M` | `0.5823` |
| `723M-732M` | `-27M` | `1.0000` | `-27M` | `0.6408` |
| `726M-735M` | `-27M` | `1.0000` | `26.5M` | `0.6495` |
| `729M-738M` | `-27M` | `1.0000` | `0M` | `0.2810` |

This changes one important reading from D-0056.

D-0056 remains correct under its own stricter dual-lag contract: at `726M-735M`, the consensus/first-full lags (`-88M` and `-14M`) both collapse observed `B`.

But N2-11 shows that `726M-735M` is not a truly dead band. Under the full lag grid, it recovers full observed `B` child-destination retention at `-27M` and `26.5M`, with strong positive separation from `matched-LZ`.

So the honest reading is now:

> Observed `B` exhibits discrete phase-channel child-destination conservation. The strongest global tested channel is `-27M`, but additional strong channels appear at `26.5M` and in some bands at `-5.5M`, `-14M`, or `0M`. `markov1` collapses everywhere, while `matched-LZ` remains a diffuse non-zero background and becomes substantially stronger in the late `729M-738M` band.

This strengthens the internal mechanism but also disciplines the language.

We should no longer say that the right boundary contains a simple dead zone. It is better read as phase-selective: some estimator lags collapse, while other tested lags recover the monitored child-destination channel.

N2-11 still does not prove:

- a wave law
- a universal lag
- global stationarity
- a completed transport law
- absolute routing outside the monitored `top-k` child universe
- final Gate 2 resolution

---

### 17. N2-12 turns the strongest phase channel into a sampled width readout

D-0059 refines the N2-11 lag-response result with a denser lag-neighborhood sweep around the two principal channels:

- the negative channel around `-27M`
- the positive channel around `26.5M`

The methodological point is narrow and important. N2-12 does not create a new observable. It reuses the audited child-destination routing object and the N2-11 lag-response wrapper, then asks whether the N2-11 signal was only a single lucky lag point or a sampled channel neighborhood.

Canonical full revalidation:

- `results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-10__20260428T162651/`

Visual readout:

- `results/hsi_v2/phase2/visual_readouts/phase2-phase-channel-width-figure__src-20260428T162651__20260430T123015/`
- `results/hsi_v2/phase2/visual_readouts/phase_channel_width_latest.png`

The full run completed all `60` band-lag targets:

- bands: `696M-705M`, `705M-714M`, `714M-723M`, `723M-732M`, `726M-735M`, `729M-738M`
- negative lags: `-30M`, `-29.5M`, `-27M`, `-26.5M`, `-26M`, `-24M`
- positive lags: `23.5M`, `24.5M`, `26.5M`, `29.5M`

The strong-cell convention is descriptive, not a statistical p-value:

> strong cell = observed `B >= 0.90` and `B - max(matched-LZ) >= 0.30`

Under that convention, the negative channel is not a single point. It forms a sampled envelope:

| Band | Negative-channel strong span | Width |
| --- | ---: | ---: |
| `696M-705M` | `-27M..-26M` | `1M` |
| `705M-714M` | `-29.5M..-27M` | `2.5M` |
| `714M-723M` | `-27M..-24M` | `3M` |
| `723M-732M` | `-30M..-26M` | `4M` |
| `726M-735M` | `-29.5M..-27M` | `2.5M` |
| `729M-738M` | none under the strong-cell criterion | - |

The positive channel is more selective:

| Band | Positive-channel strong span | Width |
| --- | ---: | ---: |
| `696M-705M` | none | - |
| `705M-714M` | `24.5M..26.5M` | `2M` |
| `714M-723M` | none | - |
| `723M-732M` | `26.5M..29.5M` | `3M` |
| `726M-735M` | `24.5M..26.5M` | `2M` |
| `729M-738M` | `29.5M` | `0M` sampled width |

The key scientific reading is:

> Observed `B` exhibits threshold-defined sampled phase-channel envelopes in the monitored top-128 child-destination universe. The negative channel around `-27M` is the strongest and most stable sampled channel; the positive channel around `26.5M` is selective and becomes the stronger separator at the late boundary.

The late band `729M-738M` is the cautionary zone. It is not a simple failure: observed `B` still reaches perfect retention in some cells, but `matched-LZ` rises strongly and contracts the margin. The best separator there is the positive edge at `29.5M`, with observed `B = 1.0000` and margin `+0.3017`.

This gives a more precise internal mechanism than N2-11 alone, but it also keeps the language disciplined:

- say "threshold-defined sampled phase-channel width"
- say "sampled channel envelope"
- do not say "formal FWHM"
- do not say "5-sigma"
- do not say "continuous resonance"
- do not say "universal lag"
- do not say "Gate 2 solved"

---

### 18. N2-12b top-k sensitivity hardens the central readout and demotes `729M-738M` to boundary status

D-0060 adds a top-k sensitivity check to the N2-12 phase-channel width line.

This is not a new observable. It is a resolution stress test on the monitored child universe:

- `top=128` remains the canonical N2-12 reference.
- `top=64` reruns the full sampled grid.
- `top=96` reruns the late frontier band `729M-738M`.

The full `top=64` run is important because it does not punch holes in the central N2-12 map. There are no missing measured cells, and the central strong-cell envelopes remain essentially the same as in the top-128 reference. Outside the late boundary, the average absolute margin drift is small, about `0.017`.

The late band behaves differently. `729M-738M` is not a stable positive-channel continuation:

| Top-k | Strong negative cell | Strong positive cell | Reading |
| ---: | --- | --- | --- |
| `64` | `-27M`, margin `0.4091` | none | negative boundary cell |
| `96` | `-27M`, margin `0.3949` | none | negative boundary cell |
| `128` | none | `29.5M`, margin `0.3017` | positive boundary cell |

This is not a measurement failure. The `top=96` frontier run has no missing expected cells. The result says something sharper: in the late boundary band, observed `B` can still retain the monitored child destination, but the strongest matched-LZ background is high enough that the surviving strong separator changes channel under top-k resolution stress.

The current honest reading is:

> N2-12 is robust in the central sampled bands, while `729M-738M` is a top-k-sensitive boundary band. It should not be used as evidence for a stable positive late channel unless future work resolves the monitored-universe dependence.

This tightens the visual interpretation too. The current `phase_channel_width_latest.png` now carries a boundary/top-k sensitivity note on the late row, so it does not visually imply that `729M-738M` belongs to the same stable envelope as the central bands.

---

### 19. N2-13a explains the late boundary as monitored null-shadowing, not observed-`B` collapse

D-0061 adds routing signatures over the N2-12b top-k sensitivity grid.

The object is deliberately bounded. It does not classify the full unbounded off-top-k universe. It classifies the **expansion delta** between lower monitored child universes (`top=64`, targeted `top=96`) and the canonical monitored reference (`top=128`), mining the already-generated child-routing subruns.

Canonical readout:

- `results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/`
- `docs/HSI-audit-Phase2-Routing-Signatures.md`

The central bands remain stable:

| Test top-k | Signature | Cells | Bands |
| ---: | --- | ---: | --- |
| `64` | `stable-monitored-channel` | `22` | `696M-705M`, `705M-714M`, `714M-723M`, `723M-732M`, `726M-735M` |

The late boundary `729M-738M` has a different mechanism:

| Signature | Meaning |
| --- | --- |
| `reference-null-shadowing` | the matched-LZ background rises under expansion and erases a lower-top strong margin |
| `reference-null-shadow-relief` | the matched-LZ background falls under expansion and lets observed `B` re-emerge as a strong cell |
| `expanded-rank-shell-B-gain` | observed `B` gains retention in the expanded monitored rank shell, but this does not by itself define a stable channel |

The canonical short reading is:

> N2-13a shows that the late `729M-738M` boundary is governed by monitored null-shadowing dynamics rather than observed-`B` collapse.

This resolves the D-0060 boundary question. `729M-738M` is not a stable positive late-channel extension, but it is also not a death of observed `B`.

The `matched-LZ` deficit shape is `diffuse-deficit` across the observed signature classes. Within the monitored expansion delta, this argues against a concentrated hidden child attractor stealing the mass. It should not be stated as a refutation of hidden structure in the unbounded off-top-k universe.

This is a methodological closure for the internal Phase 2 mechanism. It is not an editorial closure of Phase 2 in the paper:

- strict `null_pressure` remains the first and only paper-safe Gate 2 opening
- N2-11 remains the only staged Phase 2 paper-candidate figure in this child-routing line
- N2-12 and N2-13a remain internal diagnostics unless a later editorial architecture decision promotes them
- do not say "Phase 2 closed" in a paper sense

---

### 20. N2-14 assembles the sampled phase-channel cells into a pre-spectral transition scaffold

D-0062 adds a deliberately small transition-graph readout over existing N2-12b and N2-13a artifacts.

The object does not recompute raw bits and does not introduce geometry. It asks a narrower question:

> Do the already measured `band x channel` cells form adjacent-band paths, or are they isolated diagnostic points?

Canonical readout:

- `results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/`
- `docs/HSI-audit-Phase2-Transition-Graph.md`

The accepted path summary is:

| Top-k | Channel | Active bands | Boundary nodes | Max active path | Verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| `96` | negative | `1/1` | `1` | `1` | `no-stable-channel-path` |
| `96` | positive | `0/1` | `1` | `0` | `no-stable-channel-path` |
| `128` | negative | `5/6` | `1` | `5` | `structured-channel-with-boundary` |
| `128` | positive | `4/6` | `1` | `3` | `structured-channel-with-boundary` |

The strongest internal reading is therefore:

> The `top=128` negative channel is not merely a set of isolated strong cells. It forms an adjacent-band scaffold across the sampled central regime and terminates at the `729M-738M` monitored null-shadowing boundary.

The positive channel is real but more intermittent. It has activation and attenuation transitions and a shorter active path, so it should remain secondary in any later editorial architecture.

The `top=96` frontier does not falsify the `top=128` path. It shows that the transition scaffold is resolution-dependent under the monitored top-k contract; at `top=96`, the late boundary cells are visible but do not assemble into a stable path.

This is the final internal pre-editorial bridge for the current child-routing stack. It is still not a paper-safe Gate 2 opening by itself:

- strict `null_pressure` remains first in the evidentiary hierarchy
- N2-11 remains the only staged child-routing paper-candidate figure
- N2-12, N2-13a, and N2-14 remain internal diagnostics unless an editorial architecture decision promotes or summarizes them
- no graph geometry, spectral geometry, or final transport law follows from N2-14 alone

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
- the parent-density plus shell-atlas line now shows `B` preserving shell mass with high redistribution jitter-gap, while `matched-lz` remains net-deficit dominated
- N2-07 / N2-07b show that the `B` jitter-gap is mostly observable same-parent temporal compensation, while `matched-LZ` remains net-deficit dominated
- N2-08 shows that observed `B` retains about `96%` of monitored high-scale child-destination mass after jitter compensation, while `matched-LZ` remains in a `31-48%` same-destination-deficit band
- N2-09 shows that this child-destination conservation persists into the following band when the candidate lag is chosen locally (`723M-732M`, `5.5M`, observed `B = 1.0000`, `matched-LZ = 0.2088-0.4055`)
- N2-09b maps the local regime: partial `B` conservation at `696M-705M`, negative-lag strong conservation at `705M-714M`, strong conservation at `714M-723M`, and phase-shifted full conservation at `723M-732M`
- the `732M-741M` extension shows a zero-lag same-window conservation band for `B`, but with weaker separation because `matched-LZ` rises to `0.5361-0.8317`
- N2-10/N2-10c refine that map into a phase-local plateau: `696M-723M` is strong at about `-27M`, while transition bands must be read under the stricter dual-lag contract
- N2-10d/D-0056 refines the edges: the left boundary remains non-positive under both lags; `723M-732M` and `729M-738M` are positive only at the correlation-supported `-27M` lag; `726M-735M` is negative under both consensus and `first_full`
- N2-11/D-0057 shows that the broader lag-response grid is not a single-lag law but a set of discrete phase channels: `-27M` is the strongest global tested channel, `26.5M` forms a second strong channel in several bands, and `726M-735M` is phase-selective rather than truly dead
- N2-12/D-0059 shows that the main `-27M` channel is not just an isolated lag spike: under the current sampled grid it forms threshold-defined channel envelopes across adjacent lags, while the positive channel is more selective and the late `729M-738M` band becomes a matched-LZ-shadowed boundary regime
- N2-12b/D-0060 shows that this central channel-envelope reading survives a `top=64` stress test, while `729M-738M` is explicitly top-k-sensitive and must be treated as a boundary band rather than a stable positive-channel extension
- N2-13a/D-0061 shows that `729M-738M` is a monitored null-shadowing boundary: observed `B` does not collapse, the matched-LZ background alternately shadows and relaxes under top-k expansion, and the matched-LZ deficit shape is diffuse rather than concentrated inside the monitored expansion delta
- N2-14/D-0062 shows that the `top=128` sampled phase-channel cells assemble into adjacent-band paths: the negative channel gives a `5/6` active-band scaffold ending in a `null-shadow-boundary`, while the positive channel is shorter and more intermittent

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
- a paper-final Gate 2 claim based only on the shell-atlas or sensitivity readouts
- absolute topology claims from Prefix-only, Suffix-only, or Bridge-both labels
- a child-destination routing claim from N2-07
- guaranteed physical-flow identity from same-parent compensation alone
- an absolute child-routing claim outside the monitored `top-k` child universe
- cross-parent routing from N2-08
- a globally stationary child-routing regime from N2-09
- a universal `15M` lag from the child-routing persistence sweep
- a wave law from the N2-10 visual phase-lag profile
- a universal `-27M` lag from the N2-11 lag-response profile
- a formal continuous phase-channel bandwidth law from N2-12
- a statistical significance claim from the N2-12 strong-cell threshold
- a claim that all right-boundary bands are dead or alive independent of lag
- a top-k-independent routing claim for the late `729M-738M` boundary
- absolute off-top-k mass classification beyond the monitored `top=128` universe from N2-13a
- a top-k-independent phase-channel transition scaffold from N2-14
- a graph geometry, spectral geometry, or physical trajectory claim from the N2-14 transition graph
- editorial closure of Phase 2 in `main_v2_cat.tex`

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

### Canonical parent-shell atlas

- `results/hsi_v2/phase2/parent_shell_atlas_pipeline/phase2-parent-shell-atlas-pipeline__profile-canonical-all__stage-all__anchor-E__cand-B__20260423T223709/`

### Parent-shell atlas sensitivity sweep

- `results/hsi_v2/phase2/parent_shell_atlas_sensitivity/phase2-parent-shell-atlas-sensitivity__stage-all__anchor-E__cand-B__top-32-128x3__lag-14500K-15500Kx3__20260423T234050/`

### Same-parent jitter provenance

- `results/hsi_v2/phase2/jitter_provenance/phase2-jitter-provenance__anchor-E__cand-B__top-64__lag-15M__src-20260423T225436__20260424T164539/`

### Same-parent jitter provenance sensitivity

- `results/hsi_v2/phase2/jitter_provenance_sensitivity/phase2-jitter-provenance-sensitivity__top-64-128__src-20260423T234050__20260424T180722/`

### Child-destination routing

- `results/hsi_v2/phase2/child_destination_routing/phase2-child-destination-routing__anchor-E__cand-B__top-128__lag-15M__src-20260424T091639__20260424T231132/`

### Child-routing persistence

- `results/hsi_v2/phase2/child_routing_persistence/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-fixed-15M__starts-705M-723Mx3__20260425T000933/`
- `results/hsi_v2/phase2/child_routing_persistence_laglocal/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-fixed-5500K__starts-723M-723Mx1__20260425T012305/`
- `results/hsi_v2/phase2/child_routing_persistence_probe_map/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-696M-723Mx4__20260425T015723/`
- `results/hsi_v2/phase2/child_routing_persistence_boundary705_wide/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-705M-705Mx1__20260425T085419/`
- `results/hsi_v2/phase2/child_routing_persistence_probe_extension/phase2-child-routing-persistence__stage-all__anchor-E__cand-B__top-128__lag-probe-15M__starts-732M-732Mx1__20260425T025619/`
- `results/hsi_v2/phase2/child_routing_regime_map/phase2-child-routing-regime-map__20260425T091530/`
- `results/hsi_v2/phase2/phase_lag_atlas/phase2-phase-lag-atlas__anchor-E__cand-B__top-128__bands-660M-732Mx9__probe-back-45M__fwd-45M__20260425T100139/`
- `results/hsi_v2/phase2/phase_lag_tri_estimator/phase2-phase-lag-tri-estimator__anchor-E__cand-B__top-128__bands-660M-732Mx9__tol-1M__20260425T104220/`
- `results/hsi_v2/phase2/child_routing_phase_validation/phase2-child-routing-phase-validation__stage-all__conf-strong-transition__bands-687M-723Mx5__20260425T111658/`
- `results/hsi_v2/phase2/child_routing_boundary_sweep/phase2-child-routing-boundary-sweep__stage-all__top-128__bands-684M-735Mx12__20260426T105136/`
- `results/hsi_v2/phase2/child_routing_boundary_sweep_dual_transition/phase2-child-routing-boundary-sweep__stage-all__top-128__lag-dual-transition__bands-684M-735Mx12__20260427T084746/`
- `results/hsi_v2/phase2/child_routing_lag_response_n2_11/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-8__20260427T165759/`
- `results/hsi_v2/phase2/visual_readouts/phase2-lag-response-heatmap__src-20260427T165759__20260428T083731/`
- `results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval/phase2-child-routing-lag-response__stage-all__anchor-E__cand-B__top-128__bands-696M-705M-to-729M-738M__lags-10__20260428T162651/`
- `results/hsi_v2/phase2/visual_readouts/phase2-phase-channel-width-figure__src-20260428T162651__20260430T123015/`
- `results/hsi_v2/phase2/visual_readouts/phase_channel_width_latest.png`
- `results/hsi_v2/phase2/phase_channel_width_topk_sensitivity/phase2-phase-channel-width-topk-sensitivity__stage-all__top-64-128x2__lags-10__20260429T102045/`
- `results/hsi_v2/phase2/phase_channel_width_topk_sensitivity_frontier_top96/phase2-phase-channel-width-topk-sensitivity__stage-all__top-96-128x2__lags-10__20260430T065831/`
- `results/hsi_v2/phase2/off_universe_mass_readout/phase2-off-universe-mass-readout__runs-2__20260430T123656/`
- `results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724/`

---

## Reading Order For Recovery

If you want the current scientific state quickly, read:

1. `Documentation/HSI_v2_Phase1_Live_Findings.md`
2. `Documentation/HSI_v2_Phase2_Live_Findings.md`
3. `HSI_v2_Decision_Log.md`

If you want the current counterfactual result specifically, read:

1. the strict `N2-02` report under `results/hsi_v2/phase2/transport_defect_counterfactual/...`
2. the matching `summary.json`
3. `Documentation/HSI_v2_Phase2_Live_Findings.md`
4. `HSI_v2_Decision_Log.md`

If you want to know what still counts as the best Gate 2 candidate, read:

1. the strict `null_pressure` report under `results/hsi_v2/phase2/window_sweep_deep/null_pressure_strict/...`
2. D-0037 in `HSI_v2_Decision_Log.md`
3. this live-findings document

If you want the current survival-line reading specifically, read:

1. the canonical parent-shell revalidation report under `results/hsi_v2/phase2/parent_survival_revalidation/...`
2. the nested shell-lag probe summary
3. the nested lag-aware band summary
4. D-0047 in `HSI_v2_Decision_Log.md`

If you want the current parent-density reading specifically, read:

1. the corrected parent-density residual report under `results/hsi_v2/phase2/parent_density_residual_corrected/...`
2. the matching `summary.json`
3. D-0048 in `HSI_v2_Decision_Log.md`

If you want the current shell-atlas and sensitivity reading specifically, read:

1. the parent-shell atlas sensitivity report under `results/hsi_v2/phase2/parent_shell_atlas_sensitivity/...`
2. `concentration_sensitivity.csv`
3. `structure_sensitivity.csv`
4. D-0049 and D-0050 in `HSI_v2_Decision_Log.md`

If you want the current same-parent jitter-provenance reading specifically, read:

1. the N2-07b sensitivity report under `results/hsi_v2/phase2/jitter_provenance_sensitivity/...`
2. `source_envelope.csv`
3. the canonical N2-07 report under `results/hsi_v2/phase2/jitter_provenance/...`
4. D-0052 in `HSI_v2_Decision_Log.md`

If you want the current child-destination routing reading specifically, read:

1. the N2-08 report under `results/hsi_v2/phase2/child_destination_routing/...20260424T231132/`
2. `source_routing.csv`
3. `pattern_routing.csv`
4. D-0053 in `HSI_v2_Decision_Log.md`

If you want the current child-routing persistence reading specifically, read:

1. the compact regime-map report under `results/hsi_v2/phase2/child_routing_regime_map/...20260425T091530/`
2. `source_persistence.csv`
3. `band_persistence.csv`
4. the wide `705M-714M` boundary report under `results/hsi_v2/phase2/child_routing_persistence_boundary705_wide/...20260425T085419/`
5. the forward extension report under `results/hsi_v2/phase2/child_routing_persistence_probe_extension/...20260425T025619/`
6. the earlier fixed-lag report under `results/hsi_v2/phase2/child_routing_persistence/...20260425T000933/`
7. the lag-local `723M-732M` report under `results/hsi_v2/phase2/child_routing_persistence_laglocal/...20260425T012305/`
8. the N2-10 phase-lag atlas report under `results/hsi_v2/phase2/phase_lag_atlas/...20260425T100139/`
9. the N2-10b tri-estimator report under `results/hsi_v2/phase2/phase_lag_tri_estimator/...20260425T104220/`
10. the N2-10c targeted validation report under `results/hsi_v2/phase2/child_routing_phase_validation/...20260425T111658/`
11. the N2-10d fine boundary sweep report under `results/hsi_v2/phase2/child_routing_boundary_sweep/...20260426T105136/`
12. the dual-lag N2-10d rerun under `results/hsi_v2/phase2/child_routing_boundary_sweep_dual_transition/...20260427T084746/`
13. the N2-11 lag-response profile under `results/hsi_v2/phase2/child_routing_lag_response_n2_11/...20260427T165759/`
14. the N2-11 heatmap under `results/hsi_v2/phase2/visual_readouts/...20260428T083731/`
15. the N2-12 phase-channel width run under `results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval/...20260428T162651/`
16. the N2-12 phase-channel width figure under `results/hsi_v2/phase2/visual_readouts/...20260430T123015/`
17. the N2-12 top-k sensitivity runs under `results/hsi_v2/phase2/phase_channel_width_topk_sensitivity/...20260429T102045/` and `results/hsi_v2/phase2/phase_channel_width_topk_sensitivity_frontier_top96/...20260430T065831/`
18. the N2-13a routing-signature readout under `results/hsi_v2/phase2/off_universe_mass_readout/...20260430T123656/`
19. the N2-14 transition-graph readout under `results/hsi_v2/phase2/phase_channel_transition_graph/...20260430T140724/`
20. D-0054, D-0055, D-0056, D-0057, D-0058, D-0059, D-0060, D-0061, and D-0062 in `HSI_v2_Decision_Log.md`

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
- the shell atlas shows that observed `B` is dominated by redistribution jitter-gap, while `matched-lz` is dominated by net shell deficit
- the sensitivity sweep preserves that reading at the useful resolutions and classifies `top = 32` as a blind resolution control
- the jitter-provenance sensitivity readout shows that observed `B` is stable as same-parent compensated jitter, while `matched-lz` remains net-deficit dominated
- the child-destination routing readout shows that, after jitter compensation, observed `B` largely preserves monitored high-scale child destinations, while `matched-lz` remains same-destination-deficit dominated
- the child-routing persistence readout shows that this destination conservation is lag-local and phase-sensitive: the later `723M-732M` band recovers full observed `B` conservation at its measured `5.5M` lag
- the probe-mode regime map adds useful boundary structure: `696M-705M` is only partial, while `705M-714M` resolves as a strong negative-lag phase under a wider scan, so the child-routing regime is structured rather than globally stationary
- the `732M-741M` forward extension adds a same-window / zero-lag phase where observed `B` remains conserved but `matched-lz` partially shadows the monitored child destinations
- the N2-10/N2-10c phase-lag validation turns this into a sharper internal map: a strong `-27M` plateau from `696M-723M`, with transition bands kept below plateau status
- the N2-10d/D-0056 dual-lag sweep confirms that the plateau edges are aliased rather than smooth: `687M-696M` is non-positive under both lags, `726M-735M` is negative under both lags, and `723M-732M` / `729M-738M` are positive only at the correlation-supported `-27M` lag
- the N2-11/D-0057 lag-response profile corrects the broad reading of `726M-735M`: it is negative under the D-0056 estimator lags, but not dead under the full grid; it recovers strongly at `-27M` and `26.5M`
- the N2-12/D-0059 phase-channel width readout shows that the main `-27M` channel has sampled width under a conservative strong-cell convention, while the positive channel is selective and the late band is matched-LZ-shadowed
- the N2-12b/D-0060 top-k sensitivity readout preserves the central channel-envelope result, but reclassifies `729M-738M` as a top-k-sensitive boundary rather than a settled positive late channel
- the N2-13a/D-0061 routing-signature readout explains `729M-738M` as monitored null-shadowing: observed `B` does not collapse, matched-LZ alternately shadows and relaxes under top-k expansion, and the matched-LZ deficit is diffuse inside the monitored expansion delta
- the N2-14/D-0062 transition-graph readout shows that the `top=128` negative channel cells assemble into an adjacent-band path before the null-shadowing boundary, while the positive channel is secondary and more intermittent
- the current best internal reading is therefore threshold-defined sampled phase-channel conservation plus adjacent-band transition paths and a monitored null-shadowing boundary, not a smooth wave, not a universal lag, not absolute off-universe classification, not graph geometry, and not a final transport law

So the next serious move is not to overstate a transport-defect victory.

It is to:

- keep `null_pressure` as the first paper-safe Gate 2 opening
- preserve `N2-02` as a valuable negative result
- preserve the parent-density, shell-atlas, jitter-provenance, and child-destination routing line as a hardened internal diagnostic
- preserve N2-13a as the wording fence around the `729M-738M` boundary
- preserve N2-14 as the final internal pre-editorial bridge from sampled phase-channel cells to adjacent-band path structure
- and decide, in a separate editorial architecture step, how much of this derived mechanism belongs in the Phase 2 appendix versus the main paper narrative
