# 📐 Geometric Analysis Plan: Γ, 𝓡, 𝓣

**Purpose:** Implementation plan for Connection, Curvature and Torsion informational operators  
**Status:** Planning phase  
**Created:** 2025-12-06

---

## 📊 Overview

This document describes the implementation of geometric operators from the ISH paper (Section 4.3):

| Operator | Name | What it measures | Iterations needed |
|----------|------|------------------|-------------------|
| **Γ** | Connection | Pattern transport between states | 2+ consecutive |
| **𝓡** | Curvature | Rule interference (order dependency) | 2+ consecutive |
| **𝓣** | Torsion | Positional asymmetries | 1 (single iteration) |
| **𝓔** | Energy | Global rule quality | All above |

---

## 📋 Prerequisites

### Data Requirements

The geometric analyzer uses **only Level 1 results** (pre-computed pattern positions).
No need to load raw phi_snapshots - this makes analysis faster and simpler.

| Requirement | Source | Status |
|-------------|--------|--------|
| Level 1 results (iteration N) | `results/level1_analysis_var{X}_iter{N}_min{M}_max{M}.json` | Required |
| Level 1 results (iteration N-1) | `results/level1_analysis_var{X}_iter{N-1}_min{M}_max{M}.json` | For Γ, 𝓡 |
| Level 1 results (iteration N-2) | `results/level1_analysis_var{X}_iter{N-2}_min{M}_max{M}.json` | For Γ, 𝓡 |

### ⚠️ IMPORTANT: Filename Convention for Reproducibility

Level 1 analysis results now include pattern detection parameters in the filename:

```
level1_analysis_var{VARIANT}_iter{ITERATION}_min{MIN_LEN}_max{MAX_LEN}.json
```

**Example:** `level1_analysis_varB_iter19_min10_max50.json`

This ensures:
1. **Scientific reproducibility**: Parameters are visible without opening the file
2. **Comparison safety**: Different parameter sets don't overwrite each other
3. **Consistency checking**: Geometric analysis can verify all iterations use same parameters

**Critical:** When comparing multiple iterations (for Γ and 𝓡), ALL iterations MUST use the same `min` and `max` parameters. The geometric analyzer will warn if parameters don't match.

### Missing Data Warning

When required Level 1 results are not available, the script displays:

```
⚠️  INSUFFICIENT DATA FOR GEOMETRIC ANALYSIS

Required Level 1 results status:
  ❌ level1_analysis_varB_iter17_min*_max*.json
  ❌ level1_analysis_varB_iter18_min*_max*.json
  ✅ results/level1_analysis_varB_iter19_min10_max50.json (min=10, max=50)

📋 Run Level 1 analysis for missing iterations:
     python agents/level1_orchestrator.py -v B -i 17 --min-len 10 --max-len 50 --no-cache --report
     python agents/level1_orchestrator.py -v B -i 18 --min-len 10 --max-len 50 --no-cache --report

💡 Alternative: Run Torsion-only analysis (single iteration):
     python agents/geometric_analyzer.py -v B -i 19 --torsion-only
```

---

## 🔬 Mathematical Definitions

### 1. Connection Γ (Pattern Transport)

From paper (lines 414-425):
```
Γ_α^(k) : S(R_α) → S(R_{α+1})
Γ_α^(k)(P_k) = ω_k(P_k)
```

**Interpretation:** How patterns "travel" from one state to the next through rules ω_k.

**Operational definition:**
- For each pattern P found in iteration N
- Search for the same pattern in iteration N+1
- Measure: frequency change, position change, survival rate

### 2. Curvature 𝓡 (Rule Interference)

From paper (lines 427-440):
```
𝓡_α^(i,j)(P) = Γ_{α+1}^(j)(Γ_α^(i)(P)) - Γ_{α+1}^(i)(Γ_α^(j)(P))
```

**Interpretation:** 
- If 𝓡 = 0 → Transport is coherent, flat structure (no rule interference)
- If 𝓡 ≠ 0 → Transport depends on path, curvature emerges (rule interference)

**Operational definition:**
- Take two rules ω_i and ω_j
- Apply in order i→j and measure result
- Apply in order j→i and measure result
- Curvature = difference between results

### 3. Torsion 𝓣 (Positional Asymmetries)

From paper (lines 442-450):
```
𝓣_α^(k)(P) = Γ_α^(k)(P) - Γ_α^(k)(P^shift)
```

**Interpretation:** Measures persistent asymmetries - how patterns behave differently based on position.

**Operational definition:**
- Divide Φ into segments (start, middle, end)
- Count pattern occurrences per segment
- Torsion = variance of counts / mean count

### 4. Energy Functional 𝓔

From paper (lines 494-508):
```
𝓔[ω] = α|𝓡[ω]|² + β|𝓣[ω]|² + γ·Φ_incoh[ω]
```

**Interpretation:** Global quality metric for rule configurations.
- Low energy = stable, coherent rules
- High energy = unstable, incoherent rules

---

## 🗂️ Implementation Tasks

### Phase 1: Base Script & Torsion [~2h]
- [ ] Create `agents/geometric_analyzer.py`
- [ ] Implement data loading from existing JSON results
- [ ] Implement prerequisite checker with clear error messages
- [ ] Implement Torsion 𝓣 calculation (single iteration)
- [ ] Generate partial report with Torsion results

### Phase 2: Multi-Iteration Analysis [~3h]
- [ ] Implement loading of multiple `.struct.gz` files
- [ ] Implement Connection Γ calculation
- [ ] Implement Curvature 𝓡 calculation
- [ ] Add pattern matching across iterations

### Phase 3: Integration [~1h]
- [ ] Implement Energy Functional 𝓔
- [ ] Generate complete geometric report
- [ ] Add integration with existing Level 1 workflow
- [ ] Create summary visualizations

---

## 📝 Output Format

The geometric analyzer will generate:

```
results/level1/geometric_B_iter19.json
results/level1/geometric_B_iter19_report.md
```

With structure:
```json
{
  "variant": "B",
  "iteration": 19,
  "iterations_analyzed": [17, 18, 19],
  "torsion": {
    "patterns_analyzed": 150,
    "patterns_with_torsion": 42,
    "mean_asymmetry": 0.234,
    "max_asymmetry": 0.891
  },
  "connection": {
    "patterns_tracked": 120,
    "survival_rate_17_18": 0.85,
    "survival_rate_18_19": 0.78,
    "mean_transport_ratio": 1.23
  },
  "curvature": {
    "rule_pairs_analyzed": 45,
    "commuting_pairs": 12,
    "non_commuting_pairs": 33,
    "mean_curvature": 0.156
  },
  "energy": {
    "total": 2.34,
    "curvature_term": 1.12,
    "torsion_term": 0.89,
    "incoherence_term": 0.33,
    "stability": "moderate"
  }
}
```

---

## 🎯 Execution Examples

```bash
# Full geometric analysis (requires 3 iterations)
python agents/geometric_analyzer.py --variant B --iteration 19

# Torsion-only analysis (single iteration)
python agents/geometric_analyzer.py --variant B --iteration 19 --torsion-only

# Specify iterations explicitly
python agents/geometric_analyzer.py --variant B --iterations 17 18 19

# Generate report
python agents/geometric_analyzer.py --variant B --iteration 19 --report
```

---

## 📚 References

- Paper Section 4.3: "Connexió, Curvatura i Torsió Informacional"
- Paper Section 4.4: "Funcional d'Energia i Flux d'Ordre"
- `agents/level1_orchestrator.py`: Current Level 1 analysis workflow
- `agents/pattern_detector.py`: Pattern detection implementation
- `agents/rule_inferer.py`: Rule inference implementation

