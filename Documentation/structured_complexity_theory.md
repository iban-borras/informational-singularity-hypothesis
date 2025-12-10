# Structured Complexity: A New Metric for Emergent Physical Systems

## 1. The Problem with Current Metrics

### 1.1 Limitation of Lempel-Ziv Complexity

The normalized LZ complexity measures compressibility:
- LZ ≈ 0 → Highly ordered (e.g., "010101...")
- LZ ≈ 1 → Incompressible (random noise)
- LZ ≈ 0.5 → "Edge of chaos"

**The flaw**: LZ ≈ 0.5 doesn't distinguish between:
- **Meaningful complexity**: A compressed file (looks random, contains structure)
- **Empty randomness**: White noise (looks random, contains nothing)

### 1.2 The Entropy ≠ Information Paradox

Shannon entropy measures uncertainty, not meaning:
- A ZIP file has maximum entropy but contains all original information
- White noise has maximum entropy but contains zero information

**Key insight**: We need to measure *structured* information, not just randomness.

## 2. What Physical Systems Require

A stable, complex physical system (like our universe) exhibits:

| Property | Mathematical Signature | Example |
|----------|----------------------|---------|
| **Compressible laws** | Short algorithmic description | F = ma |
| **Complex states** | High number of configurations | 10^80 atoms in universe |
| **Hierarchy** | Structure at multiple scales | quarks → atoms → molecules → cells |
| **Causality** | Past conditions future | Deterministic evolution |
| **Symmetry breaking** | Patterns with variations | Crystals with defects |

## 3. Proposed Metrics for Structured Complexity

### 3.1 Hierarchical Block Entropy (HBE)

Measure entropy at multiple scales:

```
HBE = Σ_k w_k · H(blocks of size 2^k)
```

Where:
- H(block_size) = entropy of block frequency distribution
- w_k = weights (could be 1/k for scale-invariance)

**Interpretation**:
- Random sequence: H constant across all scales
- Structured sequence: H varies with scale (hierarchy)

### 3.2 Multi-Scale Mutual Information (MSMI)

Measure correlation between different scales:

```
MSMI = Σ_k MI(scale_k, scale_{k+1})
```

**Interpretation**:
- Random: MSMI ≈ 0 (scales are independent)
- Structured: MSMI > 0 (scales are correlated)

### 3.3 Transfer Entropy (Causality)

Measure how much the past predicts the future:

```
TE = H(X_t | X_{t-1}^{t-k}) - H(X_t | X_{t-1}^{t-k}, Y_{t-1}^{t-k})
```

**Interpretation**:
- Random: TE ≈ 0 (past doesn't help predict future)
- Causal structure: TE > 0

### 3.4 Statistical Complexity (Crutchfield's ε-machines)

The minimum information needed to optimally predict the future:

```
C_μ = H(causal states)
```

**Key property**: C_μ is zero for both perfectly ordered AND perfectly random sequences. It peaks for "interesting" structured sequences.

### 3.5 Logical Depth (Bennett)

The computational time needed to generate the sequence from its shortest description:

```
Depth = Time(UTM, Kolmogorov_description → sequence)
```

**Interpretation**:
- Trivial order: Low depth (easy to generate)
- Random: Low depth (just output random bits)
- Complex structure: High depth (requires computation)

## 4. Proposed "Structured Complexity Index" (SCI)

Combining the above into a single metric:

```
SCI = α · Hierarchy_Score + β · Causality_Score + γ · NonTriviality_Score
```

Where:

### 4.1 Hierarchy Score
```
Hierarchy = Variance(H(scale_k)) / Mean(H(scale_k))
```
- High when entropy varies across scales (structured)
- Low when entropy is constant (random or trivial)

### 4.2 Causality Score
```
Causality = Normalized_Transfer_Entropy
```
- High when past predicts future
- Low when sequence is memoryless

### 4.3 Non-Triviality Score
```
NonTriviality = 1 - |2·LZ - 1|   (peaks at LZ=0.5)
              × Pattern_Diversity  (number of unique patterns / theoretical max)
```
- Requires BOTH intermediate LZ AND diverse patterns

## 5. Expected Behavior

| Sequence Type | Hierarchy | Causality | NonTriviality | SCI |
|--------------|-----------|-----------|---------------|-----|
| All zeros | 0 | 0 | 0 | **0** |
| White noise | 0 | 0 | ~0.5 | **Low** |
| Simple periodic | Low | High | Low | **Low** |
| Fractal/self-similar | **High** | **High** | **Medium** | **High** |
| Physical simulation | **High** | **High** | **High** | **Very High** |

## 6. Implementation Plan

1. **Phase 1**: Implement Hierarchical Block Entropy
2. **Phase 2**: Implement Transfer Entropy
3. **Phase 3**: Implement Pattern Diversity metric
4. **Phase 4**: Combine into SCI with tunable weights
5. **Phase 5**: Validate on known sequences (random, periodic, fractal, physical)

## 7. Open Questions

1. What are the optimal weights (α, β, γ)?
2. Should we use φ (golden ratio) as a target for any metric?
3. How does SCI relate to Tononi's Integrated Information (Φ)?
4. Can we derive SCI from first principles of information theory?

## 8. References (To Investigate)

- Crutchfield & Young (1989): "Inferring Statistical Complexity"
- Grassberger (1986): "Toward a Quantitative Theory of Self-Generated Complexity"
- Tononi (2004): "An Information Integration Theory of Consciousness"
- Schreiber (2000): "Measuring Information Transfer"
- Bennett (1988): "Logical Depth and Physical Complexity"
- Bak, Tang & Wiesenfeld (1987): "Self-Organized Criticality"

