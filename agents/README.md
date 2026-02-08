# Agents: Pattern Detection and Rule Inference

This module contains the "agents" that perform Level 1 analysis on the primitive $\Phi$ field. In the HSI framework, these agents represent the emergent mechanisms that detect regularities (The Primordial Holographic Field).

## The Agent Suite

### 1. `pattern_detector.py`
Scans binary sequences to find recurrent structures.
- **Algorithms**: Sliding window, rolling hash, and block segmentation.
- **Output**: A dictionary of patterns {sequence: count}.
- **Adaptive**: Switches to sampling mode for very large sequences (>100M bits).

### 2. `rule_inferer.py`
Deduces the local rules that govern the transitions between bits.
- **Context Rules**: $P(bit | context)$.
- **Markov Rules**: Probabilistic transition matrices.
- **Output**: A set of rules $\omega_k$ with confidence scores.

### 3. `validator.py`
Verifies the robustness and predictive power of inferred rules.
- **Metrics**: Accuracy, coverage, and stability.
- **Tests**: Runs rules against unseen sequence segments.

### 4. `structural_pattern_detector.py`
Specialized detector for hierarchical and nested structures.
- **Focus**: Parentheses matching, tree-like structures.
- **Usage**: Used in `level1_nesting_tree.py`.

### 5. `geometric_analyzer.py`
Experimental agent for Level 2 analysis.
- **Task**: Computes geometric operators (Curvature $\mathcal{R}$, Torsion $\mathcal{T}$).
- **Status**: Research prototype.

## Orchestration

The **`level1_orchestrator.py`** coordinates these agents to perform a complete analysis pipeline:
1. Load $\Phi$ data.
2. Run Pattern Detector.
3. Pass patterns to Rule Inferer.
4. Validate rules.
5. Generate report.

## Usage

Agents are typically invoked via `level1_analyze_patterns.py`.

```python
from agents import Level1Orchestrator

orchestrator = Level1Orchestrator()
results = orchestrator.run_analysis(
    variant="B",
    iteration=15,
    config={"min_pattern_len": 10}
)
```
