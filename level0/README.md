# Level 0: Primitive Informational State Generator

This module implements the core iterative mechanism of the Informational Singularity Hypothesis (ISH). It is responsible for generating the primitive informational field $\Phi$ through the recursive tension between Nothingness (0) and Absolute Nothingness (1).

## Key Components

### 1. `generator.py`
The main entry point for generating sequence data.
- **Function**: Implements the iterative collapse algorithm.
- **Key Logic**: Manages the expansion (`0 -> 0 1`) and collapse (AND operation) phases.
- **Usage**: Typically called via `level0_generate.py` in the root directory.

### 2. Collapse Engines
To handle exponentially growing sequences, we use specialized engines:
- **`streaming_collapse_engine.py`**: Optimizes memory usage by processing sequences in chunks. Essential for iterations > 20.
- **`hybrid_collapse_engine.py`**: Implements hybrid stabilization strategies (for Variant F).

### 3. Managers
- **`phi_snapshot_manager.py`**: Handles storage and retrieval of $\Phi$ snapshots (compressed .bin.gz files).
- **`accumulation_manager.py`**: Manages the accumulation of states for variants that require history tracking.
- **`disk_space_manager.py`**: Monitors disk usage to prevent overflow during large experiments.

### 4. `control_variants.py`
Implements the generation logic for Control Variants:
- **A**: Random (Mersenne Twister)
- **J**: Digits of $\pi$
- **K**: Rule 30 Cellular Automaton
- **L**: Logistic Map Chaos
- **M**: Fibonacci Word (Positive Control)

## Usage

This module is primarily used by the root script `level0_generate.py`. Direct usage is for advanced debugging or custom pipeline integration.

```python
from level0.generator import simulate_phi

# Generate 15 iterations of Variant B
simulate_phi(
    variant="B",
    max_iterations=15,
    use_compression=True
)
```
