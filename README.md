# 🚀 ISH Agents System — Informational Singularity Hypothesis

This project implements a modular agent system to explore the **Informational Singularity Hypothesis (ISH)** developed by Iban Borràs. The system generates primitive binary sequences Φ and uses specialized agents to detect emergent patterns Pₖ and infer rules ωₖ from the Primordial Holographic Field.

## 📋 Project Description

The ISH proposes that all reality emerges from a primordial tension between "Absolute Nothingness" (0) and the "Absolute" (1), generating pure information that self-organizes into complex patterns. This agent system implements:

- **Level 0**: Generation of binary sequences Φ through iterative processes
- **Level 1**: Detection of recurrent patterns Pₖ(Φ)
- **Rule inference**: Discovery of emergent rules ωₖ
- **Validation**: Verification of consistency and robustness of rules

> **📖 Important**: For detailed technical documentation, see the `Documentation/` folder which contains analysis guides, variant specifications, and order metrics explanations.

## 🏗️ Project Structure

```
hsi_agents_project/
├── main.py                           # Main entry point
├── run_all_variants.py               # Run all ISH variants (B, D, E, F, G, H)
├── variant_A_control.py              # Random control for comparison
├── refresh_variant_from_cache.py     # Regenerate plots from cache
├── example_complete_level1_analysis.py # Full Level 1 analysis example
├── estimate_compression_requirements.py # Storage estimation tool
├── analyze_results.py                # Quick results analysis
├── Setup-virtual-envelop.ps1         # Setup/activation script (Windows)
├── config.json                       # Experiment configuration
├── requirements.txt                  # Python dependencies
│
├── 📚 Documentation/
│   ├── level1_analysis_guide.md      # Complete Level 1 technical guide
│   ├── order_metrics_explained.md    # Scientific explanation of metrics
│   ├── variants_spec.md              # Variant specifications (v33)
│   └── HSI_Variants_Analysis_For_Review.md # Analysis for peer review
│
├── level0/                           # Level 0 Generator
│   ├── generator.py                  # Φ sequence simulation
│   ├── streaming_collapse_engine.py  # Memory-efficient collapse
│   ├── hybrid_collapse_engine.py     # Hybrid collapse strategies
│   └── phi_snapshot_manager.py       # Snapshot management
├── level1/                           # Level 1 Analysis
│   ├── data_loader.py                # Load v33 structural data
│   └── example_usage.py              # Usage examples
├── agents/                           # Specialized agents
│   ├── pattern_detector.py           # Pₖ pattern detector
│   ├── structural_pattern_detector.py # Structural pattern analysis
│   ├── rule_inferer.py               # ωₖ rule inference
│   ├── validator.py                  # Rule validation
│   └── level1_orchestrator.py        # Level 1 orchestration
├── metrics/                          # Validation metrics
│   ├── pattern_metrics.py            # Pattern metrics
│   ├── rule_metrics.py               # Rule metrics
│   └── order_metrics.py              # Order emergence metrics
├── utils/                            # Auxiliary utilities
│   ├── visualization.py              # Visualization functions
│   ├── streaming_phi_loader.py       # Streaming data loader
│   └── bitarray_encoder.py           # Binary encoding utilities
├── data/                             # Generated experiment data
├── results/                          # Results (excluded from git)
└── docs/                             # Additional documentation
    └── GPU_ACCELERATION_PROPOSAL.md  # Future GPU integration
```

## 🔧 Installation and Dependencies

### System Requirements

- **Python 3.8 or higher** (recommended Python 3.11)
- **RAM Memory**: minimum 4GB (recommended 8GB for large experiments)
- **Operating System**: Windows 10/11, macOS, Linux

### Automatic Installation (Recommended)

#### Windows - Smart Script (Recommended)

```powershell
# Unified script that handles setup/activation
.\Setup-virtual-envelop.ps1
```

### Manual Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Python Dependencies

```bash
numpy>=1.21.0          # Scientific computing
scipy>=1.7.0           # Scientific algorithms
scikit-learn>=1.0.0    # Machine learning
matplotlib>=3.5.0      # Visualization
seaborn>=0.11.0        # Statistical visualization
```

## 🚀 System Usage

### Main Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `main.py` | Main entry point for complete agent pipeline | `python main.py` |
| `run_all_variants.py` | Run all ISH variants (B, D, E, F, G, H) with comparison | `python run_all_variants.py` |
| `variant_A_control.py` | Generate random control data for comparison | `python -m hsi_agents_project.variant_A_control -i 14` |
| `refresh_variant_from_cache.py` | Regenerate plots from cached data | `python -m hsi_agents_project.refresh_variant_from_cache -v B -i 18` |
| `example_complete_level1_analysis.py` | Complete Level 1 analysis example | `python example_complete_level1_analysis.py` |
| `estimate_compression_requirements.py` | Estimate storage needs for high iterations | `python estimate_compression_requirements.py --max-iteration 24` |
| `analyze_results.py` | Quick analysis of Level 1 results | `python analyze_results.py` |

### Basic Execution

```bash
# Run with default configuration (creates config.json if it doesn't exist)
python main.py
```

The system **always uses config.json** for configuration. If the file doesn't exist, it will be created automatically with default values.

### Configuration

Note: Current Level‑0 “basal‑pure” generator no longer uses dynamic collapse parameters (no masks, no seeds). Configuration remains for operational aspects (iterations, HW limits, outputs).

### Command Line Options

```bash
# Use different config file
python main.py --config my_config.json

# Specify output directory
python main.py --output my_results

# Disable visualizations
python main.py --no-viz

# Silent mode
python main.py --quiet
```

### Automatic Compression

The system includes **intelligent compression management** for large sequences:

- Threshold: Automatically activates when sequences exceed `memory_threshold`
- Format: Saves as `.bin.gz` + `.json` metadata
- Transparency: Loading/saving is completely transparent
- Scalability: Enables experiments for high iterations


## 🔍 System Components

### 1. Φ Generator (Level 0)

Implements the fundamental iterative process of ISH with automatic compression management:

```python
from level0 import simulate_phi

phi, snapshots, metadata = simulate_phi(
    max_iterations=20,
    collapse_rule="AND",
    phi_target=1.618,
    use_compression=True,
    memory_threshold=100000000
)
```

**Features:**
- Configurable collapse rules (AND, ASYM, CUSTOM)
- Automatic detection of convergence towards φ (1.618)
- **Intelligent compression** for sequences > 100MB
- Snapshot generation for temporal analysis
- **Scalable to iteration 32** with compression
- **φ-alignment validation** for golden ratio tendency

### 2. Pattern Detector

Identifies recurrent patterns using multiple methods:

```python
from agents import PatternDetector

detector = PatternDetector(min_pattern_length=3)
patterns = detector.detect_patterns(phi_sequence)
```

**Detection methods:**
- Sliding windows with hashing
- Block segmentation and clustering
- Spectral analysis (FFT)
- Self-similarity detection

### 3. Rule Inferer

Discovers emergent rules from patterns:

```python
from agents import RuleInferer

inferer = RuleInferer(context_window=5)
rules = inferer.infer_rules(patterns, phi_sequence)
```

**Rule types:**
- Markov transitions
- Context rules
- Composition rules
- Periodicity rules

### 4. Validator

Verifies rule robustness:

```python
from agents import Validator

validator = Validator()
results = validator.validate_rules(rules, patterns, test_sequences)
```

**Validation metrics:**
- Reproducibility
- Temporal stability
- Predictive precision
- Rule complexity

## 🧩 Absolute Token (ABS) Modes and Variants

The outer decay frame is `(Accα)ABS`, where ABS controls how the Absolute integrates prior states:
- ABS = `1` (symmetric baseline)
- ABS = `10` (directional 1→0)
- ABS = `01` (directional 0→1)

Configure via:
- `config.json → output.absolute_token`
- CLI override (generator): `--abs-mode 1|10|01`

Snapshots and outputs are segregated by variant and ABS:
- `results/phi_snapshots/var_{VAR}_abs{ABS}/phi_iterN.bin.gz`
- `results/var_{VAR}/phi_final.txt`
- Visualizations include `_abs{ABS}` in filenames

## 📑 Surviving Variants (v33) — ISH v32 Aligned

**Version:** v33 (November 2025)
**Status:** 6 active variants (A and C eliminated for incompatibility with ISH v32)

### ⭐ Core Variants (Tier 1)

**Variant B — Stratified Baseline (Gold Standard)**
- **Algorithm:** Inside→out collapse + final global simplify
- **Properties:** Preserves stratified order, accumulates ALL micro-states
- **Scientific Value:** Fully compatible with v32 principles, validated as ontologically correct
- **Why it survives:** Gold standard for order emergence through stratified containment

**Variant D — Minimal Asymmetry**
- **Algorithm:** Asymmetric rule (10→∅, 01→0) in stratified passes
- **Properties:** Breaks 0↔1 symmetry, tests robustness of order emergence
- **Scientific Value:** Tests whether order is robust to symmetry-breaking
- **Why it survives:** Legitimate hypothesis about asymmetric collapse effects

### 🧪 Experimental Variants (Tier 2)

**Variant E — Ordered Two-Pass Removal**
- **Algorithm:** Remove all 01, then all 10, then compress runs
- **Properties:** Enforces deterministic pass order (more aggressive than base)
- **Scientific Value:** Tests whether pass order affects emergent structure
- **Why it survives:** Explores sequence-dependent collapse dynamics

**Variant F — Hybrid Stabilization**
- **Algorithm:** Stabilize inside→out, then single global pass
- **Properties:** Combines stratified stabilization with final global closure
- **Scientific Value:** Tests hybrid strategy (positioned between B and A)
- **Why it survives:** Explores different closure strategies

**Variant G — Raw Stratified Structure (ABS-aware)**
- **Algorithm:** Strictly stratified (NO final global simplify)
- **Properties:** Preserves raw stratified outcome, may contain parentheses
- **Scientific Value:** Studies persistence of inner order before cleaning
- **Why it survives:** Unique value in preserving raw structure for analysis

**Variant H — Continuous Feedback (ABS-aware)**
- **Algorithm:** Temporal with per-tick global soft-closure
- **Properties:** Combines fine temporal resolution with continuous global feedback
- **Scientific Value:** Tests whether continuous feedback affects order emergence
- **Why it survives:** Explores feedback-driven collapse dynamics

### 🎲 Control Variant (v33.1)

**Variant A — Random Control (PRNG/CSPRNG Baseline)** *(REPURPOSED)*
- **Algorithm:** Pseudo-random bit generation (Mersenne Twister or cryptographic RNG)
- **Purpose:** Provides a **scientific control** for comparing ISH-generated structures against random baselines
- **Properties:** No collapse process, no stratified order — pure statistical randomness
- **Scientific Value:** Any pattern found in ISH variants (B, D, E, F, G, H) but **absent** in Variant A indicates genuine emergent structure, not algorithmic artifacts

**Usage:**
```powershell
# Generate random control matching size of Variant B at iteration 14
python -m hsi_agents_project.variant_A_control --iterations 14 --match-variant B

# Generate with fixed seed for reproducibility
python -m hsi_agents_project.variant_A_control --iterations 14 --bits 5000000 --seed 42

# Use cryptographic RNG (maximum entropy, no seed)
python -m hsi_agents_project.variant_A_control --iterations 14 --bits 5000000 --csprng

# Visualize and compare with ISH variants
python -m hsi_agents_project.refresh_variant_from_cache -v A -i 14
```

**Expected Differences (A vs ISH variants):**
| Metric | Variant A (Random) | ISH Variants (B, D, etc.) |
|--------|-------------------|---------------------------|
| Hilbert heatmap | Uniform noise | Structured patterns ("mesh") |
| FFT spectrum | Flat (white noise) | Peaks at specific frequencies |
| Power spectrum β | ~0 (white noise) | Negative β (unusual structure) |
| Ones ratio | ~0.500 (statistical) | May deviate systematically |

---

### 🔬 Experimental Validation: A vs B Comparison (Dec 2024)

The following results demonstrate that ISH collapse generates **genuine emergent structure**, not algorithmic artifacts:

| Metric | **Variant A (PRNG)** | **Variant B (ISH)** | Interpretation |
|--------|---------------------|---------------------|----------------|
| **β (power spectrum)** | **-0.013** | **-0.813** | A = white noise; B = blue spectrum (anti-correlation) |
| **R²** | **0.000** | 0.119 | A = no fit; B = measurable structure |
| **Hilbert map** | Uniform noise | Structured mesh | B shows spatial organization |
| **FFT** | Flat spectrum | Peaks near Nyquist | B has high-frequency dominance |

**Key Scientific Conclusions:**

1. **β ≈ 0 confirms white noise**: The PRNG (Mersenne Twister, seed=42) generates exactly what is expected — a completely flat spectrum with no correlations

2. **β = -0.813 is REAL structure**: The blue spectrum in Variant B is **not an artifact** — it is a genuine signature of the ISH collapse process

3. **The Hilbert mesh is information**: The "grid" pattern visible in B represents organized information, while A is pure noise

4. **Physical interpretation**: The stratified collapse (eliminating "10" and "01" pairs) generates **anti-correlations** — each bit tends to be the opposite of its neighbor, producing the characteristic blue spectrum

This comparison provides **empirical evidence** that the ISH collapse mechanism produces measurable, distinguishable structure that cannot arise from random processes.

---

**Epistemological Note:** True randomness is philosophically problematic — if ISH is correct, even quantum randomness might have underlying structure. The comparison is therefore **relative** (different patterns) rather than **absolute** (structure vs. no-structure).

### ❌ Eliminated Variants (v33)

**~~Variant A — Global Collapse~~** *(ELIMINATED, then REPURPOSED as Random Control)*
- **Original reason for elimination:** Fundamentally incompatible with stratified order principle
- **Current status:** Repurposed as PRNG control baseline (see above)

**Variant C — Temporal Cascade** *(ELIMINATED)*
- **Reason:** Violates micro-state reality principle OR redundant with Variant B
- **Why eliminated:** Intentionally discards micro-states (incompatible) or identical to B if fixed
- **v32 conflict:** "Intermediate states are ontologically real" — C discards them

### 📖 Full Documentation

See `Documentation/variants_spec.md` for:
- Complete algorithmic definitions
- Scientific justification for each variant
- Implementation cross-reference
- ISH v32 alignment analysis
### Spectral analysis configuration

- Method (config.json → output.spectral_method): `auto` (default), `welch`, `sampling`, `prefix`
- Window size (output.fft_window_size): samples per window (default 1,048,576)
- Overlap (output.fft_overlap): 0.0–0.9 (default 0.5)
- Max windows (output.fft_max_windows): cap to limit cost (default 64)

Auto mode selects:
- welch if ≥ 4 windows are available
- sampling if ≥ 1 window but < 4
- prefix if no window fits (reads a direct prefix and performs FFT)

Logs report the decision, e.g.:
- [plot] spectral auto: method=welch, nwin≈8, w=1,048,576 var=B i=15

Rationale: exposing ABS allows testing whether φ‑alignment and emergent order are robust or depend on boundary orientation.

## 🔬 Level 1 Analysis (v33 Structural Format)

**Version:** v33 (January 2025)
**Status:** Fully operational with structural pattern detection and order metrics

### Overview

Level 1 implements advanced analysis of Level 0 data to detect **emergent order** in the informational structure. The v33 format preserves the complete structural information (parentheses marking Absolutes) that is critical for understanding how order emerges from the primordial tension between 0 and 1.

### Key Concepts

**v33 Structural Format:**
- Preserves parentheses `()` that mark Absolute boundaries
- Enables analysis of **stratified containment** (how Absolutes nest within each other)
- Achieves ~99.7% compression while maintaining full structural information
- Backward compatible with v32 format (observable bits only)

**Emergent Order (not φ-convergence):**
- ISH v32 shifted focus from golden ratio convergence to **order as fundamental principle**
- "Order is the only way Nothingness can contradict itself"
- Level 1 detects and quantifies this emergent order through multiple metrics

### Components

#### 1. Data Loader

Load Level 0 data with automatic v32/v33 format detection:

```python
from level1 import load_phi_for_level1

# Load both structural and observable Φ
phi_structural, phi_observable, metadata = load_phi_for_level1(
    data_dir="results/var_B",
    iteration=10,
    return_structural=True,
    return_observable=True,
    return_metadata=True
)

# phi_structural: "(((01)1)0((10)0)1)" - with parentheses
# phi_observable: "01101001" - clean binary
```

#### 2. Structural Pattern Detector

Detect patterns in the hierarchical structure:

```python
from agents import StructuralPatternDetector

detector = StructuralPatternDetector()
patterns = detector.detect_structural_patterns(phi_structural)

# Pattern types:
# - Nesting patterns: depth distribution, transitions
# - Containment patterns: what each Absolute contains
# - Stratified order patterns: entropy at each depth level
```

#### 3. Order Metrics

Quantify emergent order with composite metrics:

```python
from metrics import OrderMetrics

# Calculate all order metrics
metrics = OrderMetrics.calculate_all_metrics(phi_structural)

# Key metric: Order Emergence Index (0-1)
order_index = metrics['order_emergence']['order_index']
# > 0.5 indicates strong emergent order

# Components:
# - Depth organization: how well-structured the nesting is
# - Entropy gradient: how entropy changes with depth
# - Containment regularity: regularity of Absolute contents
# - Balance score: structural balance quality
```

### Order Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Order Emergence Index** | 0-1 | Overall emergent order (>0.5 = strong) |
| Depth Organization | 0-1 | Nesting structure quality (>0.5 = organized) |
| Entropy Gradient | 0-1 | Entropy change smoothness (>0.5 = smooth) |
| Containment Regularity | 0-1 | Content regularity (>0.5 = regular) |
| Depth Entropy | 0-log₂(max_depth) | Depth diversity (low = concentrated) |
| Depth Gini | 0-1 | Depth inequality (low = uniform) |

### Complete Analysis Pipeline

```python
from level1 import load_phi_for_level1
from agents import PatternDetector
from metrics import OrderMetrics, create_order_report

# 1. Load data
phi_s, phi_o, meta = load_phi_for_level1("results/var_B", 10)

# 2. Detect patterns (observable + structural)
detector = PatternDetector(enable_structural_analysis=True)
obs_patterns = detector.detect_patterns(phi_o, phi_s)
struct_patterns = detector.get_structural_patterns()

# 3. Calculate order metrics
order_metrics = OrderMetrics.calculate_all_metrics(phi_s)

# 4. Generate comprehensive report
report = create_order_report(phi_s, output_path="results/order_report.json")

print(f"Order Index: {order_metrics['order_emergence']['order_index']:.3f}")
print(f"Max Depth: {order_metrics['nesting_depth']['max_depth']}")
print(f"Structural Patterns: {len(struct_patterns)}")
```

### Example Usage Script

See `example_complete_level1_analysis.py` for a complete end-to-end analysis pipeline that:
1. Loads v33 data with structural information
2. Extracts basic structural metrics
3. Detects observable and structural patterns
4. Calculates all order metrics
5. Generates comprehensive JSON report
6. Displays summary statistics

### Documentation

For detailed technical documentation, see:
- **`Documentation/level1_analysis_guide.md`** - Complete technical guide
- **`Documentation/order_metrics_explained.md`** - Scientific explanation of metrics
- **`Documentation/variants_spec.md`** - Variant specifications and rationale
- **`level1/README.md`** - Quick reference for Level 1 module

### Scientific Value

Level 1 analysis enables:
- **Quantification of emergent order** in informational structures
- **Detection of hierarchical organization** through nesting analysis
- **Validation of ISH v32 hypothesis** ("order as fundamental principle")
- **Comparison between variants** to identify which best exhibits order emergence
- **Temporal analysis** of how order evolves across iterations

## 💾 Data Management and Compression

### Automatic Compression Strategy

The system implements an intelligent compression strategy based on `level0_data_storage.md`:

| Iteration | Estimated Size | Storage Method | Compression Ratio |
|-----------|----------------|----------------|-------------------|
| 16 | ~41M bits | Memory | N/A |
| 18 | ~330M bits | **Compressed** | ~85% |
| 20 | ~1.5GB | **Compressed** | ~85% |
| 24 | ~23GB | **Compressed** | ~85% |
| 32 | ~2TB | **Compressed** | ~85% |

### Storage Structure

```
data/
├── phi_snapshots/
│   ├── phi_iter18.bin.gz      # Compressed sequence
│   ├── phi_iter18.json        # Metadata
│   ├── phi_iter20.bin.gz
│   └── phi_iter20.json
└── compression_summary.json   # Overall statistics
```

### Compression Tools

```bash
# Estimate storage requirements for high iterations
python estimate_compression_requirements.py --max-iteration 32

# Detailed analysis with breakdown
python estimate_compression_requirements.py --max-iteration 28 --detailed
```

## 📊 Metrics and Validation

### Metrics for Patterns Pₖ

- **Recurrence**: Frequency of occurrence within the Φ sequence
- **Compressibility**: Measure of internal structure and regularity
- **Self-similarity**: Fractal properties and self-organization
- **Spectral signature**: Analysis of dominant frequencies

### Metrics for Rules ωₖ

- **Reproducibility**: Consistency in repeated applications
- **Stability**: Temporal robustness between iterations
- **Precision**: Predictive capacity on new sequences
- **Complexity**: Interpretability and rule simplicity

### ISH-Specific Metrics

- **φ-alignment (Golden Tendency)**: Evaluates whether generated structures show fractal dimension or internal proportions close to φ (1.618), indicating emergent order coherent with the hypothesis of convergence towards the golden ratio
- **Informational Convergence**: Measure of how Φ sequences evolve towards stable patterns
- **Holographic Field Coherence**: Assessment of consistency of Pₖ patterns as manifestations of the primordial field

## 🎨 Visualizations
### Spectral Analysis (Welch Streaming)

The project implements streaming spectral analysis directly from compressed Φ snapshots (.bin.gz) using Welch’s method. This is robust for large iterations without loading full sequences into RAM.

- Default method: Welch (averaged PSD)
- Default FFT prefix cap: 2,000,000 bits (HSI_FFT_PREFIX)
- Configurable via config.json → output:
  - spectral_method: "welch" | "prefix" | "sampling" (planned)
  - fft_window_size (default: 1,048,576)
  - fft_overlap (default: 0.5)
  - fft_max_windows (default: 64)
  - progress_logging (default: true)

During long reads, the runner prints percentage and ETA; during Welch, it prints windows processed and ETA.

#### Performance Profiles

- Fast
  - spectral_method: prefix
  - HSI_FFT_PREFIX: 1,000,000
  - Notes: quickest, suitable for quick checks; less stable on very long sequences.

- Balanced (default)
  - spectral_method: welch
  - fft_window_size: 1,048,576
  - fft_overlap: 0.5
  - fft_max_windows: 64
  - HSI_FFT_PREFIX: 2,000,000
  - Notes: good stability and speed balance; recommended for most machines.

- High-Quality
  - spectral_method: welch
  - fft_window_size: 1,048,576
  - fft_overlap: 0.5
  - fft_max_windows: 128
  - HSI_FFT_PREFIX: 5,000,000
  - Notes: best quality; higher CPU time.

- Sampling (very fast and robust)
  - spectral_method: sampling
  - fft_window_size: 1,048,576
  - fft_max_windows: 32 (non-overlapping)
  - HSI_FFT_PREFIX: 2,000,000 (or 5,000,000)
  - Notes: averages K randomly spaced (non-overlapping) windows; good representativeness at low cost.



The system can generate:

- Φ sequence plots
- 2D Hilbert maps
- Pattern distributions
- Rule performance charts
- Comparison plots across variants (if matplotlib is available)

## 📁 Results

Results are organized in:

```
results/
├── phi_sequences/          # Generated Φ sequences
├── patterns/               # Detected patterns
├── rules/                  # Inferred rules
├── validation/             # Validation results
├── visualizations/         # Charts and dashboards
```

Reports for Level 0 generator are now saved under:

```
hsi_agents_project/reports/
```

## 🔬 Usage Examples

### Quick Start

```bash
# Run with default configuration (auto-creates config.json)
python main.py
```

### Basic Experiment (Programmatic)

```python
# Generate Φ sequence with compression
phi, _, metadata = simulate_phi(
    max_iterations=16,
    use_compression=True,
    memory_threshold=100000000
)

# Check compression status
if metadata.get('compression_activated'):
    print(f"Compression used: {metadata['compression_summary']['total_size_mb']:.1f} MB saved")

# Detect patterns
detector = PatternDetector()
patterns = detector.detect_patterns(phi)

# Infer rules
inferer = RuleInferer()
rules = inferer.infer_rules(patterns, phi)

# Validate with φ-alignment
validator = Validator()
results = validator.validate_rules(rules, patterns, [phi])
```

### High-Performance Experiment

```python
# Configure for high iterations with compression
config = {
    "phi_generation": {
        "max_iterations": 24,
        "use_compression": True,
        "memory_threshold": 50000000  # 50MB threshold
    },
    "output": {
        "create_visualizations": False,  # Disable for speed
        "verbose": False
    }
}

# Run experiment
phi, _, metadata = simulate_phi(**config["phi_generation"])
```

### Convergence Analysis

```python
# Search for convergence towards φ (1.618)
phi, _, metadata = simulate_phi(
    max_iterations=20,
    phi_target=1.618,
    collapse_rule="AND",
    use_compression=True
)

if metadata['convergence_reached']:
    print("Convergence towards φ achieved!")

# Check φ-alignment metric
if 'phi_alignment' in metadata:
    alignment = metadata['phi_alignment']
    print(f"φ-alignment score: {alignment:.4f}")
```

### Environment Configuration

Copy `.env.example` to `.env` in the repository root (or under `hsi_agents_project/`) and adjust values for your machine.
The runner will auto-load `.env`; the generator also reads these unless a CLI flag overrides them.

**Available environment variables:**

```bash
# Iterations and variant
ISH_ITERATIONS=20
ISH_VARIANT_CODE=B

# Logging and timing
ISH_LOG_EVERY=1
ISH_ENABLE_TIMING=1
ISH_COMPRESS_LOG=1

# Compression level (gzip)
ISH_COMPRESSION_LEVEL=5

# Cleanup snapshots at the end of a run (keeps the last 2 only)
ISH_CLEANUP_KEEP_LAST=2
```
```

## 🤝 Contributions

This project is part of research on the Informational Singularity Hypothesis. Contributions are welcome:

1. Fork the repository
2. Create branch for new functionality
3. Implement changes with tests
4. Create pull request with detailed description

## 📚 References and Documentation

### Technical Documentation

| Document | Description |
|----------|-------------|
| `Documentation/level1_analysis_guide.md` | Complete Level 1 technical guide |
| `Documentation/order_metrics_explained.md` | Scientific explanation of order metrics |
| `Documentation/variants_spec.md` | Variant specifications (v33) |
| `Documentation/HSI_Variants_Analysis_For_Review.md` | Analysis for peer review |
| `docs/GPU_ACCELERATION_PROPOSAL.md` | Future GPU integration proposal |
| `level1/README.md` | Quick reference for Level 1 module |

### Code Documentation

- `README.md` - This document (main guide)
- Detailed docstrings in all Python modules
- Inline comments explaining ISH theory

## 📄 License

This project is developed by Iban Borràs in collaboration with Augment Agent (Claude Sonnet 4) as part of research on the Informational Singularity Hypothesis.

## 🔮 Roadmap and Future Developments

### Planned Technological Expansions

#### 🌌 **Symbolic Quantum Computing**
- **Integration with quantum simulators** to explore ISH in Hilbert spaces
- **Quantum algorithms** for detecting Pₖ patterns in superposition
- **Informational entanglement** as manifestation of the Primordial Holographic Field

#### 🧠 **Informational Consciousness Models**
- **Application to consciousness theories** based on integrated information (IIT)
- **Modeling of cognitive processes** emerging from 0↔1 tension
- **Connections with fundamental physics** and theories of everything

#### 🎨 **Advanced Visualization**
- **3D rendering** of H(Φ) structures from the Holographic Field
- **Immersive visualization** of temporal evolution of Ψ
- **Virtual reality** to explore emergent pattern space
- **Real-time animations** of convergence towards φ

### Scientific Extensions

#### 🔬 **Experimental Validation**
- **Comparison with real physical data** (fundamental constants, cosmic structures)
- **Testable predictions** derived from ISH
- **Collaboration with theoretical physicists** for empirical validation

#### 📊 **Advanced Analysis**
- **Machine Learning** to discover non-obvious ISH patterns
- **Neural networks** trained on Φ sequences
- **Time series analysis** of informational evolution

#### 🌐 **Interdisciplinary Applications**
- **Biology**: ISH patterns in DNA and evolution
- **Cosmology**: Large-scale structures and ISH
- **Mathematics**: Connections with number theory and fractal geometry

### System Improvements

#### ⚡ **Performance Optimization**
- **Parallelization** of pattern detection algorithms
- **GPU acceleration** for fractal dimension calculations
- **Distributed algorithms** for large-scale experiments

#### 🔧 **Advanced Features**
- **Visual editor** for custom collapse rules
- **Web interface** for remote experiments
- **REST API** for integration with other systems
- **Database** of discovered patterns and rules

## 🆘 Support

For problems or questions:

1. **Configuration**: Edit `config.json` - it's self-documenting with built-in help
2. **Documentation**: Review files in `Documentation/` folder
3. **Compression**: Use `estimate_compression_requirements.py` for storage planning
4. **Research**: Contact the author for scientific questions about ISH

### Troubleshooting Tools

```bash
# Check compression requirements
python estimate_compression_requirements.py --max-iteration 24

# Test Level 0 generator (with progress and compression level)
python -m hsi_agents_project.level0.generator -v B -i 4 --log-every 1 --compression-level 5

# Verify environment setup
.\Setup-virtual-envelop.ps1
```

### Additional Resources

- **Issues**: Report technical problems on GitHub
- **Discussions**: Scientific debates about ISH
- **Wiki**: Extensive documentation (future)
- **Papers**: Publications derived from the project
- **Config Help**: All parameters documented in `config.json`

---

**Note**: This system is under active development as part of pioneering scientific research. ISH represents a new frontier in understanding informational reality. Results may open unexpected paths towards fundamental discoveries about the nature of existence.
