# ISH Agents System (Informational Singularity Hypothesis)

This project implements a modular agent system to explore the **Informational Singularity Hypothesis (ISH)** developed by Iban Borràs. The system generates primitive binary sequences Φ and uses specialized agents to detect emergent patterns Pₖ and infer rules ωₖ from the Primordial Holographic Field.

## Project Description

The ISH proposes that all reality emerges from a primordial tension between "Nothingness" (0) and the "Absolute Nothingness" (1), generating pure information that self-organizes into complex patterns. This agent system implements:

- **Level 0**: Generation of binary sequences Φ through iterative processes
- **Level 1**: Detection of recurrent patterns Pₖ(Φ)
- **Rule inference**: Discovery of emergent rules ωₖ
- **Validation**: Verification of consistency and robustness of rules

> **📖 Important**: For detailed technical documentation, see the `Documentation/` folder which contains analysis guides, variant specifications, and order metrics explanations.

## Project Structure

```
hsi_agents_project/
│
├── ════════════════════════════════════════════════════════════
│   MAIN SCRIPTS (run from project root)
├── ════════════════════════════════════════════════════════════
├── level0_generate.py                # Level 0: Generate Φ for ALL variants (A-M)
├── level0_refresh_plots.py           # Level 0: Regenerate plots from cache
├── level0_estimate_storage.py        # Level 0: Storage estimation tool
├── level1_analyze_patterns.py        # Level 1: Pattern detection and rule inference
├── level1_emergence_index.py         # Level 1: Emergence Index (Level 2 potential)
├── level1_sci_icc.py                 # Level 1: SCI & ICC metrics calculation
├── level1_trend_analysis.py          # Level 1: Trend analysis + extrapolation
├── level1_visualize.py               # Level 1: Generate publication figures
├── level1_view_results.py            # Level 1: View/analyze JSON results
│
├── ════════════════════════════════════════════════════════════
│   LEVEL 2: ADVANCED COMPLEXITY ANALYSIS
├── ════════════════════════════════════════════════════════════
├── level2_complexity_analysis.py     # LZ, Shannon, Logical Depth, BDM comparison
├── level2_takens_tda.py              # Takens embedding + Topological Data Analysis
├── level2_transfer_entropy.py        # Multi-scale Transfer Entropy (causal flow)
│
├── ════════════════════════════════════════════════════════════
│   HIDDEN φ DISCOVERY TOOLS (Level 1+)
├── ════════════════════════════════════════════════════════════
├── level1_scale_ratios.py            # Phase 1: MSRA (find φ in scale proportions)
├── level1_nesting_tree.py            # Phase 1: PCSA (tree branching ratios)
├── level1_hilbert_multires.py        # Phase 1: HMRC (Hilbert multi-resolution)
├── level1_deep_analysis.py           # Phase 2: Wavelet, Recurrence, LZ complexity
├── level1_topological.py             # Phase 3: TPA (topological persistence)
├── level1_geometric.py               # Phase 3: Γ, 𝓡, 𝓣, 𝓔 operators
│
├── Setup-virtual-envelop.ps1         # Setup/activation script (Windows)
├── config.json                       # Experiment configuration
├── requirements.txt                  # Python dependencies
│
├── 📚 Documentation/
│   ├── level1_analysis_guide.md      # Complete Level 1 technical guide
│   ├── level1_figures_plan.md        # Figure generation plan
│   ├── order_metrics_explained.md    # Scientific explanation of metrics
│   ├── variants_spec.md              # Variant specifications
│   ├── phi_emergence_mechanism.md    # φ emergence analysis (E vs B vs I)
│   ├── geometric_analysis_plan.md    # Geometric analysis plan
│   ├── hidden_phi_discovery_roadmap.md # Hidden φ research roadmap
│   ├── GPU_ACCELERATION_PROPOSAL.md  # Future GPU integration
│   └── HSI_Variants_Analysis_For_Review.md # Analysis for peer review
│
├── level0/                           # Level 0 Generator modules
│   ├── generator.py                  # Φ sequence simulation
│   ├── streaming_collapse_engine.py  # Memory-efficient collapse
│   ├── hybrid_collapse_engine.py     # Hybrid collapse strategies
│   ├── phi_snapshot_manager.py       # Snapshot management
│   └── accumulation_manager.py       # Accumulation management
├── level1/                           # Level 1 Analysis modules
│   ├── data_loader.py                # Load structural data
│   ├── level1_figures.py             # Figure generation (core)
│   └── example_usage.py              # Usage examples
├── agents/                           # Specialized agents
│   ├── pattern_detector.py           # Pₖ pattern detector
│   ├── structural_pattern_detector.py # Structural pattern analysis
│   ├── rule_inferer.py               # ωₖ rule inference
│   ├── validator.py                  # Rule validation
│   └── level1_orchestrator.py        # Level 1 orchestration (core)
├── metrics/                          # Validation metrics
│   ├── pattern_metrics.py            # Pattern metrics
│   ├── rule_metrics.py               # Rule metrics
│   ├── order_metrics.py              # Order emergence metrics
│   └── emergence_index.py            # Emergence Index (Level 2 potential)
├── utils/                            # Auxiliary utilities
│   ├── visualization.py              # Visualization functions
│   ├── streaming_phi_loader.py       # Streaming data loader
│   ├── bitarray_encoder.py           # Binary encoding utilities
│   └── progress.py                   # Progress indicators
└── results/                          # Results (excluded from git)
```

## Installation and Dependencies

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

## System Usage

### Main Scripts

All scripts can be run directly from the project root:

| Script | Purpose | Usage |
|--------|---------|-------|
| `level0_generate.py` | Generate Φ sequences for ALL variants (A-M) | `python level0_generate.py --variant B -i 18` |
| `level0_refresh_plots.py` | Regenerate plots from cached data | `python level0_refresh_plots.py -v B -i 18` |
| `level0_estimate_storage.py` | Estimate storage needs for high iterations | `python level0_estimate_storage.py --max-iteration 24` |
| `level1_analyze_patterns.py` | Pattern detection and rule inference | `python level1_analyze_patterns.py -v B -i 18 --report` |
| `level1_emergence_index.py` | Calculate Emergence Index (Level 2 potential) | `python level1_emergence_index.py -v B -i 18` |
| `level1_sci_icc.py` | Calculate SCI & ICC metrics from emergence data | `python level1_sci_icc.py --from-emergence results/emergence_*.json` |
| `level1_trend_analysis.py` | Trend analysis across iterations with extrapolation | `python level1_trend_analysis.py --variants B E I --plot` |
| `level1_visualize.py` | Generate publication-quality figures | `python level1_visualize.py --all` |
| `level1_view_results.py` | View/analyze Level 1 results | `python level1_view_results.py results/level1_*.json` |

### Level 2: Advanced Complexity Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| `level2_complexity_analysis.py` | Compare variants using algorithmic complexity (LZ, Shannon, Logical Depth, BDM) | `python level2_complexity_analysis.py --variants A B D E --iteration 18` |
| `level2_takens_tda.py` | Takens embedding + Persistent Homology (Betti numbers β₀, β₁, β₂) | `python level2_takens_tda.py --variant B --iteration 18 --visualize` |
| `level2_transfer_entropy.py` | Multi-scale Transfer Entropy (detect causal flow direction) | `python level2_transfer_entropy.py --variant B --iteration 18 --plot` |

### Hidden φ Discovery Tools

These tools implement the research roadmap for discovering hidden φ in variant structures (see `Documentation/hidden_phi_discovery_roadmap.md`).

#### Phase 1: Scale & Structure Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| `level1_scale_ratios.py` | **MSRA** — Find φ in ratios between block metrics at different scales (2^k) | `python level1_scale_ratios.py -v B -i 15` |
| `level1_nesting_tree.py` | **PCSA** — Analyze parentheses tree for φ in branching ratios | `python level1_nesting_tree.py -v B -i 15` |
| `level1_hilbert_multires.py` | **HMRC** — Compare Hilbert maps at multiple resolutions | `python level1_hilbert_multires.py -v B -i 15` |

#### Phase 2: Deep Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| `level1_deep_analysis.py` | Unified deep analysis: Wavelet φ-band, Recurrence plots, LZ complexity | `python level1_deep_analysis.py -v B -i 15` |

**Key Discovery (Dec 2025):** Variant B shows **LZ φ-scaling** with distance 0.010 from φ — the proportions of complexity between scales follow the golden ratio! Controls (A, K, L) show ratio ≈ 1.0 (random). This validates that **φ is hidden in HSI's proportions**.

##### Comparative Multi-Variant Analysis (NEW)

Analyze multiple variants in one run with comparative summary:

```bash
# Compare HSI variants vs controls (generates comparative report)
python level1_deep_analysis.py --variants A,B,F,K,L -i 15 --max-bits 100000 --analysis lz

# Output includes:
#   - Per-variant LZ ratios
#   - Best-match constant test (φ, √2, e, π/3, etc.)
#   - HSI vs Control mean comparison
#   - Cohen's d effect size
```

##### Multi-Scale Mode (100G+ bits)

For analyzing very large datasets (100 gigabits or more), the script supports **Nyquist-based multi-scale subsampling**:

```bash
# Analyze ALL bits using multi-scale subsampling (~12M samples in RAM)
python level1_deep_analysis.py -v B -i 23 --multiscale
```

**How it works (Nyquist theorem):**

To detect patterns of scale N, you only need to sample at frequency 2/N. The script creates 4 sampling levels:

| Level | Sampling | Samples | Detects patterns |
|-------|----------|---------|------------------|
| Fine | every bit | max 100k | 1-1k bits |
| Medium | every 100 bits | max 1M | 1k-1M bits |
| Large | every 10k bits | max 10M | 1M-1G bits |
| Huge | every 1M bits | max 100k | 1G+ bits |

**Total: ~12M samples ≈ 50MB RAM** to cover the ENTIRE spectrum from 1 bit to 100G+ bits.

This is scientifically valid because large-scale patterns don't require bit-level resolution to be detected. The output includes scale ranges covered and memory usage for full transparency.

#### Phase 3: Topology & Geometry

| Script | Purpose | Usage |
|--------|---------|-------|
| `level1_topological.py` | **TPA** — Topological Persistence Analysis: find φ in structural "holes" | `python level1_topological.py -v B -i 15` |
| `level1_geometric.py` | **Geometric Operators** — Γ (connection), 𝓡 (curvature), 𝓣 (torsion), 𝓔 (energy) | `python level1_geometric.py -v B -i 17` |

**Requirements for Geometric Operators:**
- **Torsion-only** (`--torsion-only`): Only needs phi_snapshots (single iteration)
- **Full analysis** (Γ, 𝓡, 𝓔): Requires Level 1 pattern analysis for 3 consecutive iterations

```bash
# Check requirements before running full geometric analysis
python level1_geometric.py -v B -i 17 --check-only

# Torsion-only mode (always works if phi data exists)
python level1_geometric.py -v B -i 15 --torsion-only
```

### Usage Examples

```bash
# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 0: Φ Sequence Generation
# ═══════════════════════════════════════════════════════════════════════════

# Generate Φ sequences (can take hours/days for high iterations)
python level0_generate.py --variant B --iterations 18

# Generate with plot-only mode (regenerate visualizations from existing data)
python level0_generate.py --variant B --iterations 20 --plot-only

# Generate Variant I (inverse order: 10→01, for φ-asymmetry study)
python level0_generate.py --variant I --iterations 18

# Refresh plots for existing variant data (without regenerating sequences)
python level0_refresh_plots.py -v B --hilbert-bits 1048576 --fft-bits 2000000

# Refresh plots for all variants (recommended Hilbert bits: 1M-4M)
python level0_refresh_plots.py -v D --hilbert-bits 1048576

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1: Pattern Analysis & Emergence
# ═══════════════════════════════════════════════════════════════════════════

# Analyze patterns and infer rules
python level1_analyze_patterns.py --variant B --iteration 18 --report
python level1_analyze_patterns.py --variant B --iteration 18 --min-len 10 --max-len 50 --report

# Calculate Emergence Index (potential for Level 2)
python level1_emergence_index.py --variant B --iteration 18

# Compare emergence potential across variants
python level1_emergence_index.py --variants A B D E F --iteration 18 --compare

# Calculate SCI & ICC metrics from emergence data
python level1_sci_icc.py --from-emergence results/emergence_vars_A_B_D_E_F_G_iter18.json

# Generate publication figures
python level1_visualize.py --variants B A --iterations 18 --format pdf

# ═══════════════════════════════════════════════════════════════════════════
# 🔬 HIDDEN φ DISCOVERY — Complete Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════

# --- Phase 1: Scale & Structure Analysis ---

# MSRA: Multi-Scale Ratio Analyzer (find φ in proportions between scales)
python level1_scale_ratios.py -v B -i 15 --max-bits 100000

# PCSA: Pre-Collapse Structure Analysis (φ in tree branching)
python level1_nesting_tree.py -v B -i 15 --max-bits 100000

# HMRC: Hilbert Multi-Resolution Comparison
python level1_hilbert_multires.py -v B -i 15 --bits 1048576

# --- Phase 2: Deep Analysis ---

# Unified deep analysis (Wavelet, Recurrence, LZ complexity)
python level1_deep_analysis.py -v B -i 15 --max-bits 50000

# Compare variants (KEY DISCOVERY: B shows LZ φ-scaling 0.019!)
python level1_deep_analysis.py -v B -i 15 --max-bits 50000 --compare F

# Multi-scale mode for 100G+ bits (Nyquist-based subsampling)
python level1_deep_analysis.py -v B -i 23 --multiscale

# --- Phase 3: Topology & Geometry ---

# Topological Persistence Analysis (φ in structural holes)
python level1_topological.py -v B -i 15 --max-bits 50000

# Compare topology between variants
python level1_topological.py -v B -i 15 --max-bits 50000 --compare F

# Geometric Operators — Check requirements first!
python level1_geometric.py -v B -i 17 --check-only

# Torsion-only (works with single iteration, no Level 1 needed)
python level1_geometric.py -v B -i 15 --torsion-only

# Full geometric analysis (requires 3 consecutive Level 1 results)
# First, generate Level 1 for iterations 15, 16, 17:
python level1_analyze_patterns.py -v B -i 15 --report
python level1_analyze_patterns.py -v B -i 16 --report
python level1_analyze_patterns.py -v B -i 17 --report
# Then run full analysis:
python level1_geometric.py -v B -i 17 --output results/geometric_B_i17.json
```

The system uses `config.json` for configuration. If the file doesn't exist, it will be created automatically with default values.

### Configuration

Note: Current Level‑0 “basal‑pure” generator no longer uses dynamic collapse parameters (no masks, no seeds). Configuration remains for operational aspects (iterations, HW limits, outputs).

### Automatic Compression

The system includes **intelligent compression management** for large sequences:

- Threshold: Automatically activates when sequences exceed `memory_threshold`
- Format: Saves as `.bin.gz` + `.json` metadata
- Transparency: Loading/saving is completely transparent
- Scalability: Enables experiments for high iterations


## System Components

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

## Absolute Token (ABS) Modes and Variants

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

## Surviving Variants

**Status:** 7 active variants + 1 control (A and C eliminated for incompatibility with ISH principles)

### Core Variants (Tier 1)

**Variant B — Stratified Baseline (Gold Standard)**
- **Algorithm:** Inside→out collapse + final global simplify
- **Properties:** Preserves stratified order, accumulates ALL micro-states
- **Scientific Value:** Fully compatible with ISH principles, validated as ontologically correct
- **Why it survives:** Gold standard for order emergence through stratified containment

**Variant D — Simultaneous Degradation**
- **Algorithm:** Degrade 10→0 and 01→0 simultaneously, then compress runs
- **Properties:** Both patterns degrade to No-Res (HSI ontology compliant)
- **Scientific Value:** Baseline comparison with B using dedicated simplify function
- **Why it survives:** Tests robustness of order emergence

### Experimental Variants (Tier 2)

**Variant E — Two-Phase Degradation**
- **Algorithm:** Phase 1: 01→0 (iterate until stable), Phase 2: 10→0 (iterate until stable)
- **Properties:** Two separate phases with accumulation between them
- **Scientific Value:** Tests whether phase order affects emergent structure
- **Why it survives:** Explores sequence-dependent collapse dynamics with inter-phase accumulation

**Variant I — Inverse of E**
- **Algorithm:** Phase 1: 10→0 (iterate until stable), Phase 2: 01→0 (iterate until stable)
- **Properties:** Inverse phase order of E (collapse before emergence)
- **Scientific Value:** Tests whether the effect of phase order is symmetric or asymmetric
- **Why it survives:** Complements E by testing the opposite ordering

**Variant F — Hybrid Stabilization**
- **Algorithm:** Stabilize inside→out, then single global pass
- **Properties:** Combines stratified stabilization with final global closure
- **Scientific Value:** Tests hybrid strategy combining B's stratified collapse with a global closure step
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

**Variant I — Inverse Order (φ-Asymmetry Study)**
- **Algorithm:** Inverse of E: remove all `10` first, then all `01`, then compress runs
- **Properties:** Tests whether order of removal affects φ-proximity
- **Scientific Value:** Tests the hypothesis that E's φ-generation comes from "emergence (01) before collapse (10)". If true, I's "collapse before emergence" should show different behavior (measured as 2.13× LESS close to φ than E)
- **Why it survives:** Critical for understanding if order has semantic meaning in collapse dynamics
- **Key Finding:** Confirms asymmetry — the order of removal IS semantically meaningful for φ emergence

### Control Variants

To validate that patterns found in ISH variants are genuine emergent structures (not algorithmic artifacts), we include a comprehensive suite of control variants. Variant A serves as the primary random baseline, while additional controls (J, K, L, M) provide diverse sources of complexity for robust hypothesis testing.

**Variant A — Random Control (PRNG/CSPRNG Baseline)** *(Primary Null Control)*
- **Algorithm:** Pseudo-random bit generation (Mersenne Twister or cryptographic RNG)
- **Purpose:** Provides a **scientific control** for comparing ISH-generated structures against random baselines
- **Properties:** No collapse process, no stratified order — pure statistical randomness
- **Scientific Value:** Any pattern found in ISH variants (B, D, E, F, G, H) but **absent** in Variant A indicates genuine emergent structure, not algorithmic artifacts

**Expected Differences (A vs ISH variants):**
| Metric | Variant A (Random) | ISH Variants (B, D, etc.) |
|--------|-------------------|---------------------------|
| Hilbert heatmap | Uniform noise | Structured patterns ("mesh") |
| FFT spectrum | Flat (white noise) | Peaks at specific frequencies |
| Power spectrum β | ~0 (white noise) | Negative β (unusual structure) |
| Ones ratio | ~0.500 (statistical) | May deviate systematically |

**Complete Control Suite (A, J, K, L, M) — Hypothesis Testing** *(Updated Dec 2025)*

| Variant | Source | Purpose | Expected LZ ratio |
|---------|--------|---------|-------------------|
| **A** | PRNG (Mersenne Twister) | Statistical random baseline | ≈ 1.0 |
| **J** | π binary (BCD) | Deterministic pseudo-random | ≈ 1.0 |
| **K** | Rule 30 (Wolfram) | Cellular automaton chaos | ≈ 1.0 |
| **L** | Logistic Map (r=3.99) | Deterministic chaos | ≈ 1.0 |
| **M** | Fibonacci Word | **POSITIVE CONTROL** (has φ) | **≈ 0.62** |

**Why these controls?**
- **A, J, K, L (Null Controls):** Different sources of complexity, ALL should show ratio ≈ 1.0
- **M (Positive Control):** Fibonacci word has φ encoded in its structure. If our algorithm works, it MUST detect φ here. Validates that φ detection in B is real.

**The Complete Validation Logic:**
1. If A, J, K, L show ≈ 1.0 → Null controls pass ✅
2. If M shows ≈ 0.62 → Positive control passes ✅
3. If B shows ≈ 0.62 → HSI has φ-structure like Fibonacci ✅

**Usage:**
```powershell
# Generate control variants (all use level0_generate.py)
python level0_generate.py --variant A -i 15   # Random (null)
python level0_generate.py --variant J -i 15   # π (null, requires mpmath)
python level0_generate.py --variant K -i 15   # Rule 30 (null)
python level0_generate.py --variant L -i 15   # Logistic Map (null)
python level0_generate.py --variant M -i 15   # Fibonacci (positive)

# Skip plots for faster generation
python level0_generate.py --variant M -i 20 --no-plots

# Comparative analysis
python level1_deep_analysis.py --variants A,B,K,L,M -i 15 --analysis lz
```

**Validated Results (Dec 2025):**
| Variant | Type | LZ Ratio | Best Match | φ-distance |
|---------|------|----------|------------|------------|
| **A** | Null Control | 0.990 | 1.0 (random) | 0.372 |
| **K** | Null Control | ~0.98 | 1.0 (random) | ~0.36 |
| **L** | Null Control | 0.984 | 1.0 (random) | 0.365 |
| **M** | Positive Control | **~0.62** | **1/φ** | ~0.01 ✅ |
| **B** | HSI | 0.628 | **1/φ** | 0.010 ✅ |

The test of multiple constants honestly reports which constant is closest. Null controls → "1.0 (random)". Positive control M and HSI B → "1/φ".

---

### 📊 Experimental Results

For detailed experimental validation comparing Control variant (A) vs ISH variants (B-I), including growth projections and φ-alignment analysis, see:

→ [`Documentation/hsi_growth_projections_analysis.md`](Documentation/hsi_growth_projections_analysis.md)

### ❌ Eliminated Variants

**~~Variant A — Global Collapse~~** *(ELIMINATED, then REPURPOSED as Random Control)*
- **Original reason for elimination:** Fundamentally incompatible with stratified order principle
- **Current status:** Repurposed as PRNG control baseline (see above)

**Variant C — Temporal Cascade** *(ELIMINATED)*
- **Reason:** Violates micro-state reality principle OR redundant with Variant B
- **Why eliminated:** Intentionally discards micro-states (incompatible) or identical to B if fixed
- **ISH conflict:** "Intermediate states are ontologically real" — C discards them

### 📖 Full Documentation

See `Documentation/variants_spec.md` for:
- Complete algorithmic definitions
- Scientific justification for each variant
- Implementation cross-reference
- ISH alignment analysis
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

### Pattern Detection Thresholds

For large Φ sequences, the pattern detector uses sampling mode to avoid memory exhaustion. These thresholds are configurable via `config.json → pattern_detection`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `large_sequence_threshold` | 10,000,000 | Sequences above this size (bits) trigger sampling mode |
| `sample_size` | 1,000,000 | When in sampling mode, analyze this many bits per sample |

**Tuning guidelines:**
- **Low RAM (8GB)**: Reduce `large_sequence_threshold` to 5,000,000
- **High RAM (32GB+)**: Increase to 50,000,000 for more thorough analysis
- **Faster results**: Reduce `sample_size` to 500,000
- **More accuracy**: Increase `sample_size` to 5,000,000

These values balance memory usage vs. analysis thoroughness. The defaults work well for systems with 8-16GB RAM.

### Adaptive Sampling (Default Behavior)

The Level 1 analysis automatically calculates optimal sample sizes based on the `max_pattern_length` parameter. This ensures **statistically valid results** regardless of sequence size.

#### How It Works

The required sample size grows exponentially with pattern length:

| max_pattern_length | Optimal Sample Size | Notes |
|-------------------|---------------------|-------|
| 20 bits | ~52M bits | Very efficient |
| 30 bits | ~54G bits | Moderate requirements |
| 40 bits | ~55T bits | Often exceeds sequence size |
| 50 bits | ~56P bits | Requires 100% coverage |

**Formula:** `optimal_sample ≈ (min_occurrences × 2) / (2^(-k) × structure_factor)`

Where `k` = max_pattern_length and `structure_factor` ≈ 50 for HSI structured data.

#### Coverage Recommendations

| Coverage | Statistical Validity | Use Case |
|----------|---------------------|----------|
| < 5% | ⚠️ Low confidence | Not recommended |
| 5-20% | ✅ Adequate for short patterns | Quick exploratory analysis |
| 20-80% | ✅ Good for most patterns | Standard analysis |
| 80-100% | ✅ Optimal | Complete scientific analysis |

#### Memory Requirements

Approximate RAM needed for analysis:

```
RAM (GB) ≈ (bits_analyzed × 2) / (1024³)
```

| Bits Analyzed | RAM Required |
|---------------|--------------|
| 1G bits | ~2 GB |
| 10G bits | ~20 GB |
| 50G bits | ~100 GB |
| 100G bits | ~200 GB |

#### Practical Recommendations

1. **For demonstrations**: Use `--max-len 30` with 10G bits (adequate coverage)
2. **For rigorous analysis**: Use `--max-len 50` with maximum available RAM
3. **For pattern-specific studies**: Adjust `--max-len` to match patterns of interest

The system will display a warning if coverage is below the minimum (5%) for the specified pattern length.

## Level 1 Analysis (Structural Format)

**Status:** Fully operational with structural pattern detection and order metrics

### Overview

Level 1 implements advanced analysis of Level 0 data to detect **emergent order** in the informational structure. The structural format preserves the complete structural information (parentheses marking Absolutes) that is critical for understanding how order emerges from the primordial tension between 0 and 1.

### Key Concepts

**Structural Format:**
- Preserves parentheses `()` that mark Absolute boundaries
- Enables analysis of **stratified containment** (how Absolutes nest within each other)
- Achieves ~99.7% compression while maintaining full structural information
- Backward compatible with legacy format (observable bits only)

**Emergent Order (not φ-convergence):**
- ISH focuses on **order as fundamental principle**, not golden ratio convergence
- "Order is the only way Nothingness can contradict itself"
- Level 1 detects and quantifies this emergent order through multiple metrics

### Components

#### 1. Data Loader

Load Level 0 data with automatic format detection:

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

### Metric Weights Configuration

Both the **Emergence Indices** and **Order Emergence Index (OEI)** are composite metrics with configurable weights. Weights can be adjusted via `config.json → metrics`:

**SEI Weights** (Structural Emergence — physical laws potential):

| Component | Default | Scientific Justification |
|-----------|---------|--------------------------|
| Order (1-LZ) | 30% | Compressibility = law existence (Chaitin, 1987) |
| Hierarchy (HBE) | 30% | Multi-scale structure = renormalization (Wilson, 1971) |
| Coherence (MI) | 20% | Long-range order = phase coherence (Anderson, 1972) |
| Non-randomness | 20% | Deterministic dynamics (Peng et al., 1994) |

**ECI Weights** (Edge of Chaos — computational capacity):

| Component | Default | Scientific Justification |
|-----------|---------|--------------------------|
| Criticality (1/f) | 25% | Classic SOC signature (Bak, Tang & Wiesenfeld, 1987) |
| Complexity (LZ) | 25% | LZ ≈ 0.5 = edge of chaos (Langton, 1990) |
| Coherence (MI) | 20% | Global coherence for information integration |
| Hierarchy (HBE) | 15% | Multi-scale structure |
| DFA (Hurst) | 15% | H ≈ 1.0 = 1/f long-range correlations |

**OEI Weights** (Structural order):

| Component | Default | Scientific Justification |
|-----------|---------|--------------------------|
| Depth Organization | 30% | Core signal: hierarchical structure is fundamental |
| Entropy Gradient | 30% | Core signal: stratification indicates real order |
| Containment Regularity | 20% | Support: regularity without hierarchy is not order |
| Balance Score | 20% | Gate: binary sanity check (0 or 1) |

To customize, edit `config.json`:
```json
"metrics": {
  "emergence_weights": { "order": 0.30, "hierarchy": 0.30, "coherence": 0.20, "non_randomness": 0.20 },
  "oei_weights": { "depth_organization": 0.40, ... }
}
```
Weights are automatically normalized to sum to 1.0.

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

For a complete end-to-end analysis pipeline, use:
```bash
python level1_analyze_patterns.py --variant B --iteration 18 --report
```

This will:
1. Load structural data for the specified variant and iteration
2. Extract basic structural metrics
3. Detect observable and structural patterns
4. Calculate all order metrics
5. Generate comprehensive JSON report and Markdown summary

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
- **Validation of ISH hypothesis** ("order as fundamental principle")
- **Comparison between variants** to identify which best exhibits order emergence
- **Temporal analysis** of how order evolves across iterations

## Data Management and Compression

### Automatic Compression Strategy

The system implements an intelligent compression strategy based on `level0_data_storage.md`:

| Iteration | Estimated Size | Storage Method | Compression Ratio |
|-----------|----------------|----------------|-------------------|
| 16 | ~41M bits | Memory | N/A |
| 18 | ~330M bits | **Compressed** | ~85% |
| 20 | ~1.5GB | **Compressed** | ~85% |
| 24 | ~23GB | **Compressed** | ~85% |
| 32 | ~2TB | **Compressed** | ~85% |

### Storage Structure for Experimental Results

All experimental data, analysis outputs, and visualizations are stored in a unified structure organized by processing level:

```
results/
│
├── level0/                                   # ═══ LEVEL 0: Sequence Generation ═══
│   │
│   ├── phi_snapshots/                        # Raw Φ sequences
│   │   ├── var_A/
│   │   │   ├── phi_iter18.struct.gz          # Structural format (preserves parentheses)
│   │   │   └── phi_iter18.json               # Iteration metadata
│   │   ├── var_B/
│   │   ├── var_E/
│   │   ├── var_I/
│   │   └── ... (var_D, var_F, var_G, var_H)
│   │
│   ├── reports/                              # Generator reports
│   │   ├── variant_B_18_20251207.json
│   │   ├── variant_B_18_20251207.enriched.json
│   │   └── metadata_{var}.json
│   │
│   └── visualizations/                       # Level 0 visualizations
│       ├── hilbert_{var}_i{N}_*.png          # Hilbert curve heatmaps
│       ├── fft_{var}_i{N}_*.png              # FFT spectrum plots
│       ├── spectrum_beta_{var}_i{N}_*.png    # 1/f spectrum analysis
│       └── raster2d_{var}_i{N}_*.png         # 2D raster visualizations
│
├── level1/                                   # ═══ LEVEL 1: Analysis ═══
│   │
│   ├── analysis/                             # Orchestrator results
│   │   ├── var_B_iter17_min10_max50.json     # Pattern/rule analysis
│   │   └── var_B_iter17_min10_max50.md       # Markdown reports
│   │
│   ├── metrics/                              # Computed metrics
│   │   ├── emergence_B_iter17.json           # Emergence index
│   │   ├── emergence_B_E_I_iter20.json       # Multi-variant emergence
│   │   ├── sci_icc_B_iter17.json             # SCI/ICC metrics
│   │   └── geometric_B_iter17.json           # Geometric analysis (Γ, 𝓡, 𝓣)
│   │
│   ├── trends/                               # Trend analysis
│   │   ├── trend_analysis.json               # Consolidated results
│   │   └── plots/                            # Trend visualizations
│   │       ├── trend_B_E_iter17-23_emergence.png
│   │       ├── trend_B_E_iter17-23_sci.png
│   │       └── trend_combined.png
│   │
│   └── figures/                              # Level 1 visualizations
│       ├── confidence_dist_{var}_iter{N}.png
│       ├── markov_heatmap_{var}_iter{N}.png
│       ├── top_patterns_{var}_iter{N}.png
│       ├── patterns_evol_{var}.png
│       ├── confidence_evol_{var}.png
│       └── survival/
│           ├── survival_analysis_{var}.json
│           ├── survival_flow_{var}.png
│           └── survival_lengths_{var}.png
│
├── level2/                                   # ═══ LEVEL 2: Advanced Complexity ═══
│   │
│   ├── complexity_analysis/                  # Multi-scale complexity metrics
│   │   ├── complexity_{variants}_iter{N}.json
│   │   └── ...
│   │
│   └── transfer_entropy/                     # Information flow analysis
│       ├── te_{var}_iter{N}.json             # Transfer entropy results
│       └── te_heatmap_{var}_iter{N}.png      # TE visualizations
│
├── cache/                                    # ═══ SHARED: Caches ═══
│   └── level1_cache_phi_iter*.pkl
│
└── temp/                                     # ═══ SHARED: Temporary ═══
    ├── accumulation_{var}.tmp                # Large data during generation
    └── checkpoint_{var}_{iter}.json          # Resumable checkpoints (auto-cleaned)
```

> **Note**: Checkpoints are stored in `temp/` and auto-cleaned after successful completion.
> Master results are stored in `level0/reports/hsi_master_results_{variants}.json`.

### Compression Tools

```bash
# Estimate storage requirements for high iterations
python level0_estimate_storage.py --max-iteration 32

# Detailed analysis with breakdown
python level0_estimate_storage.py --max-iteration 28 --detailed
```

## Metrics and Validation

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

### Hidden φ Discovery Metrics (Phase 1-3)

These advanced metrics implement the research roadmap for discovering hidden φ in variant structures. The key hypothesis: **φ exists in PROPORTIONS between structures, not in absolute structures themselves**.

#### Phase 1: Scale & Structure Metrics

| Metric | Script | What it measures | φ Detection |
|--------|--------|------------------|-------------|
| **MSRA (Multi-Scale Ratio Analyzer)** | `level1_scale_ratios.py` | Ratios between block metrics (entropy, density) at scales 2^k | Ratios near 1.618 |
| **PCSA (Pre-Collapse Structure Analysis)** | `level1_nesting_tree.py` | Branching ratios in parentheses tree structure | Children/parent ratios |
| **HMRC (Hilbert Multi-Resolution)** | `level1_hilbert_multires.py` | Density gradients across Hilbert curve resolutions | Resolution scaling |

#### Phase 2: Deep Analysis Metrics

| Metric | Script | What it measures | φ Detection |
|--------|--------|------------------|-------------|
| **Wavelet φ-Band** | `level1_deep_analysis.py` | Energy concentration in φ-related frequency bands | Band energy ratios |
| **Recurrence Plot** | `level1_deep_analysis.py` | Diagonal line distribution in recurrence matrix | Line length ratios |
| **LZ φ-Scaling** ⭐ | `level1_deep_analysis.py` | Lempel-Ziv complexity ratios between scales | **B shows 0.019 distance!** |

**Key Discovery (Dec 2025):** The LZ complexity analysis revealed that Variant B shows **extraordinary φ-scaling** with distance 0.019 from φ, while F shows 0.063 (3.3× weaker). This confirms that **φ is hidden in B's proportions**.

#### Phase 3: Topology & Geometry Metrics

| Metric | Script | What it measures | φ Detection |
|--------|--------|------------------|-------------|
| **TPA (Topological Persistence)** | `level1_topological.py` | Birth/death times of topological features (holes, components) | Lifespan ratios |
| **Betti φ-Scaling** | `level1_topological.py` | Betti numbers (H0, H1) at φ-related scales | Inter-scale ratios |
| **𝓣 Torsion** | `level1_geometric.py` | Positional asymmetries (pattern distribution variance) | Torsion ratios |
| **Γ Connection** | `level1_geometric.py` | Pattern survival rate between iterations | Survival ≈ 1/φ |
| **𝓡 Curvature** | `level1_geometric.py` | Rule interference / path-dependence | Curvature ≈ φ-1 |
| **𝓔 Energy** | `level1_geometric.py` | Global quality functional: α|𝓡|² + β|𝓣|² + γ·Φ_incoh | Low = stable |

**Geometric Operators Interpretation:**
- **𝓣 LOW** → Symmetric, patterns uniformly distributed
- **𝓣 HIGH** → Strong positional asymmetries (life-compatible?)
- **𝓡 ≈ 0** → Flat geometry, path-independent
- **𝓡 HIGH** → Curved geometry, order matters
- **𝓔 LOW** → Stable, coherent configuration (universe-like)

### Dual Emergence Indices (Level 2 Potential)

The system provides **two complementary indices** to evaluate emergence potential from different perspectives:

#### SEI — Structural Emergence Index (Dec 2025)

Designed specifically for HSI theory to detect potential for **emergence of physical laws**. Unlike "edge of chaos" metrics, SEI rewards order and structure.

| Component | Weight | What it measures | Optimal value |
|-----------|--------|------------------|---------------|
| **Order** | 30% | Compressibility (1 - LZ); physical laws are compressible | Low LZ → high score |
| **Hierarchy** | 30% | Multi-scale structure; laws operate across scales | High HBE score |
| **Coherence** | 20% | Long-range MI; constants are same everywhere | High MI ratio |
| **Non-randomness** | 20% | Combined DFA + Criticality; penalizes pure random | H > 0.5, slope < 0 |

**Scientific basis:** Physical laws are compressible (Chaitin, 1987), operate across scales (Wilson, 1971), and require non-random dynamics (Peng et al., 1994).

#### ECI — Edge of Chaos Index (Original)

Based on Langton (1990) and Kauffman (1993), seeks the computational "sweet spot" between order and chaos.

| Component | Weight | What it measures | Optimal value |
|-----------|--------|------------------|---------------|
| **Criticality (1/f)** | 25% | Power spectrum slope ≈ -1 (SOC) | slope = -1.0 |
| **Complexity (LZ)** | 25% | LZ ≈ 0.5 = edge of chaos | LZ ≈ 0.5 |
| **Coherence** | 20% | Long-range correlations | High MI ratio |
| **Hierarchy** | 15% | Multi-scale entropy | High HBE |
| **DFA (Hurst)** | 15% | H ≈ 0.8-1.0 = 1/f correlations | H ≈ 1.0 |

#### Combined Interpretation

| SEI | ECI | Verdict |
|-----|-----|---------|
| High | High | 🎯 **Best candidate** — structured AND dynamic |
| High | Low | 📐 Ordered but static — stable laws, less emergence |
| Low | High | 🌀 Dynamic but chaotic — complex but unstable laws |
| Low | Low | ❓ Neither — unlikely to generate meaningful emergence |

**Usage:**
```bash
# Single variant analysis (shows both SEI and ECI)
python level1_emergence_index.py --variant B --iteration 18

# Compare multiple variants
python level1_emergence_index.py --variants A B D E F G H --iteration 18 --compare

# Save results to JSON
python level1_emergence_index.py --variant B --iteration 18 --output results/emergence_B.json

# Statistical significance testing (compare against null model)
python level1_emergence_index.py --variant B --iteration 18 --null-test --null-samples 100
```

**Statistical Significance Testing:**

The `--null-test` flag enables comparison against a null model to determine if the observed indices are statistically significant.

| Parameter | Description |
|-----------|-------------|
| `--null-test` | Enable null model comparison |
| `--null-samples N` | Number of shuffled sequences to generate (default: 100) |

**Output includes:**
- **p-value**: Probability of observing this result by chance (p < 0.05 = significant)
- **z-score**: Standard deviations from null mean (|z| > 2 = significant)
- **Null mean/std**: Statistics of the null distribution

### Structured Complexity Index (SCI) & Cosmological Coherence Index (ICC)

These are advanced metrics designed to evaluate which HSI variant has the highest potential for generating **universe-like structures with emergent physical laws**.

#### Structured Complexity Index (SCI)

**Purpose:** Distinguishes "meaningful" complexity from empty randomness. A random sequence has high entropy but no structure; a trivial sequence (all 0s) has no complexity. Physical systems need *structured* complexity—order at multiple scales with causal relationships.

**Formula:**
```
SCI = α·Hierarchy + β·Causality + γ·NonTriviality
```

| Component | Weight | What it measures | Optimal |
|-----------|--------|------------------|---------|
| **Hierarchy** | 35% | Structure at multiple scales (variance of multi-scale entropy) | High variance |
| **Causality** | 40% | Transfer Entropy—how much the past predicts the future (physics needs causality!) | High TE |
| **NonTriviality** | 25% | LZ≈0.5 AND diverse patterns (not trivial, not random) | LZ=0.5, high diversity |

**Interpretation:**
- **SCI > 0.6**: High structured complexity—strong emergence potential for physical laws
- **SCI 0.4-0.6**: Good complexity—candidate for emergence
- **SCI < 0.4**: Low or random—limited potential

#### Cosmological Coherence Index (ICC)

**Purpose:** Evaluates compatibility with **physical universe generation**. A high ICC suggests the variant could generate physics-like structures. Inspired by known properties of our universe.

**Formula:**
```
ICC = √(Compressibility × Coherence) × Hierarchy × Criticality × (1 + φ_bonus)
```

| Component | What it measures | Why it matters |
|-----------|------------------|----------------|
| **Compressibility** | 1 - LZ (can information be compressed?) | Physical laws ARE compressible (F=ma, E=mc²) |
| **Coherence** | Long-range Mutual Information | Non-local correlations (like quantum entanglement) |
| **Hierarchy** | Multi-scale structure | Quarks → atoms → molecules → cells → galaxies |
| **Criticality** | Hurst ≈ 0.85, power slope ≈ -1 | Edge of chaos—where complexity emerges |
| **φ-Tendency** | Golden ratio in ratios (optional bonus) | φ appears throughout nature |

**Interpretation:**
- **ICC > 0.6**: Excellent cosmological coherence—strong candidate for universe generation
- **ICC 0.4-0.6**: Good coherence—shows physical-like properties
- **ICC < 0.4**: Limited coherence—less likely to generate physics

**Usage:**
```bash
# Calculate SCI & ICC from pre-computed emergence data
python level1_sci_icc.py --from-emergence results/emergence_vars_A_B_D_E_F_G_iter18.json
```

**Key Discovery (December 2025):** Variant E shows the highest ICC (0.692), suggesting it is the best candidate for generating universe-like structures. This correlates with E being 1.61× (≈φ itself!) closer to the golden ratio than Variant B.

### Trend Analysis (`level1_trend_analysis.py`)

Analyzes metric evolution across multiple iterations to identify trends and extrapolate future behavior.

**Features:**
- **Auto-discovery**: Scans `results/phi_snapshots/var_{X}/` to detect available iterations
- **Batch analysis**: Runs emergence + SCI/ICC for all discovered/specified iterations
- **Trend plots**: Generates PNG plots showing metric evolution over iterations
- **Extrapolation**: Linear regression to predict values at future iterations (default: iter 30)
- **Consolidated JSON**: Saves all data to `results/trend_analysis.json`

**Usage:**
```bash
# Discover available iterations for all variants
python level1_trend_analysis.py --variants B E I --discover-only

# Analyze specific iterations with trend plots
python level1_trend_analysis.py --variants B --iterations 17 18 19 20 --plot

# Full analysis with extrapolation to iteration 30
python level1_trend_analysis.py --variants B E I --plot --extrapolate 30

# Output saved to:
#   results/trend_analysis.json
#   results/trend_plots/trend_*.png
```

**Extrapolation Output:**
```
Variant B:
Metric               Slope/iter   R²       @iter 30
----------------------------------------------------
emergence_index      -0.009384   0.063    0.6102
sci                  -0.000780   0.068    0.7400
icc                  -0.006801   0.222    0.5215
phi_tendency         +0.011082   0.607    0.4045
```

### Algorithmic Complexity Analysis (`level2_complexity_analysis.py`)

Compares variants using multiple complexity metrics to distinguish "noise" from "music" (organized structure).

**Metrics calculated:**
- **Shannon Entropy**: Bit-level and 8-bit block entropy
- **Lempel-Ziv Complexity**: Normalized LZ76 (0=trivial, 1=random)
- **Logical Depth Proxy**: Compression time / compressed size (approximates Bennett's Logical Depth)
- **BDM**: Block Decomposition Method (requires `pybdm` library)

**HSI-Specific Scoring:**

Traditional complexity metrics reward high entropy + high LZ, which incorrectly favors chaotic sequences.
For HSI, we want **structured compressibility** (low LZ, moderate entropy, high depth).

The HSI score rewards:
1. High compressibility (structure exists)
2. Moderate entropy (0.25-0.55) — variety without chaos
3. Low but non-zero LZ (0.01-0.1) — detectable patterns

**Usage:**
```bash
# Single variant analysis
python level2_complexity_analysis.py --variant B --iteration 18

# Multi-variant comparison (recommended)
python level2_complexity_analysis.py --variants A B D E --iteration 18 --verbose

# With custom sample size
python level2_complexity_analysis.py --variants B D E --iteration 18 --max-bits 100000

# Output saved to: results/complexity_analysis/complexity_*.json
```

**Example Output:**
```
📊 COMPARISON SUMMARY
══════════════════════════════════════════════════════════════════════
   Variant  Entropy    LZ         Depth        Compress   Class
   A        0.4998     0.5420     2.2159       0.0892     ORGANIZED COMPLEXITY
   B        0.2895     0.0141     0.2080       0.0093     TRIVIAL
   D        0.3722     0.0174     0.2150       0.0089     INTERMEDIATE

   🏆 RANKING (by HSI Score):
   🥇 1. Variant D: Score = 0.996
   🥈 2. Variant B: Score = 0.996
   🥉 3. Variant A: Score = 0.699   ← High LZ penalized (chaotic)
```

### Takens Embedding & TDA (`level2_takens_tda.py`)

Discovers emergent geometry in Φ sequences using phase space reconstruction and persistent homology.

**Method:**
1. **Takens' Embedding**: Convert 1D binary sequence to 3D/nD point cloud: `v_t = (s_t, s_{t+τ}, s_{t+2τ})`
2. **Persistent Homology**: Compute Betti numbers using Vietoris-Rips complexes:
   - **β₀**: Connected components (clusters)
   - **β₁**: 1D cycles (orbital loops, recurring patterns)
   - **β₂**: 2D cavities (enclosed surfaces - strong emergence!)

**Scientific Value:**
If we detect **β₂ > 0**, we've demonstrated that Level 1 generates a REAL geometric surface (emergence of space).

**Requirements:**
```bash
pip install ripser  # Fast persistent homology
# or
pip install giotto-tda  # Full TDA suite
```

**Usage:**
```bash
# Single variant analysis
python level2_takens_tda.py --variant B --iteration 18

# With 3D visualization
python level2_takens_tda.py --variant B --iteration 18 --visualize

# Higher dimensional embedding
python level2_takens_tda.py --variant B --iteration 18 --dim 4 --delay 2

# Multi-variant comparison
python level2_takens_tda.py --variants B D E --iteration 18 --compare

# Output saved to:
#   results/tda_analysis/tda_B_iter18.json
#   results/tda_analysis/takens_3d_B_iter18.png
```

**Example Output:**
```
🌀 HSI TAKENS/TDA ANALYSIS
══════════════════════════════════════════════════════════════════════
   Variant: B
   Iteration: 18
   Embedding dim: 3
   Max points: 10,000

   📊 Computing Persistent Homology...
      🟢 β₀ = 1 significant (connected structure)
      🟢 β₁ = 3 significant (orbital loops)
      ⚪ β₂ = 0 significant (no closed surface)

   🔮 TOPOLOGICAL INTERPRETATION:
      β₀=1: ✓ Single connected component (unified structure)
      β₁=3: 🌀 3 persistent loops detected (ORBITAL STRUCTURE)
      β₂=0: No 2D cavities (no enclosed surfaces)

   📋 SUMMARY:
      🔶 MODERATE EMERGENCE: Multiple orbital structures detected
```

### Transfer Entropy Multi-Scale (`level2_transfer_entropy.py`)

Measures directed information flow between different scales of the Φ sequence to detect causal relationships.

**Method:**
1. **Coarse-Graining**: Create representations at multiple scales (2^k block sizes)
2. **Transfer Entropy**: Compute TE between all pairs of scales
   - TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
3. **Directional Analysis**:
   - If TE(fine → coarse) > TE(coarse → fine): **Bottom-up emergence**
   - If TE(coarse → fine) > TE(fine → coarse): **Top-down causation**

**Scientific Value:**
This reveals whether HSI has genuine multi-scale causal structure (like living systems) vs pure noise.

**Usage:**
```bash
# Single variant analysis
python level2_transfer_entropy.py --variant B --iteration 18

# Custom scales
python level2_transfer_entropy.py --variant B --iteration 18 --scales 1 2 4 8 16 32

# Multi-variant comparison with heatmap
python level2_transfer_entropy.py --variants B D E --iteration 18 --plot

# Output saved to:
#   results/transfer_entropy/te_B_iter18.json
#   results/transfer_entropy/te_heatmap_B_iter18.png
```

**Example Output:**
```
🔀 HSI TRANSFER ENTROPY ANALYSIS
══════════════════════════════════════════════════════════════════════
   Variant: B
   Scales: [1, 2, 4, 8, 16, 32]

   📊 TRANSFER ENTROPY MATRIX (bits):
              →1    →2    →4    →8   →16   →32
        1 →   ---  0.012 0.008 0.005 0.003 0.002
        2 →  0.010  ---  0.015 0.010 0.006 0.004
        ...

   🔮 DIRECTIONAL ANALYSIS:
      Avg TE (fine → coarse): 0.0082 bits
      Avg TE (coarse → fine): 0.0045 bits
      Asymmetry: +0.0037 bits

      🔼 BOTTOM-UP EMERGENCE: Fine scales causally influence coarse scales
```

## Visualizations
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

## Results

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

## Usage Examples

### 🚀 Quick Start (5 minutes)

```bash
# 1. Generate data (creates Hilbert visualization + Φ snapshot)
python level0_generate.py --variant B --iterations 15

# 2. Analyze patterns
python level1_analyze_patterns.py --variant B --iteration 15 --report

# 3. Calculate Emergence Index
python level1_emergence_index.py --variant B --iteration 15
```

---

### 📋 Complete Execution Guide (Recommended Order)

This guide provides a sequential workflow from data generation to advanced analysis.

#### **PHASE 0: Data Generation**

Generate Φ sequences for the variants you want to analyze. **Start with iteration 15 for quick tests**, then increase.

```bash
# Step 0.1: Generate candidate variants (B, D, E are top candidates)
python level0_generate.py --variant B --iterations 18  # ~30min, ~180MB
python level0_generate.py --variant D --iterations 18
python level0_generate.py --variant E --iterations 18

# Step 0.2: (Optional) Generate control variant A (chaotic/random)
python level0_generate.py --variant A --iterations 18  # Much larger, more chaotic

# Step 0.3: Verify what's available
python level0_generate.py --list-snapshots
```

**📁 Output:** `results/phi_snapshots/var_{X}/phi_iteration_{N}.bin.gz`

#### **PHASE 1: Pattern & Rule Analysis**

Analyze structural patterns and infer rules. **Run in order of importance**.

```bash
# Step 1.1: Pattern Detection (MOST IMPORTANT - generates base data)
python level1_analyze_patterns.py -v B -i 18 --report
python level1_analyze_patterns.py -v D -i 18 --report
python level1_analyze_patterns.py -v E -i 18 --report

# Step 1.2: Emergence Index (Level 2 potential score)
python level1_emergence_index.py -v B -i 18
python level1_emergence_index.py -v D -i 18
python level1_emergence_index.py -v E -i 18

# Step 1.3: SCI & ICC Metrics (requires emergence data)
python level1_sci_icc.py --from-emergence results/emergence_B_iter18.json

# Step 1.4: Trend Analysis (requires multiple iterations)
python level1_trend_analysis.py --variants B D E --plot

# Step 1.5: Generate Visualizations
python level1_visualize.py --all
```

**📁 Output:** `results/level1_*.json`, `reports/*.md`, `results/figures/`

#### **PHASE 1+: Hidden φ Discovery (Advanced)**

These scripts search for hidden golden ratio relationships.

```bash
# Step 1+.1: Scale Ratios (MSRA)
python level1_scale_ratios.py -v B -i 18

# Step 1+.2: Nesting Tree (PCSA)
python level1_nesting_tree.py -v B -i 18

# Step 1+.3: Hilbert Multi-Resolution (HMRC)
python level1_hilbert_multires.py -v B -i 18

# Step 1+.4: Deep Analysis (Wavelet + Recurrence + LZ)
python level1_deep_analysis.py -v B -i 18
```

#### **PHASE 2: Advanced Complexity Analysis**

These scripts provide deeper insights into the mathematical structure.

```bash
# Step 2.1: Algorithmic Complexity Comparison (LZ, Shannon, Logical Depth)
python level2_complexity_analysis.py --variants A B D E --iteration 18

# Step 2.2: Takens Embedding & TDA (detect emergent geometry)
pip install ripser  # Install first if needed
python level2_takens_tda.py --variant B --iteration 18 --visualize

# Step 2.3: Transfer Entropy (detect causal flow direction)
python level2_transfer_entropy.py --variants B D E --iteration 18 --plot
```

**📁 Output:** `results/complexity_analysis/`, `results/tda_analysis/`, `results/transfer_entropy/`

---

### 📊 Suggested Analysis Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 0: GENERATE DATA                                             │
│  level0_generate.py → Creates Φ snapshots + Hilbert visualizations  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PATTERN ANALYSIS (Core)                                   │
│  level1_analyze_patterns.py → Patterns, rules, structural metrics   │
│  level1_emergence_index.py  → Level 2 potential score               │
│  level1_sci_icc.py          → Self-organization metrics             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1+: HIDDEN φ DISCOVERY (Optional but recommended)            │
│  level1_scale_ratios.py     → φ in scale proportions                │
│  level1_deep_analysis.py    → Wavelet + Recurrence analysis         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: ADVANCED ANALYSIS                                         │
│  level2_complexity_analysis.py → Distinguish noise from structure   │
│  level2_takens_tda.py          → Detect emergent geometry (Betti)   │
│  level2_transfer_entropy.py    → Detect causal flow direction       │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Basic Experiment (Programmatic)

```python
# Generate Φ sequence with compression
phi, _, metadata = simulate_phi(
    max_iterations=16,
    use_compression=True,
    memory_threshold=100000000
)

# Check variant info
print(f"Variant: {metadata['variant']}, Iterations: {metadata['iterations_completed']}")

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

## Contributions

This project is part of research on the Informational Singularity Hypothesis. Contributions are welcome:

1. Fork the repository
2. Create branch for new functionality
3. Implement changes with tests
4. Create pull request with detailed description

## References and Documentation

### Technical Documentation

| Document | Description |
|----------|-------------|
| `Documentation/level1_analysis_guide.md` | Complete Level 1 technical guide |
| `Documentation/order_metrics_explained.md` | Scientific explanation of order metrics |
| `Documentation/variants_spec.md` | Variant specifications |
| `Documentation/HSI_Variants_Analysis_For_Review.md` | Analysis for peer review |
| `Documentation/GPU_ACCELERATION_PROPOSAL.md` | Future GPU integration proposal |
| `level1/README.md` | Quick reference for Level 1 module |

### Code Documentation

- `README.md` - This document (main guide)
- Detailed docstrings in all Python modules
- Inline comments explaining ISH theory

## License

This project is developed by Iban Borràs in collaboration with Augment Agent (Claude Sonnet 4) as part of research on the Informational Singularity Hypothesis.

## Roadmap and Future Developments

### Planned Technological Expansions

#### **Symbolic Quantum Computing**
- **Integration with quantum simulators** to explore ISH in Hilbert spaces
- **Quantum algorithms** for detecting Pₖ patterns in superposition
- **Informational entanglement** as manifestation of the Primordial Holographic Field

#### **Informational Consciousness Models**
- **Application to consciousness theories** based on integrated information (IIT)
- **Modeling of cognitive processes** emerging from 0↔1 tension
- **Connections with fundamental physics** and theories of everything

#### **Advanced Visualization**
- **3D rendering** of H(Φ) structures from the Holographic Field
- **Immersive visualization** of temporal evolution of Ψ
- **Virtual reality** to explore emergent pattern space
- **Real-time animations** of convergence towards φ

### Scientific Extensions

#### **Experimental Validation**
- **Comparison with real physical data** (fundamental constants, cosmic structures)
- **Testable predictions** derived from ISH
- **Collaboration with theoretical physicists** for empirical validation

#### **Advanced Analysis**
- **Machine Learning** to discover non-obvious ISH patterns
- **Neural networks** trained on Φ sequences
- **Time series analysis** of informational evolution

#### **Memory-Aware Streaming Analysis (Future)**

For researchers needing to analyze sequences larger than available RAM:

**Planned implementation:**
1. **Chunked streaming analysis**: Process data in memory-efficient chunks
2. **Pattern count merging**: Aggregate counts across chunks with statistical confidence
3. **Distributed processing**: Support for cluster computing environments

**Current workarounds:**
- Reduce `--max-len` to require smaller samples
- Use `--max-chars` to analyze manageable portions
- Run multiple analyses with different random seeds and average results

**RAM Estimation Tool** (coming soon):
```bash
python level1_estimate_resources.py --variant B --iteration 25 --max-len 50
# Output: Estimated RAM: 480GB, Time: ~48 hours, Coverage: 100%
```

For contributions to streaming implementation, see `docs/ADAPTIVE_SAMPLING_PROPOSAL.md`.

#### **Interdisciplinary Applications**
- **Biology**: ISH patterns in DNA and evolution
- **Cosmology**: Large-scale structures and ISH
- **Mathematics**: Connections with number theory and fractal geometry

### System Improvements

#### **Performance Optimization**
- **Parallelization** of pattern detection algorithms
- **GPU acceleration** for fractal dimension calculations
- **Distributed algorithms** for large-scale experiments

#### **Advanced Features**
- **Visual editor** for custom collapse rules
- **Web interface** for remote experiments
- **REST API** for integration with other systems
- **Database** of discovered patterns and rules

## Support

For problems or questions:

1. **Configuration**: Edit `config.json` - it's self-documenting with built-in help
2. **Documentation**: Review files in `Documentation/` folder
3. **Compression**: Use `level0_estimate_storage.py` for storage planning
4. **Research**: Contact the author for scientific questions about ISH

### Troubleshooting Tools

```bash
# Check compression requirements
python level0_estimate_storage.py --max-iteration 24

# Test Level 0 generator (with progress and compression level)
python level0_generate.py -v B -i 4 --log-every 1 --compression-level 5

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
