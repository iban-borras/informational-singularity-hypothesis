# Level 1 Visualization Plan

## Overview

Script to generate publication-quality figures from Level 1 analysis results.

**Location:** `visualizations/level1_figures.py`
**Output:** `results/figures/`

---

## Figure Types

### Type 1: Variant Comparison (requires 2+ variants, same iteration)

| Figure | Description | Filename |
|--------|-------------|----------|
| `confidence_comparison` | Bar chart comparing % deterministic rules | `confidence_cmp_{V1}_vs_{V2}_iter{N}.png` |
| `confidence_histogram` | Overlaid histograms of rule confidence | `confidence_hist_{V1}_{V2}_iter{N}.png` |
| `patterns_comparison` | Top patterns side by side | `patterns_cmp_{V1}_vs_{V2}_iter{N}.png` |

### Type 2: Evolution Over Iterations (requires 1 variant, 2+ iterations)

| Figure | Description | Filename |
|--------|-------------|----------|
| `confidence_evolution` | Line chart of avg confidence over iterations | `confidence_evol_{V}.png` |
| `patterns_evolution` | Pattern count evolution | `patterns_evol_{V}.png` |
| `determinism_evolution` | % rules ≥99% over iterations | `determinism_evol_{V}.png` |

### Type 3: Single Analysis (requires 1 variant, 1 iteration)

| Figure | Description | Filename |
|--------|-------------|----------|
| `markov_heatmap` | 2x2 or 4x4 transition probability matrix | `markov_heatmap_{V}_iter{N}.png` |
| `top_patterns` | Horizontal bar chart of top 10 patterns | `top_patterns_{V}_iter{N}.png` |
| `confidence_distribution` | Histogram of confidence values | `confidence_dist_{V}_iter{N}.png` |

---

## Design Decision: Flexible Multi-Mode Approach

### Option A: Explicit Mode Selection (User specifies what to generate)

```powershell
# Compare two variants
python level1_figures.py --compare B A --iteration 18

# Evolution for one variant
python level1_figures.py --evolution B --iterations 17 18 19

# Single figure
python level1_figures.py --single markov_heatmap --variant B --iteration 18
```

**Pros:** Precise control
**Cons:** Complex CLI, user must know what data exists

### Option B: Auto-Discovery (Script finds data and generates all possible figures)

```powershell
# Generate ALL possible figures from available data
python level1_figures.py --all

# Filter by variant(s)
python level1_figures.py --variants B A

# Filter by iteration(s)  
python level1_figures.py --iterations 17 18 19
```

**Pros:** Simple, discovers data automatically
**Cons:** May generate many figures

### Option C: Hybrid (Recommended) ✅

```powershell
# Auto-discover and generate all figures for specified variants
python level1_figures.py --variants B A --iterations 17 18 19

# If no args, scan all available data and generate everything
python level1_figures.py --all
```

The script will:
1. Scan `results/` for available `level1_analysis_var*_iter*_min*_max*.json`
2. Determine which figures can be generated
3. Generate comparison figures for all variant pairs
4. Generate evolution figures for each variant with 2+ iterations
5. Generate single figures for each variant/iteration combo

---

## CLI Specification (Option C)

```
usage: level1_figures.py [-h] [--all] [--variants V [V ...]] 
                         [--iterations N [N ...]] [--figures F [F ...]]
                         [--output-dir DIR] [--format {png,pdf,svg}]
                         [--dpi DPI]

Generate Level 1 analysis figures for scientific paper.

optional arguments:
  -h, --help            Show help message
  --all                 Generate all possible figures from available data
  --variants V [V ...]  Variants to include (default: all found)
  --iterations N [N ...]  Iterations to include (default: all found)
  --figures F [F ...]   Specific figures to generate (default: all)
                        Options: confidence_comparison, confidence_histogram,
                        markov_heatmap, top_patterns, confidence_evolution, etc.
  --output-dir DIR      Output directory (default: results/figures)
  --format FMT          Image format: png, pdf, svg (default: png)
  --dpi DPI             Resolution (default: 300)

Examples:
  # Generate all figures from all available data
  python level1_figures.py --all
  
  # Compare variants B and A at iteration 18
  python level1_figures.py --variants B A --iterations 18
  
  # Evolution figures only for variant B
  python level1_figures.py --variants B --figures confidence_evolution determinism_evolution
  
  # Generate PDF figures for publication
  python level1_figures.py --all --format pdf --dpi 600
```

---

## Data Discovery Logic

```python
def discover_data():
    """Scan results/ for available Level 1 analysis files."""
    pattern = "results/level1_analysis_var*_iter*_min*_max*.json"
    files = glob.glob(pattern)
    
    data = {}  # {variant: {iteration: filepath}}
    for f in files:
        match = re.search(r'var([A-Z])_iter(\d+)_min(\d+)_max(\d+)', f)
        if match:
            variant, iteration, min_len, max_len = match.groups()
            if variant not in data:
                data[variant] = {}
            data[variant][int(iteration)] = {
                'path': f,
                'min_len': int(min_len),
                'max_len': int(max_len)
            }
    return data
```

---

## Figure Generation Logic

```python
def generate_figures(data, variants=None, iterations=None, figures=None):
    """Generate requested figures based on available data."""
    
    # Filter data by requested variants/iterations
    filtered = filter_data(data, variants, iterations)
    
    # Determine what can be generated
    variant_list = list(filtered.keys())
    
    # Type 1: Comparison figures (need 2+ variants with same iteration)
    if len(variant_list) >= 2:
        common_iters = find_common_iterations(filtered)
        for iter_n in common_iters:
            for v1, v2 in itertools.combinations(variant_list, 2):
                generate_comparison_figures(v1, v2, iter_n, filtered)
    
    # Type 2: Evolution figures (need 1 variant with 2+ iterations)
    for variant in variant_list:
        iters = sorted(filtered[variant].keys())
        if len(iters) >= 2:
            generate_evolution_figures(variant, iters, filtered)
    
    # Type 3: Single figures (any variant/iteration)
    for variant in variant_list:
        for iter_n in filtered[variant]:
            generate_single_figures(variant, iter_n, filtered)
```

---

## Style Guidelines for Publication

```python
# Figure style for scientific papers
STYLE = {
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
}

# Color palette (colorblind-friendly)
COLORS = {
    'A': '#E69F00',  # Orange (control/random)
    'B': '#0072B2',  # Blue (HSI main)
    'D': '#009E73',  # Green
    'E': '#CC79A7',  # Pink
    'F': '#F0E442',  # Yellow
}
```

---

## Output Summary

When the script runs, it will print a summary:

```
📊 Level 1 Figure Generator
===========================

🔍 Data discovered:
   Variant A: iterations [18]
   Variant B: iterations [17, 18, 19]

📈 Figures to generate:
   Comparison (B vs A, iter 18):
     ✓ confidence_comparison
     ✓ confidence_histogram
     ✓ patterns_comparison
   
   Evolution (B, iters 17-19):
     ✓ confidence_evolution
     ✓ determinism_evolution
     ✓ patterns_evolution
   
   Single:
     ✓ markov_heatmap (A iter 18, B iter 17/18/19)
     ✓ top_patterns (A iter 18, B iter 17/18/19)

💾 Output directory: results/figures/
📐 Format: PNG @ 300 DPI

Generating... [████████████████████] 12/12 figures

✅ Complete! Generated 12 figures.
```

---

## Implementation Priority

1. **Phase 1 (Core):** Data discovery + single figures (markov_heatmap, top_patterns)
2. **Phase 2 (Comparison):** Comparison figures between variants
3. **Phase 3 (Evolution):** Evolution over iterations

---

## Dependencies

```python
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmaps
import numpy as np
import json
import glob
import re
from pathlib import Path
import argparse
```

All should already be available in the project venv.

