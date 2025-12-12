# 📁 File Storage Improvement Guide

## Executive Summary

This document analyzes the current file storage patterns across all HSI scripts and proposes a unified, organized structure. The goal is to eliminate confusion, remove unused folders, and establish consistent naming conventions.

---

## 📊 Current State Analysis

### Scripts and Their Default Output Locations

| Script | Output Type | Current Location | Naming Pattern |
|--------|-------------|------------------|----------------|
| `level0/generator.py` | Φ sequences | `results/phi_snapshots/var_{X}/` | `phi_iter{N}.struct.gz`, `.json` |
| `level0/generator.py` | Reports | `results/reports/` | `variant_{X}_{N}_{timestamp}.json` |
| `level0/generator.py` | Temp files | `results/` | `phi_final.npy`, `phi_final.txt`, `phi_metadata.json` |
| `level0_generate.py` | Visualizations | `results/visualizations/` | `hilbert_*.png`, `fft_*.png`, `raster2d_*.png`, `spectrum_*.png` |
| `level0_generate.py` | Master results | `results/` | `hsi_variants_master_results.json` |
| `level1_orchestrator.py` | Analysis | `results/` | `level1_analysis_var{X}_iter{N}_min{M}_max{M}.json` |
| `level1_orchestrator.py` | Reports | `results/` | `level1_analysis_var{X}_iter{N}_min{M}_max{M}.md` |
| `level1_orchestrator.py` | Cache | `results/cache/` | `level1_cache_*.pkl` |
| `level1/level1_figures.py` | Figures | `results/figures/` | `markov_*.png`, `confidence_*.png`, `patterns_*.png` |
| `level1/survival_analysis.py` | Figures | `results/figures/survival_analysis/` | `survival_*.png`, `.json` |
| `level1_emergence_index.py` | Metrics | `results/` | `emergence_var{X}_iter{N}.json` |
| `level1_trend_analysis.py` | Consolidated | `results/` | `trend_analysis.json` |
| `level1_trend_analysis.py` | Plots | `results/trend_plots/` | `trend_*.png` |
| `level1_sci_icc.py` | Metrics | `results/` | `sci_icc_*.json` |
| `emergence_index.py` | Checkpoints | `results/` | `checkpoint_{var}_{iter}.json` |
| `geometric_analyzer.py` | Results | `results/level1/` | `geometric_{X}_iter{N}.json` |

### Unused Folders (Empty or Never Written To)

| Folder | Status | Recommendation |
|--------|--------|----------------|
| `results/patterns/` | **Empty** - Never used | ❌ DELETE |
| `results/rules/` | **Empty** - Never used | ❌ DELETE |
| `results/validation/` | **Empty** - Never used | ❌ DELETE |
| `results/snapshots/` | **Empty** - Superseded by `phi_snapshots/` | ❌ DELETE |
| `results/temp/` | **Temp files** - Should be auto-cleaned | ⚠️ Auto-clean |

### Problems Identified

1. **Root clutter**: Many JSON files dumped directly in `results/` without organization
2. **Inconsistent naming**: Some use `var{X}`, others `_{X}`, others just `{X}`
3. **Duplicate concepts**: `snapshots/` vs `phi_snapshots/` confusion
4. **Unused folders**: `patterns/`, `rules/`, `validation/` exist but are never used
5. **Mixed levels**: Level 0 and Level 1 outputs mixed in same directories
6. **Timestamp inconsistency**: Some files have timestamps, others don't
7. **Path logging**: Not all saves show relative path in logs

---

## 🎯 Proposed Unified Structure

```
results/
├── level0/                              # All Level 0 outputs
│   ├── phi_snapshots/                   # Φ sequence data
│   │   ├── var_B/
│   │   │   ├── phi_iter17.struct.gz
│   │   │   └── phi_iter17.json
│   │   └── var_E/
│   ├── reports/                         # Generator reports
│   │   └── variant_B_18_20241211.json
│   └── visualizations/                  # Level 0 plots
│       ├── hilbert_B_i18_*.png
│       ├── fft_B_i18_*.png
│       └── spectrum_beta_B_i18_*.png
│
├── level1/                              # All Level 1 outputs
│   ├── analysis/                        # Orchestrator analysis
│   │   ├── var_B_iter17_min10_max50.json
│   │   └── var_B_iter17_min10_max50.md
│   ├── metrics/                         # Computed metrics
│   │   ├── emergence_B_iter17.json
│   │   ├── sci_icc_B_iter17.json
│   │   └── geometric_B_iter17.json
│   ├── trends/                          # Trend analysis
│   │   ├── trend_analysis.json
│   │   └── plots/
│   │       └── trend_B_E_iter17-23_*.png
│   └── figures/                         # Level 1 visualizations
│       ├── markov_B_iter17.png
│       ├── patterns_evol_B.png
│       └── survival/
│           ├── survival_analysis_B.json
│           └── survival_flow_B.png
│
├── cache/                               # All caches (auto-managed)
│   └── level1_cache_*.pkl
│
├── temp/                                # Temporary files (auto-cleaned)
│
└── master_results.json                  # Cross-variant comparison
```

---

## 📝 Implementation Tasks

### Phase 1: Delete Unused Folders

```bash
# Remove empty/unused folders
rm -r results/patterns/
rm -r results/rules/
rm -r results/validation/
rm -r results/snapshots/   # superseded by phi_snapshots/
```

### Phase 2: Create Unified Save Helper

Create a centralized `utils/file_saver.py` module:

```python
"""Unified file saving utility for consistent paths and logging."""

from pathlib import Path
from datetime import datetime

BASE_PATH = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_PATH / "results"

def get_output_path(level: int, category: str, filename: str) -> Path:
    """
    Get standardized output path.

    Args:
        level: 0 or 1
        category: 'analysis', 'metrics', 'figures', 'reports', etc.
        filename: The filename (e.g., 'emergence_B_iter17.json')

    Returns:
        Full path for the output file
    """
    path = RESULTS_DIR / f"level{level}" / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: dict, path: Path, log: bool = True) -> Path:
    """Save JSON with consistent logging."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    if log:
        rel_path = path.relative_to(BASE_PATH)
        print(f"💾 Saved: {rel_path}")
    return path

def save_figure(fig, path: Path, dpi: int = 150, log: bool = True) -> Path:
    """Save matplotlib figure with consistent logging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if log:
        rel_path = path.relative_to(BASE_PATH)
        print(f"📊 Saved: {rel_path}")
    return path
```

### Phase 3: Update Scripts

Each script needs modifications to use the new structure:

#### 3.1 `level1_emergence_index.py` / `metrics/emergence_index.py`

**Current:**
```python
output_path = results_dir / f"emergence_var{variant}_iter{args.iteration}.json"
```

**Proposed:**
```python
from utils.file_saver import get_output_path, save_json

output_path = get_output_path(1, "metrics", f"emergence_{variant}_iter{iteration}.json")
save_json(results, output_path)
```

#### 3.2 `level1_sci_icc.py`

**Current:**
```python
output_path = Path(__file__).parent / "results" / f"sci_icc_{var_str}_iter{iter_str}.json"
```

**Proposed:**
```python
output_path = get_output_path(1, "metrics", f"sci_icc_{var_str}_iter{iter_str}.json")
```

#### 3.3 `level1_trend_analysis.py`

**Current:**
```python
output_path = RESULTS_DIR / "trend_analysis.json"
plot_dir = RESULTS_DIR / "trend_plots"
```

**Proposed:**
```python
output_path = get_output_path(1, "trends", "trend_analysis.json")
plot_dir = get_output_path(1, "trends/plots", "").parent
```

#### 3.4 `agents/level1_orchestrator.py`

**Current:**
```python
output_path = f"results/level1_analysis_var{args.variant}_iter{iteration}_min{min_len}_max{max_len}.json"
```

**Proposed:**
```python
output_path = get_output_path(1, "analysis", f"var_{variant}_iter{iteration}_min{min_len}_max{max_len}.json")
```

#### 3.5 `level1/level1_figures.py`

**Current:**
```python
output_dir = script_dir.parent / args.output_dir  # results/figures
```

**Proposed:**
```python
output_dir = get_output_path(1, "figures", "").parent
```

#### 3.6 `level0_generate.py`

**Current:**
```python
VIS_DIR = RESULTS_DIR / "visualizations"
```

**Proposed:**
```python
VIS_DIR = get_output_path(0, "visualizations", "").parent
```

#### 3.7 `agents/geometric_analyzer.py`

**Current:**
```python
output_path = output_dir / f"geometric_{variant}_iter{iteration}.json"
```

**Proposed:**
```python
output_path = get_output_path(1, "metrics", f"geometric_{variant}_iter{iteration}.json")
```

### Phase 4: Move Existing Files

Script to reorganize existing files:

```python
"""Migration script: reorganize results/ folder structure."""

import shutil
from pathlib import Path

RESULTS = Path("results")

def migrate():
    # Create new structure
    for level in [0, 1]:
        for cat in ['analysis', 'metrics', 'figures', 'reports', 'visualizations', 'trends']:
            (RESULTS / f"level{level}" / cat).mkdir(parents=True, exist_ok=True)

    # Move Level 0 visualizations
    old_vis = RESULTS / "visualizations"
    new_vis = RESULTS / "level0" / "visualizations"
    if old_vis.exists():
        for f in old_vis.glob("*"):
            shutil.move(str(f), str(new_vis / f.name))

    # Move Level 0 reports
    old_reports = RESULTS / "reports"
    new_reports = RESULTS / "level0" / "reports"
    if old_reports.exists():
        for f in old_reports.glob("variant_*.json"):
            shutil.move(str(f), str(new_reports / f.name))

    # Move phi_snapshots under level0
    old_phi = RESULTS / "phi_snapshots"
    new_phi = RESULTS / "level0" / "phi_snapshots"
    if old_phi.exists() and not new_phi.exists():
        shutil.move(str(old_phi), str(new_phi))

    # Move Level 1 analysis files
    for f in RESULTS.glob("level1_analysis_*.json"):
        shutil.move(str(f), str(RESULTS / "level1" / "analysis" / f.name))
    for f in RESULTS.glob("level1_analysis_*.md"):
        shutil.move(str(f), str(RESULTS / "level1" / "analysis" / f.name))

    # Move Level 1 metrics
    for f in RESULTS.glob("emergence_*.json"):
        shutil.move(str(f), str(RESULTS / "level1" / "metrics" / f.name))
    for f in RESULTS.glob("sci_icc_*.json"):
        shutil.move(str(f), str(RESULTS / "level1" / "metrics" / f.name))

    # Move trend analysis
    if (RESULTS / "trend_analysis.json").exists():
        shutil.move(str(RESULTS / "trend_analysis.json"),
                   str(RESULTS / "level1" / "trends" / "trend_analysis.json"))

    old_trends = RESULTS / "trend_plots"
    new_trends = RESULTS / "level1" / "trends" / "plots"
    if old_trends.exists():
        for f in old_trends.glob("*"):
            shutil.move(str(f), str(new_trends / f.name))

    # Move figures
    old_figs = RESULTS / "figures"
    new_figs = RESULTS / "level1" / "figures"
    if old_figs.exists():
        for f in old_figs.glob("*"):
            if f.is_file():
                shutil.move(str(f), str(new_figs / f.name))
            elif f.is_dir():
                shutil.move(str(f), str(new_figs / f.name))

    # Clean up old empty directories
    for folder in ['patterns', 'rules', 'validation', 'snapshots',
                   'visualizations', 'reports', 'figures', 'trend_plots']:
        old = RESULTS / folder
        if old.exists() and not any(old.iterdir()):
            old.rmdir()

    print("✅ Migration complete!")

if __name__ == "__main__":
    migrate()
```

---

## 📋 Naming Conventions

### Standardized Patterns

| Type | Pattern | Example |
|------|---------|---------|
| Φ sequences | `phi_iter{N}.struct.gz` | `phi_iter17.struct.gz` |
| Analysis | `var_{X}_iter{N}_min{M}_max{M}.json` | `var_B_iter17_min10_max50.json` |
| Metrics | `{metric}_{X}_iter{N}.json` | `emergence_B_iter17.json` |
| Multi-variant | `{metric}_{X}_{Y}_{Z}_iter{N}.json` | `sci_icc_B_E_I_iter20.json` |
| Figures | `{type}_{X}_iter{N}.png` | `markov_B_iter17.png` |
| Evolution | `{type}_evol_{X}.png` | `patterns_evol_B.png` |
| Trends | `trend_{vars}_iter{min}-{max}_{metric}.png` | `trend_B_E_iter17-23_emergence.png` |
| Checkpoints | `checkpoint_{X}_iter{N}.json` | `checkpoint_B_iter23.json` |

### Logging Standard

All save operations should log:
```
💾 Saved: level1/metrics/emergence_B_iter17.json
📊 Saved: level1/figures/markov_B_iter17.png
```

Always show **relative path** from project root, not absolute path.

---

## ⚠️ Breaking Changes

Files that reference old paths will need updates:
1. Any external scripts reading from `results/emergence_*.json`
2. Documentation referring to old structure
3. README.md Storage Structure section

---

## 🔄 Rollout Plan

1. **Week 1**: Create `utils/file_saver.py`, test with one script
2. **Week 2**: Update all scripts to use new helper
3. **Week 3**: Run migration script on existing results
4. **Week 4**: Update README and documentation

---

## ✅ Checklist

- [ ] Delete unused folders (`patterns/`, `rules/`, `validation/`, `snapshots/`)
- [ ] Create `utils/file_saver.py`
- [ ] Update `metrics/emergence_index.py`
- [ ] Update `level1_sci_icc.py`
- [ ] Update `level1_trend_analysis.py`
- [ ] Update `agents/level1_orchestrator.py`
- [ ] Update `level1/level1_figures.py`
- [ ] Update `level0_generate.py`
- [ ] Update `agents/geometric_analyzer.py`
- [ ] Update `level0/generator.py`
- [ ] Run migration script
- [ ] Update README.md Storage Structure
- [ ] Test all scripts end-to-end

