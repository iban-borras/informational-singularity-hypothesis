#!/usr/bin/env python3
"""
Pattern Survival Analysis - HSI Level 1

Investigates pattern survival across iterations to understand:
- Why pattern count may decrease despite data growth
- Which patterns are statistically robust ("core" patterns)
- Threshold for pattern survival prediction

This module can be used standalone or integrated with level1_figures.py

Author: Sophia (for Iban)
Date: 2025-12-08
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt


def load_patterns_from_file(filepath: Path) -> Dict[str, dict]:
    """
    Load patterns from a Level 1 analysis JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict mapping pattern_data strings to their statistics
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = {}
    observable = data.get('patterns', {}).get('observable', [])

    for p in observable:
        patterns[p['pattern_data']] = {
            'recurrence': p['recurrence'],
            'density': p['density'],
            'autosimilarity': p.get('autosimilarity', 0),
            'length': len(p['pattern_data'])
        }

    return patterns


def load_patterns_from_data(analysis_data: dict) -> Dict[str, dict]:
    """
    Extract patterns from already-loaded analysis data.

    Args:
        analysis_data: Loaded JSON data dict

    Returns:
        Dict mapping pattern_data strings to their statistics
    """
    patterns = {}
    observable = analysis_data.get('patterns', {}).get('observable', [])

    for p in observable:
        patterns[p['pattern_data']] = {
            'recurrence': p['recurrence'],
            'density': p['density'],
            'autosimilarity': p.get('autosimilarity', 0),
            'length': len(p['pattern_data'])
        }

    return patterns


def compute_survival_stats(patterns_by_iter: Dict[int, Dict[str, dict]],
                           true_counts: Optional[Dict[int, int]] = None) -> dict:
    """Compute survival statistics between iterations.

    Dynamically handles any set of iterations (not necessarily consecutive).

    Args:
        patterns_by_iter: Dict mapping iteration -> pattern dict
        true_counts: Optional dict of true pattern counts (before truncation).
                     If provided, uses these for display instead of len(patterns).
    """
    iterations = sorted(patterns_by_iter.keys())
    if len(iterations) < 2:
        return {'counts': {}, 'core': set(), 'core_count': 0, 'transitions': {},
                'lost_patterns': {}, 'new_patterns': {}}

    sets = {i: set(p.keys()) for i, p in patterns_by_iter.items()}

    # Core patterns (present in ALL iterations)
    core = sets[iterations[0]]
    for i in iterations[1:]:
        core = core & sets[i]

    # Compute transitions between consecutive available iterations
    transitions = {}
    lost_patterns = {}
    new_patterns = {}

    for idx in range(len(iterations) - 1):
        i_from = iterations[idx]
        i_to = iterations[idx + 1]
        key = f'{i_from}→{i_to}'

        survived = sets[i_from] & sets[i_to]
        lost = sets[i_from] - sets[i_to]
        new = sets[i_to] - sets[i_from]

        survival_rate = len(survived) / len(sets[i_from]) * 100 if sets[i_from] else 0

        transitions[key] = {
            'survived': len(survived),
            'lost': len(lost),
            'new': len(new),
            'survival_rate': survival_rate
        }
        lost_patterns[key] = lost
        new_patterns[str(i_to)] = new

    # Use true_counts if provided, otherwise fall back to len(patterns)
    if true_counts:
        counts = {i: true_counts.get(i, len(sets[i])) for i in iterations}
    else:
        counts = {i: len(s) for i, s in sets.items()}

    return {
        'counts': counts,
        'core': core,
        'core_count': len(core),
        'transitions': transitions,
        'lost_patterns': lost_patterns,
        'new_patterns': new_patterns,
        'iterations': iterations  # Include for reference
    }


def analyze_lost_patterns(patterns_by_iter: Dict[int, Dict[str, dict]], 
                          lost: Set[str], 
                          source_iter: int) -> dict:
    """Analyze characteristics of patterns that were lost."""
    
    if not lost:
        return {}
    
    source = patterns_by_iter[source_iter]
    
    recurrences = [source[p]['recurrence'] for p in lost if p in source]
    densities = [source[p]['density'] for p in lost if p in source]
    lengths = [source[p]['length'] for p in lost if p in source]
    
    unique_lens, len_counts = np.unique(lengths, return_counts=True)
    length_dist = {int(k): int(v) for k, v in zip(unique_lens, len_counts)}

    return {
        'count': len(lost),
        'recurrence': {
            'mean': float(np.mean(recurrences)),
            'median': float(np.median(recurrences)),
            'min': int(np.min(recurrences)),
            'max': int(np.max(recurrences)),
            'std': float(np.std(recurrences))
        },
        'density': {
            'mean': float(np.mean(densities)),
            'median': float(np.median(densities))
        },
        'length_distribution': length_dist
    }


def analyze_survivors(patterns_by_iter: Dict[int, Dict[str, dict]],
                      core: Set[str]) -> dict:
    """Analyze characteristics of core surviving patterns.

    Dynamically handles any set of iterations.
    """
    iterations = sorted(patterns_by_iter.keys())
    if len(iterations) < 2 or not core:
        return {'count': 0, 'top_10_stable': [], 'growth_rates': {}}

    # Compare recurrence growth for core patterns across all iterations
    growth_ratios = []
    for p in core:
        entry = {'pattern': p}
        recurrences = {}
        for i in iterations:
            if p in patterns_by_iter[i]:
                recurrences[i] = patterns_by_iter[i][p]['recurrence']
                entry[f'r{i}'] = recurrences[i]

        # Compute growth between consecutive iterations
        for idx in range(len(iterations) - 1):
            i_from, i_to = iterations[idx], iterations[idx + 1]
            r_from = recurrences.get(i_from, 0)
            r_to = recurrences.get(i_to, 0)
            entry[f'growth_{i_from}_{i_to}'] = r_to / r_from if r_from > 0 else 0

        growth_ratios.append(entry)

    # Sort by recurrence at last iteration
    last_iter = iterations[-1]
    growth_ratios.sort(key=lambda x: x.get(f'r{last_iter}', 0), reverse=True)

    # Compute average growth rates for each transition
    avg_growth = {}
    for idx in range(len(iterations) - 1):
        i_from, i_to = iterations[idx], iterations[idx + 1]
        key = f'growth_{i_from}_{i_to}'
        values = [g[key] for g in growth_ratios if key in g]
        avg_growth[f'{i_from}→{i_to}'] = np.mean(values) if values else 0

    return {
        'count': len(core),
        'top_10_stable': growth_ratios[:10],
        'growth_rates': avg_growth,
        'iterations': iterations
    }


def find_threshold_hypothesis(patterns_by_iter: Dict[int, Dict[str, dict]],
                              stats: dict) -> dict:
    """
    Test if there's a recurrence threshold for pattern survival.

    Analyzes the first transition to find optimal separation threshold.
    """
    iterations = sorted(patterns_by_iter.keys())
    if len(iterations) < 2:
        return {}

    iter1, iter2 = iterations[0], iterations[1]
    trans_key = f"{iter1}→{iter2}"

    if trans_key not in stats.get('lost_patterns', {}):
        return {}

    lost = stats['lost_patterns'][trans_key]
    survived = set(patterns_by_iter[iter1].keys()) & set(patterns_by_iter[iter2].keys())

    if not lost or not survived:
        return {}

    lost_rec = [patterns_by_iter[iter1][p]['recurrence'] for p in lost if p in patterns_by_iter[iter1]]
    survived_rec = [patterns_by_iter[iter1][p]['recurrence'] for p in survived if p in patterns_by_iter[iter1]]

    if not lost_rec or not survived_rec:
        return {}

    # Find threshold that best separates
    all_rec = sorted(set(lost_rec + survived_rec))
    best_threshold = 0
    best_accuracy = 0

    for threshold in all_rec[::max(1, len(all_rec)//100)]:
        correctly_classified = (
            sum(1 for r in lost_rec if r < threshold) +
            sum(1 for r in survived_rec if r >= threshold)
        )
        accuracy = correctly_classified / (len(lost_rec) + len(survived_rec))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return {
        'transition': trans_key,
        'best_threshold': best_threshold,
        'accuracy': best_accuracy * 100,
        'lost_below_threshold': sum(1 for r in lost_rec if r < best_threshold),
        'lost_above_threshold': sum(1 for r in lost_rec if r >= best_threshold),
        'survived_below_threshold': sum(1 for r in survived_rec if r < best_threshold),
        'survived_above_threshold': sum(1 for r in survived_rec if r >= best_threshold)
    }


def run_survival_analysis(
    discovered_data: Dict[str, Dict[int, dict]],
    variant: str,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300,
    verbose: bool = True
) -> List[str]:
    """
    Run survival analysis for a variant with multiple iterations.

    This function is designed to be called from level1_figures.py

    Args:
        discovered_data: Dict from discover_data() {variant: {iter: {'data': dict}}}
        variant: Variant to analyze (e.g., 'B')
        output_dir: Directory for output figures
        fmt: Image format ('png', 'pdf', 'svg')
        dpi: Resolution
        verbose: Print progress messages

    Returns:
        List of generated filenames
    """
    generated = []

    if variant not in discovered_data:
        return generated

    variant_data = discovered_data[variant]
    iterations = sorted(variant_data.keys())

    if len(iterations) < 2:
        if verbose:
            print(f"   ⚠️  Need at least 2 iterations for survival analysis (found {len(iterations)})")
        return generated

    # Load patterns from each iteration and extract true_pattern_count
    patterns_by_iter = {}
    true_counts = {}
    for iter_num in iterations:
        data = variant_data[iter_num].get('data', {})
        patterns_by_iter[iter_num] = load_patterns_from_data(data)

        # Extract true_pattern_count from sampling_metadata if available
        sampling_meta = data.get('patterns', {}).get('sampling_metadata', {})
        true_count = sampling_meta.get('true_pattern_count', 0)
        if true_count > 0:
            true_counts[iter_num] = true_count
            if verbose:
                print(f"   📊 Iter {iter_num}: true_pattern_count = {true_count:,}")

    if not all(patterns_by_iter.values()):
        if verbose:
            print(f"   ⚠️  Some iterations have no pattern data")
        return generated

    # Compute statistics with true counts if available
    stats = compute_survival_stats(patterns_by_iter, true_counts if true_counts else None)
    threshold_analysis = find_threshold_hypothesis(patterns_by_iter, stats)

    # Create output subdirectory
    survival_dir = output_dir / "survival_analysis"
    survival_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures with variant-specific names
    fnames = generate_survival_figures(
        patterns_by_iter, stats, threshold_analysis,
        variant, survival_dir, fmt, dpi
    )
    generated.extend(fnames)

    # Save JSON results
    survivor_analysis = analyze_survivors(patterns_by_iter, stats['core'])

    iter_keys = sorted(patterns_by_iter.keys())
    lost_analysis = {}
    for i in range(len(iter_keys) - 1):
        key = f"{iter_keys[i]}→{iter_keys[i+1]}"
        if key in stats['lost_patterns']:
            lost_analysis[key] = analyze_lost_patterns(
                patterns_by_iter, stats['lost_patterns'][key], iter_keys[i]
            )

    results = {
        'variant': variant,
        'iterations_analyzed': iter_keys,
        'stats': {
            'counts': {str(k): v for k, v in stats['counts'].items()},
            'core_count': stats['core_count'],
            'transitions': stats['transitions']
        },
        'lost_analysis': lost_analysis,
        'survivor_analysis': {
            'count': survivor_analysis['count'],
            'avg_growth': {
                f"{iter_keys[i]}_{iter_keys[i+1]}": survivor_analysis.get(f'avg_growth_{iter_keys[i]}_{iter_keys[i+1]}', 0)
                for i in range(len(iter_keys) - 1)
            },
            'top_10_patterns': [p['pattern'] for p in survivor_analysis['top_10_stable'][:10]]
        },
        'threshold_analysis': threshold_analysis
    }

    json_path = survival_dir / f'survival_analysis_{variant}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"   ✓ Survival analysis for variant {variant}: {len(fnames)} figures")

    return generated


def generate_survival_figures(
    patterns_by_iter: Dict[int, Dict[str, dict]],
    stats: dict,
    threshold_analysis: dict,
    variant: str,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300
) -> List[str]:
    """Generate survival analysis figures with variant-specific names."""

    generated = []
    iterations = sorted(patterns_by_iter.keys())

    # Figure 1: Pattern Flow - Sankey-style Diagram
    counts = stats['counts']
    trans = stats['transitions']
    n_iters = len(iterations)

    # Calculate figure dimensions based on iterations
    fig_width = max(12, 3 * n_iters)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Color palette
    COLOR_SURVIVED = '#2ecc71'  # Emerald green
    COLOR_LOST = '#e74c3c'      # Alizarin red
    COLOR_NEW = '#3498db'       # Peter River blue
    COLOR_NODE = '#34495e'      # Wet Asphalt

    # Layout parameters
    x_positions = np.linspace(0.08, 0.92, n_iters)
    node_height = 0.5
    node_width = 0.06
    y_center = 0.5

    # Find max count for normalization
    max_count = max(counts.values()) if counts else 1

    # Draw nodes (rectangles proportional to pattern count)
    for i, (iter_num, x) in enumerate(zip(iterations, x_positions)):
        count = counts[iter_num]
        # Height proportional to count (min 0.15, max 0.5)
        h = max(0.15, 0.5 * (count / max_count))

        rect = plt.Rectangle((x - node_width/2, y_center - h/2),
                              node_width, h,
                              facecolor=COLOR_NODE, edgecolor='white',
                              linewidth=2, zorder=3)
        ax.add_patch(rect)

        # Label above node
        ax.text(x, y_center + h/2 + 0.05, f'Iter {iter_num}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Count inside/below node
        ax.text(x, y_center, f'{count:,}',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=4)

    # Draw flows between nodes
    for i in range(n_iters - 1):
        x1, x2 = x_positions[i], x_positions[i + 1]
        iter1, iter2 = iterations[i], iterations[i + 1]
        trans_key = f"{iter1}→{iter2}"

        if trans_key not in trans:
            continue

        t = trans[trans_key]
        survived = t['survived']
        lost = t['lost']
        new = t['new']
        total_from = counts[iter1]
        total_to = counts[iter2]

        # Normalize flow heights
        h1 = max(0.15, 0.5 * (total_from / max_count))
        h2 = max(0.15, 0.5 * (total_to / max_count))

        # Calculate proportional heights for flows
        if total_from > 0:
            surv_h1 = h1 * (survived / total_from)
            lost_h1 = h1 * (lost / total_from)
        else:
            surv_h1 = lost_h1 = 0

        if total_to > 0:
            surv_h2 = h2 * (survived / total_to)
            new_h2 = h2 * (new / total_to)
        else:
            surv_h2 = new_h2 = 0

        # Flow coordinates
        x_start = x1 + node_width/2
        x_end = x2 - node_width/2
        x_mid = (x_start + x_end) / 2

        # Draw SURVIVED flow (green, center band)
        if survived > 0:
            from matplotlib.patches import FancyBboxPatch, PathPatch
            from matplotlib.path import Path as MPath

            # Curved flow using bezier
            verts = [
                (x_start, y_center + surv_h1/2),  # top-left
                (x_mid, y_center + surv_h1/2),
                (x_mid, y_center + surv_h2/2),
                (x_end, y_center + surv_h2/2),    # top-right
                (x_end, y_center - surv_h2/2),    # bottom-right
                (x_mid, y_center - surv_h2/2),
                (x_mid, y_center - surv_h1/2),
                (x_start, y_center - surv_h1/2),  # bottom-left
                (x_start, y_center + surv_h1/2),  # close
            ]
            codes = [MPath.MOVETO] + [MPath.LINETO] * 7 + [MPath.CLOSEPOLY]
            path = MPath(verts, codes)
            patch = PathPatch(path, facecolor=COLOR_SURVIVED, alpha=0.6,
                              edgecolor=COLOR_SURVIVED, linewidth=1, zorder=2)
            ax.add_patch(patch)

            # Label - use darker green for better contrast on light green flow
            COLOR_SURVIVED_DARK = '#145a24'  # Dark forest green
            ax.text(x_mid, y_center + 0.02, f'{survived:,}',
                    ha='center', va='bottom', fontsize=10,
                    color=COLOR_SURVIVED_DARK, fontweight='bold')
            ax.text(x_mid, y_center - 0.02, f'({t["survival_rate"]:.0f}%)',
                    ha='center', va='top', fontsize=9, color=COLOR_SURVIVED_DARK,
                    fontweight='bold')

        # Draw LOST flow (red, going down)
        if lost > 0:
            y_lost = y_center - h1/2 - 0.12
            ax.annotate('', xy=(x_mid, y_lost),
                        xytext=(x_start + 0.02, y_center - h1/2 + 0.02),
                        arrowprops=dict(arrowstyle='->', color=COLOR_LOST,
                                        lw=max(1, 3 * lost/max_count),
                                        connectionstyle='arc3,rad=-0.2'))
            ax.text(x_mid - 0.04, y_lost, f'−{lost:,}',
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    color=COLOR_LOST)

        # Draw NEW flow (blue, coming from top)
        if new > 0:
            y_new = y_center + h2/2 + 0.12
            ax.annotate('', xy=(x_end - 0.02, y_center + h2/2 - 0.02),
                        xytext=(x_mid, y_new),
                        arrowprops=dict(arrowstyle='->', color=COLOR_NEW,
                                        lw=max(1, 3 * new/max_count),
                                        connectionstyle='arc3,rad=-0.2'))
            ax.text(x_mid + 0.04, y_new, f'+{new:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=COLOR_NEW)

    # Core patterns annotation
    ax.text(0.5, 0.05, f'Core Patterns (present in all iterations): {stats["core_count"]:,}',
            ha='center', fontsize=12, fontweight='bold', color=COLOR_SURVIVED,
            transform=ax.transAxes)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_SURVIVED, alpha=0.6, label='Survived'),
        Line2D([0], [0], color=COLOR_LOST, lw=2, label='Lost'),
        Line2D([0], [0], color=COLOR_NEW, lw=2, label='New'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fancybox=True, framealpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_title(f'Pattern Survival Flow - Variant {variant}', fontsize=14, fontweight='bold')

    fname = f'survival_flow_{variant}.{fmt}'
    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=dpi, bbox_inches='tight')
    plt.close()
    generated.append(fname)

    # Figure 2: Recurrence Distribution for ALL transitions (not just the first)
    # Generate one figure per transition to show complete survival dynamics
    for idx in range(len(iterations) - 1):
        iter1, iter2 = iterations[idx], iterations[idx + 1]
        trans_key = f"{iter1}→{iter2}"

        if trans_key not in stats['lost_patterns']:
            continue

        lost = stats['lost_patterns'][trans_key]
        survived = set(patterns_by_iter[iter1].keys()) & set(patterns_by_iter[iter2].keys())

        lost_rec = [patterns_by_iter[iter1][p]['recurrence'] for p in lost if p in patterns_by_iter[iter1]]
        survived_rec = [patterns_by_iter[iter1][p]['recurrence'] for p in survived if p in patterns_by_iter[iter1]]

        if not lost_rec or not survived_rec:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        all_rec = lost_rec + survived_rec
        bins = np.logspace(np.log10(max(1, min(all_rec))), np.log10(max(all_rec)), 50)
        ax.hist(lost_rec, bins=bins, alpha=0.7, label=f'Lost ({len(lost_rec):,})', 
                color=COLOR_LOST, edgecolor='black', linewidth=0.5, histtype='stepfilled')
        ax.hist(survived_rec, bins=bins, alpha=0.7, label=f'Survived ({len(survived_rec):,})', 
                color=COLOR_SURVIVED, edgecolor='black', linewidth=0.5, histtype='stepfilled')

        # Compute threshold for this specific transition
        all_rec_sorted = sorted(set(lost_rec + survived_rec))
        best_threshold = 0
        best_accuracy = 0
        for threshold in all_rec_sorted[::max(1, len(all_rec_sorted)//100)]:
            correctly_classified = (
                sum(1 for r in lost_rec if r < threshold) +
                sum(1 for r in survived_rec if r >= threshold)
            )
            accuracy = correctly_classified / (len(lost_rec) + len(survived_rec))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        if best_threshold > 0:
            ax.axvline(best_threshold, color='black',
                       linestyle='--', label=f"Threshold: {best_threshold:,.0f}")
        ax.set_xscale('log')
        ax.set_xlabel('Recurrence Count (log scale)')
        ax.set_ylabel('Number of Patterns')
        ax.set_title(f'Recurrence: Lost vs Survived ({iter1}→{iter2})')
        ax.legend()

        ax = axes[1]
        all_patterns = list(patterns_by_iter[iter1].keys())
        rec_survival = [(patterns_by_iter[iter1][p]['recurrence'], p in survived) for p in all_patterns]
        rec_survival.sort()

        n_bins = min(20, len(rec_survival) // 10)
        if n_bins >= 2:
            bin_size = len(rec_survival) // n_bins
            x_vals, y_vals = [], []
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size if i < n_bins - 1 else len(rec_survival)
                chunk = rec_survival[start:end]
                if chunk:
                    avg_rec = np.mean([r for r, _ in chunk])
                    survival_rate = sum(1 for _, s in chunk if s) / len(chunk) * 100
                    x_vals.append(avg_rec)
                    y_vals.append(survival_rate)

            ax.plot(x_vals, y_vals, 'o-', color='steelblue', linewidth=2, markersize=8)
            ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
            ax.set_xscale('log')
            ax.set_xlabel('Average Recurrence (log scale)')
            ax.set_ylabel('Survival Rate (%)')
            ax.set_title(f'Survival Probability by Recurrence ({iter1}→{iter2})')
            ax.set_ylim(0, 105)

        # Use transition-specific filename
        fname = f'survival_recurrence_{variant}_{iter1}to{iter2}.{fmt}'
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=dpi, bbox_inches='tight')
        plt.close()
        generated.append(fname)

    # Figure 3: Pattern Length Evolution
    fig, ax = plt.subplots(figsize=(10, 6))

    for iter_num in iterations:
        lengths = [p['length'] for p in patterns_by_iter[iter_num].values()]
        if lengths:
            unique, counts = np.unique(lengths, return_counts=True)
            ax.plot(unique, counts, 'o-', label=f'Iter {iter_num}', linewidth=2, markersize=6)

    ax.set_xlabel('Pattern Length')
    ax.set_ylabel('Number of Patterns')
    ax.set_title(f'Pattern Length Distribution - Variant {variant}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = f'survival_lengths_{variant}.{fmt}'
    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=dpi, bbox_inches='tight')
    plt.close()
    generated.append(fname)

    return generated


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def main():
    """Standalone execution for direct script usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Pattern Survival Analysis for HSI')
    parser.add_argument('--variant', default='B', help='Variant to analyze (default: B)')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--output-dir', default='results/level1/figures', help='Output directory')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'])
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    results_dir = Path(args.results_dir) / "level1" / "analysis"
    output_dir = Path(args.output_dir)

    print("="*70)
    print("🔬 HSI Pattern Survival Analysis")
    print("="*70)

    # Discover available data
    import glob
    import re

    results_dir.mkdir(parents=True, exist_ok=True)
    # Try both naming conventions
    pattern1 = str(results_dir / f"var_{args.variant}_iter*_min*_max*.json")
    pattern2 = str(results_dir / f"level1_analysis_var{args.variant}_iter*_min*_max*.json")
    files = glob.glob(pattern1) + glob.glob(pattern2)

    if not files:
        print(f"❌ No analysis files found for variant {args.variant}")
        return 1

    # Build discovered_data structure
    discovered = {args.variant: {}}

    for f in files:
        match = re.search(r'iter(\d+)', f)
        if match:
            iter_num = int(match.group(1))
            print(f"  Loading iteration {iter_num}...")
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            discovered[args.variant][iter_num] = {'data': data, 'path': f}

    print(f"\n📋 Found iterations: {sorted(discovered[args.variant].keys())}")

    # Run analysis
    generated = run_survival_analysis(
        discovered, args.variant, output_dir, args.format, args.dpi
    )

    if generated:
        print(f"\n✅ Generated {len(generated)} figures:")
        for fname in generated:
            print(f"   • {fname}")
    else:
        print("\n⚠️  No figures generated (need at least 2 iterations)")

    return 0


if __name__ == '__main__':
    exit(main())

