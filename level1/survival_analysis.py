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


def compute_survival_stats(patterns_by_iter: Dict[int, Dict[str, dict]]) -> dict:
    """Compute survival statistics between iterations."""
    
    sets = {i: set(p.keys()) for i, p in patterns_by_iter.items()}
    
    # Core patterns (present in ALL iterations)
    core = sets[17] & sets[18] & sets[19]
    
    # Transient patterns
    only_17 = sets[17] - sets[18] - sets[19]  # Disappeared after 17
    only_18 = sets[18] - sets[17] - sets[19]  # Only in 18
    only_19 = sets[19] - sets[17] - sets[18]  # Only in 19
    
    # Transitions
    survived_17_to_18 = sets[17] & sets[18]
    survived_18_to_19 = sets[18] & sets[19]
    new_in_18 = sets[18] - sets[17]
    new_in_19 = sets[19] - sets[18]
    lost_17_to_18 = sets[17] - sets[18]
    lost_18_to_19 = sets[18] - sets[19]
    
    return {
        'counts': {i: len(s) for i, s in sets.items()},
        'core': core,
        'core_count': len(core),
        'transitions': {
            '17→18': {
                'survived': len(survived_17_to_18),
                'lost': len(lost_17_to_18),
                'new': len(new_in_18),
                'survival_rate': len(survived_17_to_18) / len(sets[17]) * 100
            },
            '18→19': {
                'survived': len(survived_18_to_19),
                'lost': len(lost_18_to_19),
                'new': len(new_in_19),
                'survival_rate': len(survived_18_to_19) / len(sets[18]) * 100
            }
        },
        'lost_patterns': {
            '17→18': lost_17_to_18,
            '18→19': lost_18_to_19
        },
        'new_patterns': {
            '18': new_in_18,
            '19': new_in_19
        }
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
    """Analyze characteristics of core surviving patterns."""
    
    # Compare recurrence growth for core patterns
    growth_ratios = []
    for p in core:
        r17 = patterns_by_iter[17][p]['recurrence']
        r18 = patterns_by_iter[18][p]['recurrence']
        r19 = patterns_by_iter[19][p]['recurrence']
        growth_ratios.append({
            'pattern': p,
            'r17': r17, 'r18': r18, 'r19': r19,
            'growth_17_18': r18 / r17 if r17 > 0 else 0,
            'growth_18_19': r19 / r18 if r18 > 0 else 0
        })
    
    # Sort by total recurrence
    growth_ratios.sort(key=lambda x: x['r19'], reverse=True)
    
    return {
        'count': len(core),
        'top_10_stable': growth_ratios[:10],
        'avg_growth_17_18': np.mean([g['growth_17_18'] for g in growth_ratios]),
        'avg_growth_18_19': np.mean([g['growth_18_19'] for g in growth_ratios])
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

    # Load patterns from each iteration
    patterns_by_iter = {}
    for iter_num in iterations:
        data = variant_data[iter_num].get('data', {})
        patterns_by_iter[iter_num] = load_patterns_from_data(data)

    if not all(patterns_by_iter.values()):
        if verbose:
            print(f"   ⚠️  Some iterations have no pattern data")
        return generated

    # Compute statistics
    stats = compute_survival_stats(patterns_by_iter)
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

    # Figure 1: Pattern Flow
    fig, ax = plt.subplots(figsize=(12, 6))

    counts = stats['counts']
    trans = stats['transitions']

    n_iters = len(iterations)
    x_positions = np.linspace(0.15, 0.85, n_iters)

    for i, (iter_num, x) in enumerate(zip(iterations, x_positions)):
        circle = plt.Circle((x, 0.5), 0.12, color='steelblue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, 0.5, f"Iter {iter_num}\n{counts[iter_num]:,}",
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Draw arrows between consecutive iterations
    for i in range(n_iters - 1):
        x1, x2 = x_positions[i], x_positions[i + 1]
        iter1, iter2 = iterations[i], iterations[i + 1]
        trans_key = f"{iter1}→{iter2}"

        if trans_key in trans:
            t = trans[trans_key]
            ax.annotate('', xy=(x2 - 0.15, 0.5), xytext=(x1 + 0.15, 0.5),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, 0.68, f"Survived: {t['survived']:,}\n({t['survival_rate']:.1f}%)",
                    ha='center', fontsize=9, color='green')
            ax.text(mid_x, 0.32, f"Lost: {t['lost']:,}\nNew: {t['new']:,}",
                    ha='center', fontsize=9, color='red')

    ax.text(0.5, 0.92, f"Core Patterns (all iters): {stats['core_count']:,}",
            ha='center', fontsize=14, fontweight='bold', color='darkgreen')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Pattern Survival Flow - Variant {variant}', fontsize=14)

    fname = f'survival_flow_{variant}.{fmt}'
    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=dpi, bbox_inches='tight')
    plt.close()
    generated.append(fname)

    # Figure 2: Recurrence Distribution (if we have threshold analysis)
    if threshold_analysis and len(iterations) >= 2:
        iter1, iter2 = iterations[0], iterations[1]
        trans_key = f"{iter1}→{iter2}"

        if trans_key in stats['lost_patterns']:
            lost = stats['lost_patterns'][trans_key]
            survived = set(patterns_by_iter[iter1].keys()) & set(patterns_by_iter[iter2].keys())

            lost_rec = [patterns_by_iter[iter1][p]['recurrence'] for p in lost if p in patterns_by_iter[iter1]]
            survived_rec = [patterns_by_iter[iter1][p]['recurrence'] for p in survived if p in patterns_by_iter[iter1]]

            if lost_rec and survived_rec:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                all_rec = lost_rec + survived_rec
                bins = np.logspace(np.log10(max(1, min(all_rec))), np.log10(max(all_rec)), 50)
                ax.hist(lost_rec, bins=bins, alpha=0.6, label=f'Lost ({len(lost_rec):,})', color='red')
                ax.hist(survived_rec, bins=bins, alpha=0.6, label=f'Survived ({len(survived_rec):,})', color='green')
                if threshold_analysis.get('best_threshold'):
                    ax.axvline(threshold_analysis['best_threshold'], color='black',
                               linestyle='--', label=f"Threshold: {threshold_analysis['best_threshold']:,.0f}")
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
                    ax.set_title('Survival Probability by Recurrence')
                    ax.set_ylim(0, 105)

                fname = f'survival_recurrence_{variant}.{fmt}'
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
    pattern = str(results_dir / f"level1_analysis_var{args.variant}_iter*_min*_max*.json")
    files = glob.glob(pattern)

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

