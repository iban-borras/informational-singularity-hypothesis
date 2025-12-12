#!/usr/bin/env python3
"""
📈 HSI Level 1 - Trend Analysis

Analyzes evolution of HSI metrics (Emergence Index, SCI, ICC, φ-tendency)
across multiple iterations to identify trends and extrapolate to future iterations.

Usage:
    python level1_trend_analysis.py --variants B E I
    python level1_trend_analysis.py --variants B --extrapolate 30
    python level1_trend_analysis.py --from-cache results/trend_cache.json --plot
    python level1_trend_analysis.py --variants B --streaming  # Full sequence analysis

Author: Iban Borràs with Augment Agent (Sophia) collaboration
Date: December 2024
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

# Project paths
BASE_PATH = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_PATH / "results" / "level0" / "phi_snapshots"
RESULTS_DIR = BASE_PATH / "results"

# Fallback to old path if new structure doesn't exist yet
if not SNAPSHOTS_DIR.exists():
    SNAPSHOTS_DIR = BASE_PATH / "results" / "phi_snapshots"


def discover_available_iterations(variant: str) -> List[int]:
    """
    Scan phi_snapshots/var_{X}/ to find which iterations exist.
    
    Returns sorted list of available iteration numbers.
    """
    var_dir = SNAPSHOTS_DIR / f"var_{variant}"
    
    if not var_dir.exists():
        print(f"   ⚠️ Directory not found: {var_dir}")
        return []
    
    iterations = []
    
    # Look for phi_iter{N}.struct.gz (v33 format) or phi_iter{N}.bin.gz (v32)
    for pattern in ["phi_iter*.struct.gz", "phi_iter*.bin.gz"]:
        for path in var_dir.glob(pattern):
            # Extract iteration number from filename
            name = path.stem.replace('.struct', '').replace('.bin', '')
            try:
                iter_num = int(name.replace('phi_iter', ''))
                if iter_num not in iterations:
                    iterations.append(iter_num)
            except ValueError:
                continue
    
    return sorted(iterations)


def get_file_size_mb(variant: str, iteration: int) -> float:
    """Get the compressed file size in MB for a given iteration."""
    var_dir = SNAPSHOTS_DIR / f"var_{variant}"
    
    for ext in [".struct.gz", ".bin.gz"]:
        path = var_dir / f"phi_iter{iteration}{ext}"
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
    
    return 0.0


def print_discovery_summary(discoveries: Dict[str, List[int]]):
    """Print a nice summary of discovered iterations."""
    print("\n" + "=" * 60)
    print("📁 DISCOVERED ITERATIONS")
    print("=" * 60)
    
    for variant, iters in sorted(discoveries.items()):
        if iters:
            min_i, max_i = min(iters), max(iters)
            total_size = sum(get_file_size_mb(variant, i) for i in iters)
            print(f"   Variant {variant}: {len(iters)} iterations ({min_i}-{max_i})")
            print(f"              Total size: {total_size:.1f} MB")
            print(f"              Iterations: {iters}")
        else:
            print(f"   Variant {variant}: No data found")
        print()


def generate_trend_plots(results: Dict, output_dir: Path, extrapolate_to: int = 30):
    """
    Generate trend plots for each metric across iterations.
    Includes linear regression extrapolation.

    File naming convention: trend_{variants}_{metric}_iter{min}-{max}.png
    Example: trend_B_E_I_emergence_index_iter17-23.png
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import linregress
    except ImportError:
        print("   ⚠️ matplotlib/scipy not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename suffix from variants and iteration range
    variants_in_results = sorted(results['variants'].keys())
    all_iterations = []
    for v_data in results['variants'].values():
        all_iterations.extend([int(i) for i in v_data.keys()])

    if all_iterations:
        iter_min, iter_max = min(all_iterations), max(all_iterations)
        iter_suffix = f"iter{iter_min}-{iter_max}"
    else:
        iter_suffix = "iter_unknown"

    variants_str = "_".join(variants_in_results)
    file_prefix = f"trend_{variants_str}_{iter_suffix}"

    metrics = ['emergence_index', 'sci', 'icc', 'phi_tendency']
    metric_labels = {
        'emergence_index': 'Emergence Index',
        'sci': 'Structured Complexity Index (SCI)',
        'icc': 'Cosmological Coherence Index (ICC)',
        'phi_tendency': 'φ-Tendency'
    }

    colors = {'B': '#2ecc71', 'E': '#3498db', 'I': '#9b59b6',
              'A': '#e74c3c', 'D': '#f39c12', 'F': '#1abc9c'}

    # Collect all extrapolations: {variant: {metric: {slope, intercept, r2, predicted}}}
    all_extrapolations = {}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for variant, iter_data in results['variants'].items():
            if not iter_data:
                continue

            iterations = sorted([int(i) for i in iter_data.keys()])
            values = [iter_data[str(i) if str(i) in iter_data else i][metric]
                     for i in iterations]

            color = colors.get(variant, '#7f8c8d')

            # Plot actual data
            ax.plot(iterations, values, 'o-', color=color,
                   label=f'Variant {variant}', linewidth=2, markersize=8)

            # Linear regression for extrapolation
            if len(iterations) >= 2:
                slope, intercept, r_value, _, _ = linregress(iterations, values)

                # Extrapolate
                future_iters = list(range(max(iterations), extrapolate_to + 1))
                future_values = [slope * x + intercept for x in future_iters]

                ax.plot(future_iters, future_values, '--', color=color,
                       alpha=0.5, linewidth=1.5)

                # Store extrapolation per variant per metric
                if variant not in all_extrapolations:
                    all_extrapolations[variant] = {}
                all_extrapolations[variant][metric] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    f'predicted_{extrapolate_to}': slope * extrapolate_to + intercept
                }

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f'HSI {metric_labels[metric]} Trend Analysis', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=extrapolate_to, color='gray', linestyle=':', alpha=0.5)

        # Save plot with unique name
        plot_path = output_dir / f"{file_prefix}_{metric}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"   📊 Saved {plot_path.name}")

    # Generate combined plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for variant, iter_data in results['variants'].items():
            if not iter_data:
                continue

            iterations = sorted([int(i) for i in iter_data.keys()])
            values = [iter_data[str(i) if str(i) in iter_data else i][metric]
                     for i in iterations]

            color = colors.get(variant, '#7f8c8d')
            ax.plot(iterations, values, 'o-', color=color,
                   label=f'{variant}', linewidth=2, markersize=6)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'HSI Metrics Trend Analysis ({variants_str} @ {iter_suffix})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    combined_path = output_dir / f"{file_prefix}_combined.png"
    plt.savefig(combined_path, dpi=150)
    plt.close()

    print(f"   📊 Saved {combined_path.name}")

    return all_extrapolations


def run_emergence_analysis(variant: str, iteration: int,
                           output_dir: Path,
                           streaming: bool = True,
                           chunk_size: int = 50_000_000,
                           max_cpu_percent: int = 50) -> Optional[Dict]:
    """
    Run emergence index + SCI/ICC for a single variant/iteration.

    Args:
        variant: Variant code
        iteration: Iteration number
        output_dir: Output directory
        streaming: If True, process ENTIRE sequence (rigorous, default)
        chunk_size: Bits per chunk for streaming mode
        max_cpu_percent: Maximum CPU usage for parallel processing

    Returns:
        Combined metrics or None if failed
    """
    from metrics.emergence_index import (
        load_phi_sequence, calculate_power_spectrum_slope,
        calculate_lempel_ziv_complexity, calculate_long_range_mutual_info,
        calculate_hierarchical_block_entropy, calculate_dfa,
        calculate_emergence_index_streaming
    )
    from metrics.structured_complexity import (
        calculate_sci, calculate_icc, calculate_transfer_entropy,
        calculate_nontriviality, calculate_phi_tendency
    )

    print(f"\n   📊 Analyzing {variant}@{iteration}{'  [STREAMING MODE]' if streaming else '  [FAST MODE]'}...")

    # === STREAMING MODE: Process entire sequence (DEFAULT) ===
    if streaming:
        result = calculate_emergence_index_streaming(variant, iteration,
                                                     chunk_size=chunk_size,
                                                     max_cpu_percent=max_cpu_percent)

        if 'error' in result:
            print(f"      ⚠️ {result['error']}")
            return None

        # Extract values for SCI/ICC calculation
        emergence_index = result['emergence_index']
        lz_normalized = result['complexity']['avg_lz']
        mi_ratio = result['coherence']['avg_mi']
        hierarchy_score = result['hierarchy']['avg_hierarchy']
        hurst = result['dfa']['avg_hurst']
        slope = result['criticality']['avg_slope']

        # φ-tendency now comes from streaming (full sequence!)
        phi_strength = result.get('phi_tendency', 0.0)

        # Calculate SCI/ICC from streaming results
        # Only need seq for transfer_entropy (TE works well with sample)
        seq = load_phi_sequence(variant, iteration, max_bits=1_000_000)
        if seq:
            te_result = calculate_transfer_entropy(seq)
        else:
            te_result = {'transfer_entropy': 0.1}

        nt = calculate_nontriviality(lz_normalized if not np.isnan(lz_normalized) else 0.5)
        te_normalized = min(1.0, te_result['transfer_entropy'] / 0.5)

        sci_result = calculate_sci(
            hierarchy_score=min(1.0, hierarchy_score * 2) if not np.isnan(hierarchy_score) else 0.5,
            transfer_entropy=te_normalized,
            nontriviality=nt['nontriviality']
        )

        compressibility = 1 - (lz_normalized if not np.isnan(lz_normalized) else 0.5)
        coherence_scaled = min(1.0, (mi_ratio if not np.isnan(mi_ratio) else 0.05) * 5)
        hurst_val = hurst if not np.isnan(hurst) else 0.5
        hurst_factor = max(0, 1 - abs(hurst_val - 0.85) / 0.85)
        slope_val = slope if not np.isnan(slope) else -1
        slope_factor = max(0, 1 - abs(slope_val + 1) / 2)
        criticality = (hurst_factor + slope_factor) / 2

        icc_result = calculate_icc(
            compressibility=compressibility,
            coherence=coherence_scaled,
            hierarchy=min(1.0, hierarchy_score * 2) if not np.isnan(hierarchy_score) else 0.5,
            criticality=criticality,
            phi_tendency=phi_strength,  # Now from full sequence!
            include_phi=True
        )

        print(f"      ✅ EI={emergence_index:.4f} SCI={sci_result['sci']:.4f} ICC={icc_result['icc']:.4f} φ={phi_strength:.4f}")
        print(f"         (Full sequence: {result['total_bits']:,} bits in {result['chunks_processed']} chunks)")

        return {
            'emergence_index': emergence_index,
            'sci': sci_result['sci'],
            'icc': icc_result['icc'],
            'phi_tendency': phi_strength,  # From full sequence!
            'total_bits': result['total_bits'],
            'analysis_mode': 'streaming_full',
            'details': result
        }

    # === SAMPLE MODE: Fast but partial ===
    seq = load_phi_sequence(variant, iteration, max_bits=500_000)
    if seq is None:
        print(f"      ⚠️ Could not load sequence")
        return None

    print(f"      Loaded {len(seq):,} bits")

    # Calculate emergence metrics (with reduced verbosity)
    slope, r2 = calculate_power_spectrum_slope(seq)
    crit = {'slope': slope, 'r_squared': r2, 'score': max(0, 1 - abs(slope + 1))}

    lz_val = calculate_lempel_ziv_complexity(seq, verbose=False)
    lz = {'lz_normalized': lz_val, 'score': 0.7 + lz_val * 0.3 if lz_val < 0.3 else 1 - lz_val}

    mi_val = calculate_long_range_mutual_info(seq)
    mi = {'mi_ratio': mi_val, 'score': min(1.0, mi_val * 5)}

    hier_raw = calculate_hierarchical_block_entropy(seq, verbose=False)
    hier = {
        'raw_hierarchy_score': hier_raw['hierarchy_score'],
        'score': min(1.0, hier_raw['hierarchy_score'] * 2)  # Scale to 0-1
    }

    dfa_raw = calculate_dfa(seq)
    hurst = dfa_raw['hurst_exponent']
    # Score based on proximity to 0.5-0.85 (critical range)
    dfa = {
        'hurst_exponent': hurst,
        'score': hurst if 0.5 <= hurst <= 0.85 else max(0, 1 - abs(hurst - 0.7))
    }

    # Calculate emergence index
    weights = {'criticality': 0.2, 'complexity': 0.15,
               'coherence': 0.2, 'hierarchy': 0.2, 'dfa': 0.25}
    emergence_index = (
        weights['criticality'] * crit['score'] +
        weights['complexity'] * lz['score'] +
        weights['coherence'] * mi['score'] +
        weights['hierarchy'] * hier['score'] +
        weights['dfa'] * dfa['score']
    )

    # Calculate SCI/ICC
    te_result = calculate_transfer_entropy(seq)
    phi_result = calculate_phi_tendency(seq)
    nt = calculate_nontriviality(lz['lz_normalized'])

    te_normalized = min(1.0, te_result['transfer_entropy'] / 0.5)

    sci_result = calculate_sci(
        hierarchy_score=min(1.0, hier['raw_hierarchy_score'] * 2),
        transfer_entropy=te_normalized,
        nontriviality=nt['nontriviality']
    )

    compressibility = 1 - lz['lz_normalized']
    coherence_scaled = min(1.0, mi['mi_ratio'] * 5)
    hurst_factor = max(0, 1 - abs(dfa['hurst_exponent'] - 0.85) / 0.85)
    slope_factor = max(0, 1 - abs(crit['slope'] + 1) / 2)
    criticality = (hurst_factor + slope_factor) / 2

    icc_result = calculate_icc(
        compressibility=compressibility,
        coherence=coherence_scaled,
        hierarchy=min(1.0, hier['raw_hierarchy_score'] * 2),
        criticality=criticality,
        phi_tendency=phi_result['phi_strength'],
        include_phi=True
    )

    return {
        'emergence_index': emergence_index,
        'sci': sci_result['sci'],
        'icc': icc_result['icc'],
        'phi_tendency': phi_result['phi_strength'],
        'phi_detected': phi_result['phi_detected'],
        'hierarchy': hier['raw_hierarchy_score'],
        'coherence': mi['mi_ratio'],
        'hurst': dfa['hurst_exponent'],
        'lz_complexity': lz['lz_normalized']
    }


def run_batch_analysis(discoveries: Dict[str, List[int]],
                       specific_iters: Optional[List[int]] = None,
                       streaming: bool = True,
                       chunk_size: int = 50_000_000,
                       max_cpu_percent: int = 50) -> Dict:
    """
    Run analysis for all discovered variants/iterations.

    Args:
        discoveries: Dict of variant -> available iterations
        specific_iters: If provided, only analyze these iterations
        streaming: If True, process entire sequences (rigorous but slow)
        chunk_size: Bits per chunk in streaming mode

    Returns:
        Consolidated results structure
    """
    results = {
        'generated': datetime.now().isoformat(),
        'analysis_mode': 'streaming_full' if streaming else 'sample_500k',
        'variants': {}
    }

    output_dir = RESULTS_DIR / "trend_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant, available_iters in discoveries.items():
        if not available_iters:
            continue

        # Use specific iterations if provided, else use all available
        iters_to_analyze = specific_iters if specific_iters else available_iters
        # Filter to only those that exist
        iters_to_analyze = [i for i in iters_to_analyze if i in available_iters]

        if not iters_to_analyze:
            print(f"\n⚠️ Variant {variant}: No matching iterations")
            continue

        mode_str = " [STREAMING]" if streaming else ""
        print(f"\n🔬 Variant {variant}: Analyzing {len(iters_to_analyze)} iterations{mode_str}...")

        results['variants'][variant] = {}

        for iteration in iters_to_analyze:
            metrics = run_emergence_analysis(variant, iteration, output_dir,
                                            streaming=streaming, chunk_size=chunk_size,
                                            max_cpu_percent=max_cpu_percent)
            if metrics:
                results['variants'][variant][iteration] = metrics
                # Summary already printed in run_emergence_analysis if streaming

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze HSI metric trends across iterations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python level1_trend_analysis.py --variants B E I
  python level1_trend_analysis.py --variants B --iterations 17 19 21 23
  python level1_trend_analysis.py --discover-only
        """
    )
    
    parser.add_argument('--variants', '-v', nargs='+', default=['B', 'E', 'I'],
                        help='Variants to analyze (default: B E I)')
    parser.add_argument('--iterations', '-i', nargs='+', type=int,
                        help='Specific iterations to analyze (default: auto-discover)')
    parser.add_argument('--discover-only', '-d', action='store_true',
                        help='Only discover available iterations, do not analyze')
    parser.add_argument('--extrapolate', '-e', type=int, default=30,
                        help='Extrapolate to this iteration (default: 30)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file for consolidated results')
    parser.add_argument('--from-cache', type=str,
                        help='Load previous results from cache file')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate trend plots')
    parser.add_argument('--fast', '-f', action='store_true',
                        help='Fast mode: sample 500K bits instead of full sequence (for testing)')
    parser.add_argument('--chunk-size', type=int, default=50_000_000,
                        help='Bits per chunk in streaming mode (default: 50M)')
    parser.add_argument('--max-cpu', type=int, default=50,
                        help='Maximum CPU usage percentage for parallel processing (default: 50)')

    args = parser.parse_args()
    
    print("\n🌌 HSI Trend Analysis")
    print("=" * 60)
    
    # Phase 1: Discover available iterations
    print("\n📁 Phase 1: Discovering available iterations...")
    
    discoveries = {}
    for variant in args.variants:
        iters = discover_available_iterations(variant)
        discoveries[variant] = iters
    
    print_discovery_summary(discoveries)
    
    if args.discover_only:
        print("✅ Discovery complete. Use without --discover-only to run analysis.")
        return

    # Phase 2: Run batch analysis
    # Streaming is DEFAULT (rigorous). Use --fast for quick testing.
    use_streaming = not args.fast
    mode_note = " (sample mode - fast)" if args.fast else " (STREAMING - full sequence, rigorous)"
    print(f"\n⚡ Phase 2: Running emergence + SCI/ICC analysis{mode_note}...")

    results = run_batch_analysis(discoveries, args.iterations,
                                  streaming=use_streaming, chunk_size=args.chunk_size,
                                  max_cpu_percent=args.max_cpu)

    # Phase 3: Save consolidated results using unified structure
    print("\n💾 Phase 3: Saving consolidated results...")

    if args.output:
        output_path = Path(args.output)
    else:
        from utils.file_saver import get_output_path
        output_path = get_output_path(1, "trends", "trend_analysis.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Log with relative path
    try:
        from utils.file_saver import relative_path
        print(f"   💾 Saved: {relative_path(output_path)}")
    except ImportError:
        print(f"   Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TREND SUMMARY")
    print("=" * 60)

    for variant, iter_data in results['variants'].items():
        if not iter_data:
            continue
        iterations = sorted(iter_data.keys(), key=int)
        print(f"\n   Variant {variant}: {len(iterations)} iterations analyzed")
        print(f"   {'Iter':<6} {'EI':<8} {'SCI':<8} {'ICC':<8} {'φ-Tend':<8}")
        print(f"   {'-'*38}")
        for it in iterations:
            m = iter_data[it]
            phi_mark = "🔮" if m.get('phi_detected') else ""
            print(f"   {it:<6} {m['emergence_index']:<8.4f} {m['sci']:<8.4f} "
                  f"{m['icc']:<8.4f} {m['phi_tendency']:<8.4f} {phi_mark}")

    # Phase 4 & 5: Generate plots with extrapolation using unified structure
    if args.plot:
        print(f"\n📈 Phase 4 & 5: Generating trend plots (extrapolating to iter {args.extrapolate})...")
        from utils.file_saver import get_output_path
        plot_dir = get_output_path(1, "trends", "plots")
        extrapolations = generate_trend_plots(results, plot_dir, args.extrapolate)

        if extrapolations:
            print(f"\n📊 EXTRAPOLATION TO ITERATION {args.extrapolate}")
            print("=" * 60)
            pred_key = f'predicted_{args.extrapolate}'
            for variant, metrics_data in sorted(extrapolations.items()):
                print(f"\n   Variant {variant}:")
                print(f"   {'Metric':<20} {'Slope/iter':<12} {'R²':<8} {'@iter {}'.format(args.extrapolate):<10}")
                print(f"   {'-'*52}")
                for metric, ext in metrics_data.items():
                    pred_val = ext.get(pred_key, 0)
                    print(f"   {metric:<20} {ext['slope']:+.6f}   {ext['r_squared']:.3f}    {pred_val:.4f}")

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

