#!/usr/bin/env python3
"""
φ-Alignment Temporal Trend Analysis

Analyzes how φ-alignment evolves across iterations for a given variant.
This helps understand if the system converges toward φ (golden ratio) over time.

Usage:
    python level1_phi_trend.py --variant B --iterations 10,15,19,23
    python level1_phi_trend.py --variant B --range 10-23
    python level1_phi_trend.py --from-results  # Use existing Level 1 results

Author: Iban Borràs with Augment Agent (Sophia)
Date: December 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

PHI = 1.618033988749895


def load_existing_results(variant: str, results_dir: Path) -> List[Dict[str, Any]]:
    """Load existing Level 1 analysis results for a variant."""
    results = []
    pattern = f"var_{variant}_iter*_min*_max*.json"

    for result_file in sorted(results_dir.glob(pattern)):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract iteration from filename
            name = result_file.stem
            iter_part = [p for p in name.split('_') if p.startswith('iter')]
            if iter_part:
                iteration = int(iter_part[0].replace('iter', ''))
                data['iteration'] = iteration
                data['source_file'] = str(result_file)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {result_file}: {e}")

    return sorted(results, key=lambda x: x.get('iteration', 0))


def extract_phi_alignment(result: Dict[str, Any]) -> Optional[float]:
    """Extract φ-alignment from a Level 1 result."""
    validation = result.get('validation', {})
    return validation.get('phi_alignment')


def calculate_trend_metrics(iterations: List[int], phi_alignments: List[float]) -> Dict[str, Any]:
    """Calculate trend statistics for φ-alignment evolution."""
    if len(iterations) < 2:
        return {'error': 'Not enough data points'}

    # Filter valid data
    valid_pairs = [(i, p) for i, p in zip(iterations, phi_alignments) if p is not None]
    if len(valid_pairs) < 2:
        return {'error': 'Not enough valid data points'}

    iters = np.array([p[0] for p in valid_pairs])
    values = np.array([p[1] for p in valid_pairs])

    # Linear regression
    slope, intercept = np.polyfit(iters, values, 1)

    # Correlation coefficient
    correlation = np.corrcoef(iters, values)[0, 1]

    # Is it increasing or decreasing?
    trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

    # Extrapolate: at what iteration would we reach φ-alignment = 1.0?
    if slope > 0:
        iter_to_unity = (1.0 - intercept) / slope
    else:
        iter_to_unity = None

    # Distance from φ for each value (treating them as dimension estimates)
    # Note: phi_alignment is [0,1] where 1 = perfect alignment

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'correlation': float(correlation),
        'trend_direction': trend_direction,
        'iter_to_unity': float(iter_to_unity) if iter_to_unity else None,
        'first_value': float(values[0]),
        'last_value': float(values[-1]),
        'mean_value': float(np.mean(values)),
        'std_value': float(np.std(values)),
        'improvement': float(values[-1] - values[0]),
        'n_points': len(valid_pairs)
    }


def plot_phi_trend(iterations: List[int], phi_alignments: List[float],
                   variant: str, output_path: Optional[Path] = None):
    """Plot φ-alignment evolution across iterations."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping plot")
        return

    # Filter valid data
    valid_pairs = [(i, p) for i, p in zip(iterations, phi_alignments) if p is not None]
    if not valid_pairs:
        print("⚠️  No valid data to plot")
        return

    iters = [p[0] for p in valid_pairs]
    values = [p[1] for p in valid_pairs]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main plot
    ax.plot(iters, values, 'bo-', markersize=8, linewidth=2, label='φ-alignment')

    # Trend line
    if len(iters) >= 2:
        z = np.polyfit(iters, values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(iters), max(iters) * 1.2, 100)
        ax.plot(x_trend, p(x_trend), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')

    # Reference lines
    ax.axhline(y=1.0, color='gold', linestyle=':', alpha=0.8, label='Perfect alignment (1.0)')
    ax.axhline(y=0.618, color='green', linestyle=':', alpha=0.6, label='φ⁻¹ = 0.618')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('φ-alignment', fontsize=12)
    ax.set_title(f'φ-Alignment Evolution — Variant {variant}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_trend_report(variant: str, results: List[Dict[str, Any]],
                       trend_metrics: Dict[str, Any]):
    """Print a formatted trend analysis report."""
    print("\n" + "=" * 60)
    print(f"📈 φ-ALIGNMENT TEMPORAL TREND ANALYSIS — VARIANT {variant}")
    print("=" * 60)

    print(f"\n📊 Data Points: {len(results)} iterations")

    # Table of values
    print("\n┌─────────────┬──────────────┬──────────────────────────────┐")
    print("│  Iteration  │  φ-alignment │  Status                      │")
    print("├─────────────┼──────────────┼──────────────────────────────┤")

    for r in results:
        iteration = r.get('iteration', '?')
        phi = extract_phi_alignment(r)
        if phi is not None:
            if phi > 0.8:
                status = "🌟 Excellent"
            elif phi > 0.6:
                status = "✨ Good"
            elif phi > 0.4:
                status = "⚡ Moderate"
            else:
                status = "⚠️  Low"
            print(f"│  {iteration:>9}  │  {phi:>10.4f}  │  {status:<28} │")
        else:
            print(f"│  {iteration:>9}  │  {'N/A':>10}  │  {'❌ Not available':<28} │")

    print("└─────────────┴──────────────┴──────────────────────────────┘")

    # Trend analysis
    if 'error' not in trend_metrics:
        print(f"\n📈 Trend Analysis:")
        print(f"   Direction: {trend_metrics['trend_direction'].upper()}")
        print(f"   Slope: {trend_metrics['slope']:.6f} per iteration")
        print(f"   Correlation: {trend_metrics['correlation']:.4f}")
        print(f"   First → Last: {trend_metrics['first_value']:.4f} → {trend_metrics['last_value']:.4f}")
        print(f"   Improvement: {trend_metrics['improvement']:+.4f}")

        if trend_metrics['iter_to_unity']:
            print(f"   Extrapolated iteration for φ-alignment = 1.0: ~{trend_metrics['iter_to_unity']:.0f}")

        # Interpretation
        print(f"\n🔮 Interpretation:")
        if trend_metrics['slope'] > 0.01:
            print("   ✅ φ-alignment is INCREASING with iterations")
            print("   → The system is converging toward φ-ordered structures")
        elif trend_metrics['slope'] < -0.01:
            print("   ⚠️  φ-alignment is DECREASING with iterations")
            print("   → The system may be diverging from φ-order")
        else:
            print("   ➡️  φ-alignment is STABLE across iterations")
            print("   → The system maintains consistent φ-order level")
    else:
        print(f"\n⚠️  Trend analysis error: {trend_metrics['error']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze φ-alignment temporal trend across iterations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze from existing results
    python level1_phi_trend.py --variant B --from-results

    # Specify iterations manually
    python level1_phi_trend.py --variant B --iterations 10,15,19,23

    # Specify iteration range
    python level1_phi_trend.py --variant B --range 10-23

    # Save plot to file
    python level1_phi_trend.py --variant B --from-results --output phi_trend_B.png
        """
    )

    parser.add_argument('-v', '--variant', required=True,
                        help='Variant to analyze (A, B, C, D, E, F)')
    parser.add_argument('--from-results', action='store_true',
                        help='Load from existing Level 1 analysis results')
    parser.add_argument('--iterations', type=str,
                        help='Comma-separated list of iterations (e.g., 10,15,19)')
    parser.add_argument('--range', type=str, dest='iter_range',
                        help='Iteration range (e.g., 10-23)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output path for plot (PNG/PDF)')
    parser.add_argument('--results-dir', type=str,
                        default='results/level1/analysis',
                        help='Directory with Level 1 results')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.from_results:
        print(f"📂 Loading existing results for variant {args.variant}...")
        results = load_existing_results(args.variant, results_dir)

        if not results:
            print(f"❌ No results found in {results_dir}")
            print(f"   Run level1_analyze_patterns.py first for variant {args.variant}")
            sys.exit(1)

        print(f"   Found {len(results)} result files")
    else:
        print("❌ Please use --from-results or implement live analysis")
        sys.exit(1)

    # Extract data
    iterations = [r.get('iteration', 0) for r in results]
    phi_alignments = [extract_phi_alignment(r) for r in results]

    # Calculate trend
    trend_metrics = calculate_trend_metrics(iterations, phi_alignments)

    # Print report
    print_trend_report(args.variant, results, trend_metrics)

    # Plot
    output_path = Path(args.output) if args.output else None
    plot_phi_trend(iterations, phi_alignments, args.variant, output_path)

    # Save JSON report
    if output_path:
        json_path = output_path.with_suffix('.json')
        report = {
            'variant': args.variant,
            'iterations': iterations,
            'phi_alignments': phi_alignments,
            'trend_metrics': trend_metrics
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"📝 Report saved to: {json_path}")


if __name__ == "__main__":
    main()
