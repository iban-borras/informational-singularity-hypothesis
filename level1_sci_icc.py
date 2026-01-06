#!/usr/bin/env python3
"""
🌌 HSI Level 1 - Structured Complexity Index (SCI) & Cosmological Coherence Index (ICC)

Advanced analysis to evaluate which HSI variant has the highest potential
for generating universe-like structures with emergent physical laws.

SCI measures "meaningful" complexity (not trivial, not random).
ICC measures compatibility with physical universe generation.

Usage:
    python level1_sci_icc.py --variants A B D E F G --iteration 18
    python level1_sci_icc.py --variant B --iteration 18 --verbose
    python level1_sci_icc.py --from-emergence results/emergence_vars_A_B_D_E_F_G_iter18.json

Author: Iban Borràs with Augment Agent (Sophia) collaboration
Date: December 2024
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from metrics.structured_complexity import (
    analyze_structured_complexity,
    calculate_sci,
    calculate_icc,
    calculate_transfer_entropy,
    calculate_nontriviality,
    calculate_phi_tendency
)
from metrics.emergence_index import load_phi_sequence


def analyze_from_emergence_file(emergence_file: str, verbose: bool = True) -> dict:
    """
    Calculate SCI and ICC using pre-computed emergence data.
    Also loads sequences to compute Transfer Entropy and φ-tendency.

    Supports two formats:
    1. Multi-variant: {"iteration": N, "variants": {"B": {...}, "E": {...}}}
    2. Single-variant: {"emergence_index": 0.74, "criticality": {...}, ...}
       (detected from filename like emergence_B_iter23.json)
    """
    with open(emergence_file) as f:
        emergence_data = json.load(f)

    # Detect format: multi-variant vs single-variant
    if 'variants' in emergence_data:
        # Multi-variant format
        iteration = emergence_data.get('iteration', 18)
        variants_data = emergence_data.get('variants', {})
    else:
        # Single-variant format - extract variant and iteration from filename
        # Expected: emergence_B_iter23.json or emergence_E_I_iter20.json
        filename = Path(emergence_file).stem  # e.g., "emergence_B_iter23"
        parts = filename.replace('emergence_', '').split('_iter')

        if len(parts) == 2:
            variant_str = parts[0]  # e.g., "B" or "B_E_I"
            try:
                iteration = int(parts[1])
            except ValueError:
                iteration = 18
        else:
            # Fallback: try to guess from filename
            variant_str = 'X'
            iteration = 18

        # Single variant extracted from filename
        variants_data = {variant_str: emergence_data}
        print(f"   📁 Single-variant file detected: Variant {variant_str}")

    results = {
        'iteration': iteration,
        'source': emergence_file,
        'variants': {}
    }

    print(f"\n🌌 SCI & ICC Analysis")
    print(f"   Source: {emergence_file}")
    print(f"   Iteration: {iteration}")
    print("=" * 60)

    for variant, em_data in variants_data.items():
        if em_data is None:
            print(f"\n⚠️ Variant {variant}: No emergence data, skipping")
            continue
        
        print(f"\n📊 Analyzing Variant {variant}...")
        
        # Load sequence for Transfer Entropy and φ-tendency
        seq = load_phi_sequence(variant, iteration, max_bits=500_000)
        
        if seq is None:
            print(f"   ⚠️ Could not load sequence, using estimation")
            # Estimate from emergence data only
            lz = em_data.get('complexity', {}).get('lz_normalized', 0.5)
            hierarchy = em_data.get('hierarchy', {}).get('raw_hierarchy_score', 0.3)
            coherence = em_data.get('coherence', {}).get('mi_ratio', 0.1)
            hurst = em_data.get('dfa', {}).get('hurst_exponent', 0.5)
            slope = em_data.get('criticality', {}).get('slope', 0.0)
            
            # Estimate TE and φ (can't compute without sequence)
            te_normalized = 0.5
            phi_strength = 0.0
            nt = calculate_nontriviality(lz)
            
            te_result = {'transfer_entropy': 0.25, 'predictability': 0.5}
            phi_result = {'phi_strength': 0.0, 'phi_detected': False}
        else:
            print(f"   Loaded {len(seq):,} bits")
            
            # Get values from emergence data
            lz = em_data.get('complexity', {}).get('lz_normalized', 0.5)
            hierarchy = em_data.get('hierarchy', {}).get('raw_hierarchy_score', 0.3)
            coherence = em_data.get('coherence', {}).get('mi_ratio', 0.1)
            hurst = em_data.get('dfa', {}).get('hurst_exponent', 0.5)
            slope = em_data.get('criticality', {}).get('slope', 0.0)
            
            # Compute Transfer Entropy
            print("   ⏳ Computing Transfer Entropy...")
            te_result = calculate_transfer_entropy(seq)
            te_normalized = min(1.0, te_result['transfer_entropy'] / 0.5)
            
            # Compute φ-tendency
            print("   🔮 Exploring φ-tendency...")
            phi_result = calculate_phi_tendency(seq)
            phi_strength = phi_result['phi_strength']
            
            # NonTriviality
            nt = calculate_nontriviality(lz)
        
        # Calculate SCI
        sci_result = calculate_sci(
            hierarchy_score=min(1.0, hierarchy * 2),
            transfer_entropy=te_normalized,
            nontriviality=nt['nontriviality']
        )
        
        # Calculate ICC
        compressibility = 1 - lz
        coherence_scaled = min(1.0, coherence * 5)
        hurst_factor = max(0, 1 - abs(hurst - 0.85) / 0.85)
        slope_factor = max(0, 1 - abs(slope + 1) / 2)
        criticality = (hurst_factor + slope_factor) / 2
        
        icc_result = calculate_icc(
            compressibility=compressibility,
            coherence=coherence_scaled,
            hierarchy=min(1.0, hierarchy * 2),
            criticality=criticality,
            phi_tendency=phi_strength,
            include_phi=True
        )
        
        results['variants'][variant] = {
            'sci': sci_result,
            'icc': icc_result,
            'transfer_entropy': te_result,
            'phi_tendency': phi_result,
            'nontriviality': nt,
            'emergence_index': em_data.get('emergence_index', 0)
        }
        
        print(f"   ✅ SCI: {sci_result['sci']:.4f} | ICC: {icc_result['icc']:.4f}")
        if phi_result.get('phi_detected'):
            print(f"   🔮 φ detected! Strength: {phi_strength:.3f}")
    
    # Rank by SCI and ICC
    valid = {k: v for k, v in results['variants'].items() if v}
    sci_ranking = sorted(valid.keys(), key=lambda v: valid[v]['sci']['sci'], reverse=True)
    icc_ranking = sorted(valid.keys(), key=lambda v: valid[v]['icc']['icc'], reverse=True)
    
    results['sci_ranking'] = sci_ranking
    results['icc_ranking'] = icc_ranking
    results['best_sci'] = sci_ranking[0] if sci_ranking else None
    results['best_icc'] = icc_ranking[0] if icc_ranking else None

    return results


def print_summary(results: dict):
    """Print formatted summary of SCI/ICC results."""
    print("\n" + "=" * 70)
    print("📊 STRUCTURED COMPLEXITY INDEX (SCI) RANKING")
    print("=" * 70)
    print(f"{'Rank':<6}{'Variant':<10}{'SCI':<10}{'Hierarchy':<12}{'Causality':<12}{'NonTriv':<10}")
    print("-" * 70)

    for rank, variant in enumerate(results['sci_ranking'], 1):
        v = results['variants'][variant]
        sci = v['sci']
        print(f"{rank:<6}{variant:<10}{sci['sci']:<10.4f}{sci['hierarchy_score']:<12.4f}"
              f"{sci['causality_score']:<12.4f}{sci['nontriviality_score']:<10.4f}")

    print("\n" + "=" * 70)
    print("🌌 COSMOLOGICAL COHERENCE INDEX (ICC) RANKING")
    print("=" * 70)
    print(f"{'Rank':<6}{'Variant':<10}{'ICC':<10}{'Compress':<12}{'Coherence':<12}{'φ-Tend':<10}")
    print("-" * 70)

    for rank, variant in enumerate(results['icc_ranking'], 1):
        v = results['variants'][variant]
        icc = v['icc']
        print(f"{rank:<6}{variant:<10}{icc['icc']:<10.4f}{icc['compressibility']:<12.4f}"
              f"{icc['coherence']:<12.4f}{icc['phi_tendency']:<10.4f}")

    print("\n" + "=" * 70)
    print("🏆 SUMMARY")
    print("=" * 70)

    best_sci = results['best_sci']
    best_icc = results['best_icc']

    if best_sci:
        sci_score = results['variants'][best_sci]['sci']['sci']
        print(f"   Best for Structured Complexity: Variant {best_sci} (SCI = {sci_score:.4f})")

    if best_icc:
        icc_score = results['variants'][best_icc]['icc']['icc']
        print(f"   Best for Universe Generation:   Variant {best_icc} (ICC = {icc_score:.4f})")

    # φ detection summary
    phi_detected = [v for v, data in results['variants'].items()
                    if data.get('phi_tendency', {}).get('phi_detected')]
    if phi_detected:
        print(f"\n   🔮 φ (Golden Ratio) detected in: {', '.join(phi_detected)}")
    else:
        print(f"\n   🔮 φ (Golden Ratio): Not significantly detected at Level 1")
        print(f"      (May emerge at higher levels - this is expected)")

    print("=" * 70)


def print_trend_summary(results: dict):
    """Print formatted summary for trend_analysis.json data."""
    print("\n" + "=" * 70)
    print("📊 SCI & ICC FROM STREAMING ANALYSIS")
    print("=" * 70)
    print(f"{'Variant@Iter':<15}{'EI':<10}{'SCI':<10}{'ICC':<10}{'φ-Tend':<10}{'Bits':<15}{'Mode':<10}")
    print("-" * 70)

    # Sort by ICC (descending)
    sorted_keys = sorted(results['variants'].keys(),
                        key=lambda k: results['variants'][k].get('icc', 0) or 0,
                        reverse=True)

    for key in sorted_keys:
        v = results['variants'][key]
        ei = v.get('emergence_index', 0) or 0
        sci = v.get('sci', 0) or 0
        icc = v.get('icc', 0) or 0
        phi = v.get('phi_tendency', 0) or 0
        bits = v.get('total_bits', 0) or 0
        mode = v.get('analysis_mode', 'unknown')[:8]

        bits_str = f"{bits/1e9:.2f}B" if bits > 1e9 else f"{bits/1e6:.1f}M"
        print(f"{key:<15}{ei:<10.4f}{sci:<10.4f}{icc:<10.4f}{phi:<10.4f}{bits_str:<15}{mode:<10}")

    print("\n" + "=" * 70)
    print("🏆 BEST PERFORMERS (from streaming analysis)")
    print("=" * 70)

    if sorted_keys:
        best_key = sorted_keys[0]
        best = results['variants'][best_key]
        print(f"   🥇 Best ICC: {best_key} (ICC = {best.get('icc', 0):.4f})")
        print(f"      φ-tendency: {best.get('phi_tendency', 0):.4f} (calculated from full sequence)")

    print("=" * 70)


def load_from_trend_analysis(variants: List[str], iterations: List[int]) -> Optional[dict]:
    """
    Try to load pre-computed data from trend_analysis.json.

    Returns dict with data for requested variants/iterations, or None if not available.
    """
    trend_path = Path(__file__).parent / "results" / "level1" / "trends" / "trend_analysis.json"

    if not trend_path.exists():
        return None

    try:
        with open(trend_path) as f:
            trend_data = json.load(f)

        # Check if we have data for all requested variants/iterations
        available = trend_data.get('variants', {})
        results = {'variants': {}, 'source': 'trend_analysis.json'}

        for variant in variants:
            if variant not in available:
                return None  # Missing variant

            var_data = available[variant]
            for iteration in iterations:
                iter_key = str(iteration)
                if iter_key not in var_data:
                    return None  # Missing iteration

                # Extract SCI/ICC data (already computed with streaming!)
                iter_data = var_data[iter_key]
                results['variants'][f"{variant}@{iteration}"] = {
                    'variant': variant,
                    'iteration': iteration,
                    'sci': iter_data.get('sci'),
                    'icc': iter_data.get('icc'),
                    'phi_tendency': iter_data.get('phi_tendency'),
                    'emergence_index': iter_data.get('emergence_index'),
                    'analysis_mode': iter_data.get('analysis_mode', 'streaming_full'),
                    'total_bits': iter_data.get('total_bits')
                }

        return results

    except (json.JSONDecodeError, KeyError):
        return None


def run_trend_analysis(variants: List[str], iterations: List[int],
                       force_mmap: bool = False, max_cpu: int = 50,
                       no_cache: bool = False) -> bool:
    """
    Launch level1_trend_analysis.py to generate streaming data.

    Returns True if successful.
    """
    import subprocess

    mmap_note = " + MMAP" if force_mmap else ""
    cache_note = " (no-cache)" if no_cache else ""
    print(f"\n🚀 Launching trend analysis (streaming mode{mmap_note}{cache_note})...")
    print("   This may take several minutes for large iterations.\n")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "level1_trend_analysis.py"),
        "--variants", *variants,
        "--iterations", *[str(i) for i in iterations],
        "--max-cpu", str(max_cpu)
    ]

    if force_mmap:
        cmd.append("--force-mmap")
    if no_cache:
        cmd.append("--no-cache")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Trend analysis failed: {e}")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate SCI and ICC for HSI variants.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python level1_sci_icc.py --from-emergence results/emergence_vars_A_B_D_E_F_G_iter18.json
  python level1_sci_icc.py --variants A B D E F G --iteration 18
  python level1_sci_icc.py --variant B --iteration 18 --verbose
  python level1_sci_icc.py --from-trend  # Use pre-computed trend_analysis.json
        """
    )

    parser.add_argument('--from-emergence', '-f', type=str,
                        help='Use pre-computed emergence index JSON file')
    parser.add_argument('--from-trend', '-t', action='store_true',
                        help='Use pre-computed trend_analysis.json (streaming mode)')
    parser.add_argument('--variant', '-v', type=str,
                        help='Single variant to analyze')
    parser.add_argument('--variants', nargs='+',
                        help='Multiple variants to analyze')
    parser.add_argument('--iteration', '-i', type=int, default=18,
                        help='Iteration number (default: 18)')
    parser.add_argument('--iterations', nargs='+', type=int,
                        help='Multiple iterations to analyze')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--force-compute', action='store_true',
                        help='Force re-computation even if cached data exists')
    parser.add_argument('--force-mmap', action='store_true',
                        help='Force MMAP mode even for small files (saves RAM, useful for low-memory systems)')
    parser.add_argument('--max-cpu', type=int, default=50,
                        help='Maximum CPU usage percentage for parallel processing (default: 50)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore existing checkpoints and force full recalculation')

    args = parser.parse_args()

    print("\n🌌 HSI Structured Complexity & Cosmological Coherence Analysis")
    print("=" * 60)

    # Determine variants and iterations
    variants = args.variants or ([args.variant] if args.variant else ['B'])
    iterations = args.iterations or [args.iteration]

    if args.from_emergence:
        # Use pre-computed emergence data (legacy mode)
        results = analyze_from_emergence_file(args.from_emergence, verbose=True)
    elif args.from_trend or not args.force_compute:
        # Try to use trend_analysis.json first (streaming data!)
        print(f"📂 Looking for pre-computed streaming data...")
        results = load_from_trend_analysis(variants, iterations)

        if results:
            print(f"   ✅ Found data in trend_analysis.json (streaming mode)")
            print(f"   📊 Loaded {len(results['variants'])} variant-iteration pairs")
        else:
            print(f"   ⚠️ Data not found in trend_analysis.json")
            print(f"   🔄 Will compute using streaming analysis...")

            # Run trend analysis to generate the data
            if run_trend_analysis(variants, iterations,
                                  force_mmap=args.force_mmap, max_cpu=args.max_cpu,
                                  no_cache=args.no_cache):
                # Try loading again
                results = load_from_trend_analysis(variants, iterations)
                if not results:
                    print("❌ Failed to load computed data")
                    return
            else:
                print("❌ Trend analysis failed")
                return
    else:
        # Force compute: run trend analysis
        if run_trend_analysis(variants, iterations,
                              force_mmap=args.force_mmap, max_cpu=args.max_cpu,
                              no_cache=args.no_cache):
            results = load_from_trend_analysis(variants, iterations)
            if not results:
                print("❌ Failed to load computed data")
                return
        else:
            print("❌ Trend analysis failed")
            return

    # Print summary - use appropriate function based on data source
    if results.get('source') == 'trend_analysis.json':
        print_trend_summary(results)
    else:
        print_summary(results)

    # Save results using unified structure
    if args.output:
        output_path = Path(args.output)
    else:
        # Generate filename based on variants and iterations
        from utils.file_saver import get_output_path
        var_str = "_".join(variants)
        iter_str = "_".join(str(i) for i in iterations)
        filename = f"sci_icc_{var_str}_iter{iter_str}.json"
        output_path = get_output_path(1, "metrics", filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Log with relative path
    try:
        from utils.file_saver import relative_path
        print(f"\n💾 Saved: {relative_path(output_path)}")
    except ImportError:
        print(f"\n💾 Results saved to {output_path}")


if __name__ == '__main__':
    main()

