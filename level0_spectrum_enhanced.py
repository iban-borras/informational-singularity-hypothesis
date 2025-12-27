#!/usr/bin/env python3
"""
Enhanced Power Spectrum Analysis for HSI Level 0 data.

This script runs the enhanced spectrum analysis with:
- Logarithmic binning for noise reduction
- Automatic breakpoint detection
- Segmented β fitting to reveal scale transitions

Usage:
    python level0_spectrum_enhanced.py [--variant B] [--iterations 23] [--max-bits 1000000000]
    
Environment variables:
    HSI_ENHANCED_MAX_WINDOWS: Number of Welch windows (default: 512)
    HSI_SPECTRUM_BINS: Number of log bins (default: 80)
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from level0_generate import plot_spectrum_beta_fit_enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Power Spectrum Analysis for HSI"
    )
    parser.add_argument(
        '--variant', '-v', 
        type=str, 
        default='B',
        help='Variant to analyze (default: B)'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=23,
        help='Number of iterations (default: 23)'
    )
    parser.add_argument(
        '--max-bits', '-m',
        type=int,
        default=1_000_000_000,
        help='Maximum bits to analyze (default: 1B)'
    )
    parser.add_argument(
        '--windows', '-w',
        type=int,
        default=512,
        help='Number of Welch windows (default: 512)'
    )
    parser.add_argument(
        '--bins', '-b',
        type=int,
        default=80,
        help='Number of log bins (default: 80)'
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['HSI_ENHANCED_MAX_WINDOWS'] = str(args.windows)
    os.environ['HSI_SPECTRUM_BINS'] = str(args.bins)
    
    print("=" * 60)
    print("🔬 ENHANCED POWER SPECTRUM ANALYSIS")
    print("=" * 60)
    print(f"   Variant: {args.variant}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Max bits: {args.max_bits:,}")
    print(f"   Welch windows: {args.windows}")
    print(f"   Log bins: {args.bins}")
    print("=" * 60)
    print()
    
    # Create results dict for the function
    results_list = [{
        'variant': args.variant,
        'iterations': args.iterations,
    }]
    
    # Run enhanced analysis
    metrics = plot_spectrum_beta_fit_enhanced(
        results_list, 
        max_bits=args.max_bits, 
        skip=False
    )
    
    if metrics and metrics[0]:
        print()
        print("=" * 60)
        print("📊 RESULTS SUMMARY")
        print("=" * 60)
        m = metrics[0]
        print(f"   Best model: {m['best_model']}")
        print(f"   Quality: {m.get('quality', 'N/A')} (avg R²={m.get('avg_r2', 0):.3f})")
        print()
        print("   Segments:")
        for seg in m['segments']:
            print(f"      {seg['segment']}: β={seg['beta']:.3f}, R²={seg['r2']:.3f}")
            print(f"         Freq range: [{seg['freq_range'][0]:.2e}, {seg['freq_range'][1]:.2e}]")
        print("=" * 60)
    else:
        print("[ERROR] No results generated.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

