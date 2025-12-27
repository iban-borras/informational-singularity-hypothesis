#!/usr/bin/env python3
"""
Refresh plots and enriched report for a specific variant/iteration from existing reports/snapshots.
Usage:
  python level0_refresh_plots.py --variant D --iterations 23 [--hilbert-bits 1000000] [--fft-bits 2000000]

This is a thin wrapper that drives level0_generate.py in plot-only mode with explicit arguments,
so you don't need to set environment variables manually.
"""
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "level0_generate.py"

# Note: 'beta' now uses the enhanced version with segmented fits
# Note: 'entropy' generates block entropy analysis (H(L)/L vs L)
VALID_PLOTS = {'growth', 'raster', 'hilbert', 'fft', 'beta', 'autocorr', 'entropy', 'report'}

def main():
    p = argparse.ArgumentParser(description="Refresh plots for an existing variant report (plot-only)")
    p.add_argument("--variant", "-v", required=True, help="Variant code, e.g., D")
    p.add_argument("--iterations", "-i", type=int, required=False, help="Target iteration (optional)")
    p.add_argument("--report-path", "-r", type=str, required=False, help="Direct path to a JSON report")
    # Bits per plot type
    p.add_argument("--hilbert-bits", type=int, default=None, help="Max bits for Hilbert map")
    p.add_argument("--fft-bits", type=int, default=None, help="Max bits for FFT spectrum")
    p.add_argument("--autocorr-bits", type=int, default=None, help="Max bits for autocorrelation")
    p.add_argument("--beta-bits", type=int, default=None, help="Max bits for spectrum β-fit")
    p.add_argument("--raster-bits", type=int, default=None, help="Max bits for 2D raster")
    p.add_argument("--entropy-bits", type=int, default=None, help="Max bits for block entropy")
    p.add_argument("--only", type=str, default=None,
                   help=f"Comma-separated list of plots to generate. Valid: {','.join(sorted(VALID_PLOTS))}")
    args = p.parse_args()

    env = dict(os.environ)
    env["HSI_PLOT_ONLY"] = "1"
    env["HSI_VARIANT_CODE"] = args.variant.upper()
    if args.iterations is not None:
        env["HSI_REPORT_ITER"] = str(args.iterations)
    if args.report_path:
        env["HSI_REPORT_PATH"] = str(Path(args.report_path).resolve())
    if args.hilbert_bits is not None:
        env["HSI_HILBERT_PREFIX"] = str(args.hilbert_bits)
    if args.fft_bits is not None:
        env["HSI_FFT_PREFIX"] = str(args.fft_bits)
    if args.autocorr_bits is not None:
        env["HSI_AUTOCORR_PREFIX"] = str(args.autocorr_bits)
    if args.beta_bits is not None:
        env["HSI_BETA_PREFIX"] = str(args.beta_bits)
    if args.raster_bits is not None:
        env["HSI_RASTER_PREFIX"] = str(args.raster_bits)
    if args.entropy_bits is not None:
        env["HSI_ENTROPY_PREFIX"] = str(args.entropy_bits)

    # Handle --only: skip everything NOT in the list
    if args.only:
        requested = set(x.strip().lower() for x in args.only.split(','))
        invalid = requested - VALID_PLOTS
        if invalid:
            print(f"[WARN] Unknown plot types ignored: {invalid}")
        # Set skip flags for plots NOT requested
        if 'growth' not in requested:
            env["HSI_NO_GROWTH"] = "1"
        if 'raster' not in requested:
            env["HSI_NO_RASTER"] = "1"
        if 'hilbert' not in requested:
            env["HSI_NO_HILBERT"] = "1"
        if 'fft' not in requested:
            env["HSI_NO_FFT"] = "1"
        if 'beta' not in requested:
            env["HSI_NO_BETA"] = "1"
        if 'autocorr' not in requested:
            env["HSI_NO_AUTOCORR"] = "1"
        if 'entropy' not in requested:
            env["HSI_NO_ENTROPY"] = "1"
        if 'report' not in requested:
            env["HSI_NO_REPORT"] = "1"

    # Launch runner in the same Python
    cmd = [sys.executable, str(RUNNER)]
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Runner failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

