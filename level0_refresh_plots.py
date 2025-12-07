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

def main():
    p = argparse.ArgumentParser(description="Refresh plots for an existing variant report (plot-only)")
    p.add_argument("--variant", "-v", required=True, help="Variant code, e.g., D")
    p.add_argument("--iterations", "-i", type=int, required=False, help="Target iteration (optional)")
    p.add_argument("--report-path", "-r", type=str, required=False, help="Direct path to a JSON report")
    p.add_argument("--hilbert-bits", type=int, default=None, help="Max bits for Hilbert map (override config)")
    p.add_argument("--fft-bits", type=int, default=None, help="Max bits for FFT (override config)")
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

    # Launch runner in the same Python
    cmd = [sys.executable, str(RUNNER)]
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Runner failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

