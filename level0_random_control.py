#!/usr/bin/env python3
"""
Variant A: Random Control Generator for HSI Comparison

This module generates pseudo-random bit sequences with the same format
as HSI variants, allowing direct comparison of metrics (fractal dimension,
power spectrum β, Hilbert patterns, etc.) between structured HSI data
and random baselines.

Rationale:
  - Any pattern found in HSI variants should be ABSENT in random data
  - If both show similar patterns, we may have algorithmic artifacts
  - Differences indicate genuine emergent structure in HSI

Usage:
  python -m hsi_agents_project.variant_A_control --iterations 14 --seed 42
  python -m hsi_agents_project.variant_A_control --match-variant B --iterations 14

The generated files are saved in the same format as other variants,
allowing direct use with refresh_variant_from_cache.py for visualization.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import random
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project paths
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "level0" / "phi_snapshots" / "var_A"  # Same location as other variants


def generate_random_bits(num_bits: int, seed: Optional[int] = None, use_csprng: bool = False) -> str:
    """
    Generate pseudo-random bits.

    Args:
        num_bits: Number of bits to generate
        seed: Optional seed for reproducibility (PRNG only)
        use_csprng: If True, use cryptographic RNG (secrets module)

    Returns:
        String of '0' and '1' characters
    """
    # Optimized: generate in chunks using randbits (much faster than bit-by-bit)
    chunk_size = 10_000_000  # 10M bits per chunk
    chunks = []
    remaining = num_bits

    if not use_csprng and seed is not None:
        random.seed(seed)

    while remaining > 0:
        n = min(remaining, chunk_size)
        if use_csprng:
            # Cryptographically secure
            bits = bin(secrets.randbits(n))[2:].zfill(n)
        else:
            # Fast PRNG using random.getrandbits
            bits = bin(random.getrandbits(n))[2:].zfill(n)
        chunks.append(bits)
        remaining -= n
        # Progress feedback for large sequences
        if num_bits > 100_000_000:  # >100M bits
            pct = 100 * (1 - remaining / num_bits)
            print(f"[variant_A]   Generated {pct:.0f}%...", flush=True)

    return ''.join(chunks)[:num_bits]


def get_variant_size(variant: str, iterations: int) -> Optional[int]:
    """Get the size of a variant at a given iteration from its metadata."""
    # Standard structure: level0/phi_snapshots/var_{X}/
    var_dir = ROOT / "results" / "level0" / "phi_snapshots" / f"var_{variant}"
    meta_path = var_dir / f"phi_iter{iterations}.json"

    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                # Try multiple possible field names for size
                size = (
                    meta.get('sequence_length') or  # v33 format
                    meta.get('clean_length') or
                    meta.get('total_length') or
                    (meta.get('format_info', {}).get('length')) or  # nested in format_info
                    0
                )
                if size:
                    return size
        except Exception as e:
            print(f"[DEBUG] Error reading metadata: {e}")

    return None


def generate_control_variant(
    num_bits: int,
    iterations: int,
    seed: Optional[int] = None,
    use_csprng: bool = False
) -> dict:
    """
    Generate random control data in v33-compatible format.

    Uses the same bitarray encoding as other HSI variants to ensure
    the visualization pipeline can read the data correctly.

    Args:
        num_bits: Number of random bits to generate
        iterations: Iteration number (for filename compatibility)
        seed: Random seed (None for CSPRNG)
        use_csprng: Use cryptographic RNG

    Returns:
        Metadata dictionary
    """
    # Import the v33 encoder
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[variant_A] Generating {num_bits:,} random bits (~{num_bits/1e6:.1f}M)...", flush=True)
    print(f"[variant_A] ⏳ This may take a while for large sequences...", flush=True)

    # Generate random bits (no parentheses - pure random sequence)
    bits = generate_random_bits(num_bits, seed=seed, use_csprng=use_csprng)

    gen_time = time.perf_counter() - t0
    print(f"[variant_A] ✅ Generated in {gen_time:.2f}s", flush=True)

    # Save using v33 bitarray encoder (same as other variants)
    struct_path = RESULTS_DIR / f"phi_iter{iterations}.struct.gz"
    print(f"[variant_A] 💾 Saving to {struct_path.name}...", flush=True)

    t1 = time.perf_counter()
    compressed_size = save_phi_structural_gz(bits, str(struct_path))
    save_time = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0

    # Get format info (same structure as variant B)
    format_info = get_format_info(bits)

    # Metadata in exact same format as variant B
    meta = {
        "iteration": iterations,
        "sequence_length": len(bits),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "v33_structural",
        "encoding": "2bit",
        "compressed": True,
        "compression_level": 9,
        "data_directory": str(RESULTS_DIR),
        "format_info": format_info,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": compressed_size / len(bits) if len(bits) > 0 else 0.0,
        # Extra fields for variant A identification
        "variant": "A",
        "variant_description": "Random Control (PRNG/CSPRNG baseline)",
        "seed": seed,
        "generator": "CSPRNG (secrets)" if use_csprng else f"PRNG (Mersenne Twister, seed={seed})",
        "ones_count": bits.count('1'),
        "zeros_count": bits.count('0'),
        "ones_ratio": bits.count('1') / len(bits) if bits else 0.0,
        "generation_time_seconds": gen_time,
        "save_time_seconds": save_time
    }

    meta_path = RESULTS_DIR / f"phi_iter{iterations}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Also create a report file for the pipeline to find
    report_dir = ROOT / "results" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"variant_A_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "variant": "A",
        "variant_description": "Random Control (PRNG/CSPRNG baseline)",
        "iterations": iterations,
        "phi_length": len(bits),
        "total_bits": len(bits),
        "format": "v33_structural",
        "timestamp": datetime.now().isoformat(),
        "execution_time": elapsed,
        "per_iteration": [{"iteration": iterations, "phi_length": len(bits), "time_sec": elapsed}],
        "phi_lengths_per_iteration": [len(bits)],
        "generator_type": "CSPRNG (secrets)" if use_csprng else "PRNG (Mersenne Twister)",
        "seed": seed,
        "ones_ratio": meta['ones_ratio']
    }
    report_path = report_dir / report_name
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[variant_A] ✅ Saved: {struct_path.name} ({struct_path.stat().st_size:,} bytes)")
    print(f"[variant_A] ✅ Metadata: {meta_path.name}")
    print(f"[variant_A] ✅ Report: {report_path.name}")
    print(f"[variant_A] Stats: {meta['ones_count']:,} ones ({meta['ones_ratio']:.4f})")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate random control data (Variant A) for HSI comparison"
    )
    parser.add_argument("--iterations", "-i", type=int, default=None,
                       help="Iteration number (for filename and size matching). "
                            "Can also be set via HSI_ITERATIONS env var.")
    parser.add_argument("--bits", "-b", type=int, default=None,
                       help="Number of bits to generate (overrides --match-variant)")
    parser.add_argument("--match-variant", "-m", type=str, default="B",
                       help="Match size from another variant (default: B)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility (PRNG only)")
    parser.add_argument("--csprng", action="store_true",
                       help="Use cryptographic RNG (ignores --seed)")

    args = parser.parse_args()

    # Support HSI_ITERATIONS env var (for run_all_variants.py integration)
    if args.iterations is None:
        env_iter = os.environ.get("HSI_ITERATIONS")
        if env_iter:
            args.iterations = int(env_iter)
        else:
            parser.error("--iterations/-i is required (or set HSI_ITERATIONS env var)")

    # Determine number of bits
    if args.bits:
        num_bits = args.bits
        print(f"[variant_A] Using specified size: {num_bits:,} bits", flush=True)
    else:
        print(f"[variant_A] Looking up size from variant {args.match_variant} i={args.iterations}...", flush=True)
        num_bits = get_variant_size(args.match_variant, args.iterations)
        if num_bits:
            print(f"[variant_A] Matching variant {args.match_variant}: {num_bits:,} bits ({num_bits/1e6:.1f}M)", flush=True)
        else:
            print(f"[ERROR] Could not find size for variant {args.match_variant} i={args.iterations}")
            print(f"[INFO] Use --bits to specify size manually")
            return

    # Generate
    meta = generate_control_variant(
        num_bits=num_bits,
        iterations=args.iterations,
        seed=args.seed,
        use_csprng=args.csprng
    )

    print(f"\n🎲 Variant A (Random Control) generated successfully!")
    print(f"   To visualize: python -m hsi_agents_project.refresh_variant_from_cache -v A -i {args.iterations}")


if __name__ == "__main__":
    main()

