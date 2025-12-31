#!/usr/bin/env python3
"""
Control Variants Generator for HSI Comparison

This module generates control sequences for null hypothesis testing:
  - Variant A: Pseudo-random bits (PRNG/CSPRNG baseline)
  - Variant J: π (Pi) binary digits (pseudo-random mathematical constant)

These allow direct comparison of metrics (fractal dimension, power spectrum β,
Hilbert patterns, LZ complexity, etc.) between structured HSI data and known baselines.

Rationale:
  - Any pattern found in HSI variants should be ABSENT in random data
  - If both show similar patterns, we may have algorithmic artifacts
  - Differences indicate genuine emergent structure in HSI
  - Variant J (π) serves as a sanity check: if LZ ratio ≈ 0.62, algorithm is biased

Usage:
  # Variant A (Random)
  python -m hsi_agents_project.level0_random_control --iterations 14 --seed 42
  python -m hsi_agents_project.level0_random_control --match-variant B --iterations 14

  # Variant J (Pi binary)
  python -m hsi_agents_project.level0_random_control --variant J --iterations 14

The generated files are saved in the same format as other variants,
allowing direct use with the analysis pipeline.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import random
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Project paths - results are inside hsi_agents_project/
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "level0" / "phi_snapshots" / "var_A"


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


def generate_pi_bits(num_bits: int) -> str:
    """
    Generate binary representation of π digits.

    Uses BCD encoding: each decimal digit (0-9) → 4 bits.
    For num_bits binary bits, we need num_bits/4 decimal digits.

    Args:
        num_bits: Number of binary bits to generate

    Returns:
        String of '0' and '1' characters from π
    """
    # We need num_bits/4 decimal digits (BCD: 4 bits per digit)
    n_digits = (num_bits // 4) + 10  # Extra margin

    print(f"[variant_J] 📐 Computing {n_digits:,} π digits...", flush=True)

    try:
        from mpmath import mp
        # Set precision high enough
        mp.dps = n_digits + 50

        # Time estimate: ~1 min per 1M digits with mpmath
        if n_digits > 1_000_000:
            est_minutes = n_digits / 1_000_000
            print(f"[variant_J] ⏳ Estimated time: ~{est_minutes:.0f} min for {n_digits/1e6:.1f}M digits", flush=True)
            print(f"[variant_J]    (mpmath does not support progress, please wait...)", flush=True)

        t0 = time.perf_counter()
        pi_str = mp.nstr(mp.pi, n_digits + 1, strip_zeros=False)
        calc_time = time.perf_counter() - t0

        if n_digits > 1_000_000:
            print(f"[variant_J] ✅ π calculation completed in {calc_time:.1f}s", flush=True)
        # Remove "3." prefix - we want digits after decimal point
        pi_str = pi_str.replace('.', '')[1:]  # Skip the "3"
        pi_digits = pi_str[:n_digits]
        print(f"[variant_J] ✅ Generated {len(pi_digits):,} π digits using mpmath", flush=True)
    except ImportError:
        print("[variant_J] ⚠️  mpmath not available, using precomputed π digits", flush=True)
        # Fallback: first 1000 digits of π (after decimal point)
        PI_1000 = (
            "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
            "8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196"
            "4428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273"
            "7245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094"
            "3305727036575959195309218611738193261179310511854807446237996274956735188575272489122793818301194912"
            "9833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132"
            "0005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235"
            "4201995611212902196086403441815981362977477130996051870721134999999837297804995105973173281609631859"
            "5024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303"
            "5982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"
        )
        if n_digits > len(PI_1000):
            print(f"[variant_J] ⚠️  Only {len(PI_1000)} digits available (install mpmath for more)", flush=True)
        pi_digits = PI_1000[:n_digits]

    # Convert to binary using BCD (4 bits per decimal digit) with progress
    print(f"[variant_J] 🔄 Converting to binary ({num_bits:,} bits)...", flush=True)
    binary_map = {
        '0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100',
        '5': '0101', '6': '0110', '7': '0111', '8': '1000', '9': '1001'
    }

    # For large sequences, show progress
    if num_bits > 10_000_000:
        chunks = []
        chunk_size = 1_000_000  # Process 1M digits at a time
        total_digits = len(pi_digits)
        for i in range(0, total_digits, chunk_size):
            chunk = pi_digits[i:i+chunk_size]
            chunks.append(''.join(binary_map[d] for d in chunk if d in binary_map))
            if (i + chunk_size) % 10_000_000 == 0:
                pct = min(100, (i + chunk_size) * 100 // total_digits)
                print(f"[variant_J]   Converting: {pct}%...", flush=True)
        pi_binary = ''.join(chunks)
    else:
        pi_binary = ''.join(binary_map[d] for d in pi_digits if d in binary_map)

    return pi_binary[:num_bits]


def estimate_iteration_size(iteration: int) -> int:
    """
    Estimate size in bits for a given iteration.

    Based on observed HSI growth pattern: ~2x per iteration.
    Base size at iteration 1 ≈ 1000 bits.
    """
    base_size = 1000
    return base_size * (2 ** (iteration - 1))


def get_variant_size(variant: str, iterations: int) -> Optional[int]:
    """Get the size of a variant at a given iteration from its metadata, or estimate."""
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
    report_dir = ROOT / "results" / "level0" / "reports"
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

    # Also create phi_metadata_A.json for compatibility with analysis pipeline
    phi_metadata = {
        "variant": "A",
        "max_iterations": iterations,
        "iterations_completed": iterations,
        "final_length": len(bits),
        "use_compression": True,
        "compression_activated": True,
        "basal_pure": False,
        "snapshots_saved": [str(struct_path)],
        "phi_lengths_per_iteration": [len(bits)],
        "total_time_seconds": elapsed
    }
    phi_meta_path = report_dir / "phi_metadata_A.json"
    with open(phi_meta_path, 'w') as f:
        json.dump(phi_metadata, f, indent=2)

    print(f"[variant_A] ✅ Saved: {struct_path.name} ({struct_path.stat().st_size:,} bytes)")
    print(f"[variant_A] ✅ Metadata: {meta_path.name}")
    print(f"[variant_A] ✅ Report: {report_path.name}")
    print(f"[variant_A] ✅ phi_metadata: {phi_meta_path.name}")
    print(f"[variant_A] Stats: {meta['ones_count']:,} ones ({meta['ones_ratio']:.4f})")

    return meta


def generate_pi_variant(num_bits: int, iterations: int) -> dict:
    """
    Generate π (Pi) binary control data in v33-compatible format.

    Variant J serves as a sanity check for the LZ multi-scale analysis:
    - π digits pass most randomness tests (pseudo-random)
    - Expected LZ ratio ≈ 1.0 (no self-similar structure)
    - If LZ ratio ≈ 0.62 for π, the algorithm is biased!

    Args:
        num_bits: Number of binary bits to generate
        iterations: Iteration number (for filename compatibility)

    Returns:
        Metadata dictionary
    """
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_J"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[variant_J] 🥧 Generating {num_bits:,} π binary bits (~{num_bits/1e6:.1f}M)...", flush=True)

    # Generate π bits
    bits = generate_pi_bits(num_bits)

    gen_time = time.perf_counter() - t0
    print(f"[variant_J] ✅ Generated {len(bits):,} bits in {gen_time:.2f}s", flush=True)

    # Save using v33 bitarray encoder
    struct_path = results_dir / f"phi_iter{iterations}.struct.gz"
    print(f"[variant_J] 💾 Saving to {struct_path.name}...", flush=True)

    t1 = time.perf_counter()
    compressed_size = save_phi_structural_gz(bits, str(struct_path))
    save_time = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0

    format_info = get_format_info(bits)

    # Metadata
    meta = {
        "iteration": iterations,
        "sequence_length": len(bits),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "v33_structural",
        "encoding": "2bit",
        "compressed": True,
        "compression_level": 9,
        "data_directory": str(results_dir),
        "format_info": format_info,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": compressed_size / len(bits) if len(bits) > 0 else 0.0,
        "variant": "J",
        "variant_description": "π (Pi) Binary Control - Sanity Check",
        "source": "π decimal digits → BCD binary (4 bits per digit)",
        "ones_count": bits.count('1'),
        "zeros_count": bits.count('0'),
        "ones_ratio": bits.count('1') / len(bits) if bits else 0.0,
        "generation_time_seconds": gen_time,
        "save_time_seconds": save_time
    }

    meta_path = results_dir / f"phi_iter{iterations}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Report file
    report_dir = ROOT / "results" / "level0" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"variant_J_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "variant": "J",
        "variant_description": "π (Pi) Binary Control - Sanity Check",
        "iterations": iterations,
        "phi_length": len(bits),
        "total_bits": len(bits),
        "format": "v33_structural",
        "timestamp": datetime.now().isoformat(),
        "execution_time": elapsed,
        "per_iteration": [{"iteration": iterations, "phi_length": len(bits), "time_sec": elapsed}],
        "phi_lengths_per_iteration": [len(bits)],
        "generator_type": "π BCD binary",
        "ones_ratio": meta['ones_ratio'],
        "expected_lz_ratio": "≈1.0 (pseudo-random, no self-similar structure)",
        "purpose": "Sanity check - if LZ ratio ≈ 0.62, algorithm is biased"
    }
    report_path = report_dir / report_name
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Also create phi_metadata_J.json for compatibility with analysis pipeline
    phi_metadata = {
        "variant": "J",
        "max_iterations": iterations,
        "iterations_completed": iterations,
        "final_length": len(bits),
        "use_compression": True,
        "compression_activated": True,
        "basal_pure": False,
        "snapshots_saved": [str(struct_path)],
        "phi_lengths_per_iteration": [len(bits)],
        "total_time_seconds": elapsed
    }
    phi_meta_path = report_dir / "phi_metadata_J.json"
    with open(phi_meta_path, 'w') as f:
        json.dump(phi_metadata, f, indent=2)

    print(f"[variant_J] ✅ Saved: {struct_path.name} ({struct_path.stat().st_size:,} bytes)")
    print(f"[variant_J] ✅ Metadata: {meta_path.name}")
    print(f"[variant_J] ✅ Report: {report_path.name}")
    print(f"[variant_J] ✅ phi_metadata: {phi_meta_path.name}")
    print(f"[variant_J] Stats: {meta['ones_count']:,} ones ({meta['ones_ratio']:.4f})")

    return meta


def generate_rule30_bits(num_bits: int) -> str:
    """
    Generate bits using Wolfram's Rule 30 cellular automaton.

    Rule 30 is famous for generating pseudo-random behavior from simple rules.
    Used as comparison: deterministic chaos, no self-similar structure expected.

    Args:
        num_bits: Number of bits to generate

    Returns:
        Binary string from Rule 30 center column
    """
    import numpy as np

    # Width needs to be large enough to avoid edge effects
    width = max(2 * num_bits + 1, 1001)
    center = width // 2

    # Initialize with single 1 in center (numpy for speed)
    state = np.zeros(width, dtype=np.uint8)
    state[center] = 1

    bits = []
    for step in range(num_bits):
        # Collect center cell
        bits.append(str(state[center]))

        # Rule 30: new = left XOR (center OR right)
        # Using numpy vectorized operations
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        state = left ^ (state | right)
        # Fix boundaries
        state[0] = 0
        state[-1] = 0

        # Progress for large sequences
        if num_bits > 1_000_000 and step % 1_000_000 == 0:
            print(f"[variant_K]   Rule 30: {100*step/num_bits:.0f}%...", flush=True)

    return ''.join(bits)


def generate_logistic_map_bits(num_bits: int, r: float = 3.99, x0: float = 0.1) -> str:
    """
    Generate bits using the Logistic Map (deterministic chaos).

    x_{n+1} = r * x_n * (1 - x_n)

    With r ≈ 3.99 (near 4.0), produces chaotic behavior.
    Binary conversion: bit = 1 if x_n >= 0.5, else 0

    Args:
        num_bits: Number of bits to generate
        r: Logistic map parameter (default 3.99 for chaos)
        x0: Initial condition

    Returns:
        Binary string from logistic map
    """
    x = x0
    bits = []

    for step in range(num_bits):
        # Generate next value
        x = r * x * (1 - x)
        # Convert to binary
        bits.append('1' if x >= 0.5 else '0')

        # Progress for large sequences
        if num_bits > 10_000_000 and step % 10_000_000 == 0:
            print(f"[variant_L]   Logistic map: {100*step/num_bits:.0f}%...", flush=True)

    return ''.join(bits)


def generate_rule30_variant(num_bits: int, iterations: int) -> dict:
    """
    Generate Rule 30 cellular automaton control data in v33-compatible format.

    Variant K serves as deterministic chaos comparison:
    - Rule 30 is famous for pseudo-random behavior
    - Expected LZ ratio ≈ 1.0 (chaotic, no self-similar structure)

    Args:
        num_bits: Number of binary bits to generate
        iterations: Iteration number (for filename compatibility)

    Returns:
        Metadata dictionary
    """
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_K"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[variant_K] 🔲 Generating {num_bits:,} Rule 30 bits...", flush=True)

    bits = generate_rule30_bits(num_bits)

    gen_time = time.perf_counter() - t0
    print(f"[variant_K] ✅ Generated {len(bits):,} bits in {gen_time:.2f}s", flush=True)

    # Save using v33 bitarray encoder
    struct_path = results_dir / f"phi_iter{iterations}.struct.gz"
    print(f"[variant_K] 💾 Saving to {struct_path.name}...", flush=True)

    t1 = time.perf_counter()
    compressed_size = save_phi_structural_gz(bits, str(struct_path))
    save_time = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0
    format_info = get_format_info(bits)

    meta = {
        "iteration": iterations,
        "sequence_length": len(bits),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "v33_structural",
        "variant": "K",
        "variant_description": "Rule 30 Cellular Automaton - Deterministic Chaos",
        "source": "Wolfram Rule 30 center column",
        "ones_count": bits.count('1'),
        "zeros_count": bits.count('0'),
        "ones_ratio": bits.count('1') / len(bits) if bits else 0.0,
        "compressed_size_bytes": compressed_size,
        "format_info": format_info,
        "generation_time_seconds": gen_time
    }

    meta_path = results_dir / f"phi_iter{iterations}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Report
    report_dir = ROOT / "results" / "level0" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "variant": "K",
        "variant_description": "Rule 30 Cellular Automaton - Deterministic Chaos",
        "iterations": iterations,
        "phi_length": len(bits),
        "total_bits": len(bits),
        "execution_time": elapsed,
        "ones_ratio": meta['ones_ratio'],
        "expected_lz_ratio": "≈1.0 (chaotic, no self-similar structure)"
    }
    report_path = report_dir / f"variant_K_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # phi_metadata for pipeline compatibility
    phi_meta = {"variant": "K", "max_iterations": iterations, "final_length": len(bits)}
    with open(report_dir / "phi_metadata_K.json", 'w') as f:
        json.dump(phi_meta, f, indent=2)

    print(f"[variant_K] ✅ Saved: {struct_path.name}")
    return meta


def generate_logistic_variant(num_bits: int, iterations: int) -> dict:
    """
    Generate Logistic Map control data in v33-compatible format.

    Variant L serves as deterministic chaos comparison:
    - Logistic map at r≈4 produces chaotic dynamics
    - Expected LZ ratio ≈ 1.0 (chaotic, no self-similar structure)

    Args:
        num_bits: Number of binary bits to generate
        iterations: Iteration number (for filename compatibility)

    Returns:
        Metadata dictionary
    """
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_L"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[variant_L] 📈 Generating {num_bits:,} Logistic Map bits...", flush=True)

    bits = generate_logistic_map_bits(num_bits)

    gen_time = time.perf_counter() - t0
    print(f"[variant_L] ✅ Generated {len(bits):,} bits in {gen_time:.2f}s", flush=True)

    # Save using v33 bitarray encoder
    struct_path = results_dir / f"phi_iter{iterations}.struct.gz"
    print(f"[variant_L] 💾 Saving to {struct_path.name}...", flush=True)

    t1 = time.perf_counter()
    compressed_size = save_phi_structural_gz(bits, str(struct_path))
    save_time = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0
    format_info = get_format_info(bits)

    meta = {
        "iteration": iterations,
        "sequence_length": len(bits),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "v33_structural",
        "variant": "L",
        "variant_description": "Logistic Map - Deterministic Chaos",
        "source": "Logistic map x_{n+1}=r*x*(1-x) with r=3.99",
        "ones_count": bits.count('1'),
        "zeros_count": bits.count('0'),
        "ones_ratio": bits.count('1') / len(bits) if bits else 0.0,
        "compressed_size_bytes": compressed_size,
        "format_info": format_info,
        "generation_time_seconds": gen_time
    }

    meta_path = results_dir / f"phi_iter{iterations}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Report
    report_dir = ROOT / "results" / "level0" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "variant": "L",
        "variant_description": "Logistic Map - Deterministic Chaos",
        "iterations": iterations,
        "phi_length": len(bits),
        "total_bits": len(bits),
        "execution_time": elapsed,
        "ones_ratio": meta['ones_ratio'],
        "expected_lz_ratio": "≈1.0 (chaotic, no self-similar structure)"
    }
    report_path = report_dir / f"variant_L_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # phi_metadata for pipeline compatibility
    phi_meta = {"variant": "L", "max_iterations": iterations, "final_length": len(bits)}
    with open(report_dir / "phi_metadata_L.json", 'w') as f:
        json.dump(phi_meta, f, indent=2)

    print(f"[variant_L] ✅ Saved: {struct_path.name}")
    return meta


def generate_fibonacci_word(num_bits: int) -> str:
    """
    Generate the Fibonacci word (binary sequence with φ structure).

    The Fibonacci word is constructed recursively:
      S_0 = "0"
      S_1 = "01"
      S_n = S_{n-1} + S_{n-2}

    This produces: 0, 01, 010, 01001, 01001010, 0100101001001, ...

    The ratio of 0s to 1s tends to φ (golden ratio).
    The sequence is inherently self-similar with φ proportions.

    Expected LZ ratio: ≈ 0.62 (1/φ) — POSITIVE CONTROL for φ detection.
    """
    if num_bits <= 0:
        return ""

    # Build Fibonacci words until we have enough bits
    s_prev2 = "0"   # S_0
    s_prev1 = "01"  # S_1

    while len(s_prev1) < num_bits:
        s_new = s_prev1 + s_prev2
        s_prev2 = s_prev1
        s_prev1 = s_new

    return s_prev1[:num_bits]


def generate_fibonacci_variant(num_bits: int, iterations: int) -> dict:
    """
    Generate Fibonacci word control data in v33-compatible format.

    Variant M serves as POSITIVE CONTROL for φ detection:
    - Fibonacci word has φ encoded in its structure
    - Expected LZ ratio ≈ 0.62 (1/φ) — same as HSI variants
    - If algorithm detects φ here, it validates the detection in B

    Args:
        num_bits: Number of binary bits to generate
        iterations: Iteration number (for filename compatibility)

    Returns:
        Metadata dict with file info
    """
    print(f"[variant_M] Generating Fibonacci word ({num_bits:,} bits)...", flush=True)

    # Generate Fibonacci word
    bits_str = generate_fibonacci_word(num_bits)
    actual_bits = len(bits_str)

    # Statistics
    ones = bits_str.count('1')
    zeros = actual_bits - ones
    ratio_01 = zeros / ones if ones > 0 else float('inf')

    print(f"[variant_M] Generated {actual_bits:,} bits", flush=True)
    print(f"[variant_M] 0s: {zeros:,}, 1s: {ones:,}, ratio 0/1: {ratio_01:.6f} (φ≈1.618)", flush=True)

    # Create metadata (v33 format)
    meta = {
        "format_version": 33,
        "variant": "M",
        "variant_name": "Fibonacci Word (φ-structured)",
        "iteration": iterations,
        "total_bits": actual_bits,
        "generation_method": "fibonacci_word",
        "description": "Positive control: Fibonacci word with inherent φ structure",
        "expected_lz_ratio": "≈0.62 (1/φ)",
        "purpose": "Validate φ detection algorithm",
        "ones_count": ones,
        "zeros_count": zeros,
        "ones_ratio": ones / actual_bits if actual_bits > 0 else 0,
        "zeros_to_ones_ratio": ratio_01,
        "phi_reference": 1.6180339887,
        "timestamp": datetime.now().isoformat()
    }

    # Save structure file
    results_dir = Path("results/level0")
    results_dir.mkdir(parents=True, exist_ok=True)
    struct_path = results_dir / f"var_M_iter{iterations}_structure.json"

    # Pack bits into bytes
    bit_array = bitarray(bits_str)
    packed_bytes = bit_array.tobytes()
    encoded_data = base64.b64encode(packed_bytes).decode('ascii')

    structure = {
        "metadata": meta,
        "phi_sequence": {
            "total_bits": actual_bits,
            "packed_data_base64": encoded_data,
            "packed_bytes": len(packed_bytes),
            "compression": "none",
            "encoding": "bitarray"
        }
    }

    with open(struct_path, 'w') as f:
        json.dump(structure, f, indent=2)

    print(f"[variant_M] ✅ Saved: {struct_path.name}")
    return meta


def main():
    # Debug: show that we entered main()
    print(f"[DEBUG] level0_random_control.main() started", flush=True)
    print(f"[DEBUG] HSI_VARIANT_CODE={os.environ.get('HSI_VARIANT_CODE', 'NOT SET')}", flush=True)
    print(f"[DEBUG] HSI_ITERATIONS={os.environ.get('HSI_ITERATIONS', 'NOT SET')}", flush=True)

    parser = argparse.ArgumentParser(
        description="Generate control variants for HSI comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Control Variants:
  A = Random (PRNG/CSPRNG baseline)
  J = π (Pi) binary (pseudo-random mathematical constant)
  K = Rule 30 (Wolfram cellular automaton - deterministic chaos)
  L = Logistic Map (x_{n+1}=r*x*(1-x) - deterministic chaos)
  M = Fibonacci Word (POSITIVE CONTROL - has φ structure)

Examples:
  # Variant A (Random control)
  python -m hsi_agents_project.level0_random_control -i 15 --match-variant B

  # Variant J (Pi binary - sanity check)
  python -m hsi_agents_project.level0_random_control --variant J -i 15

  # Variant K (Rule 30 - cellular automaton)
  python -m hsi_agents_project.level0_random_control --variant K -i 15

  # Variant L (Logistic Map - chaos)
  python -m hsi_agents_project.level0_random_control --variant L -i 15

  # Variant M (Fibonacci - positive control for φ)
  python -m hsi_agents_project.level0_random_control --variant M -i 15
        """
    )
    parser.add_argument("--variant", "-v", type=str, default="A", choices=["A", "J", "K", "L", "M"],
                       help="Variant to generate: A=Random, J=Pi, K=Rule30, L=Logistic (default: A)")
    parser.add_argument("--iterations", "-i", type=int, default=None,
                       help="Iteration number (for filename and size matching). "
                            "Can also be set via HSI_ITERATIONS or HSI_VARIANT_CODE env vars.")
    parser.add_argument("--bits", "-b", type=int, default=None,
                       help="Number of bits to generate (overrides --match-variant)")
    parser.add_argument("--match-variant", "-m", type=str, default="B",
                       help="Match size from another variant (default: B)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility (Variant A PRNG only)")
    parser.add_argument("--csprng", action="store_true",
                       help="Use cryptographic RNG for Variant A (ignores --seed)")

    args = parser.parse_args()

    # Support HSI_VARIANT_CODE env var (for level0_generate.py integration)
    variant = os.environ.get("HSI_VARIANT_CODE", args.variant).upper()
    if variant not in ["A", "J", "K", "L", "M"]:
        variant = args.variant.upper()

    # Support HSI_ITERATIONS env var
    if args.iterations is None:
        env_iter = os.environ.get("HSI_ITERATIONS")
        if env_iter:
            args.iterations = int(env_iter)
        else:
            parser.error("--iterations/-i is required (or set HSI_ITERATIONS env var)")

    # Determine number of bits for the target iteration
    if args.bits:
        num_bits = args.bits
        print(f"[variant_{variant}] Using specified size: {num_bits:,} bits", flush=True)
    else:
        # Try to get from reference variant, otherwise estimate
        num_bits = get_variant_size(args.match_variant, args.iterations)
        if num_bits:
            print(f"[variant_{variant}] Matching {args.match_variant}: {num_bits:,} bits ({num_bits/1e6:.1f}M)", flush=True)
        else:
            num_bits = estimate_iteration_size(args.iterations)
            print(f"[variant_{variant}] Estimated size: {num_bits:,} bits (~{num_bits/1e6:.1f}M)", flush=True)

    # Generate single file for the target iteration (same as other variants)
    if variant == "J":
        meta = generate_pi_variant(num_bits=num_bits, iterations=args.iterations)
        print(f"\n🥧 Variant J (π Binary Control) generated successfully!")
        print(f"   Purpose: Sanity check for LZ algorithm - expected ratio ≈ 1.0")
    elif variant == "K":
        meta = generate_rule30_variant(num_bits=num_bits, iterations=args.iterations)
        print(f"\n🔲 Variant K (Rule 30) generated successfully!")
        print(f"   Purpose: Deterministic chaos comparison - expected ratio ≈ 1.0")
    elif variant == "L":
        meta = generate_logistic_variant(num_bits=num_bits, iterations=args.iterations)
        print(f"\n📈 Variant L (Logistic Map) generated successfully!")
        print(f"   Purpose: Deterministic chaos comparison - expected ratio ≈ 1.0")
    elif variant == "M":
        meta = generate_fibonacci_variant(num_bits=num_bits, iterations=args.iterations)
        print(f"\n🌻 Variant M (Fibonacci Word) generated successfully!")
        print(f"   Purpose: POSITIVE CONTROL — expected ratio ≈ 0.62 (1/φ)")
    else:  # Variant A
        meta = generate_control_variant(
            num_bits=num_bits,
            iterations=args.iterations,
            seed=args.seed,
            use_csprng=args.csprng
        )
        print(f"\n🎲 Variant A (Random Control) generated successfully!")

    print(f"   To analyze: python level1_deep_analysis.py -v {variant} -i {args.iterations}")


if __name__ == "__main__":
    main()

