#!/usr/bin/env python3
"""
Control Variants Generator for HSI Comparison

This module generates control sequences for hypothesis testing:
  - Variant A: Pseudo-random bits (PRNG/CSPRNG baseline) - NULL hypothesis
  - Variant J: π (Pi) binary digits - NULL hypothesis (deterministic pseudo-random)
  - Variant K: Rule 30 cellular automaton - NULL hypothesis (deterministic chaos)
  - Variant L: Logistic map at chaos - NULL hypothesis (deterministic chaos)
  - Variant M: Fibonacci word - POSITIVE control (known φ-structure)

Rationale:
  - Any pattern found in ISH variants should be ABSENT in null controls (A, J, K, L)
  - If both show similar patterns, we may have algorithmic artifacts
  - Differences indicate genuine emergent structure in ISH
  - Variant M (Fibonacci) serves as positive control: should show LZ ratio ≈ 0.62

Usage (via level0_generate.py - RECOMMENDED):
  python -m hsi_agents_project.level0_generate --variant A -i 15   # Random
  python -m hsi_agents_project.level0_generate --variant M -i 15   # Fibonacci

The generated files are saved in the same format as ISH variants,
allowing direct use with the analysis pipeline.

Configuration (config.json > level0_generation):
  variant_J_force_calculate: true  - Force mpmath calculation instead of pre-computed cache

Cache Requirements (variant J):
  Download Pi.zip from: https://archive.org/download/Math_Constants/Pi.zip
  Save to: results/cache/Pi.zip (~876 MB, 1 billion digits)
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
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Project paths - results are inside hsi_agents_project/
# Now we're inside level0/, so go up one level to hsi_agents_project/
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "level0" / "phi_snapshots" / "var_A"

# Pi cache configuration
PI_CACHE_DIR = ROOT / "results" / "cache"
PI_CACHE_FILE = PI_CACHE_DIR / "Pi.zip"
PI_ARCHIVE_URL = "https://archive.org/download/Math_Constants/Pi.zip"
PI_CACHE_MAX_DIGITS = 1_000_000_000  # 1 billion digits available


def _read_pi_digits_from_cache(n_digits: int) -> Optional[str]:
    """
    Read π digits from the cached zip file.

    Args:
        n_digits: Number of decimal digits to read (after the decimal point).

    Returns:
        String of π digits, or None if cache unavailable.
    """
    if not PI_CACHE_FILE.exists():
        return None

    if n_digits > PI_CACHE_MAX_DIGITS:
        print(f"[variant_J] ⚠️  Requested {n_digits:,} digits but cache only has {PI_CACHE_MAX_DIGITS:,}", flush=True)
        print(f"[variant_J]    Will use available digits (max iteration ~23)", flush=True)
        n_digits = PI_CACHE_MAX_DIGITS

    try:
        with zipfile.ZipFile(PI_CACHE_FILE, 'r') as zf:
            # Find the decimal digits file (usually "Pi - Dec - Chudnovsky.txt")
            txt_files = [n for n in zf.namelist() if 'Dec' in n and n.endswith('.txt')]
            if not txt_files:
                # Fallback: any .txt file
                txt_files = [n for n in zf.namelist() if n.endswith('.txt')]
            if not txt_files:
                print(f"[variant_J] ❌ No text file found in {PI_CACHE_FILE.name}", flush=True)
                return None

            txt_name = txt_files[0]
            print(f"[variant_J] 📖 Reading {n_digits:,} digits from {txt_name}...", flush=True)

            with zf.open(txt_name) as f:
                # Read enough bytes (digits are ASCII, 1 byte each)
                # File format: "3.14159..." - skip "3." prefix
                header = f.read(2).decode('ascii')  # Read "3."
                if header != "3.":
                    # Some files may start directly with digits
                    content = header + f.read(n_digits).decode('ascii')
                else:
                    content = f.read(n_digits).decode('ascii')

                # Clean: only keep digits
                digits = ''.join(c for c in content if c.isdigit())[:n_digits]
                print(f"[variant_J] ✅ Read {len(digits):,} π digits from cache", flush=True)
                return digits

    except (zipfile.BadZipFile, OSError) as e:
        print(f"[variant_J] ❌ Error reading cache: {e}", flush=True)
        return None


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


def _get_force_calculate_setting() -> bool:
    """
    Check if we should force mpmath calculation instead of using cache.

    Reads from config.json > level0_generation > variant_J_force_calculate.
    Falls back to environment variable HSI_VARIANT_J_FORCE_CALCULATE for compatibility.
    """
    # First, try config.json
    config_path = ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            force = config.get("level0_generation", {}).get("variant_J_force_calculate", False)
            if force:
                return True
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback to environment variable
    return os.environ.get('HSI_VARIANT_J_FORCE_CALCULATE', '0') == '1'


def generate_pi_bits(num_bits: int) -> str:
    """
    Generate binary representation of π digits.

    Uses BCD encoding: each decimal digit (0-9) → 4 bits.
    For num_bits binary bits, we need num_bits/4 decimal digits.

    By default, uses pre-computed π digits from cache.
    Set variant_J_force_calculate=true in config.json to force mpmath calculation.

    Args:
        num_bits: Number of binary bits to generate

    Returns:
        String of '0' and '1' characters from π

    Raises:
        SystemExit: If cache not found and force_calculate is False
    """
    # We need num_bits/4 decimal digits (BCD: 4 bits per digit)
    n_digits = (num_bits // 4) + 10  # Extra margin

    # Check if we should force mpmath calculation
    force_calculate = _get_force_calculate_setting()

    pi_digits = None

    if not force_calculate:
        # Check if cache exists
        if not PI_CACHE_FILE.exists():
            print(f"\n[variant_J] ⚠️  Cache de π no trobat!", flush=True)
            print(f"[variant_J] 📥 Descarrega manualment des de:", flush=True)
            print(f"[variant_J]    URL: {PI_ARCHIVE_URL}", flush=True)
            print(f"[variant_J]    Desa a: {PI_CACHE_FILE}", flush=True)
            print(f"[variant_J]", flush=True)
            print(f"[variant_J] 💡 Alternativament, activa el càlcul amb mpmath:", flush=True)
            print(f"[variant_J]    En config.json: \"variant_J_force_calculate\": true", flush=True)
            print(f"[variant_J]", flush=True)
            sys.exit(1)

        # Cache exists, read from it
        print(f"[variant_J] 📐 Loading {n_digits:,} π digits from cache...", flush=True)
        pi_digits = _read_pi_digits_from_cache(n_digits)

    if pi_digits is None:
        # Fallback to mpmath calculation
        print(f"[variant_J] 📐 Computing {n_digits:,} π digits with mpmath...", flush=True)
        try:
            from mpmath import mp
            mp.dps = n_digits + 50

            if n_digits > 1_000_000:
                est_minutes = n_digits / 1_000_000
                print(f"[variant_J] ⏳ Estimated time: ~{est_minutes:.0f} min for {n_digits/1e6:.1f}M digits", flush=True)
                print(f"[variant_J]    (mpmath does not support progress, please wait...)", flush=True)

            t0 = time.perf_counter()
            pi_str = mp.nstr(mp.pi, n_digits + 1, strip_zeros=False)
            calc_time = time.perf_counter() - t0

            if n_digits > 1_000_000:
                print(f"[variant_J] ✅ π calculation completed in {calc_time:.1f}s", flush=True)
            pi_str = pi_str.replace('.', '')[1:]  # Skip "3."
            pi_digits = pi_str[:n_digits]
            print(f"[variant_J] ✅ Generated {len(pi_digits):,} π digits using mpmath", flush=True)
        except ImportError:
            print("[variant_J] ⚠️  mpmath not available, using built-in π digits", flush=True)
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
                print(f"[variant_J] ⚠️  Only {len(PI_1000)} digits available", flush=True)
            pi_digits = PI_1000[:n_digits]

    # Convert to binary using BCD (4 bits per decimal digit)
    binary_map = {
        '0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100',
        '5': '0101', '6': '0110', '7': '0111', '8': '1000', '9': '1001'
    }

    # For large sequences, use progress protocol for tqdm bar in parent
    if num_bits > 10_000_000:
        from hsi_agents_project.utils.progress_protocol import ProgressReporter

        chunks = []
        chunk_size = 1_000_000
        total_digits = len(pi_digits)
        num_chunks = (total_digits + chunk_size - 1) // chunk_size

        with ProgressReporter(num_chunks, "[variant_J] π→binary", update_percent=1) as progress:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total_digits)
                chunk = pi_digits[start:end]
                chunks.append(''.join(binary_map[d] for d in chunk if d in binary_map))
                progress.update(chunk_idx + 1)

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
    compressed_size = save_phi_structural_gz(bits, str(struct_path), silent=True)
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

    Generates ALL iterations from 1 to `iterations` (like variant B),
    using pre-computed π digits from cache for speed.

    Variant J serves as a sanity check for the LZ multi-scale analysis:
    - π digits pass most randomness tests (pseudo-random)
    - Expected LZ ratio ≈ 1.0 (no self-similar structure)
    - If LZ ratio ≈ 0.62 for π, the algorithm is biased!

    Args:
        num_bits: Number of binary bits for the final iteration
        iterations: Maximum iteration number (will generate 1..iterations)

    Returns:
        Metadata dictionary for the last iteration
    """
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_J"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "results" / "level0" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    t0_total = time.perf_counter()

    # Pre-load all π digits needed for max iteration (once)
    max_bits = estimate_iteration_size(iterations)
    print(f"[variant_J] 🥧 Generating iterations 1..{iterations} (max {max_bits:,} bits)", flush=True)

    # Generate all π bits upfront (efficient: single cache read)
    all_pi_bits = generate_pi_bits(max_bits)

    per_iteration_data = []
    phi_lengths = []
    snapshots = []
    last_meta = None

    for iter_num in range(1, iterations + 1):
        t0 = time.perf_counter()

        # Calculate size for this iteration (like B: doubling growth)
        iter_bits = estimate_iteration_size(iter_num)
        bits = all_pi_bits[:iter_bits]

        gen_time = time.perf_counter() - t0

        # Save using v33 bitarray encoder
        struct_path = results_dir / f"phi_iter{iter_num}.struct.gz"

        t1 = time.perf_counter()
        compressed_size = save_phi_structural_gz(bits, str(struct_path), silent=True)
        save_time = time.perf_counter() - t1

        elapsed = time.perf_counter() - t0
        format_info = get_format_info(bits)

        # Metadata per iteration
        meta = {
            "iteration": iter_num,
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

        meta_path = results_dir / f"phi_iter{iter_num}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        per_iteration_data.append({
            "iteration": iter_num,
            "phi_length": len(bits),
            "time_sec": elapsed
        })
        phi_lengths.append(len(bits))
        snapshots.append(str(struct_path))
        last_meta = meta

        # Progress report
        size_mb = len(bits) / 8 / 1024 / 1024
        print(f"[variant_J] ✅ iter {iter_num:2d}: {len(bits):>12,} bits ({size_mb:>7.2f} MB) "
              f"in {elapsed:.2f}s → {struct_path.name}", flush=True)

    total_time = time.perf_counter() - t0_total

    # Final report
    report_name = f"variant_J_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "variant": "J",
        "variant_description": "π (Pi) Binary Control - Sanity Check",
        "iterations": iterations,
        "phi_length": phi_lengths[-1],
        "total_bits": phi_lengths[-1],
        "format": "v33_structural",
        "timestamp": datetime.now().isoformat(),
        "execution_time": total_time,
        "per_iteration": per_iteration_data,
        "phi_lengths_per_iteration": phi_lengths,
        "generator_type": "π BCD binary (pre-computed cache)",
        "ones_ratio": last_meta['ones_ratio'] if last_meta else 0.0,
        "expected_lz_ratio": "≈1.0 (pseudo-random, no self-similar structure)",
        "purpose": "Sanity check - if LZ ratio ≈ 0.62, algorithm is biased"
    }
    report_path = report_dir / report_name
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # phi_metadata_J.json for analysis pipeline compatibility
    phi_metadata = {
        "variant": "J",
        "max_iterations": iterations,
        "iterations_completed": iterations,
        "final_length": phi_lengths[-1],
        "use_compression": True,
        "compression_activated": True,
        "basal_pure": False,
        "snapshots_saved": snapshots,
        "phi_lengths_per_iteration": phi_lengths,
        "total_time_seconds": total_time
    }
    phi_meta_path = report_dir / "phi_metadata_J.json"
    with open(phi_meta_path, 'w') as f:
        json.dump(phi_metadata, f, indent=2)

    print(f"\n[variant_J] 🎉 All {iterations} iterations completed in {total_time:.1f}s")
    print(f"[variant_J] ✅ Report: {report_path.name}")
    print(f"[variant_J] ✅ phi_metadata: {phi_meta_path.name}")

    return last_meta


def generate_rule30_bits(num_bits: int, silent: bool = False) -> str:
    """
    Generate bits using Wolfram's Rule 30 cellular automaton.

    Rule 30 is famous for generating pseudo-random behavior from simple rules.
    Used as comparison: deterministic chaos, no self-similar structure expected.

    Args:
        num_bits: Number of bits to generate
        silent: If True, suppress progress output

    Returns:
        Binary string from Rule 30 center column
    """
    import numpy as np
    from hsi_agents_project.utils.progress_protocol import ProgressReporter

    # =========================================================================
    # TIER 1: Numba CPU (optimized with vectorization) - 10-20x faster
    # Note: GPU is not efficient for Rule 30 due to sequential time dependency
    # =========================================================================
    try:
        from numba import njit

        @njit(cache=True, fastmath=True)
        def _rule30_chunk_optimized(state: np.ndarray, new_state: np.ndarray,
                                     chunk_size: int, center: int) -> np.ndarray:
            """Process a chunk of Rule 30 steps with buffer swapping (no allocations)."""
            width = len(state)
            bits = np.zeros(chunk_size, dtype=np.uint8)

            for step in range(chunk_size):
                bits[step] = state[center]
                # Apply Rule 30 in-place to new_state
                for i in range(1, width - 1):
                    new_state[i] = state[i-1] ^ (state[i] | state[i+1])
                # Swap buffers by copying (numba optimizes this)
                for i in range(width):
                    state[i] = new_state[i]
                    new_state[i] = 0

            return bits

        # Width=10K balances chaos quality vs computation time
        # (100K was 10x slower, infeasible for large iterations)
        width = min(2 * num_bits + 1, 10_001)
        center = width // 2

        if not silent:
            print(f"[variant_K]   Using numba JIT optimized...", flush=True)

        # Pre-allocate both buffers ONCE
        state = np.zeros(width, dtype=np.uint8)
        state[center] = 1
        new_state = np.zeros(width, dtype=np.uint8)

        # Process in chunks to show progress
        chunk_size = 500_000  # 500K per chunk for better progress visibility
        all_bits = []
        processed = 0
        num_chunks = (num_bits + chunk_size - 1) // chunk_size

        # Use a list of strings instead of numpy arrays to avoid memory issues
        # Each chunk is converted to string immediately and numpy array discarded
        result_parts = []

        with ProgressReporter(num_chunks, "[variant_K] Rule 30", update_percent=1, silent=silent) as progress:
            chunk_idx = 0
            while processed < num_bits:
                remaining = num_bits - processed
                current_chunk = min(chunk_size, remaining)
                bits_chunk = _rule30_chunk_optimized(state, new_state, current_chunk, center)
                # Convert to string immediately and discard numpy array
                result_parts.append(''.join(str(b) for b in bits_chunk))
                del bits_chunk  # Free memory immediately
                processed += current_chunk
                chunk_idx += 1
                progress.update(chunk_idx)

        return ''.join(result_parts)

    except ImportError:
        pass  # Fall through to numpy version
    except Exception as e:
        import traceback
        if not silent:
            print(f"[variant_K]   ⚠️ Numba error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    # Fallback: optimized numpy version (no np.roll, use slicing)
    if not silent:
        print(f"[variant_K]   Using numpy (install numba for 10x speed)...", flush=True)

    # Width=10K balances chaos quality vs computation time
    width = min(2 * num_bits + 1, 10_001)
    center = width // 2

    state = np.zeros(width, dtype=np.uint8)
    state[center] = 1

    # Process in chunks for progress bar
    chunk_size = 100_000
    num_chunks = (num_bits + chunk_size - 1) // chunk_size
    all_bits = []

    with ProgressReporter(num_chunks, "[variant_K] Rule 30 (numpy)", update_percent=2, silent=silent) as progress:
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_bits)
            chunk_bits = []

            for step in range(start, end):
                chunk_bits.append(str(state[center]))
                # Rule 30 without np.roll (faster)
                new_state = np.zeros(width, dtype=np.uint8)
                new_state[1:-1] = state[:-2] ^ (state[1:-1] | state[2:])
                state = new_state

            all_bits.append(''.join(chunk_bits))
            progress.update(chunk_idx + 1)

    return ''.join(all_bits)


def generate_logistic_map_bits_streaming(num_bits: int, output_path: str, r: float = 3.99, x0: float = 0.1, silent: bool = False) -> tuple:
    """
    Generate bits using the Logistic Map (deterministic chaos) - STREAMING VERSION.

    Writes directly to gzip file without holding full sequence in memory.

    x_{n+1} = r * x_n * (1 - x_n)

    With r ≈ 3.99 (near 4.0), produces chaotic behavior.
    Binary conversion: bit = 1 if x_n >= 0.5, else 0

    Args:
        num_bits: Number of bits to generate
        output_path: Path to write .struct.gz file
        r: Logistic map parameter (default 3.99 for chaos)
        x0: Initial condition
        silent: Suppress progress output

    Returns:
        Tuple of (ones_count, zeros_count, compressed_size)
    """
    import gzip
    import numpy as np
    from hsi_agents_project.utils.progress_protocol import ProgressReporter

    # Process in chunks for memory efficiency and progress
    chunk_size = 10_000_000  # 10M per chunk
    num_chunks = (num_bits + chunk_size - 1) // chunk_size

    ones_count = 0
    zeros_count = 0
    x = x0

    with gzip.open(output_path, 'wt', encoding='utf-8', compresslevel=6) as f:
        with ProgressReporter(num_chunks, "[variant_L] Logistic map", update_percent=2, silent=silent) as progress:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_bits)
                n = end - start

                # Generate chunk
                chunk_bits = np.empty(n, dtype=np.uint8)
                for i in range(n):
                    x = r * x * (1 - x)
                    chunk_bits[i] = 1 if x >= 0.5 else 0

                # Count ones/zeros
                chunk_ones = int(np.sum(chunk_bits))
                ones_count += chunk_ones
                zeros_count += (n - chunk_ones)

                # Write to file and discard
                f.write(''.join(str(b) for b in chunk_bits))
                del chunk_bits

                progress.update(chunk_idx + 1)

    # Get compressed size
    compressed_size = Path(output_path).stat().st_size

    return ones_count, zeros_count, compressed_size


def generate_rule30_variant(num_bits: int, iterations: int) -> dict:
    """
    Generate Rule 30 cellular automaton control data in v33-compatible format.

    Generates ALL iterations from 1 to `iterations` (like variant B/J).

    Variant K serves as deterministic chaos comparison:
    - Rule 30 is famous for pseudo-random behavior
    - Expected LZ ratio ≈ 1.0 (chaotic, no self-similar structure)

    Args:
        num_bits: Number of binary bits for the final iteration
        iterations: Maximum iteration number (will generate 1..iterations)

    Returns:
        Metadata dictionary for the last iteration
    """
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_K"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "results" / "level0" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    t0_total = time.perf_counter()

    # Use num_bits as the size for the FINAL iteration
    # Scale earlier iterations proportionally using estimate_iteration_size ratios
    base_estimate = estimate_iteration_size(iterations)
    scale_factor = num_bits / base_estimate if base_estimate > 0 else 1.0

    print(f"[variant_K] 🔲 Generating iterations 1..{iterations} (final: {num_bits:,} bits)", flush=True)

    # Generate all bits upfront (efficient: single generation)
    all_rule30_bits = generate_rule30_bits(num_bits)

    per_iteration_data = []
    phi_lengths = []
    snapshots = []
    last_meta = None

    for iter_num in range(1, iterations + 1):
        t0 = time.perf_counter()

        # Calculate size for this iteration proportionally
        if iter_num == iterations:
            iter_bits = num_bits  # Final iteration uses exact size
        else:
            iter_bits = int(estimate_iteration_size(iter_num) * scale_factor)
            iter_bits = min(iter_bits, num_bits)  # Cap at available bits

        bits = all_rule30_bits[:iter_bits]

        gen_time = time.perf_counter() - t0

        # Save using v33 bitarray encoder
        struct_path = results_dir / f"phi_iter{iter_num}.struct.gz"

        t1 = time.perf_counter()
        compressed_size = save_phi_structural_gz(bits, str(struct_path), silent=True)
        save_time = time.perf_counter() - t1

        elapsed = time.perf_counter() - t0
        format_info = get_format_info(bits)

        # Metadata per iteration
        meta = {
            "iteration": iter_num,
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
            "variant": "K",
            "variant_description": "Rule 30 Cellular Automaton - Deterministic Chaos",
            "source": "Wolfram Rule 30 center column",
            "ones_count": bits.count('1'),
            "zeros_count": bits.count('0'),
            "ones_ratio": bits.count('1') / len(bits) if bits else 0.0,
            "generation_time_seconds": gen_time,
            "save_time_seconds": save_time
        }

        meta_path = results_dir / f"phi_iter{iter_num}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        per_iteration_data.append({
            "iteration": iter_num,
            "phi_length": len(bits),
            "time_sec": elapsed
        })
        phi_lengths.append(len(bits))
        snapshots.append(str(struct_path))
        last_meta = meta

        # Progress report
        size_mb = len(bits) / 8 / 1024 / 1024
        print(f"[variant_K] ✅ iter {iter_num:2d}: {len(bits):>12,} bits ({size_mb:>7.2f} MB) "
              f"in {elapsed:.2f}s → {struct_path.name}", flush=True)

    total_time = time.perf_counter() - t0_total

    # Final report
    report_name = f"variant_K_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "variant": "K",
        "variant_description": "Rule 30 Cellular Automaton - Deterministic Chaos",
        "iterations": iterations,
        "phi_length": phi_lengths[-1],
        "total_bits": phi_lengths[-1],
        "format": "v33_structural",
        "timestamp": datetime.now().isoformat(),
        "execution_time": total_time,
        "per_iteration": per_iteration_data,
        "phi_lengths_per_iteration": phi_lengths,
        "generator_type": "Wolfram Rule 30 center column",
        "ones_ratio": last_meta['ones_ratio'] if last_meta else 0.0,
        "expected_lz_ratio": "≈1.0 (chaotic, no self-similar structure)",
        "purpose": "Deterministic chaos comparison"
    }
    report_path = report_dir / report_name
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # phi_metadata_K.json for analysis pipeline compatibility
    phi_metadata = {
        "variant": "K",
        "max_iterations": iterations,
        "iterations_completed": iterations,
        "final_length": phi_lengths[-1],
        "use_compression": True,
        "compression_activated": True,
        "basal_pure": False,
        "snapshots_saved": snapshots,
        "phi_lengths_per_iteration": phi_lengths,
        "total_time_seconds": total_time
    }
    phi_meta_path = report_dir / "phi_metadata_K.json"
    with open(phi_meta_path, 'w') as f:
        json.dump(phi_metadata, f, indent=2)

    print(f"\n[variant_K] 🎉 All {iterations} iterations completed in {total_time:.1f}s")
    print(f"[variant_K] ✅ Report: {report_path.name}")
    print(f"[variant_K] ✅ phi_metadata: {phi_meta_path.name}")

    return last_meta


def generate_logistic_variant(num_bits: int, iterations: int) -> dict:
    """
    Generate Logistic Map control data in v33-compatible format.

    Variant L serves as deterministic chaos comparison:
    - Logistic map at r≈4 produces chaotic dynamics
    - Expected LZ ratio ≈ 1.0 (chaotic, no self-similar structure)

    Uses streaming generation to avoid memory issues with large sequences.

    Args:
        num_bits: Number of binary bits to generate
        iterations: Iteration number (for filename compatibility)

    Returns:
        Metadata dictionary
    """
    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_L"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[variant_L] 📈 Generating {num_bits:,} Logistic Map bits (streaming)...", flush=True)

    # Use streaming version - writes directly to disk
    struct_path = results_dir / f"phi_iter{iterations}.struct.gz"
    ones_count, zeros_count, compressed_size = generate_logistic_map_bits_streaming(
        num_bits, str(struct_path)
    )

    gen_time = time.perf_counter() - t0
    print(f"[variant_L] ✅ Generated {num_bits:,} bits in {gen_time:.2f}s", flush=True)
    print(f"[variant_L] 💾 Saved to {struct_path.name} ({compressed_size / (1024*1024):.1f} MB)", flush=True)

    elapsed = time.perf_counter() - t0
    ones_ratio = ones_count / num_bits if num_bits > 0 else 0.0

    meta = {
        "iteration": iterations,
        "sequence_length": num_bits,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "v33_structural",
        "variant": "L",
        "variant_description": "Logistic Map - Deterministic Chaos",
        "source": "Logistic map x_{n+1}=r*x*(1-x) with r=3.99",
        "ones_count": ones_count,
        "zeros_count": zeros_count,
        "ones_ratio": ones_ratio,
        "compressed_size_bytes": compressed_size,
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
        "phi_length": num_bits,
        "total_bits": num_bits,
        "execution_time": elapsed,
        "ones_ratio": ones_ratio,
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
    from hsi_agents_project.utils.bitarray_encoder import save_phi_structural_gz, get_format_info

    results_dir = ROOT / "results" / "level0" / "phi_snapshots" / "var_M"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[variant_M] 🌻 Generating {num_bits:,} Fibonacci word bits...", flush=True)

    bits = generate_fibonacci_word(num_bits)

    gen_time = time.perf_counter() - t0
    print(f"[variant_M] ✅ Generated {len(bits):,} bits in {gen_time:.2f}s", flush=True)

    # Statistics
    ones = bits.count('1')
    zeros = len(bits) - ones
    ratio_01 = zeros / ones if ones > 0 else float('inf')
    print(f"[variant_M] 0s: {zeros:,}, 1s: {ones:,}, ratio 0/1: {ratio_01:.6f} (φ≈1.618)", flush=True)

    # Save using v33 bitarray encoder (same as other variants)
    struct_path = results_dir / f"phi_iter{iterations}.struct.gz"
    print(f"[variant_M] 💾 Saving to {struct_path.name}...", flush=True)

    t1 = time.perf_counter()
    compressed_size = save_phi_structural_gz(bits, str(struct_path), silent=True)
    save_time = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0
    format_info = get_format_info(bits)

    meta = {
        "iteration": iterations,
        "sequence_length": len(bits),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "v33_structural",
        "variant": "M",
        "variant_description": "Fibonacci Word - Positive Control (φ-structured)",
        "source": "Fibonacci word S_n = S_{n-1} + S_{n-2}",
        "expected_lz_ratio": "≈0.62 (1/φ)",
        "purpose": "Validate that algorithm detects φ where it exists",
        "ones_count": ones,
        "zeros_count": zeros,
        "ones_ratio": ones / len(bits) if bits else 0.0,
        "zeros_to_ones_ratio": ratio_01,
        "phi_reference": 1.6180339887,
        "compressed_size_bytes": compressed_size,
        "format_info": format_info,
        "generation_time_seconds": gen_time
    }

    meta_path = results_dir / f"phi_iter{iterations}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Report file
    report_dir = ROOT / "results" / "level0" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "variant": "M",
        "variant_description": "Fibonacci Word - Positive Control (φ-structured)",
        "iterations": iterations,
        "phi_length": len(bits),
        "total_bits": len(bits),
        "execution_time": elapsed,
        "ones_ratio": meta['ones_ratio'],
        "zeros_to_ones_ratio": ratio_01,
        "expected_lz_ratio": "≈0.62 (1/φ)"
    }
    report_path = report_dir / f"variant_M_{iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # phi_metadata for pipeline compatibility
    phi_meta = {"variant": "M", "max_iterations": iterations, "final_length": len(bits)}
    with open(report_dir / "phi_metadata_M.json", 'w') as f:
        json.dump(phi_meta, f, indent=2)

    print(f"[variant_M] ✅ Saved: {struct_path.name}", flush=True)
    print(f"[variant_M] ✅ Metadata: {meta_path.name}", flush=True)
    print(f"[variant_M] ✅ Report: {report_path.name}", flush=True)
    print(f"[variant_M] Stats: {ones:,} ones ({meta['ones_ratio']:.4f}), ratio 0/1 = {ratio_01:.6f}", flush=True)
    return meta


def main():
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

