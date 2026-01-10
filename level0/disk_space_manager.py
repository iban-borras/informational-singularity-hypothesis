"""
Disk Space Manager — Intelligent space estimation and compression activation

This module provides utilities to:
1. Estimate space needed for next iteration
2. Check available disk space
3. Decide when to activate compression
4. Handle insufficient space gracefully

Author: Iban Borràs with Augment Agent (Sophia)
Date: January 2025
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

# Growth factor per iteration (approximately φ³ ≈ 4.236)
PHI_CUBED = 4.236
# Safety threshold: activate compression if estimated > 80% of free space
COMPRESSION_THRESHOLD = 0.80
# Compression ratio estimate for gzip level 1 (conservative)
GZIP_COMPRESSION_RATIO = 0.10  # Expect 10x compression at minimum


def get_free_space_bytes(path: Path) -> int:
    """Get free space in bytes for the drive containing path."""
    return shutil.disk_usage(path).free


def estimate_next_iteration_bytes(current_iter: int, current_size_bytes: int) -> int:
    """
    Estimate space needed for next iteration based on φ³ growth.
    
    Args:
        current_iter: Current iteration number
        current_size_bytes: Current iteration's file size in bytes
    
    Returns:
        Estimated bytes needed for next iteration
    """
    if current_iter < 14:
        # Small iterations, minimal space
        return current_size_bytes * 5  # Conservative multiplier for early iters
    
    # For larger iterations, use φ³ growth
    # Also account for temporary files (accumulation + decay_frame + hybrid temps)
    # Total temp space ≈ 3-4x the final output
    temp_multiplier = 4.0
    return int(current_size_bytes * PHI_CUBED * temp_multiplier)


def estimate_compressed_space(uncompressed_bytes: int) -> int:
    """Estimate space needed if using compression."""
    return int(uncompressed_bytes * GZIP_COMPRESSION_RATIO)


def should_use_compression(
    output_dir: Path,
    current_iter: int,
    current_size_bytes: int,
    threshold: float = COMPRESSION_THRESHOLD
) -> Tuple[bool, dict]:
    """
    Decide if compression should be activated for next iteration.
    
    Args:
        output_dir: Directory where temp files are stored
        current_iter: Current iteration number
        current_size_bytes: Current iteration's size in bytes
        threshold: Activate if estimated > threshold * free_space
    
    Returns:
        Tuple (should_compress, details_dict)
    """
    free_space = get_free_space_bytes(output_dir)
    estimated_needed = estimate_next_iteration_bytes(current_iter, current_size_bytes)
    estimated_compressed = estimate_compressed_space(estimated_needed)
    
    ratio_uncompressed = estimated_needed / free_space if free_space > 0 else float('inf')
    ratio_compressed = estimated_compressed / free_space if free_space > 0 else float('inf')
    
    details = {
        'free_space_gb': free_space / 1e9,
        'estimated_needed_gb': estimated_needed / 1e9,
        'estimated_compressed_gb': estimated_compressed / 1e9,
        'ratio_uncompressed': ratio_uncompressed,
        'ratio_compressed': ratio_compressed,
        'threshold': threshold,
        'current_iter': current_iter,
    }
    
    # Decide based on threshold
    if ratio_uncompressed > threshold:
        if ratio_compressed <= threshold:
            details['decision'] = 'COMPRESS'
            details['reason'] = f"Uncompressed ({ratio_uncompressed:.1%}) exceeds {threshold:.0%}, but compressed ({ratio_compressed:.1%}) is OK"
            return True, details
        else:
            details['decision'] = 'INSUFFICIENT'
            details['reason'] = f"Even compressed ({ratio_compressed:.1%}) exceeds {threshold:.0%} of free space"
            return True, details  # Still compress, but will likely fail
    else:
        details['decision'] = 'NORMAL'
        details['reason'] = f"Estimated ({ratio_uncompressed:.1%}) within {threshold:.0%} threshold"
        return False, details


def check_sufficient_space(
    output_dir: Path,
    needed_bytes: int,
    compressed: bool = False
) -> Tuple[bool, int]:
    """
    Check if there's sufficient space for an operation.
    
    Returns:
        Tuple (is_sufficient, free_bytes)
    """
    free_space = get_free_space_bytes(output_dir)
    # Leave 5% buffer
    usable_space = int(free_space * 0.95)
    return needed_bytes <= usable_space, free_space


def write_insufficient_space_log(
    output_dir: Path,
    iteration: int,
    details: dict,
    error_msg: Optional[str] = None
):
    """Write detailed log when space is insufficient."""
    log_dir = output_dir.parent / "errors"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"insufficient_space_{timestamp}.json"
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'iteration': iteration,
        'error': error_msg or "Insufficient disk space",
        'space_details': details,
        'recommendation': (
            f"Free at least {details.get('estimated_compressed_gb', 0):.1f} GB to continue with compression, "
            f"or {details.get('estimated_needed_gb', 0):.1f} GB for uncompressed mode."
        )
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n❌ INSUFFICIENT SPACE — Log saved to: {log_path}")
    return log_path

