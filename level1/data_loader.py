"""
Level 1 Data Loader — HSI v33

This module provides functions to load Level 0 data for Level 1 analysis.
Supports both v32 (observable bits only) and v33 (structural format with parentheses).

Key functions:
- load_phi_for_level1(): Load Φ data with both structural and observable versions
- load_metadata(): Load iteration metadata
- extract_observable_phi(): Remove parentheses from structural Φ
- extract_structural_info(): Extract nesting depth and containment info
"""

import os
import json
import gzip
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def load_metadata(data_dir: str, iteration: int) -> Dict[str, Any]:
    """
    Load metadata for a specific iteration.
    
    Args:
        data_dir: Path to Level 0 results directory (e.g., 'results/var_B')
        iteration: Iteration number to load
        
    Returns:
        Dictionary containing metadata (format, bits, timestamp, etc.)
        
    Example:
        >>> metadata = load_metadata("results/var_B", 10)
        >>> print(metadata['format'])
        'v33_structural'
    """
    json_path = Path(data_dir) / f"phi_iter{iteration}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return metadata


def extract_observable_phi(phi_structural: str) -> str:
    """
    Extract observable Φ by removing parentheses.
    
    Args:
        phi_structural: Φ string with structure (e.g., "((01)1)")
        
    Returns:
        Observable binary string without parentheses (e.g., "011")

    Example:
        >>> extract_observable_phi("((01)1)")
        '011'
    """
    return phi_structural.replace('(', '').replace(')', '')


def extract_structural_info(phi_structural: str) -> Dict[str, Any]:
    """
    Extract structural information from Φ with parentheses.
    
    Analyzes:
    - Maximum nesting depth
    - Depth distribution
    - Number of Absolutes
    - Parentheses balance
    
    Args:
        phi_structural: Φ string with structure (e.g., "((01)1)")
        
    Returns:
        Dictionary with structural metrics
        
    Example:
        >>> info = extract_structural_info("((01)1)")
        >>> print(info['max_depth'])
        2
    """
    depth = 0
    max_depth = 0
    depth_distribution = {}
    num_absolutes = 0
    
    for char in phi_structural:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
            num_absolutes += 1
        elif char == ')':
            depth -= 1
    
    # Check balance
    is_balanced = (depth == 0)
    
    return {
        "max_depth": max_depth,
        "depth_distribution": depth_distribution,
        "num_absolutes": num_absolutes,
        "is_balanced": is_balanced,
        "total_length": len(phi_structural),
        "observable_length": len(extract_observable_phi(phi_structural))
    }


def load_phi_for_level1(
    data_dir: str,
    iteration: int,
    return_structural: bool = True,
    return_observable: bool = True,
    return_metadata: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Load Φ data for Level 1 analysis with format auto-detection.

    Supports both v32 (observable bits) and v33 (structural) formats.

    Args:
        data_dir: Path to Level 0 results directory (e.g., 'results/var_B')
        iteration: Iteration number to load
        return_structural: Return Φ with parentheses (only for v33)
        return_observable: Return observable Φ without parentheses
        return_metadata: Return metadata dictionary

    Returns:
        Tuple of (phi_structural, phi_observable, metadata)
        - phi_structural: None if not requested or v32 format
        - phi_observable: Observable binary string (0s and 1s only)
        - metadata: None if not requested

    Example:
        >>> # Load v33 with structure
        >>> phi_struct, phi_obs, meta = load_phi_for_level1("results/var_B", 10)
        >>> print(f"Structural: {phi_struct[:20]}")
        >>> print(f"Observable: {phi_obs[:20]}")

        >>> # Load only observable (faster)
        >>> _, phi_obs, _ = load_phi_for_level1("results/var_B", 10,
        ...                                      return_structural=False,
        ...                                      return_metadata=False)
    """
    # Load metadata to detect format
    metadata = load_metadata(data_dir, iteration)

    # Detect format (v33_structural or v33_structural_streaming)
    format_str = metadata.get("format", "")
    is_v33 = format_str.startswith("v33_structural")

    phi_structural = None
    phi_observable = None

    if is_v33:
        # v33 format: Load structural format
        struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"

        if not struct_path.exists():
            raise FileNotFoundError(f"Structural file not found: {struct_path}")

        # Import here to avoid circular dependency
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from utils.bitarray_encoder import load_phi_structural_gz

        # Load structural Φ
        phi_structural_full = load_phi_structural_gz(str(struct_path))

        if return_structural:
            phi_structural = phi_structural_full

        if return_observable:
            phi_observable = extract_observable_phi(phi_structural_full)

    else:
        # v32 format: Load observable bits only (no structural info)
        bin_path = Path(data_dir) / f"phi_iter{iteration}.bin.gz"

        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        with gzip.open(bin_path, "rt", encoding="utf-8") as f:
            phi_observable = f.read()

        # v32 doesn't have structural information
        phi_structural = None

    # Return based on flags
    return (
        phi_structural if return_structural else None,
        phi_observable if return_observable else None,
        metadata if return_metadata else None
    )


def load_phi_range(
    data_dir: str,
    start_iteration: int,
    end_iteration: int,
    return_structural: bool = True,
    return_observable: bool = True
) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """
    Load a range of iterations.

    Args:
        data_dir: Path to Level 0 results directory
        start_iteration: First iteration to load (inclusive)
        end_iteration: Last iteration to load (inclusive)
        return_structural: Return structural Φ for each iteration
        return_observable: Return observable Φ for each iteration

    Returns:
        Dictionary mapping iteration number to (phi_structural, phi_observable)

    Example:
        >>> data = load_phi_range("results/var_B", 1, 5)
        >>> for iter_num, (phi_s, phi_o) in data.items():
        ...     print(f"Iteration {iter_num}: {len(phi_o)} bits")
    """
    results = {}

    for iteration in range(start_iteration, end_iteration + 1):
        try:
            phi_s, phi_o, _ = load_phi_for_level1(
                data_dir,
                iteration,
                return_structural=return_structural,
                return_observable=return_observable,
                return_metadata=False
            )
            results[iteration] = (phi_s, phi_o)
        except FileNotFoundError as e:
            print(f"Warning: Skipping iteration {iteration}: {e}")
            continue

    return results

