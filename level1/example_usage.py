"""
Level 1 Data Loader — Example Usage

This script demonstrates how to use the Level 1 data loader to:
1. Load v33 structural data
2. Extract observable Φ
3. Analyze structural information
4. Load ranges of iterations
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from level1.data_loader import (
    load_phi_for_level1,
    load_metadata,
    extract_observable_phi,
    extract_structural_info,
    load_phi_range
)


def example_1_load_single_iteration():
    """Example 1: Load a single iteration with both structural and observable."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Load Single Iteration")
    print("="*60)
    
    data_dir = "results/var_B"
    iteration = 10
    
    # Load both structural and observable
    phi_structural, phi_observable, metadata = load_phi_for_level1(
        data_dir,
        iteration,
        return_structural=True,
        return_observable=True,
        return_metadata=True
    )
    
    print(f"\nLoaded iteration {iteration} from {data_dir}")
    print(f"Format: {metadata['format']}")
    print(f"Structural length: {len(phi_structural):,} characters")
    print(f"Observable length: {len(phi_observable):,} bits")
    print(f"\nStructural preview: {phi_structural[:100]}...")
    print(f"Observable preview: {phi_observable[:100]}...")


def example_2_analyze_structure():
    """Example 2: Analyze structural information."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Analyze Structural Information")
    print("="*60)
    
    data_dir = "results/var_B"
    iteration = 10
    
    # Load only structural
    phi_structural, _, _ = load_phi_for_level1(
        data_dir,
        iteration,
        return_structural=True,
        return_observable=False,
        return_metadata=False
    )
    
    # Extract structural info
    info = extract_structural_info(phi_structural)
    
    print(f"\nStructural Analysis for iteration {iteration}:")
    print(f"  Max nesting depth: {info['max_depth']}")
    print(f"  Number of Absolutes: {info['num_absolutes']}")
    print(f"  Total length: {info['total_length']:,} characters")
    print(f"  Observable length: {info['observable_length']:,} bits")
    print(f"  Parentheses balanced: {info['is_balanced']}")
    print(f"\n  Depth distribution:")
    for depth, count in sorted(info['depth_distribution'].items()):
        print(f"    Depth {depth}: {count} Absolutes")


def example_3_load_only_observable():
    """Example 3: Load only observable (faster, less memory)."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Load Only Observable (Fast Mode)")
    print("="*60)
    
    data_dir = "results/var_B"
    iteration = 15
    
    # Load only observable (faster)
    _, phi_observable, _ = load_phi_for_level1(
        data_dir,
        iteration,
        return_structural=False,  # Don't load structural
        return_observable=True,
        return_metadata=False
    )
    
    print(f"\nLoaded {len(phi_observable):,} bits (observable only)")
    print(f"Preview: {phi_observable[:100]}...")
    
    # Verify it's clean binary
    assert all(c in '01' for c in phi_observable), "Should be clean binary"
    print("✅ Verified: Clean binary string")


def example_4_load_range():
    """Example 4: Load a range of iterations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Load Range of Iterations")
    print("="*60)
    
    data_dir = "results/var_B"
    start = 1
    end = 5
    
    # Load range
    data = load_phi_range(
        data_dir,
        start_iteration=start,
        end_iteration=end,
        return_structural=True,
        return_observable=True
    )
    
    print(f"\nLoaded iterations {start} to {end}:")
    for iter_num, (phi_s, phi_o) in sorted(data.items()):
        print(f"  Iteration {iter_num}:")
        print(f"    Structural: {len(phi_s):,} chars")
        print(f"    Observable: {len(phi_o):,} bits")


def example_5_compare_formats():
    """Example 5: Compare v32 and v33 formats."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare v32 and v33 Formats")
    print("="*60)
    
    # Load v32 data (if available)
    try:
        _, phi_v32, meta_v32 = load_phi_for_level1(
            "results/var_B_v32",
            10,
            return_structural=False,
            return_observable=True,
            return_metadata=True
        )
        print(f"\nv32 format:")
        print(f"  Format: {meta_v32['format']}")
        print(f"  Observable length: {len(phi_v32):,} bits")
        print(f"  Has structural info: No")
    except FileNotFoundError:
        print("\n⚠️  v32 data not found")
    
    # Load v33 data
    try:
        phi_s, phi_o, meta_v33 = load_phi_for_level1(
            "results/var_B",
            10,
            return_structural=True,
            return_observable=True,
            return_metadata=True
        )
        print(f"\nv33 format:")
        print(f"  Format: {meta_v33['format']}")
        print(f"  Structural length: {len(phi_s):,} chars")
        print(f"  Observable length: {len(phi_o):,} bits")
        print(f"  Has structural info: Yes")
        
        # Analyze structure
        info = extract_structural_info(phi_s)
        print(f"  Max nesting depth: {info['max_depth']}")
        print(f"  Number of Absolutes: {info['num_absolutes']}")
    except FileNotFoundError:
        print("\n⚠️  v33 data not found")


if __name__ == "__main__":
    print("\n" + "📚"*30)
    print("LEVEL 1 DATA LOADER — USAGE EXAMPLES")
    print("📚"*30)
    
    print("\n⚠️  Note: These examples require Level 0 data to be generated first.")
    print("Run: python -m hsi_agents_project.level0.generator")
    
    # Uncomment the examples you want to run:
    # example_1_load_single_iteration()
    # example_2_analyze_structure()
    # example_3_load_only_observable()
    # example_4_load_range()
    # example_5_compare_formats()
    
    print("\n✅ Examples ready to run!")
    print("Uncomment the examples in the script to execute them.\n")

