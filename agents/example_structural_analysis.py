"""
Example: Complete Structural Analysis Pipeline

This script demonstrates how to:
1. Load v33 data with structural information
2. Detect observable patterns
3. Detect structural patterns
4. Analyze emergent order
5. Save results

Author: Iban Borràs with Augment Agent
Date: January 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from level1.data_loader import load_phi_for_level1, extract_structural_info
from agents.pattern_detector import PatternDetector
from agents.structural_pattern_detector import StructuralPatternDetector


def analyze_iteration(data_dir: str, iteration: int, output_dir: str = "results/level1/analysis"):
    """
    Complete structural analysis of a single iteration.

    Args:
        data_dir: Path to Level 0 results (e.g., "results/level0/phi_snapshots/var_B")
        iteration: Iteration number to analyze
        output_dir: Directory to save analysis results
    """
    print("\n" + "="*60)
    print(f"STRUCTURAL ANALYSIS: {data_dir} — Iteration {iteration}")
    print("="*60)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    phi_structural, phi_observable, metadata = load_phi_for_level1(
        data_dir,
        iteration,
        return_structural=True,
        return_observable=True,
        return_metadata=True
    )
    
    print(f"   Format: {metadata['format']}")
    print(f"   Structural length: {len(phi_structural):,} chars")
    print(f"   Observable length: {len(phi_observable):,} bits")
    
    # Step 2: Extract basic structural info
    print("\n🔍 Step 2: Extracting structural information...")
    struct_info = extract_structural_info(phi_structural)
    
    print(f"   Max nesting depth: {struct_info['max_depth']}")
    print(f"   Number of Absolutes: {struct_info['num_absolutes']}")
    print(f"   Parentheses balanced: {struct_info['is_balanced']}")
    print(f"   Depth distribution: {struct_info['depth_distribution']}")
    
    # Step 3: Detect observable patterns
    print("\n🔍 Step 3: Detecting observable patterns...")
    pattern_detector = PatternDetector(
        min_pattern_length=3,
        max_pattern_length=20,
        min_occurrences=2,
        enable_structural_analysis=True
    )
    
    # Limit to first 100k bits for speed (adjust as needed)
    sample_size = min(100000, len(phi_observable))
    phi_o_sample = phi_observable[:sample_size]
    phi_s_sample = phi_structural[:sample_size * 2]  # Structural is longer
    
    print(f"   Analyzing sample: {len(phi_o_sample):,} bits...")
    observable_patterns = pattern_detector.detect_patterns(phi_o_sample, phi_s_sample)
    
    print(f"   ✅ Detected {len(observable_patterns)} observable patterns")
    
    # Step 4: Get structural patterns
    print("\n🔍 Step 4: Analyzing structural patterns...")
    structural_patterns = pattern_detector.get_structural_patterns()
    
    print(f"   ✅ Detected {len(structural_patterns)} structural patterns")
    
    # Display structural pattern summary
    if structural_patterns:
        print("\n   Structural pattern types:")
        for pattern in structural_patterns:
            print(f"     - {pattern['pattern_id']} ({pattern['pattern_type']})")
    
    # Step 5: Analyze emergent order
    print("\n📊 Step 5: Analyzing emergent order...")
    
    # Find stratified order pattern
    for pattern in structural_patterns:
        if pattern['pattern_id'] == 'stratified_order_metrics':
            print(f"\n   Stratified Order Analysis:")
            print(f"   Number of depth levels: {pattern['num_depths']}")
            
            # Show entropy at each depth
            depth_metrics = pattern['depth_metrics']
            print(f"\n   Entropy by depth level:")
            for depth in sorted(depth_metrics.keys()):
                metrics = depth_metrics[depth]
                print(f"     Depth {depth}: entropy={metrics['entropy']:.3f}, "
                      f"density={metrics['density']:.3f}, "
                      f"sequences={metrics['sequence_count']}")
    
    # Step 6: Save results
    print(f"\n💾 Step 6: Saving results to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save observable patterns
    pattern_detector.save_patterns(str(output_path / f"observable_patterns_iter{iteration}.json"))
    
    # Save structural patterns
    if hasattr(pattern_detector, 'structural_detector') and pattern_detector.structural_detector:
        pattern_detector.structural_detector.save_patterns(
            str(output_path / f"structural_patterns_iter{iteration}.json")
        )
    
    print("\n✅ Analysis complete!")
    
    return {
        'observable_patterns': len(observable_patterns),
        'structural_patterns': len(structural_patterns),
        'max_depth': struct_info['max_depth'],
        'num_absolutes': struct_info['num_absolutes']
    }


def analyze_multiple_iterations(data_dir: str, start: int, end: int):
    """
    Analyze multiple iterations and track evolution.
    
    Args:
        data_dir: Path to Level 0 results
        start: First iteration
        end: Last iteration
    """
    print("\n" + "="*60)
    print(f"MULTI-ITERATION ANALYSIS: {data_dir}")
    print(f"Iterations {start} to {end}")
    print("="*60)
    
    results = {}
    
    for iteration in range(start, end + 1):
        try:
            result = analyze_iteration(data_dir, iteration)
            results[iteration] = result
        except FileNotFoundError:
            print(f"\n⚠️  Iteration {iteration} not found, skipping...")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nIteration | Observable | Structural | Max Depth | Absolutes")
    print("-" * 60)
    for iter_num, result in sorted(results.items()):
        print(f"{iter_num:9d} | {result['observable_patterns']:10d} | "
              f"{result['structural_patterns']:10d} | {result['max_depth']:9d} | "
              f"{result['num_absolutes']:9d}")


if __name__ == "__main__":
    print("\n" + "🔬"*30)
    print("STRUCTURAL ANALYSIS PIPELINE — HSI v33")
    print("🔬"*30)
    
    # Example 1: Analyze single iteration
    print("\n📋 Example 1: Single Iteration Analysis")
    print("Uncomment to run:")
    print("# analyze_iteration('results/var_B', 10)")
    
    # Example 2: Analyze multiple iterations
    print("\n📋 Example 2: Multi-Iteration Analysis")
    print("Uncomment to run:")
    print("# analyze_multiple_iterations('results/var_B', 1, 5)")
    
    print("\n✅ Examples ready to run!")
    print("Edit this file and uncomment the examples you want to execute.\n")

