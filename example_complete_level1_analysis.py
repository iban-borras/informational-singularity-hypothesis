"""
Complete Level 1 Analysis Pipeline — HSI v33

This script demonstrates the complete Level 1 analysis workflow:
1. Load v33 data (structural + observable)
2. Detect observable patterns
3. Detect structural patterns
4. Calculate order metrics
5. Generate comprehensive report

This is the FULL pipeline for analyzing emergent order in HSI Level 0 data.

Author: Iban Borràs with Augment Agent
Date: January 2025
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from level1.data_loader import load_phi_for_level1, extract_structural_info
from agents.pattern_detector import PatternDetector
from metrics.order_metrics import OrderMetrics, create_order_report


def complete_level1_analysis(
    data_dir: str,
    iteration: int,
    output_dir: str = "results/level1_analysis",
    sample_size: int = 100000
):
    """
    Complete Level 1 analysis of a single iteration.
    
    Args:
        data_dir: Path to Level 0 results (e.g., "results/var_B")
        iteration: Iteration number to analyze
        output_dir: Directory to save analysis results
        sample_size: Number of bits to analyze (for speed)
    """
    print("\n" + "="*70)
    print(f"COMPLETE LEVEL 1 ANALYSIS — HSI v33")
    print(f"Data: {data_dir} | Iteration: {iteration}")
    print("="*70)
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n📂 STEP 1: Loading v33 data...")
    
    phi_structural, phi_observable, metadata = load_phi_for_level1(
        data_dir,
        iteration,
        return_structural=True,
        return_observable=True,
        return_metadata=True
    )
    
    print(f"   ✅ Format: {metadata.get('format', 'unknown')}")
    print(f"   ✅ Structural length: {len(phi_structural):,} chars")
    print(f"   ✅ Observable length: {len(phi_observable):,} bits")
    
    # ========== STEP 2: BASIC STRUCTURAL INFO ==========
    print("\n🔍 STEP 2: Extracting basic structural information...")
    
    struct_info = extract_structural_info(phi_structural)
    
    print(f"   ✅ Max nesting depth: {struct_info['max_depth']}")
    print(f"   ✅ Number of Absolutes: {struct_info['num_absolutes']}")
    print(f"   ✅ Parentheses balanced: {struct_info['is_balanced']}")
    
    # ========== STEP 3: DETECT PATTERNS ==========
    print("\n🔍 STEP 3: Detecting patterns (observable + structural)...")
    
    # Limit to sample size for speed
    phi_o_sample = phi_observable[:sample_size]
    phi_s_sample = phi_structural[:sample_size * 2]  # Structural is longer
    
    print(f"   Analyzing sample: {len(phi_o_sample):,} bits...")
    
    pattern_detector = PatternDetector(
        min_pattern_length=3,
        max_pattern_length=20,
        min_occurrences=2,
        enable_structural_analysis=True
    )
    
    observable_patterns = pattern_detector.detect_patterns(phi_o_sample, phi_s_sample)
    structural_patterns = pattern_detector.get_structural_patterns()
    
    print(f"   ✅ Observable patterns: {len(observable_patterns)}")
    print(f"   ✅ Structural patterns: {len(structural_patterns)}")
    
    # ========== STEP 4: CALCULATE ORDER METRICS ==========
    print("\n📊 STEP 4: Calculating order metrics...")
    
    order_metrics = OrderMetrics.calculate_all_metrics(phi_s_sample)
    
    print(f"   ✅ Order emergence index: {order_metrics['order_emergence']['order_index']:.3f}")
    print(f"   ✅ Depth organization: {order_metrics['order_emergence']['depth_organization']:.3f}")
    print(f"   ✅ Mean entropy: {order_metrics['stratified_entropy']['mean_entropy']:.3f}")
    print(f"   ✅ Containment regularity: {order_metrics['order_emergence']['containment_regularity']:.3f}")
    
    # ========== STEP 5: GENERATE REPORT ==========
    print("\n📝 STEP 5: Generating comprehensive report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive report
    report = {
        'metadata': {
            'data_dir': data_dir,
            'iteration': iteration,
            'format': metadata.get('format', 'unknown'),
            'sample_size': len(phi_o_sample),
            'full_size': len(phi_observable)
        },
        'structural_info': struct_info,
        'patterns': {
            'observable': {
                'count': len(observable_patterns),
                'patterns': observable_patterns[:10]  # First 10 only
            },
            'structural': {
                'count': len(structural_patterns),
                'patterns': structural_patterns
            }
        },
        'order_metrics': order_metrics,
        'summary': {
            'order_index': order_metrics['order_emergence']['order_index'],
            'max_depth': struct_info['max_depth'],
            'total_absolutes': struct_info['num_absolutes'],
            'observable_patterns': len(observable_patterns),
            'structural_patterns': len(structural_patterns),
            'is_balanced': struct_info['is_balanced']
        }
    }
    
    # Save report
    report_path = output_path / f"level1_analysis_iter{iteration}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Report saved to: {report_path}")
    
    # ========== STEP 6: DISPLAY SUMMARY ==========
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n🔢 DATA:")
    print(f"   Total length: {len(phi_observable):,} bits")
    print(f"   Analyzed: {len(phi_o_sample):,} bits ({100*len(phi_o_sample)/len(phi_observable):.1f}%)")
    
    print(f"\n🏗️  STRUCTURE:")
    print(f"   Max depth: {struct_info['max_depth']}")
    print(f"   Total Absolutes: {struct_info['num_absolutes']}")
    print(f"   Balanced: {struct_info['is_balanced']}")
    
    print(f"\n🔍 PATTERNS:")
    print(f"   Observable: {len(observable_patterns)}")
    print(f"   Structural: {len(structural_patterns)}")
    
    print(f"\n📊 ORDER METRICS:")
    print(f"   Order Index: {order_metrics['order_emergence']['order_index']:.3f}")
    print(f"   Depth Organization: {order_metrics['order_emergence']['depth_organization']:.3f}")
    print(f"   Entropy Gradient: {order_metrics['order_emergence']['entropy_gradient']:.3f}")
    print(f"   Containment Regularity: {order_metrics['order_emergence']['containment_regularity']:.3f}")
    print(f"   Balance Score: {order_metrics['order_emergence']['balance_score']:.3f}")
    
    print("\n✅ Analysis complete!")
    
    return report


if __name__ == "__main__":
    print("\n" + "🔬"*35)
    print("COMPLETE LEVEL 1 ANALYSIS PIPELINE — HSI v33")
    print("🔬"*35)
    
    print("\n📋 USAGE:")
    print("Edit this file and uncomment one of the examples below:\n")
    
    print("# Example 1: Analyze single iteration")
    print("# complete_level1_analysis('results/var_B', 10)")
    
    print("\n# Example 2: Analyze with custom sample size")
    print("# complete_level1_analysis('results/var_B', 15, sample_size=50000)")
    
    print("\n✅ Ready to run!")
    print("Uncomment an example and execute this script.\n")

