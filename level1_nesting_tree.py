#!/usr/bin/env python3
"""
Level 1 — Pre-Collapse Structure Analysis (PCSA)

Analyzes the parentheses structure BEFORE final collapse to find hidden φ
in branching ratios, subtree sizes, and depth proportions.

The hypothesis: The parentheses encode a tree whose branching ratios
and structural proportions approach φ.

Usage:
    # Publication mode (default): 1G chars, 100 bootstrap samples
    python level1_nesting_tree.py --variant B --iteration 15

    # Quick mode for development: 1M chars, no bootstrap
    python level1_nesting_tree.py --variant B --iteration 15 --quick
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from math import erfc, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from level1.data_loader import load_phi_for_level1

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887...

# Mathematical constants for comparison
CONSTANTS = {
    'φ (Golden Ratio)': PHI,
    '1/φ': 1/PHI,
    '√2': math.sqrt(2),
    '1/√2': 1/math.sqrt(2),
    '√3': math.sqrt(3),
    '1/√3': 1/math.sqrt(3),
    'e': math.e,
    '1/e': 1/math.e,
    'π/2': math.pi/2,
    'ln(2)': math.log(2),
    '2': 2.0,
    '1.5': 1.5,
    '1.0': 1.0
}


def test_multiple_constants(ratios: List[float], verbose: bool = True) -> Dict:
    """
    Test ratios against multiple mathematical constants.

    Returns statistical comparison with z-scores and p-values.
    """
    if not ratios:
        return {'error': 'No ratios provided'}

    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    n_samples = len(ratios)
    se_ratio = std_ratio / sqrt(n_samples) if n_samples > 0 else 0.0

    results = []
    for name, value in CONSTANTS.items():
        distance = abs(mean_ratio - value)
        z_score = distance / se_ratio if se_ratio > 0 else float('inf')
        p_value = erfc(z_score / sqrt(2)) if z_score < 6 else 0.0

        results.append({
            'constant': name,
            'value': value,
            'distance': distance,
            'z_score': z_score,
            'p_value': p_value
        })

    results.sort(key=lambda x: x['distance'])

    # Effect size vs furthest common value
    random_distance = next((r['distance'] for r in results if r['constant'] == '1.0'), None)
    best_distance = results[0]['distance']
    effect_size = (random_distance - best_distance) / std_ratio if std_ratio > 0 and random_distance else 0.0

    if verbose:
        print(f"\n🔬 Multiple Constants Test")
        print(f"   Mean ratio: {mean_ratio:.4f} ± {std_ratio:.4f} (n={n_samples})")
        print(f"   Closest matches:")
        for i, r in enumerate(results[:5]):
            marker = "🎯" if i == 0 else "  "
            sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
            print(f"   {marker} {r['constant']}: {r['value']:.4f} (dist={r['distance']:.4f}, z={r['z_score']:.1f}{sig})")

        if results[0]['p_value'] < 0.05:
            print(f"   📊 Best match is statistically significant (p<0.05)")
        if effect_size > 2:
            print(f"   📊 Effect size: {effect_size:.2f}σ closer to {results[0]['constant']} than 1.0")

    return {
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'n_samples': n_samples,
        'rankings': results,
        'best_match': results[0]['constant'],
        'best_match_value': results[0]['value'],
        'best_match_distance': results[0]['distance'],
        'best_match_p_value': results[0]['p_value'],
        'effect_size': effect_size,
        'statistically_significant': results[0]['p_value'] < 0.05
    }


def bootstrap_ratios_ci(
    ratios: List[float],
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    verbose: bool = True
) -> Dict:
    """
    Bootstrap confidence interval for the mean ratio.
    """
    if not ratios or n_bootstrap <= 0:
        return {}

    ratios_arr = np.array(ratios)
    n = len(ratios_arr)

    if verbose:
        print(f"\n📊 Bootstrap CI (n={n_bootstrap}, {int(confidence*100)}%)")

    bootstrap_means = []
    for i in range(n_bootstrap):
        sample = np.random.choice(ratios_arr, size=n, replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_means, alpha/2 * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha/2) * 100))

    mean_val = float(np.mean(bootstrap_means))
    std_val = float(np.std(bootstrap_means))

    if verbose:
        print(f"   Mean: {mean_val:.4f} ± {std_val:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Check if CI contains φ
        if ci_lower <= PHI <= ci_upper:
            print(f"   ✅ CI contains φ ({PHI:.4f})")
        else:
            dist_to_phi = min(abs(ci_lower - PHI), abs(ci_upper - PHI))
            print(f"   ❌ CI excludes φ (distance: {dist_to_phi:.4f})")

    return {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'contains_phi': ci_lower <= PHI <= ci_upper,
        'n_bootstrap': n_bootstrap,
        'confidence': confidence
    }


@dataclass
class TreeNode:
    """A node in the nesting tree."""
    content: str = ""  # Binary content (0s and 1s only)
    children: List["TreeNode"] = field(default_factory=list)
    depth: int = 0
    
    def subtree_size(self) -> int:
        """Total nodes in this subtree (including self)."""
        return 1 + sum(c.subtree_size() for c in self.children)
    
    def subtree_bits(self) -> int:
        """Total bits in this subtree (including self)."""
        return len(self.content) + sum(c.subtree_bits() for c in self.children)
    
    def max_depth(self) -> int:
        """Maximum depth in this subtree."""
        if not self.children:
            return self.depth
        return max(c.max_depth() for c in self.children)


def build_tree_from_structural(
    phi_structural: str,
    max_chars: Optional[int] = None,
    verbose: bool = False
) -> TreeNode:
    """
    Build a tree from the parentheses structure.

    Each '(' opens a new child node, ')' closes it.
    Bits between parentheses are the node's content.
    """
    if max_chars:
        phi_structural = phi_structural[:max_chars]

    total_chars = len(phi_structural)
    root = TreeNode(depth=0)
    stack = [root]

    # Progress tracking
    report_interval = max(1, total_chars // 10)
    last_report = 0

    for i, char in enumerate(phi_structural):
        if verbose and i - last_report >= report_interval:
            progress = (i / total_chars) * 100
            print(f"\r      Building tree: {progress:.0f}%...", end="", flush=True)
            last_report = i

        if char == '(':
            new_node = TreeNode(depth=len(stack))
            stack[-1].children.append(new_node)
            stack.append(new_node)
        elif char == ')':
            if len(stack) > 1:
                stack.pop()
        elif char in '01':
            stack[-1].content += char

    if verbose:
        print(f"\r      Building tree: 100% ✓", flush=True)

    return root


def collect_branching_ratios(node: TreeNode, ratios: List[float]) -> None:
    """Collect branching ratios (largest/second-largest child) recursively."""
    if len(node.children) >= 2:
        sizes = sorted([c.subtree_size() for c in node.children], reverse=True)
        if sizes[1] > 0:
            ratios.append(sizes[0] / sizes[1])
    for child in node.children:
        collect_branching_ratios(child, ratios)


def collect_depth_distribution(node: TreeNode, dist: Dict[int, int]) -> None:
    """Collect distribution of nodes by depth."""
    dist[node.depth] = dist.get(node.depth, 0) + 1
    for child in node.children:
        collect_depth_distribution(child, dist)


def collect_content_lengths(node: TreeNode, lengths: List[int]) -> None:
    """Collect content lengths at each node."""
    if node.content:
        lengths.append(len(node.content))
    for child in node.children:
        collect_content_lengths(child, lengths)


def collect_children_counts(node: TreeNode, counts: List[int]) -> None:
    """Collect number of children at each node."""
    if node.children:
        counts.append(len(node.children))
    for child in node.children:
        collect_children_counts(child, counts)


def analyze_depth_ratios(depth_dist: Dict[int, int]) -> List[float]:
    """Calculate ratios between consecutive depth counts."""
    depths = sorted(depth_dist.keys())
    ratios = []
    for i in range(len(depths) - 1):
        d1, d2 = depths[i], depths[i + 1]
        c1, c2 = depth_dist[d1], depth_dist[d2]
        if c1 > 0:
            ratios.append(c2 / c1)
    return ratios


def phi_proximity(ratios: List[float]) -> Tuple[float, float, float]:
    """Calculate φ proximity: (mean_distance, std, min_distance)."""
    if not ratios:
        return float('nan'), float('nan'), float('nan')
    distances = [abs(r - PHI) for r in ratios]
    return float(np.mean(distances)), float(np.std(distances)), float(min(distances))


def analyze_nesting_tree(
    phi_structural: str,
    max_chars: Optional[int] = None,
    n_bootstrap: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Main analysis: find φ in the nesting tree structure.

    Analyzes:
    - Branching ratios (sibling subtree sizes)
    - Depth distribution ratios
    - Content length distribution
    - Children count distribution

    Args:
        phi_structural: The structural string with parentheses
        max_chars: Maximum chars to analyze (default 1G for publication)
        n_bootstrap: Bootstrap samples for CI (default 100 for publication)
        verbose: Print progress
    """
    if verbose:
        total_chars = len(phi_structural)
        print(f"\n🌳 Pre-Collapse Structure Analysis (PCSA)")
        print(f"   Total chars: {total_chars:,}")
        if max_chars and max_chars < total_chars:
            print(f"   Analyzing first: {max_chars:,} chars")
        print(f"   Bootstrap samples: {n_bootstrap}")

    if verbose:
        print(f"   Building tree...")
    tree = build_tree_from_structural(phi_structural, max_chars, verbose=verbose)
    
    if verbose:
        print(f"   Collecting metrics...")
        print(f"      [1/4] Branching ratios...", end=" ", flush=True)

    # Collect all metrics
    branching_ratios = []
    collect_branching_ratios(tree, branching_ratios)
    if verbose:
        print(f"✓ ({len(branching_ratios):,} found)")
        print(f"      [2/4] Depth distribution...", end=" ", flush=True)

    depth_dist = {}
    collect_depth_distribution(tree, depth_dist)
    depth_ratios = analyze_depth_ratios(depth_dist)
    if verbose:
        print(f"✓ ({len(depth_dist)} depths)")

    if verbose:
        print(f"      [3/4] Content lengths...", end=" ", flush=True)
    content_lengths = []
    collect_content_lengths(tree, content_lengths)
    if verbose:
        print(f"✓ ({len(content_lengths):,} nodes)")
        print(f"      [4/4] Children counts...", end=" ", flush=True)

    children_counts = []
    collect_children_counts(tree, children_counts)
    if verbose:
        print(f"✓ ({len(children_counts):,} branching nodes)")

    # Calculate φ proximity for each metric
    br_mean, br_std, br_min = phi_proximity(branching_ratios)
    dr_mean, dr_std, dr_min = phi_proximity(depth_ratios)

    # Content length ratios - use ALL consecutive pairs, not just top 20
    sorted_lengths = sorted(content_lengths, reverse=True)
    length_ratios = []
    for i in range(len(sorted_lengths) - 1):
        if sorted_lengths[i + 1] > 0:
            length_ratios.append(sorted_lengths[i] / sorted_lengths[i + 1])
    lr_mean, lr_std, lr_min = phi_proximity(length_ratios)

    # Statistical tests on branching ratios (main metric)
    if verbose:
        print(f"\n   Running statistical tests...")

    constants_test = test_multiple_constants(branching_ratios, verbose=verbose) if branching_ratios else {}

    # Bootstrap CI if requested
    bootstrap_results = {}
    if n_bootstrap > 0 and branching_ratios:
        bootstrap_results = bootstrap_ratios_ci(branching_ratios, n_bootstrap, verbose=verbose)

    results = {
        'tree_stats': {
            'total_nodes': tree.subtree_size(),
            'total_bits': tree.subtree_bits(),
            'max_depth': tree.max_depth(),
            'depth_distribution': depth_dist
        },
        'branching_analysis': {
            'count': len(branching_ratios),
            'ratios_sample': branching_ratios[:20],
            'phi_distance_mean': br_mean,
            'phi_distance_std': br_std,
            'phi_distance_min': br_min,
            'mean_ratio': float(np.mean(branching_ratios)) if branching_ratios else float('nan'),
            'constants_test': constants_test,
            'bootstrap': bootstrap_results
        },
        'depth_analysis': {
            'ratios': depth_ratios,
            'phi_distance_mean': dr_mean,
            'phi_distance_std': dr_std,
            'phi_distance_min': dr_min
        },
        'content_analysis': {
            'total_contents': len(content_lengths),
            'mean_length': float(np.mean(content_lengths)) if content_lengths else 0,
            'length_ratios_count': len(length_ratios),
            'length_ratios_sample': length_ratios[:10],
            'phi_distance_mean': lr_mean
        },
        'children_analysis': {
            'total_branching_nodes': len(children_counts),
            'mean_children': float(np.mean(children_counts)) if children_counts else 0,
            'max_children': max(children_counts) if children_counts else 0
        },
        'phi_target': PHI
    }

    return results


def print_results(results: Dict) -> None:
    """Print analysis results with statistical interpretation."""
    print(f"\n{'='*60}")
    print(f"🌳 PRE-COLLAPSE STRUCTURE ANALYSIS — Results")
    print(f"{'='*60}")

    ts = results['tree_stats']
    print(f"\n📊 Tree Statistics:")
    print(f"   Nodes: {ts['total_nodes']:,}")
    print(f"   Bits: {ts['total_bits']:,}")
    print(f"   Max depth: {ts['max_depth']}")

    ba = results['branching_analysis']
    print(f"\n🌿 Branching Ratios Analysis:")
    print(f"   Branching nodes: {ba['count']:,}")

    # Show statistical test results
    ct = ba.get('constants_test', {})
    if ct and 'best_match' in ct:
        print(f"   Mean ratio: {ct['mean_ratio']:.4f} ± {ct['std_ratio']:.4f}")
        print(f"   Best match: {ct['best_match']} ({ct['best_match_value']:.4f})")
        print(f"   Distance to best: {ct['best_match_distance']:.4f}")
        if ct.get('statistically_significant'):
            print(f"   ✅ Statistically significant (p<0.05)")

    # Show bootstrap CI
    bs = ba.get('bootstrap', {})
    if bs and 'ci_lower' in bs:
        print(f"   Bootstrap 95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
        if bs.get('contains_phi'):
            print(f"   ✅ CI contains φ")
        else:
            print(f"   ❌ CI excludes φ")

    da = results['depth_analysis']
    print(f"\n📏 Depth Distribution:")
    if da['ratios']:
        ratios_str = ", ".join([f"{r:.3f}" for r in da['ratios'][:5]])
        print(f"   Ratios: [{ratios_str}...]")
        print(f"   φ-distance (mean): {da['phi_distance_mean']:.4f}")

    ca = results['content_analysis']
    print(f"\n📝 Content Analysis:")
    print(f"   Nodes with content: {ca['total_contents']:,}")
    print(f"   Mean content length: {ca['mean_length']:.2f} bits")

    # Summary based on statistical tests
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")

    if ct and 'best_match' in ct:
        best_match = ct['best_match']
        p_val = ct.get('best_match_p_value', 1.0)
        effect = ct.get('effect_size', 0)

        if p_val < 0.001 and effect > 3:
            print(f"   ✅ STRONG signal: {best_match} (p<0.001, effect={effect:.1f}σ)")
        elif p_val < 0.05:
            print(f"   ⚠️  Moderate signal: {best_match} (p<0.05)")
        else:
            print(f"   ❌ No significant signal (p={p_val:.3f})")
    else:
        print(f"   ❌ Insufficient data for statistical analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Collapse Structure Analysis — Find hidden φ in nesting tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Publication mode (default): 1G chars, 100 bootstrap samples
    python level1_nesting_tree.py -v B -i 15

  Quick mode for development: 1M chars, no bootstrap
    python level1_nesting_tree.py -v B -i 15 --quick
        """
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-chars", "-m", type=int, default=1_000_000_000,
                        help="Max chars per segment (default: 1G for publication)")
    parser.add_argument("--segments", "-s", type=int, default=4,
                        help="Number of segments to analyze for consistency (default: 4)")
    parser.add_argument("--bootstrap", "-b", type=int, default=100,
                        help="Bootstrap samples for CI (default: 100 for publication)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode: 1 segment, 1M chars, no bootstrap (for development)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.max_chars = 1_000_000
        args.segments = 1
        args.bootstrap = 0

    # Determine data directory (try both locations)
    data_dir = f"results/level0/phi_snapshots/var_{args.variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{args.variant}"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for variant {args.variant}")
        sys.exit(1)

    # Format for display
    def format_chars(n):
        if n >= 1_000_000_000:
            return f"{n // 1_000_000_000}G"
        elif n >= 1_000_000:
            return f"{n // 1_000_000}M"
        elif n >= 1_000:
            return f"{n // 1_000}k"
        return str(n)

    import time
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"🔬 PCSA — Variant {args.variant}, Iteration {args.iteration}")
    print(f"{'='*60}")
    print(f"   Segment size: {format_chars(args.max_chars)}")
    print(f"   Segments: {args.segments} (multi-position sampling)")
    print(f"   Bootstrap: {args.bootstrap} samples")
    print(f"   Mode: {'Quick (dev)' if args.quick else 'Publication'}")

    # Try to find structural file
    struct_path = Path(data_dir) / f"phi_iter{args.iteration}.struct.gz"

    if not struct_path.exists():
        print(f"\n❌ Structural file not found: {struct_path}")
        sys.exit(1)

    # Load metadata to get total size
    meta_path = Path(data_dir) / f"phi_iter{args.iteration}.json"
    total_chars = None
    if meta_path.exists():
        import json as json_module
        with open(meta_path) as f:
            meta = json_module.load(f)
            total_chars = meta.get('sequence_length')
            if total_chars:
                print(f"   Total sequence: {format_chars(total_chars)} chars")

    # Use multi-segment streaming for scientific rigor
    print(f"\n📥 Streaming {args.segments} segments from different positions...")
    load_start = time.time()

    try:
        if args.segments == 1:
            # Single segment (quick mode) - use simple prefix streaming
            from utils.bitarray_encoder import stream_phi_prefix_gz
            phi_structural = stream_phi_prefix_gz(
                str(struct_path),
                max_chars=args.max_chars,
                clean=False,
                verbose=True
            )
            segments = [(0, phi_structural)]
        else:
            # Multi-segment mode (publication) - scientific rigor
            from utils.bitarray_encoder import stream_multi_segment_gz
            segments = stream_multi_segment_gz(
                str(struct_path),
                segment_size=args.max_chars,
                num_segments=args.segments,
                total_chars=total_chars,
                clean=False,
                verbose=True
            )
    except Exception as e:
        import traceback
        print(f"\n❌ Error loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not segments:
        print(f"\n❌ No segments loaded")
        sys.exit(1)

    load_time = time.time() - load_start
    print(f"   ✓ Loaded {len(segments)} segments in {load_time:.1f}s")

    # Analyze each segment
    all_results = []
    for seg_idx, (position, phi_structural) in enumerate(segments):
        print(f"\n{'─'*60}")
        print(f"📊 Analyzing Segment {seg_idx + 1}/{len(segments)} (position {position:,})")

        seg_results = analyze_nesting_tree(
            phi_structural,
            max_chars=None,  # Already sized
            n_bootstrap=args.bootstrap if seg_idx == 0 else 0,  # Bootstrap only on first
            verbose=True
        )
        seg_results['segment_index'] = seg_idx
        seg_results['segment_position'] = position
        all_results.append(seg_results)

    # Combine results and check consistency
    results = {
        'variant': args.variant,
        'iteration': args.iteration,
        'mode': 'quick' if args.quick else 'publication',
        'segments_analyzed': len(segments),
        'segment_size': args.max_chars,
        'segments': all_results
    }

    # Calculate consistency metrics
    if len(all_results) > 1:
        mean_ratios = [r['branching_analysis']['mean_ratio'] for r in all_results
                       if not math.isnan(r['branching_analysis']['mean_ratio'])]
        if mean_ratios:
            results['consistency'] = {
                'mean_ratio_mean': float(np.mean(mean_ratios)),
                'mean_ratio_std': float(np.std(mean_ratios)),
                'mean_ratio_cv': float(np.std(mean_ratios) / np.mean(mean_ratios)) if np.mean(mean_ratios) > 0 else 0,
                'is_consistent': float(np.std(mean_ratios)) < 0.1  # CV < 10% is consistent
            }

            print(f"\n{'='*60}")
            print(f"📊 CONSISTENCY CHECK (Multi-Segment)")
            print(f"{'='*60}")
            print(f"   Mean branching ratio across segments: {results['consistency']['mean_ratio_mean']:.4f} ± {results['consistency']['mean_ratio_std']:.4f}")
            cv_pct = results['consistency']['mean_ratio_cv'] * 100
            if results['consistency']['is_consistent']:
                print(f"   ✅ CONSISTENT (CV={cv_pct:.1f}% < 10%)")
            else:
                print(f"   ⚠️ INCONSISTENT (CV={cv_pct:.1f}% >= 10%)")

    # Print main results (from first segment or combined)
    print_results(all_results[0])

    # Auto-save results
    if args.output is None and not args.quick:
        from utils.file_saver import get_output_path
        filename = f"nesting_tree_var_{args.variant}_iter{args.iteration}_{format_chars(args.max_chars)}.json"
        args.output = str(get_output_path(1, "analysis", filename))

    if args.output:
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            return obj

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(convert_for_json(results), f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

