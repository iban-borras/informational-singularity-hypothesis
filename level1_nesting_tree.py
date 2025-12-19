#!/usr/bin/env python3
"""
Level 1 — Pre-Collapse Structure Analysis (PCSA)

Analyzes the parentheses structure BEFORE final collapse to find hidden φ
in branching ratios, subtree sizes, and depth proportions.

The hypothesis: The parentheses encode a tree whose branching ratios
and structural proportions approach φ.

Usage:
    python level1_nesting_tree.py --variant B --iteration 15
    python level1_nesting_tree.py --variant B --iteration 15 --max-chars 1000000
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from level1.data_loader import load_phi_for_level1

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887...


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
    verbose: bool = True
) -> Dict:
    """
    Main analysis: find φ in the nesting tree structure.
    
    Analyzes:
    - Branching ratios (sibling subtree sizes)
    - Depth distribution ratios
    - Content length distribution
    - Children count distribution
    """
    if verbose:
        total_chars = len(phi_structural)
        print(f"\n🌳 Pre-Collapse Structure Analysis (PCSA)")
        print(f"   Total chars: {total_chars:,}")
        if max_chars and max_chars < total_chars:
            print(f"   Analyzing first: {max_chars:,} chars")
        print(f"   Target: φ = {PHI:.6f}")

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

    # Content length ratios (consecutive sorted lengths)
    sorted_lengths = sorted(content_lengths, reverse=True)
    length_ratios = []
    for i in range(min(20, len(sorted_lengths) - 1)):
        if sorted_lengths[i + 1] > 0:
            length_ratios.append(sorted_lengths[i] / sorted_lengths[i + 1])
    lr_mean, lr_std, lr_min = phi_proximity(length_ratios)

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
            'mean_ratio': float(np.mean(branching_ratios)) if branching_ratios else float('nan')
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
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"🌳 PRE-COLLAPSE STRUCTURE ANALYSIS — Hidden φ Discovery")
    print(f"{'='*60}")

    ts = results['tree_stats']
    print(f"\n📊 Tree Statistics:")
    print(f"   Nodes: {ts['total_nodes']:,}")
    print(f"   Bits: {ts['total_bits']:,}")
    print(f"   Max depth: {ts['max_depth']}")

    ba = results['branching_analysis']
    print(f"\n🌿 Branching Ratios (sibling subtree sizes):")
    print(f"   Branching nodes analyzed: {ba['count']:,}")
    if ba['count'] > 0:
        print(f"   Mean ratio: {ba['mean_ratio']:.4f}")
        print(f"   φ-distance (mean): {ba['phi_distance_mean']:.4f}")
        print(f"   φ-distance (min): {ba['phi_distance_min']:.4f}")
        if ba['ratios_sample']:
            sample_str = ", ".join([f"{r:.2f}" for r in ba['ratios_sample'][:5]])
            print(f"   Sample: [{sample_str}...]")

    da = results['depth_analysis']
    print(f"\n📏 Depth Distribution Ratios:")
    if da['ratios']:
        ratios_str = ", ".join([f"{r:.3f}" for r in da['ratios']])
        print(f"   Ratios: [{ratios_str}]")
        print(f"   φ-distance (mean): {da['phi_distance_mean']:.4f}")

    ca = results['content_analysis']
    print(f"\n📝 Content Analysis:")
    print(f"   Nodes with content: {ca['total_contents']:,}")
    print(f"   Mean content length: {ca['mean_length']:.2f} bits")
    if not math.isnan(ca['phi_distance_mean']):
        print(f"   Length ratios φ-distance: {ca['phi_distance_mean']:.4f}")

    # Overall assessment
    print(f"\n🎯 Overall φ-Alignment Assessment:")
    best_dist = min(
        ba['phi_distance_mean'] if not math.isnan(ba['phi_distance_mean']) else float('inf'),
        da['phi_distance_mean'] if not math.isnan(da['phi_distance_mean']) else float('inf'),
        ca['phi_distance_mean'] if not math.isnan(ca['phi_distance_mean']) else float('inf')
    )

    if best_dist < 0.1:
        print(f"   ✅ STRONG φ signal in tree structure! (best distance: {best_dist:.4f})")
    elif best_dist < 0.3:
        print(f"   ⚠️  Moderate φ signal (best distance: {best_dist:.4f})")
    elif best_dist < float('inf'):
        print(f"   ❌ Weak φ signal (best distance: {best_dist:.4f})")
    else:
        print(f"   ❌ Insufficient data for φ analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Collapse Structure Analysis — Find hidden φ in nesting tree"
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-chars", "-m", type=int, default=None, help="Max chars to analyze")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Determine data directory (try both locations)
    data_dir = f"results/level0/phi_snapshots/var_{args.variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{args.variant}"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for variant {args.variant}")
        sys.exit(1)

    print(f"\n🔬 Loading Variant {args.variant}, iteration {args.iteration}...")

    try:
        phi_structural, _, metadata = load_phi_for_level1(
            data_dir, args.iteration,
            return_structural=True,
            return_observable=False,
            return_metadata=True
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not phi_structural:
        print(f"❌ No structural data available (might be v32 format)")
        sys.exit(1)

    print(f"   Loaded {len(phi_structural):,} chars (structural)")

    results = analyze_nesting_tree(phi_structural, max_chars=args.max_chars)
    results['variant'] = args.variant
    results['iteration'] = args.iteration

    print_results(results)

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

