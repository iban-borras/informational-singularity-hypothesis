#!/usr/bin/env python3
"""
Level 1 Figure Generator for HSI Scientific Paper.

Generates publication-quality figures from Level 1 analysis results.
Auto-discovers available data and generates appropriate figures.

Usage:
    python visualizations/level1_figures.py --all
    python visualizations/level1_figures.py --variants B A --iterations 18
    python visualizations/level1_figures.py --figures markov_heatmap top_patterns
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Try to import seaborn for heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not available, heatmaps will use matplotlib only")

# Import survival analysis module
try:
    from . import survival_analysis
    HAS_SURVIVAL = True
except ImportError:
    try:
        import survival_analysis
        HAS_SURVIVAL = True
    except ImportError:
        HAS_SURVIVAL = False

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

STYLE = {
    'figure.figsize': (10, 7),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
}

# Colorblind-friendly palette
COLORS = {
    'A': '#E69F00',  # Orange (control/random)
    'B': '#0072B2',  # Blue (HSI main)
    'C': '#56B4E9',  # Light blue
    'D': '#009E73',  # Green
    'E': '#CC79A7',  # Pink
    'F': '#F0E442',  # Yellow
}

VARIANT_LABELS = {
    'A': 'Variant A (Random Control)',
    'B': 'Variant B (HSI Collapse)',
    'C': 'Variant C',
    'D': 'Variant D',
    'E': 'Variant E',
    'F': 'Variant F',
}


# =============================================================================
# PROGRESS BAR
# =============================================================================

class ProgressBar:
    """Simple progress bar for console output."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.bar_length = 30
    
    def update(self, step: int = 1):
        self.current += step
        self._draw()
    
    def _draw(self):
        filled = int(self.bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        percent = 100 * self.current / self.total
        sys.stdout.write(f'\r{self.description} [{bar}] {self.current}/{self.total} ({percent:.0f}%)')
        sys.stdout.flush()
        if self.current >= self.total:
            print()  # Newline at end


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def discover_data(results_dir: Path) -> Dict[str, Dict[int, dict]]:
    """
    Scan results directory for available Level 1 analysis files.
    
    Returns:
        {variant: {iteration: {'path': str, 'min_len': int, 'max_len': int}}}
    """
    pattern = str(results_dir / "level1_analysis_var*_iter*_min*_max*.json")
    files = glob.glob(pattern)
    
    data = {}
    for f in files:
        match = re.search(r'var([A-Z])_iter(\d+)_min(\d+)_max(\d+)', f)
        if match:
            variant, iteration, min_len, max_len = match.groups()
            if variant not in data:
                data[variant] = {}
            data[variant][int(iteration)] = {
                'path': f,
                'min_len': int(min_len),
                'max_len': int(max_len)
            }
    
    return data


def load_analysis(filepath: str) -> dict:
    """Load a Level 1 analysis JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# SINGLE FIGURES (1 variant, 1 iteration)
# =============================================================================

def generate_markov_heatmap(data: dict, variant: str, iteration: int, 
                            output_dir: Path, fmt: str, dpi: int) -> str:
    """Generate Markov transition probability heatmap."""
    rules = data.get('rules', [])
    
    # Extract order-2 Markov transitions
    transitions = {'00': {}, '01': {}, '10': {}, '11': {}}
    
    for rule in rules:
        if rule.get('order') == 2 and rule.get('rule_type') == 'markov_transition':
            ctx = rule.get('context', '')
            pred = rule.get('prediction', '')
            conf = rule.get('confidence', 0)
            if ctx in transitions and pred in ['0', '1']:
                transitions[ctx][pred] = conf
    
    # Build 4x2 matrix (contexts x predictions)
    contexts = ['00', '01', '10', '11']
    matrix = np.zeros((4, 2))
    
    for i, ctx in enumerate(contexts):
        matrix[i, 0] = transitions[ctx].get('0', 0)
        matrix[i, 1] = transitions[ctx].get('1', 0)
    
    # Create figure
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6, 8))
    
    if HAS_SEABORN:
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['→ 0', '→ 1'],
                    yticklabels=contexts,
                    ax=ax, vmin=0, vmax=1,
                    cbar_kws={'label': 'Probability'})
    else:
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['→ 0', '→ 1'])
        ax.set_yticks(range(4))
        ax.set_yticklabels(contexts)
        for i in range(4):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center')
        plt.colorbar(im, ax=ax, label='Probability')
    
    ax.set_xlabel('Next Bit')
    ax.set_ylabel('Context (Previous 2 bits)')
    ax.set_title(f'Markov Transition Matrix\n{VARIANT_LABELS.get(variant, variant)} - Iteration {iteration}')
    
    plt.tight_layout()
    
    filename = f'markov_heatmap_{variant}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


def generate_top_patterns(data: dict, variant: str, iteration: int,
                          output_dir: Path, fmt: str, dpi: int, top_n: int = 10) -> str:
    """Generate horizontal bar chart of top patterns by frequency."""
    patterns = data.get('patterns', {}).get('observable', [])

    if not patterns:
        return None

    # Sort by recurrence (or occurrences for backwards compatibility) and take top N
    def get_count(p):
        return p.get('recurrence', 0) or p.get('occurrences', 0) or 0

    sorted_patterns = sorted(patterns, key=get_count, reverse=True)[:top_n]

    # Get pattern data (field may be 'pattern' or 'pattern_data')
    labels = [p.get('pattern_data', p.get('pattern', ''))[:20] for p in sorted_patterns]
    values = [get_count(p) for p in sorted_patterns]

    # Create figure
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    color = COLORS.get(variant, '#333333')
    y_pos = np.arange(len(labels))

    # Determine scale factor for large numbers
    max_val = max(values) if values else 0
    if max_val >= 1e9:
        scale = 1e9
        scale_label = '×10⁹'
    elif max_val >= 1e8:
        scale = 1e8
        scale_label = '×10⁸'
    elif max_val >= 1e6:
        scale = 1e6
        scale_label = '×10⁶'
    else:
        scale = 1
        scale_label = ''

    scaled_values = [v / scale for v in values]

    bars = ax.barh(y_pos, scaled_values, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'`{l}`' for l in labels], fontfamily='monospace')
    ax.invert_yaxis()  # Top pattern at top

    xlabel = f'Occurrences ({scale_label})' if scale_label else 'Occurrences'
    ax.set_xlabel(xlabel)
    ax.set_title(f'Top {top_n} Patterns by Frequency\n{VARIANT_LABELS.get(variant, variant)} - Iteration {iteration}')

    # Add value labels (show original values formatted)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(scaled_values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=9)

    if scaled_values and max(scaled_values) > 0:
        ax.set_xlim(0, max(scaled_values) * 1.15)
    plt.tight_layout()

    filename = f'top_patterns_{variant}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


def generate_confidence_distribution(data: dict, variant: str, iteration: int,
                                     output_dir: Path, fmt: str, dpi: int) -> str:
    """Generate histogram of rule confidence distribution."""
    rules = data.get('rules', [])

    if not rules:
        return None

    confidences = [r.get('confidence', 0) * 100 for r in rules]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    color = COLORS.get(variant, '#333333')
    ax.hist(confidences, bins=50, color=color, alpha=0.7, edgecolor='white')

    ax.axvline(x=99, color='red', linestyle='--', linewidth=2, label='99% threshold')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2, label='Random (50%)')

    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Number of Rules')
    ax.set_title(f'Rule Confidence Distribution\n{VARIANT_LABELS.get(variant, variant)} - Iteration {iteration}')
    ax.legend()

    # Add stats annotation
    pct_deterministic = sum(1 for c in confidences if c >= 99) / len(confidences) * 100
    avg_conf = np.mean(confidences)
    ax.text(0.02, 0.98, f'Avg: {avg_conf:.1f}%\n≥99%: {pct_deterministic:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    filename = f'confidence_dist_{variant}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


# =============================================================================
# COMPARISON FIGURES (2+ variants, same iteration)
# =============================================================================

def generate_confidence_comparison(data1: dict, data2: dict,
                                   v1: str, v2: str, iteration: int,
                                   output_dir: Path, fmt: str, dpi: int) -> str:
    """Generate bar chart comparing confidence metrics between variants."""

    def get_metrics(data):
        rules = data.get('rules', [])
        if not rules:
            return {'avg': 0, 'pct_99': 0, 'pct_90': 0}
        confidences = [r.get('confidence', 0) * 100 for r in rules]
        return {
            'avg': np.mean(confidences),
            'pct_99': sum(1 for c in confidences if c >= 99) / len(confidences) * 100,
            'pct_90': sum(1 for c in confidences if c >= 90) / len(confidences) * 100,
        }

    m1 = get_metrics(data1)
    m2 = get_metrics(data2)

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35

    metrics = ['Average\nConfidence', 'Rules ≥99%\n(Deterministic)', 'Rules ≥90%']
    vals1 = [m1['avg'], m1['pct_99'], m1['pct_90']]
    vals2 = [m2['avg'], m2['pct_99'], m2['pct_90']]

    bars1 = ax.bar(x - width/2, vals1, width, label=VARIANT_LABELS.get(v1, v1), color=COLORS.get(v1, '#333'))
    bars2 = ax.bar(x + width/2, vals2, width, label=VARIANT_LABELS.get(v2, v2), color=COLORS.get(v2, '#666'))

    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Rule Confidence Comparison - Iteration {iteration}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    filename = f'confidence_cmp_{v1}_vs_{v2}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


def generate_confidence_histogram_comparison(data1: dict, data2: dict,
                                              v1: str, v2: str, iteration: int,
                                              output_dir: Path, fmt: str, dpi: int) -> str:
    """Generate overlaid histograms comparing confidence distributions."""
    rules1 = data1.get('rules', [])
    rules2 = data2.get('rules', [])

    if not rules1 or not rules2:
        return None

    conf1 = [r.get('confidence', 0) * 100 for r in rules1]
    conf2 = [r.get('confidence', 0) * 100 for r in rules2]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(conf1, bins=50, alpha=0.6, label=VARIANT_LABELS.get(v1, v1), color=COLORS.get(v1, '#333'))
    ax.hist(conf2, bins=50, alpha=0.6, label=VARIANT_LABELS.get(v2, v2), color=COLORS.get(v2, '#666'))

    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2, label='Random baseline')

    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Number of Rules')
    ax.set_title(f'Confidence Distribution Comparison - Iteration {iteration}')
    ax.legend()

    plt.tight_layout()

    filename = f'confidence_hist_{v1}_{v2}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


# =============================================================================
# EVOLUTION FIGURES (1 variant, 2+ iterations)
# =============================================================================

def generate_confidence_evolution(variant_data: Dict[int, dict], variant: str,
                                   output_dir: Path, fmt: str, dpi: int) -> str:
    """Generate line chart of confidence metrics evolution over iterations."""
    iterations = sorted(variant_data.keys())

    avg_confs = []
    pct_99s = []

    for it in iterations:
        data = variant_data[it]['data']
        rules = data.get('rules', [])
        if rules:
            confs = [r.get('confidence', 0) * 100 for r in rules]
            avg_confs.append(np.mean(confs))
            pct_99s.append(sum(1 for c in confs if c >= 99) / len(confs) * 100)
        else:
            avg_confs.append(0)
            pct_99s.append(0)

    plt.rcParams.update(STYLE)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = COLORS.get(variant, '#333')

    ax1.plot(iterations, avg_confs, 'o-', color=color, linewidth=2, markersize=8, label='Avg Confidence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Confidence (%)')
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(iterations, pct_99s, 's--', color=color, alpha=0.6, linewidth=2, markersize=8, label='% Deterministic (≥99%)')
    ax2.set_ylabel('% Deterministic Rules')
    ax2.set_ylim(0, 105)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    ax1.set_title(f'Confidence Evolution\n{VARIANT_LABELS.get(variant, variant)}')
    ax1.set_xticks(iterations)

    plt.tight_layout()

    filename = f'confidence_evol_{variant}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


def generate_patterns_evolution(variant_data: Dict[int, dict], variant: str,
                                 output_dir: Path, fmt: str, dpi: int) -> str:
    """Generate line chart of pattern count evolution."""
    iterations = sorted(variant_data.keys())

    pattern_counts = []
    for it in iterations:
        data = variant_data[it]['data']
        patterns = data.get('patterns', {}).get('observable', [])
        pattern_counts.append(len(patterns))

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    color = COLORS.get(variant, '#333')
    ax.plot(iterations, pattern_counts, 'o-', color=color, linewidth=2, markersize=10)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Unique Patterns')
    ax.set_title(f'Pattern Count Evolution\n{VARIANT_LABELS.get(variant, variant)}')
    ax.set_xticks(iterations)

    # Add value labels
    for x, y in zip(iterations, pattern_counts):
        ax.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()

    filename = f'patterns_evol_{variant}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filename


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def generate_all_figures(discovered: Dict[str, Dict[int, dict]],
                         variants: Optional[List[str]] = None,
                         iterations: Optional[List[int]] = None,
                         output_dir: Path = None,
                         fmt: str = 'png',
                         dpi: int = 300) -> List[str]:
    """Generate all possible figures from discovered data."""

    generated = []

    # Filter by requested variants/iterations
    if variants:
        discovered = {v: discovered[v] for v in variants if v in discovered}

    if iterations:
        for v in discovered:
            discovered[v] = {i: discovered[v][i] for i in iterations if i in discovered[v]}

    # Remove empty variants
    discovered = {v: d for v, d in discovered.items() if d}

    if not discovered:
        print("❌ No data found matching the criteria.")
        return []

    # Load all data
    print("\n📂 Loading analysis data...")
    for variant in discovered:
        for iteration in discovered[variant]:
            filepath = discovered[variant][iteration]['path']
            print(f"   Loading {variant} iter {iteration}...")
            discovered[variant][iteration]['data'] = load_analysis(filepath)

    # Count total figures to generate
    variant_list = list(discovered.keys())
    total_figures = 0

    # Single figures: 3 per variant/iteration combo
    for v in variant_list:
        total_figures += len(discovered[v]) * 3

    # Comparison figures: for each pair with common iterations
    if len(variant_list) >= 2:
        from itertools import combinations
        for v1, v2 in combinations(variant_list, 2):
            common = set(discovered[v1].keys()) & set(discovered[v2].keys())
            total_figures += len(common) * 2  # 2 comparison figures per common iteration

    # Evolution figures: 2 per variant with 2+ iterations
    for v in variant_list:
        if len(discovered[v]) >= 2:
            total_figures += 2

    # Survival analysis figures: 3 per variant with 2+ iterations
    if HAS_SURVIVAL:
        for v in variant_list:
            if len(discovered[v]) >= 2:
                total_figures += 3  # flow, recurrence, lengths

    print(f"\n📊 Generating {total_figures} figures...")
    progress = ProgressBar(total_figures, "Generating")

    # =========================
    # SINGLE FIGURES
    # =========================
    for variant in variant_list:
        for iteration in sorted(discovered[variant].keys()):
            data = discovered[variant][iteration]['data']

            # Markov heatmap
            fname = generate_markov_heatmap(data, variant, iteration, output_dir, fmt, dpi)
            if fname:
                generated.append(fname)
            progress.update()

            # Top patterns
            fname = generate_top_patterns(data, variant, iteration, output_dir, fmt, dpi)
            if fname:
                generated.append(fname)
            progress.update()

            # Confidence distribution
            fname = generate_confidence_distribution(data, variant, iteration, output_dir, fmt, dpi)
            if fname:
                generated.append(fname)
            progress.update()

    # =========================
    # COMPARISON FIGURES
    # =========================
    if len(variant_list) >= 2:
        from itertools import combinations
        for v1, v2 in combinations(variant_list, 2):
            common_iters = set(discovered[v1].keys()) & set(discovered[v2].keys())
            for iteration in sorted(common_iters):
                data1 = discovered[v1][iteration]['data']
                data2 = discovered[v2][iteration]['data']

                # Confidence comparison bars
                fname = generate_confidence_comparison(data1, data2, v1, v2, iteration, output_dir, fmt, dpi)
                if fname:
                    generated.append(fname)
                progress.update()

                # Confidence histogram overlay
                fname = generate_confidence_histogram_comparison(data1, data2, v1, v2, iteration, output_dir, fmt, dpi)
                if fname:
                    generated.append(fname)
                progress.update()

    # =========================
    # EVOLUTION FIGURES
    # =========================
    for variant in variant_list:
        if len(discovered[variant]) >= 2:
            # Confidence evolution
            fname = generate_confidence_evolution(discovered[variant], variant, output_dir, fmt, dpi)
            if fname:
                generated.append(fname)
            progress.update()

            # Pattern evolution
            fname = generate_patterns_evolution(discovered[variant], variant, output_dir, fmt, dpi)
            if fname:
                generated.append(fname)
            progress.update()

    # =========================
    # SURVIVAL ANALYSIS FIGURES
    # =========================
    if HAS_SURVIVAL:
        for variant in variant_list:
            if len(discovered[variant]) >= 2:
                # Run survival analysis for variants with multiple iterations
                fnames = survival_analysis.run_survival_analysis(
                    discovered, variant, output_dir, fmt, dpi, verbose=True
                )
                generated.extend(fnames)
                progress.update()

    return generated


def main():
    parser = argparse.ArgumentParser(
        description='Generate Level 1 analysis figures for HSI scientific paper.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python level1_figures.py --all
  python level1_figures.py --variants B A --iterations 18
  python level1_figures.py --format pdf --dpi 600
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Generate all possible figures from available data')
    parser.add_argument('--variants', nargs='+', metavar='V',
                        help='Variants to include (default: all found)')
    parser.add_argument('--iterations', nargs='+', type=int, metavar='N',
                        help='Iterations to include (default: all found)')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                        help='Output directory (default: results/figures)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='Image format (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Determine paths using unified structure
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"

    # Use unified output path
    if args.output_dir == 'results/figures':
        # Default - use new structure
        from utils.file_saver import get_output_path, relative_path
        output_dir = get_output_path(1, "figures", "")
    else:
        # Custom path specified
        output_dir = script_dir.parent / args.output_dir

    print("=" * 60)
    print("📊 ISH Level 1 Figure Generator")
    print("=" * 60)

    # Discover data - check both old and new locations
    print("\n🔍 Scanning for available data...")
    discovered = discover_data(results_dir)

    # Also check new structure
    new_analysis_dir = results_dir / "level1" / "analysis"
    if new_analysis_dir.exists():
        discovered_new = discover_data(new_analysis_dir)
        for v, data in discovered_new.items():
            if v not in discovered:
                discovered[v] = data
            else:
                discovered[v].update(data)

    if not discovered:
        print("❌ No Level 1 analysis files found in results/")
        print("   Run level1_orchestrator.py first to generate analysis data.")
        return 1

    # Print discovered data
    print("\n📋 Data discovered:")
    for variant in sorted(discovered.keys()):
        iters = sorted(discovered[variant].keys())
        print(f"   Variant {variant}: iterations {iters}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from utils.file_saver import relative_path
        print(f"\n💾 Output directory: {relative_path(output_dir)}")
    except ImportError:
        print(f"\n💾 Output directory: {output_dir}")
    print(f"📐 Format: {args.format.upper()} @ {args.dpi} DPI")

    # Generate figures
    generated = generate_all_figures(
        discovered,
        variants=args.variants,
        iterations=args.iterations,
        output_dir=output_dir,
        fmt=args.format,
        dpi=args.dpi
    )

    # Summary
    print(f"\n✅ Complete! Generated {len(generated)} figures:")
    for fname in generated:
        print(f"   • {fname}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

