"""
📐 Geometric Analyzer: Connection (Γ), Curvature (𝓡), Torsion (𝓣)

Implements informational geometric operators from ISH paper Section 4.3:
- Γ (Connection): Pattern transport between states
- 𝓡 (Curvature): Rule interference measurement
- 𝓣 (Torsion): Positional asymmetries

Author: Iban Borràs with collaboration of Augment Agent (Sophia)
Date: December 2025
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import time


class GeometricAnalyzer:
    """
    Analyzes geometric properties of ISH patterns and rules.
    
    Implements operators Γ, 𝓡, 𝓣 from the ISH paper.
    """
    
    def __init__(self, base_path: str = "results/level1/analysis"):
        """
        Initialize the geometric analyzer.

        Args:
            base_path: Base path for Level 1 analysis results directory
        """
        self.base_path = Path(base_path)
        self.results: Dict[str, Any] = {}
        
    def check_prerequisites(
        self,
        variant: str,
        iteration: int,
        require_multi_iteration: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if required Level 1 result files exist.

        Only Level 1 results are needed - we use pre-computed pattern positions,
        so no need to load raw phi_snapshots.

        Args:
            variant: Variant code (B, D, etc.)
            iteration: Target iteration
            require_multi_iteration: Whether to require 3 consecutive iterations

        Returns:
            Tuple of (all_ok, status_dict)
        """
        import glob
        import re

        status = {
            "variant": variant,
            "iteration": iteration,
            "files": {},
            "missing": [],
            "available": [],
            "configs": {}  # Track configs per iteration for consistency check
        }

        # Check Level 1 results for required iterations
        iterations_to_check = [iteration]
        if require_multi_iteration:
            iterations_to_check = [iteration - 2, iteration - 1, iteration]

        for i in iterations_to_check:
            level1_file = None
            level1_config = None

            # Try to find any matching file with glob pattern
            patterns_to_try = [
                str(self.base_path / f"level1_analysis_var{variant}_iter{i}_min*_max*.json"),
                str(self.base_path / f"level1_analysis_var{variant}_iter{i}.json"),
            ]

            for pattern in patterns_to_try:
                matches = glob.glob(pattern)
                if matches:
                    level1_file = Path(matches[0])
                    # Extract min/max from filename if present
                    match = re.search(r'_min(\d+)_max(\d+)\.json$', str(level1_file))
                    if match:
                        level1_config = {
                            'min_pattern_length': int(match.group(1)),
                            'max_pattern_length': int(match.group(2))
                        }
                    break

            status["files"][f"level1_iter{i}"] = {
                "path": str(level1_file) if level1_file else f"level1_analysis_var{variant}_iter{i}_min*_max*.json",
                "exists": level1_file is not None,
                "config": level1_config
            }

            if level1_file:
                status["available"].append(str(level1_file))
                if level1_config:
                    status["configs"][i] = level1_config
            else:
                status["missing"].append(f"level1_analysis_var{variant}_iter{i}_min*_max*.json")

        # Check config consistency across iterations
        if require_multi_iteration and len(status["configs"]) > 1:
            configs_list = list(status["configs"].values())
            first_config = configs_list[0]
            all_match = all(c == first_config for c in configs_list)
            status["configs_consistent"] = all_match
            if not all_match:
                status["config_warning"] = "Different min/max parameters used across iterations!"

        all_ok = len(status["missing"]) == 0
        return all_ok, status
    
    def print_missing_data_warning(self, status: Dict[str, Any]) -> None:
        """Print detailed warning about missing data with instructions."""
        variant = status["variant"]
        iteration = status["iteration"]

        print("\n" + "=" * 70)
        print("⚠️  INSUFFICIENT DATA FOR GEOMETRIC ANALYSIS")
        print("=" * 70)
        print("\nRequired Level 1 results status:")

        for name, info in status["files"].items():
            icon = "✅" if info["exists"] else "❌"
            config_str = ""
            if info.get("config"):
                cfg = info["config"]
                config_str = f" (min={cfg['min_pattern_length']}, max={cfg['max_pattern_length']})"
            print(f"  {icon} {info['path']}{config_str}")

        # Config consistency warning
        if status.get("config_warning"):
            print(f"\n⚠️  {status['config_warning']}")
            print("   All iterations must use the same --min-len and --max-len parameters!")

        # Missing files instructions
        missing_iters = []
        for name, info in status["files"].items():
            if not info["exists"] and name.startswith("level1_iter"):
                iter_num = name.replace("level1_iter", "")
                missing_iters.append(iter_num)

        if missing_iters:
            print(f"\n📋 Run Level 1 analysis for missing iterations:")
            for iter_num in missing_iters:
                print(f"     python agents/level1_orchestrator.py -v {variant} -i {iter_num} --min-len 10 --max-len 50 --no-cache --report")

        print(f"\n💡 Alternative: Run Torsion-only analysis (single iteration):")
        print(f"     python agents/geometric_analyzer.py -v {variant} -i {iteration} --torsion-only")
        print("=" * 70 + "\n")

    def load_level1_results(self, variant: str, iteration: int) -> Optional[Dict[str, Any]]:
        """Load Level 1 results from JSON file."""
        import glob

        # Try multiple formats including min/max in filename
        patterns_to_try = [
            str(self.base_path / f"level1_analysis_var{variant}_iter{iteration}_min*_max*.json"),
            str(self.base_path / f"level1_analysis_var{variant}_iter{iteration}.json"),
            str(self.base_path / "level1" / f"results_{variant}_iter{iteration}.json"),
        ]

        for pattern in patterns_to_try:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], 'r', encoding='utf-8') as f:
                    return json.load(f)
        return None

    def calculate_torsion(
        self,
        patterns: List[Dict[str, Any]],
        seq_len: int,
        num_segments: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate Torsion 𝓣 for patterns using pre-computed positions.

        Torsion measures positional asymmetries:
        𝓣_α^(k)(P) = Γ_α^(k)(P) - Γ_α^(k)(P^shift)

        Operationally: Compare pattern frequency across segments.

        Uses pattern positions from Level 1 results (no need to load Φ sequence).
        """
        print(f"\n🔄 Calculating Torsion (𝓣) for {len(patterns)} patterns...")

        segment_size = seq_len // num_segments

        torsion_results = {
            "num_segments": num_segments,
            "segment_size": segment_size,
            "sequence_length": seq_len,
            "patterns_analyzed": 0,
            "patterns_with_torsion": 0,
            "pattern_details": [],
            "mean_asymmetry": 0.0,
            "max_asymmetry": 0.0
        }

        asymmetries = []

        for p in patterns:
            pattern_data = p.get('pattern_data', p.get('pattern', ''))
            positions = p.get('positions', [])

            if not pattern_data or len(pattern_data) < 2 or not positions:
                continue

            # Count occurrences in each segment using pre-computed positions
            segment_counts = []
            for seg_idx in range(num_segments):
                start = seg_idx * segment_size
                end = start + segment_size if seg_idx < num_segments - 1 else seq_len
                # Count positions within this segment
                count = len([pos for pos in positions if start <= pos < end])
                segment_counts.append(count)

            total_count = sum(segment_counts)
            if total_count == 0:
                continue

            # Calculate asymmetry (normalized variance)
            mean_count = total_count / num_segments
            variance = np.var(segment_counts)
            asymmetry = variance / (mean_count ** 2) if mean_count > 0 else 0

            # Determine if there's significant torsion
            has_torsion = asymmetry > 0.1  # Threshold for "significant" asymmetry

            pattern_info = {
                "pattern": pattern_data[:20] + "..." if len(pattern_data) > 20 else pattern_data,
                "pattern_id": p.get('pattern_id', 'unknown'),
                "segment_counts": segment_counts,
                "total_count": total_count,
                "asymmetry": float(asymmetry),
                "has_torsion": has_torsion
            }

            torsion_results["pattern_details"].append(pattern_info)
            torsion_results["patterns_analyzed"] += 1

            if has_torsion:
                torsion_results["patterns_with_torsion"] += 1

            asymmetries.append(asymmetry)

        if asymmetries:
            torsion_results["mean_asymmetry"] = float(np.mean(asymmetries))
            torsion_results["max_asymmetry"] = float(np.max(asymmetries))

        # Sort by asymmetry for top patterns
        torsion_results["pattern_details"].sort(
            key=lambda x: x["asymmetry"], reverse=True
        )

        # Keep only top 50 for report
        torsion_results["top_torsion_patterns"] = torsion_results["pattern_details"][:50]
        del torsion_results["pattern_details"]  # Remove full list to save space

        print(f"   ✅ Analyzed: {torsion_results['patterns_analyzed']} patterns")
        print(f"   🔄 With torsion: {torsion_results['patterns_with_torsion']}")
        print(f"   📊 Mean asymmetry: {torsion_results['mean_asymmetry']:.4f}")

        return torsion_results

    def analyze(
        self,
        variant: str,
        iteration: int,
        torsion_only: bool = False,
        max_chars: int = 50_000_000
    ) -> Dict[str, Any]:
        """
        Run geometric analysis on a variant/iteration.

        Args:
            variant: Variant code (B, D, etc.)
            iteration: Target iteration
            torsion_only: If True, only calculate Torsion (single iteration)
            max_chars: Maximum characters to analyze

        Returns:
            Complete geometric analysis results
        """
        start_time = time.time()

        print(f"\n📐 Geometric Analysis - ISH")
        print("=" * 60)
        print(f"📁 Variant: {variant}")
        print(f"🔢 Iteration: {iteration}")
        print(f"📊 Mode: {'Torsion only' if torsion_only else 'Full (Γ, 𝓡, 𝓣)'}")

        # Check prerequisites
        ok, status = self.check_prerequisites(
            variant, iteration,
            require_multi_iteration=not torsion_only
        )

        if not ok:
            self.print_missing_data_warning(status)
            return {"error": "Missing required data", "status": status}

        # Load Level 1 results only (no need for phi_snapshots - we use pre-computed positions)
        print("\n📥 Loading Level 1 results...")
        level1_results = self.load_level1_results(variant, iteration)

        if not level1_results:
            print("❌ Failed to load Level 1 results")
            return {"error": "Failed to load data"}

        # Extract patterns from Level 1 results
        patterns = level1_results.get('patterns', {})
        observable = patterns.get('observable', [])
        structural = patterns.get('structural', [])
        all_patterns = observable + structural

        # Get sequence length from Level 1 metadata
        char_counts = level1_results.get('char_counts', {})
        seq_len = char_counts.get('observable', char_counts.get('structural', 0))

        # Get config used for this analysis
        config = level1_results.get('config', {})

        print(f"   Loaded {len(all_patterns)} patterns")
        print(f"   Φ length: {seq_len:,} chars")
        if config:
            print(f"   Config: min={config.get('min_pattern_length')}, max={config.get('max_pattern_length')}")

        # Initialize results
        self.results = {
            "variant": variant,
            "iteration": iteration,
            "analysis_mode": "torsion_only" if torsion_only else "full",
            "phi_length": seq_len,
            "patterns_count": len(all_patterns),
            "level1_config": config,
            "torsion": None,
            "connection": None,
            "curvature": None,
            "energy": None
        }

        # Calculate Torsion (always) - uses pre-computed positions from Level 1
        self.results["torsion"] = self.calculate_torsion(all_patterns, seq_len)

        if not torsion_only:
            # TODO: Implement Connection and Curvature in Phase 2
            print("\n⚠️  Connection (Γ) and Curvature (𝓡) not yet implemented")
            print("   Use --torsion-only for now, or wait for Phase 2")

        elapsed = time.time() - start_time
        self.results["analysis_time_seconds"] = elapsed

        print(f"\n✅ Analysis complete in {elapsed:.1f}s")

        return self.results

    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save results to JSON file using unified structure."""
        if not self.results:
            raise ValueError("No results to save. Run analyze() first.")

        variant = self.results.get("variant", "X")
        iteration = self.results.get("iteration", 0)

        if output_path is None:
            # Use unified structure
            try:
                from utils.file_saver import get_output_path, relative_path
                filename = f"geometric_{variant}_iter{iteration}.json"
                output_path = get_output_path(1, "metrics", filename)
            except ImportError:
                output_dir = self.base_path / "level1"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"geometric_{variant}_iter{iteration}.json"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Log with relative path
        try:
            from utils.file_saver import relative_path
            print(f"💾 Saved: {relative_path(output_path)}")
        except ImportError:
            print(f"📝 Results saved to: {output_path}")
        return str(output_path)

    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        if not self.results:
            print("No results available. Run analyze() first.")
            return

        print("\n" + "=" * 60)
        print("📐 GEOMETRIC ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📁 Variant: {self.results['variant']}")
        print(f"🔢 Iteration: {self.results['iteration']}")
        print(f"⏱️  Time: {self.results.get('analysis_time_seconds', 0):.1f}s")

        torsion = self.results.get("torsion")
        if torsion:
            print(f"\n🔄 TORSION (𝓣):")
            print(f"   Patterns analyzed: {torsion['patterns_analyzed']}")
            print(f"   Patterns with torsion: {torsion['patterns_with_torsion']}")
            print(f"   Mean asymmetry: {torsion['mean_asymmetry']:.4f}")
            print(f"   Max asymmetry: {torsion['max_asymmetry']:.4f}")

            top = torsion.get("top_torsion_patterns", [])[:5]
            if top:
                print(f"\n   Top asymmetric patterns:")
                for i, p in enumerate(top, 1):
                    print(f"   {i}. {p['pattern']} (asymmetry: {p['asymmetry']:.3f})")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Geometric Analysis: Connection (Γ), Curvature (𝓡), Torsion (𝓣)"
    )
    parser.add_argument("--variant", "-v", required=True, help="Variant code (B, D, etc.)")
    parser.add_argument("--iteration", "-i", type=int, required=True, help="Iteration number")
    parser.add_argument("--torsion-only", action="store_true", help="Only calculate Torsion")
    parser.add_argument("--max-chars", type=int, default=50_000_000, help="Max chars to analyze")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--report", action="store_true", help="Print summary report")

    args = parser.parse_args()

    analyzer = GeometricAnalyzer()
    results = analyzer.analyze(
        variant=args.variant,
        iteration=args.iteration,
        torsion_only=args.torsion_only,
        max_chars=args.max_chars
    )

    if "error" not in results:
        analyzer.save_results(args.output)
        if args.report:
            analyzer.print_summary()


if __name__ == "__main__":
    main()

