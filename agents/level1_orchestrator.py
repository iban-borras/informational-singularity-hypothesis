#!/usr/bin/env python3
"""
HSI Level 1 Analysis Orchestrator
==================================

Orchestrates Level 1 agents (pattern detection, rule inference, validation)
with streaming file loading for memory-efficient processing of large Φ files.

This orchestrator:
1. Loads Φ files using streaming loader (memory-efficient)
2. Coordinates pattern detection agents
3. Manages rule inference and validation
4. Generates consolidated reports
5. Supports intermediate caching to resume long analyses

Author: Iban Borràs with Augment Agent (Sophia)
Date: December 2025
"""

import sys
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.streaming_phi_loader import StreamingPhiLoader, load_phi_for_agents
from agents.pattern_detector import PatternDetector
from agents.structural_pattern_detector import StructuralPatternDetector
from agents.rule_inferer import RuleInferer
from agents.validator import Validator


class Level1Orchestrator:
    """
    Orchestrates Level 1 analysis with memory-efficient file loading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator with optional configuration.
        
        Args:
            config: Configuration dictionary with agent parameters
        """
        self.config = config or self._default_config()
        
        # Initialize agents with unpacked config dicts
        pd_config = self.config.get('pattern_detector', {})
        self.pattern_detector = PatternDetector(
            min_pattern_length=pd_config.get('min_pattern_length', 3),
            max_pattern_length=pd_config.get('max_pattern_length', 50),
            min_occurrences=pd_config.get('min_occurrences', 2),
            similarity_threshold=pd_config.get('similarity_threshold', 0.8)
        )
        self.structural_detector = StructuralPatternDetector()

        ri_config = self.config.get('rule_inferer', {})
        pd_config = self.config.get('pattern_detector', {})
        # Use min_pattern_length as default for min_context_length (consistency)
        default_min_context = pd_config.get('min_pattern_length', 3)
        self.rule_inferer = RuleInferer(
            context_window=ri_config.get('context_window', 5),
            min_context_length=ri_config.get('min_context_length', default_min_context),
            min_rule_confidence=ri_config.get('min_confidence', 0.7),
            max_rule_complexity=ri_config.get('max_rule_complexity', 10)
        )

        val_config = self.config.get('validator', {})
        self.validator = Validator(
            validation_split=val_config.get('validation_ratio', 0.3),
            min_validation_score=val_config.get('confidence_threshold', 0.6),
            stability_threshold=val_config.get('stability_threshold', 0.1)
        )
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'max_chars_in_memory': 50_000_000,  # 50M chars max in memory
            'pattern_detector': {
                'min_pattern_length': 4,
                'max_pattern_length': 32,
                'min_occurrences': 3,
                'similarity_threshold': 0.85
            },
            'rule_inferer': {
                'min_confidence': 0.7,
                'min_support': 0.05,
                'max_antecedent_length': 3
            },
            'validator': {
                'validation_ratio': 0.2,
                'confidence_threshold': 0.8
            },
            'adaptive_sampling': {
                'min_coverage_percent': 5.0,
                'structure_factor': 50.0,
                'target_confidence': 0.95
            }
        }

    def calculate_optimal_sample_size(
        self,
        full_sequence_length: int,
        max_pattern_length: int,
        min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate optimal sample size for reliable pattern detection.

        Args:
            full_sequence_length: Total bits in the sequence
            max_pattern_length: Longest patterns to detect
            min_occurrences: Minimum occurrences to count as detected

        Returns:
            Dictionary with sampling recommendations
        """
        import math

        config = self.config.get('adaptive_sampling', {})
        min_coverage = config.get('min_coverage_percent', 5.0) / 100.0
        structure_factor = config.get('structure_factor', 50.0)

        # Expected frequency of rarest pattern (with HSI structure factor)
        # For random: 2^(-k), for structured HSI: 2^(-k) * structure_factor
        rarest_freq = (2 ** (-max_pattern_length)) * structure_factor

        # Poisson lambda needed for reliable detection (5 expected occurrences)
        poisson_lambda = max(min_occurrences * 2, 5)

        # Theoretical optimal sample (may exceed sequence length)
        theoretical_optimal = int(poisson_lambda / rarest_freq) if rarest_freq > 0 else full_sequence_length

        # Practical limits
        optimal_sample = min(theoretical_optimal, full_sequence_length)
        minimum_sample = int(full_sequence_length * min_coverage)

        # Memory estimation (2 bytes per bit for processing)
        memory_optimal_gb = (optimal_sample * 2) / (1024**3)
        memory_minimum_gb = (minimum_sample * 2) / (1024**3)

        # Coverage percentages
        optimal_coverage = (optimal_sample / full_sequence_length) * 100 if full_sequence_length > 0 else 100

        return {
            'full_sequence_length': full_sequence_length,
            'max_pattern_length': max_pattern_length,
            'optimal_sample': optimal_sample,
            'optimal_coverage_percent': optimal_coverage,
            'minimum_sample': minimum_sample,
            'minimum_coverage_percent': min_coverage * 100,
            'memory_optimal_gb': memory_optimal_gb,
            'memory_minimum_gb': memory_minimum_gb,
            'theoretical_optimal': theoretical_optimal,
            'exceeds_sequence': theoretical_optimal > full_sequence_length
        }

    def _get_cache_path(self, struct_gz_path: str, max_chars: int) -> Path:
        """Generate cache file path based on input file and parameters."""
        path_hash = hashlib.md5(f"{struct_gz_path}_{max_chars}".encode()).hexdigest()[:12]
        filename = Path(struct_gz_path).stem.replace('.struct', '')
        cache_dir = Path(__file__).parent.parent / "results" / "level1" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"level1_cache_{filename}_{path_hash}.pkl"

    def _load_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load cached intermediate results if available."""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                print(f"📦 Cache found: {cache_path.name}")
                print(f"   Cached phases: {list(cache_data.keys())}")
                return cache_data
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}")
                return None
        return None

    def _save_cache(self, cache_path: Path, cache_data: Dict[str, Any], phase: str) -> None:
        """Save intermediate results to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cache saved after {phase}: {cache_path.name}")
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")

    def analyze_file(
        self,
        struct_gz_path: str,
        max_chars: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform full Level 1 analysis on a .struct.gz file.

        Args:
            struct_gz_path: Path to .struct.gz file
            max_chars: Maximum characters to analyze (None = use config default)
            use_cache: Whether to use/save intermediate cache (default: True)

        Returns:
            Dictionary with complete analysis results
        """
        start_time = time.time()

        max_chars = max_chars or self.config.get('max_chars_in_memory', 50_000_000)

        print(f"🔬 Level 1 Analysis - HSI")
        print("=" * 60)
        print(f"📁 File: {struct_gz_path}")
        max_chars_display = "unlimited" if max_chars >= 10_000_000_000 else f"{max_chars:,}"
        print(f"📊 Max chars: {max_chars_display}")
        print(f"💾 Cache: {'enabled' if use_cache else 'disabled'}")

        # Cache setup
        cache_path = self._get_cache_path(struct_gz_path, max_chars)
        cache_data: Dict[str, Any] = {}

        if use_cache:
            cache_data = self._load_cache(cache_path) or {}

        # Load data (always needed for validation)
        print("\n📥 Loading Φ data...")

        # Show progress for large files (>100M chars target)
        show_progress = max_chars is None or max_chars > 100_000_000

        # Load structural (with parentheses) and observable (0/1 only)
        print("   [1/2] Structural sequence...")
        phi_structural, metadata = load_phi_for_agents(
            struct_gz_path, max_chars=max_chars, show_progress=show_progress
        )
        print("   [2/2] Observable sequence...")
        phi_observable, _ = load_phi_for_agents(
            struct_gz_path, max_chars=max_chars, observable_only=True, show_progress=show_progress
        )

        print(f"\n   Structural length: {len(phi_structural):,}")
        print(f"   Observable length: {len(phi_observable):,}")
        print(f"   Metadata format: {metadata.get('format', 'unknown')}")

        # Adaptive Sampling Analysis
        full_sequence_length = metadata.get('sequence_length', len(phi_observable))
        pd_config = self.config.get('pattern_detector', {})
        max_pattern_len = pd_config.get('max_pattern_length', 32)
        min_occurrences = pd_config.get('min_occurrences', 3)

        sampling_info = self.calculate_optimal_sample_size(
            full_sequence_length, max_pattern_len, min_occurrences
        )

        actual_analyzed = len(phi_observable)
        actual_coverage = (actual_analyzed / full_sequence_length * 100) if full_sequence_length > 0 else 100

        print(f"\n📐 Adaptive Sampling Analysis")
        print(f"   Full sequence: {full_sequence_length:,} bits ({full_sequence_length/1e9:.1f}G)")
        print(f"   Bits analyzed: {actual_analyzed:,} ({actual_analyzed/1e9:.1f}G)")
        print(f"   Coverage: {actual_coverage:.2f}%")
        print(f"   Max pattern length: {max_pattern_len} bits")

        # Calculate maximum reliable pattern length for current sample
        import math
        config = self.config.get('adaptive_sampling', {})
        structure_factor = config.get('structure_factor', 50.0)
        min_coverage_percent = config.get('min_coverage_percent', 5.0)

        # k_max where sample is statistically valid: sample >= (min_occ * 2) / (2^(-k) * factor)
        # Solving for k: k = log2(sample * factor / (min_occ * 2))
        if actual_analyzed > 0 and min_occurrences > 0:
            max_reliable_k = math.log2(actual_analyzed * structure_factor / (min_occurrences * 2))
            max_reliable_k = int(max_reliable_k)
        else:
            max_reliable_k = max_pattern_len

        # Confidence assessment with specific pattern length guidance
        if max_reliable_k < max_pattern_len:
            print(f"\n   ⚠️  STATISTICAL COVERAGE WARNING")
            print(f"   Reliable detection up to: {max_reliable_k} bits")
            print(f"   Requested max pattern: {max_pattern_len} bits")
            print(f"   Patterns of {max_reliable_k+1}-{max_pattern_len} bits: MAY BE UNDERDETECTED")

            # Calculate what would be needed for full coverage
            needed_sample = self.calculate_optimal_sample_size(
                full_sequence_length, max_pattern_len, min_occurrences
            )
            needed_bits = min(needed_sample['optimal_sample'], full_sequence_length)
            needed_ram = (needed_bits * 2) / (1024**3)

            print(f"\n   For {max_pattern_len}-bit reliability:")
            print(f"   └─ Need: {needed_bits:,} bits ({needed_bits/1e9:.1f}G)")
            print(f"   └─ RAM: ~{needed_ram:.1f}GB")
            print(f"   └─ Coverage: {(needed_bits/full_sequence_length)*100:.1f}%")
        elif actual_coverage < min_coverage_percent:
            print(f"\n   ⚠️  Coverage below minimum {min_coverage_percent}%")
            print(f"   Recommended sample: {sampling_info['minimum_sample']:,} bits")
        else:
            print(f"\n   ✅ Coverage optimal for {max_pattern_len}-bit pattern detection")
            print(f"   Reliable detection up to: {max_reliable_k} bits")

        # Store sampling info for results
        self._sampling_info = {
            'full_sequence_length': full_sequence_length,
            'analyzed_length': actual_analyzed,
            'coverage_percent': actual_coverage,
            'optimal_coverage_percent': sampling_info['optimal_coverage_percent'],
            'min_coverage_percent': min_coverage_percent,
            'max_pattern_length': max_pattern_len,
            'max_reliable_pattern_length': max_reliable_k,
            'extrapolation_factor': full_sequence_length / actual_analyzed if actual_analyzed > 0 else 1.0
        }

        # Phase 1: Pattern Detection
        if 'patterns' in cache_data and 'structural_patterns' in cache_data:
            print("\n🔍 Phase 1: Pattern Detection... [CACHED]")
            patterns = cache_data['patterns']
            structural_patterns = cache_data['structural_patterns']
            print(f"   ✅ Observable patterns: {len(patterns)} (from cache)")
            print(f"   ✅ Structural patterns: {len(structural_patterns)} (from cache)")
        else:
            print("\n🔍 Phase 1: Pattern Detection...")
            patterns = self.pattern_detector.detect_patterns(phi_observable, phi_structural)
            structural_patterns = self.pattern_detector.get_structural_patterns()

            print(f"   ✅ Observable patterns: {len(patterns)}")
            print(f"   ✅ Structural patterns: {len(structural_patterns)}")

            if use_cache:
                cache_data['patterns'] = patterns
                cache_data['structural_patterns'] = structural_patterns
                self._save_cache(cache_path, cache_data, "pattern_detection")

        # Phase 2: Rule Inference
        all_patterns = patterns + structural_patterns

        if 'rules' in cache_data:
            print("\n🧠 Phase 2: Rule Inference... [CACHED]")
            rules = cache_data['rules']
            print(f"   ✅ Rules inferred: {len(rules)} (from cache)")
        else:
            print("\n🧠 Phase 2: Rule Inference...")
            rules = self.rule_inferer.infer_rules(all_patterns, phi_observable)

            print(f"   ✅ Rules inferred: {len(rules)}")

            if use_cache:
                cache_data['rules'] = rules
                self._save_cache(cache_path, cache_data, "rule_inference")

        # Phase 3: Validation (always run fresh to allow parameter tuning)
        print("\n✓ Phase 3: Validation...")
        # Get pre-computed context index from rule_inferer for O(1) lookups
        context_index = self.rule_inferer.get_context_index()
        # Validator expects: rules, patterns, phi_sequences (list), optional context_index
        validation = self.validator.validate_rules(rules, all_patterns, [phi_observable], context_index=context_index)

        overall = validation.get('overall_metrics', {})
        validated_count = overall.get('rules_above_threshold', 0)
        avg_accuracy = overall.get('mean_accuracy', 0)
        print(f"   ✅ Rules validated: {validated_count}")
        print(f"   📊 Average confidence: {avg_accuracy:.2%}")
        
        elapsed = time.time() - start_time
        
        # Compile results (include config for reproducibility)
        pd_config = self.config.get('pattern_detector', {})
        coverage_info = self.pattern_detector.get_coverage_info()

        self.results = {
            'file': str(struct_gz_path),
            'metadata': metadata,
            'analysis_time_seconds': elapsed,
            'config': {
                'min_pattern_length': pd_config.get('min_pattern_length', 4),
                'max_pattern_length': pd_config.get('max_pattern_length', 32),
                'min_occurrences': pd_config.get('min_occurrences', 3),
                'similarity_threshold': pd_config.get('similarity_threshold', 0.85)
            },
            'coverage': {
                'percentage': coverage_info.get('coverage_percentage', 100.0),
                'sampling_used': coverage_info.get('sampling_used', False),
                'method': coverage_info.get('method', 'full_sequential'),
                'sequence_length': coverage_info.get('sequence_length', len(phi_observable))
            },
            'sampling': getattr(self, '_sampling_info', {
                'full_sequence_length': len(phi_observable),
                'analyzed_length': len(phi_observable),
                'coverage_percent': 100.0,
                'extrapolation_factor': 1.0
            }),
            'char_counts': {
                'structural': len(phi_structural),
                'observable': len(phi_observable)
            },
            'patterns': {
                'observable': self._limit_pattern_positions(patterns),
                'structural': self._limit_pattern_positions(structural_patterns),
                'total': len(patterns) + len(structural_patterns)
            },
            'rules': self._limit_rules_for_storage(rules),
            'rules_total_count': len(rules),  # Store original count for reference
            'validation': validation
        }

    def _limit_pattern_positions(self, patterns: List[Dict], max_positions: int = 100) -> List[Dict]:
        """
        Limit positions per pattern for storage efficiency.

        The 'recurrence' field contains the TOTAL count (unchanged).
        The 'positions' field is limited to a sample for examples/context.
        This does NOT affect statistical analyses that use 'recurrence'.
        """
        total_before = sum(len(p.get('positions', [])) for p in patterns)

        for pattern in patterns:
            positions = pattern.get('positions', [])
            if len(positions) > max_positions:
                pattern['positions'] = positions[:max_positions]

        total_after = sum(len(p.get('positions', [])) for p in patterns)

        if total_before > total_after:
            reduction = (1 - total_after / total_before) * 100 if total_before > 0 else 0
            print(f"   📦 Limiting positions: {total_before:,} → {total_after:,} ({reduction:.0f}% reduction)")

        return patterns

    def _limit_rules_for_storage(self, rules: List[Dict], max_rules: int = 10000) -> List[Dict]:
        """
        Limit rules to top N by confidence for storage efficiency.

        This reduces JSON file size significantly while preserving
        the most scientifically relevant rules.
        """
        if len(rules) <= max_rules:
            return rules

        # Sort by confidence (descending), then by support (descending)
        sorted_rules = sorted(
            rules,
            key=lambda r: (r.get('confidence', 0), r.get('support', 0)),
            reverse=True
        )

        print(f"   📦 Limiting rules for storage: {len(rules):,} → {max_rules:,} (top by confidence)")
        return sorted_rules[:max_rules]
        
        print(f"\n✅ Analysis complete in {elapsed:.1f}s")

        return self.results

    def analyze_variant(
        self,
        variant: str,
        iteration: int,
        base_path: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a specific variant and iteration.

        Args:
            variant: Variant letter (e.g., 'B', 'D', 'E')
            iteration: Iteration number
            base_path: Base path for results (default: auto-detect)
            use_cache: Whether to use/save intermediate cache (default: True)

        Returns:
            Analysis results
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / "results" / "level0" / "phi_snapshots"
        else:
            base_path = Path(base_path)

        struct_path = base_path / f"var_{variant}" / f"phi_iter{iteration}.struct.gz"

        if not struct_path.exists():
            available = self.get_available_iterations(variant)
            max_available = max(available) if available else 0

            print("\n" + "=" * 70)
            print("❌ LEVEL 0 DATA NOT FOUND")
            print("=" * 70)
            print(f"\n📁 Missing file: {struct_path.name}")
            print(f"   Full path: {struct_path}")

            if iteration > max_available and max_available > 0:
                print(f"\n📊 Reason: You requested iteration {iteration}, but Level 0 data")
                print(f"   only exists up to iteration {max_available} for variant {variant}.")
                print(f"\n   The HSI generator must first compute iterations 1→{iteration}")
                print(f"   before Level 1 analysis can be performed.")
            elif max_available == 0:
                print(f"\n📊 Reason: No Level 0 data exists for variant {variant}.")
                print(f"   The HSI generator must first compute the Φ sequences.")
            else:
                print(f"\n📊 Available iterations for variant {variant}: {available}")

            print(f"\n💡 Solution: Generate Level 0 data first with:")
            print(f"   python -m level0.generator --variant {variant} --iterations {iteration}")
            print("\n" + "=" * 70)

            # Raise a custom exception that can be caught cleanly
            raise SystemExit(1)

        print(f"🔬 Analyzing Variant {variant}, Iteration {iteration}")
        return self.analyze_file(str(struct_path), use_cache=use_cache)

    def get_available_iterations(self, variant: str) -> List[int]:
        """Get list of available iterations for a variant."""
        base_path = Path(__file__).parent.parent / "results" / "level0" / "phi_snapshots" / f"var_{variant}"

        if not base_path.exists():
            return []

        iterations = []
        for f in base_path.glob("phi_iter*.struct.gz"):
            try:
                num = int(f.stem.replace("phi_iter", "").replace(".struct", ""))
                iterations.append(num)
            except ValueError:
                continue

        return sorted(iterations)

    def _find_phi_file(self, variant: str, iteration: int) -> Optional[str]:
        """Find the phi file path for a given variant and iteration."""
        base_path = Path(__file__).parent.parent / "results" / "level0" / "phi_snapshots" / f"var_{variant}"
        phi_path = base_path / f"phi_iter{iteration}.struct.gz"

        if phi_path.exists():
            return str(phi_path)
        return None

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📝 Results saved to: {output_path}")

    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        if not self.results:
            print("No results available. Run analyze_file() first.")
            return

        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📁 File: {self.results.get('file', 'unknown')}")
        print(f"⏱️  Time: {self.results.get('analysis_time_seconds', 0):.1f}s")

        counts = self.results.get('char_counts', {})
        print(f"\n📏 Sequence lengths:")
        print(f"   Structural: {counts.get('structural', 0):,}")
        print(f"   Observable: {counts.get('observable', 0):,}")

        patterns = self.results.get('patterns', {})
        print(f"\n🔍 Patterns detected:")
        print(f"   Observable: {len(patterns.get('observable', []))}")
        print(f"   Structural: {len(patterns.get('structural', []))}")

        rules = self.results.get('rules', [])
        print(f"\n🧠 Rules inferred: {len(rules)}")

        if rules:
            # Top 5 rules by confidence
            top_rules = sorted(rules, key=lambda r: r.get('confidence', 0), reverse=True)[:5]
            print("   Top 5 by confidence:")
            for i, rule in enumerate(top_rules, 1):
                conf = rule.get('confidence', 0)
                # Build description from rule fields
                rule_type = rule.get('rule_type', '')
                if rule_type == 'markov_transition':
                    desc = f"'{rule.get('context', '')}' → '{rule.get('prediction', '')}'"
                elif rule_type in ['context_before', 'context_after']:
                    ctx = rule.get('context', '')[:20]
                    produces = rule.get('produces', rule.get('prediction', ''))
                    desc = f"ctx({ctx}...) → {produces}"
                elif rule_type in ['composition', 'overlap_composition']:
                    components = rule.get('components', [])
                    produces = rule.get('produces', [])
                    desc = f"{components} → {produces}"
                elif rule_type == 'periodicity':
                    pattern = rule.get('pattern', '')[:10]
                    period = rule.get('period', 0)
                    desc = f"'{pattern}' period={period}"
                else:
                    desc = rule.get('rule_description', '')[:40]
                print(f"     {i}. {desc} (conf: {conf:.2%})")

        validation = self.results.get('validation', {})
        overall = validation.get('overall_metrics', {})
        phi_align = validation.get('phi_alignment')
        print(f"\n✓ Validation:")
        print(f"   Validated rules: {overall.get('rules_above_threshold', 0)}")
        print(f"   Avg confidence: {overall.get('mean_accuracy', 0):.2%}")
        if phi_align is not None:
            print(f"   φ-alignment: {phi_align:.4f}")
        else:
            print(f"   φ-alignment: N/A")


if __name__ == "__main__":
    import argparse

    epilog = """
IMPORTANT FOR REPRODUCIBILITY:
  Output filename includes pattern length parameters:
    level1_analysis_var{X}_iter{N}_min{MIN}_max{MAX}.json

  When comparing iterations for geometric analysis (Γ, 𝓡, 𝓣),
  use CONSISTENT --min-len and --max-len across ALL iterations.

EXAMPLES:
  # Analyze iteration 19 with recommended parameters:
  python agents/level1_orchestrator.py -v B -i 19 --min-len 10 --max-len 50 --report

  # Analyze 3 consecutive iterations for geometric analysis:
  python agents/level1_orchestrator.py -v B -i 17 --min-len 10 --max-len 50 --no-cache
  python agents/level1_orchestrator.py -v B -i 18 --min-len 10 --max-len 50 --no-cache
  python agents/level1_orchestrator.py -v B -i 19 --min-len 10 --max-len 50 --no-cache
"""

    parser = argparse.ArgumentParser(
        description="HSI Level 1 Analysis: Pattern detection and rule inference",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant to analyze (default: B)")
    parser.add_argument("--iteration", "-i", type=int, help="Iteration number (default: latest)")
    parser.add_argument("--max-chars", "-m", type=int, default=0,
                       help="Max chars to analyze (0 = no limit, analyze full file)")
    parser.add_argument("--min-len", type=int, default=0,
                       help="Min pattern length (0 = use config default of 4)")
    parser.add_argument("--max-len", type=int, default=0,
                       help="Max pattern length (0 = use config default of 32)")
    parser.add_argument("--output", "-o", help="Output JSON file path (auto-generated if not specified)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable cache (force recalculation of all phases)")
    parser.add_argument("--no-spectral", action="store_true",
                       help="Skip spectral (FFT) pattern analysis (faster, useful when spectral yields no patterns)")
    parser.add_argument("--report", "-r", action="store_true",
                       help="Generate Markdown analysis report after execution")

    args = parser.parse_args()

    orchestrator = Level1Orchestrator()

    # Override pattern length config if specified
    if args.min_len > 0:
        orchestrator.config['pattern_detector']['min_pattern_length'] = args.min_len
        print(f"📏 Using min_pattern_length = {args.min_len}")
    if args.max_len > 0:
        orchestrator.config['pattern_detector']['max_pattern_length'] = args.max_len
        print(f"📏 Using max_pattern_length = {args.max_len}")

    # Handle --no-spectral flag
    enable_spectral = not args.no_spectral
    if args.no_spectral:
        print(f"⏭️  Spectral analysis: DISABLED (--no-spectral)")

    # Re-initialize pattern detector with all config options
    pd_config = orchestrator.config.get('pattern_detector', {})
    ri_config = orchestrator.config.get('rule_inferer', {})

    orchestrator.pattern_detector = PatternDetector(
        min_pattern_length=pd_config.get('min_pattern_length', 3),
        max_pattern_length=pd_config.get('max_pattern_length', 50),
        min_occurrences=pd_config.get('min_occurrences', 2),
        similarity_threshold=pd_config.get('similarity_threshold', 0.8),
        enable_spectral_analysis=enable_spectral
    )

    # Re-initialize rule inferer if pattern lengths changed
    if args.min_len > 0 or args.max_len > 0:
        min_ctx = pd_config.get('min_pattern_length', 3)
        orchestrator.rule_inferer = RuleInferer(
            context_window=ri_config.get('context_window', 5),
            min_context_length=min_ctx,
            min_rule_confidence=ri_config.get('min_confidence', 0.7),
            max_rule_complexity=ri_config.get('max_rule_complexity', 10)
        )
        print(f"📏 Using min_context_length = {min_ctx} for rule inference")

    # Find latest iteration if not specified
    iterations = orchestrator.get_available_iterations(args.variant)

    if not iterations:
        print(f"❌ No iterations found for variant {args.variant}")
        sys.exit(1)

    iteration = args.iteration or max(iterations)
    print(f"📋 Available iterations for variant {args.variant}: {iterations}")
    print(f"📌 Selected iteration: {iteration}")

    # Run analysis with adaptive sample sizing
    user_specified_max_chars = args.max_chars > 0

    if user_specified_max_chars:
        # User explicitly set max_chars - respect it but warn if insufficient
        max_chars = args.max_chars
        print(f"📊 Using user-specified max_chars: {max_chars:,}")
    else:
        # Auto-calculate optimal sample for max_pattern_length
        # First, we need to know the sequence length - get from latest phi file
        phi_path = orchestrator._find_phi_file(args.variant, iteration)
        if phi_path:
            from utils.streaming_phi_loader import StreamingPhiLoader
            loader = StreamingPhiLoader(phi_path)
            metadata = loader.metadata  # Property that loads from .json file
            full_seq_length = metadata.get('sequence_length', 10_000_000_000)

            pd_config = orchestrator.config.get('pattern_detector', {})
            max_pattern_len = pd_config.get('max_pattern_length', 32)
            min_occurrences = pd_config.get('min_occurrences', 3)

            # Calculate optimal sample
            sampling_info = orchestrator.calculate_optimal_sample_size(
                full_seq_length, max_pattern_len, min_occurrences
            )

            optimal_sample = sampling_info['optimal_sample']
            min_sample = sampling_info['minimum_sample']

            # Use optimal sample, but cap at full sequence and reasonable RAM limit
            max_ram_gb = 64  # Reasonable default, could be made configurable
            max_bits_for_ram = int(max_ram_gb * (1024**3) / 2)  # 2 bytes per bit

            # Choose the most practical sample size
            max_chars = min(optimal_sample, full_seq_length, max_bits_for_ram)
            max_chars = max(max_chars, min_sample)  # At least minimum coverage

            print(f"\n🔧 AUTO-ADAPTIVE SAMPLING")
            print(f"   Sequence length: {full_seq_length:,} bits")
            print(f"   Max pattern length: {max_pattern_len} bits")
            print(f"   Optimal sample: {optimal_sample:,} bits")
            if optimal_sample > full_seq_length:
                print(f"   └─ Capped to sequence length: {full_seq_length:,}")
            if optimal_sample > max_bits_for_ram:
                print(f"   └─ Capped to RAM limit ({max_ram_gb}GB): {max_bits_for_ram:,}")
            print(f"   Selected sample: {max_chars:,} bits ({max_chars/1e9:.1f}G)")
        else:
            max_chars = 10_000_000_000
            print(f"⚠️  Could not determine sequence length, using default: {max_chars:,}")

    orchestrator.config['max_chars_in_memory'] = max_chars
    orchestrator.config['user_specified_max_chars'] = user_specified_max_chars
    use_cache = not args.no_cache
    results = orchestrator.analyze_variant(args.variant, iteration, use_cache=use_cache)

    # Print summary
    orchestrator.print_summary()

    # Save results using unified structure
    if args.output:
        output_path = args.output
    else:
        # Build filename with pattern limits for reproducibility
        pd_config = orchestrator.config.get('pattern_detector', {})
        min_len = pd_config.get('min_pattern_length', 4)
        max_len = pd_config.get('max_pattern_length', 32)
        from utils.file_saver import get_output_path
        filename = f"var_{args.variant}_iter{iteration}_min{min_len}_max{max_len}.json"
        output_path = str(get_output_path(1, "analysis", filename))

    orchestrator.save_results(output_path)

    # Generate analysis report if requested
    if args.report:
        try:
            # Import here to avoid circular dependency
            from level1_view_results import ResultsReporter
            reporter = ResultsReporter(output_path)
            md_path = reporter.save_markdown_report()
            print(f"\n📝 Analysis report saved to: {md_path}")
        except ImportError:
            print("\n⚠️ Could not import ResultsReporter. Run level1_view_results.py manually.")
        except Exception as e:
            print(f"\n⚠️ Error generating report: {e}")

    # Reminder about analysis tool
    print(f"\n💡 Tip: Analyze these results with:")
    print(f"   python level1_view_results.py {output_path}")

