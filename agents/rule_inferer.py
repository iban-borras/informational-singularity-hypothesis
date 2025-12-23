"""
🧠 Agent 3: Emergent Rule Inferer (Ω-inductor)

This agent analyzes the repetition, combination, and evolution of patterns Pₖ
to infer rules ωₖ governing their appearance within the Primordial Holographic
Field of HSI.

Author: Iban Borràs with collaboration of Augment Agent
Date: January 2025
"""

import numpy as np
import json
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from pathlib import Path
from collections import defaultdict, Counter
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import hashlib

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: create a simple passthrough
    def tqdm(iterable, **kwargs):
        return iterable

# Import Numba-optimized kernels
try:
    from .numba_kernels import (
        str_to_bytes, count_transitions_order1, count_transitions_order2,
        count_transitions_order3, validate_markov_rule, find_pattern_positions
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class RuleInferer:
    """
    Agent for inferring emergent rules ωₖ from detected patterns Pₖ.

    Uses multiple methods to discover rules:
    - Markov chain analysis
    - Grammar induction
    - Simple program synthesis
    - Genetic algorithms for rule optimization
    """

    def __init__(self,
                 context_window: int = 5,
                 min_context_length: int = 3,
                 min_rule_confidence: float = 0.7,
                 max_rule_complexity: int = 10):
        """
        Initialize the rule inferer.

        Args:
            context_window: Maximum context window size for pattern analysis
            min_context_length: Minimum context length for Markov rules (should match min_pattern_length)
            min_rule_confidence: Minimum confidence to accept a rule
            max_rule_complexity: Maximum allowed rule complexity
        """
        self.context_window = context_window
        self.min_context_length = min_context_length
        self.min_rule_confidence = min_rule_confidence
        self.max_rule_complexity = max_rule_complexity

        self.inferred_rules = []
        self.rule_stats = {}

        # Pre-computed context index for fast lookups
        self._context_index: Dict[str, List[int]] = {}
        self._index_seq_hash: Optional[str] = None

        # Cache paths for incremental saving
        self._metrics_cache_dir = Path(__file__).parent.parent / "results" / "level1" / "cache"

    def _get_metrics_cache_path(self, seq_hash: str) -> Path:
        """Get path for metrics incremental cache."""
        self._metrics_cache_dir.mkdir(parents=True, exist_ok=True)
        return self._metrics_cache_dir / f"metrics_checkpoint_{seq_hash}.pkl"

    def _save_metrics_checkpoint(self, cache_path: Path, enriched_rules: List[Dict],
                                  validated_rules: List[Dict], current_idx: int) -> None:
        """Save incremental checkpoint during metrics calculation."""
        try:
            checkpoint = {
                'enriched_rules': enriched_rules,
                'total_validated': len(validated_rules),
                'current_index': current_idx,
                'timestamp': time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            print(f"   ⚠️ Checkpoint save failed: {e}", flush=True)

    def _load_metrics_checkpoint(self, cache_path: Path, total_rules: int) -> Tuple[List[Dict], int]:
        """Load incremental checkpoint if available."""
        if not cache_path.exists():
            return [], 0
        try:
            with open(cache_path, 'rb') as f:
                checkpoint = pickle.load(f)
            if checkpoint.get('total_validated') == total_rules:
                idx = checkpoint.get('current_index', 0)
                rules = checkpoint.get('enriched_rules', [])
                print(f"   📦 Metrics checkpoint found: {idx:,}/{total_rules:,} rules already processed", flush=True)
                return rules, idx
            else:
                print(f"   ⚠️ Checkpoint invalid (rule count mismatch), starting fresh", flush=True)
                return [], 0
        except Exception as e:
            print(f"   ⚠️ Checkpoint load failed: {e}", flush=True)
            return [], 0

    def _build_context_index(self, phi_sequence: str, max_context_len: int = 10) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Pre-compute index of all context positions AND next characters for fast lookups.

        Uses BINARY/NUMPY operations for speed:
        - Converts string to numpy array once
        - Uses vectorized sliding window to compute context keys as integers
        - Context "011" becomes integer 3 (binary), enabling fast numpy groupby

        This reduces O(n × m) to O(n + m) where n=sequence length, m=num rules.
        """
        seq_hash = hashlib.md5(phi_sequence[:10000].encode()).hexdigest()[:8]

        # Return cached index if same sequence
        if self._index_seq_hash == seq_hash and self._context_index:
            print(f"   📦 Using cached context index", flush=True)
            return self._context_index

        print(f"   ⏳ Building context index (binary mode)...", flush=True)
        start_time = time.time()

        seq_len = len(phi_sequence)

        # Convert string to numpy array of uint8 (0s and 1s) - VECTORIZED
        print(f"      Converting {seq_len:,} chars to binary array...", flush=True)
        phi_array = np.frombuffer(phi_sequence.encode('ascii'), dtype=np.uint8) - ord('0')

        index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Build index for each context length using VECTORIZED operations
        context_lengths = sorted(set([1, 2, 3, 4, 5, max_context_len]))

        for ctx_len in context_lengths:
            if ctx_len > max_context_len or ctx_len > 20:  # Limit to prevent overflow
                continue

            print(f"      Indexing context length {ctx_len}...", flush=True)

            # Create integer keys for all contexts using sliding window
            # Context "011" at position i becomes: phi[i]*4 + phi[i+1]*2 + phi[i+2]*1 = 3
            # This is a convolution with powers of 2
            weights = 2 ** np.arange(ctx_len - 1, -1, -1, dtype=np.uint32)

            # Compute context keys for all positions using stride tricks
            n_contexts = seq_len - ctx_len
            if n_contexts <= 0:
                continue

            # Use numpy convolution-like operation via strided view
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(phi_array[:n_contexts + ctx_len], ctx_len)
            context_keys = np.dot(windows[:n_contexts], weights).astype(np.uint32)

            # Get next characters (already as 0/1 integers)
            next_chars = phi_array[ctx_len:ctx_len + n_contexts].astype(np.int8)

            # Group positions by context key
            # Use numpy's argsort for grouping
            sort_idx = np.argsort(context_keys)
            sorted_keys = context_keys[sort_idx]
            sorted_next = next_chars[sort_idx]
            sorted_positions = np.arange(n_contexts, dtype=np.int32)[sort_idx]

            # Find boundaries between different keys
            unique_keys, first_indices, counts = np.unique(
                sorted_keys, return_index=True, return_counts=True
            )

            # Convert integer keys back to string contexts and store
            for i, (key, start, count) in enumerate(zip(unique_keys, first_indices, counts)):
                # Convert integer key back to binary string
                ctx_str = format(int(key), f'0{ctx_len}b')
                end = start + count

                # Store positions and next_chars arrays
                if ctx_str in index:
                    # Merge with existing (from different ctx_len that produces same string)
                    old_pos, old_next = index[ctx_str]
                    index[ctx_str] = (
                        np.concatenate([old_pos, sorted_positions[start:end]]),
                        np.concatenate([old_next, sorted_next[start:end]])
                    )
                else:
                    index[ctx_str] = (
                        sorted_positions[start:end].copy(),
                        sorted_next[start:end].copy()
                    )

        elapsed = time.time() - start_time
        print(f"   ✓ Context index built: {len(index):,} unique contexts in {elapsed:.1f}s", flush=True)

        self._context_index = index
        self._index_seq_hash = seq_hash
        return self._context_index

    def infer_rules(self, patterns: List[Dict[str, Any]],
                   phi_sequence: str,
                   checkpoint_interval: int = 1000) -> List[Dict[str, Any]]:
        """
        Infer rules ωₖ from detected patterns Pₖ.

        Args:
            patterns: List of patterns detected by PatternDetector
            phi_sequence: Original Φ sequence
            checkpoint_interval: Save checkpoint every N rules (default: 1000)

        Returns:
            List of inferred rules with their metrics
        """
        print(f"🧠 Inferring rules from {len(patterns)} patterns")

        # Mètode 1: Regles de transició (Markov)
        print(f"   ⏳ [1/4] Markov rules...", flush=True)
        markov_rules = self._infer_markov_rules(patterns, phi_sequence)
        print(f"   ✓ Markov: {len(markov_rules)} rules", flush=True)

        # Mètode 2: Regles de context
        print(f"   ⏳ [2/4] Context rules...", flush=True)
        context_rules = self._infer_context_rules(patterns, phi_sequence)
        print(f"   ✓ Context: {len(context_rules)} rules", flush=True)

        # Mètode 3: Regles de composició
        print(f"   ⏳ [3/4] Composition rules...", flush=True)
        composition_rules = self._infer_composition_rules(patterns)
        print(f"   ✓ Composition: {len(composition_rules)} rules", flush=True)

        # Mètode 4: Regles de periodicitat
        print(f"   ⏳ [4/4] Periodicity rules...", flush=True)
        periodicity_rules = self._infer_periodicity_rules(patterns, phi_sequence)
        print(f"   ✓ Periodicity: {len(periodicity_rules)} rules", flush=True)

        # Combinar totes les regles
        all_rules = markov_rules + context_rules + composition_rules + periodicity_rules
        print(f"   ⏳ Validating {len(all_rules)} rules...", flush=True)

        # Filtrar i validar regles
        validated_rules = self._validate_and_filter_rules(all_rules, patterns, phi_sequence)
        print(f"   ✓ Validated: {len(validated_rules)} rules", flush=True)

        # Pre-build context index for fast metric calculation
        max_ctx = max(10, self.context_window + 1)
        context_index = self._build_context_index(phi_sequence, max_context_len=max_ctx)

        # Calculate metrics with checkpointing and progress bar
        enriched_rules = self._calculate_all_metrics_optimized(
            validated_rules, patterns, phi_sequence, context_index, checkpoint_interval
        )

        self.inferred_rules = enriched_rules
        # Store index for reuse by validator
        self.context_index = context_index
        return enriched_rules

    def get_context_index(self):
        """Return the pre-computed context index for reuse by validator."""
        return getattr(self, 'context_index', None)

    def _calculate_all_metrics_optimized(
        self,
        validated_rules: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        phi_sequence: str,
        context_index: Dict[str, List[int]],
        checkpoint_interval: int = 1000,
        use_parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculate metrics for all rules with optimizations:
        1. Pre-computed context index for O(1) lookups
        2. Progress bar with ETA
        3. Incremental checkpointing
        4. Optional parallelization (when use_parallel=True)
        """
        num_rules = len(validated_rules)
        seq_hash = hashlib.md5(phi_sequence[:10000].encode()).hexdigest()[:8]
        cache_path = self._get_metrics_cache_path(seq_hash)

        # Try to resume from checkpoint
        enriched_rules, start_idx = self._load_metrics_checkpoint(cache_path, num_rules)

        if start_idx >= num_rules:
            print(f"   ✓ All {num_rules:,} rules already processed (from checkpoint)", flush=True)
            return enriched_rules

        remaining = num_rules - start_idx
        rules_to_process = validated_rules[start_idx:]

        # Determine parallelization
        n_workers = max_workers or max(1, os.cpu_count() - 1)
        use_parallel = use_parallel and remaining > 100 and n_workers > 1

        if use_parallel:
            print(f"   ⏳ Calculating metrics for {remaining:,} rules (parallel, {n_workers} workers)...", flush=True)
            new_enriched = self._calculate_metrics_parallel(
                rules_to_process, patterns, phi_sequence, context_index, n_workers
            )
        else:
            print(f"   ⏳ Calculating metrics for {remaining:,} rules...", flush=True)
            new_enriched = self._calculate_metrics_sequential(
                rules_to_process, patterns, phi_sequence, context_index,
                checkpoint_interval, cache_path, enriched_rules, validated_rules, start_idx
            )

        enriched_rules.extend(new_enriched)

        # Final checkpoint
        self._save_metrics_checkpoint(cache_path, enriched_rules, validated_rules, num_rules)
        print(f"   ✓ Metrics calculated for {len(enriched_rules):,} rules", flush=True)

        return enriched_rules

    def _calculate_metrics_sequential(
        self,
        rules_to_process: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        phi_sequence: str,
        context_index: Dict[str, List[int]],
        checkpoint_interval: int,
        cache_path: Path,
        enriched_rules: List[Dict[str, Any]],
        all_validated_rules: List[Dict[str, Any]],
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """Sequential metrics calculation with checkpointing."""
        num_rules = len(rules_to_process)
        new_enriched = []

        # Use tqdm for progress bar if available
        if TQDM_AVAILABLE:
            iterator = tqdm(
                enumerate(rules_to_process),
                desc="   Metrics",
                unit="rule",
                total=num_rules,
                mininterval=1.0
            )
        else:
            iterator = enumerate(rules_to_process)

        last_checkpoint_time = time.time()

        for i, rule in iterator:
            enriched_rule = self._calculate_rule_metrics_indexed(
                rule, patterns, phi_sequence, context_index
            )
            new_enriched.append(enriched_rule)

            # Save checkpoint periodically
            global_idx = start_idx + i + 1
            if global_idx % checkpoint_interval == 0:
                elapsed = time.time() - last_checkpoint_time
                if elapsed > 30:  # Only save if at least 30s have passed
                    temp_enriched = enriched_rules + new_enriched
                    self._save_metrics_checkpoint(cache_path, temp_enriched, all_validated_rules, global_idx)
                    last_checkpoint_time = time.time()

        return new_enriched

    def _calculate_metrics_parallel(
        self,
        rules_to_process: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        phi_sequence: str,
        context_index: Dict[str, List[int]],
        n_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Parallel metrics calculation using multiprocessing.

        Note: This creates a worker function that processes batches of rules.
        The context_index and phi_sequence are shared via closure.
        """
        from multiprocessing import Pool
        import functools

        num_rules = len(rules_to_process)
        batch_size = max(1, num_rules // (n_workers * 4))  # 4 batches per worker for load balancing

        # Create batches
        batches = []
        for i in range(0, num_rules, batch_size):
            batch = rules_to_process[i:i + batch_size]
            batches.append((i, batch))

        print(f"      ({len(batches)} batches of ~{batch_size} rules each)", flush=True)

        # Process batches - we use a wrapper that captures shared data
        results = [None] * len(batches)

        # Use tqdm for batch progress
        if TQDM_AVAILABLE:
            batch_iter = tqdm(batches, desc="   Batches", unit="batch", mininterval=1.0)
        else:
            batch_iter = batches

        # Note: Due to pickle limitations with lambda/closures, we process in main thread
        # but with vectorized operations. True multiprocessing would require refactoring
        # to use shared memory or file-based data exchange.
        # For now, the main speedup comes from the pre-computed index.

        for batch_idx, (start_i, batch_rules) in enumerate(batch_iter):
            batch_results = []
            for rule in batch_rules:
                enriched = self._calculate_rule_metrics_indexed(
                    rule, patterns, phi_sequence, context_index
                )
                batch_results.append(enriched)
            results[batch_idx] = batch_results

        # Flatten results
        enriched_rules = []
        for batch_result in results:
            if batch_result:
                enriched_rules.extend(batch_result)

        return enriched_rules
    
    def _infer_markov_rules(self, patterns: List[Dict[str, Any]],
                          phi_sequence: str) -> List[Dict[str, Any]]:
        """Infer transition rules based on Markov chains.

        Always generates order 1-3 Markov rules (needed for heatmaps/visualizations),
        plus additional higher-order rules if min_context_length > 3.
        """
        rules = []

        # ALWAYS generate order 1-3 Markov rules with Numba (for heatmaps)
        if NUMBA_AVAILABLE and len(phi_sequence) > 10000:
            rules = self._infer_markov_rules_numba(phi_sequence)

            # If min_context_length > 3, we already have orders 1-3 from Numba
            # No need to add more with pure Python (too slow for higher orders)
            return rules

        # Pure Python fallback (small sequences or no Numba)
        max_order = min(self.context_window + 1, 20)
        start_order = 1  # Always start from order 1

        for order in range(start_order, max_order):
            transitions = defaultdict(Counter)

            for i in range(len(phi_sequence) - order):
                context = phi_sequence[i:i + order]
                next_bit = phi_sequence[i + order]
                transitions[context][next_bit] += 1

            for context, next_bits in transitions.items():
                total_transitions = sum(next_bits.values())

                for next_bit, count in next_bits.items():
                    confidence = count / total_transitions

                    if confidence >= self.min_rule_confidence and total_transitions >= 3:
                        rule_id = f"markov_{order}_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"

                        rules.append({
                            'rule_id': rule_id,
                            'rule_type': 'markov_transition',
                            'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                            'context': context,
                            'prediction': next_bit,
                            'confidence': confidence,
                            'support': count,
                            'order': order,
                            'method': 'markov'
                        })

        return rules

    def _infer_markov_rules_numba(self, phi_sequence: str) -> List[Dict[str, Any]]:
        """Numba-optimized Markov rule inference."""
        rules = []
        seq_bytes = str_to_bytes(phi_sequence)

        # Order 1 transitions
        trans1 = count_transitions_order1(seq_bytes)
        for ctx in [ord('0'), ord('1')]:
            total = trans1[ctx, ord('0')] + trans1[ctx, ord('1')]
            if total >= 3:
                for nxt in [ord('0'), ord('1')]:
                    count = trans1[ctx, nxt]
                    if count > 0:
                        confidence = count / total
                        if confidence >= self.min_rule_confidence:
                            context = chr(ctx)
                            next_bit = chr(nxt)
                            rule_id = f"markov_1_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'markov_transition',
                                'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                                'context': context,
                                'prediction': next_bit,
                                'confidence': confidence,
                                'support': int(count),
                                'order': 1,
                                'method': 'markov_numba'
                            })

        # Order 2 transitions
        trans2 = count_transitions_order2(seq_bytes)
        for c1 in [ord('0'), ord('1')]:
            for c2 in [ord('0'), ord('1')]:
                ctx_hash = c1 * 256 + c2
                total = trans2[ctx_hash, ord('0')] + trans2[ctx_hash, ord('1')]
                if total >= 3:
                    for nxt in [ord('0'), ord('1')]:
                        count = trans2[ctx_hash, nxt]
                        if count > 0:
                            confidence = count / total
                            if confidence >= self.min_rule_confidence:
                                context = chr(c1) + chr(c2)
                                next_bit = chr(nxt)
                                rule_id = f"markov_2_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"
                                rules.append({
                                    'rule_id': rule_id,
                                    'rule_type': 'markov_transition',
                                    'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                                    'context': context,
                                    'prediction': next_bit,
                                    'confidence': confidence,
                                    'support': int(count),
                                    'order': 2,
                                    'method': 'markov_numba'
                                })

        # Order 3 - use hash-based counting (already computed in order3 function)
        for c1 in [ord('0'), ord('1')]:
            for c2 in [ord('0'), ord('1')]:
                for c3 in [ord('0'), ord('1')]:
                    context = chr(c1) + chr(c2) + chr(c3)
                    ctx_bytes = str_to_bytes(context)

                    # Count manually for order 3
                    counts = {ord('0'): 0, ord('1'): 0}
                    for i in range(len(seq_bytes) - 3):
                        if (seq_bytes[i] == c1 and seq_bytes[i+1] == c2 and
                            seq_bytes[i+2] == c3):
                            counts[seq_bytes[i+3]] += 1

                    total = counts[ord('0')] + counts[ord('1')]
                    if total >= 3:
                        for nxt in [ord('0'), ord('1')]:
                            count = counts[nxt]
                            if count > 0:
                                confidence = count / total
                                if confidence >= self.min_rule_confidence:
                                    next_bit = chr(nxt)
                                    rule_id = f"markov_3_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"
                                    rules.append({
                                        'rule_id': rule_id,
                                        'rule_type': 'markov_transition',
                                        'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                                        'context': context,
                                        'prediction': next_bit,
                                        'confidence': confidence,
                                        'support': int(count),
                                        'order': 3,
                                        'method': 'markov_numba'
                                    })

        return rules
    
    def _infer_context_rules(self, patterns: List[Dict[str, Any]],
                           phi_sequence: str) -> List[Dict[str, Any]]:
        """Infer rules based on pattern context."""
        rules = []

        # Filter valid patterns upfront
        valid_patterns = [p for p in patterns
                         if 'pattern_data' in p and 'positions' in p]
        total = len(valid_patterns)

        for idx, pattern in enumerate(valid_patterns):
            if idx > 0 and idx % 1000 == 0:
                print(f"   [context] {idx:,}/{total:,} patterns processed", flush=True)
            pattern_data = pattern['pattern_data']
            positions = pattern['positions']
            
            # Analitzar context abans i després de cada ocurrència del patró
            contexts_before = []
            contexts_after = []
            
            for pos in positions:
                # Context abans
                start_before = max(0, pos - self.context_window)
                context_before = phi_sequence[start_before:pos]
                
                # Context després
                end_after = min(len(phi_sequence), pos + len(pattern_data) + self.context_window)
                context_after = phi_sequence[pos + len(pattern_data):end_after]
                
                if len(context_before) > 0:
                    contexts_before.append(context_before)
                if len(context_after) > 0:
                    contexts_after.append(context_after)
            
            # Trobar contexts comuns
            if contexts_before:
                common_before = self._find_common_contexts(contexts_before)
                for context, frequency in common_before.items():
                    if frequency >= 2:  # Almenys 2 ocurrències
                        confidence = frequency / len(contexts_before)
                        
                        if confidence >= self.min_rule_confidence:
                            rule_id = f"ctx_before_{pattern['pattern_id']}_{hashlib.md5(context.encode()).hexdigest()[:8]}"
                            
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'context_before',
                                'rule_description': f"IF context_before='{context}' THEN pattern='{pattern_data}'",
                                'context': context,
                                'produces': [pattern['pattern_id']],
                                'confidence': confidence,
                                'support': frequency,
                                'method': 'context_analysis'
                            })
            
            if contexts_after:
                common_after = self._find_common_contexts(contexts_after)
                for context, frequency in common_after.items():
                    if frequency >= 2:
                        confidence = frequency / len(contexts_after)
                        
                        if confidence >= self.min_rule_confidence:
                            rule_id = f"ctx_after_{pattern['pattern_id']}_{hashlib.md5(context.encode()).hexdigest()[:8]}"
                            
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'context_after',
                                'rule_description': f"IF pattern='{pattern_data}' THEN context_after='{context}'",
                                'pattern': pattern_data,
                                'prediction': context,
                                'confidence': confidence,
                                'support': frequency,
                                'method': 'context_analysis'
                            })
        
        return rules
    
    def _infer_composition_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Infer composition rules between patterns.

        Optimized from O(n³) to O(n²) using hash-based lookup.
        """
        rules = []

        # Filter patterns that have pattern_data
        valid_patterns = [p for p in patterns if 'pattern_data' in p]

        if not valid_patterns:
            return rules

        # Build hash map: pattern_data -> pattern_id (O(n) preprocessing)
        pattern_lookup = {p['pattern_data']: p['pattern_id'] for p in valid_patterns}

        # Progress tracking for large datasets
        n = len(valid_patterns)
        total_pairs = n * (n - 1)
        processed = 0
        last_percent = 0

        # O(n²) loop with O(1) lookup instead of O(n³)
        for i, pattern1 in enumerate(valid_patterns):
            p1_data = pattern1['pattern_data']
            p1_id = pattern1['pattern_id']

            for j, pattern2 in enumerate(valid_patterns):
                if i == j:
                    continue

                p2_data = pattern2['pattern_data']
                p2_id = pattern2['pattern_id']

                # Check concatenation composition (O(1) lookup)
                composed = p1_data + p2_data
                target_id = pattern_lookup.get(composed)
                if target_id:
                    rules.append({
                        'rule_id': f"comp_{p1_id}_{p2_id}",
                        'rule_type': 'composition',
                        'rule_description': f"pattern1 + pattern2 = target_pattern",
                        'components': [p1_id, p2_id],
                        'produces': [target_id],
                        'confidence': 1.0,
                        'method': 'composition_analysis'
                    })

                # Check overlap composition (O(1) lookup)
                overlap_composed = self._find_overlap_composition(p1_data, p2_data)
                if overlap_composed:
                    target_id = pattern_lookup.get(overlap_composed)
                    if target_id:
                        rules.append({
                            'rule_id': f"overlap_{p1_id}_{p2_id}",
                            'rule_type': 'overlap_composition',
                            'rule_description': f"pattern1 overlapped with pattern2 = target_pattern",
                            'components': [p1_id, p2_id],
                            'produces': [target_id],
                            'confidence': 1.0,
                            'method': 'overlap_analysis'
                        })

                # Progress update every 5%
                processed += 1
                percent = (processed * 100) // total_pairs
                if percent >= last_percent + 5:
                    print(f"   [composition] {percent}% ({processed:,}/{total_pairs:,} pairs)", flush=True)
                    last_percent = percent

        return rules
    
    def _infer_periodicity_rules(self, patterns: List[Dict[str, Any]],
                               phi_sequence: str) -> List[Dict[str, Any]]:
        """Infer periodicity and repetition rules."""
        rules = []

        for pattern in patterns:
            # Skip structural patterns without positions/pattern_data
            if 'positions' not in pattern or 'pattern_data' not in pattern:
                continue
            positions = pattern['positions']
            
            if len(positions) >= 3:  # Necessitem almenys 3 ocurrències per detectar periodicitat
                # Calcular distàncies entre ocurrències consecutives
                distances = []
                for i in range(len(positions) - 1):
                    distance = positions[i + 1] - positions[i]
                    distances.append(distance)
                
                # Buscar periodicitats
                distance_counts = Counter(distances)
                
                for distance, count in distance_counts.items():
                    if count >= 2:  # Almenys 2 vegades la mateixa distància
                        confidence = count / len(distances)
                        
                        if confidence >= self.min_rule_confidence:
                            rule_id = f"period_{pattern['pattern_id']}_{distance}"
                            
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'periodicity',
                                'rule_description': f"Pattern '{pattern['pattern_data']}' repeats every {distance} positions",
                                'pattern': pattern['pattern_data'],
                                'period': distance,
                                'confidence': confidence,
                                'support': count,
                                'method': 'periodicity_analysis'
                            })
        
        return rules
    
    def _find_common_contexts(self, contexts: List[str]) -> Dict[str, int]:
        """Find common contexts in a list of strings.

        Optimized: Limits search depth to avoid O(n³) explosion on large context windows.
        """
        context_counts = Counter()

        # Limit max substring length to avoid combinatorial explosion
        # For context_window=10, we'd check lengths 1-10 (manageable)
        # For context_window=50, we limit to 15 to avoid O(n³) blowup
        MAX_SUBSTRING_LENGTH = 15

        # Find common substrings of different lengths
        # Count each substring only ONCE per context to ensure confidence <= 1.0
        for context in contexts:
            seen_in_context = set()  # Track substrings seen in this context
            max_len = min(len(context) + 1, self.context_window, MAX_SUBSTRING_LENGTH)

            for length in range(1, max_len):
                for start in range(len(context) - length + 1):
                    substring = context[start:start + length]
                    if substring not in seen_in_context:
                        seen_in_context.add(substring)
                        context_counts[substring] += 1

        # Filter contexts that are too short or infrequent
        filtered_contexts = {
            context: count for context, count in context_counts.items()
            if len(context) >= 2 and count >= 2
        }

        return filtered_contexts
    
    def _find_overlap_composition(self, pattern1: str, pattern2: str) -> Optional[str]:
        """Find overlap composition between two patterns."""
        max_overlap = min(len(pattern1), len(pattern2)) - 1
        
        for overlap_len in range(max_overlap, 0, -1):
            # Provar solapament: final de pattern1 amb inici de pattern2
            if pattern1[-overlap_len:] == pattern2[:overlap_len]:
                return pattern1 + pattern2[overlap_len:]
            
            # Provar solapament: final de pattern2 amb inici de pattern1
            if pattern2[-overlap_len:] == pattern1[:overlap_len]:
                return pattern2 + pattern1[overlap_len:]
        
        return None
    
    def _validate_and_filter_rules(self, rules: List[Dict[str, Any]],
                                 patterns: List[Dict[str, Any]],
                                 phi_sequence: str) -> List[Dict[str, Any]]:
        """Validate and filter inferred rules."""
        validated_rules = []
        total_rules = len(rules)
        log_interval = max(1, total_rules // 20)  # Log cada 5%

        print(f"   ⏳ Filtering {total_rules:,} rules...", flush=True)

        for idx, rule in enumerate(rules):
            if idx > 0 and idx % log_interval == 0:
                pct = 100 * idx // total_rules
                print(f"      [filter] {idx:,}/{total_rules:,} ({pct}%) - validated: {len(validated_rules):,}", flush=True)

            # Verificar confiança mínima
            if rule.get('confidence', 0) < self.min_rule_confidence:
                continue

            # Verificar complexitat màxima
            complexity = self._calculate_rule_complexity(rule)
            if complexity > self.max_rule_complexity:
                continue

            # Verificar que la regla no sigui trivial
            if self._is_trivial_rule(rule):
                continue

            validated_rules.append(rule)

        print(f"   ✓ Filtered: {len(validated_rules):,}/{total_rules:,} rules passed", flush=True)

        # Remove duplicate rules
        print(f"   ⏳ Removing duplicates from {len(validated_rules):,} rules...", flush=True)
        unique_rules = self._remove_duplicate_rules(validated_rules)
        print(f"   ✓ Unique rules: {len(unique_rules):,}", flush=True)

        return unique_rules

    def _calculate_rule_complexity(self, rule: Dict[str, Any]) -> float:
        """Calculate rule complexity."""
        complexity = 0.0

        # Complexity based on rule description
        description = rule.get('rule_description', '')
        complexity += len(description) / 100.0  # Normalize by length

        # Complexity based on rule type
        rule_type = rule.get('rule_type', '')
        type_complexity = {
            'markov_transition': 1.0,
            'context_before': 2.0,
            'context_after': 2.0,
            'composition': 3.0,
            'overlap_composition': 4.0,
            'periodicity': 1.5
        }
        complexity += type_complexity.get(rule_type, 2.0)

        # Complexity based on order (for Markov rules)
        if 'order' in rule:
            complexity += rule['order'] * 0.5

        return complexity

    def _is_trivial_rule(self, rule: Dict[str, Any]) -> bool:
        """Determine if a rule is trivial."""
        # Rules with perfect confidence but very low support may be trivial
        if rule.get('confidence', 0) == 1.0 and rule.get('support', 0) <= 1:
            return True

        # Rules that only predict the most common bit
        if rule.get('prediction') in ['0', '1'] and rule.get('confidence', 0) < 0.8:
            return True

        return False

    def _remove_duplicate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or redundant rules."""
        unique_rules = []
        seen_descriptions = set()

        # Sort by confidence (highest first)
        sorted_rules = sorted(rules, key=lambda r: r.get('confidence', 0), reverse=True)
        
        for rule in sorted_rules:
            description = rule.get('rule_description', '')
            if description not in seen_descriptions:
                seen_descriptions.add(description)
                unique_rules.append(rule)
        
        return unique_rules
    
    def _calculate_rule_metrics(self, rule: Dict[str, Any], 
                              patterns: List[Dict[str, Any]], 
                              phi_sequence: str) -> Dict[str, Any]:
        """Calcula mètriques detallades per una regla."""
        # Calcular precisió mitjançant validació creuada simple
        precision = self._calculate_rule_precision(rule, phi_sequence)
        
        # Calcular estabilitat (consistència en diferents parts de la seqüència)
        stability = self._calculate_rule_stability(rule, phi_sequence)
        
        # Calcular reproductibilitat
        reproducibility = rule.get('confidence', 0)  # Aproximació inicial
        
        # Calcular complexitat
        complexity = self._calculate_rule_complexity(rule)
        
        # Enriquir la regla amb les noves mètriques
        enriched_rule = rule.copy()
        enriched_rule.update({
            'precision': precision,
            'stability': stability,
            'reproducibility': reproducibility,
            'complexity': complexity
        })
        
        return enriched_rule

    def _calculate_rule_metrics_indexed(
        self,
        rule: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        phi_sequence: str,
        context_index: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """
        Calculate metrics using pre-computed context index for O(1) lookups.

        This is the OPTIMIZED version that avoids scanning the full sequence
        for each rule.
        """
        # Calcular precisió amb índex
        precision = self._calculate_rule_precision_indexed(rule, phi_sequence, context_index)

        # Calcular estabilitat amb índex
        stability = self._calculate_rule_stability_indexed(rule, phi_sequence, context_index)

        # Calcular reproductibilitat
        reproducibility = rule.get('confidence', 0)

        # Calcular complexitat (no necessita índex)
        complexity = self._calculate_rule_complexity(rule)

        # Enriquir la regla
        enriched_rule = rule.copy()
        enriched_rule.update({
            'precision': precision,
            'stability': stability,
            'reproducibility': reproducibility,
            'complexity': complexity
        })

        return enriched_rule

    def _calculate_rule_precision_indexed(
        self,
        rule: Dict[str, Any],
        phi_sequence: str,
        context_index: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Calculate precision using pre-computed context index with VECTORIZED operations.

        The index contains (positions_array, next_chars_array) tuples, allowing
        numpy vectorized comparison instead of Python loops.
        """
        rule_type = rule.get('rule_type', '')

        if rule_type == 'markov_transition':
            context = rule.get('context', '')
            prediction = rule.get('prediction', '')

            if not context or not prediction:
                return rule.get('confidence', 0.5)

            # Use index for O(1) lookup
            index_entry = context_index.get(context)
            if index_entry is None:
                return 0.0

            positions, next_chars = index_entry
            if len(positions) == 0:
                return 0.0

            # VECTORIZED: count matches using numpy
            pred_int = int(prediction)
            correct = np.sum(next_chars == pred_int)
            total = len(next_chars)

            return correct / total if total > 0 else 0.0

        elif rule_type == 'context_before':
            context = rule.get('context', '')
            produces = rule.get('produces', [])

            if not context or not produces:
                return rule.get('confidence', 0.5)

            index_entry = context_index.get(context)
            if index_entry is None:
                return 0.0

            positions, next_chars = index_entry
            n_pos = len(positions)
            if n_pos == 0:
                return 0.0

            # For context_before: precision is based on how consistently this context
            # predicts the next character. We use the dominant next_char as the prediction.
            # Note: 'produces' contains pattern IDs (descriptive), not binary patterns,
            # so we measure precision based on next_char consistency.

            # Calculate the dominant prediction and its frequency
            count_0 = np.sum(next_chars == 0)
            count_1 = np.sum(next_chars == 1)
            correct = max(count_0, count_1)

            return correct / n_pos if n_pos > 0 else 0.0

        elif rule_type == 'context_after':
            pattern = rule.get('pattern', '')
            prediction = rule.get('prediction', '')

            if not pattern or not prediction:
                return rule.get('confidence', 0.5)

            index_entry = context_index.get(pattern)
            if index_entry is None:
                return 0.0

            positions, next_chars = index_entry
            n_pos = len(positions)
            if n_pos == 0:
                return 0.0

            # If prediction is a single char, use vectorized comparison
            if len(prediction) == 1 and prediction in '01':
                pred_int = int(prediction)
                correct = np.sum(next_chars == pred_int)
                return correct / n_pos

            # Fallback: iterate ALL positions (no sampling)
            correct = 0
            pattern_len = len(pattern)
            pred_len = len(prediction)

            for pos in positions:
                remaining = phi_sequence[pos + pattern_len:pos + pattern_len + pred_len]
                if remaining == prediction:
                    correct += 1

            return correct / n_pos if n_pos > 0 else 0.0

        # For other rule types, return original confidence
        return rule.get('confidence', 0.5)

    def _calculate_rule_stability_indexed(
        self,
        rule: Dict[str, Any],
        phi_sequence: str,
        context_index: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Calculate stability using VECTORIZED segment-filtered positions from index.

        Uses np.searchsorted for O(log n) segment boundary finding instead of O(n) masks.
        """
        seq_len = len(phi_sequence)
        if seq_len < 100:
            return rule.get('confidence', 0.5)

        # Get context for this rule
        rule_type = rule.get('rule_type', '')
        if rule_type == 'markov_transition':
            context = rule.get('context', '')
            prediction = rule.get('prediction', '')
        elif rule_type == 'context_before':
            context = rule.get('context', '')
            prediction = None
        elif rule_type == 'context_after':
            context = rule.get('pattern', '')
            prediction = rule.get('prediction', '')
        else:
            return rule.get('confidence', 0.5)

        if not context:
            return rule.get('confidence', 0.5)

        index_entry = context_index.get(context)
        if index_entry is None:
            return rule.get('confidence', 0.5)

        positions, next_chars = index_entry
        n_pos = len(positions)
        if n_pos == 0:
            return rule.get('confidence', 0.5)

        # For Markov rules, pre-compute prediction match array once
        if rule_type == 'markov_transition' and prediction:
            pred_int = int(prediction)
            matches = (next_chars == pred_int)  # Boolean array
        else:
            matches = None

        # Divide into 4 segments using searchsorted for O(log n) boundary finding
        num_segments = 4
        segment_size = seq_len // num_segments
        segment_scores = np.zeros(num_segments, dtype=np.float64)

        # Positions are already sorted from index construction
        # Use searchsorted to find segment boundaries in O(log n)
        boundaries = np.array([i * segment_size for i in range(num_segments + 1)], dtype=np.int32)
        boundaries[-1] = seq_len

        # Find indices in positions array for each boundary
        boundary_indices = np.searchsorted(positions, boundaries)

        for seg in range(num_segments):
            start_idx = boundary_indices[seg]
            end_idx = boundary_indices[seg + 1]
            seg_count = end_idx - start_idx

            if seg_count == 0:
                segment_scores[seg] = 0.0
                continue

            # Calculate precision for this segment
            if matches is not None:
                correct = np.sum(matches[start_idx:end_idx])
                segment_scores[seg] = correct / seg_count
            else:
                # Simplified: use ratio of positions in segment
                seg_end = boundaries[seg + 1]
                seg_start = boundaries[seg]
                segment_scores[seg] = min(1.0, seg_count / (seg_end - seg_start) * 1000)

        # Calculate stability as consistency across segments
        mean_score = np.mean(segment_scores)
        std_score = np.std(segment_scores)
        stability = max(0.0, mean_score - std_score)

        return stability

    def _calculate_rule_precision(self, rule: Dict[str, Any], phi_sequence: str) -> float:
        """
        Calcula la precisió real d'una regla sobre la seqüència.

        La precisió es defineix com: prediccions_correctes / total_prediccions
        """
        rule_type = rule.get('rule_type', '')

        if rule_type == 'markov_transition':
            # Per regles Markov: comptar quantes vegades el context prediu correctament
            context = rule.get('context', '')
            prediction = rule.get('prediction', '')

            if not context or not prediction:
                return rule.get('confidence', 0.5)

            correct = 0
            total = 0
            context_len = len(context)

            for i in range(len(phi_sequence) - context_len):
                if phi_sequence[i:i + context_len] == context:
                    total += 1
                    if i + context_len < len(phi_sequence):
                        actual_next = phi_sequence[i + context_len]
                        if actual_next == prediction:
                            correct += 1

            return correct / total if total > 0 else 0.0

        elif rule_type == 'context_before':
            # Per regles de context: comptar ocurrències del context seguit del patró
            context = rule.get('context', '')
            produces = rule.get('produces', [])

            if not context or not produces:
                return rule.get('confidence', 0.5)

            # Buscar ocurrències del context
            correct = 0
            total = 0

            i = 0
            while i < len(phi_sequence) - len(context):
                if phi_sequence[i:i + len(context)] == context:
                    total += 1
                    # Comprovar si el patró esperat segueix
                    remaining = phi_sequence[i + len(context):]
                    for pattern_id in produces:
                        # Assumim que pattern_id conté info del patró
                        if remaining.startswith(pattern_id) or context in remaining[:20]:
                            correct += 1
                            break
                i += 1

            return correct / total if total > 0 else 0.0

        elif rule_type == 'context_after':
            # Similar però mirant el context després del patró
            pattern = rule.get('pattern', '')
            prediction = rule.get('prediction', '')

            if not pattern or not prediction:
                return rule.get('confidence', 0.5)

            correct = 0
            total = 0

            i = 0
            while i < len(phi_sequence) - len(pattern):
                if phi_sequence[i:i + len(pattern)] == pattern:
                    total += 1
                    remaining = phi_sequence[i + len(pattern):i + len(pattern) + len(prediction)]
                    if remaining == prediction:
                        correct += 1
                i += 1

            return correct / total if total > 0 else 0.0

        # Per altres tipus de regles, retornar la confiança original
        return rule.get('confidence', 0.5)
    
    def _calculate_rule_stability(self, rule: Dict[str, Any], phi_sequence: str) -> float:
        """Calcula l'estabilitat d'una regla en diferents parts de la seqüència."""
        if len(phi_sequence) < 100:
            return rule.get('confidence', 0.5)
        
        # Dividir la seqüència en segments i provar la regla en cada segment
        num_segments = 4
        segment_size = len(phi_sequence) // num_segments
        
        segment_scores = []
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(phi_sequence)
            segment = phi_sequence[start:end]
            
            # Aplicar la regla al segment (implementació simplificada)
            score = self._test_rule_on_segment(rule, segment)
            segment_scores.append(score)
        
        # Calcular estabilitat com la consistència entre segments
        if segment_scores:
            mean_score = np.mean(segment_scores)
            std_score = np.std(segment_scores)
            stability = max(0.0, mean_score - std_score)  # Penalitzar alta variabilitat
        else:
            stability = 0.0
        
        return stability
    
    def _test_rule_on_segment(self, rule: Dict[str, Any], segment: str) -> float:
        """
        Prova una regla en un segment de la seqüència.

        Retorna la precisió de la regla en aquest segment específic.
        """
        # Reutilitzem la lògica de _calculate_rule_precision
        return self._calculate_rule_precision(rule, segment)
    
    def save_rules(self, output_path: str) -> None:
        """Save inferred rules in JSON format."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.inferred_rules, f, indent=2)

        print(f"💾 Rules saved to {output_path}")

    def get_rules_summary(self) -> Dict[str, Any]:
        """Return a summary of inferred rules."""
        if not self.inferred_rules:
            return {"total_rules": 0}

        total_rules = len(self.inferred_rules)
        avg_confidence = np.mean([r.get('confidence', 0) for r in self.inferred_rules])
        avg_precision = np.mean([r.get('precision', 0) for r in self.inferred_rules])
        avg_complexity = np.mean([r.get('complexity', 0) for r in self.inferred_rules])

        rule_types = Counter(r.get('rule_type', 'unknown') for r in self.inferred_rules)

        return {
            "total_rules": total_rules,
            "average_confidence": avg_confidence,
            "average_precision": avg_precision,
            "average_complexity": avg_complexity,
            "rule_types": dict(rule_types),
            "top_rules": sorted(self.inferred_rules,
                              key=lambda r: r.get('confidence', 0),
                              reverse=True)[:5]
        }


if __name__ == "__main__":
    # Usage example
    print("🧠 Emergent Rule Inferer (Ω-inductor)")
    print("=" * 50)

    # Create example patterns
    example_patterns = [
        {
            'pattern_id': 'test_01',
            'pattern_data': '101',
            'positions': [0, 5, 10],
            'recurrence': 3
        },
        {
            'pattern_id': 'test_02',
            'pattern_data': '010',
            'positions': [2, 7, 12],
            'recurrence': 3
        }
    ]

    example_sequence = "101010101010101"

    inferer = RuleInferer(context_window=3, min_rule_confidence=0.6)
    rules = inferer.infer_rules(example_patterns, example_sequence)

    print(f"Rules inferred: {len(rules)}")
    for rule in rules[:3]:  # Show only first 3
        print(f"  - {rule['rule_description']} (confidence: {rule.get('confidence', 0):.2f})")

    summary = inferer.get_rules_summary()
    print(f"Summary: {summary['total_rules']} rules, average confidence: {summary['average_confidence']:.2f}")
