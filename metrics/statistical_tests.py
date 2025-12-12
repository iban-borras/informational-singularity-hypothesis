"""
Statistical Significance Tests for HSI Metrics

This module provides null model comparison and statistical significance testing
for HSI metrics. These tests help distinguish genuine emergent properties from
random artifacts.

Author: Iban Borràs with Augment Agent (Sophia)
Date: December 2025
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import random


def generate_null_model(length: int, n_samples: int = 100, 
                        p_one: float = 0.5) -> List[str]:
    """
    Generate random binary sequences for null model comparison.
    
    Args:
        length: Length of each random sequence
        n_samples: Number of random samples to generate
        p_one: Probability of generating '1' (default 0.5 for uniform)
        
    Returns:
        List of random binary strings
    """
    samples = []
    for _ in range(n_samples):
        if p_one == 0.5:
            # Fast path for uniform distribution
            seq = ''.join(random.choices('01', k=length))
        else:
            # Custom probability
            seq = ''.join('1' if random.random() < p_one else '0' for _ in range(length))
        samples.append(seq)
    return samples


def generate_shuffled_null(phi_sequence: str, n_samples: int = 100) -> List[str]:
    """
    Generate null samples by shuffling the original Φ sequence.
    
    This preserves the bit distribution (proportion of 0s and 1s)
    while destroying any structural patterns.
    
    Args:
        phi_sequence: Original Φ sequence to shuffle
        n_samples: Number of shuffled samples to generate
        
    Returns:
        List of shuffled sequences
    """
    chars = list(phi_sequence)
    samples = []
    for _ in range(n_samples):
        shuffled = chars.copy()
        random.shuffle(shuffled)
        samples.append(''.join(shuffled))
    return samples


def permutation_test(observed_value: float, 
                     null_values: List[float],
                     alternative: str = 'greater') -> Dict[str, float]:
    """
    Calculate p-value for observed value against null distribution.
    
    Args:
        observed_value: The observed metric value from Φ
        null_values: List of metric values from null model
        alternative: 'greater', 'less', or 'two-sided'
        
    Returns:
        Dictionary with p_value, z_score, and interpretation
    """
    null_array = np.array(null_values)
    null_mean = np.mean(null_array)
    null_std = np.std(null_array)
    
    # Avoid division by zero
    if null_std < 1e-10:
        z_score = 0.0 if abs(observed_value - null_mean) < 1e-10 else np.inf
    else:
        z_score = (observed_value - null_mean) / null_std
    
    # Calculate p-value
    if alternative == 'greater':
        p_value = np.mean(null_array >= observed_value)
    elif alternative == 'less':
        p_value = np.mean(null_array <= observed_value)
    else:  # two-sided
        p_value = np.mean(np.abs(null_array - null_mean) >= abs(observed_value - null_mean))
    
    # Interpret significance
    if p_value < 0.001:
        significance = 'highly significant (p < 0.001)'
    elif p_value < 0.01:
        significance = 'very significant (p < 0.01)'
    elif p_value < 0.05:
        significance = 'significant (p < 0.05)'
    elif p_value < 0.1:
        significance = 'marginally significant (p < 0.1)'
    else:
        significance = 'not significant'
    
    return {
        'observed': observed_value,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
        'p_value': p_value,
        'significance': significance,
        'n_samples': len(null_values)
    }


def compare_to_null_model(phi_sequence: str,
                          metric_fn: Callable[[str], float],
                          n_samples: int = 100,
                          null_type: str = 'shuffle',
                          alternative: str = 'greater') -> Dict[str, any]:
    """
    Compare a Φ metric to a null model distribution.
    
    Args:
        phi_sequence: The Φ sequence to analyze
        metric_fn: Function that takes a sequence and returns a metric value
        n_samples: Number of null samples to generate
        null_type: 'shuffle' (preserves bit distribution) or 'random' (uniform 50/50)
        alternative: 'greater', 'less', or 'two-sided'
        
    Returns:
        Dictionary with observed value, null statistics, and significance
    """
    # Calculate observed value
    observed = metric_fn(phi_sequence)
    
    # Generate null samples
    if null_type == 'shuffle':
        null_sequences = generate_shuffled_null(phi_sequence, n_samples)
    else:
        null_sequences = generate_null_model(len(phi_sequence), n_samples)
    
    # Calculate metric on null samples
    null_values = [metric_fn(seq) for seq in null_sequences]
    
    # Run permutation test
    result = permutation_test(observed, null_values, alternative)
    result['null_type'] = null_type
    result['metric_name'] = metric_fn.__name__ if hasattr(metric_fn, '__name__') else 'unknown'
    
    return result

