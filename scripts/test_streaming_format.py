#!/usr/bin/env python3
"""Test that the streaming generators produce correct v33 binary format."""

import sys
import tempfile
import os
sys.path.insert(0, '.')

from level0.control_variants import (
    generate_logistic_map_bits_streaming,
    generate_fibonacci_word_streaming
)
from utils.streaming_phi_loader import load_phi_for_agents
from collections import Counter

def test_format(name, generator_func, num_bits=1000):
    """Test a streaming generator produces correct format."""
    print(f"\n{'='*50}")
    print(f"Testing {name}...")
    
    with tempfile.NamedTemporaryFile(suffix='.struct.gz', delete=False) as f:
        temp_path = f.name
    
    try:
        # Generate
        ones, zeros, size = generator_func(num_bits, temp_path, silent=True)
        print(f"  Generated: {ones} ones, {zeros} zeros, {size} bytes")
        
        # Load and verify
        phi, meta = load_phi_for_agents(temp_path, max_chars=num_bits*2)
        
        print(f"  Loaded length: {len(phi)}")
        print(f"  First 100 chars: {phi[:100]}")
        print(f"  Unique chars: {set(phi)}")
        
        counts = Counter(phi[:num_bits])  # Only count expected chars
        print(f"  Char counts (first {num_bits}): {dict(counts)}")
        
        # Verify
        has_parens = '(' in phi or ')' in phi
        if has_parens:
            print(f"  ❌ FAIL: Found parentheses in flat sequence!")
            return False
        else:
            print(f"  ✅ PASS: No parentheses (correct for flat sequence)")
            
        # Verify counts match
        loaded_ones = phi[:num_bits].count('1')
        loaded_zeros = phi[:num_bits].count('0')
        if loaded_ones == ones and loaded_zeros == zeros:
            print(f"  ✅ PASS: Counts match (ones={ones}, zeros={zeros})")
            return True
        else:
            print(f"  ❌ FAIL: Count mismatch! Expected ones={ones}, zeros={zeros}, got ones={loaded_ones}, zeros={loaded_zeros}")
            return False
            
    finally:
        os.unlink(temp_path)

if __name__ == '__main__':
    print("Testing streaming generators with v33 binary format...")
    
    # Test with small sequences
    test_l = test_format("Logistic Map (L)", 
                         lambda n, p, silent: generate_logistic_map_bits_streaming(n, p, silent=silent),
                         num_bits=1000)
    
    test_m = test_format("Fibonacci Word (M)",
                         generate_fibonacci_word_streaming,
                         num_bits=1000)
    
    print(f"\n{'='*50}")
    if test_l and test_m:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

