#!/usr/bin/env python3
"""Debug encoding/decoding of Fibonacci sequence."""

import sys
sys.path.insert(0, '.')

from utils.bitarray_encoder import encode_phi_with_structure, decode_phi_with_structure
from bitarray import bitarray

# Test with a simple Fibonacci sequence (just 0s and 1s)
fib = "0100101001001"  # First 13 chars of Fibonacci word

print(f"Original: {fib}")
print(f"Length: {len(fib)}")

# Encode
bits = encode_phi_with_structure(fib)
print(f"\nEncoded bits: {bits.to01()}")
print(f"Bits length: {len(bits)}")
print(f"Bytes needed: {(len(bits) + 7) // 8}")

# Decode
decoded = decode_phi_with_structure(bits)
print(f"\nDecoded: {decoded}")
print(f"Match: {decoded == fib}")

# Now test what happens when we save to bytes and reload
print("\n--- Testing byte storage ---")
import gzip
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
    temp_path = f.name

# Save
with gzip.open(temp_path, 'wb') as f:
    bits.tofile(f)
    
# Check file size
file_size = os.path.getsize(temp_path)
print(f"File size: {file_size} bytes")

# Load
with gzip.open(temp_path, 'rb') as f:
    loaded_bits = bitarray()
    loaded_bits.fromfile(f)
    
print(f"Loaded bits: {loaded_bits.to01()}")
print(f"Loaded length: {len(loaded_bits)}")

# The issue: bitarray pads to 8 bits!
if len(loaded_bits) != len(bits):
    print(f"\n⚠️ PADDING DETECTED!")
    print(f"Original: {len(bits)} bits")
    print(f"Loaded: {len(loaded_bits)} bits")
    print(f"Extra bits: {len(loaded_bits) - len(bits)}")
    
# Decode the loaded bits
decoded_from_file = decode_phi_with_structure(loaded_bits)
print(f"\nDecoded from file: {decoded_from_file}")
print(f"Length: {len(decoded_from_file)}")

# Clean up
os.unlink(temp_path)

