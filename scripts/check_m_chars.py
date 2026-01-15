#!/usr/bin/env python3
"""Check what characters are in variant M file."""

import sys
sys.path.insert(0, '.')

from utils.streaming_phi_loader import load_phi_for_agents

# Load first 1000 chars of M
path = 'results/level0/phi_snapshots/var_M/phi_iter19.struct.gz'
print('Loading first 10000 chars of variant M...')
phi, meta = load_phi_for_agents(path, max_chars=10000)

print(f'Length: {len(phi)}')
print(f'First 200 chars: {phi[:200]}')
print()
print(f'Contains open paren: {"(" in phi}')
print(f'Contains close paren: {")" in phi}')
print(f'Unique chars: {set(phi)}')

# Count each char
from collections import Counter
counts = Counter(phi)
print(f'\nChar counts:')
for char, count in sorted(counts.items()):
    print(f'  {repr(char)}: {count}')

