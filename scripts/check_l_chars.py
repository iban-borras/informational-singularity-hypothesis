#!/usr/bin/env python3
"""Check what characters are in variant L file."""

import sys
sys.path.insert(0, '.')

from utils.streaming_phi_loader import load_phi_for_agents
from collections import Counter

# Load first 10000 chars of L
path = 'results/level0/phi_snapshots/var_L/phi_iter15.struct.gz'
print('Loading first 10000 chars of variant L...')
phi, meta = load_phi_for_agents(path, max_chars=10000)

print(f'Length: {len(phi)}')
print(f'First 200 chars: {phi[:200]}')
print(f'\nUnique chars: {set(phi)}')

counts = Counter(phi)
print(f'\nChar counts:')
for char, count in sorted(counts.items()):
    print(f'  {repr(char)}: {count}')

