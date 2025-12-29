#!/usr/bin/env python3
"""Quick script to check max_depth of variants using streaming loader."""
import sys
import re
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.streaming_phi_loader import StreamingPhiLoader

def check_depth_streaming(path: Path, sample_size: int = 10_000_000) -> dict:
    """Stream through file and count max depth without loading all to RAM."""
    loader = StreamingPhiLoader(path)

    depth = 0
    max_depth = 0
    chars_read = 0

    for char in loader.iter_chars():
        if char == '(':
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif char == ')':
            depth -= 1
        chars_read += 1
        if chars_read >= sample_size:
            break

    return {
        'max_depth': max_depth,
        'final_depth': depth,
        'chars_sampled': chars_read,
        'exceeds_256': max_depth > 256
    }

def extract_iter_num(path: Path) -> int:
    """Extract iteration number from filename."""
    match = re.search(r'iter(\d+)', path.stem)
    return int(match.group(1)) if match else 0

def analyze_depth_growth(var: str, base: Path):
    """Analyze depth growth across iterations for a variant."""
    var_path = base / f"var_{var}"
    if not var_path.exists():
        print(f"  {var}: No data")
        return

    files = sorted(var_path.glob("phi_iter*.struct.gz"), key=extract_iter_num)
    if not files:
        print(f"  {var}: No struct.gz files")
        return

    print(f"\n📊 Variant {var} - Depth Growth Analysis")
    print(f"   {'Iter':<6} {'max_depth':<12} {'Status'}")
    print(f"   {'-'*30}")

    for f in files:
        iter_num = extract_iter_num(f)
        try:
            result = check_depth_streaming(f, sample_size=50_000_000)
            status = "⚠️ >256" if result['exceeds_256'] else "✓"
            print(f"   {iter_num:<6} {result['max_depth']:<12} {status}")
        except Exception as e:
            print(f"   {iter_num:<6} Error: {e}")

if __name__ == "__main__":
    base = Path("results/level0/phi_snapshots")

    # Check if specific variant requested
    if len(sys.argv) > 1:
        var = sys.argv[1].upper()
        analyze_depth_growth(var, base)
    else:
        # Quick check of latest iteration per variant
        variants = ['F', 'B', 'E', 'A', 'H']

        print("Variant Depth Check (latest iteration, sample: 10M chars)")
        print("=" * 55)

        for var in variants:
            var_path = base / f"var_{var}"
            if not var_path.exists():
                print(f"  {var}: No data")
                continue

            files = sorted(var_path.glob("phi_iter*.struct.gz"), key=extract_iter_num)
            if not files:
                print(f"  {var}: No struct.gz files")
                continue

            latest = files[-1]
            iter_num = extract_iter_num(latest)

            print(f"  {var} (iter {iter_num:>2}): ", end="", flush=True)
            try:
                result = check_depth_streaming(latest)
                status = "⚠️ EXCEEDS 256!" if result['exceeds_256'] else "✓ OK"
                print(f"max_depth = {result['max_depth']} {status}")
            except Exception as e:
                print(f"Error: {e}")

        print("\n💡 Per veure creixement: python scripts/check_depth.py B")

