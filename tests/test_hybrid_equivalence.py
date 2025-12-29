"""
Test Hybrid Engine Equivalence

Verifies that HybridCollapseEngine produces identical results to in-memory
regex collapse. This is critical for scientific validity: results must be
independent of the processing method used.

Author: Iban Borràs with Augment Agent (Sophia)
Date: December 2025
"""

import re
import tempfile
from pathlib import Path
import sys

# pytest is optional - only needed for pytest runner
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy decorator
    class pytest:
        @staticmethod
        def mark():
            pass
        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator

# Import engines
sys.path.insert(0, str(Path(__file__).parent.parent))

from level0.hybrid_collapse_engine import HybridCollapseEngine


# =============================================================================
# REFERENCE IN-MEMORY COLLAPSE (the "gold standard")
# =============================================================================

def _simplify_base(seq: str) -> str:
    """Base AND-rule: 01→0, 10→0, compress runs. Returns '0' if empty."""
    prev = None
    result = seq
    while prev != result:
        prev = result
        result = re.sub(r'01', '0', result)
        result = re.sub(r'10', '0', result)
        result = re.sub(r'0+', '0', result)
        result = re.sub(r'1+', '1', result)
    return result if result else '0'


def _simplify_variant_d(seq: str) -> str:
    """Variant D: 10→0, 01→0, compress runs."""
    prev = None
    result = seq
    while prev != result:
        prev = result
        result = re.sub(r'10', '0', result)
        result = re.sub(r'01', '0', result)
        result = re.sub(r'0+', '0', result)
        result = re.sub(r'1+', '1', result)
    return result if result else '0'


def _simplify_variant_e(seq: str) -> str:
    """Variant E: Phase 1 (01→0) then Phase 2 (10→0)."""
    # Phase 1
    result = re.sub(r'01', '0', seq)
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)
    # Phase 2
    result = re.sub(r'10', '0', result)
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)
    return result if result else '0'


def _simplify_variant_i(seq: str) -> str:
    """Variant I: Phase 1 (10→0) then Phase 2 (01→0). Inverse of E."""
    # Phase 1
    result = re.sub(r'10', '0', seq)
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)
    # Phase 2
    result = re.sub(r'01', '0', result)
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)
    return result if result else '0'


VARIANT_SIMPLIFY_FNS = {
    'B': _simplify_base,
    'D': _simplify_variant_d,
    'E': _simplify_variant_e,
    'F': _simplify_base,
    'G': _simplify_base,
    'H': _simplify_base,
    'I': _simplify_variant_i,
}


def collapse_inmemory_one_pass(data: str, simplify_fn=None) -> tuple:
    """
    Reference in-memory collapse: one pass of innermost parentheses.
    
    Returns:
        Tuple (result, had_changes)
    """
    sf = simplify_fn or _simplify_base
    pattern = re.compile(r'\(([01]+)\)')
    had_changes = [False]
    
    def replacer(m):
        had_changes[0] = True
        return sf(m.group(1))
    
    result = pattern.sub(replacer, data)
    return result, had_changes[0]


def collapse_inmemory_full(data: str, simplify_fn=None) -> str:
    """Collapse fully using in-memory regex (reference implementation)."""
    sf = simplify_fn or _simplify_base
    result = data
    while True:
        new_result, had_changes = collapse_inmemory_one_pass(result, sf)
        if not had_changes:
            break
        result = new_result
    return result


# =============================================================================
# TEST CASES
# =============================================================================

class TestHybridEquivalence:
    """Test that HybridEngine produces identical results to in-memory."""
    
    # Test inputs of increasing complexity
    TEST_CASES = [
        # Simple cases
        "(01)",
        "(10)",
        "(11)",
        "(00)",
        "((01))",
        # Nested
        "((01)1)",
        "(0(10))",
        "((01)(10))",
        # Multiple groups
        "(01)(10)(11)",
        "((01)1)0((10)0)1",
        # Deep nesting
        "(((01)1)0)",
        "((((01)0)1)0)",
        # Mixed
        "0(01)1(10)0",
        "1((01)(10))0",
    ]
    
    @pytest.mark.parametrize("test_input", TEST_CASES)
    def test_one_pass_equivalence(self, test_input):
        """Verify one-pass collapse produces identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt"
            output_file = tmpdir / "output.txt"
            
            # Write input
            input_file.write_text(test_input)
            
            # In-memory reference
            expected, _ = collapse_inmemory_one_pass(test_input)
            
            # Hybrid engine (force small blocks to test chunking)
            engine = HybridCollapseEngine(max_ram_bytes=100)
            engine.collapse_one_pass(input_file, output_file, log_progress=False)
            actual = output_file.read_text()
            
            assert actual == expected, (
                f"Mismatch!\n"
                f"  Input:    {test_input}\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )

    @pytest.mark.parametrize("test_input", TEST_CASES)
    def test_full_collapse_equivalence(self, test_input):
        """Verify full collapse produces identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt"
            output_file = tmpdir / "output.txt"

            input_file.write_text(test_input)

            # In-memory reference
            expected = collapse_inmemory_full(test_input)

            # Hybrid engine
            engine = HybridCollapseEngine(max_ram_bytes=100)
            engine.collapse_fully(input_file, output_file, log_progress=False)
            actual = output_file.read_text()

            assert actual == expected, (
                f"Full collapse mismatch!\n"
                f"  Input:    {test_input}\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )

    @pytest.mark.parametrize("variant", ['B', 'D', 'E', 'I'])
    def test_variant_equivalence(self, variant):
        """Verify each variant produces identical results with hybrid."""
        test_input = "((01)1)0((10)0)1" * 10  # Larger input
        simplify_fn = VARIANT_SIMPLIFY_FNS[variant]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt"
            output_file = tmpdir / "output.txt"

            input_file.write_text(test_input)

            # In-memory reference
            expected = collapse_inmemory_full(test_input, simplify_fn)

            # Hybrid engine with same simplify function
            engine = HybridCollapseEngine(max_ram_bytes=500, simplify_fn=simplify_fn)
            engine.collapse_fully(input_file, output_file, log_progress=False)
            actual = output_file.read_text()

            assert actual == expected, (
                f"Variant {variant} mismatch!\n"
                f"  Expected: {expected[:100]}...\n"
                f"  Actual:   {actual[:100]}..."
            )

    def test_large_input_chunking(self):
        """Test with input large enough to force multiple blocks."""
        # Create input that will need multiple chunks
        base = "((01)1)0((10)0)1"
        test_input = base * 1000  # ~18KB

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt"
            output_file = tmpdir / "output.txt"

            input_file.write_text(test_input)

            # In-memory reference
            expected = collapse_inmemory_full(test_input)

            # Hybrid with very small blocks to force many chunks
            engine = HybridCollapseEngine(max_ram_bytes=1000)  # 1KB blocks
            engine.collapse_fully(input_file, output_file, log_progress=False)
            actual = output_file.read_text()

            assert actual == expected, (
                f"Large input mismatch!\n"
                f"  Input length: {len(test_input)}\n"
                f"  Expected length: {len(expected)}\n"
                f"  Actual length: {len(actual)}"
            )


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    print("🧪 Running Hybrid Engine Equivalence Tests")
    print("=" * 60)

    test = TestHybridEquivalence()

    print("\n1. One-pass equivalence tests:")
    for i, case in enumerate(TestHybridEquivalence.TEST_CASES):
        try:
            test.test_one_pass_equivalence(case)
            print(f"   ✓ Case {i+1}: {case[:30]}...")
        except AssertionError as e:
            print(f"   ✗ Case {i+1}: FAILED - {e}")

    print("\n2. Full collapse equivalence tests:")
    for i, case in enumerate(TestHybridEquivalence.TEST_CASES):
        try:
            test.test_full_collapse_equivalence(case)
            print(f"   ✓ Case {i+1}: {case[:30]}...")
        except AssertionError as e:
            print(f"   ✗ Case {i+1}: FAILED - {e}")

    print("\n3. Variant equivalence tests:")
    for variant in ['B', 'D', 'E', 'I']:
        try:
            test.test_variant_equivalence(variant)
            print(f"   ✓ Variant {variant}")
        except AssertionError as e:
            print(f"   ✗ Variant {variant}: FAILED - {e}")

    print("\n4. Large input chunking test:")
    try:
        test.test_large_input_chunking()
        print("   ✓ Large input (18KB, 1KB blocks)")
    except AssertionError as e:
        print(f"   ✗ FAILED - {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")

