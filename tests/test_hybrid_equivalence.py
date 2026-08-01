"""
Test Hybrid Engine Equivalence

Verifies that HybridCollapseEngine produces identical results to in-memory
regex collapse. This is critical for scientific validity: results must be
independent of the processing method used.

Author: Iban Borràs with Augment Agent (Sophia)
Date: December 2025
"""

import gzip
import random
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


    def test_gzip_checkpoint_resume_equivalence(self):
        """An interrupted gzip pass resumes from its last atomic part."""
        test_input = "((01)1)0((10)0)1" * 200
        expected, expected_changed = collapse_inmemory_one_pass(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt.gz"
            output_file = tmpdir / "output.txt.gz"
            with gzip.open(input_file, "wt", encoding="utf-8", compresslevel=1) as f:
                f.write(test_input)

            interrupted = HybridCollapseEngine(
                max_ram_bytes=128,
                compress=True,
                stream_chunk_chars=31,
                checkpoint_chars=93,
            )
            original_write = interrupted._write_json_atomic

            def stop_after_second_part(path, payload):
                original_write(path, payload)
                if path.name == "checkpoint.json" and len(payload.get("parts", [])) == 2:
                    raise RuntimeError("simulated interruption")

            interrupted._write_json_atomic = stop_after_second_part
            try:
                interrupted.collapse_one_pass(
                    input_file,
                    output_file,
                    log_progress=False,
                    checkpoint_key="resume-test",
                    expected_input_chars=len(test_input),
                )
                raise AssertionError("The simulated interruption did not fire")
            except RuntimeError as exc:
                assert str(exc) == "simulated interruption"

            resumed = HybridCollapseEngine(
                max_ram_bytes=128,
                compress=True,
                stream_chunk_chars=31,
                checkpoint_chars=93,
            )
            output_size, had_changes = resumed.collapse_one_pass(
                input_file,
                output_file,
                log_progress=False,
                checkpoint_key="resume-test",
                expected_input_chars=len(test_input),
            )
            with gzip.open(output_file, "rt", encoding="utf-8") as f:
                actual = f.read()

            assert actual == expected
            assert output_size == len(expected)
            assert had_changes == expected_changed

    def test_match_crossing_every_chunk_boundary(self):
        """Regex matches spanning read boundaries remain exact."""
        test_input = "0" * 7 + "(" + "01" * 101 + ")" + "1" * 9
        expected, _ = collapse_inmemory_one_pass(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt"
            output_file = tmpdir / "output.txt"
            input_file.write_text(test_input)

            engine = HybridCollapseEngine(
                max_ram_bytes=64,
                stream_chunk_chars=13,
                checkpoint_chars=39,
            )
            engine.collapse_one_pass(
                input_file,
                output_file,
                log_progress=False,
                checkpoint_key="boundary-test",
                expected_input_chars=len(test_input),
            )
            assert output_file.read_text() == expected

    def test_completed_pass_is_reused(self):
        """A completed pass is returned without re-running simplification."""
        test_input = "((01)1)0" * 50

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "input.txt"
            output_file = tmpdir / "output.txt"
            input_file.write_text(test_input)

            first = HybridCollapseEngine(
                max_ram_bytes=64,
                stream_chunk_chars=17,
                checkpoint_chars=51,
            )
            expected_size, expected_changed = first.collapse_one_pass(
                input_file,
                output_file,
                log_progress=False,
                checkpoint_key="complete-cache-test",
                expected_input_chars=len(test_input),
            )

            def must_not_run(_):
                raise AssertionError("Completed pass cache was not reused")

            cached = HybridCollapseEngine(
                max_ram_bytes=64,
                simplify_fn=must_not_run,
                stream_chunk_chars=17,
                checkpoint_chars=51,
            )
            actual_size, actual_changed = cached.collapse_one_pass(
                input_file,
                output_file,
                log_progress=False,
                checkpoint_key="complete-cache-test",
                expected_input_chars=len(test_input),
            )
            assert actual_size == expected_size
            assert actual_changed == expected_changed

    def test_randomized_chunk_boundary_equivalence(self):
        """Random structural strings agree with re.sub at many chunk sizes."""
        rng = random.Random(42)
        for case_number in range(12):
            length = rng.randint(1, 600)
            test_input = "".join(rng.choice("01()") for _ in range(length))
            expected, expected_changed = collapse_inmemory_one_pass(test_input)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                input_file = tmpdir / "input.txt"
                input_file.write_text(test_input)
                for chunk_size in (1, 2, 7, 17, 64):
                    output_file = tmpdir / f"output-{chunk_size}.txt"
                    engine = HybridCollapseEngine(
                        max_ram_bytes=max(4, chunk_size * 4),
                        stream_chunk_chars=chunk_size,
                        checkpoint_chars=10_000,
                    )
                    output_size, had_changes = engine.collapse_one_pass(
                        input_file,
                        output_file,
                        log_progress=False,
                        checkpoint_key=f"fuzz-{case_number}-{chunk_size}",
                        expected_input_chars=len(test_input),
                    )
                    actual = output_file.read_text()
                    assert actual == expected
                    assert output_size == len(expected)
                    assert had_changes == expected_changed

    def test_compiled_base_collapse_equivalence(self):
        """The Numba base-rule path is byte-exact across random structures."""
        rng = random.Random(20260801)
        for case_number in range(20):
            test_input = "".join(rng.choice("01()") for _ in range(2_000))
            expected, expected_changed = collapse_inmemory_one_pass(
                test_input,
                _simplify_base,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                input_file = tmpdir / "input.txt"
                output_file = tmpdir / "output.txt"
                input_file.write_text(test_input)
                engine = HybridCollapseEngine(
                    max_ram_bytes=256,
                    simplify_fn=_simplify_base,
                    stream_chunk_chars=127,
                    checkpoint_chars=508,
                    compiled_base_collapse=True,
                )
                output_size, had_changes = engine.collapse_one_pass(
                    input_file,
                    output_file,
                    log_progress=False,
                    checkpoint_key=f"compiled-base-{case_number}",
                    expected_input_chars=len(test_input),
                )
                actual = output_file.read_text()
                assert actual == expected
                assert output_size == len(expected)
                assert had_changes == expected_changed


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

    print("\n5. Gzip checkpoint resume test:")
    test.test_gzip_checkpoint_resume_equivalence()
    print("   ✓ Interrupted gzip pass resumes exactly")

    print("\n6. Cross-boundary regex test:")
    test.test_match_crossing_every_chunk_boundary()
    print("   ✓ Match crossing chunk boundaries")

    print("\n7. Completed-pass cache test:")
    test.test_completed_pass_is_reused()
    print("   ✓ Completed pass reused without recomputation")

    print("\n8. Randomized boundary equivalence test:")
    test.test_randomized_chunk_boundary_equivalence()
    print("   ✓ 12 random structures across 5 chunk sizes")

    print("\n9. Compiled base-collapse equivalence test:")
    test.test_compiled_base_collapse_equivalence()
    print("   Compiled path matches 20 random reference cases")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
