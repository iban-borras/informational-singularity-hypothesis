import gzip
import tempfile
import unittest
from unittest import mock

from level0.accumulation_manager import AccumulationManager


class AccumulationManagerFlushTests(unittest.TestCase):
    def test_flush_preserves_exact_content_with_bounded_chunks(self):
        for compress in (False, True):
            with self.subTest(compress=compress), tempfile.TemporaryDirectory() as tmp:
                manager = AccumulationManager(tmp, "T", compress=compress)
                manager.write_chunk_chars = 4
                manager.buffer_char_limit = 5

                manager.append("(01)")
                manager.append("10(1)")

                opener = gzip.open if compress else open
                with opener(manager.file_path, "rt", encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "(01)10(1)")

                self.assertEqual(manager.get_length(), 9)
                self.assertEqual(manager.get_clean_bits_count(), 5)

    def test_flush_splits_a_single_large_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = AccumulationManager(tmp, "T")
            manager.write_chunk_chars = 3
            manager.buffer_char_limit = 1

            writes = []

            class RecordingWriter:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def write(self, value):
                    writes.append(value)

            with mock.patch.object(manager, "_open_file", return_value=RecordingWriter()):
                manager.append("abcdefgh")

            self.assertEqual(writes, ["abc", "def", "gh"])
            self.assertEqual(manager.buffer, [])
            self.assertEqual(manager.buffer_chars, 0)

    def test_failed_flush_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = AccumulationManager(tmp, "T")
            manager.write_chunk_chars = 3
            manager.buffer_char_limit = 1

            class FailingWriter:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def write(self, value):
                    raise MemoryError("simulated")

            with mock.patch.object(manager, "_open_file", return_value=FailingWriter()):
                with self.assertRaisesRegex(MemoryError, "simulated"):
                    manager.append("abcdef")

            self.assertTrue(manager._write_failed)
            self.assertEqual(manager.buffer, [])
            with self.assertRaisesRegex(RuntimeError, "last durable checkpoint"):
                manager._flush()


if __name__ == "__main__":
    unittest.main()
