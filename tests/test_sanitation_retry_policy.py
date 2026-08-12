#!/usr/bin/env python3
"""Regression checks for the sanitation runner's retry classification."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import hsi_v2_q6a_level0_sanitation_runner as runner


def assert_retry_control() -> None:
    original_run_command = runner.run_command
    original_wait = runner.wait_for_results_base
    args = SimpleNamespace(
        transient_retries=2,
        results_ready_timeout_seconds=1,
        results_ready_check_interval_seconds=0.01,
        no_retry_resume=False,
    )

    try:
        runner.wait_for_results_base = lambda *_args, **_kwargs: None
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "runner.log"
            calls: list[list[str]] = []

            def deterministic_failure(command: list[str], _log_path: Path) -> tuple[int, bool]:
                calls.append(command)
                return 1, False

            runner.run_command = deterministic_failure
            code = runner.run_command_with_retries(
                ["python", "generator.py", "--no-resume"],
                log_path,
                Path(temp_dir),
                args,
            )
            assert code == 1
            assert len(calls) == 1

            outcomes = iter(((1, True), (0, False)))
            calls.clear()

            def transient_then_success(command: list[str], _log_path: Path) -> tuple[int, bool]:
                calls.append(command)
                return next(outcomes)

            runner.run_command = transient_then_success
            code = runner.run_command_with_retries(
                ["python", "generator.py", "--no-resume"],
                log_path,
                Path(temp_dir),
                args,
            )
            assert code == 0
            assert len(calls) == 2
            assert "--no-resume" in calls[0]
            assert "--no-resume" not in calls[1]
    finally:
        runner.run_command = original_run_command
        runner.wait_for_results_base = original_wait


def main() -> None:
    transient_examples = (
        "OSError: [WinError 21] The device is not ready",
        "OSError: [WinError 1117] The request could not be performed because of an I/O device error",
        "OSError: [WinError 1167] The device is not connected",
    )
    deterministic_examples = (
        "ModuleNotFoundError: No module named 'level0'",
        "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 9.32 MiB",
        "MemoryError",
        "Structural snapshot not found",
    )

    for line in transient_examples:
        assert runner.is_transient_storage_error(line), line
    for line in deterministic_examples:
        assert not runner.is_transient_storage_error(line), line

    assert_retry_control()

    print("Sanitation retry policy tests passed.")


if __name__ == "__main__":
    main()
