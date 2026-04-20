from __future__ import annotations

import argparse
from pathlib import Path


def parse_variants(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def resolve_dir(raw_path: str, *, anchor_file: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(anchor_file).parent / path
    return path.resolve()


def parse_single_policy(
    raw: str,
    *,
    parser: argparse.ArgumentParser,
    flag: str,
    parse_policies_fn,
) -> str:
    try:
        policies = parse_policies_fn(raw)
    except ValueError as exc:
        parser.error(str(exc))
    if len(policies) != 1:
        parser.error(f"{flag} expects exactly one policy.")
    return policies[0]
