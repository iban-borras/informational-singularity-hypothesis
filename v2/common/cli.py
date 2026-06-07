from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_dotenv_for_results() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        print(f"[WARN] Failed to load .env for results path: {exc}")


_load_dotenv_for_results()


def parse_variants(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def resolve_dir(raw_path: str, *, anchor_file: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(anchor_file).parent / path
    return path.resolve()


def resolve_results_dir(raw_path: str, *, anchor_file: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.expanduser().resolve()

    parts = path.parts
    results_base = os.environ.get("HSI_RESULTS_BASE_DIR")
    if results_base and parts and parts[0].lower() == "results":
        return (Path(results_base).expanduser() / Path(*parts[1:])).resolve()

    return (Path(anchor_file).parent / path).resolve()


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
