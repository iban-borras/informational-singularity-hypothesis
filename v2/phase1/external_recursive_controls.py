from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from utils.bitarray_encoder import save_phi_structural_gz


CONTRACT_DOC = "docs/HSI_v2_Phase1_External_Recursive_Controls_Proposal_v0_3.md"
CONTRACT_SHA256 = "66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0"
TARGET_OBSERVABLE_BITS = 3_100_000
MIN_OBSERVABLE_BITS = 3_000_000
MAX_TERMINALIZATION_RATIO_CLEAN = 0.05
MAX_TERMINALIZATION_RATIO_VALID = 0.20
MIN_DENSITY = 0.15
MAX_DENSITY = 0.85
ITERATION_LABEL = 20

SHARED_TERMINAL_BLOCKS = {
    "U0": "0011010110010110100101101100101001101001011010010110110010100110",
    "U1": "1100101001101001011010010011010110010110100101101001011001011001",
    "U2": "0101100110100101101001100101101001011001011010011010010110010110",
    "U3": "1010011001011010010110011010010110100110010110100101101001101001",
}


def _chars(*parts: str) -> list[str]:
    output: list[str] = []
    for part in parts:
        token = part.strip()
        if token in SHARED_TERMINAL_BLOCKS:
            output.extend(SHARED_TERMINAL_BLOCKS[token])
        else:
            output.extend(part.replace(" ", ""))
    return output


CFG_RULES: dict[str, list[list[str]]] = {
    "S": [_chars("(A)D")],
    "A": [
        _chars("U0", "A", "U1", "B"),
        _chars("(", "A", ")", "C", "U2"),
        _chars("U3", "B", "A"),
        [],
        _chars("C", "U1", "A"),
        _chars("U2"),
    ],
    "B": [
        _chars("U1", "B", "U2", "C"),
        _chars("(", "B", ")", "A", "U3"),
        _chars("U0", "A", "B"),
        [],
        _chars("C", "U2", "B"),
        _chars("U3"),
    ],
    "C": [
        _chars("U2", "C", "U0", "A"),
        _chars("(", "A", ")", "B", "U1"),
        _chars("U3", "B", "C"),
        [],
        _chars("A", "U0"),
    ],
    "D": [
        _chars("U0", "D", "U1"),
        _chars("U2", "A"),
        _chars("U3"),
    ],
}

CFG_TERMINALIZATION = {
    "A": _chars("01"),
    "B": _chars("10"),
    "C": _chars("01"),
    "D": _chars("10"),
    "S": _chars("0"),
}

LSYSTEM_RULES: dict[str, list[str]] = {
    "A": _chars("(", "U0", "B", "U1", "C", "U2", "D", ")"),
    "B": _chars("(", "U1", "C", "U2", "A", "U3", "D", ")"),
    "C": _chars("U2", "(", "B", ")", "U3", "D", "U0", "A"),
    "D": _chars("(", "C", "U3", ")", "A", "U0", "B"),
}

LSYSTEM_TERMINALIZATION = {
    "A": _chars("01"),
    "B": _chars("10"),
    "C": _chars("0"),
    "D": _chars("1"),
}


@dataclass(frozen=True)
class GeneratedControl:
    variant: str
    structural_stream: str
    metadata: dict


def generate_control(variant: str) -> GeneratedControl:
    variant = variant.strip().upper()
    _validate_terminal_blocks()
    if variant == "O":
        return _generate_cfg_control()
    if variant == "P":
        return _generate_lsystem_control()
    raise ValueError(f"Unsupported external recursive control: {variant}")


def write_control_snapshot(
    *,
    generated: GeneratedControl,
    level0_root: Path,
    command: list[str],
    force: bool = False,
    quiet: bool = False,
) -> dict:
    var_dir = level0_root / f"var_{generated.variant}"
    struct_path = var_dir / f"phi_iter{ITERATION_LABEL}.struct.gz"
    metadata_path = var_dir / f"phi_iter{ITERATION_LABEL}.json"

    if struct_path.exists() or metadata_path.exists():
        if not force:
            return _reuse_existing_snapshot(
                variant=generated.variant,
                struct_path=struct_path,
                metadata_path=metadata_path,
            )
        if struct_path.exists():
            struct_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    var_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(
            f"[Snapshot] Writing var_{generated.variant} "
            f"({generated.metadata['final_observable_bits']:,} observable bits)"
        )

    save_phi_structural_gz(
        generated.structural_stream,
        str(struct_path),
        compresslevel=6,
        silent=quiet,
    )
    struct_sha = sha256_file(struct_path)
    metadata = {
        **generated.metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "hsi_v2_generate_external_recursive_controls.py",
        "command": command,
        "cwd": str(Path.cwd()),
        "snapshot_paths": {
            "struct_gz": str(struct_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
        "sha256_struct_gz": struct_sha,
    }
    _validate_metadata(metadata)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    return {
        "variant": generated.variant,
        "status": "generated",
        "struct_path": str(struct_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "sha256_struct_gz": struct_sha,
        "metadata": metadata,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _generate_cfg_control() -> GeneratedControl:
    sequence = ["S"]
    counters: Counter[str] = Counter()
    pass_index = 0

    while True:
        stats = _projected_stats(sequence, CFG_TERMINALIZATION)
        if _passes_generation_contract(stats):
            break
        if pass_index >= 50:
            raise RuntimeError(
                "Variant O failed v0.3 contract: reached max_passes=50 before target with clean terminalization."
            )

        next_sequence: list[str] = []
        for token in sequence:
            if token in CFG_RULES:
                productions = CFG_RULES[token]
                production = productions[counters[token] % len(productions)]
                counters[token] += 1
                next_sequence.extend(production)
            else:
                next_sequence.append(token)
        sequence = next_sequence
        pass_index += 1

    final_sequence, terminalization_bits = _terminalize(sequence, CFG_TERMINALIZATION)
    metadata = _build_metadata(
        variant="O",
        control_family="deterministic-cfg-with-deletion",
        derivation_strategy={
            "mode": "left-to-right breadth-style passes",
            "cycle_policy": "per-nonterminal deterministic production counter",
            "max_passes": 50,
            "stop_condition": (
                "pre_terminalization_observable_bits >= 3100000 and "
                "projected_terminalization_ratio <= 0.05"
            ),
            "passes_reached": pass_index,
            "production_counters": dict(sorted(counters.items())),
        },
        rule_set=_serialize_cfg_rules(),
        terminalization_map=_serialize_map(CFG_TERMINALIZATION),
        structural_stream=final_sequence,
        pre_terminalization_observable_bits=stats["pre_terminalization_observable_bits"],
        terminalization_observable_bits=terminalization_bits,
    )
    return GeneratedControl("O", "".join(final_sequence), metadata)


def _generate_lsystem_control() -> GeneratedControl:
    sequence = ["A"]
    round_index = 0

    while True:
        stats = _projected_stats(sequence, LSYSTEM_TERMINALIZATION)
        if _passes_generation_contract(stats):
            break
        if round_index >= 12:
            raise RuntimeError(
                "Variant P failed v0.3 contract: reached max_rounds=12 before target with clean terminalization."
            )

        sequence = [
            replacement
            for token in sequence
            for replacement in (LSYSTEM_RULES[token] if token in LSYSTEM_RULES else [token])
        ]
        round_index += 1

    final_sequence, terminalization_bits = _terminalize(sequence, LSYSTEM_TERMINALIZATION)
    metadata = _build_metadata(
        variant="P",
        control_family="deterministic-stratified-lsystem",
        derivation_strategy={
            "mode": "deterministic parallel rewrite",
            "active_symbols": sorted(LSYSTEM_RULES),
            "max_rounds": 12,
            "stop_condition": (
                "pre_terminalization_observable_bits >= 3100000 and "
                "projected_terminalization_ratio <= 0.05"
            ),
            "rounds_reached": round_index,
        },
        rule_set=_serialize_rules(LSYSTEM_RULES),
        terminalization_map=_serialize_map(LSYSTEM_TERMINALIZATION),
        structural_stream=final_sequence,
        pre_terminalization_observable_bits=stats["pre_terminalization_observable_bits"],
        terminalization_observable_bits=terminalization_bits,
    )
    return GeneratedControl("P", "".join(final_sequence), metadata)


def _build_metadata(
    *,
    variant: str,
    control_family: str,
    derivation_strategy: dict,
    rule_set: dict,
    terminalization_map: dict,
    structural_stream: list[str],
    pre_terminalization_observable_bits: int,
    terminalization_observable_bits: int,
) -> dict:
    final_observable_bits = _observable_count(structural_stream)
    zeros = structural_stream.count("0")
    ones = structural_stream.count("1")
    terminalization_ratio = _safe_ratio(terminalization_observable_bits, final_observable_bits)
    return {
        "variant": variant,
        "variant_code": variant,
        "control_family": control_family,
        "pre_registration_doc": CONTRACT_DOC,
        "pre_registration_sha256": CONTRACT_SHA256,
        "iteration": ITERATION_LABEL,
        "format": "v33_structural",
        "encoding": "2bit",
        "shared_terminal_blocks": dict(SHARED_TERMINAL_BLOCKS),
        "rule_set": rule_set,
        "derivation_strategy": derivation_strategy,
        "terminalization_map": terminalization_map,
        "target_observable_bits": TARGET_OBSERVABLE_BITS,
        "minimum_observable_bits": MIN_OBSERVABLE_BITS,
        "pre_terminalization_observable_bits": pre_terminalization_observable_bits,
        "terminalization_observable_bits": terminalization_observable_bits,
        "final_observable_bits": final_observable_bits,
        "observable_bits": final_observable_bits,
        "terminalization_ratio": terminalization_ratio,
        "freq_zero": _safe_ratio(zeros, final_observable_bits),
        "freq_one": _safe_ratio(ones, final_observable_bits),
        "structural_length": len(structural_stream),
        "density_band": [MIN_DENSITY, MAX_DENSITY],
        "terminalization_ratio_bands": {
            "clean_max": MAX_TERMINALIZATION_RATIO_CLEAN,
            "valid_max": MAX_TERMINALIZATION_RATIO_VALID,
        },
    }


def _passes_generation_contract(stats: dict) -> bool:
    return (
        stats["pre_terminalization_observable_bits"] >= TARGET_OBSERVABLE_BITS
        and stats["terminalization_ratio"] <= MAX_TERMINALIZATION_RATIO_CLEAN
    )


def _projected_stats(sequence: list[str], terminalization_map: dict[str, list[str]]) -> dict:
    pre_bits = _observable_count(sequence)
    term_bits = 0
    for token in sequence:
        replacement = terminalization_map.get(token)
        if replacement is not None:
            term_bits += _observable_count(replacement)
    final_bits = pre_bits + term_bits
    return {
        "pre_terminalization_observable_bits": pre_bits,
        "terminalization_observable_bits": term_bits,
        "projected_final_observable_bits": final_bits,
        "terminalization_ratio": _safe_ratio(term_bits, final_bits),
    }


def _terminalize(
    sequence: Iterable[str],
    terminalization_map: dict[str, list[str]],
) -> tuple[list[str], int]:
    final_sequence: list[str] = []
    terminalization_bits = 0
    for token in sequence:
        replacement = terminalization_map.get(token)
        if replacement is None:
            final_sequence.append(token)
        else:
            final_sequence.extend(replacement)
            terminalization_bits += _observable_count(replacement)
    return final_sequence, terminalization_bits


def _observable_count(sequence: Iterable[str]) -> int:
    return sum(1 for token in sequence if token in ("0", "1"))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _serialize_cfg_rules() -> dict:
    return {
        symbol: ["".join(production) if production else "epsilon" for production in productions]
        for symbol, productions in CFG_RULES.items()
    }


def _serialize_rules(rules: dict[str, list[str]]) -> dict[str, str]:
    return {symbol: "".join(production) for symbol, production in rules.items()}


def _serialize_map(mapping: dict[str, list[str]]) -> dict[str, str]:
    return {symbol: "".join(replacement) for symbol, replacement in mapping.items()}


def _validate_terminal_blocks() -> None:
    for name, block in SHARED_TERMINAL_BLOCKS.items():
        if len(block) != 64:
            raise RuntimeError(f"{name} must be exactly 64 bits.")
        if set(block) != {"0", "1"}:
            raise RuntimeError(f"{name} must contain both 0 and 1 only.")
        if block.count("0") != 32 or block.count("1") != 32:
            raise RuntimeError(f"{name} must be balanced 32/32.")


def _validate_metadata(metadata: dict) -> None:
    final_bits = int(metadata["final_observable_bits"])
    ratio = float(metadata["terminalization_ratio"])
    freq_zero = float(metadata["freq_zero"])
    freq_one = float(metadata["freq_one"])
    if final_bits < MIN_OBSERVABLE_BITS:
        raise RuntimeError(f"{metadata['variant']} generated only {final_bits} observable bits.")
    if ratio > MAX_TERMINALIZATION_RATIO_VALID:
        raise RuntimeError(
            f"{metadata['variant']} terminalization_ratio={ratio:.6f} exceeds {MAX_TERMINALIZATION_RATIO_VALID}."
        )
    if not (MIN_DENSITY <= freq_zero <= MAX_DENSITY and MIN_DENSITY <= freq_one <= MAX_DENSITY):
        raise RuntimeError(
            f"{metadata['variant']} density outside [{MIN_DENSITY}, {MAX_DENSITY}]: "
            f"freq_zero={freq_zero:.6f}, freq_one={freq_one:.6f}."
        )
    if metadata["pre_registration_sha256"] != CONTRACT_SHA256:
        raise RuntimeError("Pre-registration SHA mismatch.")


def _reuse_existing_snapshot(
    *,
    variant: str,
    struct_path: Path,
    metadata_path: Path,
) -> dict:
    if not struct_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Partial existing snapshot for var_{variant}. Use --force only after checking the directory."
        )
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read existing metadata for var_{variant}: {exc}") from exc

    actual_sha = sha256_file(struct_path)
    expected_sha = str(metadata.get("sha256_struct_gz", "")).upper()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Existing var_{variant} snapshot SHA mismatch. Use --force only if regenerating intentionally."
        )
    if metadata.get("pre_registration_sha256") != CONTRACT_SHA256:
        raise RuntimeError(
            f"Existing var_{variant} snapshot is not anchored to D-0087. Use --force only if regenerating intentionally."
        )
    _validate_metadata(metadata)
    return {
        "variant": variant,
        "status": "reused",
        "struct_path": str(struct_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "sha256_struct_gz": actual_sha,
        "metadata": metadata,
    }


def default_command() -> list[str]:
    return [Path(sys.argv[0]).name, *sys.argv[1:]]
