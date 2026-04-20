from __future__ import annotations

from datetime import datetime


def compact_int(value: int) -> str:
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def build_preflight_run_slug(
    *,
    variant: str,
    iteration: int | None,
    segment_bits: int,
    num_segments: int,
    m_min: int,
    m_max: int,
    nulls: list[str],
) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    null_part = "observed" if not nulls else "observed-" + "-".join(nulls)
    iter_part = f"iter-{iteration}" if iteration is not None else "iter-custom"
    return (
        f"preflight-fc__var-{variant}__{iter_part}__seg-{num_segments}x{compact_int(segment_bits)}"
        f"__m-{m_min}-{m_max}__ctrl-{null_part}__{timestamp}"
    )


def build_phase1_run_slug(
    *,
    variant: str,
    iteration: int | None,
    segment_bits: int,
    num_segments: int,
    segment_offset_bits: int = 0,
    scales: list[int],
    policies: list[str],
    null_model: str | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    iter_part = f"iter-{iteration}" if iteration is not None else "iter-custom"
    scale_part = "-".join(str(value) for value in scales)
    policy_part = "-".join(policies)
    offset_part = f"__off-{compact_int(segment_offset_bits)}" if segment_offset_bits > 0 else ""
    null_part = f"__null-{null_model}" if null_model else ""
    return (
        f"phase1-tower__var-{variant}__{iter_part}__seg-{num_segments}x{compact_int(segment_bits)}"
        f"{offset_part}__m-{scale_part}__proj-{policy_part}{null_part}__{timestamp}"
    )
