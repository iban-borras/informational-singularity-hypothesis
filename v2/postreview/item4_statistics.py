from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta


ALPHA = 0.05
NULL_COUNT = 511
HISTORICAL_SEEDS = (17, 101, 211, 307, 401, 503)
COORDINATES = ("R", "S", "A")


def validate_tail_vector(vector: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for coordinate in COORDINATES:
        value = float(vector[coordinate])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid tail coordinate {coordinate}={value!r}.")
        output[coordinate] = value
    return output


def pareto_exceeds(null: dict[str, Any], observed: dict[str, Any]) -> bool:
    null_vector = validate_tail_vector(null)
    observed_vector = validate_tail_vector(observed)
    return all(null_vector[key] >= observed_vector[key] for key in COORDINATES)


def clopper_pearson_interval(
    count: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    if total <= 0 or not 0 <= count <= total:
        raise ValueError("Clopper-Pearson requires 0 <= count <= total and total > 0.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if count == 0 else float(beta.ppf(tail, count, total - count + 1))
    upper = (
        1.0 if count == total else float(beta.ppf(1.0 - tail, count + 1, total - count))
    )
    return lower, upper


def coordinate_dispersion(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError(
            "Dispersion requires a non-empty finite one-dimensional sample."
        )
    median = float(np.median(array))
    return {
        "mean": float(np.mean(array)),
        "sample_variance": float(np.var(array, ddof=1)) if array.size > 1 else 0.0,
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "quantile_2_5": float(np.quantile(array, 0.025, method="linear")),
        "quantile_97_5": float(np.quantile(array, 0.975, method="linear")),
    }


def empirical_percentile_rank(values: Iterable[float], observed: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Percentile rank requires a non-empty one-dimensional sample.")
    value = float(observed)
    if not math.isfinite(value) or not bool(np.all(np.isfinite(array))):
        raise ValueError("Percentile rank requires finite values.")
    return float(np.count_nonzero(array <= value) / array.size)


def summarize_source_family(
    source: str,
    family: str,
    observed: dict[str, Any],
    null_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_vector = validate_tail_vector(observed)
    if len(null_rows) != NULL_COUNT:
        raise ValueError(
            f"{source}/{family} requires {NULL_COUNT} null vectors, got {len(null_rows)}."
        )
    seeds = [int(row["seed"]) for row in null_rows]
    if sorted(seeds) != list(range(1, NULL_COUNT + 1)) or len(set(seeds)) != NULL_COUNT:
        raise ValueError(
            f"{source}/{family} does not contain the frozen seed ledger 1..511."
        )

    vectors = [validate_tail_vector(row) for row in null_rows]
    count = sum(pareto_exceeds(row, observed_vector) for row in vectors)
    p_value = float((1 + count) / (NULL_COUNT + 1))
    interval = clopper_pearson_interval(count, NULL_COUNT)
    historical = [row for row in null_rows if int(row["seed"]) in HISTORICAL_SEEDS]
    historical_count = sum(pareto_exceeds(row, observed_vector) for row in historical)

    dispersion = {
        coordinate: coordinate_dispersion(row[coordinate] for row in vectors)
        for coordinate in COORDINATES
    }
    percentiles = {
        coordinate: empirical_percentile_rank(
            (row[coordinate] for row in vectors), observed_vector[coordinate]
        )
        for coordinate in COORDINATES
    }
    return {
        "source": source,
        "family": family,
        "observed": observed_vector,
        "null_count": NULL_COUNT,
        "pareto_exceedance_count": int(count),
        "p_value": p_value,
        "rank_resolution": 1.0 / (NULL_COUNT + 1),
        "clopper_pearson_95": {"lower": interval[0], "upper": interval[1]},
        "coordinate_dispersion": dispersion,
        "observed_percentile_ranks": percentiles,
        "historical_six_seed_subset": {
            "seeds": list(HISTORICAL_SEEDS),
            "pareto_exceedance_count": int(historical_count),
            "p_value": float((1 + historical_count) / (len(HISTORICAL_SEEDS) + 1)),
        },
    }


def holm_stepdown(
    rows: list[dict[str, Any]], *, alpha: float = ALPHA
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Holm step-down requires at least one test.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    ordered = sorted(
        rows,
        key=lambda row: (float(row["p_value"]), str(row["source"]), str(row["family"])),
    )
    total = len(ordered)
    running_adjusted = 0.0
    still_rejecting = True
    trace: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        raw = float(row["p_value"])
        if not 0.0 <= raw <= 1.0:
            raise ValueError(f"Invalid p-value: {raw!r}.")
        multiplier = total - index + 1
        threshold = alpha / multiplier
        running_adjusted = max(running_adjusted, min(1.0, multiplier * raw))
        reject = bool(still_rejecting and raw <= threshold)
        if not reject:
            still_rejecting = False
        trace.append(
            {
                "rank": index,
                "source": str(row["source"]),
                "family": str(row["family"]),
                "raw_p_value": raw,
                "holm_threshold": float(threshold),
                "holm_adjusted_p_value": float(running_adjusted),
                "reject": reject,
            }
        )
    return trace


def classify_outcome(
    *,
    derivable: bool,
    degenerate: bool,
    baseline_reproduced: bool,
    holm_trace: list[dict[str, Any]],
    sources: Iterable[str],
    families: Iterable[str],
) -> dict[str, Any]:
    source_order = tuple(sources)
    family_order = tuple(families)
    if not derivable:
        return {"outcome": "inputs-or-run-not-derivable", "source_passes": {}}
    if degenerate:
        return {"outcome": "gate1-tail-readout-degenerate", "source_passes": {}}
    if not baseline_reproduced:
        return {
            "outcome": "canonical-tail-baseline-not-reproduced",
            "source_passes": {},
        }

    decisions = {
        (str(row["source"]), str(row["family"])): bool(row["reject"])
        for row in holm_trace
    }
    expected = {(source, family) for source in source_order for family in family_order}
    if set(decisions) != expected:
        raise ValueError("Holm trace does not contain the complete source-family grid.")
    source_passes = {
        source: all(decisions[(source, family)] for family in family_order)
        for source in source_order
    }
    passing_sources = sum(source_passes.values())
    rejected_tests = sum(decisions.values())
    if passing_sources == len(source_order):
        outcome = "gate1-tail-multiseed-reinforced-across-tested-sources"
    elif passing_sources:
        outcome = "gate1-tail-multiseed-source-limited"
    elif rejected_tests:
        outcome = "gate1-tail-multiseed-null-family-sensitive"
    else:
        outcome = "gate1-tail-multiseed-not-reinforced"
    return {
        "outcome": outcome,
        "source_passes": source_passes,
        "passing_source_count": int(passing_sources),
        "rejected_test_count": int(rejected_tests),
    }
