from __future__ import annotations

import json
from pathlib import Path


def load_phase2_dataset(run_path: Path) -> dict:
    path = run_path if run_path.name == "dataset.json" else run_path / "dataset.json"
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def select_row(payload: dict, variant: str | None) -> dict:
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("Phase 2 dataset contains no rows.")
    if variant is None:
        if len(rows) != 1:
            available = ", ".join(row.get("variant", "?") for row in rows)
            raise ValueError(
                "Dataset contains multiple variants; specify one explicitly. "
                f"Available: {available}"
            )
        return rows[0]

    variant_lower = variant.lower()
    for row in rows:
        if str(row.get("variant", "")).lower() == variant_lower:
            return row
    available = ", ".join(row.get("variant", "?") for row in rows)
    raise ValueError(f"Variant {variant!r} not found in dataset. Available: {available}")


def build_rarity_report(observed_row: dict, null_row: dict) -> dict:
    policies = observed_row.get("policies", [])
    null_policies = null_row.get("policies", [])
    if policies != null_policies:
        raise ValueError("Observed and null rows do not share the same policy list.")

    window_scales = observed_row.get("window_scales", [])
    if window_scales != null_row.get("window_scales", []):
        raise ValueError("Observed and null rows do not share the same defect window.")

    policy_reports = {}
    for policy in policies:
        observed_weights = _family_weights(observed_row["policy_results"][policy])
        null_weights = _family_weights(null_row["policy_results"][policy])
        policy_reports[policy] = _compare_family_weights(observed_weights, null_weights)

    return {
        "observed_variant": observed_row.get("variant"),
        "null_variant": null_row.get("variant"),
        "observed_iteration": observed_row.get("iteration"),
        "null_iteration": null_row.get("iteration"),
        "window_scales": list(window_scales),
        "policies": list(policies),
        "policy_reports": policy_reports,
    }


def render_console_summary(report: dict) -> str:
    lines = [
        "Phase 2 null-conditioned defect rarity",
        "-" * 118,
        f"{'policy':<10}{'obs_fam':>10}{'null_fam':>10}{'shared':>10}{'jacc':>10}{'w_jacc':>10}{'rarity':>10}{'cov_obs':>10}{'null_exc':>10}",
    ]
    for policy in report["policies"]:
        item = report["policy_reports"][policy]
        lines.append(
            f"{policy:<10}"
            f"{item['observed_family_count']:>10}"
            f"{item['null_family_count']:>10}"
            f"{item['shared_family_count']:>10}"
            f"{_fmt(item['family_jaccard']):>10}"
            f"{_fmt(item['weighted_jaccard']):>10}"
            f"{_fmt(item['rarity_mass_fraction']):>10}"
            f"{_fmt(item['observed_mass_covered_fraction']):>10}"
            f"{_fmt(item['null_excess_mass_fraction']):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Phase 2 Null-Conditioned Defect Rarity",
        "",
        "## Selection",
        "",
        f"- Observed variant: {report['observed_variant']} @ iter {report['observed_iteration']}",
        f"- Null variant: {report['null_variant']} @ iter {report['null_iteration']}",
        f"- Defect window: {', '.join(str(value) for value in report['window_scales'])}",
        f"- Policies: {', '.join(report['policies'])}",
        "",
        "## Compact Readout",
        "",
        "| Policy | Observed fam. | Null fam. | Shared fam. | Family Jaccard | Weighted Jaccard | Observed rarity mass | Observed mass covered by null | Null excess mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in report["policies"]:
        item = report["policy_reports"][policy]
        lines.append(
            f"| {policy} | {item['observed_family_count']} | {item['null_family_count']} | {item['shared_family_count']} | "
            f"{_fmt(item['family_jaccard'])} | {_fmt(item['weighted_jaccard'])} | {_fmt(item['rarity_mass_fraction'])} | "
            f"{_fmt(item['observed_mass_covered_fraction'])} | {_fmt(item['null_excess_mass_fraction'])} |"
        )

    for policy in report["policies"]:
        item = report["policy_reports"][policy]
        lines.extend(
            [
                "",
                f"## Policy `{policy}`",
                "",
                f"- Observed family count: {item['observed_family_count']}",
                f"- Null family count: {item['null_family_count']}",
                f"- Shared family count: {item['shared_family_count']}",
                f"- Family Jaccard: {_fmt(item['family_jaccard'])}",
                f"- Weighted Jaccard: {_fmt(item['weighted_jaccard'])}",
                f"- Observed rarity mass fraction: {_fmt(item['rarity_mass_fraction'])}",
                f"- Observed mass covered by null: {_fmt(item['observed_mass_covered_fraction'])}",
                f"- Null excess mass fraction: {_fmt(item['null_excess_mass_fraction'])}",
                "",
                "Top observed-rare families:",
            ]
        )
        if item["top_rare_families"]:
            for family in item["top_rare_families"]:
                lines.append(
                    f"- `{family['family']}` | obs={_fmt(family['observed_weight'])} | "
                    f"null={_fmt(family['null_weight'])} | excess={_fmt(family['rarity_weight'])}"
                )
        else:
            lines.append("- none")

        lines.append("")
        lines.append("Top shared families:")
        if item["top_shared_families"]:
            for family in item["top_shared_families"]:
                lines.append(
                    f"- `{family['family']}` | obs={_fmt(family['observed_weight'])} | "
                    f"null={_fmt(family['null_weight'])} | shared={_fmt(family['shared_weight'])}"
                )
        else:
            lines.append("- none")

    return "\n".join(lines)


def _family_weights(policy_result: dict) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in policy_result.get("parent_results", []):
        family = item.get("defect_family")
        if not family:
            continue
        weights[family] = weights.get(family, 0.0) + float(item.get("parent_weight", 0.0) or 0.0)
    return weights


def _compare_family_weights(observed: dict[str, float], null: dict[str, float]) -> dict:
    observed_set = set(observed)
    null_set = set(null)
    shared = observed_set & null_set
    union = observed_set | null_set

    observed_total = sum(observed.values())
    null_total = sum(null.values())
    shared_mass = sum(min(observed[key], null[key]) for key in shared)
    rarity_mass = sum(max(observed.get(key, 0.0) - null.get(key, 0.0), 0.0) for key in union)
    null_excess_mass = sum(max(null.get(key, 0.0) - observed.get(key, 0.0), 0.0) for key in union)
    weighted_union = sum(max(observed.get(key, 0.0), null.get(key, 0.0)) for key in union)

    top_rare_families = [
        {
            "family": family,
            "observed_weight": observed.get(family, 0.0),
            "null_weight": null.get(family, 0.0),
            "rarity_weight": max(observed.get(family, 0.0) - null.get(family, 0.0), 0.0),
        }
        for family in sorted(
            union,
            key=lambda key: (
                -max(observed.get(key, 0.0) - null.get(key, 0.0), 0.0),
                -observed.get(key, 0.0),
                key,
            ),
        )
        if observed.get(family, 0.0) > null.get(family, 0.0)
    ][:10]

    top_shared_families = [
        {
            "family": family,
            "observed_weight": observed[family],
            "null_weight": null[family],
            "shared_weight": min(observed[family], null[family]),
        }
        for family in sorted(
            shared,
            key=lambda key: (-min(observed[key], null[key]), key),
        )
    ][:10]

    return {
        "observed_family_count": len(observed_set),
        "null_family_count": len(null_set),
        "shared_family_count": len(shared),
        "family_jaccard": _safe_ratio(len(shared), len(union)),
        "weighted_jaccard": _safe_ratio(shared_mass, weighted_union),
        "observed_mass_covered_fraction": _safe_ratio(shared_mass, observed_total),
        "rarity_mass_fraction": _safe_ratio(rarity_mass, observed_total),
        "null_excess_mass_fraction": _safe_ratio(null_excess_mass, null_total),
        "top_rare_families": top_rare_families,
        "top_shared_families": top_shared_families,
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
