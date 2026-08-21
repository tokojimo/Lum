"""Biological-replicate-aware directional inference.

Technical wells must be collapsed before calling this module.  Consequently,
every row supplied to the test represents one independent biological block.
"""

from __future__ import annotations

import json
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


RESULT_COLUMNS = [
    "condition_1", "condition_2", "test", "test_name", "alternative", "transformation",
    "statistical_transform", "paired", "value_column", "pairing_columns", "n", "n_pairs",
    "n_total", "n_used", "n_excluded_nonpositive", "n_excluded_nonfinite",
    "statistic", "statistic_t", "degrees_freedom", "mean_transformed_difference",
    "mean_log10_difference", "p_raw", "p_adjusted", "p_holm",
    "multiple_testing_method", "holm_family_size", "alpha",
    "significance", "calculation_status", "non_calculable_reason",
    "paired_values_json",
]

STATISTICAL_TRANSFORMS = ("none", "log10")


def transform_for_statistics(values, transform: str = "log10") -> np.ndarray:
    """Transform finite test inputs without filtering or adding an offset."""
    if transform not in STATISTICAL_TRANSFORMS:
        raise ValueError(f"Unknown statistical transform: {transform!r}.")
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Statistical values must be finite.")
    if transform == "log10":
        if np.any(result <= 0):
            raise ValueError("log10 statistical values must be strictly positive.")
        return np.log10(result)
    return result.copy()


def all_pairwise_comparisons(conditions) -> tuple[tuple[object, object], ...]:
    """Return every unordered pair once, preserving condition order."""
    unique = list(dict.fromkeys(conditions))
    return tuple(combinations(unique, 2))


def _paired_value_table(
    biological: pd.DataFrame, ids: list[str], condition: str, value: str,
) -> pd.DataFrame:
    """Pivot values without grouping NUL-delimited condition strings.

    pandas' hash grouping treats NUL as a C-string terminator in some code
    paths.  Group each exact condition subset separately so ``A\0M1`` and
    ``A\0M2`` can never collapse into the same pivot column.
    """
    columns = []
    for item in dict.fromkeys(biological[condition]):
        selected = biological.loc[biological[condition].map(lambda candidate: candidate == item)]
        series = selected.groupby(ids, dropna=False)[value].mean().rename(item)
        columns.append(series)
    if not columns:
        return pd.DataFrame()
    result = pd.concat(columns, axis=1)
    # A list of 2-tuples is normally promoted to a pandas MultiIndex.  Force a
    # plain object Index: the tuple is one atomic scientific condition.
    result.columns = pd.Index(np.asarray([series.name for series in columns], dtype=object))
    return result


def _biological_identity(biological: pd.DataFrame,
                         candidates: tuple[str, ...]) -> list[str]:
    """Select one real biological identity, in explicit priority order."""
    for column in candidates:
        if column in biological and biological[column].notna().any():
            return [column]
    return []


def directional_test_diagnostics(
    biological: pd.DataFrame, *, value: str, condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "experience", "biological_replicate_id"),
    comparisons: tuple[tuple[str, str], ...] = (),
) -> list[dict[str, object]]:
    """Capture the exact inputs presented to the directional test.

    This intentionally performs no matching or statistical inference.  It is
    an audit aid for the Streamlit interface, built from the same biological
    table, condition column, and identity candidates as the real test.
    """
    ids = _biological_identity(biological, identity)
    pivot_columns: list[object] = []
    if ids and condition in biological and value in biological:
        pivot_columns = list(_paired_value_table(biological, ids, condition, value).columns)

    unique_values = {
        column: list(pd.unique(biological[column].dropna())) if column in biological else []
        for column in ("souche", "Groupe", "_comparison")
    }
    row_columns = [
        column for column in
        ("experience", "experience_id", "souche", "Groupe", "_comparison",
         "lum_norm_peak", "lum_norm_auc")
        if column in biological
    ]
    diagnostics = []
    table = (_paired_value_table(biological, ids, condition, value)
             if ids and value in biological else pd.DataFrame())
    canonical_columns = {_canonical_condition(column): column for column in table.columns}
    for requested_left, requested_right in comparisons:
        left = canonical_columns.get(_canonical_condition(requested_left))
        right = canonical_columns.get(_canonical_condition(requested_right))
        if left is not None and right is not None:
            pair_table = pd.concat([table[left], table[right]], axis=1)
            pair_table.columns = ["A", "B"]
            pair_table = pair_table.reset_index()
        else:
            pair_table = pd.DataFrame()
        complete = pair_table[["A", "B"]].notna().all(axis=1) if not pair_table.empty else pd.Series(dtype=bool)
        finite = (np.isfinite(pair_table[["A", "B"]].to_numpy(float)).all(axis=1)
                  if not pair_table.empty else np.array([], dtype=bool))
        positive = ((pair_table["A"] > 0) & (pair_table["B"] > 0)
                    if not pair_table.empty else pd.Series(dtype=bool))
        diagnostics.append({
        "requested_left": requested_left,
        "canonical_left": _canonical_condition(requested_left),
        "requested_right": requested_right,
        "canonical_right": _canonical_condition(requested_right),
        "condition_column": condition,
        "identity_columns": list(ids),
        "pivot_columns": [
            {"value": column, "repr": repr(column),
             "canonical": _canonical_condition(column)}
            for column in pivot_columns
        ],
        "unique_values": unique_values,
        "biological_rows": biological[row_columns].copy(),
        "pair_table": pair_table,
        "condition_a": left,
        "condition_b": right,
        "complete_pairs": int(complete.sum()),
        "finite_pairs": int(np.sum(complete.to_numpy() & finite)) if len(complete) else 0,
        "positive_pairs": int(np.sum(complete.to_numpy() & finite & positive.to_numpy())) if len(complete) else 0,
        "only_a": pair_table.loc[pair_table["A"].notna() & pair_table["B"].isna(), ids].to_dict("records") if not pair_table.empty else [],
        "only_b": pair_table.loc[pair_table["A"].isna() & pair_table["B"].notna(), ids].to_dict("records") if not pair_table.empty else [],
    })
    return diagnostics


def _canonical_condition(value: object) -> str:
    """Normalize a UI/figure condition without changing its scientific identity."""
    parts = list(value) if isinstance(value, tuple) and len(value) == 2 else str(value).split("\0", 1)
    normalized = []
    for index, part in enumerate(parts):
        text = " ".join(part.strip().split())
        if index == 0:
            # Guided comparisons may have been selected from imported plate
            # labels (``14.1Ac attB::PspeD2-1A-lux``), while an already-built
            # metric table contains the reporter construct only
            # (``PspeD2-1A-lux``).  Both name the same plotted reporter.  Keep
            # arbitrary strain names intact, but remove the well-known host /
            # integration prefix and MiniCTX wrapper when they are present.
            _prefix, separator, construct = text.rpartition("::")
            if separator:
                text = construct.strip()
            wrapped = re.fullmatch(r"MiniCTXlux\s*\((.+)\)", text,
                                   flags=re.IGNORECASE)
            if wrapped:
                text = wrapped.group(1).strip()
            text = re.sub(r"-lux$", "-lux", text, flags=re.IGNORECASE)
        if index == 1:
            text = re.sub(
                r"^exp(?:eriment)?\s*\d+\s*\|\s*", "", text,
                flags=re.IGNORECASE,
            )
        normalized.append(text.casefold())
    return "\0".join(normalized)


def _significance(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < .0001:
        return "****"
    if p_value < .001:
        return "***"
    if p_value < .01:
        return "**"
    if p_value < .05:
        return "*"
    return "ns"


def paired_directional_t_tests(
    biological: pd.DataFrame, *, value: str, condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "experience", "biological_replicate_id"),
    comparisons: tuple[tuple[str, str], ...] | None = None, transform: str = "log10",
    alternative: str = "two-sided", multiple_testing_method: str = "Holm",
) -> pd.DataFrame:
    """Test pairs using an explicit transform, independently of plot settings.

    Complete finite pairs are required. With log10, non-positive pairs are
    excluded, preserving Lum's historical rule. Holm treats all estimable
    contrasts as one comparison family. Passing an empty tuple explicitly
    disables inference; omitting comparisons tests every unordered pair once.
    The historical directional test remains available with ``alternative="greater"``.
    """
    if transform not in STATISTICAL_TRANSFORMS:
        raise ValueError(f"Unknown statistical transform: {transform!r}.")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("Invalid t-test alternative.")
    if multiple_testing_method.casefold() != "holm":
        raise ValueError("Only Holm correction is supported.")
    ids = _biological_identity(biological, identity)
    if not ids or condition not in biological or value not in biological:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    table = _paired_value_table(biological, ids, condition, value)
    available = list(table.columns)
    if comparisons is None:
        comparisons = all_pairwise_comparisons(available)
    if not comparisons:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    canonical_available: dict[str, list[object]] = {}
    for item in available:
        canonical_available.setdefault(_canonical_condition(item), []).append(item)
    rows: list[dict[str, object]] = []
    for requested_left, requested_right in comparisons:
        left_matches = canonical_available.get(_canonical_condition(requested_left), [])
        right_matches = canonical_available.get(_canonical_condition(requested_right), [])
        left = requested_left if requested_left in table else (left_matches[0] if len(left_matches) == 1 else None)
        right = requested_right if requested_right in table else (right_matches[0] if len(right_matches) == 1 else None)
        if left is None or right is None or left == right:
            missing = []
            if left is None:
                missing.append(f"condition A introuvable ({requested_left})")
            if right is None:
                missing.append(f"condition B introuvable ({requested_right})")
            if left is not None and left == right:
                missing.append("les deux conditions correspondent à la même boîte")
            rows.append({
                "condition_1": requested_left, "condition_2": requested_right,
                "test": "paired t-test", "test_name": "paired_t_test", "alternative": alternative,
                "transformation": transform, "statistical_transform": transform, "paired": True,
                "value_column": value, "pairing_columns": "|".join(ids), "n": 0, "n_pairs": 0,
                "n_total": 0, "n_used": 0, "n_excluded_nonpositive": 0,
                "n_excluded_nonfinite": 0, "statistic": np.nan, "statistic_t": np.nan,
                "degrees_freedom": np.nan, "mean_transformed_difference": np.nan,
                "mean_log10_difference": np.nan, "p_raw": np.nan, "p_adjusted": np.nan,
                "p_holm": np.nan, "multiple_testing_method": "Holm", "holm_family_size": 0,
                "alpha": .05,
                "significance": "NA", "calculation_status": "non calculable",
                "non_calculable_reason": "; ".join(missing),
                "paired_values_json": "[]",
            })
            continue
        candidates = table[[left, right]]
        finite = np.isfinite(candidates.to_numpy(float)).all(axis=1)
        n_nonfinite = int((~finite).sum())
        pairs = candidates.loc[finite]
        positive = pairs[left].gt(0) & pairs[right].gt(0)
        n_nonpositive = int((~positive).sum()) if transform == "log10" else 0
        if transform == "log10":
            pairs = pairs.loc[positive]
        transformed_left = transform_for_statistics(pairs[left], transform)
        transformed_right = transform_for_statistics(pairs[right], transform)
        enough_pairs = len(pairs) >= 3
        if enough_pairs:
            test_result = ttest_rel(
                transformed_left, transformed_right, alternative=alternative,
            )
            statistic = float(test_result.statistic)
            raw = float(test_result.pvalue)
        else:
            statistic = raw = float("nan")
        paired_records = []
        for pair_identity, pair_values in pairs.iterrows():
            identity_values = pair_identity if isinstance(pair_identity, tuple) else (pair_identity,)
            paired_records.append({
                **{column: value for column, value in zip(ids, identity_values)},
                "condition_1_value": float(pair_values[left]),
                "condition_2_value": float(pair_values[right]),
                "condition_1_transformed": float(transformed_left[len(paired_records)]),
                "condition_2_transformed": float(transformed_right[len(paired_records)]),
                **({
                    "condition_1_log10": float(transformed_left[len(paired_records)]),
                    "condition_2_log10": float(transformed_right[len(paired_records)]),
                } if transform == "log10" else {}),
            })
        rows.append({
            "condition_1": left, "condition_2": right,
            "test": "paired t-test", "test_name": "paired_t_test", "alternative": alternative,
            "transformation": transform, "statistical_transform": transform, "paired": True,
            "value_column": value, "pairing_columns": "|".join(ids), "n": len(pairs),
            "n_pairs": len(pairs), "n_total": len(candidates), "n_used": len(pairs),
            "n_excluded_nonpositive": n_nonpositive, "n_excluded_nonfinite": n_nonfinite,
            "statistic": statistic, "statistic_t": statistic,
            "degrees_freedom": len(pairs) - 1 if enough_pairs else float("nan"),
            "mean_transformed_difference": float(np.mean(transformed_left - transformed_right)) if len(pairs) else float("nan"),
            "mean_log10_difference": float(np.mean(transformed_left - transformed_right)) if len(pairs) and transform == "log10" else float("nan"),
            "p_raw": raw, "p_adjusted": np.nan, "p_holm": np.nan,
            "multiple_testing_method": "Holm", "holm_family_size": 0, "alpha": .05,
            "calculation_status": "calculé" if enough_pairs else "non calculable",
            "non_calculable_reason": "" if enough_pairs else (
                f"moins de 3 paires biologiques{' positives' if transform == 'log10' else ' complètes'} ({len(pairs)} disponible(s))"
            ),
            "paired_values_json": json.dumps(paired_records, ensure_ascii=False, default=str),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    estimable = result["p_raw"].dropna()
    order = estimable.sort_values().index
    adjusted = pd.Series(index=result.index, dtype=float)
    running = 0.0
    total = len(estimable)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, result.at[index, "p_raw"] * (total - rank)))
        adjusted.at[index] = running
    result["p_holm"] = adjusted
    result["p_adjusted"] = adjusted
    result["holm_family_size"] = total
    # Each explicitly selected contrast is a separately planned, pairwise
    # hypothesis.  Keep Holm available in the exported audit table, but report
    # the raw significance used by the figure rather than silently treating all
    # simultaneously displayed brackets as one inferential family.
    result["significance"] = result["p_raw"].map(_significance)
    return result[RESULT_COLUMNS]
