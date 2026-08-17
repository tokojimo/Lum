"""Biological-replicate-aware directional inference.

Technical wells must be collapsed before calling this module.  Consequently,
every row supplied to the test represents one independent biological block.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


RESULT_COLUMNS = [
    "condition_1", "condition_2", "test", "alternative", "transformation",
    "value_column", "pairing_columns", "n_pairs", "statistic_t", "degrees_freedom",
    "mean_log10_difference", "p_raw", "p_holm", "holm_family_size", "alpha",
    "significance", "calculation_status", "non_calculable_reason",
    "paired_values_json",
]


def _canonical_condition(value: object) -> str:
    """Normalize a UI/figure condition without changing its scientific identity."""
    parts = str(value).split("\0", 1)
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
    comparisons: tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    """Test explicitly requested ``left > right`` hypotheses on log10 values.

    A paired, one-tailed t-test is calculated only for positive, complete
    biological pairs.  Holm adjustment treats all requested, estimable
    contrasts as one comparison family.  With no pre-specified hypotheses no
    inferential test is performed.
    """
    ids = [column for column in identity
           if column in biological and biological[column].notna().any()]
    ids = ids or [column for column in ("Groupe",) if column in biological]
    if not comparisons or not ids or condition not in biological or value not in biological:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    table = biological.pivot_table(index=ids, columns=condition, values=value, aggfunc="mean")
    available = list(table.columns)
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
                "test": "paired t-test", "alternative": "condition_1 > condition_2",
                "transformation": "log10", "value_column": value,
                "pairing_columns": "|".join(ids), "n_pairs": 0,
                "statistic_t": np.nan, "degrees_freedom": np.nan,
                "mean_log10_difference": np.nan, "p_raw": np.nan, "alpha": .05,
                "significance": "NA", "calculation_status": "non calculable",
                "non_calculable_reason": "; ".join(missing),
                "paired_values_json": "[]",
            })
            continue
        pairs = table[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
        pairs = pairs.loc[pairs[left].gt(0) & pairs[right].gt(0)]
        enough_pairs = len(pairs) >= 3
        if enough_pairs:
            test_result = ttest_rel(
                np.log10(pairs[left].to_numpy(float)),
                np.log10(pairs[right].to_numpy(float)),
                alternative="greater",
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
                "condition_1_log10": float(np.log10(pair_values[left])),
                "condition_2_log10": float(np.log10(pair_values[right])),
            })
        rows.append({
            "condition_1": left, "condition_2": right,
            "test": "paired t-test", "alternative": "condition_1 > condition_2",
            "transformation": "log10", "value_column": value,
            "pairing_columns": "|".join(ids), "n_pairs": len(pairs),
            "statistic_t": statistic,
            "degrees_freedom": len(pairs) - 1 if enough_pairs else float("nan"),
            "mean_log10_difference": float(
                (np.log10(pairs[left]) - np.log10(pairs[right])).mean()
            ) if len(pairs) else float("nan"),
            "p_raw": raw, "alpha": .05,
            "calculation_status": "calculé" if enough_pairs else "non calculable",
            "non_calculable_reason": "" if enough_pairs else (
                f"moins de 3 paires biologiques positives ({len(pairs)} disponible(s))"
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
    result["holm_family_size"] = total
    # Each explicitly selected contrast is a separately planned, pairwise
    # hypothesis.  Keep Holm available in the exported audit table, but report
    # the raw significance used by the figure rather than silently treating all
    # simultaneously displayed brackets as one inferential family.
    result["significance"] = result["p_raw"].map(_significance)
    return result[RESULT_COLUMNS]
