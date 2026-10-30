"""Reusable preparation helpers for animal-level chronometric functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


ABSOLUTE_CHRONOMETRIC_METHODS = {"side_average", "pooled_absolute"}


def absolute_chronometric_per_animal(
    data: pd.DataFrame,
    *,
    condition_cols: Sequence[str] = (),
    method: str = "side_average",
    require_both_sides: bool = True,
    animal_col: str = "animal",
    dataset_col: str = "dataset_key",
    ild_col: str = "ILD",
    value_col: str = "timed_rt",
) -> pd.DataFrame:
    """Return one mean RT per animal, condition, and absolute ILD.

    ``side_average`` first calculates separate animal means at signed ILDs and
    then gives the negative and positive sides equal weight at each magnitude.
    When ``require_both_sides`` is true, non-zero magnitudes without both signs
    are omitted; ILD zero remains valid as its own point.

    ``pooled_absolute`` reproduces the older calculation: trials are collapsed
    to absolute ILD before their mean is calculated, so the side with more
    trials receives more weight.
    """
    if method not in ABSOLUTE_CHRONOMETRIC_METHODS:
        allowed = ", ".join(sorted(ABSOLUTE_CHRONOMETRIC_METHODS))
        raise ValueError(f"method must be one of: {allowed}")

    condition_cols = list(condition_cols)
    required = [animal_col, ild_col, value_col, *condition_cols]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required chronometric columns: {missing}")

    frame = data.copy()
    if dataset_col not in frame.columns:
        frame[dataset_col] = "single_cohort"
    frame[dataset_col] = frame[dataset_col].astype("string").fillna("unknown")
    frame["_signed_ild"] = pd.to_numeric(frame[ild_col], errors="coerce")
    frame["_chronometric_value"] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(
        subset=[animal_col, "_signed_ild", "_chronometric_value", *condition_cols]
    ).copy()

    output_columns = [
        dataset_col,
        animal_col,
        *condition_cols,
        "abs_ILD",
        "mean_rt",
        "n_sides",
        "n_trials",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    identity_cols = [dataset_col, animal_col, *condition_cols]
    frame["abs_ILD"] = frame["_signed_ild"].abs()
    frame["_ild_side"] = np.sign(frame["_signed_ild"]).astype(int)

    if method == "pooled_absolute":
        result = (
            frame.groupby([*identity_cols, "abs_ILD"], observed=False, dropna=False)
            .agg(
                mean_rt=("_chronometric_value", "mean"),
                n_sides=("_ild_side", "nunique"),
                n_trials=("_chronometric_value", "size"),
            )
            .reset_index()
        )
    else:
        side_means = (
            frame.groupby(
                [*identity_cols, "_signed_ild", "abs_ILD", "_ild_side"],
                observed=False,
                dropna=False,
            )["_chronometric_value"]
            .agg(side_mean="mean", side_trials="size")
            .reset_index()
        )
        result = (
            side_means.groupby([*identity_cols, "abs_ILD"], observed=False, dropna=False)
            .agg(
                mean_rt=("side_mean", "mean"),
                n_sides=("_ild_side", "nunique"),
                n_trials=("side_trials", "sum"),
            )
            .reset_index()
        )

    if require_both_sides:
        result = result[result["abs_ILD"].eq(0) | result["n_sides"].ge(2)].copy()

    return result[output_columns].sort_values(
        [*identity_cols, "abs_ILD"], ignore_index=True
    )


def summarize_animal_chronometric(
    per_animal: pd.DataFrame,
    *,
    condition_cols: Sequence[str] = (),
    value_col: str = "mean_rt",
) -> pd.DataFrame:
    """Summarize animal chronometric means with between-animal SEM."""
    group_cols = [*condition_cols, "abs_ILD"]
    output_columns = [*group_cols, "mean", "sem", "n"]
    if per_animal is None or per_animal.empty:
        return pd.DataFrame(columns=output_columns)

    frame = per_animal.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[value_col, *group_cols])
    summary = (
        frame.groupby(group_cols, observed=False, dropna=False)[value_col]
        .agg(
            mean="mean",
            sem=lambda values: (
                values.std(ddof=1) / np.sqrt(values.count())
                if values.count() > 1
                else np.nan
            ),
            n="count",
        )
        .reset_index()
    )
    return summary[output_columns].sort_values(group_cols, ignore_index=True)
