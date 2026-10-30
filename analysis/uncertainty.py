"""Reusable animal-level uncertainty summaries for comparison plots."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


CI95_COLUMNS = ["mean", "sem", "ci95_low", "ci95_high", "n", "n_strata"]


def _bootstrap_mean_ci(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Return a percentile CI after resampling animals within each stratum."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata, dtype=object)
    finite = np.isfinite(values)
    values = values[finite]
    strata = strata[finite]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    stratum_values = [values[strata == label] for label in pd.unique(strata)]
    boot_means = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        samples = [rng.choice(group, size=len(group), replace=True) for group in stratum_values]
        boot_means[index] = np.concatenate(samples).mean()

    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_animal_mean_ci(
    data: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Sequence[str],
    animal_col: str = "animal",
    strata_col: str = "dataset_key",
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 20261030,
) -> pd.DataFrame:
    """Summarize one-row-per-animal values with a stratified bootstrap CI.

    ``strata_col`` normally identifies a line/cohort dataset. Resampling within
    each observed stratum retains its animal count while keeping the animal,
    rather than its trials or sessions, as the unit of replication.
    """
    output_columns = [*group_cols, *CI95_COLUMNS]
    if data is None or data.empty:
        return pd.DataFrame(columns=output_columns)

    frame = data.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[value_col, animal_col]).copy()
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    if strata_col not in frame.columns:
        frame[strata_col] = "single_cohort"
    frame[strata_col] = frame[strata_col].astype("string").fillna("unknown")

    rows = []
    grouped = frame.groupby(list(group_cols), dropna=False, sort=True)
    for group_index, (keys, group) in enumerate(grouped):
        keys = keys if isinstance(keys, tuple) else (keys,)
        # Guard against accidental duplicate animal rows within a plotted point.
        animal_values = (
            group.groupby([strata_col, animal_col], dropna=False)[value_col]
            .mean()
            .reset_index()
        )
        values = animal_values[value_col].to_numpy(dtype=float)
        strata = animal_values[strata_col].to_numpy(dtype=object)
        mean = float(np.mean(values)) if values.size else np.nan
        sem = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1 else np.nan
        )
        low, high = _bootstrap_mean_ci(
            values,
            strata,
            n_boot=n_boot,
            confidence=confidence,
            seed=int(seed) + group_index,
        )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "mean": mean,
                "sem": sem,
                "ci95_low": low,
                "ci95_high": high,
                "n": int(values.size),
                "n_strata": int(pd.Series(strata).nunique()),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def ci95_yerr(frame: pd.DataFrame, *, mean_col: str = "mean") -> np.ndarray:
    """Return Matplotlib-compatible asymmetric 95% CI errors."""
    mean = pd.to_numeric(frame[mean_col], errors="coerce").to_numpy(dtype=float)
    if {"ci95_low", "ci95_high"}.issubset(frame.columns):
        low = pd.to_numeric(frame["ci95_low"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(frame["ci95_high"], errors="coerce").to_numpy(dtype=float)
        return np.vstack([
            np.maximum(0.0, mean - low),
            np.maximum(0.0, high - mean),
        ])

    # Compatibility for older prepared objects: visibly use an approximate
    # 95% interval rather than silently reverting to SEM.
    sem_source = frame["sem"] if "sem" in frame.columns else pd.Series(np.nan, index=frame.index)
    sem = pd.to_numeric(sem_source, errors="coerce").to_numpy(dtype=float)
    error = 1.96 * np.nan_to_num(sem, nan=0.0)
    return np.vstack([error, error])


def add_ci95_figure_note(fig) -> None:
    """Add the common uncertainty definition to a Matplotlib figure."""
    if any(getattr(text, "get_gid", lambda: None)() == "animal-ci95-note" for text in fig.texts):
        return
    note = fig.text(
        0.995,
        0.005,
        "Error bars/shading: 95% animal-level CI; pooled cohorts use cohort-stratified bootstrap",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    note.set_gid("animal-ci95-note")
