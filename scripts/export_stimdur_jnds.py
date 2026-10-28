"""Export individual JNDs by animal, ABL, and stimulus duration.

This module contains the code used for the Notebook 4 JND export:

* restricts the source data to one session type (default: session type 2),
* treats ``short_duration == 0`` or ``stim_dur == 6000`` as RT,
* fits one ``my_psycho`` curve per animal and ABL,
* computes lapse-normalized JNDs from the fitted asymptotes,
* records the fitted parameters and flags slope-bound fits,
* checks that there is only one row per view/animal/ABL/duration.

The psychometric model and JND calculation are imported from the shared
project modules, so this exporter stays consistent with the notebooks.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from analysis import psychometric as Psychometric
import Helpers.DataHelpers as DataHelpers


DEFAULT_MODEL = "my_psycho"
DEFAULT_SESSION_TYPE = 2
DEFAULT_SKIP_ABL = 50
DEFAULT_SLOPE_LOWER_BOUND = 0.1


def _duration_mask(
    df: pd.DataFrame,
    duration_value: int,
) -> pd.Series:
    """Return the Notebook 4 selection mask for one duration."""
    if duration_value == 0:
        # RT is represented by short_duration=0 and, where available,
        # by the raw stim_dur value 6000.
        mask = pd.to_numeric(
            df["short_duration"],
            errors="coerce",
        ).eq(0)

        if "stim_dur" in df.columns:
            mask |= pd.to_numeric(
                df["stim_dur"],
                errors="coerce",
            ).eq(6000)

        return mask

    return pd.to_numeric(
        df["short_duration"],
        errors="coerce",
    ).eq(duration_value)


def export_jnds(
    *,
    df: pd.DataFrame,
    views: Sequence[Any],
    duration_values: Iterable[int],
    output_csv: str | Path,
    session_type: int = DEFAULT_SESSION_TYPE,
    skip_abl: int | None = DEFAULT_SKIP_ABL,
    model: str = DEFAULT_MODEL,
    slope_lower_bound: float = DEFAULT_SLOPE_LOWER_BOUND,
) -> pd.DataFrame:
    """Fit and export individual JNDs.

    Parameters
    ----------
    df:
        DataFrame containing the merged trials.
    views:
        View objects with ``name`` and ``selector(DataFrame)`` attributes,
        as used by Notebook 4.
    duration_values:
        Stimulus-duration values. Use ``0`` for RT.
    output_csv:
        Destination CSV path.
    session_type:
        Session type retained before fitting. Defaults to 2.
    skip_abl:
        ABL excluded from JND calculation. ``50`` matches the notebook.
        Use ``None`` to include ABL 50.
    model:
        Psychometric model passed to the shared fitting function.
    slope_lower_bound:
        Lower slope bound used to flag boundary-limited JNDs.
    """
    required = {"animal", "ABL", "short_duration", "session_type"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    source = df.copy()
    source["animal"] = source["animal"].astype(str)
    source["ABL"] = pd.to_numeric(source["ABL"], errors="coerce")
    source["short_duration"] = pd.to_numeric(
        source["short_duration"],
        errors="coerce",
    )
    source["session_type"] = pd.to_numeric(
        source["session_type"],
        errors="coerce",
    )

    # Notebook 4 JND export uses session type 2 only.
    source = source[source["session_type"].eq(session_type)].copy()

    rows: list[dict[str, Any]] = []

    for view in views:
        view_df = view.selector(source.copy())
        if view_df.empty:
            continue

        for raw_duration in duration_values:
            duration_value = int(raw_duration)
            selected = view_df.loc[
                _duration_mask(view_df, duration_value)
            ].copy()

            if selected.empty:
                continue

            # Fit across all sessions belonging to each animal. This avoids
            # emitting one row per session when a view contains one animal.
            for animal, animal_df in selected.groupby(
                "animal",
                sort=True,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    fits = Psychometric.compute_psychometrics_by_ABL(
                        animal_df,
                        model=model,
                    )
                    jnd_df = DataHelpers.compute_jnd_by_ABL(
                        fits,
                        skip_ABL=skip_abl,
                    )

                if jnd_df is None or jnd_df.empty:
                    continue

                for _, fit in jnd_df.iterrows():
                    slope_a = float(fit["a"])
                    rows.append(
                        {
                            "view": str(view.name),
                            "animal": str(animal),
                            "ABL": int(fit["ABL"]),
                            "stim_duration": (
                                "RT"
                                if duration_value == 0
                                else str(duration_value)
                            ),
                            "stim_duration_value": duration_value,
                            "JND": float(fit["JND"]),
                            "slope_a": slope_a,
                            "bias_b": float(fit["b"]),
                            "lower_c": float(fit["c"]),
                            "upper_d": float(fit["d"]),
                            "slope_at_lower_bound": bool(
                                np.isclose(
                                    slope_a,
                                    slope_lower_bound,
                                    rtol=1e-8,
                                    atol=1e-10,
                                )
                            ),
                        }
                    )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No JND estimates were generated.")

    key_columns = [
        "view",
        "animal",
        "ABL",
        "stim_duration",
    ]
    duplicates = result[result.duplicated(key_columns, keep=False)]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate view/animal/ABL/duration rows found:\n"
            + duplicates.to_string(index=False)
        )

    result = result.sort_values(
        ["view", "animal", "stim_duration_value", "ABL"]
    ).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


__all__ = ["export_jnds"]
