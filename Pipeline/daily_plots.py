from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import Helpers.DataHelpers as DataHelpers
from analysis import psychometric as Psychometric


def _normalize_stakes_sound_ramp_safe(df: pd.DataFrame) -> pd.DataFrame:
    normalize_ramp = getattr(DataHelpers, "normalize_stakes_sound_ramp", None)
    if normalize_ramp is not None:
        return normalize_ramp(df)
    out = df.copy()
    if "sound_ramp_time" in out.columns:
        ramp = pd.to_numeric(out["sound_ramp_time"], errors="coerce")
        out["sound_ramp_time"] = ramp.fillna(0.005)
    return out
from analysis.daily_merge import get_base_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = ROOT / "DataFiles" / "Old Data" / "ILD_task"


def apply_daily_plot_style() -> None:
    mpl.rcParams.update({
        "savefig.pad_inches": 0.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "axes.linewidth": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def style_axes(
    ax,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    title_fs=22,
    label_fs=22,
    tick_fs=20,
    title_pad=16,
    box_aspect=1,
):
    if title:
        ax.set_title(title, fontsize=title_fs, pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fs, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fs, color="black")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fs,
        colors="black",
        width=1.5,
        length=6,
    )
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_reference_data(
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> dict[str, Any]:
    reference_dir = Path(reference_dir)
    makefig1_data = load_pickle(reference_dir / "fig1_plot_data.pkl")
    makefig1_chrono = load_pickle(reference_dir / "fig1_chrono_plot_data.pkl")
    makefig1_data = DataHelpers.normalize_ABL_labels(makefig1_data)
    return {
        "psychometric": makefig1_data,
        "chrono": makefig1_chrono,
    }


def load_daily_animal_data(
    *,
    subject_file: str,
    line: str,
    cohort: str,
) -> tuple[pd.DataFrame, Path]:
    data_dir = Path(get_base_dir(line, cohort))
    path = data_dir / subject_file
    if not path.exists():
        raise FileNotFoundError(f"Subject file not found: {path}")
    return pd.read_csv(path), data_dir


def load_stakes_group_data(
    *,
    cohorts: list[str] | tuple[str, ...] = ("cohort2",),
    dataset_selections: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    selected_cohorts = list(dataset_selections) if dataset_selections is not None else list(cohorts)
    if not selected_cohorts:
        raise ValueError("At least one Stakes cohort must be selected.")

    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for cohort in selected_cohorts:
        base_dir = Path(get_base_dir("Stakes", cohort))
        path = base_dir / "merged_all_subjects.csv"
        if not path.exists():
            raise FileNotFoundError(f"Group file not found: {path}")
        df = pd.read_csv(path)
        df["line"] = "Stakes"
        df["cohort"] = cohort
        df["dataset_key"] = f"Stakes:{cohort}"
        frames.append(df)
        loaded.append(cohort)

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df_all, loaded


def _has_psychometric_fit(res: dict[str, Any] | None) -> bool:
    if not res:
        return False
    return (
        res.get("pars") is not None
        and res.get("xx") is not None
        and res.get("yy") is not None
    )


def _has_any_psychometric_fit(results: dict[Any, dict[str, Any]]) -> bool:
    return any(_has_psychometric_fit(res) for res in results.values())


def _psychometric_ticks_and_limits(results: dict[Any, dict[str, Any]]) -> tuple[list[float], tuple[float, float]]:
    point_xs: list[float] = []
    curve_xs: list[float] = []

    for res in results.values():
        ilds = res.get("ILDs")
        if ilds is not None:
            point_xs.extend(DataHelpers.shift_ILD_for_ABL50(ilds).tolist())
        if _has_psychometric_fit(res):
            curve_xs.extend(DataHelpers.shift_ILD_for_ABL50(res["xx"]).tolist())

    if not point_xs and not curve_xs:
        return [-18, 0, 18], (-19, 19)

    all_xs = point_xs + curve_xs
    x_min = min(all_xs)
    x_max = max(all_xs)
    x_pad = max(1.0, 0.05 * (x_max - x_min if x_max != x_min else 2.0))

    ticks = sorted(set(point_xs) | {0.0})
    return ticks, (x_min - x_pad, x_max + x_pad)


def _format_psychometric_tick(x: float) -> str:
    if np.isclose(x, -18):
        return "-50"
    if np.isclose(x, 18):
        return "50"
    if float(x).is_integer():
        return str(int(x))
    return f"{x:g}"


def _display_abl_label(abl: float | int, ilds=None) -> str:
    abl_num = pd.to_numeric(pd.Series([abl]), errors="coerce").iloc[0]
    if pd.isna(abl_num):
        return str(abl)

    return f"{float(abl_num):g}"


def _choice_right_series(df: pd.DataFrame) -> pd.Series:
    resp = pd.to_numeric(df.get("response_poke"), errors="coerce")
    if resp.dropna().isin([2, 3]).any():
        return pd.Series(
            np.where(resp == 3, 1.0, np.where(resp == 2, 0.0, np.nan)),
            index=df.index,
            dtype=float,
        )
    if resp.dropna().isin([-1, 1]).any():
        return pd.Series(
            np.where(resp == 1, 1.0, np.where(resp == -1, 0.0, np.nan)),
            index=df.index,
            dtype=float,
        )
    return pd.Series(np.nan, index=df.index, dtype=float)


def _sem(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return np.nan
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _match_numeric_filter(
    values: pd.Series,
    targets: int | float | list[int | float] | tuple[int | float, ...] | None,
    *,
    atol: float = 1e-9,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if targets is None:
        return pd.Series(True, index=values.index, dtype=bool)

    target_list = targets if isinstance(targets, (list, tuple, set)) else [targets]
    wanted = pd.to_numeric(pd.Series(list(target_list)), errors="coerce").dropna().to_numpy(dtype=float)
    if wanted.size == 0:
        return pd.Series(False, index=values.index, dtype=bool)

    mask = pd.Series(False, index=values.index, dtype=bool)
    for target in wanted:
        mask |= np.isclose(numeric, target, atol=atol, rtol=0.0)
    return mask


def _apply_training_and_abl_filters(
    df: pd.DataFrame,
    *,
    training_level: int | None = None,
    training_level_min: int | None = None,
    training_level_max: int | None = None,
    abl_filter: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    sound_ramp_filter: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    sound_ramp_exclude: int | float | list[int | float] | tuple[int | float, ...] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    training_num = pd.to_numeric(out["training_level"], errors="coerce")

    if training_level is not None:
        out = out[training_num == training_level].copy()
    else:
        if training_level_min is not None:
            out = out[training_num >= training_level_min].copy()
            training_num = pd.to_numeric(out["training_level"], errors="coerce")
        if training_level_max is not None:
            out = out[training_num < training_level_max].copy()

    if abl_filter is not None and "ABL" in out.columns:
        out = out[_match_numeric_filter(out["ABL"], abl_filter)].copy()

    if sound_ramp_filter is not None and "sound_ramp_time" in out.columns:
        out = out[_match_numeric_filter(out["sound_ramp_time"], sound_ramp_filter)].copy()

    if sound_ramp_exclude is not None and "sound_ramp_time" in out.columns:
        out = out[~_match_numeric_filter(out["sound_ramp_time"], sound_ramp_exclude)].copy()

    return out


def _prepare_sound_ramp_comparison_data(prepared: dict[str, Any]) -> dict[str, Any]:
    df = prepared["df_last"].copy()
    if df.empty or "sound_ramp_time" not in df.columns or "ABL" not in df.columns:
        return {
            "df": pd.DataFrame(),
            "eligible_sessions": [],
            "rt_summary": pd.DataFrame(),
            "prop_summary": pd.DataFrame(),
            "abls": [],
        }

    df["sound_ramp_time_num"] = pd.to_numeric(df["sound_ramp_time"], errors="coerce")
    session_col = pd.to_numeric(df.get("session"), errors="coerce")
    df = df.assign(session_num=session_col)

    ramp_counts = (
        df.dropna(subset=["session_num"])
        .groupby("session_num", observed=False)["sound_ramp_time_num"]
        .nunique(dropna=True)
    )
    eligible_sessions = sorted(ramp_counts[ramp_counts > 1].index.tolist())
    if not eligible_sessions:
        return {
            "df": pd.DataFrame(),
            "eligible_sessions": [],
            "rt_summary": pd.DataFrame(),
            "prop_summary": pd.DataFrame(),
            "abls": [],
        }

    subset = df[
        df["session_num"].isin(eligible_sessions) & df["sound_ramp_time_num"].notna()
    ].copy()
    if subset.empty:
        return {
            "df": pd.DataFrame(),
            "eligible_sessions": eligible_sessions,
            "rt_summary": pd.DataFrame(),
            "prop_summary": pd.DataFrame(),
            "abls": [],
        }

    subset["choice_right"] = _choice_right_series(subset)
    valid_trials = subset[pd.to_numeric(subset["success"], errors="coerce") != 0].copy()

    rt_summary = (
        valid_trials.dropna(subset=["timed_rt"])
        .groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["timed_rt"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
    )
    prop_summary = (
        valid_trials.dropna(subset=["choice_right"])
        .groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["choice_right"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
    )
    abls = sorted(
        set(pd.to_numeric(rt_summary["ABL"], errors="coerce").dropna().tolist())
        | set(pd.to_numeric(prop_summary["ABL"], errors="coerce").dropna().tolist())
    )

    return {
        "df": subset,
        "eligible_sessions": eligible_sessions,
        "rt_summary": rt_summary,
        "prop_summary": prop_summary,
        "abls": abls,
    }


def _format_values(values) -> str:
    out = []
    for value in values:
        if pd.isna(value):
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric).is_integer():
            out.append(str(int(numeric)))
        else:
            out.append(str(value))
    return "[" + ", ".join(out) + "]"


def _sessions_by_type(df: pd.DataFrame) -> dict[str, list]:
    if df.empty or "session_type" not in df or "session" not in df:
        return {}

    sessions_by_type = {}
    for session_type, sub in df.groupby("session_type", dropna=True):
        type_label = _format_values([session_type]).strip("[]")
        sessions = sorted(pd.to_numeric(sub["session"], errors="coerce").dropna().astype(int).unique())
        sessions_by_type[type_label] = sessions
    return sessions_by_type


def prepare_daily_animal_data(
    df: pd.DataFrame,
    *,
    training_level: int | None = 16,
    training_level_min: int | None = None,
    training_level_max: int | None = None,
    abl_filter: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    sound_ramp_filter: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    sound_ramp_exclude: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    include_long_duration: bool = True,
) -> dict[str, Any]:
    df = DataHelpers.prepare_data(
        df.copy(),
        session_col="session",
        trial_col="trial",
        apply_abl_fixes=False,
    )
    df = DataHelpers.normalize_stakes_abl(df)
    df = _normalize_stakes_sound_ramp_safe(df)
    df = df[df["trial_is_repeat"] == False].copy()
    df = _apply_training_and_abl_filters(
        df,
        training_level=training_level,
        training_level_min=training_level_min,
        training_level_max=training_level_max,
        abl_filter=abl_filter,
        sound_ramp_filter=sound_ramp_filter,
        sound_ramp_exclude=sound_ramp_exclude,
    )

    if include_long_duration:
        sess = pd.to_numeric(df["session_type"], errors="coerce")
        sd = pd.to_numeric(df["stim_dur"], errors="coerce")
        short_duration = pd.to_numeric(df["short_duration"], errors="coerce")
        df = df[(sess == 1) | (sd == 6000) | (short_duration == 0)].copy()

    df_last = df
    mean_rt = (
        df_last[df_last["success"] == 1]
        .groupby(["ABL", "ILD"])["timed_rt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    mean_mt = (
        df_last[df_last["success"] == 1]
        .groupby(["ABL", "ILD"])["timed_mt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    psychometric_results = Psychometric.compute_psychometrics_by_ABL(df_last)
    jnd = DataHelpers.compute_jnd_by_ABL(psychometric_results, skip_ABL=50)

    info = {
        "animal": df_last["animal"].iloc[0] if not df_last.empty and "animal" in df_last else None,
        "setup": df_last["box"].unique() if not df_last.empty and "box" in df_last else [],
        "session_type": df_last["session_type"].unique() if not df_last.empty and "session_type" in df_last else [],
        "n_sessions": df_last["session"].nunique() if "session" in df_last else 0,
        "sessions_by_type": _sessions_by_type(df_last),
        "n_trials": len(df_last),
    }

    return {
        "df": df,
        "df_last": df_last,
        "info": info,
        "mean_rt": mean_rt,
        "mean_mt": mean_mt,
        "psychometric_results": psychometric_results,
        "jnd": jnd,
    }


def prepare_stakes_group_data(
    df: pd.DataFrame,
    *,
    training_level: int | None = None,
    training_level_min: int | None = None,
    training_level_max: int | None = 16,
    abl_filter: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    sound_ramp_filter: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    sound_ramp_exclude: int | float | list[int | float] | tuple[int | float, ...] | None = None,
    include_long_duration: bool = True,
) -> dict[str, Any]:
    df = DataHelpers.prepare_data(
        df.copy(),
        session_col="session",
        trial_col="trial",
        apply_abl_fixes=False,
    )
    df = DataHelpers.normalize_stakes_abl(df)
    df = _normalize_stakes_sound_ramp_safe(df)
    df = df[df["trial_is_repeat"] == False].copy()
    df = _apply_training_and_abl_filters(
        df,
        training_level=training_level,
        training_level_min=training_level_min,
        training_level_max=training_level_max,
        abl_filter=abl_filter,
        sound_ramp_filter=sound_ramp_filter,
        sound_ramp_exclude=sound_ramp_exclude,
    )

    if include_long_duration:
        sess = pd.to_numeric(df["session_type"], errors="coerce")
        sd = pd.to_numeric(df["stim_dur"], errors="coerce")
        short_duration = pd.to_numeric(df["short_duration"], errors="coerce")
        df = df[(sess == 1) | (sd == 6000) | (short_duration == 0)].copy()

    df_last = df.copy()
    df_last["sound_ramp_time_num"] = pd.to_numeric(df_last.get("sound_ramp_time"), errors="coerce")

    rt_per_animal = (
        df_last[df_last["success"] == 1]
        .groupby(["animal", "ABL", "ILD"], observed=False)["timed_rt"]
        .agg(mean_rt="mean")
        .reset_index()
    )
    mean_rt = (
        rt_per_animal.groupby(["ABL", "ILD"], observed=False)["mean_rt"]
        .agg(mean="mean", sem=_sem, std="std", n="count")
        .reset_index()
    )

    mt_per_animal = (
        df_last[df_last["success"] == 1]
        .groupby(["animal", "ABL", "ILD"], observed=False)["timed_mt"]
        .agg(mean_mt="mean")
        .reset_index()
    )
    mean_mt = (
        mt_per_animal.groupby(["ABL", "ILD"], observed=False)["mean_mt"]
        .agg(mean="mean", sem=_sem, std="std", n="count")
        .reset_index()
    )

    rt_ramp_summary = pd.DataFrame(columns=["ABL", "sound_ramp_time_num", "ILD", "mean", "sem", "std", "n"])
    mt_ramp_summary = pd.DataFrame(columns=["ABL", "sound_ramp_time_num", "ILD", "mean", "sem", "std", "n"])
    if df_last["sound_ramp_time_num"].notna().any():
        rt_ramp_per_animal = (
            df_last[(df_last["success"] == 1) & df_last["sound_ramp_time_num"].notna()]
            .groupby(["animal", "ABL", "sound_ramp_time_num", "ILD"], observed=False)["timed_rt"]
            .agg(mean_rt="mean")
            .reset_index()
        )
        if not rt_ramp_per_animal.empty:
            rt_ramp_summary = (
                rt_ramp_per_animal.groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["mean_rt"]
                .agg(mean="mean", sem=_sem, std="std", n="count")
                .reset_index()
                .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
            )

        mt_ramp_per_animal = (
            df_last[(df_last["success"] == 1) & df_last["sound_ramp_time_num"].notna()]
            .groupby(["animal", "ABL", "sound_ramp_time_num", "ILD"], observed=False)["timed_mt"]
            .agg(mean_mt="mean")
            .reset_index()
        )
        if not mt_ramp_per_animal.empty:
            mt_ramp_summary = (
                mt_ramp_per_animal.groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["mean_mt"]
                .agg(mean="mean", sem=_sem, std="std", n="count")
                .reset_index()
                .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
            )

    psychometric_points: list[dict[str, Any]] = []
    psychometric_points_by_ramp: list[dict[str, Any]] = []
    psychometric_results_by_animal: dict[str, dict[Any, dict[str, Any]]] = {}
    psychometric_results_by_animal_ramp: dict[str, dict[float, dict[Any, dict[str, Any]]]] = {}
    jnd_rows: list[dict[str, Any]] = []
    jnd_rows_by_ramp: list[dict[str, Any]] = []
    for animal, df_animal in df_last.groupby("animal", sort=False):
        results = Psychometric.compute_psychometrics_by_ABL(df_animal, model="my_psycho")
        psychometric_results_by_animal[str(animal)] = results
        for abl, res in results.items():
            for ild, val in zip(np.asarray(res["ILDs"]), np.asarray(res["PropLeft"], dtype=float)):
                psychometric_points.append(
                    {"animal": str(animal), "ABL": float(abl), "ILD": float(ild), "PropRight": float(val)}
                )
        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=50)
        if jnd_df is not None and not jnd_df.empty:
            for _, row in jnd_df.iterrows():
                jnd_rows.append({"animal": str(animal), "ABL": float(row["ABL"]), "JND": float(row["JND"])})

        ramp_results_for_animal: dict[float, dict[Any, dict[str, Any]]] = {}
        for ramp, df_animal_ramp in df_animal.groupby("sound_ramp_time_num", sort=True):
            if pd.isna(ramp):
                continue
            results_ramp = Psychometric.compute_psychometrics_by_ABL(df_animal_ramp, model="my_psycho")
            ramp_results_for_animal[float(ramp)] = results_ramp
            for abl, res in results_ramp.items():
                for ild, val in zip(np.asarray(res["ILDs"]), np.asarray(res["PropLeft"], dtype=float)):
                    psychometric_points_by_ramp.append(
                        {
                            "animal": str(animal),
                            "sound_ramp_time_num": float(ramp),
                            "ABL": float(abl),
                            "ILD": float(ild),
                            "PropRight": float(val),
                        }
                    )
            jnd_df_ramp = DataHelpers.compute_jnd_by_ABL(results_ramp, skip_ABL=50)
            if jnd_df_ramp is not None and not jnd_df_ramp.empty:
                for _, row in jnd_df_ramp.iterrows():
                    jnd_rows_by_ramp.append(
                        {
                            "animal": str(animal),
                            "sound_ramp_time_num": float(ramp),
                            "ABL": float(row["ABL"]),
                            "JND": float(row["JND"]),
                        }
                    )
        psychometric_results_by_animal_ramp[str(animal)] = ramp_results_for_animal

    psy_points = pd.DataFrame(psychometric_points, columns=["animal", "ABL", "ILD", "PropRight"])
    if psy_points.empty:
        psy_group = pd.DataFrame(columns=["ABL", "ILD", "mean", "sem", "std", "n_animals"])
        mean_fits: dict[int, dict[str, Any] | None] = {}
    else:
        psy_group = (
            psy_points.groupby(["ABL", "ILD"], observed=False)["PropRight"]
            .agg(mean="mean", sem=_sem, std="std", n_animals="count")
            .reset_index()
            .sort_values(["ABL", "ILD"])
        )
        mean_fits = {}
        for abl, sub in psy_group.groupby("ABL", sort=True):
            ilds = sub["ILD"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            if len(ilds) < 4 or not np.isfinite(y).all():
                mean_fits[int(abl)] = None
                continue
            try:
                _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                    ilds,
                    y,
                    model="my_psycho",
                    n_trials=np.full_like(ilds, 50, dtype=int),
                    show_plot=False,
                )
                mean_fits[int(abl)] = {"xx": xx, "yy": yy}
            except Exception:
                mean_fits[int(abl)] = None

    psy_points_by_ramp = pd.DataFrame(
        psychometric_points_by_ramp,
        columns=["animal", "sound_ramp_time_num", "ABL", "ILD", "PropRight"],
    )
    if psy_points_by_ramp.empty:
        psy_group_by_ramp = pd.DataFrame(
            columns=["sound_ramp_time_num", "ABL", "ILD", "mean", "sem", "std", "n_animals"]
        )
        mean_fits_by_ramp: dict[tuple[float, int], dict[str, Any] | None] = {}
    else:
        psy_group_by_ramp = (
            psy_points_by_ramp.groupby(["sound_ramp_time_num", "ABL", "ILD"], observed=False)["PropRight"]
            .agg(mean="mean", sem=_sem, std="std", n_animals="count")
            .reset_index()
            .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
        )
        mean_fits_by_ramp = {}
        for (ramp, abl), sub in psy_group_by_ramp.groupby(["sound_ramp_time_num", "ABL"], sort=True):
            ilds = sub["ILD"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            key = (float(ramp), int(abl))
            if len(ilds) < 4 or not np.isfinite(y).all():
                mean_fits_by_ramp[key] = None
                continue
            try:
                _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                    ilds,
                    y,
                    model="my_psycho",
                    n_trials=np.full_like(ilds, 50, dtype=int),
                    show_plot=False,
                )
                mean_fits_by_ramp[key] = {"xx": xx, "yy": yy}
            except Exception:
                mean_fits_by_ramp[key] = None

    jnd_indiv = pd.DataFrame(jnd_rows, columns=["animal", "ABL", "JND"])
    if jnd_indiv.empty:
        group_jnd = pd.DataFrame(columns=["ABL", "mean", "sem", "std", "n_animals"])
    else:
        group_jnd = (
            jnd_indiv.groupby("ABL", observed=False)["JND"]
            .agg(mean="mean", sem=_sem, std="std", n_animals="count")
            .reset_index()
            .sort_values("ABL")
        )

    jnd_indiv_by_ramp = pd.DataFrame(
        jnd_rows_by_ramp,
        columns=["animal", "sound_ramp_time_num", "ABL", "JND"],
    )
    if jnd_indiv_by_ramp.empty:
        group_jnd_by_ramp = pd.DataFrame(columns=["sound_ramp_time_num", "ABL", "mean", "sem", "std", "n_animals"])
    else:
        group_jnd_by_ramp = (
            jnd_indiv_by_ramp.groupby(["sound_ramp_time_num", "ABL"], observed=False)["JND"]
            .agg(mean="mean", sem=_sem, std="std", n_animals="count")
            .reset_index()
            .sort_values(["sound_ramp_time_num", "ABL"])
        )

    info = {
        "line": sorted(df_last["line"].astype(str).unique()) if "line" in df_last and not df_last.empty else ["Stakes"],
        "cohorts": sorted(df_last["cohort"].astype(str).unique()) if "cohort" in df_last and not df_last.empty else [],
        "n_animals": int(df_last["animal"].nunique()) if "animal" in df_last else 0,
        "animals": sorted(df_last["animal"].astype(str).unique()) if "animal" in df_last and not df_last.empty else [],
        "n_sessions": int(df_last.groupby(["dataset_key", "animal", "session"], observed=False).ngroups) if {"dataset_key", "animal", "session"}.issubset(df_last.columns) else int(df_last["session"].nunique()) if "session" in df_last else 0,
        "n_trials": len(df_last),
    }

    return {
        "df": df,
        "df_last": df_last,
        "info": info,
        "rt_per_animal": rt_per_animal,
        "mt_per_animal": mt_per_animal,
        "mean_rt": mean_rt,
        "mean_mt": mean_mt,
        "mean_rt_by_ramp": rt_ramp_summary,
        "mean_mt_by_ramp": mt_ramp_summary,
        "psy_points": psy_points,
        "psy_group": psy_group,
        "psy_mean_fits": mean_fits,
        "psy_points_by_ramp": psy_points_by_ramp,
        "psy_group_by_ramp": psy_group_by_ramp,
        "psy_mean_fits_by_ramp": mean_fits_by_ramp,
        "psychometric_results_by_animal": psychometric_results_by_animal,
        "psychometric_results_by_animal_ramp": psychometric_results_by_animal_ramp,
        "jnd_indiv": jnd_indiv,
        "group_jnd": group_jnd,
        "jnd_indiv_by_ramp": jnd_indiv_by_ramp,
        "group_jnd_by_ramp": group_jnd_by_ramp,
    }


def _prepare_group_sound_ramp_comparison_data(prepared: dict[str, Any]) -> dict[str, Any]:
    df = prepared["df_last"].copy()
    if df.empty or "sound_ramp_time" not in df.columns or "ABL" not in df.columns:
        return {
            "eligible_session_keys": [],
            "rt_summary": pd.DataFrame(),
            "mt_summary": pd.DataFrame(),
            "prop_summary": pd.DataFrame(),
        }

    df["sound_ramp_time_num"] = pd.to_numeric(df["sound_ramp_time"], errors="coerce")
    df["session_num"] = pd.to_numeric(df.get("session"), errors="coerce")
    session_cols = [col for col in ["animal", "session_num"] if col in df.columns]
    if not session_cols:
        return {
            "eligible_session_keys": [],
            "rt_summary": pd.DataFrame(),
            "mt_summary": pd.DataFrame(),
            "prop_summary": pd.DataFrame(),
        }

    ramp_counts = (
        df.dropna(subset=["session_num"])
        .groupby(session_cols, observed=False)["sound_ramp_time_num"]
        .nunique(dropna=True)
        .reset_index(name="n_ramps")
    )
    eligible = ramp_counts[ramp_counts["n_ramps"] > 1][session_cols].copy()
    if eligible.empty:
        return {
            "eligible_session_keys": [],
            "rt_summary": pd.DataFrame(),
            "prop_summary": pd.DataFrame(),
        }

    subset = df.merge(eligible.assign(_keep=True), on=session_cols, how="inner")
    subset = subset[subset["sound_ramp_time_num"].notna()].copy()
    subset["choice_right"] = _choice_right_series(subset)
    valid_trials = subset[pd.to_numeric(subset["success"], errors="coerce") != 0].copy()

    rt_per_animal = (
        valid_trials.dropna(subset=["timed_rt"])
        .groupby(["animal", "ABL", "sound_ramp_time_num", "ILD"], observed=False)["timed_rt"]
        .agg(animal_mean="mean")
        .reset_index()
    )
    rt_summary = (
        rt_per_animal.groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["animal_mean"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
    )

    mt_per_animal = (
        valid_trials[(pd.to_numeric(valid_trials["success"], errors="coerce") == 1)]
        .dropna(subset=["timed_mt"])
        .groupby(["animal", "ABL", "sound_ramp_time_num", "ILD"], observed=False)["timed_mt"]
        .agg(animal_mean="mean")
        .reset_index()
    )
    mt_summary = (
        mt_per_animal.groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["animal_mean"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
    )

    prop_per_animal = (
        valid_trials.dropna(subset=["choice_right"])
        .groupby(["animal", "ABL", "sound_ramp_time_num", "ILD"], observed=False)["choice_right"]
        .agg(animal_mean="mean")
        .reset_index()
    )
    prop_summary = (
        prop_per_animal.groupby(["ABL", "sound_ramp_time_num", "ILD"], observed=False)["animal_mean"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["ABL", "sound_ramp_time_num", "ILD"])
    )

    eligible_session_keys = list(eligible.itertuples(index=False, name=None))
    return {
        "eligible_session_keys": eligible_session_keys,
        "rt_summary": rt_summary,
        "mt_summary": mt_summary,
        "prop_summary": prop_summary,
    }


def _prepare_group_ramp_across_sessions_data(prepared: dict[str, Any]) -> dict[str, Any]:
    df = prepared["df_last"].copy()
    empty = {
        "eligible_session_keys": [],
        "ramp_values": [],
        "completed_trials": pd.DataFrame(),
        "mean_rt": pd.DataFrame(),
        "mean_mt": pd.DataFrame(),
        "mean_acc": pd.DataFrame(),
        "bias_summary": pd.DataFrame(),
        "mean_fa": pd.DataFrame(),
        "mean_mta": pd.DataFrame(),
    }
    if df.empty or "sound_ramp_time" not in df.columns or "ABL" not in df.columns:
        return empty

    df["sound_ramp_time_num"] = pd.to_numeric(df["sound_ramp_time"], errors="coerce")
    df["session_num"] = pd.to_numeric(df.get("session"), errors="coerce")
    session_cols = [col for col in ["animal", "session_num"] if col in df.columns]
    if not session_cols:
        return empty

    ramp_counts = (
        df.dropna(subset=["session_num"])
        .groupby(session_cols, observed=False)["sound_ramp_time_num"]
        .nunique(dropna=True)
        .reset_index(name="n_ramps")
    )
    eligible = ramp_counts[ramp_counts["n_ramps"] >= 1][session_cols].copy()
    if eligible.empty:
        return empty

    subset = df.merge(eligible.assign(_keep=True), on=session_cols, how="inner")
    subset = subset[subset["sound_ramp_time_num"].notna()].copy()

    def _animal_then_group(
        frame: pd.DataFrame,
        value_col: str,
        per_animal_name: str,
        *,
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        cols = group_cols or ["animal", "session_num", "sound_ramp_time_num"]
        if frame.empty:
            return pd.DataFrame(columns=["session_num", "sound_ramp_time_num", "mean", "sem", "std", "n_animals"])
        per_animal = (
            frame.groupby(cols, observed=False)[value_col]
            .agg(**{per_animal_name: "mean"})
            .reset_index()
        )
        return (
            per_animal.groupby(["session_num", "sound_ramp_time_num"], observed=False)[per_animal_name]
            .agg(mean="mean", sem=_sem, std="std", n_animals="count")
            .reset_index()
            .sort_values(["sound_ramp_time_num", "session_num"])
        )

    valid_trials = subset[pd.to_numeric(subset["success"], errors="coerce") != 0].copy()
    completed_trials = subset[pd.to_numeric(subset["success"], errors="coerce") == 1].copy()
    non_cnp_trials = subset[subset["abort_type"] != "CNP"].copy()

    mean_rt = _animal_then_group(
        completed_trials.dropna(subset=["timed_rt"]),
        "timed_rt",
        "animal_mean",
    )
    completed_per_animal = (
        valid_trials.groupby(["animal", "session_num", "sound_ramp_time_num"], observed=False)
        .size()
        .reset_index(name="animal_completed_trials")
    )
    completed_summary = (
        completed_per_animal.groupby(["session_num", "sound_ramp_time_num"], observed=False)["animal_completed_trials"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["sound_ramp_time_num", "session_num"])
    )
    mean_mt = _animal_then_group(
        completed_trials.dropna(subset=["timed_mt"]),
        "timed_mt",
        "animal_mean",
    )

    acc_per_animal = (
        valid_trials.groupby(["animal", "session_num", "sound_ramp_time_num"], observed=False)["success"]
        .agg(animal_accuracy=lambda x: (pd.to_numeric(x, errors="coerce") == 1).mean())
        .reset_index()
    )
    mean_acc = (
        acc_per_animal.groupby(["session_num", "sound_ramp_time_num"], observed=False)["animal_accuracy"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["sound_ramp_time_num", "session_num"])
    )

    bias_per_animal = (
        subset.groupby(["animal", "session_num", "sound_ramp_time_num"], observed=False)
        .apply(DataHelpers.compute_bias, include_groups=False)
        .pipe(lambda x: x if isinstance(x, pd.Series) else pd.Series(dtype=float))
        .reset_index(name="animal_bias")
    )
    bias_summary = (
        bias_per_animal.groupby(["session_num", "sound_ramp_time_num"], observed=False)["animal_bias"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["sound_ramp_time_num", "session_num"])
    )

    fa_per_animal = (
        non_cnp_trials.groupby(["animal", "session_num", "sound_ramp_time_num"], observed=False)["abort_type"]
        .agg(animal_rate=lambda x: (x == "Fixation").mean())
        .reset_index()
    )
    mean_fa = (
        fa_per_animal.groupby(["session_num", "sound_ramp_time_num"], observed=False)["animal_rate"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["sound_ramp_time_num", "session_num"])
    )

    mta_per_animal = (
        non_cnp_trials.groupby(["animal", "session_num", "sound_ramp_time_num"], observed=False)["abort_type"]
        .agg(animal_rate=lambda x: (x == "MT+").mean())
        .reset_index()
    )
    mean_mta = (
        mta_per_animal.groupby(["session_num", "sound_ramp_time_num"], observed=False)["animal_rate"]
        .agg(mean="mean", sem=_sem, std="std", n_animals="count")
        .reset_index()
        .sort_values(["sound_ramp_time_num", "session_num"])
    )

    eligible_session_keys = list(eligible.itertuples(index=False, name=None))
    ramp_values = sorted(subset["sound_ramp_time_num"].dropna().unique().tolist())
    return {
        "eligible_session_keys": eligible_session_keys,
        "ramp_values": ramp_values,
        "completed_trials": completed_summary,
        "mean_rt": mean_rt,
        "mean_mt": mean_mt,
        "mean_acc": mean_acc,
        "bias_summary": bias_summary,
        "mean_fa": mean_fa,
        "mean_mta": mean_mta,
    }


def plot_stakes_group_summary_with_jnd(
    prepared: dict[str, Any],
    *,
    figsize=(18, 10),
    colors=("C0", "C1", "C2", "C3"),
):
    apply_daily_plot_style()
    mean_rt = prepared["mean_rt"]
    mean_mt = prepared["mean_mt"]
    psy_group = prepared["psy_group"]
    mean_fits = prepared["psy_mean_fits"]
    group_jnd = prepared["group_jnd"]
    mean_rt_by_ramp = prepared.get("mean_rt_by_ramp", pd.DataFrame())
    mean_mt_by_ramp = prepared.get("mean_mt_by_ramp", pd.DataFrame())
    psy_group_by_ramp = prepared.get("psy_group_by_ramp", pd.DataFrame())
    mean_fits_by_ramp = prepared.get("psy_mean_fits_by_ramp", {})
    group_jnd_by_ramp = prepared.get("group_jnd_by_ramp", pd.DataFrame())
    info_data = prepared["info"]

    ramp_values = sorted(
        set(pd.to_numeric(mean_rt_by_ramp.get("sound_ramp_time_num"), errors="coerce").dropna().tolist())
        | set(pd.to_numeric(mean_mt_by_ramp.get("sound_ramp_time_num"), errors="coerce").dropna().tolist())
        | set(pd.to_numeric(psy_group_by_ramp.get("sound_ramp_time_num"), errors="coerce").dropna().tolist())
    )
    use_ramp_split = len(ramp_values) > 0
    ramp_colors = {ramp: colors[i % len(colors)] for i, ramp in enumerate(ramp_values)}

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_mt = fig.add_subplot(gs[1, 0])
    ax_psy = fig.add_subplot(gs[1, 1])

    ax_text.axis("off")
    info = (
        f"Line: {info_data['line']}\n"
        f"Cohorts: {info_data['cohorts']}\n"
        f"Animals: {info_data['n_animals']}\n"
        f"Sessions: {info_data['n_sessions']}\n"
        f"Trials: {info_data['n_trials']}\n"
    )
    ax_text.text(
        0.5,
        0.5,
        info,
        fontsize=18,
        ha="center",
        va="center",
        family="monospace",
    )

    if use_ramp_split:
        for abl in sorted(pd.to_numeric(mean_rt_by_ramp["ABL"], errors="coerce").dropna().unique()):
            sub_abl = mean_rt_by_ramp[mean_rt_by_ramp["ABL"] == abl]
            abl_label = _display_abl_label(abl, sub_abl["ILD"])
            for ramp in ramp_values:
                sub = sub_abl[sub_abl["sound_ramp_time_num"] == ramp]
                if sub.empty:
                    continue
                ax_rt.errorbar(
                    DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
                    sub["mean"],
                    yerr=sub["sem"],
                    label=f"ABL {abl_label}, ramp={ramp:g}",
                    color=ramp_colors[ramp],
                    linestyle="none",
                    marker="o",
                    capsize=5,
                    linewidth=1.8,
                )
    else:
        for i, abl in enumerate(sorted(pd.to_numeric(mean_rt["ABL"], errors="coerce").dropna().unique())):
            sub = mean_rt[mean_rt["ABL"] == abl]
            ax_rt.errorbar(
                DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
                sub["mean"],
                yerr=sub["sem"],
                label=f"ABL {_display_abl_label(abl, sub['ILD'])}",
                color=colors[i % len(colors)],
                linestyle="none",
                marker="o",
                capsize=5,
                linewidth=1.8,
            )
    style_axes(ax_rt, title="Reaction Time", xlabel="ILD (dB)", ylabel="Mean RT (s)", box_aspect=None)

    if use_ramp_split:
        for abl in sorted(pd.to_numeric(mean_mt_by_ramp["ABL"], errors="coerce").dropna().unique()):
            sub_abl = mean_mt_by_ramp[mean_mt_by_ramp["ABL"] == abl]
            abl_label = _display_abl_label(abl, sub_abl["ILD"])
            for ramp in ramp_values:
                sub = sub_abl[sub_abl["sound_ramp_time_num"] == ramp]
                if sub.empty:
                    continue
                ax_mt.errorbar(
                    DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
                    sub["mean"],
                    yerr=sub["sem"],
                    label=f"ABL {abl_label}, ramp={ramp:g}",
                    color=ramp_colors[ramp],
                    linestyle="none",
                    marker="o",
                    capsize=5,
                    linewidth=1.8,
                )
    else:
        for i, abl in enumerate(sorted(pd.to_numeric(mean_mt["ABL"], errors="coerce").dropna().unique())):
            sub = mean_mt[mean_mt["ABL"] == abl]
            ax_mt.errorbar(
                DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
                sub["mean"],
                yerr=sub["sem"],
                label=f"ABL {_display_abl_label(abl, sub['ILD'])}",
                color=colors[i % len(colors)],
                linestyle="none",
                marker="o",
                capsize=5,
                linewidth=1.8,
            )
    style_axes(ax_mt, title="Movement Time", xlabel="ILD (dB)", ylabel="Mean MT (s)", box_aspect=None)

    if use_ramp_split:
        for abl in sorted(pd.to_numeric(psy_group_by_ramp["ABL"], errors="coerce").dropna().unique()):
            sub_abl = psy_group_by_ramp[psy_group_by_ramp["ABL"] == abl]
            abl_label = _display_abl_label(abl, sub_abl["ILD"])
            for ramp in ramp_values:
                sub = sub_abl[sub_abl["sound_ramp_time_num"] == ramp]
                if sub.empty:
                    continue
                color = ramp_colors[ramp]
                ax_psy.errorbar(
                    DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
                    sub["mean"],
                    yerr=sub["sem"],
                    label=f"ABL={abl_label}, ramp={ramp:g}",
                    color=color,
                    linestyle="none",
                    marker="o",
                    capsize=5,
                    linewidth=1.8,
                    zorder=3,
                )
                fit = mean_fits_by_ramp.get((float(ramp), int(abl)))
                if fit is not None:
                    ax_psy.plot(
                        DataHelpers.shift_ILD_for_ABL50(fit["xx"]),
                        fit["yy"],
                        color=color,
                        linewidth=1.8,
                    )
    else:
        for i, abl in enumerate(sorted(pd.to_numeric(psy_group["ABL"], errors="coerce").dropna().unique())):
            sub = psy_group[psy_group["ABL"] == abl]
            color = colors[i % len(colors)]
            ax_psy.errorbar(
                DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
                sub["mean"],
                yerr=sub["sem"],
                label=f"ABL={_display_abl_label(abl, sub['ILD'])}",
                color=color,
                linestyle="none",
                marker="o",
                capsize=5,
                linewidth=1.8,
                zorder=3,
            )
            fit = mean_fits.get(int(abl))
            if fit is not None:
                ax_psy.plot(
                    DataHelpers.shift_ILD_for_ABL50(fit["xx"]),
                    fit["yy"],
                    color=color,
                    linewidth=1.8,
                )
    ax_psy.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    ax_psy.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    style_axes(ax_psy, title="Psychometric", xlabel="ILD (dB)", ylabel="Proportion Right", box_aspect=None)
    psy_for_ticks = psy_group_by_ramp if use_ramp_split else psy_group
    xticks = sorted(set(DataHelpers.shift_ILD_for_ABL50(psy_for_ticks["ILD"])) | {0.0}) if not psy_for_ticks.empty else [-18, 0, 18]
    ax_psy.set_xticks(xticks)
    ax_psy.set_xticklabels([_format_psychometric_tick(x) for x in xticks])
    if not psy_for_ticks.empty:
        ax_psy.set_xlim(min(xticks) - 1, max(xticks) + 1)
    ax_psy.legend(fontsize=13, frameon=False, loc="upper left")

    jnd_for_plot = group_jnd_by_ramp if use_ramp_split and not group_jnd_by_ramp.empty else group_jnd
    if not jnd_for_plot.empty:
        ax_jnd = ax_psy.inset_axes([0.7, 0.12, 0.25, 0.25])
        ax_jnd.set_facecolor("white")
        if use_ramp_split and not group_jnd_by_ramp.empty:
            for ramp in ramp_values:
                ramp_sub = group_jnd_by_ramp[group_jnd_by_ramp["sound_ramp_time_num"] == ramp]
                if ramp_sub.empty:
                    continue
                for _, row in ramp_sub.iterrows():
                    ax_jnd.errorbar(
                        row["ABL"],
                        row["mean"],
                        yerr=row["sem"],
                        marker="o",
                        linestyle="none",
                        color=ramp_colors[ramp],
                        markersize=7,
                        capsize=3,
                        elinewidth=1.2,
                    )
        else:
            full_abl_order = sorted(pd.to_numeric(mean_rt["ABL"], errors="coerce").dropna().unique())
            abl_colors = {abl: colors[i % len(colors)] for i, abl in enumerate(full_abl_order)}
            for _, row in group_jnd.iterrows():
                ax_jnd.errorbar(
                    row["ABL"],
                    row["mean"],
                    yerr=row["sem"],
                    marker="o",
                    linestyle="none",
                    color=abl_colors.get(row["ABL"], "black"),
                    markersize=7,
                    capsize=3,
                    elinewidth=1.2,
                )
        style_axes(
            ax_jnd,
            xlabel="ABL",
            ylabel="JND",
            title_fs=13,
            label_fs=12,
            tick_fs=10,
            title_pad=4,
            box_aspect=None,
        )
        if use_ramp_split and not group_jnd_by_ramp.empty:
            jnd_abls = sorted(group_jnd_by_ramp["ABL"].unique())
            ax_jnd.set_xticks(jnd_abls)
            ax_jnd.set_xticklabels([_display_abl_label(abl) for abl in jnd_abls])
        else:
            ax_jnd.set_xticks(sorted(group_jnd["ABL"].unique()))
            ax_jnd.set_xticklabels([])
        ax_jnd.grid(True, linestyle="--", alpha=0.3, zorder=-10)

    return fig


def plot_stakes_group_sound_ramp_comparison(
    prepared: dict[str, Any],
    *,
    figsize_per_row=(16, 3.8),
):
    apply_daily_plot_style()
    ramp_data = _prepare_group_sound_ramp_comparison_data(prepared)
    rt_summary = ramp_data["rt_summary"]
    mt_summary = ramp_data["mt_summary"]
    prop_summary = ramp_data["prop_summary"]
    info_data = prepared["info"]

    if rt_summary.empty and mt_summary.empty and prop_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            (
                f"Stakes group ({', '.join(info_data.get('cohorts', []))})\n"
                "No sessions contain more than one non-NaN sound ramp duration\n"
                "for the current group-analysis subset."
            ),
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    ramp_values = sorted(
        set(pd.to_numeric(rt_summary["sound_ramp_time_num"], errors="coerce").dropna().tolist())
        | set(pd.to_numeric(mt_summary["sound_ramp_time_num"], errors="coerce").dropna().tolist())
        | set(pd.to_numeric(prop_summary["sound_ramp_time_num"], errors="coerce").dropna().tolist())
    )
    ramp_colors = {ramp: f"C{i % 10}" for i, ramp in enumerate(ramp_values)}
    abls = sorted(
        set(pd.to_numeric(rt_summary["ABL"], errors="coerce").dropna().tolist())
        | set(pd.to_numeric(mt_summary["ABL"], errors="coerce").dropna().tolist())
        | set(pd.to_numeric(prop_summary["ABL"], errors="coerce").dropna().tolist())
    )

    fig_width, row_height = figsize_per_row
    fig, axes = plt.subplots(
        len(abls),
        3,
        figsize=(fig_width, max(row_height * len(abls), row_height)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Stakes group · Sound Ramp Comparison ({len(ramp_data['eligible_session_keys'])} eligible animal-sessions)",
        fontsize=18,
    )

    for row_idx, abl in enumerate(abls):
        ax_rt = axes[row_idx, 0]
        ax_mt = axes[row_idx, 1]
        ax_prop = axes[row_idx, 2]

        rt_sub = rt_summary[rt_summary["ABL"] == abl]
        for ramp in ramp_values:
            ramp_rt = rt_sub[rt_sub["sound_ramp_time_num"] == ramp]
            if ramp_rt.empty:
                continue
            ax_rt.errorbar(
                DataHelpers.shift_ILD_for_ABL50(ramp_rt["ILD"]),
                ramp_rt["mean"],
                yerr=ramp_rt["sem"],
                linestyle="none",
                marker="o",
                color=ramp_colors[ramp],
                capsize=4,
                markersize=5,
                elinewidth=1.4,
                label=f"ramp={ramp:g}" if row_idx == 0 else None,
            )

        mt_sub = mt_summary[mt_summary["ABL"] == abl]
        for ramp in ramp_values:
            ramp_mt = mt_sub[mt_sub["sound_ramp_time_num"] == ramp]
            if ramp_mt.empty:
                continue
            ax_mt.errorbar(
                DataHelpers.shift_ILD_for_ABL50(ramp_mt["ILD"]),
                ramp_mt["mean"],
                yerr=ramp_mt["sem"],
                linestyle="none",
                marker="o",
                color=ramp_colors[ramp],
                capsize=4,
                markersize=5,
                elinewidth=1.4,
                label=f"ramp={ramp:g}" if row_idx == 0 else None,
            )

        prop_sub = prop_summary[prop_summary["ABL"] == abl]
        for ramp in ramp_values:
            ramp_prop = prop_sub[prop_sub["sound_ramp_time_num"] == ramp]
            if ramp_prop.empty:
                continue
            ax_prop.errorbar(
                DataHelpers.shift_ILD_for_ABL50(ramp_prop["ILD"]),
                ramp_prop["mean"],
                yerr=ramp_prop["sem"],
                linestyle="none",
                marker="o",
                color=ramp_colors[ramp],
                capsize=4,
                markersize=5,
                elinewidth=1.4,
                label=f"ramp={ramp:g}" if row_idx == 0 else None,
            )

        style_axes(
            ax_rt,
            title=f"ABL {_display_abl_label(abl, rt_sub['ILD'])} · RT",
            xlabel="ILD (dB)",
            ylabel="Mean RT (s)",
            title_fs=16,
            label_fs=13,
            tick_fs=11,
            box_aspect=None,
        )
        rt_ticks = sorted(set(DataHelpers.shift_ILD_for_ABL50(rt_sub["ILD"]))) if not rt_sub.empty else []
        ax_rt.set_xticks(rt_ticks)
        ax_rt.set_xticklabels([_format_psychometric_tick(x) for x in rt_ticks])
        ax_rt.grid(True, linestyle="--", alpha=0.25, zorder=-10)

        style_axes(
            ax_mt,
            title=f"ABL {_display_abl_label(abl, mt_sub['ILD'])} · MT",
            xlabel="ILD (dB)",
            ylabel="Mean MT (s)",
            title_fs=16,
            label_fs=13,
            tick_fs=11,
            box_aspect=None,
        )
        mt_ticks = sorted(set(DataHelpers.shift_ILD_for_ABL50(mt_sub["ILD"]))) if not mt_sub.empty else []
        ax_mt.set_xticks(mt_ticks)
        ax_mt.set_xticklabels([_format_psychometric_tick(x) for x in mt_ticks])
        ax_mt.grid(True, linestyle="--", alpha=0.25, zorder=-10)

        style_axes(
            ax_prop,
            title=f"ABL {_display_abl_label(abl, prop_sub['ILD'])} · Proportion Right",
            xlabel="ILD (dB)",
            ylabel="Proportion Right",
            title_fs=16,
            label_fs=13,
            tick_fs=11,
            box_aspect=None,
        )
        prop_ticks = sorted(set(DataHelpers.shift_ILD_for_ABL50(prop_sub["ILD"]))) if not prop_sub.empty else []
        ax_prop.set_xticks(prop_ticks)
        ax_prop.set_xticklabels([_format_psychometric_tick(x) for x in prop_ticks])
        ax_prop.set_ylim(-0.05, 1.05)
        ax_prop.grid(True, linestyle="--", alpha=0.25, zorder=-10)

    axes[0, 2].legend(fontsize=10, frameon=False, loc="best", title="Ramp")

    return fig


def plot_stakes_group_ramp_across_sessions(
    prepared: dict[str, Any],
    *,
    figsize=(18, 10),
):
    apply_daily_plot_style()
    session_data = _prepare_group_ramp_across_sessions_data(prepared)

    if not session_data["ramp_values"]:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            (
                f"Stakes group ({', '.join(prepared['info'].get('cohorts', []))})\n"
                "No sessions contain more than one non-NaN sound ramp duration\n"
                "for the current group-analysis subset."
            ),
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    ramp_values = session_data["ramp_values"]
    ramp_colors = {ramp: f"C{i % 10}" for i, ramp in enumerate(ramp_values)}
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 2.0])
    ax_counts = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 3)
    axes = [fig.add_subplot(right[r, c]) for r in range(2) for c in range(3)]
    fig.suptitle(
        f"Stakes group · Across Sessions by Sound Ramp ({len(session_data['eligible_session_keys'])} eligible animal-sessions)",
        fontsize=18,
    )

    metric_specs = [
        ("mean_rt", "Mean RT (s)", "RT", axes[0]),
        ("mean_mt", "Mean MT (s)", "MT", axes[1]),
        ("mean_acc", "Accuracy", "Accuracy", axes[2]),
        ("bias_summary", "Bias", "Bias", axes[3]),
        ("mean_fa", "Fixation Abort Rate", "Fixation Aborts", axes[4]),
        ("mean_mta", "MT+ Abort Rate", "MT+ Aborts", axes[5]),
    ]

    for table_name, ylabel, short_title, ax in metric_specs:
        for ramp in ramp_values:
            table = session_data[table_name]
            ramp_table = table[table["sound_ramp_time_num"] == ramp]
            if not ramp_table.empty:
                ax.errorbar(
                    ramp_table["session_num"],
                    ramp_table["mean"],
                    yerr=ramp_table["sem"],
                    color=ramp_colors[ramp],
                    linestyle="-",
                    marker="o",
                    capsize=4,
                    linewidth=1.6,
                    markersize=4.8,
                    label=f"Ramp {ramp:g}",
                )

        style_axes(
            ax,
            title=short_title,
            xlabel="Session",
            ylabel=ylabel,
            title_fs=15,
            label_fs=12,
            tick_fs=10,
            box_aspect=None,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.25, zorder=-10)
        if short_title in {"Accuracy", "Fixation Aborts", "MT+ Aborts"}:
            ax.set_ylim(-0.05, 1.05)
        if short_title == "Bias":
            ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)

    for ramp in ramp_values:
        completed = session_data["completed_trials"]
        completed_sub = completed[completed["sound_ramp_time_num"] == ramp]
        if not completed_sub.empty:
            ax_counts.errorbar(
                completed_sub["session_num"],
                completed_sub["mean"],
                yerr=completed_sub["sem"],
                color=ramp_colors[ramp],
                linestyle="-",
                marker="o",
                capsize=4,
                linewidth=1.6,
                markersize=4.8,
                label=f"Ramp {ramp:g}",
            )

    style_axes(
        ax_counts,
        title="Completed Trials",
        xlabel="Session",
        ylabel="Completed Trials",
        title_fs=15,
        label_fs=12,
        tick_fs=10,
        box_aspect=None,
    )
    ax_counts.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_counts.grid(True, linestyle="--", alpha=0.25, zorder=-10)
    ax_counts.set_box_aspect(1)
    ax_counts.legend(
        fontsize=9,
        frameon=False,
        loc="best",
        title="Ramp",
    )
    return fig


def plot_daily_animal_summary(
    prepared: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    overlay_reference_psychometric: bool = True,
    overlay_reference_only_when_fit: bool = True,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(12, 10),
):
    apply_daily_plot_style()
    mean_rt = prepared["mean_rt"]
    mean_mt = prepared["mean_mt"]
    results = prepared["psychometric_results"]
    info_data = prepared["info"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax_text = axes[0, 0]
    ax_text.axis("off")
    info = (
        f"Animal: {info_data['animal']}\n"
        f"Setup number: {info_data['setup']}\n"
        f"Session type: {info_data['session_type']}\n"
        f"Sessions: {info_data['n_sessions']}\n"
        + "\n".join(
            f"  Type {session_type}: {len(sessions)} sessions"
            for session_type, sessions in info_data.get("sessions_by_type", {}).items()
        )
        + "\n"
        f"Number of trials: {info_data['n_trials']}\n"
    )
    ax_text.text(
        0.5,
        0.5,
        info,
        fontsize=16,
        ha="center",
        va="center",
        family="monospace",
    )

    for i, abl in enumerate(sorted(mean_rt["ABL"].unique())):
        sub = mean_rt[mean_rt["ABL"] == abl]
        axes[0, 1].errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {_display_abl_label(abl, sub['ILD'])}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )

    style_axes(
        axes[0, 1],
        title="Reaction Time",
        xlabel="ILD (dB)",
        ylabel="Mean Reaction Time (s)",
    )

    for i, abl in enumerate(sorted(mean_mt["ABL"].unique())):
        sub = mean_mt[mean_mt["ABL"] == abl]
        axes[1, 0].errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {_display_abl_label(abl, sub['ILD'])}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )

    style_axes(
        axes[1, 0],
        title="Movement Time",
        xlabel="ILD (dB)",
        ylabel="Mean Movement Time (s)",
    )

    ax_psy = axes[1, 1]
    for color, (abl, res) in zip(colors, results.items()):
        ax_psy.scatter(
            DataHelpers.shift_ILD_for_ABL50(res["ILDs"]),
            res["PropLeft"],
            label=f"ABL={_display_abl_label(abl, res['ILDs'])}",
            color=color,
            s=70,
            edgecolor=color,
            linewidth=0.6,
            zorder=3,
        )
        if _has_psychometric_fit(res):
            ax_psy.plot(
                DataHelpers.shift_ILD_for_ABL50(res["xx"]),
                res["yy"],
                color=color,
                linewidth=1.8,
            )

    if (
        reference is not None
        and overlay_reference_psychometric
        and (
            not overlay_reference_only_when_fit
            or _has_any_psychometric_fit(results)
        )
    ):
        DataHelpers.overlay_makefig1_psychometrics(
            ax_psy,
            reference["psychometric"],
            color="black",
            show_individuals=True,
            use_abl_colors=False,
        )

    ax_psy.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    ax_psy.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    style_axes(
        ax_psy,
        title="Psychometric by ABL",
        xlabel="ILD (dB)",
        ylabel="Proportion Right",
    )

    xticks, xlim = _psychometric_ticks_and_limits(results)
    ax_psy.set_xticks(xticks)
    ax_psy.set_xticklabels([_format_psychometric_tick(x) for x in xticks])
    ax_psy.set_xlim(*xlim)
    ax_psy.legend(fontsize=16, frameon=False)

    fig.tight_layout()
    return fig


def plot_daily_animal_jnd(
    prepared: dict[str, Any],
    *,
    figsize=(2.5, 2),
):
    apply_daily_plot_style()
    jnd = prepared["jnd"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        jnd["ABL"],
        jnd["JND"],
        "-o",
        color="black",
        linewidth=1.2,
        markersize=4,
    )
    style_axes(
        ax,
        xlabel="ABL (dB)",
        ylabel="JND (dB)",
        title_fs=12,
        label_fs=10,
        tick_fs=10,
    )
    ax.set_xticks(sorted(jnd["ABL"].unique()))
    ax.grid(True, linestyle="--", alpha=0.3, zorder=-10)
    fig.tight_layout()
    return fig


def plot_daily_animal_summary_with_jnd(
    prepared: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    overlay_reference_psychometric: bool = True,
    overlay_reference_only_when_fit: bool = True,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(18, 10),
):
    apply_daily_plot_style()
    mean_rt = prepared["mean_rt"]
    mean_mt = prepared["mean_mt"]
    results = prepared["psychometric_results"]
    jnd = prepared["jnd"]
    info_data = prepared["info"]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_mt = fig.add_subplot(gs[1, 0])
    ax_psy = fig.add_subplot(gs[1, 1])

    ax_text.axis("off")
    info = (
        f"Animal: {info_data['animal']}\n"
        f"Setup number: {info_data['setup']}\n"
        f"Session type: {info_data['session_type']}\n"
        f"Sessions: {info_data['n_sessions']}\n"
        + "\n".join(
            f"  Type {session_type}: {len(sessions)} sessions"
            for session_type, sessions in info_data.get("sessions_by_type", {}).items()
        )
        + "\n"
        f"Number of trials: {info_data['n_trials']}\n"
    )
    ax_text.text(
        0.5,
        0.5,
        info,
        fontsize=18,
        ha="center",
        va="center",
        family="monospace",
    )

    for i, abl in enumerate(sorted(mean_rt["ABL"].unique())):
        sub = mean_rt[mean_rt["ABL"] == abl]
        ax_rt.errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {_display_abl_label(abl, sub['ILD'])}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )
    style_axes(
        ax_rt,
        title="Reaction Time",
        xlabel="ILD (dB)",
        ylabel="Mean RT (s)",
        box_aspect=None,
    )

    for i, abl in enumerate(sorted(mean_mt["ABL"].unique())):
        sub = mean_mt[mean_mt["ABL"] == abl]
        ax_mt.errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {_display_abl_label(abl, sub['ILD'])}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )
    style_axes(
        ax_mt,
        title="Movement Time",
        xlabel="ILD (dB)",
        ylabel="Mean MT (s)",
        box_aspect=None,
    )

    for color, (abl, res) in zip(colors, results.items()):
        ax_psy.scatter(
            DataHelpers.shift_ILD_for_ABL50(res["ILDs"]),
            res["PropLeft"],
            label=f"ABL={_display_abl_label(abl, res['ILDs'])}",
            color=color,
            s=70,
            edgecolor=color,
            linewidth=0.6,
            zorder=3,
        )
        if _has_psychometric_fit(res):
            ax_psy.plot(
                DataHelpers.shift_ILD_for_ABL50(res["xx"]),
                res["yy"],
                color=color,
                linewidth=1.8,
            )
    if (
        reference is not None
        and overlay_reference_psychometric
        and (
            not overlay_reference_only_when_fit
            or _has_any_psychometric_fit(results)
        )
    ):
        DataHelpers.overlay_makefig1_psychometrics(
            ax_psy,
            reference["psychometric"],
            color="black",
            show_individuals=True,
            use_abl_colors=False,
        )
    ax_psy.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    ax_psy.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    style_axes(
        ax_psy,
        title="Psychometric",
        xlabel="ILD (dB)",
        ylabel="Proportion Right",
        box_aspect=None,
    )
    xticks, xlim = _psychometric_ticks_and_limits(results)
    ax_psy.set_xticks(xticks)
    ax_psy.set_xticklabels([_format_psychometric_tick(x) for x in xticks])
    ax_psy.set_xlim(*xlim)
    ax_psy.legend(fontsize=13, frameon=False, loc="upper left")

    if not jnd.empty:
        ax_jnd = ax_psy.inset_axes([0.7, 0.12, 0.25, 0.25])
        ax_jnd.set_facecolor("white")
        full_abl_order = sorted(mean_rt["ABL"].unique())
        abl_colors = {
            abl: colors[i % len(colors)]
            for i, abl in enumerate(full_abl_order)
        }
        jnd_sorted = jnd.sort_values("ABL").reset_index(drop=True)
        for _, row in jnd_sorted.iterrows():
            ax_jnd.plot(
                row["ABL"],
                row["JND"],
                marker="o",
                linestyle="none",
                color=abl_colors.get(row["ABL"], "black"),
                markersize=7,
            )
        style_axes(
            ax_jnd,
            xlabel="ABL",
            ylabel="JND",
            title_fs=13,
            label_fs=12,
            tick_fs=10,
            title_pad=4,
            box_aspect=None,
        )
        ax_jnd.set_xticks(sorted(jnd["ABL"].unique()))
        ax_jnd.set_xticklabels([])
        ax_jnd.grid(True, linestyle="--", alpha=0.3, zorder=-10)

    return fig


def plot_sound_ramp_duration_comparison(
    prepared: dict[str, Any],
    *,
    figsize=(12, 4.5),
):
    apply_daily_plot_style()
    ramp_data = _prepare_sound_ramp_comparison_data(prepared)
    abls = ramp_data["abls"]
    info_data = prepared["info"]

    if not abls:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.axis("off")
        animal = info_data.get("animal") or "Animal"
        ax.text(
            0.5,
            0.5,
            (
                f"{animal}\n"
                "No sessions contain more than one non-NaN sound ramp duration\n"
                "for the current daily-review subset."
            ),
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    animal = info_data.get("animal") or "Animal"
    eligible_sessions = ramp_data["eligible_sessions"]
    fig.suptitle(
        f"{animal} · Sound Ramp Comparison ({len(eligible_sessions)} eligible sessions)",
        fontsize=18,
    )

    rt_summary = ramp_data["rt_summary"]
    prop_summary = ramp_data["prop_summary"]
    ramp_values = sorted(
        set(pd.to_numeric(rt_summary["sound_ramp_time_num"], errors="coerce").dropna().tolist())
        | set(pd.to_numeric(prop_summary["sound_ramp_time_num"], errors="coerce").dropna().tolist())
    )
    ramp_colors = {
        ramp: f"C{i % 10}"
        for i, ramp in enumerate(ramp_values)
    }

    ax_rt = axes[0, 0]
    ax_prop = axes[0, 1]

    for abl in abls:
        rt_sub = rt_summary[rt_summary["ABL"] == abl].copy()
        for ramp in ramp_values:
            ramp_rt = rt_sub[rt_sub["sound_ramp_time_num"] == ramp].copy()
            if ramp_rt.empty:
                continue
            ax_rt.errorbar(
                DataHelpers.shift_ILD_for_ABL50(ramp_rt["ILD"]),
                ramp_rt["mean"],
                yerr=ramp_rt["std"].fillna(0.0),
                linestyle="none",
                marker="o",
                color=ramp_colors[ramp],
                capsize=4,
                markersize=5,
                elinewidth=1.4,
                label=f"ramp={ramp:g}" if abl == abls[0] else None,
            )

        prop_sub = prop_summary[prop_summary["ABL"] == abl].copy()
        for ramp in ramp_values:
            ramp_prop = prop_sub[prop_sub["sound_ramp_time_num"] == ramp].copy()
            if ramp_prop.empty:
                continue
            ax_prop.errorbar(
                DataHelpers.shift_ILD_for_ABL50(ramp_prop["ILD"]),
                ramp_prop["mean"],
                yerr=ramp_prop["std"].fillna(0.0),
                linestyle="none",
                marker="o",
                color=ramp_colors[ramp],
                capsize=4,
                markersize=5,
                elinewidth=1.4,
                label=f"ramp={ramp:g}" if abl == abls[0] else None,
            )

    style_axes(
        ax_rt,
        title="RT",
        xlabel="ILD (dB)",
        ylabel="Mean RT (s)",
        title_fs=16,
        label_fs=13,
        tick_fs=11,
        box_aspect=None,
    )
    rt_ticks = sorted(set(DataHelpers.shift_ILD_for_ABL50(rt_summary["ILD"]))) if not rt_summary.empty else []
    ax_rt.set_xticks(rt_ticks)
    ax_rt.set_xticklabels([_format_psychometric_tick(x) for x in rt_ticks])
    ax_rt.grid(True, linestyle="--", alpha=0.25, zorder=-10)

    style_axes(
        ax_prop,
        title="Proportion Right",
        xlabel="ILD (dB)",
        ylabel="Proportion Right",
        title_fs=16,
        label_fs=13,
        tick_fs=11,
        box_aspect=None,
    )
    prop_ticks = sorted(set(DataHelpers.shift_ILD_for_ABL50(prop_summary["ILD"]))) if not prop_summary.empty else []
    ax_prop.set_xticks(prop_ticks)
    ax_prop.set_xticklabels([_format_psychometric_tick(x) for x in prop_ticks])
    ax_prop.set_ylim(-0.05, 1.05)
    ax_prop.grid(True, linestyle="--", alpha=0.25, zorder=-10)
    ax_prop.legend(fontsize=10, frameon=False, loc="best", title="Ramp")

    return fig


def _make_bins(values, bin_width, *, start=None, stop=None):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.array([0, bin_width])
    vmin = values.min() if start is None else start
    vmax = values.max() if stop is None else stop
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmax = vmin + bin_width
    return np.arange(vmin, vmax + bin_width, bin_width)


def plot_timing_histograms(
    prepared: dict[str, Any],
    *,
    subject_id: str | None = None,
    figsize=(16, 9),
    bin_width_ft=20,
    bin_width_cnp=0.5,
    bin_width_rt=0.01,
    bin_width_mt=0.01,
    bin_width_lnp=0.001,
):
    apply_daily_plot_style()

    df = prepared["df"].copy()
    df_valid = df[df["abort_type"] != "CNP"].copy()
    df_aborts = df[df["abort_type"] == "Fixation"].copy()
    subject_id = subject_id or prepared["info"].get("animal") or "Animal"

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    cols_top = ["fix_time", "intended_fix_time", "cnp_time"]
    cols_bottom = ["timed_rt", "timed_mt", "timed_lnp"]
    xlabel_map = {
        "fix_time": "Fixation Time (ms)",
        "intended_fix_time": "Intended Fixation Time (ms)",
        "cnp_time": "CNP Time (s)",
        "timed_rt": "Timed RT (s)",
        "timed_mt": "Timed MT (s)",
        "timed_lnp": "Timed LNP (s)",
    }

    bins_top = {
        "fix_time": _make_bins(df_valid["fix_time"], bin_width_ft, start=0),
        "intended_fix_time": _make_bins(df_valid["intended_fix_time"], bin_width_ft, start=0),
        "cnp_time": _make_bins(df_valid["cnp_time"], bin_width_cnp),
    }
    bins_bottom = {
        "timed_rt": _make_bins(df["timed_rt"], bin_width_rt),
        "timed_mt": _make_bins(df_valid["timed_mt"], bin_width_mt),
        "timed_lnp": _make_bins(df_valid["timed_lnp"], bin_width_lnp),
    }
    bin_labels = {
        "fix_time": bin_width_ft,
        "intended_fix_time": bin_width_ft,
        "cnp_time": bin_width_cnp,
        "timed_rt": bin_width_rt,
        "timed_mt": bin_width_mt,
        "timed_lnp": bin_width_lnp,
    }

    for ax, col in zip(axes[0, :], cols_top):
        ax.hist(
            df_valid[col].dropna(),
            bins=bins_top[col],
            histtype="step",
            color="black",
            linewidth=1.4,
            label="All initiated trials",
        )
        if col == "fix_time":
            ax.hist(
                df_aborts["fix_time"].dropna(),
                bins=bins_top[col],
                histtype="step",
                color="red",
                linewidth=1.4,
                label="Fixation aborts",
            )
            ax.legend(fontsize=11, frameon=False)

    for ax, col in zip(axes[1, :], cols_bottom):
        source = df if col == "timed_rt" else df_valid
        ax.hist(
            source[col].dropna(),
            bins=bins_bottom[col],
            histtype="step",
            color="black",
            linewidth=1.4,
        )

    for ax, col in zip(axes.flat, cols_top + cols_bottom):
        ax.set_xlabel(xlabel_map[col], fontsize=15)
        ax.set_ylabel("Count", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.3, color="black")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")
        ax.text(
            0.72,
            0.94,
            f"bin = {bin_labels[col]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )

    axes[0, 0].set_xlim(-2, 2000)
    axes[0, 2].set_xlim(-1, 10)
    axes[1, 0].set_xlim(-0.5, 0.5)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 2].set_xlim(0, 0.02)

    fig.suptitle(f"{subject_id}: Timing Distributions", fontsize=20, fontweight="bold")
    return fig


def plot_rt_histogram_by_abl(
    prepared: dict[str, Any],
    *,
    subject_id: str | None = None,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(7, 4),
    bin_width=0.01,
    exclude_abls=(50,),
):
    df = prepared["df"].copy()
    df_valid = df[df["abort_type"] != "CNP"].copy()
    df_rt = df_valid.dropna(subset=["timed_rt", "ABL"]).copy()
    if exclude_abls:
        df_rt = df_rt[~df_rt["ABL"].isin(exclude_abls)]
    subject_id = subject_id or prepared["info"].get("animal") or "Animal"

    bins = _make_bins(df_rt["timed_rt"], bin_width)
    fig, ax = plt.subplots(figsize=figsize)
    for i, abl in enumerate(sorted(df_rt["ABL"].unique())):
        df_abl = df_rt[df_rt["ABL"] == abl]
        subset = df_abl["timed_rt"]
        ax.hist(
            subset,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=colors[i % len(colors)],
            label=f"ABL {_display_abl_label(abl, df_abl['ILD'])}",
        )

    ax.set_xlabel("Reaction Time (s)", fontsize=15)
    ax.set_ylabel("Count", fontsize=15)
    ax.set_title(f"{subject_id}: Reaction Times by ABL", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 0.75)
    ax.legend(title="ABL", fontsize=12, frameon=False)
    ax.tick_params(axis="both", labelsize=12, width=1.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")
    fig.tight_layout()
    return fig


def run_daily_animal_plots(
    *,
    subject_file: str,
    line: str,
    cohort: str,
    training_level: int | None = 16,
    training_level_max: int | None = None,
    load_reference: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    df_raw, data_dir = load_daily_animal_data(
        subject_file=subject_file,
        line=line,
        cohort=cohort,
    )
    prepared = prepare_daily_animal_data(
        df_raw,
        training_level=training_level,
        training_level_max=training_level_max,
    )
    reference = load_reference_data() if load_reference else None
    figures = {
        "summary": plot_daily_animal_summary(prepared, reference=reference),
        "jnd": plot_daily_animal_jnd(prepared),
        "summary_with_jnd": plot_daily_animal_summary_with_jnd(prepared, reference=reference),
        "timing_histograms": plot_timing_histograms(prepared),
        "rt_by_abl_histogram": plot_rt_histogram_by_abl(prepared),
    }
    if show:
        plt.show()
    return {
        "data_dir": data_dir,
        "raw": df_raw,
        "prepared": prepared,
        "reference": reference,
        "figures": figures,
    }
