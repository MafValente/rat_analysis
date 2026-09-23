from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, minimize

import Helpers.DataHelpers as DataHelpers
from analysis import psychometric as Psychometric
from GroupComparison.config import FilterConfig, GroupComparisonConfig, OverlaySpec, PlotStyle, ViewSpec
from GroupComparison.layouts import plot_abls_4x3
from GroupComparison.plots import apply_50_tick_labels, plot_mt_on_ax, plot_psy_on_ax, plot_rt_on_ax, style_axes
from GroupComparison.prepare import (
    apply_filters,
    compute_group_jnd_by_view,
    compute_jnd_individuals_by_view,
    prep_mt,
    prep_psy,
    sem,
)


BIASED_SESSION_TYPES = (3, 23)
UNBIASED_RT_SESSION_TYPES = (1, 2)
SHORT_DURATION_VALUE = 0
PSYCHOMETRIC_L2 = 0.0
DEFAULT_UNBIASED_IMBALANCE = 0.35
BLOCK_ORDER = ["unbiased", "rightward", "leftward"]
BLOCK_COLORS = {
    "unbiased": "#4D4D4D",
    "rightward": "#1F77B4",
    "leftward": "#D62728",
}
BLOCK_STYLES = {
    "unbiased": {"linestyle": "-", "marker": "o", "markerfacecolor": None},
    "rightward": {"linestyle": "-", "marker": "s", "markerfacecolor": None},
    "leftward": {"linestyle": "-", "marker": "^", "markerfacecolor": None},
}
PSY_PARAM_SPECS = [
    ("slope_a", "Slope (a)"),
    ("bias_b", "Bias (b)"),
    ("lower_c", "Lower (c)"),
    ("upper_d", "Upper (d)"),
]


def default_config() -> GroupComparisonConfig:
    return GroupComparisonConfig(
        error_mode="individuals",
        skip_psy_fits=(50,),
        psychometric_aggregation="animal_trials",
        ild_shift_for_abl50=True,
        xlim_abs=(-18.5, 18.5),
    )


def default_filter_config() -> FilterConfig:
    return FilterConfig(
        training_min=16,
        session_min=0,
        drop_repeat_trials=True,
        session_type_values=None,
        stim_dur_values=None,
    )


def default_style() -> PlotStyle:
    return PlotStyle(title_fs=24, label_fs=25, tick_fs=24, legend_fs=16)


def _psychometric_prior() -> np.ndarray:
    return np.asarray([1.0, 0.0, 0.05, 0.95], dtype=float)


def _psychometric_scale() -> np.ndarray:
    return np.asarray([2.0, 10.0, 0.20, 0.20], dtype=float)


def _fit_psychometric_biased(
    x_data,
    y_data,
    *,
    n_trials=None,
    l2_strength: float = PSYCHOMETRIC_L2,
):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite]
    y_data = y_data[finite]
    if x_data.size < 4:
        raise ValueError("Need at least 4 finite ILD points for psychometric fit.")

    if n_trials is None:
        n_trials = np.full(x_data.shape, 50.0, dtype=float)
    else:
        n_trials = np.asarray(n_trials, dtype=float)[finite]
        n_trials = np.clip(n_trials, 1.0, None)

    lower_bounds = np.asarray([0.1, float(np.min(x_data)), 0.0, 0.5], dtype=float)
    upper_bounds = np.asarray([10.0, float(np.max(x_data)), 0.5, 1.0], dtype=float)
    prior = np.clip(_psychometric_prior(), lower_bounds, upper_bounds)
    scale = _psychometric_scale()

    if float(l2_strength) <= 0:
        return Psychometric.fit_and_plot_psychometric(
            x_data,
            y_data,
            model="my_psycho",
            n_trials=n_trials.astype(int),
            show_plot=False,
        )

    def objective(pars):
        pred = Psychometric.my_psycho_model(x_data, *pars)
        resid = pred - y_data
        data_term = np.sum(n_trials * resid * resid)
        penalty = float(l2_strength) * np.sum(((pars - prior) / scale) ** 2)
        return float(data_term + penalty)

    result = minimize(
        objective,
        x0=prior,
        method="L-BFGS-B",
        bounds=list(zip(lower_bounds, upper_bounds)),
    )
    pars = np.asarray(result.x if result.success else prior, dtype=float)
    xx = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 200)
    yy = Psychometric.my_psycho_model(xx, *pars)
    return pars, np.nan, xx, yy


def _compute_psychometrics_by_ABL_local(
    df_last: pd.DataFrame,
    *,
    l2_strength: float = PSYCHOMETRIC_L2,
    min_ilds_for_fit: int = 4,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for abl in sorted(df_last["ABL"].dropna().unique()):
        df_sub = df_last[df_last["ABL"] == abl]
        ilds = np.sort(pd.to_numeric(df_sub["ILD"], errors="coerce").dropna().unique())
        if len(ilds) == 0:
            continue
        prop_left = np.array([
            (((df_sub["ILD"] == ild) & (pd.to_numeric(df_sub["success"], errors="coerce") == 1)).sum()) /
            max(1, (((df_sub["ILD"] == ild) & (pd.to_numeric(df_sub["success"], errors="coerce") != 0)).sum()))
            for ild in ilds
        ], dtype=float)
        prop_left = np.where(ilds < 0, 1 - prop_left, prop_left)
        n_trials = np.array([(df_sub["ILD"] == ild).sum() for ild in ilds], dtype=int)
        pars, L, xx, yy = None, np.nan, None, None
        if len(ilds) >= min_ilds_for_fit and np.isfinite(prop_left).all():
            try:
                pars, L, xx, yy = _fit_psychometric_biased(
                    ilds,
                    prop_left,
                    n_trials=n_trials,
                    l2_strength=l2_strength,
                )
            except Exception:
                pars, L, xx, yy = None, np.nan, None, None
        results[int(abl)] = {
            "ILDs": ilds,
            "PropLeft": prop_left,
            "n_trials": n_trials,
            "pars": pars,
            "L": L,
            "xx": xx,
            "yy": yy,
        }
    return results


def _prep_psy_local(
    df_in: pd.DataFrame,
    do_individual_fits: bool,
    *,
    aggregation: str = "animal_trials",
    skip_jnd_abl: int = 50,
    l2_strength: float = PSYCHOMETRIC_L2,
):
    all_pts: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    per_subject_curves: dict[tuple, dict[str, Any]] = {}
    jnd_rows: list[dict[str, Any]] = []
    valid_aggregations = {"animal_trials", "session_then_animal"}
    if aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {sorted(valid_aggregations)}")

    for subject, df_subj in df_in.groupby("animal", sort=False):
        if aggregation == "session_then_animal" and "session" in df_subj.columns:
            session_rows: list[dict[str, Any]] = []
            for _, df_sess in df_subj.groupby("session", sort=False):
                results = _compute_psychometrics_by_ABL_local(df_sess, l2_strength=l2_strength)
                for abl, res in results.items():
                    for ild, val in zip(np.asarray(res["ILDs"]), np.asarray(res["PropLeft"], dtype=float)):
                        session_rows.append(
                            {
                                "subject": subject,
                                "session": df_sess["session"].iloc[0] if "session" in df_sess.columns and not df_sess.empty else pd.NA,
                                "ABL": int(abl),
                                "ILD": float(ild),
                                "PropLeft": float(val),
                            }
                        )
            if session_rows:
                session_points = pd.DataFrame(session_rows)
                session_points["ABL"] = pd.to_numeric(session_points["ABL"], errors="coerce")
                session_points["ILD"] = pd.to_numeric(session_points["ILD"], errors="coerce")
                session_points["PropLeft"] = pd.to_numeric(session_points["PropLeft"], errors="coerce")
                session_points = session_points.dropna(subset=["ABL", "ILD", "PropLeft"]).copy()
                session_points["ABL"] = session_points["ABL"].astype(int)
                subj_points = (
                    session_points.groupby(["subject", "ABL", "ILD"])["PropLeft"]
                    .agg(mean="mean", sem=sem, n="count")
                    .reset_index()
                )
            else:
                subj_points = pd.DataFrame(columns=["subject", "ABL", "ILD", "mean", "sem", "n"])

            results = {}
            for abl, df_abl in subj_points.groupby("ABL", sort=False):
                df_abl = df_abl.sort_values("ILD")
                ilds = df_abl["ILD"].to_numpy(dtype=float)
                mean_vals = df_abl["mean"].to_numpy(dtype=float)
                n_trials = np.full_like(ilds, 50)
                pars, L, xx, yy = None, np.nan, None, None
                if len(ilds) >= 4 and np.isfinite(mean_vals).all():
                    try:
                        pars, L, xx, yy = _fit_psychometric_biased(
                            ilds,
                            mean_vals,
                            n_trials=n_trials,
                            l2_strength=l2_strength,
                        )
                    except Exception:
                        pars, L, xx, yy = None, np.nan, None, None
                results[int(abl)] = {
                    "ILDs": ilds,
                    "PropLeft": mean_vals,
                    "n_trials": n_trials,
                    "pars": pars,
                    "L": L,
                    "xx": xx,
                    "yy": yy,
                }
        else:
            results = _compute_psychometrics_by_ABL_local(df_subj, l2_strength=l2_strength)

        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_jnd_abl)
        if jnd_df is not None and not jnd_df.empty:
            for _, r in jnd_df.iterrows():
                jnd_rows.append({"subject": subject, "ABL": int(r["ABL"]), "JND": float(r["JND"])})

        if aggregation == "session_then_animal":
            for _, row in subj_points.iterrows():
                all_pts.append(
                    {
                        "subject": subject,
                        "ABL": int(row["ABL"]),
                        "ILD": float(row["ILD"]),
                        "PropLeft": float(row["mean"]),
                    }
                )
        else:
            for abl, res in results.items():
                for ild, val in zip(np.asarray(res["ILDs"]), np.asarray(res["PropLeft"], dtype=float)):
                    all_pts.append({"subject": subject, "ABL": int(abl), "ILD": float(ild), "PropLeft": float(val)})

        if do_individual_fits:
            for abl, res in results.items():
                if res.get("xx") is not None and res.get("yy") is not None:
                    per_subject_curves[(subject, int(abl))] = dict(xx=res["xx"], yy=res["yy"])

        for abl, res in results.items():
            pars = res.get("pars")
            if pars is None or len(pars) < 4 or not np.all(np.isfinite(np.asarray(pars[:4], dtype=float))):
                continue
            param_rows.append(
                {
                    "animal": subject,
                    "line": df_subj["line"].iloc[0] if "line" in df_subj.columns and not df_subj.empty else pd.NA,
                    "cohort": df_subj["cohort"].iloc[0] if "cohort" in df_subj.columns and not df_subj.empty else pd.NA,
                    "genotype": df_subj["genotype"].iloc[0] if "genotype" in df_subj.columns and not df_subj.empty else pd.NA,
                    "dataset_key": df_subj["dataset_key"].iloc[0] if "dataset_key" in df_subj.columns and not df_subj.empty else pd.NA,
                    "ABL": int(abl),
                    "slope_a": float(pars[0]),
                    "bias_b": float(pars[1]),
                    "lower_c": float(pars[2]),
                    "upper_d": float(pars[3]),
                }
            )

    points = pd.DataFrame(all_pts, columns=["subject", "ABL", "ILD", "PropLeft"])
    jnd_indiv = pd.DataFrame(jnd_rows, columns=["subject", "ABL", "JND"])
    params = pd.DataFrame(
        param_rows,
        columns=["animal", "line", "cohort", "genotype", "dataset_key", "ABL", "slope_a", "bias_b", "lower_c", "upper_d"],
    )
    if points.empty:
        psy_group = pd.DataFrame(columns=["ABL", "ILD", "mean", "sem", "n"])
        return points, psy_group, per_subject_curves, {}, jnd_indiv, params

    psy_group = (
        points.groupby(["ABL", "ILD"])["PropLeft"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    mean_fits: dict[int, dict[str, Any] | None] = {}
    for abl in sorted(psy_group["ABL"].unique()):
        sub = psy_group[psy_group["ABL"] == abl]
        ilds = sub["ILD"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        if len(ilds) < 4 or not np.isfinite(y).all():
            mean_fits[int(abl)] = None
            continue
        n_trials = np.full_like(ilds, 50)
        try:
            _, _, xx, yy = _fit_psychometric_biased(
                ilds,
                y,
                n_trials=n_trials,
                l2_strength=l2_strength,
            )
            mean_fits[int(abl)] = dict(xx=xx, yy=yy)
        except Exception:
            mean_fits[int(abl)] = None
    return points, psy_group, per_subject_curves, mean_fits, jnd_indiv, params


def _numeric_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _match_numeric_filter(series: pd.Series, values: Any) -> pd.Series:
    if values is None:
        return pd.Series(True, index=series.index, dtype=bool)
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return series.isin(list(values))
    return series.eq(values)


def _session_key_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ["dataset_key", "animal", "session"] if c in df.columns]


def _block_valid_mask(block_df: pd.DataFrame) -> pd.Series:
    if "success" not in block_df.columns:
        return pd.Series(True, index=block_df.index, dtype=bool)
    success = pd.to_numeric(block_df["success"], errors="coerce")
    return success.ne(0)


def _contiguous_block_runs(session_df: pd.DataFrame) -> list[pd.Index]:
    if session_df.empty:
        return []

    if "trial" in session_df.columns:
        ordered = session_df.sort_values("trial", kind="stable")
    else:
        ordered = session_df.sort_index(kind="stable")

    if "block" not in ordered.columns or not ordered["block"].notna().any():
        return [ordered.index]

    block_values = pd.to_numeric(ordered["block"], errors="coerce")
    transitions = block_values.ne(block_values.shift())
    transitions.iloc[0] = True
    run_ids = transitions.cumsum()
    return [sub.index for _, sub in ordered.groupby(run_ids, sort=False, dropna=False)]


def _session_block_number(block_df: pd.DataFrame) -> int | pd._libs.missing.NAType:
    if "block" not in block_df.columns:
        return pd.NA
    block_number = pd.to_numeric(block_df["block"], errors="coerce").dropna()
    if block_number.empty:
        return pd.NA
    return int(block_number.iloc[0])


def _looks_like_balanced_unbiased_block(block_df: pd.DataFrame) -> bool:
    ild = pd.to_numeric(block_df.get("ILD"), errors="coerce")
    ild = ild[ild.notna() & ild.ne(0)]
    if ild.empty:
        return False

    counts = ild.value_counts().sort_index()
    if counts.empty or counts.nunique() != 1:
        return False

    ild_values = counts.index.to_numpy(dtype=float)
    if len(ild_values) % 2 != 0:
        return False

    negatives = np.sort(ild_values[ild_values < 0])
    positives = np.sort(ild_values[ild_values > 0])
    if len(negatives) == 0 or len(positives) == 0 or len(negatives) != len(positives):
        return False
    if not np.array_equal(np.abs(negatives), positives):
        return False

    return True


def classify_block_condition(
    block_df: pd.DataFrame,
    *,
    rightward_ild_sign: int = 1,
    min_imbalance: float = 0.0,
    max_unbiased_imbalance: float = DEFAULT_UNBIASED_IMBALANCE,
) -> str | pd._libs.missing.NAType:
    valid_mask = _block_valid_mask(block_df)
    valid_count = int(valid_mask.sum())
    ild = pd.to_numeric(block_df.loc[valid_mask, "ILD"], errors="coerce")
    ild = ild[ild.notna() & ild.ne(0)]
    if ild.empty:
        return pd.NA

    signed_imbalance = np.sign(ild).mean()
    if not np.isfinite(signed_imbalance):
        return pd.NA

    if abs(signed_imbalance) <= float(max_unbiased_imbalance):
        if "trials_per_block" in block_df.columns:
            target = pd.to_numeric(block_df["trials_per_block"], errors="coerce").dropna()
            if not target.empty:
                target_trials = float(target.iloc[0])
                if np.isfinite(target_trials) and target_trials > 0 and valid_count > (1.5 * target_trials):
                    return pd.NA
        return "unbiased"

    if abs(signed_imbalance) <= min_imbalance:
        return pd.NA

    block_sign = 1 if signed_imbalance > 0 else -1
    return "rightward" if block_sign == int(rightward_ild_sign) else "leftward"


def assign_session_block_conditions(
    session_df: pd.DataFrame,
    *,
    rightward_ild_sign: int = 1,
    min_imbalance: float = 0.0,
    max_unbiased_imbalance: float = DEFAULT_UNBIASED_IMBALANCE,
) -> list[tuple[pd.Index, str | pd._libs.missing.NAType]]:
    assignments: list[tuple[pd.Index, str | pd._libs.missing.NAType]] = []
    for block_idx in _contiguous_block_runs(session_df):
        block_df = session_df.loc[block_idx]
        block_number = _session_block_number(block_df)
        if pd.notna(block_number) and int(block_number) == 1:
            condition = "unbiased"
        elif _looks_like_balanced_unbiased_block(block_df):
            condition = "unbiased"
        else:
            condition = classify_block_condition(
                block_df,
                rightward_ild_sign=rightward_ild_sign,
                min_imbalance=min_imbalance,
                max_unbiased_imbalance=max_unbiased_imbalance,
            )
        assignments.append((block_idx, condition))
    return assignments


def add_biased_block_condition(
    df: pd.DataFrame,
    *,
    biased_session_types: tuple[int, ...] = BIASED_SESSION_TYPES,
    unbiased_rt_session_types: tuple[int, ...] = UNBIASED_RT_SESSION_TYPES,
    short_duration_value: Any = SHORT_DURATION_VALUE,
    rightward_ild_sign: int = 1,
    min_direction_imbalance: float = 0.0,
    max_unbiased_imbalance: float = DEFAULT_UNBIASED_IMBALANCE,
) -> pd.DataFrame:
    df = df.copy()
    sess = _numeric_col(df, "session_type")
    short_duration = _numeric_col(df, "short_duration")
    is_rt = _match_numeric_filter(short_duration, short_duration_value)

    df["block_condition"] = pd.NA
    unbiased_rt = sess.isin(unbiased_rt_session_types) & is_rt
    df.loc[unbiased_rt, "block_condition"] = "unbiased"

    biased_rt = sess.isin(biased_session_types) & is_rt
    key_cols = _session_key_columns(df)
    if not key_cols:
        raise KeyError("Need at least one session key column such as animal/session to label biased blocks.")

    for _, session_df in df[biased_rt].groupby(key_cols, sort=False, dropna=False):
        if session_df.empty:
            continue

        for block_idx, condition in assign_session_block_conditions(
            session_df,
            rightward_ild_sign=rightward_ild_sign,
            min_imbalance=min_direction_imbalance,
            max_unbiased_imbalance=max_unbiased_imbalance,
        ):
            if pd.notna(condition):
                df.loc[block_idx, "block_condition"] = condition

    return df[df["block_condition"].notna()].copy()


def prepare_biased_blocks(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    cfg: GroupComparisonConfig | None = None,
    fcfg: FilterConfig | None = None,
    style: PlotStyle | None = None,
    biased_session_types: tuple[int, ...] = BIASED_SESSION_TYPES,
    unbiased_rt_session_types: tuple[int, ...] = UNBIASED_RT_SESSION_TYPES,
    short_duration_value: Any = SHORT_DURATION_VALUE,
    rightward_ild_sign: int = 1,
    min_direction_imbalance: float = 0.0,
    max_unbiased_imbalance: float = DEFAULT_UNBIASED_IMBALANCE,
    keep_only_animals_with_biased_sessions: bool = True,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> dict[str, Any]:
    cfg = cfg or default_config()
    fcfg = fcfg or default_filter_config()
    style = style or default_style()

    # Preserve trial identity so temporal layouts can use full-session order even
    # when the plotted analysis is restricted to a duration subset.
    df = df.copy()
    df["_biased_blocks_row_id"] = np.arange(len(df), dtype=np.int64)

    prefilter_session_types = set(unbiased_rt_session_types) | set(biased_session_types)
    prefilter_sess = _numeric_col(df, "session_type")
    df_prefiltered = df[prefilter_sess.isin(prefilter_session_types)].copy()

    if keep_only_animals_with_biased_sessions and "animal" in df_prefiltered.columns:
        biased_animals = set(
            df_prefiltered.loc[prefilter_sess.loc[df_prefiltered.index].isin(biased_session_types), "animal"]
            .dropna()
            .astype(str)
        )
        df_prefiltered = df_prefiltered[df_prefiltered["animal"].astype(str).isin(biased_animals)].copy()

    df_filtered_timing = apply_filters(df_prefiltered, fcfg)
    df_blocks_timing = add_biased_block_condition(
        df_filtered_timing,
        biased_session_types=biased_session_types,
        unbiased_rt_session_types=unbiased_rt_session_types,
        short_duration_value=None,
        rightward_ild_sign=rightward_ild_sign,
        min_direction_imbalance=min_direction_imbalance,
        max_unbiased_imbalance=max_unbiased_imbalance,
    )
    duration_mask = _match_numeric_filter(
        _numeric_col(df_blocks_timing, "short_duration"),
        short_duration_value,
    )
    df_blocks = df_blocks_timing[duration_mask].copy()
    df_filtered = df_filtered_timing[
        _match_numeric_filter(_numeric_col(df_filtered_timing, "short_duration"), short_duration_value)
    ].copy()

    summary_cols = [
        c
        for c in ["line", "cohort", "genotype", "animal", "session_type", "block_condition"]
        if c in df_blocks.columns
    ]
    block_summary = (
        df_blocks.groupby(summary_cols, dropna=False)
        .size()
        .rename("trials")
        .reset_index()
        .sort_values(summary_cols)
        if summary_cols
        else pd.DataFrame({"trials": [len(df_blocks)]})
    )

    return {
        "df_filtered": df_filtered,
        "df_blocks": df_blocks,
        "df_blocks_timing": df_blocks_timing,
        "block_summary": block_summary,
        "views": views,
        "cfg": cfg,
        "fcfg": fcfg,
        "style": style,
        "short_duration_value": short_duration_value,
        "psychometric_l2": float(psychometric_l2),
        "max_unbiased_imbalance": float(max_unbiased_imbalance),
    }


def prep_rt_signed(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_s = df_in[df_in["success"] == 1].copy()
    per_subj = (
        df_s.groupby(["animal", "ABL", "ILD"])["timed_rt"]
        .agg(mean_rt="mean")
        .reset_index()
    )
    grouped = (
        per_subj.groupby(["ABL", "ILD"])
        .agg(mean=("mean_rt", "mean"), sem=("mean_rt", sem), n=("mean_rt", "count"))
        .reset_index()
    )
    return per_subj, grouped


def build_prepared_signed_rt(
    df: pd.DataFrame,
    selected_views: list[ViewSpec],
    cfg: GroupComparisonConfig,
    *,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for view in selected_views:
        df_v = view.selector(df)
        rt_per_subj, rt_group = prep_rt_signed(df_v)
        mt_per_subj, mt_group = prep_mt(df_v)
        with contextlib.redirect_stdout(io.StringIO()):
            if float(psychometric_l2) > 0:
                psy_points, psy_group, psy_indiv, psy_mean, jnd_indiv, psy_params = _prep_psy_local(
                    df_v,
                    do_individual_fits=(cfg.error_mode == "individuals"),
                    aggregation=cfg.psychometric_aggregation,
                    skip_jnd_abl=50,
                    l2_strength=psychometric_l2,
                )
            else:
                psy_points, psy_group, psy_indiv, psy_mean, jnd_indiv, psy_params = prep_psy(
                    df_v,
                    do_individual_fits=(cfg.error_mode == "individuals"),
                    aggregation=cfg.psychometric_aggregation,
                    skip_jnd_abl=50,
                )
        prepared[view.name] = dict(
            rt_per_subj=rt_per_subj,
            rt_group=rt_group,
            mt_per_subj=mt_per_subj,
            mt_group=mt_group,
            psy_points=psy_points,
            psy_group=psy_group,
            psy_indiv_curves=psy_indiv,
            psy_mean_fits=psy_mean,
            jnd_indiv=jnd_indiv,
            psy_params=psy_params,
            df_view=df_v,
        )
    return prepared


def make_block_views(df: pd.DataFrame) -> list[ViewSpec]:
    present = [name for name in BLOCK_ORDER if name in set(df["block_condition"].dropna().astype(str))]
    return [
        ViewSpec(name, lambda d, _name=name: d[d["block_condition"].astype(str) == _name].copy())
        for name in present
    ]


def _format_signed_rt_axes(fig: plt.Figure, cfg: GroupComparisonConfig) -> None:
    for ax in fig.axes[0::3]:
        _set_signed_ild_ticks(ax, cfg)


def _set_signed_ild_ticks(ax: plt.Axes, cfg: GroupComparisonConfig) -> None:
    ax.set_xlim(*cfg.xlim_sym)
    ticks = [-18, -15, -10, -5, 0, 5, 10, 15, 18]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["-50", "-15", "-10", "-5", "0", "5", "10", "15", "50"])


def _replace_figure_legend_at_bottom(
    fig: plt.Figure,
    labels: list[str],
    colors: dict[str, str],
    styles: dict[str, dict] | None,
    style: PlotStyle,
) -> None:
    for legend in list(fig.legends):
        legend.remove()
    styles = styles or {}
    handles = []
    for label in labels:
        style_cfg = styles.get(label, {})
        color = colors.get(label, "gray")
        markerfacecolor = style_cfg.get("markerfacecolor")
        if markerfacecolor is None:
            markerfacecolor = color
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=style_cfg.get("marker", "o"),
                linestyle=style_cfg.get("linestyle", "None"),
                markerfacecolor=markerfacecolor,
                markeredgecolor=color,
            )
        )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        bbox_transform=fig.transFigure,
        ncol=min(6, max(1, len(labels))),
        fontsize=style.legend_fs,
        frameon=False,
    )


def _safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(":", "_").replace(" ", "_")


def _plot_prepared_abls(
    *,
    prepared: dict[str, dict[str, Any]],
    views: list[ViewSpec],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    group_jnd: dict[str, pd.DataFrame],
    view_colors: dict[str, str],
    view_styles: dict[str, dict],
    title: str,
) -> plt.Figure:
    view_names = [v.name for v in views]
    fig = plot_abls_4x3(
        prepared=prepared,
        views=views,
        cfg=cfg,
        style=style,
        overlay=OverlaySpec(),
        view_colors={name: view_colors.get(name, f"C{i % 10}") for i, name in enumerate(view_names)},
        group_jnd_by_view=group_jnd,
        add_inset=True,
        view_styles={name: view_styles.get(name, {}) for name in view_names},
    )
    fig.suptitle(title, fontsize=26, y=0.995)
    _replace_figure_legend_at_bottom(
        fig,
        view_names,
        {name: view_colors.get(name, f"C{i % 10}") for i, name in enumerate(view_names)},
        {name: view_styles.get(name, {}) for name in view_names},
        style,
    )
    _format_signed_rt_axes(fig, cfg)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    return fig


MEAN_ABL_SYNTHETIC = 999
MEAN_ABL_TARGETS = (20, 40, 60)


def _fit_mean_psy_curve(
    points_df: pd.DataFrame,
    *,
    psychometric_l2: float,
) -> dict[int, dict[str, Any]]:
    if points_df.empty:
        return {}
    sub = points_df.sort_values("ILD").copy()
    x = pd.to_numeric(sub["ILD"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 4:
        return {}
    n_trials = np.full(len(x), 50, dtype=int)
    try:
        if float(psychometric_l2) > 0:
            pars, _, xx, yy = _fit_psychometric_biased(x, y, n_trials=n_trials, l2_strength=psychometric_l2)
        else:
            pars, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                x, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
    except Exception:
        return {}
    return {MEAN_ABL_SYNTHETIC: {"xx": xx, "yy": yy, "pars": pars}}


def _collapse_tables_over_abls(
    tables: dict[str, Any],
    *,
    include_abls: tuple[int, ...] = MEAN_ABL_TARGETS,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> dict[str, Any]:
    include_abls = tuple(int(a) for a in include_abls)

    rt_per_subj = tables.get("rt_per_subj", pd.DataFrame()).copy()
    rt_per_subj = rt_per_subj[pd.to_numeric(rt_per_subj.get("ABL"), errors="coerce").isin(include_abls)].copy()
    if not rt_per_subj.empty:
        rt_per_subj = (
            rt_per_subj.groupby(["animal", "ILD"], dropna=False)["mean_rt"]
            .mean()
            .rename("mean_rt")
            .reset_index()
        )
        rt_per_subj["ABL"] = MEAN_ABL_SYNTHETIC
        rt_group = (
            rt_per_subj.groupby(["ABL", "ILD"], dropna=False)["mean_rt"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
    else:
        rt_per_subj = pd.DataFrame(columns=["animal", "ILD", "mean_rt", "ABL"])
        rt_group = pd.DataFrame(columns=["ABL", "ILD", "mean", "sem", "n"])

    mt_per_subj = tables.get("mt_per_subj", pd.DataFrame()).copy()
    mt_per_subj = mt_per_subj[pd.to_numeric(mt_per_subj.get("ABL"), errors="coerce").isin(include_abls)].copy()
    if not mt_per_subj.empty:
        mt_per_subj = (
            mt_per_subj.groupby(["animal", "ILD"], dropna=False)["mean_mt"]
            .mean()
            .rename("mean_mt")
            .reset_index()
        )
        mt_per_subj["ABL"] = MEAN_ABL_SYNTHETIC
        mt_group = (
            mt_per_subj.groupby(["ABL", "ILD"], dropna=False)["mean_mt"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
    else:
        mt_per_subj = pd.DataFrame(columns=["animal", "ILD", "mean_mt", "ABL"])
        mt_group = pd.DataFrame(columns=["ABL", "ILD", "mean", "sem", "n"])

    psy_points = tables.get("psy_points", pd.DataFrame()).copy()
    psy_points = psy_points[pd.to_numeric(psy_points.get("ABL"), errors="coerce").isin(include_abls)].copy()
    if not psy_points.empty:
        psy_points = (
            psy_points.groupby(["subject", "ILD"], dropna=False)["PropLeft"]
            .mean()
            .rename("PropLeft")
            .reset_index()
        )
        psy_points["ABL"] = MEAN_ABL_SYNTHETIC
        psy_group = (
            psy_points.groupby(["ABL", "ILD"], dropna=False)["PropLeft"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
        psy_indiv_curves = {}
        for subject, df_sub in psy_points.groupby("subject", dropna=False, sort=False):
            fit_input = df_sub.rename(columns={"PropLeft": "mean"})
            fit = _fit_mean_psy_curve(fit_input, psychometric_l2=psychometric_l2)
            if fit:
                psy_indiv_curves[(subject, MEAN_ABL_SYNTHETIC)] = fit[MEAN_ABL_SYNTHETIC]
        psy_mean_fits = _fit_mean_psy_curve(psy_group, psychometric_l2=psychometric_l2)
    else:
        psy_points = pd.DataFrame(columns=["subject", "ILD", "PropLeft", "ABL"])
        psy_group = pd.DataFrame(columns=["ABL", "ILD", "mean", "sem", "n"])
        psy_indiv_curves = {}
        psy_mean_fits = {}

    return {
        "rt_per_subj": rt_per_subj,
        "rt_group": rt_group,
        "mt_per_subj": mt_per_subj,
        "mt_group": mt_group,
        "psy_points": psy_points,
        "psy_group": psy_group,
        "psy_indiv_curves": psy_indiv_curves,
        "psy_mean_fits": psy_mean_fits,
        "jnd_indiv": pd.DataFrame(),
        "psy_params": pd.DataFrame(),
        "df_view": tables.get("df_view", pd.DataFrame()).copy(),
    }


def _plot_genotype_mean_abl_summary(
    *,
    collapsed_prepared_by_view: dict[str, dict[str, dict[str, Any]]],
    views: list[ViewSpec],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
) -> plt.Figure | None:
    view_names = [v.name for v in views if v.name in collapsed_prepared_by_view]
    if not view_names:
        return None

    fig, axes = plt.subplots(len(view_names), 3, figsize=(18, 4.8 * len(view_names)), squeeze=False, sharex="col")

    for r, view_name in enumerate(view_names):
        ax_rt, ax_mt, ax_psy = axes[r]
        prepared = collapsed_prepared_by_view[view_name]
        block_names = [name for name in BLOCK_ORDER if name in prepared]
        if not block_names:
            continue
        for block_name in block_names:
            tables = prepared[block_name]
            color = BLOCK_COLORS.get(block_name, "gray")
            style_cfg = BLOCK_STYLES.get(block_name, {})
            plot_rt_on_ax(ax_rt, tables, MEAN_ABL_SYNTHETIC, color, cfg, **style_cfg)
            plot_mt_on_ax(ax_mt, tables, MEAN_ABL_SYNTHETIC, color, cfg, **style_cfg)
            plot_psy_on_ax(ax_psy, tables, MEAN_ABL_SYNTHETIC, color, cfg, **style_cfg)

        style_axes(ax_rt, style, f"{view_name} - RT", "ILD (dB)", "Mean RT (s)")
        style_axes(ax_mt, style, f"{view_name} - MT", "ILD (dB)", "Mean MT (s)")
        style_axes(ax_psy, style, f"{view_name} - Psychometric", "ILD (dB)", "P(Left)")

        ax_rt.set_xlim(*cfg.xlim_sym)
        ax_rt.set_ylim(*cfg.ylim_rt)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_mt.set_ylim(*cfg.ylim_mt)
        ax_psy.set_xlim(*cfg.xlim_sym)
        _set_signed_ild_ticks(ax_rt, cfg)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

    _replace_figure_legend_at_bottom(
        fig,
        [name for name in BLOCK_ORDER if any(name in collapsed_prepared_by_view[vn] for vn in view_names)],
        BLOCK_COLORS,
        BLOCK_STYLES,
        style,
    )
    fig.suptitle("Biased blocks - mean of ABLs 20, 40, 60", fontsize=26, y=0.995)
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    return fig


def plot_genotype_block_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    df_blocks = bundle["df_blocks"]
    cfg = bundle["cfg"]
    style = bundle["style"]
    psychometric_l2 = float(bundle.get("psychometric_l2", PSYCHOMETRIC_L2))
    views = views or bundle["views"]

    outputs = {}
    collapsed_prepared_by_view: dict[str, dict[str, dict[str, Any]]] = {}
    for view in views:
        df_view = view.selector(df_blocks)
        if df_view.empty:
            continue
        block_views = make_block_views(df_view)
        prepared = build_prepared_signed_rt(df_view, block_views, cfg, psychometric_l2=psychometric_l2)
        collapsed_prepared = {
            block_view.name: _collapse_tables_over_abls(
                prepared.get(block_view.name, {}),
                include_abls=MEAN_ABL_TARGETS,
                psychometric_l2=psychometric_l2,
            )
            for block_view in block_views
        }
        collapsed_prepared_by_view[view.name] = collapsed_prepared
        jnd_indiv = compute_jnd_individuals_by_view(prepared, skip_abl=50)
        group_jnd = compute_group_jnd_by_view(jnd_indiv)
        block_names = [v.name for v in block_views]
        fig = _plot_prepared_abls(
            prepared=prepared,
            views=block_views,
            cfg=cfg,
            style=style,
            group_jnd=group_jnd,
            view_colors={name: BLOCK_COLORS[name] for name in block_names},
            view_styles={name: BLOCK_STYLES[name] for name in block_names},
            title=f"{view.name} - biased blocks",
        )
        outputs[view.name] = {
            "figure": fig,
            "prepared": prepared,
            "jnd_indiv": jnd_indiv,
            "group_jnd": group_jnd,
            "counts": df_view["block_condition"].value_counts().to_dict(),
        }
        if show:
            plt.show()
    mean_abl_figure = _plot_genotype_mean_abl_summary(
        collapsed_prepared_by_view=collapsed_prepared_by_view,
        views=views,
        cfg=cfg,
        style=style,
    )
    if mean_abl_figure is not None:
        outputs["mean_abl_summary"] = {
            "figure": mean_abl_figure,
            "prepared": collapsed_prepared_by_view,
            "abls": MEAN_ABL_TARGETS,
        }
        if show:
            plt.show()
    return outputs


def plot_animal_block_figures(
    bundle: dict[str, Any],
    *,
    max_animals: int | None = None,
    show: bool = True,
) -> dict[str, Any]:
    df_blocks = bundle["df_blocks"]
    cfg = bundle["cfg"]
    style = bundle["style"]
    psychometric_l2 = float(bundle.get("psychometric_l2", PSYCHOMETRIC_L2))

    outputs = {}
    animals = sorted(df_blocks["animal"].dropna().astype(str).unique())
    if max_animals is not None:
        animals = animals[:max_animals]

    for animal in animals:
        df_animal = df_blocks[df_blocks["animal"].astype(str) == animal].copy()
        if df_animal.empty:
            continue
        genotype = (
            df_animal["genotype"].dropna().astype(str).iloc[0]
            if "genotype" in df_animal and df_animal["genotype"].notna().any()
            else ""
        )
        block_views = make_block_views(df_animal)
        prepared = build_prepared_signed_rt(df_animal, block_views, cfg, psychometric_l2=psychometric_l2)
        jnd_indiv = compute_jnd_individuals_by_view(prepared, skip_abl=50)
        group_jnd = compute_group_jnd_by_view(jnd_indiv)
        block_names = [v.name for v in block_views]
        fig = _plot_prepared_abls(
            prepared=prepared,
            views=block_views,
            cfg=cfg,
            style=style,
            group_jnd=group_jnd,
            view_colors={name: BLOCK_COLORS[name] for name in block_names},
            view_styles={name: BLOCK_STYLES[name] for name in block_names},
            title=f"{animal} {genotype} - biased blocks",
        )
        outputs[animal] = {
            "figure": fig,
            "prepared": prepared,
            "jnd_indiv": jnd_indiv,
            "group_jnd": group_jnd,
            "counts": df_animal["block_condition"].value_counts().to_dict(),
        }
        if show:
            plt.show()
    return outputs


def plot_block_condition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    view_colors: dict[str, str] | None = None,
    view_styles: dict[str, dict] | None = None,
    block_conditions: list[str] | tuple[str, ...] = ("rightward", "leftward", "unbiased"),
    show: bool = True,
) -> dict[str, Any]:
    df_blocks = bundle["df_blocks"]
    cfg = bundle["cfg"]
    style = bundle["style"]
    psychometric_l2 = float(bundle.get("psychometric_l2", PSYCHOMETRIC_L2))
    views = views or bundle["views"]
    view_colors = view_colors or {}
    view_styles = view_styles or {}

    outputs = {}
    for condition in block_conditions:
        df_condition = df_blocks[df_blocks["block_condition"].astype(str) == condition].copy()
        if df_condition.empty:
            continue
        condition_views = [v for v in views if not v.selector(df_condition).empty]
        if not condition_views:
            continue

        prepared = build_prepared_signed_rt(df_condition, condition_views, cfg, psychometric_l2=psychometric_l2)
        jnd_indiv = compute_jnd_individuals_by_view(prepared, skip_abl=50)
        group_jnd = compute_group_jnd_by_view(jnd_indiv)
        fig = _plot_prepared_abls(
            prepared=prepared,
            views=condition_views,
            cfg=cfg,
            style=style,
            group_jnd=group_jnd,
            view_colors=view_colors,
            view_styles=view_styles,
            title=f"{condition} blocks - genotype comparison",
        )
        outputs[condition] = {
            "figure": fig,
            "prepared": prepared,
            "jnd_indiv": jnd_indiv,
            "group_jnd": group_jnd,
            "counts": {v.name: len(v.selector(df_condition)) for v in condition_views},
        }
        if show:
            plt.show()
    return outputs


def _collect_block_condition_params(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec],
    block_conditions: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    df_blocks = bundle["df_blocks"]
    cfg = bundle["cfg"]
    psychometric_l2 = float(bundle.get("psychometric_l2", PSYCHOMETRIC_L2))
    rows = []

    for condition in block_conditions:
        df_condition = df_blocks[df_blocks["block_condition"].astype(str) == condition].copy()
        if df_condition.empty:
            continue
        condition_views = [v for v in views if not v.selector(df_condition).empty]
        if not condition_views:
            continue

        prepared = build_prepared_signed_rt(df_condition, condition_views, cfg, psychometric_l2=psychometric_l2)
        for view in condition_views:
            params = prepared.get(view.name, {}).get("psy_params", pd.DataFrame()).copy()
            if params.empty:
                continue
            params["view"] = view.name
            params["block_condition"] = condition
            rows.append(params)

    if not rows:
        return pd.DataFrame(
            columns=[
                "animal", "line", "cohort", "genotype", "dataset_key", "ABL",
                "slope_a", "bias_b", "lower_c", "upper_d", "view", "block_condition",
            ]
        )
    return pd.concat(rows, ignore_index=True, sort=False)


def plot_block_condition_psy_params(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    block_conditions: list[str] | tuple[str, ...] = ("rightward", "leftward", "unbiased"),
    show: bool = True,
) -> dict[str, Any]:
    """Plot psychometric parameters by genotype/view and block condition.

    Returns one 2x2 figure per ABL. Within each parameter panel, each marker is
    the mean across animals for one genotype/view and block condition; error
    bars are SEM across animals.
    """
    style = bundle["style"]
    views = views or bundle["views"]
    view_names = [v.name for v in views]
    params = _collect_block_condition_params(
        bundle,
        views=views,
        block_conditions=block_conditions,
    )
    if params.empty:
        raise ValueError("No psychometric parameter rows available for the selected block conditions/views.")

    params = params.copy()
    params["ABL"] = pd.to_numeric(params["ABL"], errors="coerce")
    params = params.dropna(subset=["ABL"]).copy()
    params["ABL"] = params["ABL"].astype(int)

    condition_names = [
        condition for condition in block_conditions
        if condition in set(params["block_condition"].dropna().astype(str))
    ]
    offsets = (
        np.linspace(-0.22, 0.22, len(condition_names))
        if len(condition_names) > 1
        else np.array([0.0])
    )
    x_positions = {name: i for i, name in enumerate(view_names)}
    outputs: dict[str, Any] = {}

    def _draw_param_figure(fig_params: pd.DataFrame, title_text: str, output_key: str) -> None:
        fs = style.legend_fs
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(max(9.5, 2.6 * len(view_names)), 8.8),
            squeeze=False,
            gridspec_kw={"hspace": 0.46, "wspace": 0.52},
        )
        axes = axes.ravel()

        for ax, (param_col, title) in zip(axes, PSY_PARAM_SPECS):
            for cond_i, condition in enumerate(condition_names):
                cond_df = fig_params[fig_params["block_condition"].astype(str) == condition].copy()
                if cond_df.empty:
                    continue
                color = BLOCK_COLORS.get(condition, f"C{cond_i}")
                marker = BLOCK_STYLES.get(condition, {}).get("marker", "o")
                offset = float(offsets[cond_i])

                for view_name in view_names:
                    sub = cond_df[cond_df["view"].astype(str) == view_name].copy()
                    values = pd.to_numeric(sub[param_col], errors="coerce").dropna()
                    if values.empty:
                        continue
                    x = x_positions[view_name] + offset
                    mean = float(values.mean())
                    err = sem(values.to_numpy())
                    ax.errorbar(
                        x,
                        mean,
                        yerr=err,
                        fmt=marker,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=8.5,
                        elinewidth=1.5,
                        capsize=3,
                        linestyle="None",
                        zorder=4,
                    )

            ax.set_title(title, fontsize=fs, pad=style.title_pad)
            ax.set_ylabel("Parameter value", fontsize=fs, color="black")
            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels(view_names, rotation=25, ha="right", fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            ax.grid(True, axis="x", linestyle=":", alpha=0.25)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

        handles = [
            Line2D(
                [],
                [],
                color=BLOCK_COLORS.get(condition, f"C{i}"),
                marker=BLOCK_STYLES.get(condition, {}).get("marker", "o"),
                linestyle="None",
                markerfacecolor=BLOCK_COLORS.get(condition, f"C{i}"),
                markeredgecolor=BLOCK_COLORS.get(condition, f"C{i}"),
                label=condition,
            )
            for i, condition in enumerate(condition_names)
        ]
        fig.legend(
            handles=handles,
            labels=condition_names,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, max(1, len(condition_names))),
            fontsize=fs,
            frameon=False,
        )
        fig.suptitle(title_text, fontsize=fs, y=0.985)
        fig.tight_layout(rect=[0, 0.12, 1, 0.93])
        outputs[output_key] = {
            "figure": fig,
            "params": fig_params,
        }
        if show:
            plt.show()

    for abl in sorted(params["ABL"].dropna().astype(int).unique()):
        abl_params = params[params["ABL"] == abl].copy()
        _draw_param_figure(
            abl_params,
            title_text=f"Psychometric parameters - ABL {abl}",
            output_key=f"ABL_{abl}",
        )

    collapse_cols = [
        c
        for c in ["animal", "line", "cohort", "genotype", "dataset_key", "view", "block_condition"]
        if c in params.columns
    ]
    if collapse_cols:
        collapsed = (
            params.groupby(collapse_cols, dropna=False)[[col for col, _ in PSY_PARAM_SPECS]]
            .mean()
            .reset_index()
        )
        collapsed["ABL"] = "all"
        _draw_param_figure(
            collapsed,
            title_text="Psychometric parameters - all ABLs",
            output_key="ABL_all",
        )

    return outputs


def _collect_block_condition_summary_metrics(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec],
    block_conditions: list[str] | tuple[str, ...],
    include_abls: tuple[int, ...] = MEAN_ABL_TARGETS,
) -> pd.DataFrame:
    df_blocks = bundle["df_blocks"]
    cfg = bundle["cfg"]
    psychometric_l2 = float(bundle.get("psychometric_l2", PSYCHOMETRIC_L2))
    include_abls = tuple(int(a) for a in include_abls)
    metric_rows: list[dict[str, Any]] = []

    for condition in block_conditions:
        df_condition = df_blocks[df_blocks["block_condition"].astype(str) == condition].copy()
        if df_condition.empty:
            continue
        condition_views = [v for v in views if not v.selector(df_condition).empty]
        if not condition_views:
            continue

        prepared = build_prepared_signed_rt(df_condition, condition_views, cfg, psychometric_l2=psychometric_l2)
        for view in condition_views:
            view_tables = prepared.get(view.name, {})
            view_df = view_tables.get("df_view", pd.DataFrame()).copy()
            if not view_df.empty and {"animal", "ABL", "success"}.issubset(view_df.columns):
                perf = view_df.copy()
                perf["ABL"] = pd.to_numeric(perf["ABL"], errors="coerce")
                perf["success"] = pd.to_numeric(perf["success"], errors="coerce")
                perf = perf[
                    perf["ABL"].isin(include_abls)
                    & perf["success"].notna()
                    & perf["success"].ne(0)
                ].copy()
                if not perf.empty:
                    perf["value"] = perf["success"].eq(1).astype(float)
                    perf_rows = (
                        perf.groupby(["animal", "ABL"], dropna=False)["value"]
                        .mean()
                        .reset_index()
                    )
                    for _, row in perf_rows.iterrows():
                        metric_rows.append(
                            {
                                "animal": row["animal"],
                                "view": view.name,
                                "block_condition": condition,
                                "ABL": int(row["ABL"]),
                                "metric": "prop_correct",
                                "value": float(row["value"]),
                            }
                        )

            params = view_tables.get("psy_params", pd.DataFrame()).copy()
            if not params.empty and {"animal", "ABL", "bias_b"}.issubset(params.columns):
                params["ABL"] = pd.to_numeric(params["ABL"], errors="coerce")
                params["bias_b"] = pd.to_numeric(params["bias_b"], errors="coerce")
                params = params[params["ABL"].isin(include_abls)].dropna(subset=["bias_b"]).copy()
                for _, row in params.iterrows():
                    metric_rows.append(
                        {
                            "animal": row["animal"],
                            "view": view.name,
                            "block_condition": condition,
                            "ABL": int(row["ABL"]),
                            "metric": "bias_b",
                            "value": float(row["bias_b"]),
                        }
                    )

            jnd = view_tables.get("jnd_indiv", pd.DataFrame()).copy()
            if not jnd.empty and {"subject", "ABL", "JND"}.issubset(jnd.columns):
                jnd["ABL"] = pd.to_numeric(jnd["ABL"], errors="coerce")
                jnd["JND"] = pd.to_numeric(jnd["JND"], errors="coerce")
                jnd = jnd[jnd["ABL"].isin(include_abls)].dropna(subset=["JND"]).copy()
                for _, row in jnd.iterrows():
                    metric_rows.append(
                        {
                            "animal": row["subject"],
                            "view": view.name,
                            "block_condition": condition,
                            "ABL": int(row["ABL"]),
                            "metric": "JND",
                            "value": float(row["JND"]),
                        }
                    )

    if not metric_rows:
        return pd.DataFrame(columns=["animal", "view", "block_condition", "metric", "value", "n_abls"])

    metrics = pd.DataFrame(metric_rows)
    collapsed = (
        metrics.groupby(["animal", "view", "block_condition", "metric"], dropna=False)["value"]
        .agg(value="mean", n_abls="count")
        .reset_index()
    )
    return collapsed


def plot_block_condition_summary_metrics(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    block_conditions: list[str] | tuple[str, ...] = ("rightward", "leftward"),
    show: bool = True,
) -> dict[str, Any]:
    """Plot ABL-averaged bias, proportion correct, and JND by genotype/view."""
    style = bundle["style"]
    views = views or bundle["views"]
    view_names = [v.name for v in views]
    condition_names = [
        condition
        for condition in block_conditions
        if condition in {"rightward", "leftward"}
    ]
    if not condition_names:
        condition_names = ["rightward", "leftward"]

    metrics = _collect_block_condition_summary_metrics(
        bundle,
        views=views,
        block_conditions=condition_names,
    )
    if metrics.empty:
        raise ValueError("No ABL-averaged summary metrics available for rightward/leftward blocks and selected views.")

    specs = [
        ("bias_b", "Psychometric bias", "Bias (b)"),
        ("prop_correct", "Proportion correct", "Proportion correct"),
        ("JND", "JND", "JND"),
    ]
    condition_names = [
        condition for condition in condition_names
        if condition in set(metrics["block_condition"].dropna().astype(str))
    ]
    x_positions = {name: i for i, name in enumerate(view_names)}
    offsets = np.linspace(-0.14, 0.14, len(condition_names)) if len(condition_names) > 1 else np.array([0.0])
    fs = style.legend_fs
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.8), squeeze=False)
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, specs):
        metric_df = metrics[metrics["metric"].astype(str) == metric].copy()
        for cond_i, condition in enumerate(condition_names):
            cond_df = metric_df[metric_df["block_condition"].astype(str) == condition].copy()
            if cond_df.empty:
                continue
            color = BLOCK_COLORS.get(condition, f"C{cond_i}")
            marker = BLOCK_STYLES.get(condition, {}).get("marker", "o")
            offset = float(offsets[cond_i])
            for view_name in view_names:
                sub = cond_df[cond_df["view"].astype(str) == view_name].copy()
                values = pd.to_numeric(sub["value"], errors="coerce").dropna()
                if values.empty:
                    continue
                x = x_positions[view_name] + offset
                jitter = rng.uniform(-0.035, 0.035, size=len(values))
                ax.scatter(
                    np.full(len(values), x, dtype=float) + jitter,
                    values.to_numpy(dtype=float),
                    s=28,
                    color=color,
                    marker=marker,
                    alpha=0.55,
                    edgecolors="none",
                    zorder=3,
                )
                ax.errorbar(
                    x,
                    float(values.mean()),
                    yerr=sem(values.to_numpy(dtype=float)),
                    fmt=marker,
                    color="black",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=8.0,
                    elinewidth=1.5,
                    capsize=3,
                    linestyle="None",
                    zorder=5,
                )

        if metric == "bias_b":
            ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
        elif metric == "prop_correct":
            ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0, zorder=0)
            ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=fs, pad=style.title_pad)
        ax.set_ylabel(ylabel, fontsize=fs, color="black")
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(view_names, rotation=25, ha="right", fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.grid(True, axis="x", linestyle=":", alpha=0.25)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            color=BLOCK_COLORS.get(condition, f"C{i}"),
            marker=BLOCK_STYLES.get(condition, {}).get("marker", "o"),
            linestyle="None",
            markerfacecolor=BLOCK_COLORS.get(condition, f"C{i}"),
            markeredgecolor=BLOCK_COLORS.get(condition, f"C{i}"),
            label=condition,
        )
        for i, condition in enumerate(condition_names)
    ]
    fig.legend(
        handles=handles,
        labels=condition_names,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, max(1, len(condition_names))),
        fontsize=fs,
        frameon=False,
    )
    fig.suptitle("Biased blocks summary - mean of ABLs 20, 40, 60", fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    if show:
        plt.show()
    return {
        "summary": {
            "figure": fig,
            "metrics": metrics,
            "abls": MEAN_ABL_TARGETS,
        }
    }


def _std(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(arr.std(ddof=1))


def _df_for_views(df: pd.DataFrame, views: list[ViewSpec]) -> pd.DataFrame:
    parts = [view.selector(df).copy() for view in views]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return df.iloc[0:0].copy()
    out = pd.concat(parts, axis=0, sort=False)
    return out.loc[~out.index.duplicated()].copy()


def _compute_session_bias(df_blocks: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "session", "block_condition"]
        if c in df_blocks.columns
    ]
    rows = []
    for key, sub in df_blocks.groupby(key_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(key_cols, key_values))
        row["bias"] = DataHelpers.compute_bias(sub)
        row["n_trials"] = len(sub)
        row["n_valid"] = int((pd.to_numeric(sub["success"], errors="coerce") != 0).sum()) if "success" in sub else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_bias_by_animal(session_bias: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "block_condition"] if c in session_bias.columns]
    if session_bias.empty:
        return pd.DataFrame(columns=[*meta_cols, "mean", "std", "sem", "n_sessions"])
    return (
        session_bias.groupby(meta_cols, dropna=False)["bias"]
        .agg(mean="mean", std=_std, sem=sem, n_sessions="count")
        .reset_index()
    )


def _plot_block_bias_panels(
    *,
    panel_df: pd.DataFrame,
    mean_df: pd.DataFrame,
    panel_col: str,
    panel_order: list[str],
    title: str,
    style: PlotStyle,
) -> plt.Figure:
    condition_order = [c for c in BLOCK_ORDER if c in set(panel_df["block_condition"].dropna().astype(str))]
    if not condition_order:
        condition_order = sorted(panel_df["block_condition"].dropna().astype(str).unique())
    x_positions = {condition: i for i, condition in enumerate(condition_order)}

    panel_names = panel_order + ["Mean"]
    n_cols = min(4, max(1, len(panel_names)))
    n_rows = int(np.ceil(len(panel_names) / n_cols))
    fs = style.legend_fs
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.ravel()

    for ax_i, panel_name in enumerate(panel_names):
        ax = flat_axes[ax_i]
        if panel_name == "Mean":
            sub = mean_df.copy()
            err_col = "sem"
        else:
            sub = panel_df[panel_df[panel_col].astype(str) == str(panel_name)].copy()
            err_col = "std"

        for condition in condition_order:
            rows = sub[sub["block_condition"].astype(str) == condition]
            if rows.empty:
                continue
            row = rows.iloc[0]
            x = x_positions[condition]
            color = BLOCK_COLORS.get(condition, "gray")
            marker = BLOCK_STYLES.get(condition, {}).get("marker", "o")
            ax.errorbar(
                x,
                row["mean"],
                yerr=row.get(err_col, np.nan),
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7.5,
                elinewidth=1.3,
                capsize=3,
                linestyle="None",
                zorder=4,
            )

        ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_title(str(panel_name), fontsize=fs, pad=8)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(condition_order, rotation=25, ha="right", fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_ylabel("Bias", fontsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

    for ax in flat_axes[len(panel_names):]:
        ax.axis("off")

    handles = [
        Line2D(
            [],
            [],
            color=BLOCK_COLORS.get(condition, f"C{i}"),
            marker=BLOCK_STYLES.get(condition, {}).get("marker", "o"),
            linestyle="None",
            markerfacecolor=BLOCK_COLORS.get(condition, f"C{i}"),
            markeredgecolor=BLOCK_COLORS.get(condition, f"C{i}"),
            label=condition,
        )
        for i, condition in enumerate(condition_order)
    ]
    fig.legend(
        handles=handles,
        labels=condition_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, max(1, len(condition_order))),
        fontsize=fs,
        frameon=False,
    )
    fig.suptitle(title, fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    return fig


def plot_block_bias_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot block-condition bias by animal and by genotype/view."""
    df_blocks = bundle["df_blocks"]
    style = bundle["style"]
    views = views or bundle["views"]
    df_scope = _df_for_views(df_blocks, views)
    if df_scope.empty:
        raise ValueError("No rows available for the selected views.")

    session_bias = _compute_session_bias(df_scope)
    animal_bias = _summarize_bias_by_animal(session_bias)

    animal_order = sorted(animal_bias["animal"].dropna().astype(str).unique())
    mean_by_animal = (
        animal_bias.groupby("block_condition", dropna=False)["mean"]
        .agg(mean="mean", sem=sem, std=_std, n_animals="count")
        .reset_index()
    )
    fig_animals = _plot_block_bias_panels(
        panel_df=animal_bias,
        mean_df=mean_by_animal,
        panel_col="animal",
        panel_order=animal_order,
        title="Block bias by animal",
        style=style,
    )
    if show:
        plt.show()

    view_rows = []
    for view in views:
        sub = view.selector(df_scope)
        if sub.empty:
            continue
        animals = set(sub["animal"].dropna().astype(str))
        tmp = animal_bias[animal_bias["animal"].astype(str).isin(animals)].copy()
        tmp["view"] = view.name
        view_rows.append(tmp)
    genotype_panel = pd.concat(view_rows, ignore_index=True, sort=False) if view_rows else pd.DataFrame()
    if genotype_panel.empty:
        raise ValueError("No animal-bias rows available for the selected genotype/view panels.")

    genotype_summary = (
        genotype_panel.groupby(["view", "block_condition"], dropna=False)["mean"]
        .agg(mean="mean", sem=sem, std=_std, n_animals="count")
        .reset_index()
    )
    mean_by_view = (
        genotype_panel.groupby("block_condition", dropna=False)["mean"]
        .agg(mean="mean", sem=sem, std=_std, n_animals="count")
        .reset_index()
    )
    view_order = [v.name for v in views if v.name in set(genotype_summary["view"].astype(str))]
    fig_genotypes = _plot_block_bias_panels(
        panel_df=genotype_summary,
        mean_df=mean_by_view,
        panel_col="view",
        panel_order=view_order,
        title="Block bias by genotype/view",
        style=style,
    )
    if show:
        plt.show()

    return {
        "animals": {
            "figure": fig_animals,
            "session_bias": session_bias,
            "animal_bias": animal_bias,
            "mean_bias": mean_by_animal,
        },
        "genotypes": {
            "figure": fig_genotypes,
            "genotype_bias": genotype_summary,
            "mean_bias": mean_by_view,
        },
    }


def _choice_right_series(df: pd.DataFrame) -> pd.Series:
    resp = pd.to_numeric(df["response_poke"], errors="coerce")
    if resp.dropna().isin([2, 3]).any():
        return pd.Series(np.where(resp == 3, 1.0, np.where(resp == 2, 0.0, np.nan)), index=df.index)
    if resp.dropna().isin([-1, 1]).any():
        return pd.Series(np.where(resp == 1, 1.0, np.where(resp == -1, 0.0, np.nan)), index=df.index)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _signed_ild_group_series(ild: pd.Series) -> pd.Series:
    ild = pd.to_numeric(ild, errors="coerce")
    return pd.Series(
        np.select(
            [
                ild.isin([-2, -1]),
                ild.isin([1, 2]),
                ild.isin([-16, -8]),
                ild.isin([8, 16]),
            ],
            ["hard left", "hard right", "easy left", "easy right"],
            default=pd.NA,
        ),
        index=ild.index,
    )


def _transition_window_rows(
    df_blocks: pd.DataFrame,
    *,
    window: int = 20,
    pre_window: int | None = None,
    post_window: int | None = None,
    from_condition: str = "leftward",
    to_condition: str = "rightward",
    analysis_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pre_window = int(window if pre_window is None else pre_window)
    post_window = int(window if post_window is None else post_window)
    key_cols = [c for c in ["dataset_key", "animal", "session"] if c in df_blocks.columns]
    if not key_cols:
        raise KeyError("Need animal/session columns to find block transitions.")

    rows = []
    for _, session_df in df_blocks.groupby(key_cols, dropna=False, sort=False):
        if "block" not in session_df.columns:
            continue
        session_df = session_df.copy()
        session_df["_block_num"] = pd.to_numeric(session_df["block"], errors="coerce")
        block_order = sorted(session_df["_block_num"].dropna().unique())
        if len(block_order) < 2:
            continue

        block_condition = (
            session_df.groupby("_block_num")["block_condition"]
            .agg(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().size else pd.NA)
            .to_dict()
        )
        for prev_block, next_block in zip(block_order[:-1], block_order[1:]):
            if block_condition.get(prev_block) != from_condition or block_condition.get(next_block) != to_condition:
                continue

            prev_rows = session_df[session_df["_block_num"] == prev_block].sort_values("trial").copy()
            next_rows = session_df[session_df["_block_num"] == next_block].sort_values("trial").copy()
            prev_success = pd.to_numeric(prev_rows["success"], errors="coerce")
            next_success = pd.to_numeric(next_rows["success"], errors="coerce")
            prev_rows = prev_rows[prev_success.ne(0)].tail(pre_window).copy()
            next_rows = next_rows[next_success.ne(0)].head(post_window).copy()
            if prev_rows.empty or next_rows.empty:
                continue

            prev_rows["relative_trial"] = np.arange(-len(prev_rows), 0)
            next_rows["relative_trial"] = np.arange(1, len(next_rows) + 1)
            transition_id = f"{prev_rows['animal'].iloc[0]}:{prev_rows['session'].iloc[0]}:{int(prev_block)}-{int(next_block)}"
            prev_rows["transition_id"] = transition_id
            next_rows["transition_id"] = transition_id
            rows.append(prev_rows)
            rows.append(next_rows)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    success = pd.to_numeric(out["success"], errors="coerce")
    out = out[success.ne(0)].copy()
    out["prob_correct"] = success.loc[out.index].eq(1).astype(float)
    out["choice_right"] = _choice_right_series(out)
    out["signed_ild_group"] = _signed_ild_group_series(out["ILD"])
    out = out[out["signed_ild_group"].notna() & out["choice_right"].notna()].copy()
    if analysis_df is not None:
        if "_biased_blocks_row_id" not in out or "_biased_blocks_row_id" not in analysis_df:
            raise KeyError("Transition timing requires _biased_blocks_row_id in both timing and analysis data.")
        analysis_rows = set(analysis_df["_biased_blocks_row_id"].dropna().astype(int))
        out = out[out["_biased_blocks_row_id"].astype(int).isin(analysis_rows)].copy()
    out["ABL"] = pd.to_numeric(out["ABL"], errors="coerce")
    return out


def _aligned_biased_transition_window_rows(
    df_blocks: pd.DataFrame,
    *,
    window: int = 20,
    analysis_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect windows around any biased->biased transition and align to the new favored side."""
    key_cols = [c for c in ["dataset_key", "animal", "session"] if c in df_blocks.columns]
    if not key_cols:
        raise KeyError("Need animal/session columns to find block transitions.")

    rows = []
    for _, session_df in df_blocks.groupby(key_cols, dropna=False, sort=False):
        if "block" not in session_df.columns:
            continue
        session_df = session_df.copy()
        session_df["_block_num"] = pd.to_numeric(session_df["block"], errors="coerce")
        block_order = sorted(session_df["_block_num"].dropna().unique())
        if len(block_order) < 2:
            continue

        block_condition = (
            session_df.groupby("_block_num")["block_condition"]
            .agg(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().size else pd.NA)
            .to_dict()
        )
        for prev_block, next_block in zip(block_order[:-1], block_order[1:]):
            prev_condition = block_condition.get(prev_block)
            next_condition = block_condition.get(next_block)
            if prev_condition not in {"leftward", "rightward"}:
                continue
            if next_condition not in {"leftward", "rightward"}:
                continue
            if prev_condition == next_condition:
                continue

            prev_rows = session_df[session_df["_block_num"] == prev_block].sort_values("trial").copy()
            next_rows = session_df[session_df["_block_num"] == next_block].sort_values("trial").copy()
            prev_rows = prev_rows[pd.to_numeric(prev_rows["success"], errors="coerce").ne(0)].tail(window).copy()
            next_rows = next_rows[pd.to_numeric(next_rows["success"], errors="coerce").ne(0)].head(window).copy()
            if prev_rows.empty or next_rows.empty:
                continue

            prev_rows["relative_trial"] = np.arange(-len(prev_rows), 0)
            next_rows["relative_trial"] = np.arange(1, len(next_rows) + 1)
            transition_id = f"{prev_rows['animal'].iloc[0]}:{prev_rows['session'].iloc[0]}:{int(prev_block)}-{int(next_block)}"
            transition_direction = f"{prev_condition}_to_{next_condition}"
            prev_rows["transition_id"] = transition_id
            next_rows["transition_id"] = transition_id
            prev_rows["transition_direction"] = transition_direction
            next_rows["transition_direction"] = transition_direction
            prev_rows["new_favored_side"] = next_condition
            next_rows["new_favored_side"] = next_condition
            rows.append(prev_rows)
            rows.append(next_rows)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    success = pd.to_numeric(out["success"], errors="coerce")
    out = out[success.ne(0)].copy()
    out["choice_right"] = _choice_right_series(out)
    out["ild_raw"] = pd.to_numeric(out["ILD"], errors="coerce")

    side_sign = np.where(out["new_favored_side"].astype(str) == "rightward", 1.0, -1.0)
    out["ild_aligned"] = out["ild_raw"] * side_sign
    out["choice_toward_new_side"] = np.where(side_sign > 0, out["choice_right"], 1.0 - out["choice_right"])

    ild_aligned = pd.to_numeric(out["ild_aligned"], errors="coerce")
    out["aligned_ild_group"] = np.select(
        [
            ild_aligned.isin([-2, -1]),
            ild_aligned.isin([1, 2]),
            ild_aligned.isin([-16, -8]),
            ild_aligned.isin([8, 16]),
        ],
        ["hard away", "hard toward", "easy away", "easy toward"],
        default=pd.NA,
    )
    out = out[out["aligned_ild_group"].notna() & pd.notna(out["choice_toward_new_side"])].copy()
    if analysis_df is not None:
        if "_biased_blocks_row_id" not in out or "_biased_blocks_row_id" not in analysis_df:
            raise KeyError("Transition timing requires _biased_blocks_row_id in both timing and analysis data.")
        analysis_rows = set(analysis_df["_biased_blocks_row_id"].dropna().astype(int))
        out = out[out["_biased_blocks_row_id"].astype(int).isin(analysis_rows)].copy()
    out["ABL"] = pd.to_numeric(out["ABL"], errors="coerce")
    return out


def _animal_transition_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "view",
                "ABL",
                "signed_ild_group",
                "relative_trial",
                "prob_correct",
            ]
        )
    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "signed_ild_group", "relative_trial"]
        if c in window_df.columns
    ]
    return (
        window_df.groupby(group_cols, dropna=False)["prob_correct"]
        .mean()
        .rename("prob_correct")
        .reset_index()
    )


def _animal_transition_trace_all_ilds(window_df: pd.DataFrame) -> pd.DataFrame:
    base = _animal_transition_trace(window_df)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                "prob_correct",
            ]
        )
    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial"]
        if c in base.columns
    ]
    return (
        base.groupby(group_cols, dropna=False)["prob_correct"]
        .mean()
        .rename("prob_correct")
        .reset_index()
    )


def _animal_aligned_transition_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(columns=["animal", "genotype", "view", "ABL", "aligned_ild_group", "relative_trial", "frac_toward_new_side"])
    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "aligned_ild_group", "relative_trial"]
        if c in window_df.columns
    ]
    return (
        window_df.groupby(group_cols, dropna=False)["choice_toward_new_side"]
        .mean()
        .rename("frac_toward_new_side")
        .reset_index()
    )


def _collapsed_biased_transition_window_rows(
    df_blocks: pd.DataFrame,
    *,
    window: int = 20,
    analysis_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect any biased->biased transition, align choices to the new side, and collapse ILDs by difficulty."""
    out = _aligned_biased_transition_window_rows(
        df_blocks,
        window=window,
        analysis_df=analysis_df,
    )
    if out.empty:
        return out
    ild_aligned = pd.to_numeric(out["ild_aligned"], errors="coerce")
    out["difficulty_group"] = np.select(
        [
            ild_aligned.abs().isin([1, 2]),
            ild_aligned.abs().isin([8, 16]),
        ],
        ["hard", "easy"],
        default=pd.NA,
    )
    out = out[out["difficulty_group"].notna()].copy()
    return out


def _animal_collapsed_transition_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(columns=["animal", "genotype", "view", "ABL", "difficulty_group", "relative_trial", "frac_toward_new_side"])
    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "difficulty_group", "relative_trial"]
        if c in window_df.columns
    ]
    return (
        window_df.groupby(group_cols, dropna=False)["choice_toward_new_side"]
        .mean()
        .rename("frac_toward_new_side")
        .reset_index()
    )


def _unbiased_choice_right_baseline(df_blocks: pd.DataFrame) -> pd.DataFrame:
    if df_blocks.empty or "block_condition" not in df_blocks.columns:
        return pd.DataFrame(columns=["animal", "dataset_key", "ABL", "signed_ild_group", "baseline_choice_right"])

    df_u = df_blocks[df_blocks["block_condition"].astype(str) == "unbiased"].copy()
    if df_u.empty:
        return pd.DataFrame(columns=["animal", "dataset_key", "ABL", "signed_ild_group", "baseline_choice_right"])

    success = pd.to_numeric(df_u["success"], errors="coerce")
    df_u = df_u[success.ne(0)].copy()
    if df_u.empty:
        return pd.DataFrame(columns=["animal", "dataset_key", "ABL", "signed_ild_group", "baseline_choice_right"])

    df_u["choice_right"] = _choice_right_series(df_u)
    df_u["signed_ild_group"] = _signed_ild_group_series(df_u["ILD"])
    df_u["ABL"] = pd.to_numeric(df_u["ABL"], errors="coerce")
    df_u = df_u[df_u["choice_right"].notna() & df_u["signed_ild_group"].notna() & df_u["ABL"].notna()].copy()
    if df_u.empty:
        return pd.DataFrame(columns=["animal", "dataset_key", "ABL", "signed_ild_group", "baseline_choice_right"])

    group_cols = [c for c in ["animal", "dataset_key", "ABL", "signed_ild_group"] if c in df_u.columns]
    return (
        df_u.groupby(group_cols, dropna=False)["choice_right"]
        .mean()
        .rename("baseline_choice_right")
        .reset_index()
    )


def _animal_baseline_subtracted_transition_trace(
    window_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "view",
                "ABL",
                "signed_ild_group",
                "relative_trial",
                "delta_choice_right",
            ]
        )

    join_cols = [c for c in ["animal", "dataset_key", "ABL", "signed_ild_group"] if c in window_df.columns and c in baseline_df.columns]
    if not join_cols:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "view",
                "ABL",
                "signed_ild_group",
                "relative_trial",
                "delta_choice_right",
            ]
        )

    merged = window_df.merge(baseline_df, on=join_cols, how="left")
    merged["delta_choice_right"] = merged["choice_right"] - pd.to_numeric(merged["baseline_choice_right"], errors="coerce")
    merged = merged[merged["delta_choice_right"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "view",
                "ABL",
                "signed_ild_group",
                "relative_trial",
                "delta_choice_right",
            ]
        )

    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "signed_ild_group", "relative_trial"]
        if c in merged.columns
    ]
    return (
        merged.groupby(group_cols, dropna=False)["delta_choice_right"]
        .mean()
        .rename("delta_choice_right")
        .reset_index()
    )


def _animal_baseline_subtracted_transition_trace_all_ilds(
    window_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    base = _animal_baseline_subtracted_transition_trace(window_df, baseline_df)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                "delta_choice_right",
            ]
        )
    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial"]
        if c in base.columns
    ]
    return (
        base.groupby(group_cols, dropna=False)["delta_choice_right"]
        .mean()
        .rename("delta_choice_right")
        .reset_index()
    )


def _animal_positive_minus_negative_accuracy_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                "accuracy_pos_minus_neg",
            ]
        )

    df = window_df.copy()
    df["ild_value"] = pd.to_numeric(df["ILD"], errors="coerce")
    df["ild_sign"] = np.select(
        [df["ild_value"].gt(0), df["ild_value"].lt(0)],
        ["positive", "negative"],
        default=pd.NA,
    )
    df = df[df["ild_sign"].notna() & df["prob_correct"].notna()].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                "accuracy_pos_minus_neg",
            ]
        )

    group_cols = [
        c
        for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial", "ild_sign"]
        if c in df.columns
    ]
    accuracy = (
        df.groupby(group_cols, dropna=False)["prob_correct"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
    index_cols = [c for c in group_cols if c != "ild_sign"]
    wide = accuracy.pivot_table(index=index_cols, columns="ild_sign", values="accuracy", aggfunc="mean").reset_index()
    if "positive" not in wide.columns or "negative" not in wide.columns:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                "accuracy_pos_minus_neg",
            ]
        )

    wide["accuracy_pos_minus_neg"] = wide["positive"] - wide["negative"]
    wide = wide[wide["accuracy_pos_minus_neg"].notna()].copy()
    keep_cols = index_cols + ["accuracy_pos_minus_neg"]
    return wide[keep_cols].reset_index(drop=True)


def _animal_transition_accuracy_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                "prob_correct",
            ]
        )

    group_cols = [
        c for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial"]
        if c in window_df.columns
    ]
    return (
        window_df.groupby(group_cols, dropna=False)["prob_correct"]
        .mean()
        .rename("prob_correct")
        .reset_index()
    )


def _animal_mean_rt_transition_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    """Pool all ILDs into one correct-trial mean RT for each animal and trial index."""
    columns = [
        "animal",
        "genotype",
        "line",
        "cohort",
        "dataset_key",
        "ABL",
        "relative_trial",
        "mean_rt",
    ]
    if window_df.empty or "timed_rt" not in window_df:
        return pd.DataFrame(columns=columns)

    df = window_df.copy()
    df["timed_rt"] = pd.to_numeric(df["timed_rt"], errors="coerce")
    success = pd.to_numeric(df["success"], errors="coerce")
    df = df[success.eq(1) & df["timed_rt"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    group_cols = [
        column
        for column in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial"]
        if column in df.columns
    ]
    return (
        df.groupby(group_cols, dropna=False)["timed_rt"]
        .mean()
        .rename("mean_rt")
        .reset_index()
    )


def _animal_positive_minus_negative_rt_transition_trace(window_df: pd.DataFrame) -> pd.DataFrame:
    """Compute positive-ILD minus negative-ILD mean RT for each animal and trial index."""
    columns = [
        "animal",
        "genotype",
        "line",
        "cohort",
        "dataset_key",
        "ABL",
        "relative_trial",
        "rt_pos_minus_neg",
    ]
    if window_df.empty or "timed_rt" not in window_df:
        return pd.DataFrame(columns=columns)

    df = window_df.copy()
    df["timed_rt"] = pd.to_numeric(df["timed_rt"], errors="coerce")
    ild = pd.to_numeric(df["ILD"], errors="coerce")
    df["ild_sign"] = np.select([ild.gt(0), ild.lt(0)], ["positive", "negative"], default=pd.NA)
    success = pd.to_numeric(df["success"], errors="coerce")
    df = df[success.eq(1) & df["timed_rt"].notna() & df["ild_sign"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    group_cols = [
        column
        for column in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial", "ild_sign"]
        if column in df.columns
    ]
    mean_rt = (
        df.groupby(group_cols, dropna=False)["timed_rt"]
        .mean()
        .rename("mean_rt")
        .reset_index()
    )
    index_cols = [column for column in group_cols if column != "ild_sign"]
    wide = mean_rt.pivot_table(index=index_cols, columns="ild_sign", values="mean_rt", aggfunc="mean").reset_index()
    if "positive" not in wide.columns or "negative" not in wide.columns:
        return pd.DataFrame(columns=columns)
    wide["rt_pos_minus_neg"] = wide["positive"] - wide["negative"]
    return wide[index_cols + ["rt_pos_minus_neg"]].dropna(subset=["rt_pos_minus_neg"]).reset_index(drop=True)


def _format_stim_duration_title(value: Any) -> str:
    if value is None:
        return "stim durations: all"
    if isinstance(value, str):
        return f"stim durations: {value}"
    if isinstance(value, Iterable):
        values = []
        for item in value:
            try:
                item_float = float(item)
            except (TypeError, ValueError):
                values.append(str(item))
                continue
            values.append(str(int(item_float)) if item_float.is_integer() else f"{item_float:g}")
        return "stim durations: " + ", ".join(values)
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return f"stim durations: {value}"
    label = str(int(value_float)) if value_float.is_integer() else f"{value_float:g}"
    return f"stim duration: {label}"


def _stim_duration_title_from_bundle(bundle: dict[str, Any]) -> str:
    if "short_duration_value" in bundle:
        return _format_stim_duration_title(bundle.get("short_duration_value"))

    df_blocks = bundle.get("df_blocks", pd.DataFrame())
    if isinstance(df_blocks, pd.DataFrame) and "short_duration" in df_blocks.columns:
        values = pd.to_numeric(df_blocks["short_duration"], errors="coerce").dropna().unique()
        values = sorted(values)
        if len(values):
            return _format_stim_duration_title(values)

    return "stim durations: unknown"


def _half_difference_direction_trace(
    left_to_right: pd.DataFrame,
    right_to_left: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    if left_to_right.empty or right_to_left.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                out_col,
            ]
        )

    join_cols = [
        c
        for c in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL", "relative_trial"]
        if c in left_to_right.columns and c in right_to_left.columns
    ]
    if not join_cols:
        return pd.DataFrame(
            columns=[
                "animal",
                "genotype",
                "line",
                "cohort",
                "dataset_key",
                "ABL",
                "relative_trial",
                out_col,
            ]
        )

    merged = left_to_right[join_cols + [value_col]].merge(
        right_to_left[join_cols + [value_col]],
        on=join_cols,
        how="inner",
        suffixes=("_left_to_right", "_right_to_left"),
    )
    if merged.empty:
        return pd.DataFrame(columns=join_cols + [out_col])

    merged[out_col] = (
        pd.to_numeric(merged[f"{value_col}_left_to_right"], errors="coerce")
        - pd.to_numeric(merged[f"{value_col}_right_to_left"], errors="coerce")
    ) / 2.0
    merged = merged[merged[out_col].notna()].copy()
    return merged[join_cols + [out_col]].reset_index(drop=True)


def _half_difference_direction_summary(
    left_to_right: pd.DataFrame,
    right_to_left: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Subtract direction-level animal means without requiring paired animals at every trial."""
    summary_cols = ["ABL", "relative_trial"]
    if left_to_right.empty or right_to_left.empty:
        return pd.DataFrame(columns=summary_cols + [out_col, "sem", "n_left_to_right", "n_right_to_left"])

    def summarize(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return (
            df.groupby(summary_cols, dropna=False)[value_col]
            .agg(mean="mean", sem=sem, n="count")
            .rename(columns={"mean": f"mean_{suffix}", "sem": f"sem_{suffix}", "n": f"n_{suffix}"})
            .reset_index()
        )

    merged = summarize(left_to_right, "left_to_right").merge(
        summarize(right_to_left, "right_to_left"),
        on=summary_cols,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=summary_cols + [out_col, "sem", "n_left_to_right", "n_right_to_left"])

    merged[out_col] = (merged["mean_left_to_right"] - merged["mean_right_to_left"]) / 2.0
    merged["sem"] = np.hypot(
        pd.to_numeric(merged["sem_left_to_right"], errors="coerce"),
        pd.to_numeric(merged["sem_right_to_left"], errors="coerce"),
    ) / 2.0
    return merged[summary_cols + [out_col, "sem", "n_left_to_right", "n_right_to_left"]]


def _relative_trial_bin_center(relative_trial: pd.Series, bin_size: int | None) -> pd.Series:
    rel = pd.to_numeric(relative_trial, errors="coerce")
    if not bin_size or int(bin_size) <= 1:
        return rel.astype(float)

    bin_size = int(bin_size)
    centers = pd.Series(np.nan, index=rel.index, dtype=float)
    neg = rel < 0
    pos = rel > 0

    if neg.any():
        neg_dist = (-rel[neg] - 1).astype(int)
        neg_group = neg_dist // bin_size
        neg_start = -((neg_group + 1) * bin_size)
        neg_end = -(neg_group * bin_size + 1)
        centers.loc[neg] = (neg_start + neg_end) / 2.0

    if pos.any():
        pos_dist = (rel[pos] - 1).astype(int)
        pos_group = pos_dist // bin_size
        pos_start = pos_group * bin_size + 1
        pos_end = (pos_group + 1) * bin_size
        centers.loc[pos] = (pos_start + pos_end) / 2.0

    return centers


def _bin_transition_trace(trace_df: pd.DataFrame, value_col: str, bin_size: int | None) -> pd.DataFrame:
    if trace_df.empty or not bin_size or int(bin_size) <= 1:
        return trace_df.copy()

    out = trace_df.copy()
    out["relative_trial"] = _relative_trial_bin_center(out["relative_trial"], int(bin_size))
    group_cols = [c for c in out.columns if c != value_col]
    return (
        out.groupby(group_cols, dropna=False)[value_col]
        .mean()
        .reset_index()
        .sort_values("relative_trial")
    )


def _fit_post_transition_exponential(
    x: np.ndarray,
    y: np.ndarray,
    *,
    include_pre_transition_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit an exponential with an optional pre-transition a-b baseline."""
    valid = np.isfinite(x) & np.isfinite(y)
    if not include_pre_transition_baseline:
        valid &= x > 0
    x_fit = np.asarray(x[valid], dtype=float)
    y_fit = np.asarray(y[valid], dtype=float)
    post_x = x_fit[x_fit > 0]
    if len(x_fit) < 3 or len(np.unique(post_x)) < 2:
        return None

    def decay(trial: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return np.where(trial <= 0, a - b, a - b * np.exp(-c * trial))

    post_y = y_fit[x_fit > 0]
    initial_a = float(np.clip(post_y[-1], 1e-4, 1.5))
    pre_y = y_fit[x_fit <= 0]
    baseline = float(np.nanmean(pre_y)) if len(pre_y) else float(post_y[0])
    initial_b = float(np.clip(initial_a - baseline, 1e-4, 3.0))
    try:
        params, _ = curve_fit(
            decay,
            x_fit,
            y_fit,
            p0=(initial_a, initial_b, 0.2),
            bounds=((0.0, 0.0, 0.0), (1.5, 3.0, 2.0)),
            maxfev=10_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    x_curve = np.linspace(float(x_fit.min()), float(x_fit.max()), 200)
    return x_curve, decay(x_curve, *params), params


def _fit_individual_post_transition_traces(
    trace_df: pd.DataFrame,
    value_col: str,
    *,
    include_pre_transition_baseline: bool = True,
) -> pd.DataFrame:
    """Fit the post-transition exponential separately for each animal and ABL."""
    fit_cols = [
        col
        for col in ["animal", "genotype", "line", "cohort", "dataset_key", "ABL"]
        if col in trace_df.columns
    ]
    if trace_df.empty or not fit_cols:
        return pd.DataFrame(columns=[*fit_cols, "a", "b", "c", "max_post_trial"])

    rows = []
    for key, sub in trace_df.groupby(fit_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        x = pd.to_numeric(sub["relative_trial"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        fitted = _fit_post_transition_exponential(
            x,
            y,
            include_pre_transition_baseline=include_pre_transition_baseline,
        )
        if fitted is None:
            continue
        _, _, params = fitted
        row = dict(zip(fit_cols, key_values))
        row.update(a=float(params[0]), b=float(params[1]), c=float(params[2]))
        row["max_post_trial"] = float(np.nanmax(x[np.isfinite(x) & (x > 0)]))
        rows.append(row)
    return pd.DataFrame(rows, columns=[*fit_cols, "a", "b", "c", "max_post_trial"])


def _mean_individual_fit_curve(
    individual_fits: pd.DataFrame,
    *,
    post_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return mean/SEM fitted curve and mean/SEM a,b,c across animals."""
    if individual_fits.empty:
        return None
    max_post_trial = min(float(post_window), float(individual_fits["max_post_trial"].max()))
    if not np.isfinite(max_post_trial) or max_post_trial < 1:
        return None
    x_curve = np.linspace(1.0, max_post_trial, 200)
    curves = []
    for row in individual_fits.itertuples(index=False):
        active = x_curve <= float(row.max_post_trial)
        curve = np.full_like(x_curve, np.nan)
        curve[active] = float(row.a) - float(row.b) * np.exp(-float(row.c) * x_curve[active])
        curves.append(curve)
    curves = np.asarray(curves, dtype=float)
    n = np.sum(np.isfinite(curves), axis=0)
    mean_curve = np.nanmean(curves, axis=0)
    sem_curve = np.full_like(mean_curve, np.nan)
    enough = n > 1
    if enough.any():
        sem_curve[enough] = np.nanstd(curves[:, enough], axis=0, ddof=1) / np.sqrt(n[enough])
    params = individual_fits[["a", "b", "c"]].to_numpy(dtype=float)
    parameter_mean = np.nanmean(params, axis=0)
    parameter_sem = np.full(3, np.nan)
    if len(params) > 1:
        parameter_sem = np.nanstd(params, axis=0, ddof=1) / np.sqrt(len(params))
    return x_curve, mean_curve, sem_curve, parameter_mean, parameter_sem


def _add_fit_parameter_inset(
    ax: plt.Axes,
    parameter_rows: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    label_header: str,
    fontsize: float,
) -> None:
    """Add a compact fit-parameter scatter plot inside a transition panel."""
    if not parameter_rows:
        return

    inset = ax.inset_axes([0.53, 0.06, 0.42, 0.32])
    labels = [str(label) for label, _, _ in parameter_rows]
    x = np.arange(len(parameter_rows), dtype=float)
    parameter_values = np.asarray([params for _, params, _ in parameter_rows], dtype=float)
    parameter_errors = np.asarray([errors for _, _, errors in parameter_rows], dtype=float)
    parameter_colors = {"a": "#1B9E77", "b": "#D95F02", "c": "#7570B3"}
    for index, parameter_name in enumerate(["a", "b", "c"]):
        inset.errorbar(
            x,
            parameter_values[:, index],
            yerr=parameter_errors[:, index],
            fmt="o",
            color=parameter_colors[parameter_name],
            markersize=3.5,
            capsize=2,
            label=parameter_name,
            zorder=3,
        )
    inset.axhline(0.0, color="0.75", linewidth=0.7)
    inset.set_xticks(x)
    inset.set_xticklabels(labels, fontsize=max(5, fontsize - 12))
    inset.tick_params(axis="y", labelsize=max(5, fontsize - 12), length=2)
    inset.set_title(f"{label_header} fit parameters", fontsize=max(5, fontsize - 11), pad=2)
    inset.legend(fontsize=max(5, fontsize - 13), frameon=False, ncol=3, loc="upper center", handletextpad=0.2, columnspacing=0.5)
    for spine in ["right", "top"]:
        inset.spines[spine].set_visible(False)


def _history_short_label(condition: str) -> str:
    return {
        "unbiased": "U",
        "leftward": "L",
        "rightward": "R",
    }.get(str(condition), str(condition))


def _duration_pair_label(value: float, duration_groups: tuple[tuple[int, ...], ...]) -> str | pd._libs.missing.NAType:
    if not np.isfinite(value):
        return pd.NA
    for group in duration_groups:
        if int(round(float(value))) in {int(v) for v in group}:
            return "+".join(str(int(v)) for v in group)
    return pd.NA


def _duration_pair_order(df: pd.DataFrame) -> list[str]:
    preferred = ["8+16", "32+64", "120+0"]
    present = set(df["duration_pair"].dropna().astype(str)) if "duration_pair" in df else set()
    return [name for name in preferred if name in present]


def _duration_pair_markers() -> dict[str, str]:
    return {"8+16": "o", "32+64": "s", "120+0": "^"}


def _duration_pair_offsets(order: list[str]) -> dict[str, float]:
    if len(order) == 1:
        return {order[0]: 0.0}
    if len(order) == 2:
        return {order[0]: -0.10, order[1]: 0.10}
    default = [-0.18, 0.0, 0.18]
    return {name: default[i] for i, name in enumerate(order)}


def _global_param_ylim(params: pd.DataFrame, param_cols: list[str], pad_frac: float = 0.08) -> dict[str, tuple[float, float]]:
    ylims: dict[str, tuple[float, float]] = {}
    for col in param_cols:
        if col not in params.columns:
            continue
        vals = pd.to_numeric(params[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size >= 6:
            vmin = float(np.nanpercentile(vals, 5))
            vmax = float(np.nanpercentile(vals, 95))
            core = vals[(vals >= vmin) & (vals <= vmax)]
            if core.size >= 2:
                vals = core
                vmin = float(vals.min())
                vmax = float(vals.max())
        else:
            vmin = float(vals.min())
            vmax = float(vals.max())
        if np.isclose(vmin, vmax):
            pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * pad_frac
        else:
            pad = (vmax - vmin) * pad_frac
        ylims[col] = (vmin - pad, vmax + pad)
    return ylims


def _full_param_ylim(params: pd.DataFrame, param_cols: list[str], pad_frac: float = 0.08) -> dict[str, tuple[float, float]]:
    ylims: dict[str, tuple[float, float]] = {}
    for col in param_cols:
        if col not in params.columns:
            continue
        vals = pd.to_numeric(params[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vmin = float(vals.min())
        vmax = float(vals.max())
        if np.isclose(vmin, vmax):
            pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * pad_frac
        else:
            pad = (vmax - vmin) * pad_frac
        ylims[col] = (vmin - pad, vmax + pad)
    return ylims


def _duration_pair_colors() -> dict[str, str]:
    return {"8+16": "#1F77B4", "32+64": "#E69F00", "120+0": "#4D4D4D"}


def _add_slope_metrics(params: pd.DataFrame) -> pd.DataFrame:
    out = params.copy()
    needed = {"slope_a", "lower_c", "upper_d"}
    if not needed.issubset(out.columns):
        return out
    slope_a = pd.to_numeric(out["slope_a"], errors="coerce")
    bias_b = pd.to_numeric(out["bias_b"], errors="coerce") if "bias_b" in out.columns else pd.Series(np.nan, index=out.index)
    lower_c = pd.to_numeric(out["lower_c"], errors="coerce")
    upper_d = pd.to_numeric(out["upper_d"], errors="coerce")
    out["effective_slope"] = 0.5 * (upper_d - lower_c) * slope_a
    exp_term = np.exp(2 * slope_a * bias_b)
    denom = (1.0 + exp_term) ** 2
    out["slope_at_0"] = (upper_d - lower_c) * (2.0 * slope_a) * exp_term / denom
    return out


def _find_suspicious_slope_cases(
    params: pd.DataFrame,
    *,
    condition_col: str,
) -> pd.DataFrame:
    metric_col = "slope_at_0" if "slope_at_0" in params.columns else ("effective_slope" if "effective_slope" in params.columns else "slope_a")
    if params.empty or metric_col not in params.columns or condition_col not in params.columns:
        return pd.DataFrame(columns=[condition_col, "ABL"])

    duration_order = _duration_pair_order(params)
    if len(duration_order) < 2:
        return pd.DataFrame(columns=[condition_col, "ABL"])

    rows = []
    grouped = (
        params.groupby([condition_col, "ABL", "duration_pair"], dropna=False)[metric_col]
        .median()
        .reset_index()
    )
    for (condition_value, abl), sub in grouped.groupby([condition_col, "ABL"], dropna=False, sort=False):
        ordered = []
        for duration_pair in duration_order:
            vals = pd.to_numeric(
                sub.loc[sub["duration_pair"].astype(str) == duration_pair, metric_col],
                errors="coerce",
            ).dropna()
            if vals.empty:
                continue
            ordered.append((duration_pair, float(vals.iloc[0])))
        if len(ordered) < 2:
            continue
        slopes = [value for _, value in ordered]
        if slopes[0] > slopes[-1]:
            rows.append({condition_col: condition_value, "ABL": int(abl), "slope_metric": metric_col})
    return pd.DataFrame(rows)


def _trimmed_mean(values: pd.Series | np.ndarray, trim_frac: float = 0.2) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    if arr.size < 5 or trim_frac <= 0:
        return float(arr.mean())
    k = int(np.floor(arr.size * trim_frac))
    if k <= 0 or (2 * k) >= arr.size:
        return float(arr.mean())
    arr = np.sort(arr)[k: arr.size - k]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def _median_and_iqr_errors(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.median(arr))
    if arr.size == 1:
        return center, 0.0, 0.0
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    lower = max(0.0, center - q25)
    upper = max(0.0, q75 - center)
    return center, lower, upper


def _compute_psy_curve_for_subset(
    df_sub: pd.DataFrame,
    *,
    cfg: GroupComparisonConfig,
    robust_mean: bool = False,
    trim_frac: float = 0.2,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> dict[str, Any] | None:
    if df_sub.empty:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        if float(psychometric_l2) > 0:
            psy_points, psy_group, _, psy_mean, _, _ = _prep_psy_local(
                df_sub,
                do_individual_fits=False,
                aggregation=cfg.psychometric_aggregation,
                skip_jnd_abl=50,
                l2_strength=psychometric_l2,
            )
        else:
            psy_points, psy_group, _, psy_mean, _, _ = prep_psy(
                df_sub,
                do_individual_fits=False,
                aggregation=cfg.psychometric_aggregation,
                skip_jnd_abl=50,
            )
    if psy_group.empty:
        return None
    if not robust_mean:
        return {
            "psy_points": psy_points,
            "psy_group": psy_group,
            "psy_mean_fits": psy_mean,
        }

    if psy_points.empty or "ABL" not in psy_points.columns or "ILD" not in psy_points.columns:
        return None

    value_col = "mean" if "mean" in psy_points.columns else ("PropLeft" if "PropLeft" in psy_points.columns else None)
    if value_col is None:
        return None

    base = psy_points.copy()
    base["ABL"] = pd.to_numeric(base["ABL"], errors="coerce")
    base["ILD"] = pd.to_numeric(base["ILD"], errors="coerce")
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    base = base.dropna(subset=["ABL", "ILD", value_col]).copy()
    if base.empty:
        return None
    base["ABL"] = base["ABL"].astype(int)

    robust_group = (
        base.groupby(["ABL", "ILD"], sort=False)[value_col]
        .agg(
            mean=lambda s: _trimmed_mean(s, trim_frac=trim_frac),
            sem=sem,
            n="count",
        )
        .reset_index()
    )
    robust_group["mean"] = pd.to_numeric(robust_group["mean"], errors="coerce")
    robust_group = robust_group.dropna(subset=["mean"]).copy()
    if robust_group.empty:
        return None

    robust_fits: dict[int, dict[str, Any]] = {}
    for abl, df_abl in robust_group.groupby("ABL", sort=False):
        df_abl = df_abl.sort_values("ILD")
        ilds = df_abl["ILD"].to_numpy(dtype=float)
        mean_vals = df_abl["mean"].to_numpy(dtype=float)
        n_trials = np.full(ilds.shape, 50, dtype=int)
        pars = None
        L = np.nan
        xx = yy = None
        if len(ilds) >= 4 and np.isfinite(mean_vals).all():
            try:
                pars, L, xx, yy = _fit_psychometric_biased(
                    ilds,
                    mean_vals,
                    n_trials=n_trials,
                    l2_strength=psychometric_l2,
                )
            except Exception:
                pars = None
                L = np.nan
                xx = yy = None
        robust_fits[int(abl)] = {
            "ILDs": ilds,
            "PropLeft": mean_vals,
            "n_trials": n_trials,
            "pars": pars,
            "L": L,
            "xx": xx,
            "yy": yy,
        }

    return {
        "psy_points": psy_points,
        "psy_group": robust_group,
        "psy_mean_fits": robust_fits,
    }


def _displayed_curve_slope_at_zero(xx, yy) -> float | None:
    if xx is None or yy is None:
        return None
    x = np.asarray(xx, dtype=float)
    y = np.asarray(yy, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        return None
    dydx = np.gradient(y, x)
    idx = int(np.argmin(np.abs(x - 0.0)))
    if idx < 0 or idx >= len(dydx) or not np.isfinite(dydx[idx]):
        return None
    return float(dydx[idx])


def _compute_group_curve_slopes(
    history_df: pd.DataFrame,
    *,
    cfg: GroupComparisonConfig,
    condition_col: str,
    robust_mean: bool = False,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if history_df.empty or condition_col not in history_df.columns:
        return pd.DataFrame()

    for (condition_value, abl, duration_pair), sub in history_df.groupby(
        [condition_col, "ABL", "duration_pair"], dropna=False, sort=False
    ):
        curve = _compute_psy_curve_for_subset(sub, cfg=cfg, robust_mean=robust_mean, psychometric_l2=psychometric_l2)
        if curve is None:
            continue
        fit = curve["psy_mean_fits"].get(int(abl)) if curve["psy_mean_fits"] is not None else None
        if fit is None:
            continue
        shown_slope = _displayed_curve_slope_at_zero(fit["xx"], fit["yy"])
        if shown_slope is None:
            continue
        rows.append(
            {
                condition_col: condition_value,
                "ABL": int(abl),
                "duration_pair": duration_pair,
                "group_curve_slope": shown_slope,
            }
        )
    return pd.DataFrame(rows)


def _plot_group_curve_slope_summary(
    slope_df: pd.DataFrame,
    *,
    title_prefix: str,
    style: PlotStyle,
    condition_col: str,
    x_label: str,
    abls: tuple[int, ...] = (20, 40, 60),
) -> dict[str, plt.Figure]:
    if slope_df.empty:
        return {}

    if condition_col == "transition_type":
        condition_order = _history_transition_order(slope_df)
        condition_colors = {
            "U->R": "#1F77B4",
            "L->R": "#17BECF",
            "U->L": "#D62728",
            "R->L": "#FF9896",
        }
    else:
        condition_order = _current_block_order(slope_df)
        condition_colors = {"rightward": "#1F77B4", "leftward": "#D62728"}
    if not condition_order:
        return {}

    duration_order = _duration_pair_order(slope_df)
    if not duration_order:
        return {}

    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()
    fs = style.legend_fs
    ylims = _global_param_ylim(slope_df.rename(columns={"group_curve_slope": "effective_slope"}), ["effective_slope"])
    figs: dict[str, plt.Figure] = {}

    for abl in abls:
        abl_df = slope_df[pd.to_numeric(slope_df["ABL"], errors="coerce").eq(int(abl))].copy()
        if abl_df.empty:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.3))
        x_positions = {name: i for i, name in enumerate(condition_order)}
        for condition in condition_order:
            for duration_pair in duration_order:
                sub = abl_df[
                    (abl_df[condition_col].astype(str) == str(condition))
                    & (abl_df["duration_pair"].astype(str) == duration_pair)
                ].copy()
                if sub.empty:
                    continue
                x0 = x_positions[condition] + duration_offsets[duration_pair]
                color = condition_colors.get(condition, "0.4")
                marker = duration_markers.get(duration_pair, "o")
                y = float(pd.to_numeric(sub["group_curve_slope"], errors="coerce").iloc[0])
                ax.plot(
                    x0,
                    y,
                    marker=marker,
                    linestyle="None",
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=8.0,
                    zorder=4,
                )

        ax.set_title("Group psychometric slope", fontsize=fs, pad=8)
        ax.set_xlabel(x_label, fontsize=fs)
        ax.set_ylabel("Effective slope", fontsize=fs)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(condition_order, fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        if "effective_slope" in ylims:
            ax.set_ylim(*ylims["effective_slope"])
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

        handles = [
            Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name)
            for name in duration_order
        ]
        fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
        fig.suptitle(f"{title_prefix} - ABL {abl}", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.08, 1, 0.92])
        figs[f"ABL_{abl}"] = fig
    return figs


def _plot_suspicious_psychometric_cases(
    history_df: pd.DataFrame,
    params: pd.DataFrame,
    *,
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    title_prefix: str,
    condition_col: str,
    condition_label: str,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> tuple[dict[str, plt.Figure], pd.DataFrame]:
    suspicious = _find_suspicious_slope_cases(params, condition_col=condition_col)
    if suspicious.empty:
        return {}, pd.DataFrame()

    duration_order = _duration_pair_order(history_df)
    duration_colors = _duration_pair_colors()
    duration_markers = _duration_pair_markers()
    figures: dict[str, plt.Figure] = {}
    slope_rows: list[dict[str, Any]] = []
    fs = style.legend_fs

    for _, row in suspicious.iterrows():
        condition_value = row[condition_col]
        abl = int(row["ABL"])
        metric_col = row["slope_metric"] if "slope_metric" in row and pd.notna(row["slope_metric"]) else ("slope_at_0" if "slope_at_0" in params.columns else ("effective_slope" if "effective_slope" in params.columns else "slope_a"))
        fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.8))
        plotted_any = False
        slope_pairs = []
        for duration_pair in duration_order:
            sub = history_df[
                (history_df[condition_col].astype(str) == str(condition_value))
                & (pd.to_numeric(history_df["ABL"], errors="coerce").eq(abl))
                & (history_df["duration_pair"].astype(str) == duration_pair)
            ].copy()
            curve = _compute_psy_curve_for_subset(sub, cfg=cfg, psychometric_l2=psychometric_l2)
            if curve is None:
                continue
            slope_vals = pd.to_numeric(
                params.loc[
                    (params[condition_col].astype(str) == str(condition_value))
                    & (pd.to_numeric(params["ABL"], errors="coerce").eq(abl))
                    & (params["duration_pair"].astype(str) == duration_pair),
                    metric_col,
                ],
                errors="coerce",
            ).dropna()
            if not slope_vals.empty:
                slope_summary = float(slope_vals.mean())
                slope_pairs.append((duration_pair, slope_summary))
                slope_rows.append(
                    {
                        "condition_type": condition_col,
                        "condition": condition_value,
                        "ABL": abl,
                        "duration_pair": duration_pair,
                        "slope_metric": metric_col,
                        "slope_summary": slope_summary,
                        "slope_summary_type": "mean",
                        "n_animals_with_slope": int(len(slope_vals)),
                        "used_trimmed_mean": pd.NA,
                    }
                )
            psy_group = curve["psy_group"]
            fit = curve["psy_mean_fits"].get(abl) if curve["psy_mean_fits"] is not None else None
            color = duration_colors.get(duration_pair, "0.4")
            marker = duration_markers.get(duration_pair, "o")
            shown_slope = _displayed_curve_slope_at_zero(fit["xx"], fit["yy"]) if fit is not None else None
            ax.scatter(
                DataHelpers.shift_ILD_for_ABL50(psy_group["ILD"]),
                psy_group["mean"],
                color=color,
                marker=marker,
                s=52,
                linewidth=0.6,
                edgecolor=color,
                label=duration_pair,
                zorder=3,
            )
            if fit is not None:
                ax.plot(
                    DataHelpers.shift_ILD_for_ABL50(fit["xx"]),
                    fit["yy"],
                    color=color,
                    linewidth=1.8,
                )
            if shown_slope is not None:
                slope_rows.append(
                    {
                        "condition_type": condition_col,
                        "condition": condition_value,
                        "ABL": abl,
                        "duration_pair": duration_pair,
                        "slope_metric": "displayed_curve_slope",
                        "slope_summary": shown_slope,
                        "slope_summary_type": "displayed_curve",
                        "n_animals_with_slope": pd.NA,
                        "used_trimmed_mean": pd.NA,
                    }
                )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=-10)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=-10)
        ax.set_title(f"{condition_label}: {condition_value} - ABL {abl}", fontsize=fs, pad=8)
        ax.set_xlabel("ILD (dB)", fontsize=fs)
        ax.set_ylabel("Proportion Left", fontsize=fs)
        ax.tick_params(axis="both", labelsize=fs)
        xticks = sorted(set(ax.get_xticks()) | {-18, 18})
        ax.set_xticks(xticks)
        ax.set_xticklabels(["-50" if x == -18 else "50" if x == 18 else str(int(x)) for x in xticks])
        ax.set_xlim(-19, 19)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=fs, frameon=False, loc="best")
        if slope_pairs:
            lines = []
            for name, value in slope_pairs:
                shown_vals = pd.to_numeric(
                    pd.DataFrame(slope_rows).loc[
                        (pd.DataFrame(slope_rows)["condition"] == condition_value)
                        & (pd.to_numeric(pd.DataFrame(slope_rows)["ABL"], errors="coerce").eq(abl))
                        & (pd.DataFrame(slope_rows)["duration_pair"] == name)
                        & (pd.DataFrame(slope_rows)["slope_metric"] == "displayed_curve_slope"),
                        "slope_summary",
                    ],
                    errors="coerce",
                ).dropna()
                if not shown_vals.empty:
                    lines.append(f"{name}: shown={shown_vals.iloc[0]:.3f}  mean={value:.3f}")
                else:
                    lines.append(f"{name}: mean={value:.3f}")
            slope_text = "\n".join(lines)
            ax.text(
                0.02,
                0.98,
                f"slope at 0\n{slope_text}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=max(9, fs - 2),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.9),
            )
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
        fig.suptitle(f"{title_prefix} - slope at 0 check", fontsize=fs, y=0.98)
        fig.tight_layout()
        figures[f"{condition_value}_ABL_{abl}"] = fig

    return figures, pd.DataFrame(slope_rows)


def _history_transition_rows(
    df_blocks: pd.DataFrame,
    *,
    duration_groups: tuple[tuple[int, ...], ...] = ((8, 16), (32, 64), (120, 0)),
    analysis_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    key_cols = [c for c in ["dataset_key", "animal", "session"] if c in df_blocks.columns]
    if not key_cols:
        raise KeyError("Need animal/session columns to compute block-history summaries.")

    rows = []
    for _, session_df in df_blocks.groupby(key_cols, dropna=False, sort=False):
        if "block" not in session_df.columns:
            continue
        session_df = session_df.copy()
        session_df["_block_num"] = pd.to_numeric(session_df["block"], errors="coerce")
        block_order = sorted(session_df["_block_num"].dropna().unique())
        if len(block_order) < 2:
            continue

        block_condition = (
            session_df.groupby("_block_num")["block_condition"]
            .agg(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().size else pd.NA)
            .to_dict()
        )
        for prev_block, next_block in zip(block_order[:-1], block_order[1:]):
            prev_condition = block_condition.get(prev_block)
            current_condition = block_condition.get(next_block)
            if current_condition not in {"leftward", "rightward"}:
                continue
            if prev_condition not in {"unbiased", "leftward", "rightward"}:
                continue
            if prev_condition == current_condition:
                continue

            next_rows = session_df[session_df["_block_num"] == next_block].sort_values("trial").copy()
            if next_rows.empty:
                continue

            next_rows["prev_block_condition"] = prev_condition
            next_rows["current_block_condition"] = current_condition
            next_rows["transition_type"] = f"{_history_short_label(prev_condition)}->{_history_short_label(current_condition)}"
            next_rows["block_id"] = (
                next_rows["animal"].astype(str).iloc[0]
                + ":"
                + next_rows["session"].astype(str).iloc[0]
                + ":"
                + str(int(next_block))
            )
            rows.append(next_rows)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    success = pd.to_numeric(out["success"], errors="coerce")
    out = out[success.ne(0)].copy()
    out["prob_correct"] = success.loc[out.index].eq(1).astype(float)
    out = out.sort_values(["animal", "session", "block_id", "trial"]).copy()
    out["valid_trial_in_block"] = out.groupby("block_id", dropna=False).cumcount() + 1
    out["n_valid_in_block"] = out.groupby("block_id", dropna=False)["valid_trial_in_block"].transform("max")
    if analysis_df is not None:
        if "_biased_blocks_row_id" not in out or "_biased_blocks_row_id" not in analysis_df:
            raise KeyError("Transition timing requires _biased_blocks_row_id in both timing and analysis data.")
        analysis_rows = set(analysis_df["_biased_blocks_row_id"].dropna().astype(int))
        out = out[out["_biased_blocks_row_id"].astype(int).isin(analysis_rows)].copy()
    short_duration = pd.to_numeric(out.get("short_duration"), errors="coerce")
    out["duration_pair"] = short_duration.map(lambda x: _duration_pair_label(x, duration_groups))
    out = out[out["duration_pair"].notna()].copy()
    out["ABL"] = pd.to_numeric(out["ABL"], errors="coerce")
    out = out[out["ABL"].notna()].copy()
    out["ABL"] = out["ABL"].astype(int)
    return out


def _history_transition_order(df: pd.DataFrame) -> list[str]:
    preferred = ["U->R", "L->R", "U->L", "R->L"]
    present = set(df["transition_type"].dropna().astype(str)) if "transition_type" in df else set()
    return [name for name in preferred if name in present]


def _current_block_order(df: pd.DataFrame) -> list[str]:
    preferred = ["rightward", "leftward"]
    present = set(df["current_block_condition"].dropna().astype(str)) if "current_block_condition" in df else set()
    return [name for name in preferred if name in present]


def _plot_history_performance(
    history_df: pd.DataFrame,
    *,
    title: str,
    style: PlotStyle,
    abls: tuple[int, ...] = (20, 40, 60),
) -> plt.Figure | None:
    transition_order = _history_transition_order(history_df)
    if history_df.empty or not transition_order:
        return None
    duration_order = _duration_pair_order(history_df)
    if not duration_order:
        return None
    transition_colors = {
        "U->R": "#1F77B4",
        "L->R": "#17BECF",
        "U->L": "#D62728",
        "R->L": "#FF9896",
    }
    animal_means = (
        history_df.groupby(["animal", "ABL", "transition_type", "duration_pair"], dropna=False)["prob_correct"]
        .mean()
        .reset_index()
    )

    fs = style.legend_fs
    fig, axes = plt.subplots(1, len(abls), figsize=(5.2 * len(abls), 4.4), squeeze=False, sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()

    for ax, abl in zip(axes, abls):
        abl_df = animal_means[animal_means["ABL"] == int(abl)].copy()
        x_positions = {name: i for i, name in enumerate(transition_order)}
        for transition_type in transition_order:
            for duration_pair in duration_order:
                sub = abl_df[
                    (abl_df["transition_type"].astype(str) == transition_type)
                    & (abl_df["duration_pair"].astype(str) == duration_pair)
                ].copy()
                if sub.empty:
                    continue
                x0 = x_positions[transition_type] + duration_offsets[duration_pair]
                color = transition_colors.get(transition_type, "0.4")
                marker = duration_markers.get(duration_pair, "o")
                jitter = rng.uniform(-0.04, 0.04, size=len(sub))
                ax.scatter(
                    np.full(len(sub), x0, dtype=float) + jitter,
                    sub["prob_correct"].to_numpy(dtype=float),
                    s=28,
                    color=color,
                    marker=marker,
                    alpha=0.55,
                    edgecolors="none",
                    zorder=3,
                )
                mean = float(sub["prob_correct"].mean())
                err = sem(sub["prob_correct"].to_numpy(dtype=float))
                ax.errorbar(
                    x0,
                    mean,
                    yerr=err,
                    fmt=marker,
                    color="black",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=7.5,
                    elinewidth=1.5,
                    capsize=3,
                    zorder=5,
                )

        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
        ax.set_xlabel("Previous -> current block", fontsize=fs)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(transition_order, fontsize=fs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Probability correct", fontsize=fs)
    handles = [
        Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name)
        for name in duration_order
    ]
    fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
    fig.suptitle(title, fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def _compute_history_psy_params(
    history_df: pd.DataFrame,
    *,
    cfg: GroupComparisonConfig,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> pd.DataFrame:
    rows = []
    for (transition_type, duration_pair), sub in history_df.groupby(["transition_type", "duration_pair"], dropna=False, sort=False):
        if sub.empty:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            if float(psychometric_l2) > 0:
                _, _, _, _, jnd_indiv, psy_params = _prep_psy_local(
                    sub,
                    do_individual_fits=True,
                    aggregation=cfg.psychometric_aggregation,
                    skip_jnd_abl=50,
                    l2_strength=psychometric_l2,
                )
            else:
                _, _, _, _, jnd_indiv, psy_params = prep_psy(
                    sub,
                    do_individual_fits=True,
                    aggregation=cfg.psychometric_aggregation,
                    skip_jnd_abl=50,
                )
        if not psy_params.empty:
            tmp = psy_params.copy()
            tmp["transition_type"] = transition_type
            tmp["duration_pair"] = duration_pair
            rows.append(tmp)
        if not jnd_indiv.empty:
            tmp = jnd_indiv.rename(columns={"subject": "animal"}).copy()
            tmp["transition_type"] = transition_type
            tmp["duration_pair"] = duration_pair
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()

    params = pd.concat([r for r in rows if "slope_a" in r.columns], ignore_index=True, sort=False) if any("slope_a" in r.columns for r in rows) else pd.DataFrame()
    if params.empty:
        return params
    params = _add_slope_metrics(params)

    jnd_parts = [r for r in rows if "JND" in r.columns]
    if jnd_parts:
        jnd_df = pd.concat(jnd_parts, ignore_index=True, sort=False)
        params = params.merge(jnd_df[["animal", "ABL", "transition_type", "duration_pair", "JND"]], on=["animal", "ABL", "transition_type", "duration_pair"], how="left")
    return params


def _plot_history_psy_params(
    params: pd.DataFrame,
    *,
    title_prefix: str,
    style: PlotStyle,
    abls: tuple[int, ...] = (20, 40, 60),
) -> dict[str, plt.Figure]:
    if params.empty:
        return {}

    transition_order = _history_transition_order(params)
    if not transition_order:
        return {}
    duration_order = _duration_pair_order(params)
    if not duration_order:
        return {}
    transition_colors = {
        "U->R": "#1F77B4",
        "L->R": "#17BECF",
        "U->L": "#D62728",
        "R->L": "#FF9896",
    }
    specs = [
        ("bias_b", "Bias (b)"),
        ("slope_at_0", "Slope at 0"),
        ("JND", "JND"),
    ]
    fs = style.legend_fs
    figs = {}
    rng = np.random.default_rng(1)
    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()
    global_ylims = _global_param_ylim(params, [col for col, _ in specs])
    global_ylims.update(_full_param_ylim(params, ["slope_at_0"]))

    for abl in abls:
        abl_df = params[pd.to_numeric(params["ABL"], errors="coerce").eq(int(abl))].copy()
        if abl_df.empty:
            continue
        fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.3), squeeze=False)
        axes = axes.ravel()
        x_positions = {name: i for i, name in enumerate(transition_order)}

        for ax, (col, label) in zip(axes, specs):
            sub_all = abl_df[["animal", "transition_type", "duration_pair", col]].copy()
            sub_all[col] = pd.to_numeric(sub_all[col], errors="coerce")
            sub_all = sub_all.dropna(subset=[col])
            for transition_type in transition_order:
                for duration_pair in duration_order:
                    sub = sub_all[
                        (sub_all["transition_type"].astype(str) == transition_type)
                        & (sub_all["duration_pair"].astype(str) == duration_pair)
                    ].copy()
                    if sub.empty:
                        continue
                    x0 = x_positions[transition_type] + duration_offsets[duration_pair]
                    color = transition_colors.get(transition_type, "0.4")
                    marker = duration_markers.get(duration_pair, "o")
                    jitter = rng.uniform(-0.04, 0.04, size=len(sub))
                    ax.scatter(
                        np.full(len(sub), x0, dtype=float) + jitter,
                        sub[col].to_numpy(dtype=float),
                        s=28,
                        color=color,
                        marker=marker,
                        alpha=0.75,
                        edgecolors="black" if col == "slope_at_0" else "none",
                        linewidths=0.25 if col == "slope_at_0" else 0.0,
                        zorder=3,
                    )
                    mean = float(sub[col].mean())
                    err = sem(sub[col].to_numpy(dtype=float))
                    ax.errorbar(
                        x0,
                        mean,
                        yerr=err,
                        fmt=marker,
                        color="black",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markersize=7.5,
                        elinewidth=1.5,
                        capsize=3,
                        zorder=5,
                    )
            ax.set_title(label, fontsize=fs, pad=8)
            ax.set_xlabel("Previous -> current block", fontsize=fs)
            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels(transition_order, fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            if col in global_ylims:
                ax.set_ylim(*global_ylims[col])
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

        handles = [
            Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name)
            for name in duration_order
        ]
        fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
        fig.suptitle(f"{title_prefix} - ABL {abl}", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.08, 1, 0.92])
        figs[f"ABL_{abl}"] = fig
    return figs


def _plot_history_early_late(
    history_df: pd.DataFrame,
    *,
    title: str,
    style: PlotStyle,
    early_n: int = 10,
    late_n: int = 10,
    abls: tuple[int, ...] = (20, 40, 60),
) -> plt.Figure | None:
    df = history_df.copy()
    df["phase"] = pd.NA
    df.loc[df["valid_trial_in_block"] <= int(early_n), "phase"] = "early"
    df.loc[(df["n_valid_in_block"] - df["valid_trial_in_block"] + 1) <= int(late_n), "phase"] = (
        df.loc[(df["n_valid_in_block"] - df["valid_trial_in_block"] + 1) <= int(late_n), "phase"].fillna("late")
    )
    df = df[df["phase"].notna()].copy()
    if df.empty:
        return None

    animal_means = (
        df.groupby(["animal", "ABL", "transition_type", "phase", "duration_pair"], dropna=False)["prob_correct"]
        .mean()
        .reset_index()
    )
    transition_order = _history_transition_order(df)
    if not transition_order:
        return None
    duration_order = _duration_pair_order(df)
    if not duration_order:
        return None
    phase_order = ["early", "late"]
    phase_colors = {"early": "#7F7F7F", "late": "#111111"}
    duration_markers = _duration_pair_markers()
    fs = style.legend_fs
    fig, axes = plt.subplots(1, len(abls), figsize=(5.2 * len(abls), 4.4), squeeze=False, sharey=True)
    axes = axes.ravel()
    x_positions = {name: i for i, name in enumerate(transition_order)}
    phase_offsets = {"early": -0.12, "late": 0.12}
    duration_offsets = _duration_pair_offsets(duration_order)

    for ax, abl in zip(axes, abls):
        abl_df = animal_means[animal_means["ABL"] == int(abl)].copy()
        for phase in phase_order:
            for transition_type in transition_order:
                for duration_pair in duration_order:
                    sub = abl_df[
                        (abl_df["phase"].astype(str) == phase)
                        & (abl_df["transition_type"].astype(str) == transition_type)
                        & (abl_df["duration_pair"].astype(str) == duration_pair)
                    ].copy()
                    if sub.empty:
                        continue
                    x0 = x_positions[transition_type] + phase_offsets[phase] + duration_offsets[duration_pair] * 0.5
                    mean = float(sub["prob_correct"].mean())
                    err = sem(sub["prob_correct"].to_numpy(dtype=float))
                    ax.errorbar(
                        x0,
                        mean,
                        yerr=err,
                        fmt=duration_markers[duration_pair],
                        color=phase_colors[phase],
                        markerfacecolor=phase_colors[phase],
                        markeredgecolor=phase_colors[phase],
                        markersize=7.0,
                        elinewidth=1.5,
                        capsize=3,
                        linestyle="None",
                        zorder=5,
                    )
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
        ax.set_xlabel("Previous -> current block", fontsize=fs)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(transition_order, fontsize=fs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Probability correct", fontsize=fs)
    handles = [
        Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name)
        for name in duration_order
    ]
    fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
    fig.suptitle(title, fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    return fig


def _plot_current_block_performance(
    history_df: pd.DataFrame,
    *,
    title: str,
    style: PlotStyle,
    abls: tuple[int, ...] = (20, 40, 60),
) -> plt.Figure | None:
    block_order = _current_block_order(history_df)
    if history_df.empty or not block_order:
        return None
    duration_order = _duration_pair_order(history_df)
    if not duration_order:
        return None
    colors = {"rightward": "#1F77B4", "leftward": "#D62728"}
    animal_means = (
        history_df.groupby(["animal", "ABL", "current_block_condition", "duration_pair"], dropna=False)["prob_correct"]
        .mean()
        .reset_index()
    )

    fs = style.legend_fs
    fig, axes = plt.subplots(1, len(abls), figsize=(5.2 * len(abls), 4.4), squeeze=False, sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(2)
    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()

    for ax, abl in zip(axes, abls):
        abl_df = animal_means[animal_means["ABL"] == int(abl)].copy()
        x_positions = {name: i for i, name in enumerate(block_order)}
        for block_name in block_order:
            for duration_pair in duration_order:
                sub = abl_df[
                    (abl_df["current_block_condition"].astype(str) == block_name)
                    & (abl_df["duration_pair"].astype(str) == duration_pair)
                ].copy()
                if sub.empty:
                    continue
                x0 = x_positions[block_name] + duration_offsets[duration_pair]
                color = colors.get(block_name, "0.4")
                marker = duration_markers.get(duration_pair, "o")
                jitter = rng.uniform(-0.04, 0.04, size=len(sub))
                ax.scatter(
                    np.full(len(sub), x0, dtype=float) + jitter,
                    sub["prob_correct"].to_numpy(dtype=float),
                    s=28,
                    color=color,
                    marker=marker,
                    alpha=0.55,
                    edgecolors="none",
                    zorder=3,
                )
                mean = float(sub["prob_correct"].mean())
                err = sem(sub["prob_correct"].to_numpy(dtype=float))
                ax.errorbar(
                    x0,
                    mean,
                    yerr=err,
                    fmt=marker,
                    color="black",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=7.5,
                    elinewidth=1.5,
                    capsize=3,
                    zorder=5,
                )
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
        ax.set_xlabel("Current block", fontsize=fs)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(block_order, fontsize=fs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Probability correct", fontsize=fs)
    handles = [
        Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name)
        for name in duration_order
    ]
    fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
    fig.suptitle(title, fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def _compute_current_block_psy_params(
    history_df: pd.DataFrame,
    *,
    cfg: GroupComparisonConfig,
    psychometric_l2: float = PSYCHOMETRIC_L2,
) -> pd.DataFrame:
    rows = []
    for (block_name, duration_pair), sub in history_df.groupby(["current_block_condition", "duration_pair"], dropna=False, sort=False):
        if sub.empty:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            if float(psychometric_l2) > 0:
                _, _, _, _, jnd_indiv, psy_params = _prep_psy_local(
                    sub,
                    do_individual_fits=True,
                    aggregation=cfg.psychometric_aggregation,
                    skip_jnd_abl=50,
                    l2_strength=psychometric_l2,
                )
            else:
                _, _, _, _, jnd_indiv, psy_params = prep_psy(
                    sub,
                    do_individual_fits=True,
                    aggregation=cfg.psychometric_aggregation,
                    skip_jnd_abl=50,
                )
        if not psy_params.empty:
            tmp = psy_params.copy()
            tmp["current_block_condition"] = block_name
            tmp["duration_pair"] = duration_pair
            rows.append(tmp)
        if not jnd_indiv.empty:
            tmp = jnd_indiv.rename(columns={"subject": "animal"}).copy()
            tmp["current_block_condition"] = block_name
            tmp["duration_pair"] = duration_pair
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()

    params = pd.concat([r for r in rows if "slope_a" in r.columns], ignore_index=True, sort=False) if any("slope_a" in r.columns for r in rows) else pd.DataFrame()
    if params.empty:
        return params
    params = _add_slope_metrics(params)

    jnd_parts = [r for r in rows if "JND" in r.columns]
    if jnd_parts:
        jnd_df = pd.concat(jnd_parts, ignore_index=True, sort=False)
        params = params.merge(jnd_df[["animal", "ABL", "current_block_condition", "duration_pair", "JND"]], on=["animal", "ABL", "current_block_condition", "duration_pair"], how="left")
    return params


def _plot_current_block_psy_params(
    params: pd.DataFrame,
    *,
    title_prefix: str,
    style: PlotStyle,
    abls: tuple[int, ...] = (20, 40, 60),
) -> dict[str, plt.Figure]:
    if params.empty:
        return {}

    block_order = _current_block_order(params)
    if not block_order:
        return {}
    duration_order = _duration_pair_order(params)
    if not duration_order:
        return {}
    colors = {"rightward": "#1F77B4", "leftward": "#D62728"}
    specs = [("bias_b", "Bias (b)"), ("slope_at_0", "Slope at 0"), ("JND", "JND")]
    fs = style.legend_fs
    figs = {}
    rng = np.random.default_rng(3)
    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()
    global_ylims = _global_param_ylim(params, [col for col, _ in specs])
    global_ylims.update(_full_param_ylim(params, ["slope_at_0"]))

    for abl in abls:
        abl_df = params[pd.to_numeric(params["ABL"], errors="coerce").eq(int(abl))].copy()
        if abl_df.empty:
            continue
        fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.3), squeeze=False)
        axes = axes.ravel()
        x_positions = {name: i for i, name in enumerate(block_order)}

        for ax, (col, label) in zip(axes, specs):
            sub_all = abl_df[["animal", "current_block_condition", "duration_pair", col]].copy()
            sub_all[col] = pd.to_numeric(sub_all[col], errors="coerce")
            sub_all = sub_all.dropna(subset=[col])
            for block_name in block_order:
                for duration_pair in duration_order:
                    sub = sub_all[
                        (sub_all["current_block_condition"].astype(str) == block_name)
                        & (sub_all["duration_pair"].astype(str) == duration_pair)
                    ].copy()
                    if sub.empty:
                        continue
                    x0 = x_positions[block_name] + duration_offsets[duration_pair]
                    color = colors.get(block_name, "0.4")
                    marker = duration_markers.get(duration_pair, "o")
                    jitter = rng.uniform(-0.04, 0.04, size=len(sub))
                    ax.scatter(
                        np.full(len(sub), x0, dtype=float) + jitter,
                        sub[col].to_numpy(dtype=float),
                        s=28,
                        color=color,
                        marker=marker,
                        alpha=0.75,
                        edgecolors="black" if col == "slope_at_0" else "none",
                        linewidths=0.25 if col == "slope_at_0" else 0.0,
                        zorder=3,
                    )
                    mean = float(sub[col].mean())
                    err = sem(sub[col].to_numpy(dtype=float))
                    ax.errorbar(
                        x0,
                        mean,
                        yerr=err,
                        fmt=marker,
                        color="black",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markersize=7.5,
                        elinewidth=1.5,
                        capsize=3,
                        zorder=5,
                    )
            ax.set_title(label, fontsize=fs, pad=8)
            ax.set_xlabel("Current block", fontsize=fs)
            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels(block_order, fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            if col in global_ylims:
                ax.set_ylim(*global_ylims[col])
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
        handles = [
            Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name)
            for name in duration_order
        ]
        fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
        fig.suptitle(f"{title_prefix} - ABL {abl}", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.08, 1, 0.92])
        figs[f"ABL_{abl}"] = fig
    return figs


def _plot_current_block_early_late(
    history_df: pd.DataFrame,
    *,
    title: str,
    style: PlotStyle,
    early_n: int = 10,
    late_n: int = 10,
    abls: tuple[int, ...] = (20, 40, 60),
) -> plt.Figure | None:
    df = history_df.copy()
    df["phase"] = pd.NA
    df.loc[df["valid_trial_in_block"] <= int(early_n), "phase"] = "early"
    df.loc[(df["n_valid_in_block"] - df["valid_trial_in_block"] + 1) <= int(late_n), "phase"] = (
        df.loc[(df["n_valid_in_block"] - df["valid_trial_in_block"] + 1) <= int(late_n), "phase"].fillna("late")
    )
    df = df[df["phase"].notna()].copy()
    if df.empty:
        return None

    animal_means = (
        df.groupby(["animal", "ABL", "current_block_condition", "phase", "duration_pair"], dropna=False)["prob_correct"]
        .mean()
        .reset_index()
    )
    block_order = _current_block_order(df)
    if not block_order:
        return None
    duration_order = _duration_pair_order(df)
    if not duration_order:
        return None
    phase_order = ["early", "late"]
    phase_colors = {"early": "#7F7F7F", "late": "#111111"}
    duration_markers = _duration_pair_markers()
    fs = style.legend_fs
    fig, axes = plt.subplots(1, len(abls), figsize=(5.2 * len(abls), 4.4), squeeze=False, sharey=True)
    axes = axes.ravel()
    x_positions = {name: i for i, name in enumerate(block_order)}
    phase_offsets = {"early": -0.12, "late": 0.12}
    duration_offsets = _duration_pair_offsets(duration_order)

    for ax, abl in zip(axes, abls):
        abl_df = animal_means[animal_means["ABL"] == int(abl)].copy()
        for phase in phase_order:
            for block_name in block_order:
                for duration_pair in duration_order:
                    sub = abl_df[
                        (abl_df["phase"].astype(str) == phase)
                        & (abl_df["current_block_condition"].astype(str) == block_name)
                        & (abl_df["duration_pair"].astype(str) == duration_pair)
                    ].copy()
                    if sub.empty:
                        continue
                    x0 = x_positions[block_name] + phase_offsets[phase] + duration_offsets[duration_pair] * 0.5
                    mean = float(sub["prob_correct"].mean())
                    err = sem(sub["prob_correct"].to_numpy(dtype=float))
                    ax.errorbar(
                        x0,
                        mean,
                        yerr=err,
                        fmt=duration_markers[duration_pair],
                        color=phase_colors[phase],
                        markerfacecolor=phase_colors[phase],
                        markeredgecolor=phase_colors[phase],
                        markersize=7.0,
                        elinewidth=1.5,
                        capsize=3,
                        linestyle="None",
                        zorder=5,
                    )
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
        ax.set_xlabel("Current block", fontsize=fs)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(block_order, fontsize=fs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Probability correct", fontsize=fs)
    handles = [Line2D([], [], color="black", marker=duration_markers[name], linestyle="None", markerfacecolor="white", markeredgecolor="black", label=name) for name in duration_order]
    fig.legend(handles=handles, labels=duration_order, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(duration_order)), fontsize=fs, frameon=False)
    fig.suptitle(title, fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    return fig


def plot_block_history_exploration(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    abls: tuple[int, ...] = (20, 40, 60),
    early_n: int = 10,
    late_n: int = 10,
    duration_groups: tuple[tuple[int, ...], ...] = ((8, 16), (32, 64), (120, 0)),
    show: bool = True,
) -> dict[str, Any]:
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    cfg = bundle["cfg"]
    psychometric_l2 = float(bundle.get("psychometric_l2", PSYCHOMETRIC_L2))
    views = views or bundle["views"]
    outputs = {}

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        history_df = _history_transition_rows(
            df_timing_view,
            duration_groups=duration_groups,
            analysis_df=df_view,
        )
        if history_df.empty:
            continue

        performance_fig = _plot_history_performance(
            history_df,
            title=f"{view.name} - performance by previous/current block",
            style=style,
            abls=abls,
        )
        params = _compute_history_psy_params(history_df, cfg=cfg, psychometric_l2=psychometric_l2)
        param_figs = _plot_history_psy_params(
            params,
            title_prefix=f"{view.name} - psychometric summary by previous/current block",
            style=style,
            abls=abls,
        )
        diagnostic_param_figs, diagnostic_slope_table = _plot_suspicious_psychometric_cases(
            history_df,
            params,
            cfg=cfg,
            style=style,
            title_prefix=f"{view.name} - previous/current psychometrics",
            condition_col="transition_type",
            condition_label="Transition",
            psychometric_l2=psychometric_l2,
        )
        phase_fig = _plot_history_early_late(
            history_df,
            title=f"{view.name} - early vs late within biased blocks",
            style=style,
            early_n=early_n,
            late_n=late_n,
            abls=abls,
        )
        current_performance_fig = _plot_current_block_performance(
            history_df,
            title=f"{view.name} - performance by current block",
            style=style,
            abls=abls,
        )
        current_params = _compute_current_block_psy_params(history_df, cfg=cfg, psychometric_l2=psychometric_l2)
        current_param_figs = _plot_current_block_psy_params(
            current_params,
            title_prefix=f"{view.name} - psychometric summary by current block",
            style=style,
            abls=abls,
        )
        current_diagnostic_param_figs, current_diagnostic_slope_table = _plot_suspicious_psychometric_cases(
            history_df,
            current_params,
            cfg=cfg,
            style=style,
            title_prefix=f"{view.name} - current-block psychometrics",
            condition_col="current_block_condition",
            condition_label="Current block",
            psychometric_l2=psychometric_l2,
        )
        current_phase_fig = _plot_current_block_early_late(
            history_df,
            title=f"{view.name} - early vs late by current biased block",
            style=style,
            early_n=early_n,
            late_n=late_n,
            abls=abls,
        )
        status = {
            "performance_figure": "ok" if performance_fig is not None else "skipped_no_data",
            "phase_figure": "ok" if phase_fig is not None else "skipped_no_data",
            "psy_param_figures": sorted(param_figs.keys()),
            "diagnostic_psychometric_figures": sorted(diagnostic_param_figs.keys()),
            "current_performance_figure": "ok" if current_performance_fig is not None else "skipped_no_data",
            "current_phase_figure": "ok" if current_phase_fig is not None else "skipped_no_data",
            "current_psy_param_figures": sorted(current_param_figs.keys()),
            "current_diagnostic_psychometric_figures": sorted(current_diagnostic_param_figs.keys()),
        }

        outputs[view.name] = {
            "history_rows": history_df,
            "performance_figure": performance_fig,
            "psy_params": params,
            "psy_param_figures": param_figs,
            "diagnostic_psychometric_figures": diagnostic_param_figs,
            "diagnostic_slope_table": diagnostic_slope_table,
            "phase_figure": phase_fig,
            "current_performance_figure": current_performance_fig,
            "current_psy_params": current_params,
            "current_psy_param_figures": current_param_figs,
            "current_diagnostic_psychometric_figures": current_diagnostic_param_figs,
            "current_diagnostic_slope_table": current_diagnostic_slope_table,
            "current_phase_figure": current_phase_fig,
            "status": status,
        }
        if show:
            if performance_fig is not None:
                plt.figure(performance_fig.number)
                plt.show()
            for fig in param_figs.values():
                plt.figure(fig.number)
                plt.show()
            for fig in diagnostic_param_figs.values():
                plt.figure(fig.number)
                plt.show()
            if phase_fig is not None:
                plt.figure(phase_fig.number)
                plt.show()
            if current_performance_fig is not None:
                plt.figure(current_performance_fig.number)
                plt.show()
            for fig in current_param_figs.values():
                plt.figure(fig.number)
                plt.show()
            for fig in current_diagnostic_param_figs.values():
                plt.figure(fig.number)
                plt.show()
            if current_phase_fig is not None:
                plt.figure(current_phase_fig.number)
                plt.show()
        else:
            figures_to_close = []
            if performance_fig is not None:
                figures_to_close.append(performance_fig)
            figures_to_close.extend(param_figs.values())
            figures_to_close.extend(diagnostic_param_figs.values())
            if phase_fig is not None:
                figures_to_close.append(phase_fig)
            if current_performance_fig is not None:
                figures_to_close.append(current_performance_fig)
            figures_to_close.extend(current_param_figs.values())
            figures_to_close.extend(current_diagnostic_param_figs.values())
            if current_phase_fig is not None:
                figures_to_close.append(current_phase_fig)
            for fig in figures_to_close:
                plt.close(fig)

    return outputs


def plot_left_to_right_transition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    show: bool = True,
) -> dict[str, Any]:
    """Plot probability-correct traces around left->right and right->left transitions."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    fs = style.legend_fs
    outputs = {}

    ild_group_order = ["hard left", "hard right", "easy left", "easy right"]
    ild_group_colors = {
        "hard left": "#6A3D9A",
        "hard right": "#1B9E77",
        "easy left": "#E69F00",
        "easy right": "#D55E00",
    }
    ild_group_markers = {
        "hard left": "o",
        "hard right": "o",
        "easy left": "s",
        "easy right": "s",
    }

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        if df_view.empty or df_timing_view.empty:
            continue
        direction_specs = [
            ("leftward", "rightward", "leftward to rightward"),
            ("rightward", "leftward", "rightward to leftward"),
        ]
        direction_payloads = []
        for from_condition, to_condition, label in direction_specs:
            window_df = _transition_window_rows(
                df_timing_view,
                window=window,
                from_condition=from_condition,
                to_condition=to_condition,
                analysis_df=df_view,
            )
            animal_trace = _animal_transition_trace(window_df)
            if animal_trace.empty:
                continue
            animal_trace = _bin_transition_trace(animal_trace, "prob_correct", transition_bin_size)
            animal_trace["view"] = view.name
            direction_payloads.append((label, window_df, animal_trace))

        if not direction_payloads:
            continue

        fig, axes = plt.subplots(
            len(direction_payloads),
            len(abls),
            figsize=(5.2 * len(abls), 4.4 * len(direction_payloads)),
            squeeze=False,
            sharey=True,
        )

        for row_i, (direction_label, window_df, animal_trace) in enumerate(direction_payloads):
            row_axes = axes[row_i]
            for ax, abl in zip(row_axes, abls):
                abl_df = animal_trace[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl)].copy()
                for ild_group in ild_group_order:
                    sub = abl_df[abl_df["signed_ild_group"].astype(str) == ild_group].copy()
                    if sub.empty:
                        continue
                    summary = (
                        sub.groupby("relative_trial")["prob_correct"]
                        .agg(mean="mean", sem=sem, n_animals="count")
                        .reset_index()
                    )
                    x = summary["relative_trial"].to_numpy(dtype=float)
                    y = summary["mean"].to_numpy(dtype=float)
                    yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
                    ax.plot(
                        x,
                        y,
                        color=ild_group_colors[ild_group],
                        marker=ild_group_markers[ild_group],
                        linestyle="-",
                        markersize=4.0,
                        linewidth=1.6,
                        label=ild_group,
                    )
                    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                    if finite.any():
                        ax.fill_between(
                            x[finite],
                            np.clip(y[finite] - yerr[finite], 0, 1),
                            np.clip(y[finite] + yerr[finite], 0, 1),
                            color=ild_group_colors[ild_group],
                            alpha=0.18,
                            linewidth=0,
                        )

                ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
                ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
                ax.set_title(f"{direction_label} - ABL {abl}", fontsize=fs, pad=8)
                ax.set_xlabel("Trials from block transition", fontsize=fs)
                ax.set_ylim(0, 1)
                ax.set_xlim(-window - 1, window + 1)
                ax.set_xticks([-20, -10, 0, 10, 20])
                ax.set_xticklabels(["-20", "-10", "0", "10", "20"])
                ax.tick_params(axis="both", labelsize=fs)
                for spine in ["right", "top"]:
                    ax.spines[spine].set_visible(False)
            row_axes[0].set_ylabel("Probability correct", fontsize=fs)

        handles = [
            Line2D([], [], color=ild_group_colors[group], marker=ild_group_markers[group], linestyle="-", label=group)
            for group in ild_group_order
        ]
        fig.legend(
            handles=handles,
            labels=["hard left (-1,-2)", "hard right (1,2)", "easy left (-8,-16)", "easy right (8,16)"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=fs,
            frameon=False,
        )
        fig.suptitle(f"{view.name} - signed biased-block transitions", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.10, 1, 0.95])
        outputs[view.name] = {
            "figure": fig,
            "transition_bin_size": transition_bin_size,
            "direction_payloads": {
                direction_label: {
                    "animal_trace": animal_trace,
                    "transition_rows": window_df,
                }
                for direction_label, window_df, animal_trace in direction_payloads
            },
        }
        if show:
            plt.show()

    return outputs


def plot_left_to_right_transition_genotype_summary(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    view_colors: dict[str, str] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot genotype/view averages around biased-block transitions after unbiased-baseline subtraction and ILD averaging within animal."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    view_colors = view_colors or {}
    fs = style.legend_fs

    direction_specs = [
        ("leftward", "rightward", "leftward to rightward"),
        ("rightward", "leftward", "rightward to leftward"),
    ]
    view_payloads: dict[str, dict[str, pd.DataFrame]] = {}

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        if df_view.empty or df_timing_view.empty:
            continue
        baseline_df = _unbiased_choice_right_baseline(df_view)
        if baseline_df.empty:
            continue
        direction_payloads: dict[str, pd.DataFrame] = {}
        for from_condition, to_condition, label in direction_specs:
            window_df = _transition_window_rows(
                df_timing_view,
                window=window,
                from_condition=from_condition,
                to_condition=to_condition,
                analysis_df=df_view,
            )
            animal_trace = _animal_baseline_subtracted_transition_trace_all_ilds(window_df, baseline_df)
            if animal_trace.empty:
                continue
            animal_trace = _bin_transition_trace(animal_trace, "delta_choice_right", transition_bin_size)
            animal_trace["view"] = view.name
            direction_payloads[label] = animal_trace
        if direction_payloads:
            view_payloads[view.name] = direction_payloads

    if not view_payloads:
        return {}

    fig, axes = plt.subplots(
        len(direction_specs),
        len(abls),
        figsize=(5.2 * len(abls), 4.4 * len(direction_specs)),
        squeeze=False,
        sharey=True,
    )

    all_summary_values = []
    for direction_map in view_payloads.values():
        for animal_trace in direction_map.values():
            vals = pd.to_numeric(animal_trace["delta_choice_right"], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                all_summary_values.append(vals.to_numpy(dtype=float))
    if all_summary_values:
        all_summary = np.concatenate(all_summary_values)
        ymax = np.nanpercentile(np.abs(all_summary), 95)
        ymax = max(0.15, float(ymax))
        ymax = min(ymax, 0.75)
    else:
        ymax = 0.25

    for row_i, (_, _, direction_label) in enumerate(direction_specs):
        row_axes = axes[row_i]
        for ax, abl in zip(row_axes, abls):
            for i, view in enumerate(views):
                animal_trace = view_payloads.get(view.name, {}).get(direction_label)
                if animal_trace is None or animal_trace.empty:
                    continue
                sub = animal_trace[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl)].copy()
                if sub.empty:
                    continue
                summary = (
                    sub.groupby("relative_trial")["delta_choice_right"]
                    .agg(mean="mean", sem=sem, n_animals="count")
                    .reset_index()
                )
                x = summary["relative_trial"].to_numpy(dtype=float)
                y = summary["mean"].to_numpy(dtype=float)
                yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
                color = view_colors.get(view.name, f"C{i % 10}")
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker="o",
                    linestyle="-",
                    markersize=4.0,
                    linewidth=1.8,
                    label=view.name,
                )
                finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                if finite.any():
                    ax.fill_between(
                        x[finite],
                        y[finite] - yerr[finite],
                        y[finite] + yerr[finite],
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )

            ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
            ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
            ax.set_title(f"{direction_label} - ABL {abl}", fontsize=fs, pad=8)
            ax.set_xlabel("Trials from block transition", fontsize=fs)
            ax.set_ylim(-ymax, ymax)
            ax.set_xlim(-window - 1, window + 1)
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_xticklabels(["-20", "-10", "0", "10", "20"])
            ax.tick_params(axis="both", labelsize=fs)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
        row_axes[0].set_ylabel("Delta frac. rightward choices", fontsize=fs)

    handles = [
        Line2D([], [], color=view_colors.get(view.name, f"C{i % 10}"), marker="o", linestyle="-", label=view.name)
        for i, view in enumerate(views)
        if view.name in view_payloads
    ]
    labels = [view.name for view in views if view.name in view_payloads]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, min(4, len(labels))),
        fontsize=fs,
        frameon=False,
    )
    fig.suptitle("Genotype summary - matched transitions vs unbiased baseline", fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    if show:
        plt.show()
    return {
        "figure": fig,
        "transition_bin_size": transition_bin_size,
        "view_payloads": view_payloads,
    }


def plot_aligned_biased_transition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    show: bool = True,
) -> dict[str, Any]:
    """Plot any biased->biased transitions aligned to the new favored side."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    fs = style.legend_fs
    outputs = {}

    ild_group_order = ["hard away", "hard toward", "easy away", "easy toward"]
    ild_group_colors = {
        "hard away": "#6A3D9A",
        "hard toward": "#1B9E77",
        "easy away": "#E69F00",
        "easy toward": "#D55E00",
    }
    ild_group_markers = {
        "hard away": "o",
        "hard toward": "o",
        "easy away": "s",
        "easy toward": "s",
    }

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        if df_view.empty or df_timing_view.empty:
            continue
        window_df = _aligned_biased_transition_window_rows(
            df_timing_view,
            window=window,
            analysis_df=df_view,
        )
        animal_trace = _animal_aligned_transition_trace(window_df)
        if animal_trace.empty:
            continue
        animal_trace = _bin_transition_trace(animal_trace, "frac_toward_new_side", transition_bin_size)
        animal_trace["view"] = view.name

        fig, axes = plt.subplots(
            1,
            len(abls),
            figsize=(5.2 * len(abls), 4.6),
            squeeze=False,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, abl in zip(axes, abls):
            abl_df = animal_trace[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl)].copy()
            for ild_group in ild_group_order:
                sub = abl_df[abl_df["aligned_ild_group"].astype(str) == ild_group].copy()
                if sub.empty:
                    continue
                summary = (
                    sub.groupby("relative_trial")["frac_toward_new_side"]
                    .agg(mean="mean", sem=sem, n_animals="count")
                    .reset_index()
                )
                x = summary["relative_trial"].to_numpy(dtype=float)
                y = summary["mean"].to_numpy(dtype=float)
                yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=ild_group_colors[ild_group],
                    marker=ild_group_markers[ild_group],
                    linestyle="-",
                    markersize=4.0,
                    linewidth=1.6,
                    label=ild_group,
                )
                finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                if finite.any():
                    ax.fill_between(
                        x[finite],
                        np.clip(y[finite] - yerr[finite], 0, 1),
                        np.clip(y[finite] + yerr[finite], 0, 1),
                        color=ild_group_colors[ild_group],
                        alpha=0.18,
                        linewidth=0,
                    )

            ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
            ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
            ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
            ax.set_xlabel("Trials from biased-block transition", fontsize=fs)
            ax.set_ylim(0, 1)
            ax.set_xlim(-window - 1, window + 1)
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_xticklabels(["-20", "-10", "0", "10", "20"])
            ax.tick_params(axis="both", labelsize=fs)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
        axes[0].set_ylabel("Frac. choices to new bias side", fontsize=fs)

        handles = [
            Line2D([], [], color=ild_group_colors[group], marker=ild_group_markers[group], linestyle="-", label=group)
            for group in ild_group_order
        ]
        fig.legend(
            handles=handles,
            labels=[
                "hard away from new side",
                "hard toward new side",
                "easy away from new side",
                "easy toward new side",
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=fs,
            frameon=False,
        )
        fig.suptitle(f"{view.name} - biased transitions aligned to new favored side", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.10, 1, 0.92])
        outputs[view.name] = {
            "figure": fig,
            "animal_trace": animal_trace,
            "transition_rows": window_df,
            "transition_bin_size": transition_bin_size,
        }
        if show:
            plt.show()

    return outputs


def plot_collapsed_biased_transition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    show: bool = True,
) -> dict[str, Any]:
    """Plot any biased->biased transitions with ILDs collapsed to hard vs easy, aligned to the new favored side."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    fs = style.legend_fs
    outputs = {}

    difficulty_order = ["hard", "easy"]
    difficulty_colors = {"hard": "#6A3D9A", "easy": "#E69F00"}
    difficulty_markers = {"hard": "o", "easy": "s"}

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        if df_view.empty or df_timing_view.empty:
            continue
        window_df = _collapsed_biased_transition_window_rows(
            df_timing_view,
            window=window,
            analysis_df=df_view,
        )
        animal_trace = _animal_collapsed_transition_trace(window_df)
        if animal_trace.empty:
            continue
        animal_trace = _bin_transition_trace(animal_trace, "frac_toward_new_side", transition_bin_size)
        animal_trace["view"] = view.name

        fig, axes = plt.subplots(
            1,
            len(abls),
            figsize=(5.2 * len(abls), 4.6),
            squeeze=False,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, abl in zip(axes, abls):
            abl_df = animal_trace[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl)].copy()
            for difficulty in difficulty_order:
                sub = abl_df[abl_df["difficulty_group"].astype(str) == difficulty].copy()
                if sub.empty:
                    continue
                summary = (
                    sub.groupby("relative_trial")["frac_toward_new_side"]
                    .agg(mean="mean", sem=sem, n_animals="count")
                    .reset_index()
                )
                x = summary["relative_trial"].to_numpy(dtype=float)
                y = summary["mean"].to_numpy(dtype=float)
                yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=difficulty_colors[difficulty],
                    marker=difficulty_markers[difficulty],
                    linestyle="-",
                    markersize=4.0,
                    linewidth=1.6,
                    label=difficulty,
                )
                finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                if finite.any():
                    ax.fill_between(
                        x[finite],
                        np.clip(y[finite] - yerr[finite], 0, 1),
                        np.clip(y[finite] + yerr[finite], 0, 1),
                        color=difficulty_colors[difficulty],
                        alpha=0.18,
                        linewidth=0,
                    )

            ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
            ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
            ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
            ax.set_xlabel("Trials from biased-block transition", fontsize=fs)
            ax.set_ylim(0, 1)
            ax.set_xlim(-window - 1, window + 1)
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_xticklabels(["-20", "-10", "0", "10", "20"])
            ax.tick_params(axis="both", labelsize=fs)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
        axes[0].set_ylabel("Frac. choices to new bias side", fontsize=fs)

        handles = [
            Line2D([], [], color=difficulty_colors[group], marker=difficulty_markers[group], linestyle="-", label=group)
            for group in difficulty_order
        ]
        fig.legend(
            handles=handles,
            labels=["hard |1,2| from new side", "easy |8,16| from new side"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=fs,
            frameon=False,
        )
        fig.suptitle(f"{view.name} - biased transitions collapsed by difficulty", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.10, 1, 0.92])
        outputs[view.name] = {
            "figure": fig,
            "animal_trace": animal_trace,
            "transition_rows": window_df,
            "transition_bin_size": transition_bin_size,
        }
        if show:
            plt.show()

    return outputs


def plot_baseline_subtracted_biased_transition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    show: bool = True,
) -> dict[str, Any]:
    """Plot stimulus-matched changes in rightward choice relative to unbiased-block baseline."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    fs = style.legend_fs
    outputs = {}

    ild_group_order = ["hard left", "hard right", "easy left", "easy right"]
    ild_group_colors = {
        "hard left": "#6A3D9A",
        "hard right": "#1B9E77",
        "easy left": "#E69F00",
        "easy right": "#D55E00",
    }
    ild_group_markers = {
        "hard left": "o",
        "hard right": "o",
        "easy left": "s",
        "easy right": "s",
    }

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        if df_view.empty or df_timing_view.empty:
            continue

        baseline_df = _unbiased_choice_right_baseline(df_view)
        if baseline_df.empty:
            continue

        direction_specs = [
            ("leftward", "rightward", "leftward to rightward"),
            ("rightward", "leftward", "rightward to leftward"),
        ]
        direction_payloads = []
        for from_condition, to_condition, label in direction_specs:
            window_df = _transition_window_rows(
                df_timing_view,
                window=window,
                from_condition=from_condition,
                to_condition=to_condition,
                analysis_df=df_view,
            )
            animal_trace = _animal_baseline_subtracted_transition_trace(window_df, baseline_df)
            if animal_trace.empty:
                continue
            animal_trace = _bin_transition_trace(animal_trace, "delta_choice_right", transition_bin_size)
            animal_trace["view"] = view.name
            direction_payloads.append((label, window_df, animal_trace))

        if not direction_payloads:
            continue

        fig, axes = plt.subplots(
            len(direction_payloads),
            len(abls),
            figsize=(5.2 * len(abls), 4.4 * len(direction_payloads)),
            squeeze=False,
            sharey=True,
        )

        all_summary_values = []
        for _, _, animal_trace in direction_payloads:
            vals = pd.to_numeric(animal_trace["delta_choice_right"], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                all_summary_values.append(vals.to_numpy(dtype=float))
        if all_summary_values:
            all_summary = np.concatenate(all_summary_values)
            ymax = np.nanpercentile(np.abs(all_summary), 95)
            ymax = max(0.15, float(ymax))
            ymax = min(ymax, 0.75)
        else:
            ymax = 0.25

        for row_i, (direction_label, window_df, animal_trace) in enumerate(direction_payloads):
            row_axes = axes[row_i]
            for ax, abl in zip(row_axes, abls):
                abl_df = animal_trace[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl)].copy()
                for ild_group in ild_group_order:
                    sub = abl_df[abl_df["signed_ild_group"].astype(str) == ild_group].copy()
                    if sub.empty:
                        continue
                    summary = (
                        sub.groupby("relative_trial")["delta_choice_right"]
                        .agg(mean="mean", sem=sem, n_animals="count")
                        .reset_index()
                    )
                    x = summary["relative_trial"].to_numpy(dtype=float)
                    y = summary["mean"].to_numpy(dtype=float)
                    yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
                    ax.plot(
                        x,
                        y,
                        color=ild_group_colors[ild_group],
                        marker=ild_group_markers[ild_group],
                        linestyle="-",
                        markersize=4.0,
                        linewidth=1.6,
                        label=ild_group,
                    )
                    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                    if finite.any():
                        ax.fill_between(
                            x[finite],
                            y[finite] - yerr[finite],
                            y[finite] + yerr[finite],
                            color=ild_group_colors[ild_group],
                            alpha=0.18,
                            linewidth=0,
                        )

                ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
                ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
                ax.set_title(f"{direction_label} - ABL {abl}", fontsize=fs, pad=8)
                ax.set_xlabel("Trials from block transition", fontsize=fs)
                ax.set_ylim(-ymax, ymax)
                ax.set_xlim(-window - 1, window + 1)
                ax.set_xticks([-20, -10, 0, 10, 20])
                ax.set_xticklabels(["-20", "-10", "0", "10", "20"])
                ax.tick_params(axis="both", labelsize=fs)
                for spine in ["right", "top"]:
                    ax.spines[spine].set_visible(False)
            row_axes[0].set_ylabel("Delta frac. rightward choices", fontsize=fs)

        handles = [
            Line2D([], [], color=ild_group_colors[group], marker=ild_group_markers[group], linestyle="-", label=group)
            for group in ild_group_order
        ]
        fig.legend(
            handles=handles,
            labels=["hard left (-1,-2)", "hard right (1,2)", "easy left (-8,-16)", "easy right (8,16)"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=fs,
            frameon=False,
        )
        fig.suptitle(f"{view.name} - matched transitions vs unbiased baseline", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0, 0.10, 1, 0.95])
        outputs[view.name] = {
            "figure": fig,
            "baseline_rows": baseline_df,
            "transition_bin_size": transition_bin_size,
            "direction_payloads": {
                direction_label: {
                    "animal_trace": animal_trace,
                    "transition_rows": window_df,
                }
                for direction_label, window_df, animal_trace in direction_payloads
            },
        }
        if show:
            plt.show()

    return outputs


def plot_positive_minus_negative_accuracy_transition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    pre_window: int = 10,
    post_window: int = 30,
    abls: tuple[int, ...] = (20, 40, 60),
    separate_abls: bool = True,
    transition_bin_size: int | None = 1,
    view_colors: dict[str, str] | None = None,
    fit_post_transition: bool = True,
    fit_pre_transition_baseline: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Plot the directional half-difference of pooled positive-minus-negative accuracy."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    view_colors = view_colors or {}
    fs = style.legend_fs
    plot_abls = abls if separate_abls else (0,)

    direction_specs = [
        ("leftward", "rightward", "leftward to rightward"),
        ("rightward", "leftward", "rightward to leftward"),
    ]
    view_payloads: dict[str, dict[str, pd.DataFrame]] = {}
    combined_payloads: dict[str, pd.DataFrame] = {}
    individual_fit_payloads: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []

    for view in views:
        df_view = view.selector(df_blocks)
        df_timing_view = view.selector(df_blocks_timing)
        if df_view.empty or df_timing_view.empty:
            continue

        direction_payloads: dict[str, pd.DataFrame] = {}
        for from_condition, to_condition, label in direction_specs:
            window_df = _transition_window_rows(
                df_timing_view,
                window=window,
                pre_window=pre_window,
                post_window=post_window,
                from_condition=from_condition,
                to_condition=to_condition,
                analysis_df=df_view,
            )
            if not separate_abls:
                window_df = window_df.copy()
                window_df["ABL"] = 0
            # Pool all positive and all negative ILDs separately before contrasting them.
            animal_trace = _animal_positive_minus_negative_accuracy_trace(window_df)
            if animal_trace.empty:
                continue
            animal_trace = _bin_transition_trace(
                animal_trace,
                "accuracy_pos_minus_neg",
                transition_bin_size,
            )
            animal_trace["view"] = view.name
            direction_payloads[label] = animal_trace

            for abl in plot_abls:
                raw_rel = pd.to_numeric(
                    window_df.loc[pd.to_numeric(window_df["ABL"], errors="coerce").eq(abl), "relative_trial"],
                    errors="coerce",
                )
                contrast_rel = pd.to_numeric(
                    animal_trace.loc[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl), "relative_trial"],
                    errors="coerce",
                )
                coverage_rows.append(
                    {
                        "view": view.name,
                        "direction": label,
                        "ABL": abl,
                        "last_raw_post_trial": raw_rel[raw_rel.gt(0)].max(),
                        "last_pos_minus_neg_post_trial": contrast_rel[contrast_rel.gt(0)].max(),
                    }
                )

        if direction_payloads:
            view_payloads[view.name] = direction_payloads
            combined_trace = _half_difference_direction_summary(
                direction_payloads.get("leftward to rightward", pd.DataFrame()),
                direction_payloads.get("rightward to leftward", pd.DataFrame()),
                "accuracy_pos_minus_neg",
                "transition_half_difference",
            )
            if not combined_trace.empty:
                combined_trace["view"] = view.name
                combined_payloads[view.name] = combined_trace
                individual_trace = _half_difference_direction_trace(
                    direction_payloads.get("leftward to rightward", pd.DataFrame()),
                    direction_payloads.get("rightward to leftward", pd.DataFrame()),
                    "accuracy_pos_minus_neg",
                    "transition_half_difference",
                )
                individual_fits = _fit_individual_post_transition_traces(
                    individual_trace,
                    "transition_half_difference",
                    include_pre_transition_baseline=fit_pre_transition_baseline,
                )
                if not individual_fits.empty:
                    individual_fit_payloads[view.name] = individual_fits

                for abl in plot_abls:
                    rel = pd.to_numeric(
                        combined_trace.loc[
                            pd.to_numeric(combined_trace["ABL"], errors="coerce").eq(abl),
                            "relative_trial",
                        ],
                        errors="coerce",
                    )
                    coverage_rows.append(
                        {
                            "view": view.name,
                            "direction": "combined half-difference",
                            "ABL": abl,
                            "last_raw_post_trial": np.nan,
                            "last_pos_minus_neg_post_trial": rel[rel.gt(0)].max(),
                        }
                    )

    if not combined_payloads:
        return {}

    fig, axes = plt.subplots(
        1,
        len(plot_abls),
        figsize=(5.2 * len(plot_abls), 5.8),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    all_summary_values = []
    for animal_trace in combined_payloads.values():
        vals = pd.to_numeric(animal_trace["transition_half_difference"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals):
            all_summary_values.append(vals.to_numpy(dtype=float))
    if all_summary_values:
        all_summary = np.concatenate(all_summary_values)
        ymax = np.nanpercentile(np.abs(all_summary), 95)
        ymax = max(0.15, float(ymax))
        ymax = min(ymax, 1.0)
    else:
        ymax = 0.25

    for ax, abl in zip(axes, plot_abls):
        fit_parameter_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        for i, view in enumerate(views):
            animal_trace = combined_payloads.get(view.name)
            if animal_trace is None or animal_trace.empty:
                continue
            sub = animal_trace[pd.to_numeric(animal_trace["ABL"], errors="coerce").eq(abl)].copy()
            if sub.empty:
                continue
            summary = sub.rename(columns={"transition_half_difference": "mean"})[
                ["relative_trial", "mean", "sem", "n_left_to_right", "n_right_to_left"]
            ].copy()
            x = summary["relative_trial"].to_numpy(dtype=float)
            y = summary["mean"].to_numpy(dtype=float)
            yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
            color = view_colors.get(view.name, f"C{i % 10}")
            ax.plot(
                x,
                y,
                color=color,
                marker="o",
                linestyle="None",
                markersize=4.0,
                label=view.name,
            )
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            if finite.any():
                ax.errorbar(x[finite], y[finite], yerr=yerr[finite], fmt="none", color=color, alpha=0.55, capsize=2)
            if fit_post_transition:
                individual_fits = individual_fit_payloads.get(view.name, pd.DataFrame())
                if "ABL" in individual_fits:
                    individual_fits = individual_fits[
                        pd.to_numeric(individual_fits["ABL"], errors="coerce").eq(abl)
                    ].copy()
                fitted = _mean_individual_fit_curve(individual_fits, post_window=post_window)
                if fitted is not None:
                    x_curve, y_curve, _, parameter_mean, parameter_sem = fitted
                    ax.plot(x_curve, y_curve, color=color, linewidth=2.0)
                    fit_parameter_rows.append((view.name, parameter_mean, parameter_sem))

        if fit_post_transition:
            _add_fit_parameter_inset(ax, fit_parameter_rows, label_header="genotype", fontsize=fs)

        ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(f"ABL {abl}" if separate_abls else "All ABLs", fontsize=fs, pad=8)
        ax.set_xlabel("Trials from block transition", fontsize=fs)
        ax.set_ylim(-0.25, ymax)
        ax.set_xlim(-pre_window - 1, post_window + 1)
        ax.set_xticks([-10, 0, 10, 20, 30])
        ax.set_xticklabels(["-10", "0", "10", "20", "30"])
        ax.tick_params(axis="both", labelsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

    handles = [
        Line2D([], [], color=view_colors.get(view.name, f"C{i % 10}"), marker="o", linestyle="-", label=view.name)
        for i, view in enumerate(views)
        if view.name in combined_payloads
    ]
    labels = [view.name for view in views if view.name in combined_payloads]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, min(4, len(labels))),
        fontsize=fs,
        frameon=False,
    )
    duration_label = _stim_duration_title_from_bundle(bundle)
    fig.supylabel("Half Transition Difference", fontsize=fs, x=0.01)
    fig.suptitle(
        f"Genotype summary ({duration_label})",
        fontsize=fs,
        y=0.99,
    )
    fig.tight_layout(rect=[0.04, 0.10, 1, 0.95])
    if show:
        plt.show()
    return {
        "figure": fig,
        "transition_bin_size": transition_bin_size,
        "pre_window": pre_window,
        "post_window": post_window,
        "separate_abls": separate_abls,
        "view_payloads": view_payloads,
        "combined_payloads": combined_payloads,
        "individual_fit_payloads": individual_fit_payloads,
        "coverage": pd.DataFrame(coverage_rows),
    }


def plot_half_difference_rt_transition_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    window: int = 20,
    pre_window: int = 10,
    post_window: int = 30,
    abls: tuple[int, ...] = (20, 40, 60),
    separate_abls: bool = True,
    transition_bin_size: int | None = 1,
    view_colors: dict[str, str] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot directional half-differences of positive-minus-negative correct-trial RT."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    view_colors = view_colors or {}
    fs = style.legend_fs
    plot_abls = abls if separate_abls else (0,)
    direction_specs = [
        ("leftward", "rightward", "leftward to rightward"),
        ("rightward", "leftward", "rightward to leftward"),
    ]
    direction_payloads_by_view: dict[str, dict[str, pd.DataFrame]] = {}
    combined_payloads: dict[str, pd.DataFrame] = {}

    for view in views:
        analysis_view = view.selector(df_blocks)
        timing_view = view.selector(df_blocks_timing)
        if analysis_view.empty or timing_view.empty:
            continue
        direction_payloads: dict[str, pd.DataFrame] = {}
        for from_condition, to_condition, direction_label in direction_specs:
            window_df = _transition_window_rows(
                timing_view,
                window=window,
                pre_window=pre_window,
                post_window=post_window,
                from_condition=from_condition,
                to_condition=to_condition,
                analysis_df=analysis_view,
            )
            if not separate_abls:
                window_df = window_df.copy()
                window_df["ABL"] = 0
            animal_trace = _animal_positive_minus_negative_rt_transition_trace(window_df)
            if animal_trace.empty:
                continue
            direction_payloads[direction_label] = _bin_transition_trace(
                animal_trace,
                "rt_pos_minus_neg",
                transition_bin_size,
            )

        if not direction_payloads:
            continue
        direction_payloads_by_view[view.name] = direction_payloads
        combined = _half_difference_direction_summary(
            direction_payloads.get("leftward to rightward", pd.DataFrame()),
            direction_payloads.get("rightward to leftward", pd.DataFrame()),
            "rt_pos_minus_neg",
            "half_transition_rt",
        )
        if not combined.empty:
            combined_payloads[view.name] = combined

    if not combined_payloads:
        return {}

    fig, axes = plt.subplots(
        1,
        len(plot_abls),
        figsize=(5.2 * len(plot_abls), 5.8),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    values = [
        pd.to_numeric(payload["half_transition_rt"], errors="coerce").dropna().to_numpy(dtype=float)
        for payload in combined_payloads.values()
    ]
    values = [value for value in values if len(value)]
    y_limit = max(0.025, float(np.nanpercentile(np.abs(np.concatenate(values)), 95))) if values else 0.10

    for ax, abl in zip(axes, plot_abls):
        for index, view in enumerate(views):
            payload = combined_payloads.get(view.name)
            if payload is None or payload.empty:
                continue
            sub = payload[pd.to_numeric(payload["ABL"], errors="coerce").eq(abl)].copy()
            if sub.empty:
                continue
            x = pd.to_numeric(sub["relative_trial"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub["half_transition_rt"], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(sub["sem"], errors="coerce").to_numpy(dtype=float)
            color = view_colors.get(view.name, f"C{index % 10}")
            ax.plot(x, y, color=color, marker="o", linestyle="-", markersize=4.0, linewidth=1.8, label=view.name)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            if finite.any():
                ax.fill_between(x[finite], y[finite] - yerr[finite], y[finite] + yerr[finite], color=color, alpha=0.18, linewidth=0)
        ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(f"ABL {abl}" if separate_abls else "All ABLs", fontsize=fs, pad=8)
        ax.set_xlabel("Trials from block transition", fontsize=fs)
        ax.set_xlim(-pre_window - 1, post_window + 1)
        ax.set_xticks([-10, 0, 10, 20, 30])
        ax.set_ylim(-y_limit, y_limit)
        ax.tick_params(axis="both", labelsize=fs)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Half Transition Difference in RT (s)", fontsize=fs)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=max(1, min(4, len(labels))), fontsize=fs, frameon=False)
    fig.suptitle(f"Genotype summary - RT positive minus negative ({_stim_duration_title_from_bundle(bundle)})", fontsize=fs, y=0.99)
    fig.tight_layout(rect=[0.04, 0.10, 1, 0.95])
    if show:
        plt.show()
    return {
        "figure": fig,
        "transition_bin_size": transition_bin_size,
        "pre_window": pre_window,
        "post_window": post_window,
        "separate_abls": separate_abls,
        "direction_payloads": direction_payloads_by_view,
        "combined_payloads": combined_payloads,
    }


def plot_duration_grouped_half_transition_accuracy_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    duration_groups: tuple[tuple[int, ...], ...] = ((8, 16), (32, 64), (120, 0)),
    window: int = 20,
    pre_window: int = 10,
    post_window: int = 30,
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    fit_post_transition: bool = True,
    fit_pre_transition_baseline: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Plot one duration-group trace per ABL panel for each genotype/view."""
    df_blocks = bundle["df_blocks"]
    df_blocks_timing = bundle.get("df_blocks_timing", df_blocks)
    style = bundle["style"]
    views = views or bundle["views"]
    fs = style.legend_fs
    direction_specs = [
        ("leftward", "rightward", "leftward to rightward"),
        ("rightward", "leftward", "rightward to leftward"),
    ]
    outputs: dict[str, dict[str, Any]] = {}

    for view in views:
        timing_view = view.selector(df_blocks_timing)
        if timing_view.empty:
            continue

        duration_payloads: dict[str, pd.DataFrame] = {}
        duration_individual_fit_payloads: dict[str, pd.DataFrame] = {}
        for duration_group in duration_groups:
            duration_values = tuple(int(value) for value in duration_group)
            duration_label = "+".join(str(value) for value in duration_values)
            analysis_df = timing_view[
                pd.to_numeric(timing_view["short_duration"], errors="coerce").isin(duration_values)
            ].copy()
            if analysis_df.empty:
                continue

            direction_payloads: dict[str, pd.DataFrame] = {}
            for from_condition, to_condition, direction_label in direction_specs:
                window_df = _transition_window_rows(
                    timing_view,
                    window=window,
                    pre_window=pre_window,
                    post_window=post_window,
                    from_condition=from_condition,
                    to_condition=to_condition,
                    analysis_df=analysis_df,
                )
                animal_trace = _animal_positive_minus_negative_accuracy_trace(window_df)
                if animal_trace.empty:
                    continue
                direction_payloads[direction_label] = _bin_transition_trace(
                    animal_trace,
                    "accuracy_pos_minus_neg",
                    transition_bin_size,
                )

            combined = _half_difference_direction_summary(
                direction_payloads.get("leftward to rightward", pd.DataFrame()),
                direction_payloads.get("rightward to leftward", pd.DataFrame()),
                "accuracy_pos_minus_neg",
                "transition_half_difference",
            )
            if not combined.empty:
                duration_payloads[duration_label] = combined
                individual_trace = _half_difference_direction_trace(
                    direction_payloads.get("leftward to rightward", pd.DataFrame()),
                    direction_payloads.get("rightward to leftward", pd.DataFrame()),
                    "accuracy_pos_minus_neg",
                    "transition_half_difference",
                )
                individual_fits = _fit_individual_post_transition_traces(
                    individual_trace,
                    "transition_half_difference",
                    include_pre_transition_baseline=fit_pre_transition_baseline,
                )
                if not individual_fits.empty:
                    duration_individual_fit_payloads[duration_label] = individual_fits

        if not duration_payloads:
            continue

        fig, axes = plt.subplots(
            1,
            len(abls),
            figsize=(5.2 * len(abls), 5.8),
            squeeze=False,
            sharey=True,
        )
        axes = axes.ravel()
        values = [
            pd.to_numeric(payload["transition_half_difference"], errors="coerce").dropna().to_numpy(dtype=float)
            for payload in duration_payloads.values()
        ]
        values = [value for value in values if len(value)]
        ymax = max(0.15, float(np.nanpercentile(np.abs(np.concatenate(values)), 95))) if values else 0.25
        ymax = min(ymax, 1.0)

        for ax, abl in zip(axes, abls):
            fit_parameter_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
            for index, (duration_label, payload) in enumerate(duration_payloads.items()):
                summary = payload[pd.to_numeric(payload["ABL"], errors="coerce").eq(abl)].copy()
                if summary.empty:
                    continue
                summary = summary.rename(columns={"transition_half_difference": "mean"})
                x = summary["relative_trial"].to_numpy(dtype=float)
                y = summary["mean"].to_numpy(dtype=float)
                yerr = pd.to_numeric(summary["sem"], errors="coerce").to_numpy(dtype=float)
                color = plt.get_cmap("tab10")(index % 10)
                ax.plot(x, y, color=color, marker="o", linestyle="None", markersize=4.0, label=duration_label)
                finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                if finite.any():
                    ax.errorbar(x[finite], y[finite], yerr=yerr[finite], fmt="none", color=color, alpha=0.55, capsize=2)
                if fit_post_transition:
                    individual_fits = duration_individual_fit_payloads.get(duration_label, pd.DataFrame())
                    if "ABL" in individual_fits:
                        individual_fits = individual_fits[
                            pd.to_numeric(individual_fits["ABL"], errors="coerce").eq(abl)
                        ].copy()
                    fitted = _mean_individual_fit_curve(individual_fits, post_window=post_window)
                    if fitted is not None:
                        x_curve, y_curve, _, parameter_mean, parameter_sem = fitted
                        ax.plot(x_curve, y_curve, color=color, linewidth=2.0)
                        fit_parameter_rows.append((duration_label, parameter_mean, parameter_sem))
            if fit_post_transition:
                _add_fit_parameter_inset(ax, fit_parameter_rows, label_header="duration", fontsize=fs)
            ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
            ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
            ax.set_title(f"ABL {abl}", fontsize=fs, pad=8)
            ax.set_xlabel("Trials from block transition", fontsize=fs)
            ax.set_xlim(-pre_window - 1, post_window + 1)
            ax.set_xticks([-10, 0, 10, 20, 30])
            ax.set_ylim(-0.25, ymax)
            ax.tick_params(axis="both", labelsize=fs)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=max(1, min(3, len(labels))), fontsize=fs, frameon=False)
        duration_label = ", ".join(duration_payloads)
        fig.supylabel("Half Transition Difference", fontsize=fs, x=0.01)
        fig.suptitle(f"Genotype summary - {view.name} (stim durations: {duration_label})", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0.04, 0.10, 1, 0.95])
        if show:
            plt.show()
        outputs[view.name] = {
            "figure": fig,
            "duration_payloads": duration_payloads,
            "duration_individual_fit_payloads": duration_individual_fit_payloads,
            "duration_groups": duration_groups,
        }

    return outputs


def _individual_half_transition_fits(
    timing_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    *,
    pre_window: int,
    post_window: int,
    transition_bin_size: int | None,
    pool_abls: bool,
    fit_pre_transition_baseline: bool,
) -> pd.DataFrame:
    direction_payloads: dict[str, pd.DataFrame] = {}
    for from_condition, to_condition, direction_label in [
        ("leftward", "rightward", "leftward to rightward"),
        ("rightward", "leftward", "rightward to leftward"),
    ]:
        window_df = _transition_window_rows(
            timing_df,
            pre_window=pre_window,
            post_window=post_window,
            from_condition=from_condition,
            to_condition=to_condition,
            analysis_df=analysis_df,
        )
        if pool_abls:
            window_df = window_df.copy()
            window_df["ABL"] = 0
        animal_trace = _animal_positive_minus_negative_accuracy_trace(window_df)
        if not animal_trace.empty:
            direction_payloads[direction_label] = _bin_transition_trace(
                animal_trace,
                "accuracy_pos_minus_neg",
                transition_bin_size,
            )

    individual_trace = _half_difference_direction_trace(
        direction_payloads.get("leftward to rightward", pd.DataFrame()),
        direction_payloads.get("rightward to leftward", pd.DataFrame()),
        "accuracy_pos_minus_neg",
        "transition_half_difference",
    )
    return _fit_individual_post_transition_traces(
        individual_trace,
        "transition_half_difference",
        include_pre_transition_baseline=fit_pre_transition_baseline,
    )


def _plot_fit_parameter_summary(
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    x_order: list[Any],
    x_labels: list[str],
    view_colors: dict[str, str],
    title: str,
    x_label: str,
    fontsize: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.8), squeeze=False)
    axes = axes.ravel()
    _draw_fit_parameter_summary_axes(
        axes,
        summary_df,
        x_col=x_col,
        x_order=x_order,
        x_labels=x_labels,
        view_colors=view_colors,
        x_label=x_label,
        fontsize=fontsize,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=max(1, min(4, len(labels))), fontsize=fontsize, frameon=False)
    fig.suptitle(title, fontsize=fontsize, y=0.99)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    return fig


def _draw_fit_parameter_summary_axes(
    axes: np.ndarray,
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    x_order: list[Any],
    x_labels: list[str],
    view_colors: dict[str, str],
    x_label: str,
    fontsize: float,
) -> None:
    x_positions = np.arange(len(x_order), dtype=float)
    view_names = list(summary_df["view"].dropna().astype(str).unique())

    for ax, parameter in zip(axes, ["a", "b", "c"]):
        for index, view_name in enumerate(view_names):
            sub = summary_df[summary_df["view"].astype(str) == view_name].copy()
            sub = sub.set_index(x_col).reindex(x_order).reset_index()
            y = pd.to_numeric(sub[f"{parameter}_mean"], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(sub[f"{parameter}_sem"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y)
            if not valid.any():
                continue
            color = view_colors.get(view_name, f"C{index % 10}")
            ax.plot(x_positions[valid], y[valid], color=color, marker="o", linewidth=1.8, label=view_name)
            ribbon = valid & np.isfinite(yerr)
            if ribbon.any():
                ax.fill_between(x_positions[ribbon], y[ribbon] - yerr[ribbon], y[ribbon] + yerr[ribbon], color=color, alpha=0.18, linewidth=0)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.set_title(parameter, fontsize=fontsize, pad=8)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis="both", labelsize=fontsize)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Fit parameter value", fontsize=fontsize)


def _draw_half_transition_fit_axis(
    ax: plt.Axes,
    trace_payloads: dict[str, pd.DataFrame],
    individual_fit_payloads: dict[str, pd.DataFrame],
    *,
    abl: int,
    colors: dict[str, Any],
    post_window: int,
) -> None:
    """Draw raw half-transition summaries and the mean individual-animal fits."""
    for label, payload in trace_payloads.items():
        sub = payload[pd.to_numeric(payload["ABL"], errors="coerce").eq(abl)].copy()
        if sub.empty:
            continue
        x = pd.to_numeric(sub["relative_trial"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["transition_half_difference"], errors="coerce").to_numpy(dtype=float)
        yerr = pd.to_numeric(sub["sem"], errors="coerce").to_numpy(dtype=float)
        color = colors[label]
        ax.plot(x, y, color=color, marker="o", linestyle="None", markersize=4.0, label=label)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
        if finite.any():
            ax.errorbar(x[finite], y[finite], yerr=yerr[finite], fmt="none", color=color, alpha=0.55, capsize=2)

        individual_fits = individual_fit_payloads.get(label, pd.DataFrame())
        if "ABL" in individual_fits:
            individual_fits = individual_fits[
                pd.to_numeric(individual_fits["ABL"], errors="coerce").eq(abl)
            ].copy()
        fitted = _mean_individual_fit_curve(individual_fits, post_window=post_window)
        if fitted is None:
            continue
        x_curve, y_curve, _, _, _ = fitted
        ax.plot(x_curve, y_curve, color=color, linewidth=2.0)


def _style_half_transition_fit_axis(
    ax: plt.Axes,
    *,
    title: str,
    y_max: float,
    fontsize: float,
) -> None:
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
    ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_title(title, fontsize=fontsize, pad=8)
    ax.set_xlabel("Trials from block transition", fontsize=fontsize)
    ax.set_xlim(-11, 31)
    ax.set_xticks([-10, 0, 10, 20, 30])
    ax.set_ylim(-0.25, y_max)
    ax.tick_params(axis="both", labelsize=fontsize)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)


def plot_half_transition_fit_parameter_summaries(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    duration_groups: tuple[tuple[int, ...], ...] = ((8, 16), (32, 64), (120, 0)),
    abls: tuple[int, ...] = (20, 40, 60),
    transition_bin_size: int | None = 1,
    view_colors: dict[str, str] | None = None,
    fit_pre_transition_baseline: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Summarize individual-animal half-transition fit parameters across ABL and duration."""
    df_blocks = bundle["df_blocks"]
    timing_df = bundle.get("df_blocks_timing", df_blocks)
    views = views or bundle["views"]
    view_colors = view_colors or {}
    fs = bundle["style"].legend_fs
    by_abl_rows: list[pd.DataFrame] = []
    by_duration_rows: list[pd.DataFrame] = []

    for view in views:
        timing_view = view.selector(timing_df)
        if timing_view.empty:
            continue
        all_duration_fits = _individual_half_transition_fits(
            timing_view,
            timing_view,
            pre_window=10,
            post_window=30,
            transition_bin_size=transition_bin_size,
            pool_abls=False,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
        )
        if not all_duration_fits.empty:
            all_duration_fits["view"] = view.name
            by_abl_rows.append(all_duration_fits)

        for duration_group in duration_groups:
            duration_values = tuple(int(value) for value in duration_group)
            analysis_df = timing_view[
                pd.to_numeric(timing_view["short_duration"], errors="coerce").isin(duration_values)
            ].copy()
            if analysis_df.empty:
                continue
            duration_fits = _individual_half_transition_fits(
                timing_view,
                analysis_df,
                pre_window=10,
                post_window=30,
                transition_bin_size=transition_bin_size,
                pool_abls=True,
                fit_pre_transition_baseline=fit_pre_transition_baseline,
            )
            if not duration_fits.empty:
                duration_fits["view"] = view.name
                duration_fits["duration_group"] = "+".join(str(value) for value in duration_values)
                by_duration_rows.append(duration_fits)

    outputs: dict[str, Any] = {}
    if by_abl_rows:
        fits_by_abl = pd.concat(by_abl_rows, ignore_index=True)
        summary_by_abl = fits_by_abl.groupby(["view", "ABL"], dropna=False)[["a", "b", "c"]].agg(["mean", sem]).reset_index()
        summary_by_abl.columns = ["view", "ABL", "a_mean", "a_sem", "b_mean", "b_sem", "c_mean", "c_sem"]
        timing_bundle = {**bundle, "df_blocks": timing_df}
        trace_output = plot_positive_minus_negative_accuracy_transition_figures(
            timing_bundle,
            views=views,
            pre_window=10,
            post_window=30,
            abls=abls,
            separate_abls=True,
            transition_bin_size=transition_bin_size,
            view_colors=view_colors,
            fit_post_transition=False,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
            show=False,
        )
        if trace_output:
            plt.close(trace_output["figure"])
        fig, axes = plt.subplots(2, len(abls), figsize=(5.2 * len(abls), 11.6), squeeze=False, sharey="row")
        trace_payloads = trace_output.get("combined_payloads", {})
        trace_fits = trace_output.get("individual_fit_payloads", {})
        values = [
            pd.to_numeric(payload["transition_half_difference"], errors="coerce").dropna().to_numpy(dtype=float)
            for payload in trace_payloads.values()
        ]
        values = [value for value in values if len(value)]
        y_max = min(1.0, max(0.15, float(np.nanpercentile(np.abs(np.concatenate(values)), 95)))) if values else 0.25
        trace_colors = {view.name: view_colors.get(view.name, f"C{index % 10}") for index, view in enumerate(views)}
        for ax, abl in zip(axes[0], abls):
            _draw_half_transition_fit_axis(ax, trace_payloads, trace_fits, abl=abl, colors=trace_colors, post_window=30)
            _style_half_transition_fit_axis(ax, title=f"ABL {abl}", y_max=y_max, fontsize=fs)
        _draw_fit_parameter_summary_axes(
            axes[1], summary_by_abl, x_col="ABL", x_order=list(abls), x_labels=[str(abl) for abl in abls],
            view_colors=view_colors, x_label="ABL", fontsize=fs,
        )
        axes[0, 0].set_ylabel("Half Transition Difference", fontsize=fs)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=max(1, min(4, len(labels))), fontsize=fs, frameon=False)
        fig.suptitle("Genotype summary (all stim durations)", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0.04, 0.10, 1, 0.96])
        outputs["all_durations_by_abl"] = {"figure": fig, "individual_fits": fits_by_abl, "summary": summary_by_abl}
        if show:
            plt.show()
    if by_duration_rows:
        fits_by_duration = pd.concat(by_duration_rows, ignore_index=True)
        duration_order = ["+".join(str(value) for value in group) for group in duration_groups]
        summary_by_duration = fits_by_duration.groupby(["view", "duration_group"], dropna=False)[["a", "b", "c"]].agg(["mean", sem]).reset_index()
        summary_by_duration.columns = ["view", "duration_group", "a_mean", "a_sem", "b_mean", "b_sem", "c_mean", "c_sem"]
        trace_outputs = plot_duration_grouped_half_transition_accuracy_figures(
            bundle,
            views=views,
            duration_groups=duration_groups,
            pre_window=10,
            post_window=30,
            abls=abls,
            transition_bin_size=transition_bin_size,
            fit_post_transition=False,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
            show=False,
        )
        for trace_output in trace_outputs.values():
            plt.close(trace_output["figure"])
        plotted_views = [view for view in views if view.name in trace_outputs]
        fig, axes = plt.subplots(len(plotted_views) + 1, len(abls), figsize=(5.2 * len(abls), 5.8 * (len(plotted_views) + 1)), squeeze=False, sharey="row")
        duration_colors = {label: plt.get_cmap("tab10")(index % 10) for index, label in enumerate(duration_order)}
        for row, view in enumerate(plotted_views):
            trace_output = trace_outputs[view.name]
            trace_payloads = trace_output["duration_payloads"]
            trace_fits = trace_output["duration_individual_fit_payloads"]
            values = [
                pd.to_numeric(payload["transition_half_difference"], errors="coerce").dropna().to_numpy(dtype=float)
                for payload in trace_payloads.values()
            ]
            values = [value for value in values if len(value)]
            y_max = min(1.0, max(0.15, float(np.nanpercentile(np.abs(np.concatenate(values)), 95)))) if values else 0.25
            for ax, abl in zip(axes[row], abls):
                _draw_half_transition_fit_axis(ax, trace_payloads, trace_fits, abl=abl, colors=duration_colors, post_window=30)
                _style_half_transition_fit_axis(ax, title=f"ABL {abl}", y_max=y_max, fontsize=fs)
            axes[row, 0].set_ylabel(f"{view.name}\nHalf Transition Difference", fontsize=fs)
        _draw_fit_parameter_summary_axes(
            axes[-1], summary_by_duration, x_col="duration_group", x_order=duration_order, x_labels=duration_order,
            view_colors=view_colors, x_label="Stim duration", fontsize=fs,
        )
        duration_handles, duration_labels = axes[0, 0].get_legend_handles_labels()
        genotype_handles, genotype_labels = axes[-1, 0].get_legend_handles_labels()
        fig.legend(duration_handles, duration_labels, loc="lower center", bbox_to_anchor=(0.31, 0.01), ncol=max(1, min(3, len(duration_labels))), fontsize=fs, frameon=False)
        fig.legend(genotype_handles, genotype_labels, loc="lower center", bbox_to_anchor=(0.76, 0.01), ncol=max(1, min(4, len(genotype_labels))), fontsize=fs, frameon=False)
        fig.suptitle("Genotype summary by stim duration", fontsize=fs, y=0.99)
        fig.tight_layout(rect=[0.04, 0.10, 1, 0.97])
        outputs["all_abls_by_duration"] = {"figure": fig, "individual_fits": fits_by_duration, "summary": summary_by_duration}
        if show:
            plt.show()
    return outputs


def plot_biased_blocks(
    *,
    bundle: dict[str, Any],
    views: list[ViewSpec] | None = None,
    layout: str = "block_conditions",
    view_colors: dict[str, str] | None = None,
    view_styles: dict[str, dict] | None = None,
    block_conditions: list[str] | tuple[str, ...] = ("rightward", "leftward", "unbiased"),
    max_animals: int | None = None,
    transition_bin_size: int | None = 1,
    separate_abls: bool = True,
    duration_groups: tuple[tuple[int, ...], ...] = ((8, 16), (32, 64), (120, 0)),
    fit_post_transition: bool = True,
    fit_pre_transition_baseline: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    layout = layout.lower()
    figures = {}

    if layout in {"genotype_blocks", "by_genotype", "views"}:
        figures["genotype_blocks"] = plot_genotype_block_figures(bundle, views=views, show=show)
    elif layout in {"animal_blocks", "by_animal", "animals"}:
        figures["animal_blocks"] = plot_animal_block_figures(bundle, max_animals=max_animals, show=show)
    elif layout in {"block_conditions", "by_block_condition", "conditions"}:
        figures["block_conditions"] = plot_block_condition_figures(
            bundle,
            views=views,
            view_colors=view_colors,
            view_styles=view_styles,
            block_conditions=block_conditions,
            show=show,
        )
    elif layout in {"block_condition_params", "psy_params", "params"}:
        figures["block_condition_params"] = plot_block_condition_psy_params(
            bundle,
            views=views,
            block_conditions=block_conditions,
            show=show,
        )
    elif layout in {"block_condition_summary", "bias_pc_jnd", "block_condition_bias_pc_jnd"}:
        figures["block_condition_summary"] = plot_block_condition_summary_metrics(
            bundle,
            views=views,
            block_conditions=block_conditions,
            show=show,
        )
    elif layout in {"block_bias", "bias"}:
        figures["block_bias"] = plot_block_bias_figures(
            bundle,
            views=views,
            show=show,
        )
    elif layout in {"left_to_right_transition", "transition_left_to_right", "ltr_transition"}:
        figures["left_to_right_transition"] = plot_left_to_right_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
    elif layout in {"left_to_right_transition_genotypes", "ltr_transition_genotypes", "transition_genotype_summary"}:
        figures["left_to_right_transition_genotypes"] = plot_left_to_right_transition_genotype_summary(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            view_colors=view_colors,
            show=show,
        )
    elif layout in {"biased_transition_aligned", "aligned_biased_transition", "biased_transition_test"}:
        figures["biased_transition_aligned"] = plot_aligned_biased_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
    elif layout in {"biased_transition_collapsed", "collapsed_biased_transition", "biased_transition_abs"}:
        figures["biased_transition_collapsed"] = plot_collapsed_biased_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
    elif layout in {"biased_transition_baseline", "matched_transition_baseline", "baseline_subtracted_transition"}:
        figures["biased_transition_baseline"] = plot_baseline_subtracted_biased_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
    elif layout in {
        "biased_transition_accuracy_delta",
        "positive_minus_negative_accuracy_transition",
    }:
        figures["biased_transition_accuracy_delta"] = plot_positive_minus_negative_accuracy_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            separate_abls=separate_abls,
            view_colors=view_colors,
            fit_post_transition=fit_post_transition,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
            show=show,
        )
    elif layout in {"biased_transition_rt", "transition_rt", "half_transition_rt"}:
        figures["biased_transition_rt"] = plot_half_difference_rt_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            separate_abls=separate_abls,
            view_colors=view_colors,
            show=show,
        )
    elif layout in {"biased_transition_accuracy_delta_durations", "duration_grouped_transition_accuracy_delta"}:
        figures["biased_transition_accuracy_delta_durations"] = plot_duration_grouped_half_transition_accuracy_figures(
            bundle,
            views=views,
            duration_groups=duration_groups,
            transition_bin_size=transition_bin_size,
            fit_post_transition=fit_post_transition,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
            show=show,
        )
    elif layout in {"biased_transition_fit_parameter_summary", "half_transition_fit_parameters"}:
        figures["biased_transition_fit_parameter_summary"] = plot_half_transition_fit_parameter_summaries(
            bundle,
            views=views,
            duration_groups=duration_groups,
            transition_bin_size=transition_bin_size,
            view_colors=view_colors,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
            show=show,
        )
    elif layout == "all":
        figures["genotype_blocks"] = plot_genotype_block_figures(bundle, views=views, show=show)
        figures["animal_blocks"] = plot_animal_block_figures(bundle, max_animals=max_animals, show=show)
        figures["block_conditions"] = plot_block_condition_figures(
            bundle,
            views=views,
            view_colors=view_colors,
            view_styles=view_styles,
            block_conditions=block_conditions,
            show=show,
        )
        figures["block_condition_params"] = plot_block_condition_psy_params(
            bundle,
            views=views,
            block_conditions=block_conditions,
            show=show,
        )
        figures["block_condition_summary"] = plot_block_condition_summary_metrics(
            bundle,
            views=views,
            block_conditions=block_conditions,
            show=show,
        )
        figures["block_bias"] = plot_block_bias_figures(
            bundle,
            views=views,
            show=show,
        )
        figures["left_to_right_transition"] = plot_left_to_right_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
        figures["left_to_right_transition_genotypes"] = plot_left_to_right_transition_genotype_summary(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            view_colors=view_colors,
            show=show,
        )
        figures["biased_transition_aligned"] = plot_aligned_biased_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
        figures["biased_transition_collapsed"] = plot_collapsed_biased_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
        figures["biased_transition_baseline"] = plot_baseline_subtracted_biased_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            show=show,
        )
        figures["biased_transition_accuracy_delta"] = plot_positive_minus_negative_accuracy_transition_figures(
            bundle,
            views=views,
            transition_bin_size=transition_bin_size,
            separate_abls=separate_abls,
            view_colors=view_colors,
            fit_post_transition=fit_post_transition,
            fit_pre_transition_baseline=fit_pre_transition_baseline,
            show=show,
        )
    else:
        raise ValueError(
            "layout must be one of: genotype_blocks, block_conditions, block_condition_params, block_condition_summary, block_bias, left_to_right_transition, left_to_right_transition_genotypes, biased_transition_aligned, biased_transition_collapsed, biased_transition_baseline, biased_transition_accuracy_delta, biased_transition_rt, biased_transition_accuracy_delta_durations, biased_transition_fit_parameter_summary, animal_blocks, all."
        )

    return {"figures": figures}


def save_biased_block_figures(
    figures: dict[str, Any],
    out_dir,
    prefix: str = "biased_blocks",
    formats: tuple[str, ...] = ("png", "pdf"),
) -> None:
    """Save every figure in the nested layout output in one or more formats."""
    out_dir = pd.io.common.stringify_path(out_dir)
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    formats = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    if not formats:
        raise ValueError("Provide at least one output format, for example ('png', 'eps').")

    def _save(obj: dict[str, Any], folder: Path, name_prefix: str) -> None:
        for name, value in obj.items():
            if isinstance(value, dict) and "figure" in value:
                for fmt in formats:
                    value["figure"].savefig(
                        folder / f"{name_prefix}_{_safe_name(name)}.{fmt}",
                        dpi=250,
                        bbox_inches="tight",
                    )
            elif isinstance(value, dict):
                subdir = folder / _safe_name(name)
                subdir.mkdir(parents=True, exist_ok=True)
                _save(value, subdir, name_prefix)

    _save(figures, out_path, prefix)
