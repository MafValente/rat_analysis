# StimDur/prepare.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from StimDur.config import FilterConfig, StimDurComparisonConfig, ViewSpec, StimDurSpec

import Psychometric
import Helpers.DataHelpers as DataHelpers


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def apply_filters(df: pd.DataFrame, fcfg: FilterConfig) -> pd.DataFrame:
    df = df.copy()
    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")

    if fcfg.drop_repeat_trials and "trial_is_repeat" in df.columns:
        df = df[df["trial_is_repeat"] == False].copy()

    if "training_level" in df.columns:
        df = df[df["training_level"] >= fcfg.training_min].copy()

    if "session" in df.columns:
        df = df[df["session"] >= fcfg.session_min].copy()

    if fcfg.session_type_values is not None:
        if "session_type" not in df.columns:
            raise KeyError("Expected 'session_type' column for session_type filter.")
        df = df[df["session_type"].isin(list(fcfg.session_type_values))].copy()

    # IMPORTANT: keep ABL dtype consistent so fit dict keys match plotting ABLs
    if "ABL" in df.columns:
        df["ABL"] = df["ABL"].astype(int)

    return df


def prep_rt(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_s = df_in[df_in["success"] == 1].copy()
    df_s["abs_ILD"] = df_s["ILD"].abs()

    per_subj = (
        df_s.groupby(["animal", "ABL", "abs_ILD"])["timed_rt"]
        .mean()
        .reset_index()
        .rename(columns={"abs_ILD": "ILD", "timed_rt": "mean_rt"})
    )

    grp = (
        per_subj.groupby(["ABL", "ILD"])["mean_rt"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    return per_subj, grp


def prep_mt(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_s = df_in[df_in["success"] == 1].copy()

    per_subj = (
        df_s.groupby(["animal", "ABL", "ILD"])["timed_mt"]
        .mean()
        .reset_index()
        .rename(columns={"timed_mt": "mean_mt"})
    )

    grp = (
        per_subj.groupby(["ABL", "ILD"])["mean_mt"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    return per_subj, grp


def prep_psy(
    df_in: pd.DataFrame,
    do_individual_fits: bool,
    skip_abl_for_jnd: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, pd.DataFrame]:
    """
    Returns:
      points_df, psy_group_df, indiv_curves, mean_fits, jnd_indiv_df

    NOTE: JND is computed from the same per-subject psychometric results
    to avoid recomputing psychometrics later.
    """
    all_pts: List[dict] = []
    indiv_curves: Dict[tuple, dict] = {}
    mean_fits: Dict[int, dict | None] = {}
    jnd_rows: List[dict] = []

    # loop subjects once
    for subject, df_subj in df_in.groupby("animal", sort=False):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")

        # ---- JND from the same 'results' (no extra psychometric computation) ----
        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_abl_for_jnd)
        if jnd_df is not None and not jnd_df.empty:
            for _, r in jnd_df.iterrows():
                jnd_rows.append(
                    {"subject": subject, "ABL": int(r["ABL"]), "JND": float(r["JND"])}
                )

        # ---- points + optional individual fits ----
        for abl, res in results.items():
            abl_i = int(abl)
            ILDs = np.asarray(res["ILDs"])
            pleft = np.asarray(res["PropLeft"], dtype=float)

            for ild, val in zip(ILDs, pleft):
                all_pts.append(
                    {"subject": subject, "ABL": abl_i, "ILD": float(ild), "PropLeft": float(val)}
                )

            if do_individual_fits and len(ILDs) >= 4 and np.isfinite(pleft).all():
                n_trials = np.full_like(ILDs, 50)
                try:
                    _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="my_psycho", n_trials=n_trials, show_plot=False
                    )
                    indiv_curves[(subject, abl_i)] = {"xx": xx, "yy": yy}
                except Exception:
                    # keep silent like your current style
                    pass

    points = pd.DataFrame(all_pts, columns=["subject", "ABL", "ILD", "PropLeft"])
    jnd_indiv = pd.DataFrame(jnd_rows, columns=["subject", "ABL", "JND"])

    if points.empty:
        psy_group = pd.DataFrame(columns=["ABL", "ILD", "mean", "sem", "n"])
        return points, psy_group, indiv_curves, {}, jnd_indiv

    psy_group = (
        points.groupby(["ABL", "ILD"])["PropLeft"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )

    # ---- mean fit per ABL ----
    for abl in sorted(psy_group["ABL"].unique()):
        sub = psy_group[psy_group["ABL"] == abl]
        ILDs = sub["ILD"].values
        y = sub["mean"].values

        # require enough points for a stable fit
        if len(ILDs) < 4 or not np.isfinite(y).all():
            mean_fits[int(abl)] = None
            continue

        n_trials = np.full_like(ILDs, 50)
        try:
            _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                ILDs, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
            mean_fits[int(abl)] = {"xx": xx, "yy": yy}
        except Exception:
            mean_fits[int(abl)] = None

    return points, psy_group, indiv_curves, mean_fits, jnd_indiv


def build_prepared_by_view_and_stimdur(
    df: pd.DataFrame,
    views: List[ViewSpec],
    stimdur_specs: List[StimDurSpec],
    cfg: StimDurComparisonConfig,
) -> Dict[str, Dict[str, dict]]:
    """
    prepared[view_name][stimdur_name] -> tables dict
    """
    prepared: Dict[str, Dict[str, dict]] = {}

    for v in views:
        df_v = v.selector(df.copy())
        prepared[v.name] = {}

        for s in stimdur_specs:
            df_vs = s.selector(df_v.copy())

            rt_per, rt_grp = prep_rt(df_vs)
            mt_per, mt_grp = prep_mt(df_vs)

            psy_pts, psy_grp, psy_indiv, psy_mean, jnd_indiv = prep_psy(
                df_vs,
                do_individual_fits=(cfg.error_mode == "individuals"),
                skip_abl_for_jnd=50,
            )

            prepared[v.name][s.name] = dict(
                rt_per_subj=rt_per, rt_group=rt_grp,
                mt_per_subj=mt_per, mt_group=mt_grp,
                psy_points=psy_pts, psy_group=psy_grp,
                psy_indiv_curves=psy_indiv, psy_mean_fits=psy_mean,
                jnd_indiv=jnd_indiv,          # <<<< store here (NO recomputation later)
                df_view=df_vs,                # keep if you still want it for debugging
            )

    return prepared


def compute_group_jnd_by_view_and_stimdur(
    prepared: Dict[str, Dict[str, dict]],
    skip_abl: int = 50,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    group_jnd[view_name][stimdur_name] -> df[ABL, mean, sem, n]

    Uses tables["jnd_indiv"] (computed during prep_psy) and DOES NOT recompute psychometrics.
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}

    for vname, by_sd in prepared.items():
        out[vname] = {}

        for sd_name, tables in by_sd.items():
            dfj = tables.get("jnd_indiv", None)

            if dfj is None or dfj.empty:
                out[vname][sd_name] = pd.DataFrame(columns=["ABL", "mean", "sem", "n"])
                continue

            # safety: if skip_abl passed different from prep_psy, apply it here too
            if skip_abl is not None:
                dfj = dfj[dfj["ABL"] != int(skip_abl)].copy()

            out[vname][sd_name] = (
                dfj.groupby("ABL")["JND"]
                .agg(mean="mean", sem=sem, n="count")
                .reset_index()
            )

    return out
