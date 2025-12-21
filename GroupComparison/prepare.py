# analysis/groupcomparison/prepare.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from .config import ViewSpec, FilterConfig, GroupComparisonConfig

import os
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis")  # <- the folder that contains Helpers/ and GroupComparison/
import Psychometric
import Helpers.DataHelpers as DataHelpers


def sem(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else x.std(ddof=1) / np.sqrt(len(x))


def apply_filters(df: pd.DataFrame, fcfg: FilterConfig) -> pd.DataFrame:
    df = df.copy()
    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")

    if fcfg.drop_repeat_trials and "trial_is_repeat" in df.columns:
        df = df[df["trial_is_repeat"] == False].copy()

    if "training_level" in df.columns:
        df = df[df["training_level"] >= fcfg.training_min]

    if "session" in df.columns:
        df = df[df["session"] >= fcfg.session_min]

    # ----------restrict to session_type OR stim_dur ----------
    if fcfg.session_type_values is not None or fcfg.stim_dur_values is not None:
        masks = []

        if fcfg.session_type_values is not None:
            if "session_type" not in df.columns:
                raise KeyError("FilterConfig asked for session_type_values but df has no 'session_type' column.")
            masks.append(df["session_type"].isin(list(fcfg.session_type_values)))

        if fcfg.stim_dur_values is not None:
            # adjust this column name if yours is different (e.g. 'stimdur' or 'stim_dur_ms')
            if "stim_dur" not in df.columns:
                raise KeyError("FilterConfig asked for stim_dur_values but df has no 'stim_dur' column.")
            masks.append(df["stim_dur"].isin(list(fcfg.stim_dur_values)))

        if len(masks) == 1:
            mask = masks[0]
        else:
            if fcfg.sessiontype_or_stimdur.lower() == "or":
                mask = masks[0] | masks[1]
            elif fcfg.sessiontype_or_stimdur.lower() == "and":
                mask = masks[0] & masks[1]
            else:
                raise ValueError("sessiontype_or_stimdur must be 'or' or 'and'.")

        df = df[mask].copy()
    # ---------------------------------------------------------------

    return df


def prep_rt(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_s = df_in[df_in["success"] == 1].copy()
    df_s["abs_ILD"] = df_s["ILD"].abs()

    per_subj = (
        df_s.groupby(["animal", "ABL", "abs_ILD"])["timed_rt"]
        .agg(mean_rt="mean")
        .reset_index()
        .rename(columns={"abs_ILD": "ILD"})
    )

    grouped = (
        per_subj.groupby(["ABL", "ILD"])
        .agg(mean=("mean_rt", "mean"), sem=("mean_rt", sem), n=("mean_rt", "count"))
        .reset_index()
    )
    return per_subj, grouped


def prep_mt(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_subj = (
        df_in[df_in["success"] == 1]
        .groupby(["animal", "ABL", "ILD"])["timed_mt"]
        .agg(mean_mt="mean")
        .reset_index()
    )

    grouped = (
        per_subj.groupby(["ABL", "ILD"])
        .agg(mean=("mean_mt", "mean"), sem=("mean_mt", sem), n=("mean_mt", "count"))
        .reset_index()
    )
    return per_subj, grouped


def prep_psy(df_in: pd.DataFrame, do_individual_fits: bool) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    all_pts = []
    per_subject_curves = {}  # (subject, abl) -> dict(xx, yy)

    for subject, df_subj in df_in.groupby("animal"):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
        for abl, res in results.items():
            ILDs = np.asarray(res["ILDs"])
            pleft = np.asarray(res["PropLeft"], dtype=float)

            for ild, val in zip(ILDs, pleft):
                all_pts.append({"subject": subject, "ABL": abl, "ILD": ild, "PropLeft": val})

            if do_individual_fits and len(ILDs) >= 4 and np.isfinite(pleft).all():
                n_trials = np.full_like(ILDs, 50)
                try:
                    _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="my_psycho", n_trials=n_trials, show_plot=False
                    )
                    per_subject_curves[(subject, abl)] = dict(xx=xx, yy=yy)
                except Exception:
                    pass

    points = pd.DataFrame(all_pts)

    agg = (
        points.groupby(["ABL", "ILD"])
        .agg(mean=("PropLeft", "mean"), sem=("PropLeft", sem), n=("PropLeft", "count"))
        .reset_index()
    )

    mean_fits = {}
    for abl in sorted(agg["ABL"].unique()):
        sub = agg[agg["ABL"] == abl]
        ILDs, y = sub["ILD"].values, sub["mean"].values
        n_trials = np.full_like(ILDs, 50)
        try:
            _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                ILDs, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
            mean_fits[abl] = dict(xx=xx, yy=yy)
        except Exception:
            mean_fits[abl] = None

    return points, agg, per_subject_curves, mean_fits


def build_prepared(df: pd.DataFrame, views: List[ViewSpec], cfg: GroupComparisonConfig) -> Dict[str, dict]:
    prepared: Dict[str, dict] = {}
    for v in views:
        df_v = v.selector(df.copy())
        rt_per_subj, rt_group = prep_rt(df_v)
        mt_per_subj, mt_group = prep_mt(df_v)
        psy_points, psy_group, psy_indiv, psy_mean = prep_psy(
            df_v, do_individual_fits=(cfg.error_mode == "individuals")
        )
        prepared[v.name] = dict(
            rt_per_subj=rt_per_subj, rt_group=rt_group,
            mt_per_subj=mt_per_subj, mt_group=mt_group,
            psy_points=psy_points, psy_group=psy_group,
            psy_indiv_curves=psy_indiv, psy_mean_fits=psy_mean,
            df_view=df_v,  # for JND extraction
        )
    return prepared


def compute_jnd_individuals_by_view(prepared: Dict[str, dict], skip_abl: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Returns {view_name: all_jnds_df} with columns [subject, ABL, JND]
    """
    out: Dict[str, pd.DataFrame] = {}
    for view_name, tables in prepared.items():
        df_v = tables["df_view"]
        rows = []
        for subject, df_subj in df_v.groupby("animal"):
            results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
            jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_abl)
            if jnd_df is None or jnd_df.empty:
                continue
            for _, r in jnd_df.iterrows():
                rows.append({"subject": subject, "ABL": int(r["ABL"]), "JND": float(r["JND"])})
        out[view_name] = pd.DataFrame(rows, columns=["subject", "ABL", "JND"])
    return out


def compute_group_jnd_by_view(jnd_indiv_by_view: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Returns {view_name: group_jnd_df} with columns [ABL, mean, sem, n]
    """
    out: Dict[str, pd.DataFrame] = {}
    for view_name, dfj in jnd_indiv_by_view.items():
        if dfj is None or dfj.empty:
            out[view_name] = pd.DataFrame(columns=["ABL", "mean", "sem", "n"])
            continue
        g = (
            dfj.groupby("ABL")["JND"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
        out[view_name] = g
    return out
