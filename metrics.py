import numpy as np
import pandas as pd

import Psychometric
import Helpers.DataHelpers as DataHelpers


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else x.std(ddof=1) / np.sqrt(len(x))


def boot_sem_mean(x, n_boot=2000, seed=0):
    """Bootstrap SEM of the mean (std of bootstrap means)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return means.std(ddof=1)


def _ensure_ild_for_chrono(df: pd.DataFrame, ild_col: str = "ILD") -> pd.DataFrame:
    """Chronometric plots in this codebase expect a single column named 'ILD'.

    Historically that 'ILD' was *absolute* ILD (0..18, with 18 relabeled as 50 for ABL50).
    Some later edits introduced an 'abs_ILD' column and started grouping on it, which broke
    plotting code that expects 'ILD'. This helper restores the old contract.
    """
    out = df.copy()
    if ild_col not in out.columns:
        raise KeyError(f"Missing required column '{ild_col}' for chronometric metrics.")
    out[ild_col] = pd.to_numeric(out[ild_col], errors="coerce").abs()
    return out


def prep_rt(
    df_v: pd.DataFrame,
    RT_COL: str = "timed_rt",
    *,
    ild_col: str = "ILD",
    groupby_stim_dur: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
):
    """Prepare RT tables used by the GroupComparison plots.

    Default behavior matches the older plotting code:
      - RT is computed as a function of (ABL, |ILD|), i.e. ILD is absolute.
      - If 'stim_dur' exists, it is averaged over unless groupby_stim_dur=True.

    Returns:
      rt_per_subj: per-animal mean RT per bin (+ boot_sem for single-animal cases)
      rt_group:    mean across animals + SEM across animals
    """
    df = df_v.copy()
    df = _ensure_ild_for_chrono(df, ild_col=ild_col)

    # numeric coercion
    for c in ["ABL", ild_col, RT_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need = ["animal", "ABL", ild_col, RT_COL]
    if groupby_stim_dur and "stim_dur" in df.columns:
        df["stim_dur"] = pd.to_numeric(df["stim_dur"], errors="coerce")
        need.append("stim_dur")

    df = df.dropna(subset=need)

    keys = ["ABL", ild_col] + (["stim_dur"] if (groupby_stim_dur and "stim_dur" in df.columns) else [])
    is_single = df["animal"].nunique() == 1

    gb_trial = df.groupby(["animal"] + keys)[RT_COL]
    rt_per_subj = gb_trial.mean().reset_index(name="mean_rt")

    if is_single:
        rt_per_subj["boot_sem"] = gb_trial.apply(
            lambda s: boot_sem_mean(s.values, n_boot=n_boot, seed=seed)
        ).values
    else:
        rt_per_subj["boot_sem"] = np.nan  # keep column for API stability

    rt_group = (
        rt_per_subj.groupby(keys)
        .agg(
            mean=("mean_rt", "mean"),
            sem=("mean_rt", sem),
            n_animals=("animal", "nunique"),
        )
        .reset_index()
    )

    return rt_per_subj, rt_group


def prep_mt(
    df_v: pd.DataFrame,
    MT_COL: str = "timed_mt",
    *,
    ild_col: str = "ILD",
    groupby_stim_dur: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
):
    """Same as prep_rt, but for MT."""
    df = df_v.copy()
    df = _ensure_ild_for_chrono(df, ild_col=ild_col)

    for c in ["ABL", ild_col, MT_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need = ["animal", "ABL", ild_col, MT_COL]
    if groupby_stim_dur and "stim_dur" in df.columns:
        df["stim_dur"] = pd.to_numeric(df["stim_dur"], errors="coerce")
        need.append("stim_dur")

    df = df.dropna(subset=need)

    keys = ["ABL", ild_col] + (["stim_dur"] if (groupby_stim_dur and "stim_dur" in df.columns) else [])
    is_single = df["animal"].nunique() == 1

    gb_trial = df.groupby(["animal"] + keys)[MT_COL]
    mt_per_subj = gb_trial.mean().reset_index(name="mean_mt")

    if is_single:
        mt_per_subj["boot_sem"] = gb_trial.apply(
            lambda s: boot_sem_mean(s.values, n_boot=n_boot, seed=seed)
        ).values
    else:
        mt_per_subj["boot_sem"] = np.nan

    mt_group = (
        mt_per_subj.groupby(keys)
        .agg(
            mean=("mean_mt", "mean"),
            sem=("mean_mt", sem),
            n_animals=("animal", "nunique"),
        )
        .reset_index()
    )

    return mt_per_subj, mt_group


def prep_psy(df_in: pd.DataFrame, do_individual_fits: bool = False):
    """Psychometric points + fits, as before."""
    all_pts = []
    per_subject_curves = {}

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
                    pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="my_psycho", n_trials=n_trials, show_plot=False
                    )
                    per_subject_curves[(subject, abl)] = dict(xx=xx, yy=yy)
                except Exception:
                    pass

    points = pd.DataFrame(all_pts)
    agg = (
        points.groupby(["ABL", "ILD"])
        .agg(mean=("PropLeft", "mean"), sem=("PropLeft", sem))
        .reset_index()
    )

    mean_fits = {}
    for abl in sorted(agg["ABL"].unique()):
        sub = agg[agg["ABL"] == abl]
        ILDs, y = sub["ILD"].values, sub["mean"].values
        n_trials = np.full_like(ILDs, 50)
        try:
            pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                ILDs, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
            mean_fits[abl] = dict(xx=xx, yy=yy)
        except Exception:
            mean_fits[abl] = None

    return points, agg, per_subject_curves, mean_fits


def compute_group_jnd(df_view: pd.DataFrame, skip_ABL=50):
    """Per-animal JNDs + group mean/SEM."""
    all_jnds = []
    for subject, df_subj in df_view.groupby("animal"):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_ABL)
        if not jnd_df.empty:
            jnd_df = jnd_df.copy()
            jnd_df["subject"] = subject
            all_jnds.append(jnd_df)

    if not all_jnds:
        return pd.DataFrame(columns=["ABL", "mean", "sem", "n"]), pd.DataFrame()

    all_jnds_df = pd.concat(all_jnds, ignore_index=True)
    group_jnd = (
        all_jnds_df.groupby("ABL")["JND"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    return group_jnd, all_jnds_df
