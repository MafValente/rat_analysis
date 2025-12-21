# analysis/pipeline.py
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import pandas as pd
import Helpers.DataHelpers as DataHelpers

from datasets import DatasetSpec, load_cohort_df
from session_profiles import SessionProfile, NORMAL
from metrics import prep_rt, prep_mt, prep_psy

@dataclass(frozen=True)
class ViewSpec:
    name: str
    selector: Callable[[pd.DataFrame], pd.DataFrame]
    color: Optional[str] = None

@dataclass(frozen=True)
class AnalysisConfig:
    training_min: int = 16
    session_min: int = 13
    error_mode: str = "individuals"   # "sem" or "individuals"
    skip_psy_fits: frozenset = frozenset({50})
    min_ilds_for_psy_fit: int = 8


def load_and_filter(spec: DatasetSpec, cfg: AnalysisConfig) -> tuple[pd.DataFrame, str]:
    df, data_dir = load_cohort_df(spec)

    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
    # Only apply repeat filter if the column exists
    if "trial_is_repeat" in df.columns:
        df = df[df["trial_is_repeat"] == False].copy()
    elif "repeat_trial" in df.columns:
        df = df[df["repeat_trial"] == 0].copy()   # or == False depending on dtype
    # else: do nothing

    if "training_level" in df.columns:
        df = df[df["training_level"] >= cfg.training_min]
    if "session" in df.columns:
        df = df[df["session"] >= cfg.session_min]

    return df, data_dir

def build_prepared(
    spec: DatasetSpec,
    views: List[ViewSpec],
    cfg: AnalysisConfig,
    profile: SessionProfile = NORMAL,
) -> Dict:
    df, data_dir = load_and_filter(spec, cfg)

    prepared = {}
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
            df_view=df_v,  # keep around for JND etc.
        )

    abl_rows = sorted(set().union(*[
        set(p["rt_group"]["ABL"].unique()) for p in prepared.values()
    ]))

    return dict(
        spec=spec,
        data_dir=data_dir,
        cfg=cfg,
        profile=profile,
        views=views,
        prepared=prepared,
        abl_rows=abl_rows,
    )


def build_prepared_from_df(df, views, cfg, *, assume_prepared=False):

    """
    Same output structure as build_prepared(spec,...), but uses an already-loaded df.
    Useful for: single animal, session_type subsets, cross-session slices, etc.
    """

    if not assume_prepared:
        df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
        # Only apply repeat filter if the column exists (keeps compatibility with legacy cohorts)
        if "trial_is_repeat" in df.columns:
            df = df[df["trial_is_repeat"] == False].copy()
        elif "repeat_trial" in df.columns:
            df = df[df["repeat_trial"] == 0].copy()
        if "training_level" in df.columns:
            df = df[df["training_level"] >= cfg.training_min]
        if "session" in df.columns:
            df = df[df["session"] >= cfg.session_min]


    prepared = {}
    for v in views:
        df_v = v.selector(df.copy())
        rt_per_subj, rt_group = prep_rt(df_v)
        mt_per_subj, mt_group = prep_mt(df_v)
        psy_points, psy_group, psy_indiv_curves, psy_mean_fits = prep_psy(
            df_v,
            do_individual_fits=(cfg.error_mode == "individuals"),
        )

        prepared[v.name] = dict(
            df_view=df_v,
            rt_per_subj=rt_per_subj,
            rt_group=rt_group,
            mt_per_subj=mt_per_subj,
            mt_group=mt_group,
            psy_points=psy_points,
            psy_group=psy_group,
            psy_indiv_curves=psy_indiv_curves,
            psy_mean_fits=psy_mean_fits,
        )

    abl_rows = sorted(set().union(*[set(p["rt_group"]["ABL"].unique()) for p in prepared.values()]))

    return dict(prepared=prepared, views=views, abl_rows=abl_rows, cfg=cfg)