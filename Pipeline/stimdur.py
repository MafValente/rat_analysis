from __future__ import annotations

import contextlib
import io
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from analysis import psychometric as Psychometric
from analysis.datasets import dataset_key, load_dataset_selections
from Pipeline.biased_blocks import add_biased_block_condition
from StimDur.config import (
    FilterConfig,
    PlotStyle,
    StimDurComparisonConfig,
    ViewSpec,
    make_stimdur_specs,
)
from StimDur.layouts import (
    plot_absild_perf_3x5_all_genotypes,
    plot_absild_perf_across_stimdur_1x3_for_view,
    plot_biased_metric_grid_for_stimdur,
    plot_block_conditions_4x3_for_stimdur_and_view,
    plot_psychometric_stimdur_grid_all_conditions,
    plot_psychometric_stimdur_grid_for_condition,
    plot_genotypes_4x3_for_stimdur,
    plot_kreg_4x3_by_abl_for_view,
    plot_stimdur_4x3_for_view,
)
from StimDur.prepare import (
    apply_filters,
    build_prepared_by_view_and_stimdur,
    compute_group_jnd_by_view_and_stimdur,
    sem,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_DIR = ROOT / "DataFiles"
STIMDUR_COL = "short_duration"
DEFAULT_STIM_DURS = [8, 15, 16, 32, 60, 64, 120, 0]
BLOCK_CONDITION_ORDER = ("unbiased", "rightward", "leftward")
BLOCK_CONDITION_COLORS = {
    "unbiased": "#4D4D4D",
    "rightward": "#1F77B4",
    "leftward": "#D62728",
}
BLOCK_CONDITION_PRETTY = {
    "unbiased": "Unbiased",
    "rightward": "Right Bias",
    "leftward": "Left Bias",
}
ABL_COLORS = {
    20: "#0072B2",
    40: "#E69F00",
    60: "#009E73",
}
STIMDUR_PRETTY = {
    "8": "SD = 8 ms",
    "15": "SD = 15 ms",
    "16": "SD = 16 ms",
    "32": "SD = 32 ms",
    "60": "SD = 60 ms",
    "64": "SD = 64 ms",
    "120": "SD = 120 ms",
    "0": "SD = RT",
}
STIMDUR_PALETTE = [
    "#B2A706", "#0072B2", "#56B4E9", "#E69F00",
    "#009E73", "#D55E00", "#CC79A7", "#4D4D4D",
]
GENOTYPE_COLORS = {
    "wt": "#4D4D4D",
    "het": "#7A8F28",
    "hom": "#C24A7A",
}


def dataset_sort_key(name: str):
    line, _, cohort = str(name).partition(":")
    match = pd.Series([cohort]).str.extract(r"cohort(\d+)", expand=False).iloc[0]
    cohort_num = int(match) if pd.notna(match) else 10**9
    return (line.lower(), cohort_num, cohort.lower())


def normalize_dataset_selections(
    *,
    lines: list[str] | tuple[str, ...],
    cohorts: list[str] | tuple[str, ...],
    dataset_selections: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if dataset_selections is not None:
        return [(str(line), str(cohort)) for line, cohort in dataset_selections]
    return [(str(line), str(cohort)) for line, cohort in product(lines, cohorts)]


def load_stimdur_data(
    *,
    lines: list[str] | tuple[str, ...] = ("CNTNAP2",),
    cohorts: list[str] | tuple[str, ...] = ("cohort3",),
    dataset_selections: list[tuple[str, str]] | None = None,
    base_dir: str | Path = BASE_DATA_DIR,
    cohort_file: str | None = None,
    require_meta: bool = False,
    animal_selection: str | None = None,
) -> dict[str, Any]:
    selections = normalize_dataset_selections(
        lines=lines,
        cohorts=cohorts,
        dataset_selections=dataset_selections,
    )
    df_all, meta_all, dataset_info = load_dataset_selections(
        selections=selections,
        base_dir=str(base_dir),
        cohort_file=cohort_file,
        require_meta=require_meta,
    )

    df_all = df_all.copy()
    for col in ("animal", "line", "cohort", "genotype", "dataset_key"):
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(str).str.strip()

    if "dataset_key" not in df_all.columns:
        df_all["dataset_key"] = df_all.apply(lambda row: dataset_key(row["line"], row["cohort"]), axis=1)

    if not meta_all.empty:
        usable_dataset_names = sorted(meta_all["dataset_key"].dropna().astype(str).unique(), key=dataset_sort_key)
        df_plot = df_all[df_all["dataset_key"].isin(usable_dataset_names)].copy()
    else:
        usable_dataset_names = sorted(df_all["dataset_key"].dropna().astype(str).unique(), key=dataset_sort_key)
        df_plot = df_all.copy()

    if animal_selection is not None:
        animal_id = str(animal_selection).strip()
        df_plot = df_plot[df_plot["animal"].astype(str).str.strip() == animal_id].copy()
        if df_plot.empty:
            raise ValueError(f"No rows found for animal_selection={animal_id!r} after dataset filters.")

    return {
        "df_all": df_all,
        "df_plot": df_plot,
        "meta_all": meta_all,
        "dataset_info": dataset_info,
        "selections": selections,
        "usable_dataset_names": usable_dataset_names,
    }


def _as_set(values):
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def make_selector(
    *,
    genotypes=None,
    lines=None,
    cohorts=None,
    dataset_keys=None,
    animals=None,
    experimenters=None,
    experimentor=None,
):
    genotype_set = _as_set(genotypes)
    line_set = _as_set(lines)
    cohort_set = _as_set(cohorts)
    dataset_key_set = _as_set(dataset_keys)
    animal_set = _as_set(animals)
    experimenter_set = _as_set(experimenters if experimenters is not None else experimentor)

    def _selector(df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if genotype_set is not None:
            mask &= df["genotype"].astype(str).str.strip().isin(genotype_set)
        if line_set is not None:
            mask &= df["line"].astype(str).str.strip().isin(line_set)
        if cohort_set is not None:
            mask &= df["cohort"].astype(str).str.strip().isin(cohort_set)
        if dataset_key_set is not None:
            mask &= df["dataset_key"].astype(str).str.strip().isin(dataset_key_set)
        if animal_set is not None:
            mask &= df["animal"].astype(str).str.strip().isin(animal_set)
        if experimenter_set is not None:
            if "experimenter" not in df.columns:
                return df.iloc[0:0].copy()
            mask &= df["experimenter"].astype(str).str.strip().isin(experimenter_set)
        return df[mask].copy()

    return _selector


def build_custom_views(custom_specs: list[dict[str, Any]]) -> list[ViewSpec]:
    views: list[ViewSpec] = []
    for spec in custom_specs:
        name = spec["name"]
        views.append(
            ViewSpec(
                str(name),
                make_selector(
                    genotypes=spec.get("genotype", spec.get("genotypes")),
                    lines=spec.get("line", spec.get("lines")),
                    cohorts=spec.get("cohort", spec.get("cohorts")),
                    dataset_keys=spec.get("dataset_key", spec.get("dataset_keys")),
                    animals=spec.get("animal", spec.get("animals")),
                    experimenters=spec.get("experimenters", spec.get("experimentor", spec.get("experimenter"))),
                ),
            )
        )
    return views


def build_stimdur_views(
    df: pd.DataFrame,
    *,
    comparison: str = "genotypes",
    split_by: str = "none",
    genotypes: list[str] | tuple[str, ...] | None = ("wt", "het", "hom"),
    lines: list[str] | tuple[str, ...] | None = None,
    cohorts: list[str] | tuple[str, ...] | None = None,
    custom_specs: list[dict[str, Any]] | None = None,
) -> list[ViewSpec]:
    comparison = "experimentor" if comparison == "experimenter" else comparison

    if comparison == "custom":
        if not custom_specs:
            raise ValueError("comparison='custom' requires custom_specs.")
        views = build_custom_views(custom_specs)
    else:
        df_meta = df.dropna(subset=["dataset_key"]).copy()
        available_genotypes = [
            g for g in ("wt", "het", "hom")
            if g in set(df_meta.get("genotype", pd.Series(dtype=str)).astype(str))
        ]
        genotype_list = list(genotypes) if genotypes is not None else available_genotypes
        line_list = list(lines) if lines is not None else sorted(df_meta["line"].dropna().astype(str).unique())
        cohort_list = list(cohorts) if cohorts is not None else sorted(df_meta["cohort"].dropna().astype(str).unique())
        dataset_names = sorted(df_meta["dataset_key"].dropna().astype(str).unique(), key=dataset_sort_key)
        experimenter_list = (
            sorted(df_meta["experimenter"].dropna().astype(str).str.strip().unique())
            if "experimenter" in df_meta.columns
            else []
        )
        views = []

        if comparison == "genotypes":
            for genotype in genotype_list:
                if split_by == "none":
                    views.append(ViewSpec(str(genotype), make_selector(genotypes=genotype, lines=lines, cohorts=cohorts)))
                elif split_by == "dataset":
                    for dataset_name in dataset_names:
                        key_line, _, key_cohort = dataset_name.partition(":")
                        if lines is not None and key_line not in set(map(str, lines)):
                            continue
                        if cohorts is not None and key_cohort not in set(map(str, cohorts)):
                            continue
                        views.append(ViewSpec(f"{genotype} {dataset_name}", make_selector(genotypes=genotype, dataset_keys=dataset_name)))
                elif split_by == "line":
                    for line in line_list:
                        views.append(ViewSpec(f"{genotype} {line}", make_selector(genotypes=genotype, lines=line, cohorts=cohorts)))
                elif split_by == "cohort":
                    for cohort in cohort_list:
                        views.append(ViewSpec(f"{genotype} {cohort}", make_selector(genotypes=genotype, lines=lines, cohorts=cohort)))
                else:
                    raise ValueError("split_by must be one of: none, dataset, line, cohort.")
        elif comparison == "datasets":
            for dataset_name in dataset_names:
                views.append(ViewSpec(dataset_name, make_selector(genotypes=genotypes, dataset_keys=dataset_name)))
        elif comparison == "lines":
            for line in line_list:
                label = line if genotypes is None else f"{line} {'/'.join(map(str, genotype_list))}"
                views.append(ViewSpec(label, make_selector(genotypes=genotypes, lines=line, cohorts=cohorts)))
        elif comparison == "cohorts":
            for cohort in cohort_list:
                label = cohort if genotypes is None else f"{cohort} {'/'.join(map(str, genotype_list))}"
                views.append(ViewSpec(label, make_selector(genotypes=genotypes, lines=lines, cohorts=cohort)))
        elif comparison == "experimentor":
            for experimenter in experimenter_list:
                label = experimenter if genotypes is None else f"{experimenter} {'/'.join(map(str, genotype_list))}"
                views.append(
                    ViewSpec(
                        label,
                        make_selector(
                            genotypes=genotypes,
                            lines=lines,
                            cohorts=cohorts,
                            experimenters=experimenter,
                        ),
                    )
                )
        elif comparison == "animals":
            animals = sorted(df["animal"].dropna().astype(str).str.strip().unique())
            views.extend(ViewSpec(animal, make_selector(animals=animal)) for animal in animals)
        else:
            raise ValueError("comparison must be one of: genotypes, datasets, lines, cohorts, experimentor/experimenter, animals, custom.")

    nonempty_views = [view for view in views if not view.selector(df).empty]
    if not nonempty_views:
        raise ValueError("No non-empty StimDur views were built. Check selected lines/cohorts/genotypes.")
    return nonempty_views


def build_view_labels(views: list[ViewSpec]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for view in views:
        name = str(view.name)
        parts = name.split(" ", 1)
        if len(parts) == 2 and ":" in parts[1]:
            genotype, dataset_name = parts
            line, _, cohort = dataset_name.partition(":")
            labels[name] = f"{genotype.upper()} {line} {cohort}"
        else:
            labels[name] = name.upper() if name in {"wt", "het", "hom"} else name
    return labels


def build_view_colors(views: list[ViewSpec]) -> dict[str, str]:
    colors: dict[str, str] = {}
    fallback_idx = 0
    for view in views:
        name = str(view.name)
        genotype = name.split()[0] if name.split() else name
        if genotype in GENOTYPE_COLORS:
            colors[name] = GENOTYPE_COLORS[genotype]
        else:
            colors[name] = f"C{fallback_idx % 10}"
            fallback_idx += 1
    return colors


def summarize_views(df: pd.DataFrame, views: list[ViewSpec]) -> pd.DataFrame:
    rows = []
    for view in views:
        sub = view.selector(df)
        rows.append(
            {
                "view": view.name,
                "animals": sub["animal"].nunique() if "animal" in sub.columns else None,
                "trials": len(sub),
                "datasets": ", ".join(sorted(sub["dataset_key"].dropna().astype(str).unique())) if "dataset_key" in sub.columns else "",
                "experimentor": ", ".join(sorted(sub["experimenter"].dropna().astype(str).unique())) if "experimenter" in sub.columns else "",
            }
        )
    return pd.DataFrame(rows)


def prepare_stimdur_comparison(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    stim_durs: list[int] | tuple[int, ...] = tuple(DEFAULT_STIM_DURS),
    stimdur_col: str = STIMDUR_COL,
    cfg: StimDurComparisonConfig | None = None,
    fcfg: FilterConfig | None = None,
    style: PlotStyle | None = None,
    stimdur_pretty: dict[str, str] | None = None,
    stimdur_colors: dict[str, str] | None = None,
    view_colors: dict[str, str] | None = None,
    view_pretty: dict[str, str] | None = None,
) -> dict[str, Any]:
    cfg = cfg or StimDurComparisonConfig(
        error_mode="individuals",
        skip_psy_fits=(50,),
        ild_shift_for_abl50=True,
    )
    fcfg = fcfg or FilterConfig(
        training_min=16,
        session_min=13,
        drop_repeat_trials=True,
        session_type_values=[2],
    )
    style = style or PlotStyle()
    stimdur_pretty = stimdur_pretty or STIMDUR_PRETTY
    view_pretty = view_pretty or build_view_labels(views)
    view_colors = view_colors or build_view_colors(views)

    df_filtered = apply_filters(df, fcfg)
    available_stim_durs = {
        int(float(x))
        for x in pd.to_numeric(df_filtered[stimdur_col], errors="coerce").dropna().unique()
    }
    active_stim_durs = [int(sd) for sd in stim_durs if int(sd) in available_stim_durs]
    if not active_stim_durs:
        raise ValueError(f"No stim durations from stim_durs={stim_durs!r} were present after filtering.")

    stimdur_specs = make_stimdur_specs(active_stim_durs, stim_dur_col=stimdur_col)
    if stimdur_colors is None:
        stimdur_colors = {s.name: STIMDUR_PALETTE[i % len(STIMDUR_PALETTE)] for i, s in enumerate(stimdur_specs)}

    df_by_view = {view.name: view.selector(df_filtered.copy()) for view in views}
    prepared = build_prepared_by_view_and_stimdur(df_filtered, views, stimdur_specs, cfg)
    group_jnd = compute_group_jnd_by_view_and_stimdur(prepared, skip_abl=50)

    return {
        "df": df_filtered,
        "df_by_view": df_by_view,
        "prepared": prepared,
        "group_jnd": group_jnd,
        "stimdur_specs": stimdur_specs,
        "stimdur_colors": stimdur_colors,
        "stimdur_pretty": stimdur_pretty,
        "view_colors": view_colors,
        "view_pretty": view_pretty,
        "cfg": cfg,
        "fcfg": fcfg,
        "style": style,
        "stimdur_col": stimdur_col,
    }


def prepare_unbiased_stimdur_comparison(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    stim_durs: list[int] | tuple[int, ...] = tuple(DEFAULT_STIM_DURS),
    stimdur_col: str = STIMDUR_COL,
    cfg: StimDurComparisonConfig | None = None,
    fcfg: FilterConfig | None = None,
    style: PlotStyle | None = None,
    stimdur_pretty: dict[str, str] | None = None,
    stimdur_colors: dict[str, str] | None = None,
    view_colors: dict[str, str] | None = None,
    view_pretty: dict[str, str] | None = None,
    unbiased_session_types: tuple[int, ...] = (2,),
    biased_session_types: tuple[int, ...] = (23,),
    rightward_ild_sign: int = 1,
    min_direction_imbalance: float = 0.0,
    max_unbiased_imbalance: float = 0.2,
) -> dict[str, Any]:
    cfg = cfg or StimDurComparisonConfig(
        error_mode="individuals",
        skip_psy_fits=(50,),
        ild_shift_for_abl50=True,
    )
    fcfg = fcfg or FilterConfig(
        training_min=16,
        session_min=13,
        drop_repeat_trials=True,
        session_type_values=[2, 23],
    )
    style = style or PlotStyle()
    stimdur_pretty = stimdur_pretty or STIMDUR_PRETTY
    view_pretty = view_pretty or build_view_labels(views)
    view_colors = view_colors or build_view_colors(views)

    prefilter_session_types = sorted(set(unbiased_session_types) | set(biased_session_types))
    mixed_filter = FilterConfig(
        training_min=fcfg.training_min,
        session_min=fcfg.session_min,
        drop_repeat_trials=fcfg.drop_repeat_trials,
        session_type_values=prefilter_session_types,
    )
    df_filtered = apply_filters(df, mixed_filter)
    df_labeled = add_biased_block_condition(
        df_filtered,
        biased_session_types=tuple(int(x) for x in biased_session_types),
        unbiased_rt_session_types=tuple(int(x) for x in unbiased_session_types),
        short_duration_value=None,
        rightward_ild_sign=rightward_ild_sign,
        min_direction_imbalance=min_direction_imbalance,
        max_unbiased_imbalance=max_unbiased_imbalance,
    )
    df_unbiased = df_labeled[df_labeled["block_condition"] == "unbiased"].copy()

    bundle = prepare_stimdur_comparison(
        df=df_unbiased,
        views=views,
        stim_durs=stim_durs,
        stimdur_col=stimdur_col,
        cfg=cfg,
        fcfg=FilterConfig(
            training_min=0,
            session_min=0,
            drop_repeat_trials=False,
            session_type_values=None,
        ),
        style=style,
        stimdur_pretty=stimdur_pretty,
        stimdur_colors=stimdur_colors,
        view_colors=view_colors,
        view_pretty=view_pretty,
    )
    bundle["df_prefiltered"] = df_filtered
    bundle["df_labeled"] = df_labeled
    bundle["selection_summary"] = {
        "session_type_2_trials": int((pd.to_numeric(df_filtered["session_type"], errors="coerce") == 2).sum()),
        "session_type_23_unbiased_trials": int(
            (
                (pd.to_numeric(df_labeled["session_type"], errors="coerce") == 23)
                & (df_labeled["block_condition"] == "unbiased")
            ).sum()
        ),
    }
    return bundle


def prepare_biased_block_stimdur_comparison(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    stim_durs: list[int] | tuple[int, ...] = tuple(DEFAULT_STIM_DURS),
    stimdur_col: str = STIMDUR_COL,
    cfg: StimDurComparisonConfig | None = None,
    fcfg: FilterConfig | None = None,
    style: PlotStyle | None = None,
    stimdur_pretty: dict[str, str] | None = None,
    stimdur_colors: dict[str, str] | None = None,
    view_colors: dict[str, str] | None = None,
    view_pretty: dict[str, str] | None = None,
    biased_session_types: tuple[int, ...] = (23,),
    block_conditions: tuple[str, ...] = BLOCK_CONDITION_ORDER,
    rightward_ild_sign: int = 1,
    min_direction_imbalance: float = 0.0,
    max_unbiased_imbalance: float = 0.2,
) -> dict[str, Any]:
    cfg = cfg or StimDurComparisonConfig(
        error_mode="individuals",
        skip_psy_fits=(50,),
        ild_shift_for_abl50=True,
    )
    fcfg = fcfg or FilterConfig(
        training_min=16,
        session_min=13,
        drop_repeat_trials=True,
        session_type_values=[23],
    )
    style = style or PlotStyle()
    stimdur_pretty = stimdur_pretty or STIMDUR_PRETTY
    view_pretty = view_pretty or build_view_labels(views)
    view_colors = view_colors or build_view_colors(views)

    biased_filter = FilterConfig(
        training_min=fcfg.training_min,
        session_min=fcfg.session_min,
        drop_repeat_trials=fcfg.drop_repeat_trials,
        session_type_values=list(biased_session_types),
    )
    df_filtered = apply_filters(df, biased_filter)
    df_blocks = add_biased_block_condition(
        df_filtered,
        biased_session_types=tuple(int(x) for x in biased_session_types),
        unbiased_rt_session_types=(),
        short_duration_value=None,
        rightward_ild_sign=rightward_ild_sign,
        min_direction_imbalance=min_direction_imbalance,
        max_unbiased_imbalance=max_unbiased_imbalance,
    )

    per_condition_filter = FilterConfig(
        training_min=0,
        session_min=0,
        drop_repeat_trials=False,
        session_type_values=None,
    )
    condition_bundles: dict[str, dict[str, Any]] = {}
    condition_views: dict[str, list[ViewSpec]] = {}
    summary_rows: list[dict[str, Any]] = []

    for condition in block_conditions:
        df_condition = df_blocks[df_blocks["block_condition"] == condition].copy()
        available_views = [view for view in views if not view.selector(df_condition).empty]
        if not available_views:
            continue

        condition_bundles[condition] = prepare_stimdur_comparison(
            df=df_condition,
            views=available_views,
            stim_durs=stim_durs,
            stimdur_col=stimdur_col,
            cfg=cfg,
            fcfg=per_condition_filter,
            style=style,
            stimdur_pretty=stimdur_pretty,
            stimdur_colors=stimdur_colors,
            view_colors=view_colors,
            view_pretty=view_pretty,
        )
        condition_views[condition] = available_views

        for view in available_views:
            df_view = view.selector(df_condition)
            summary_rows.append(
                {
                    "block_condition": condition,
                    "view": view.name,
                    "animals": df_view["animal"].nunique() if "animal" in df_view.columns else pd.NA,
                    "trials": len(df_view),
                }
            )

    if not condition_bundles:
        raise ValueError("No biased-block stim-duration data remained after filtering.")

    block_summary = pd.DataFrame(summary_rows)
    if not block_summary.empty:
        block_summary = block_summary.sort_values(["block_condition", "view"]).reset_index(drop=True)

    return {
        "df": df_blocks,
        "df_filtered": df_filtered,
        "condition_bundles": condition_bundles,
        "condition_views": condition_views,
        "block_summary": block_summary,
        "cfg": cfg,
        "fcfg": biased_filter,
        "style": style,
        "stimdur_col": stimdur_col,
        "stim_durs": list(stim_durs),
        "view_colors": view_colors,
        "view_pretty": view_pretty,
        "stimdur_pretty": stimdur_pretty,
    }


def plot_stimdur_comparison(
    *,
    bundle: dict[str, Any],
    views: list[ViewSpec],
    plot_mode: str = "by_view",
    show: bool = True,
    abls: list[int] | tuple[int, ...] = (20, 40, 60),
    absilds: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
    xlim: tuple[float, float] = (0.0, 0.5),
    debug: bool = False,
) -> dict[str, Any]:
    prepared = bundle["prepared"]
    group_jnd = bundle["group_jnd"]
    stimdur_specs = bundle["stimdur_specs"]
    stimdur_colors = bundle["stimdur_colors"]
    stimdur_pretty = bundle["stimdur_pretty"]
    view_colors = bundle["view_colors"]
    view_pretty = bundle["view_pretty"]
    cfg = bundle["cfg"]
    style = bundle["style"]

    figures: dict[str, Any] = {}

    if plot_mode in {"by_view", "all"}:
        figures["by_view"] = {}
        for view in views:
            fig = plot_stimdur_4x3_for_view(
                prepared_for_view=prepared[view.name],
                group_jnd_for_view=group_jnd[view.name],
                stimdur_specs=stimdur_specs,
                stimdur_colors=stimdur_colors,
                stimdur_pretty=stimdur_pretty,
                view_name=view_pretty.get(view.name, view.name),
                cfg=cfg,
                style=style,
            )
            figures["by_view"][view.name] = fig
            if show:
                plt.show()

    if plot_mode in {"by_stimdur", "all"}:
        figures["by_stimdur"] = {}
        for stimdur in stimdur_specs:
            fig = plot_genotypes_4x3_for_stimdur(
                prepared=prepared,
                group_jnd=group_jnd,
                views=views,
                stimdur_name=stimdur.name,
                view_colors=view_colors,
                cfg=cfg,
                style=style,
                stimdur_pretty=stimdur_pretty,
                view_pretty=view_pretty,
            )
            figures["by_stimdur"][stimdur.name] = fig
            if show:
                plt.show()

    if plot_mode in {"performance_by_view", "all"}:
        figures["performance_by_view"] = {}
        for view in views:
            fig = plot_absild_perf_across_stimdur_1x3_for_view(
                prepared_for_view=prepared[view.name],
                stimdur_specs=stimdur_specs,
                view_name=view_pretty.get(view.name, view.name),
                cfg=cfg,
                style=style,
                stimdur_pretty=stimdur_pretty,
                abls=list(abls),
            )
            figures["performance_by_view"][view.name] = fig
            if show:
                plt.show()

    if plot_mode in {"kreg_by_view", "all"}:
        figures["kreg_by_view"] = {}
        for view in views:
            fig = plot_kreg_4x3_by_abl_for_view(
                df_view=bundle["df_by_view"][view.name],
                view_name=view_pretty.get(view.name, view.name),
                stimdur_specs=stimdur_specs,
                stimdur_col=bundle["stimdur_col"],
                stimdur_colors=stimdur_colors,
                stimdur_pretty=stimdur_pretty,
                abls=tuple(abls),
                xlim=xlim,
                debug=debug,
            )
            figures["kreg_by_view"][view.name] = fig
            if show:
                plt.show()

    if plot_mode in {"performance_all", "all"}:
        figures["performance_all"] = plot_absild_perf_3x5_all_genotypes(
            prepared=prepared,
            views=views,
            stimdur_specs=stimdur_specs,
            abls=list(abls),
            style=style,
            stimdur_pretty=stimdur_pretty,
            view_colors=view_colors,
            view_pretty=view_pretty,
            absilds=list(absilds),
        )
        if show:
            plt.show()

    valid_modes = {"by_view", "by_stimdur", "performance_by_view", "performance_all", "kreg_by_view", "all"}
    if plot_mode not in valid_modes:
        raise ValueError(f"plot_mode must be one of {sorted(valid_modes)}.")

    return {"figures": figures, **bundle}


STIMDUR_SUMMARY_GROUPS = {
    "8": ("8",),
    "16": ("16",),
    "32": ("32",),
    "64": ("64",),
    "RT": ("0",),
}
STIMDUR_SUMMARY_COLORS = {
    "8": "#B2A706",
    "16": "#0072B2",
    "32": "#E69F00",
    "64": "#D55E00",
    "RT": "#4D4D4D",
}
STIMDUR_SUMMARY_MARKERS = {
    "rightward": "s",
    "leftward": "^",
}


def _stimdur_summary_group_for_name(stimdur_name: str) -> str | None:
    stimdur_name = str(stimdur_name)
    for group_name, names in STIMDUR_SUMMARY_GROUPS.items():
        if stimdur_name in names:
            return group_name
    return None


def _normalize_abls(abls: int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(abls, (int, np.integer)):
        return (int(abls),)
    return tuple(int(a) for a in abls)


def _abls_title(abls: tuple[int, ...]) -> str:
    return ", ".join(str(abl) for abl in abls)


PSY_PARAM_SPECS = [
    ("slope_a", "Slope (a)"),
    ("bias_b", "Bias (b)"),
    ("lower_c", "Lower (c)"),
    ("upper_d", "Upper (d)"),
    ("asymmetry_1_minus_d_minus_c", "1 - d - c"),
]

TEMP_PSY_PARAM_SPECS = [
    ("temp_slope_nearest_ild_regression", "TEMP: slope from nearest ILDs"),
    ("temp_slope_bias_flank_mean", "TEMP: flank evidence around bias"),
    ("bias_b", "Bias (b)"),
    ("lower_c", "Lower (c)"),
    ("upper_d", "Upper (d)"),
    ("asymmetry_1_minus_d_minus_c", "1 - d - c"),
    ("JND", "JND"),
]

TEMP_PSY_PARAM_LINE_COLUMNS = [
    [
        ("temp_slope_nearest_ild_regression", "TEMP: slope from nearest ILDs"),
        ("temp_slope_bias_flank_mean", "TEMP: flank evidence around bias"),
        ("bias_b", "Bias (b)"),
        ("JND", "JND"),
    ],
    [
        ("lower_c", "Lower (c)"),
        ("upper_d", "Upper (d)"),
        ("asymmetry_1_minus_d_minus_c", "1 - d - c"),
    ],
]

TEMP_PSY_PARAM_COLLAPSED_LINE_COLUMNS = [
    [
        ("temp_slope_nearest_ild_regression_collapsed", "TEMP: slope"),
        ("temp_slope_bias_flank_mean_collapsed", "TEMP: flank evidence"),
        ("bias_b_collapsed", "Bias (L + R) / 2"),
    ],
    [
        ("lapse_collapsed", "Lapses (L)"),
        ("asymmetry_1_minus_d_minus_c_collapsed", "1 - d - c"),
    ],
]


def _temporary_bias_flank_slope_rows(
    *,
    animal: Any,
    abl: Any,
    res: dict[str, Any],
    bias: float,
) -> list[dict[str, Any]]:
    ilds = pd.to_numeric(pd.Series(res.get("ILDs", [])), errors="coerce").to_numpy(dtype=float)
    prop_right = pd.to_numeric(pd.Series(res.get("PropLeft", [])), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ilds) & np.isfinite(prop_right)
    ilds = ilds[valid]
    prop_right = prop_right[valid]
    if len(ilds) < 2 or not np.isfinite(bias):
        return []

    left_candidates = np.where(ilds < bias)[0]
    right_candidates = np.where(ilds > bias)[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return []

    left_idx = left_candidates[np.argmax(ilds[left_candidates])]
    right_idx = right_candidates[np.argmin(ilds[right_candidates])]
    left_ild = ilds[left_idx]
    right_ild = ilds[right_idx]
    left_prop = prop_right[left_idx]
    right_prop = prop_right[right_idx]
    if not np.isfinite(left_ild) or not np.isfinite(right_ild) or right_ild == left_ild:
        return []

    rows = []
    nearest_regression_slope = (right_prop - left_prop) / (right_ild - left_ild)
    if np.isfinite(nearest_regression_slope):
        rows.append(
            {
                "animal": animal,
                "ABL": int(abl),
                "metric": "temp_slope_nearest_ild_regression",
                "value": float(nearest_regression_slope),
            }
        )

    flank_mean = np.mean([right_prop, 1.0 - left_prop])
    if np.isfinite(flank_mean):
        rows.append(
            {
                "animal": animal,
                "ABL": int(abl),
                "metric": "temp_slope_bias_flank_mean",
                "value": float(flank_mean),
            }
        )
    return rows


def _fit_psychometric_params_by_animal_abl(
    df_view: pd.DataFrame,
    *,
    include_temp_slopes: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df_view.empty or "animal" not in df_view.columns:
        return pd.DataFrame(columns=["animal", "ABL", "metric", "value"])

    for animal, df_animal in df_view.groupby("animal", dropna=False, sort=False):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                results = Psychometric.compute_psychometrics_by_ABL(df_animal, model="my_psycho")
            except Exception:
                results = {}
        for abl, res in results.items():
            pars = res.get("pars") if isinstance(res, dict) else None
            if pars is None or len(pars) < 4:
                continue
            for param_i, (metric, _) in enumerate(PSY_PARAM_SPECS[:4]):
                value = pd.to_numeric(pd.Series([pars[param_i]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    rows.append({"animal": animal, "ABL": int(abl), "metric": metric, "value": float(value)})
            lower_c = pd.to_numeric(pd.Series([pars[2]]), errors="coerce").iloc[0]
            upper_d = pd.to_numeric(pd.Series([pars[3]]), errors="coerce").iloc[0]
            if pd.notna(lower_c) and pd.notna(upper_d):
                rows.append(
                    {
                        "animal": animal,
                        "ABL": int(abl),
                        "metric": "asymmetry_1_minus_d_minus_c",
                        "value": float(1.0 - upper_d - lower_c),
                    }
                )
            if include_temp_slopes:
                bias = pd.to_numeric(pd.Series([pars[1]]), errors="coerce").iloc[0]
                rows.extend(
                    _temporary_bias_flank_slope_rows(
                        animal=animal,
                        abl=abl,
                        res=res,
                        bias=float(bias),
                    )
                )
    return pd.DataFrame(rows, columns=["animal", "ABL", "metric", "value"])


def _collect_biased_block_stimdur_summary_metrics(
    bundle: dict[str, Any],
    *,
    include_abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
    condition_order: tuple[str, ...] = ("rightward", "leftward"),
) -> pd.DataFrame:
    include_abls = _normalize_abls(include_abls)
    rows: list[dict[str, Any]] = []

    for condition in condition_order:
        condition_bundle = bundle.get("condition_bundles", {}).get(condition)
        if condition_bundle is None:
            continue
        prepared = condition_bundle.get("prepared", {})
        for view_name, by_stimdur in prepared.items():
            for stimdur_name, tables in by_stimdur.items():
                duration_group = _stimdur_summary_group_for_name(stimdur_name)
                if duration_group is None:
                    continue

                df_view = tables.get("df_view", pd.DataFrame()).copy()
                if not df_view.empty and {"animal", "ABL", "success"}.issubset(df_view.columns):
                    perf = df_view.copy()
                    perf["ABL"] = pd.to_numeric(perf["ABL"], errors="coerce")
                    perf["success"] = pd.to_numeric(perf["success"], errors="coerce")
                    perf = perf[
                        perf["ABL"].isin(include_abls)
                        & perf["success"].notna()
                        & perf["success"].ne(0)
                    ].copy()
                    if not perf.empty:
                        perf["value"] = perf["success"].eq(1).astype(float)
                        perf_rows = perf.groupby(["animal", "ABL"], dropna=False)["value"].mean().reset_index()
                        for _, row in perf_rows.iterrows():
                            rows.append(
                                {
                                    "animal": row["animal"],
                                    "view": view_name,
                                    "block_condition": condition,
                                    "stimdur": stimdur_name,
                                    "duration_group": duration_group,
                                    "ABL": int(row["ABL"]),
                                    "metric": "prop_correct",
                                    "value": float(row["value"]),
                                }
                            )

                    bias_rows = _fit_psychometric_params_by_animal_abl(df_view)
                    if not bias_rows.empty:
                        bias_rows["ABL"] = pd.to_numeric(bias_rows["ABL"], errors="coerce")
                        bias_rows = bias_rows[bias_rows["metric"].astype(str) == "bias_b"].copy()
                        bias_rows = bias_rows[bias_rows["ABL"].isin(include_abls)].dropna(subset=["value"]).copy()
                        for _, row in bias_rows.iterrows():
                            rows.append(
                                {
                                    "animal": row["animal"],
                                    "view": view_name,
                                    "block_condition": condition,
                                    "stimdur": stimdur_name,
                                    "duration_group": duration_group,
                                    "ABL": int(row["ABL"]),
                                    "metric": "bias_b",
                                    "value": float(row["value"]),
                                }
                            )

                jnd = tables.get("jnd_indiv", pd.DataFrame()).copy()
                if not jnd.empty and {"subject", "ABL", "JND"}.issubset(jnd.columns):
                    jnd["ABL"] = pd.to_numeric(jnd["ABL"], errors="coerce")
                    jnd["JND"] = pd.to_numeric(jnd["JND"], errors="coerce")
                    jnd = jnd[jnd["ABL"].isin(include_abls)].dropna(subset=["JND"]).copy()
                    for _, row in jnd.iterrows():
                        rows.append(
                            {
                                "animal": row["subject"],
                                "view": view_name,
                                "block_condition": condition,
                                "stimdur": stimdur_name,
                                "duration_group": duration_group,
                                "ABL": int(row["ABL"]),
                                "metric": "JND",
                                "value": float(row["JND"]),
                            }
                        )

    if not rows:
        return pd.DataFrame(
            columns=["animal", "view", "block_condition", "duration_group", "metric", "value", "n_values"]
        )

    metrics = pd.DataFrame(rows)
    return (
        metrics.groupby(["animal", "view", "block_condition", "duration_group", "metric"], dropna=False)["value"]
        .agg(value="mean", n_values="count")
        .reset_index()
    )


def _collect_biased_block_stimdur_psy_params(
    bundle: dict[str, Any],
    *,
    include_abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
    condition_order: tuple[str, ...] = ("rightward", "leftward"),
    include_temp_slopes: bool = False,
    include_jnd: bool = False,
    average_abls: bool = True,
) -> pd.DataFrame:
    include_abls = _normalize_abls(include_abls)
    rows: list[dict[str, Any]] = []

    for condition in condition_order:
        condition_bundle = bundle.get("condition_bundles", {}).get(condition)
        if condition_bundle is None:
            continue
        prepared = condition_bundle.get("prepared", {})
        for view_name, by_stimdur in prepared.items():
            for stimdur_name, tables in by_stimdur.items():
                duration_group = _stimdur_summary_group_for_name(stimdur_name)
                if duration_group is None:
                    continue

                df_view = tables.get("df_view", pd.DataFrame()).copy()
                params = _fit_psychometric_params_by_animal_abl(
                    df_view,
                    include_temp_slopes=include_temp_slopes,
                )
                if not params.empty:
                    params["ABL"] = pd.to_numeric(params["ABL"], errors="coerce")
                    params = params[params["ABL"].isin(include_abls)].dropna(subset=["value"]).copy()
                    for _, row in params.iterrows():
                        rows.append(
                            {
                                "animal": row["animal"],
                                "view": view_name,
                                "block_condition": condition,
                                "stimdur": stimdur_name,
                                "duration_group": duration_group,
                                "ABL": int(row["ABL"]),
                                "metric": row["metric"],
                                "value": float(row["value"]),
                            }
                        )

                if include_jnd:
                    jnd = tables.get("jnd_indiv", pd.DataFrame()).copy()
                    if not jnd.empty and {"subject", "ABL", "JND"}.issubset(jnd.columns):
                        jnd["ABL"] = pd.to_numeric(jnd["ABL"], errors="coerce")
                        jnd["JND"] = pd.to_numeric(jnd["JND"], errors="coerce")
                        jnd = jnd[jnd["ABL"].isin(include_abls)].dropna(subset=["JND"]).copy()
                        for _, row in jnd.iterrows():
                            rows.append(
                                {
                                    "animal": row["subject"],
                                    "view": view_name,
                                    "block_condition": condition,
                                    "stimdur": stimdur_name,
                                    "duration_group": duration_group,
                                    "ABL": int(row["ABL"]),
                                    "metric": "JND",
                                    "value": float(row["JND"]),
                                }
                            )

    group_cols = ["animal", "view", "block_condition", "duration_group", "metric"]
    if not average_abls:
        group_cols.insert(4, "ABL")

    if not rows:
        return pd.DataFrame(columns=[*group_cols, "value", "n_values"])

    metrics = pd.DataFrame(rows)
    return (
        metrics.groupby(group_cols, dropna=False)["value"]
        .agg(value="mean", n_values="count")
        .reset_index()
    )


def _collect_biased_block_stimdur_collapsed_temp_params(
    bundle: dict[str, Any],
    *,
    include_abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
) -> pd.DataFrame:
    side_metrics = _collect_biased_block_stimdur_psy_params(
        bundle,
        include_abls=include_abls,
        condition_order=("rightward", "leftward"),
        include_temp_slopes=True,
        include_jnd=False,
        average_abls=False,
    )
    if side_metrics.empty:
        return pd.DataFrame(columns=["animal", "view", "duration_group", "metric", "value", "n_abls"])

    rows: list[dict[str, Any]] = []
    group_cols = ["animal", "view", "duration_group", "ABL"]
    for keys, sub in side_metrics.groupby(group_cols, dropna=False, sort=False):
        animal, view_name, duration_group, abl = keys
        pivot = sub.pivot_table(
            index="metric",
            columns="block_condition",
            values="value",
            aggfunc="mean",
        )

        def side_value(metric: str, condition: str) -> float:
            if metric not in pivot.index or condition not in pivot.columns:
                return np.nan
            value = pd.to_numeric(pd.Series([pivot.loc[metric, condition]]), errors="coerce").iloc[0]
            return float(value) if pd.notna(value) else np.nan

        collapsed_specs = [
            (
                "temp_slope_nearest_ild_regression_collapsed",
                (
                    side_value("temp_slope_nearest_ild_regression", "leftward")
                    + side_value("temp_slope_nearest_ild_regression", "rightward")
                )
                / 2.0,
            ),
            (
                "temp_slope_bias_flank_mean_collapsed",
                (
                    side_value("temp_slope_bias_flank_mean", "leftward")
                    + side_value("temp_slope_bias_flank_mean", "rightward")
                )
                / 2.0,
            ),
            (
                "bias_b_collapsed",
                (side_value("bias_b", "leftward") + side_value("bias_b", "rightward")) / 2.0,
            ),
            (
                "asymmetry_1_minus_d_minus_c_collapsed",
                (
                    side_value("asymmetry_1_minus_d_minus_c", "leftward")
                    + side_value("asymmetry_1_minus_d_minus_c", "rightward")
                )
                / 2.0,
            ),
        ]

        lapse_left = (
            side_value("lower_c", "leftward")
            + (1.0 - side_value("upper_d", "leftward"))
        ) / 2.0
        lapse_right = (
            side_value("lower_c", "rightward")
            + (1.0 - side_value("upper_d", "rightward"))
        ) / 2.0
        collapsed_specs.append(("lapse_collapsed", (lapse_left + lapse_right) / 2.0))

        for metric, value in collapsed_specs:
            if np.isfinite(value):
                rows.append(
                    {
                        "animal": animal,
                        "view": view_name,
                        "duration_group": duration_group,
                        "ABL": int(abl),
                        "metric": metric,
                        "value": float(value),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["animal", "view", "duration_group", "metric", "value", "n_abls"])
    collapsed_by_abl = pd.DataFrame(rows)
    return (
        collapsed_by_abl.groupby(["animal", "view", "duration_group", "metric"], dropna=False)["value"]
        .agg(value="mean", n_abls="count")
        .reset_index()
    )


def plot_biased_block_stimdur_summary_metrics(
    *,
    bundle: dict[str, Any],
    show: bool = True,
    abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
) -> dict[str, Any]:
    """Plot ABL-averaged bias, proportion correct, and JND by genotype panels."""
    style = bundle["style"]
    view_pretty = bundle.get("view_pretty", {})
    include_abls = _normalize_abls(abls)
    metrics = _collect_biased_block_stimdur_summary_metrics(bundle, include_abls=include_abls)
    if metrics.empty:
        raise ValueError("No ABL-averaged biased-block stim-duration summary metrics were available.")

    view_names: list[str] = []
    for condition in ("rightward", "leftward"):
        for view in bundle.get("condition_views", {}).get(condition, []):
            if view.name not in view_names:
                view_names.append(view.name)
    if not view_names:
        view_names = sorted(metrics["view"].dropna().astype(str).unique())

    duration_groups = [name for name in STIMDUR_SUMMARY_GROUPS if name in set(metrics["duration_group"].astype(str))]
    condition_order = [name for name in ("leftward", "rightward") if name in set(metrics["block_condition"].astype(str))]
    group_width = len(duration_groups)
    group_gap = 1.0
    condition_bases = {name: i * (group_width + group_gap) for i, name in enumerate(condition_order)}
    condition_centers = {
        name: base + (group_width - 1) / 2.0
        for name, base in condition_bases.items()
    }
    duration_positions = {
        (condition, duration_group): condition_bases[condition] + duration_i
        for condition in condition_order
        for duration_i, duration_group in enumerate(duration_groups)
    }
    specs = [
        ("bias_b", "Psychometric bias", "Bias (b)"),
        ("prop_correct", "Proportion correct", "Proportion correct"),
        ("JND", "JND", "JND"),
    ]
    fs = style.legend_fs
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(
        len(specs),
        len(view_names),
        figsize=(5.4 * len(view_names), 3.6 * len(specs)),
        squeeze=False,
        sharey="row",
    )

    for row_i, (metric, title, ylabel) in enumerate(specs):
        metric_df = metrics[metrics["metric"].astype(str) == metric].copy()
        for col_i, view_name in enumerate(view_names):
            ax = axes[row_i, col_i]
            for condition in condition_order:
                marker = STIMDUR_SUMMARY_MARKERS.get(condition, "o")
                for duration_group in duration_groups:
                    color = STIMDUR_SUMMARY_COLORS.get(duration_group, "0.4")
                    sub = metric_df[
                        (metric_df["view"].astype(str) == view_name)
                        & (metric_df["block_condition"].astype(str) == condition)
                        & (metric_df["duration_group"].astype(str) == duration_group)
                    ].copy()
                    values = pd.to_numeric(sub["value"], errors="coerce").dropna()
                    if values.empty:
                        continue
                    x = duration_positions[(condition, duration_group)]
                    jitter = rng.uniform(-0.018, 0.018, size=len(values))
                    ax.scatter(
                        np.full(len(values), x, dtype=float) + jitter,
                        values.to_numpy(dtype=float),
                        s=24,
                        facecolors="none",
                        edgecolors=color,
                        marker=marker,
                        alpha=0.40,
                        linewidths=0.9,
                        zorder=3,
                    )
                    ax.errorbar(
                        x,
                        float(values.mean()),
                        yerr=sem(values.to_numpy(dtype=float)),
                        color=color,
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=7.5,
                        linestyle="None",
                        linewidth=1.5,
                        elinewidth=1.3,
                        capsize=3,
                        zorder=5,
                    )

            if metric in {"bias_b", "asymmetry_1_minus_d_minus_c"}:
                ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
            elif metric == "prop_correct":
                ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0, zorder=0)
                ax.set_ylim(0, 1)
            if row_i == 0:
                ax.set_title(view_pretty.get(view_name, view_name), fontsize=fs, pad=style.title_pad)
            tick_positions = [
                duration_positions[(condition, duration_group)]
                for condition in condition_order
                for duration_group in duration_groups
            ]
            tick_labels = [
                duration_group
                for condition in condition_order
                for duration_group in duration_groups
            ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                tick_labels,
                rotation=35,
                ha="center",
                fontsize=max(8, fs - 3),
            )
            if row_i == len(specs) - 1:
                for condition, center in condition_centers.items():
                    group_label = ax.text(
                        center,
                        0,
                        BLOCK_CONDITION_PRETTY.get(condition, condition),
                        transform=ax.get_xaxis_transform()
                        + ScaledTranslation(0, -42 / 72, fig.dpi_scale_trans),
                        ha="center",
                        va="top",
                        fontsize=max(8, fs - 2),
                        clip_on=False,
                    )
                    group_label.set_in_layout(False)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.grid(True, axis="x", linestyle=":", alpha=0.25)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=fs, color="black")
                ax.tick_params(axis="y", labelsize=fs)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
            if col_i != 0:
                ax.spines["left"].set_visible(False)

    duration_handles = [
        Line2D(
            [],
            [],
            color=STIMDUR_SUMMARY_COLORS.get(group, "0.4"),
            marker="o",
            linestyle="None",
            markerfacecolor=STIMDUR_SUMMARY_COLORS.get(group, "0.4"),
            markeredgecolor=STIMDUR_SUMMARY_COLORS.get(group, "0.4"),
            label=group,
        )
        for group in duration_groups
    ]
    fill_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="black", label="Animal"),
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", label="Mean"),
    ]
    fig.legend(
        handles=duration_handles + fill_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=max(1, len(duration_handles) + len(fill_handles)),
        fontsize=fs,
        frameon=False,
    )
    fig.suptitle(f"Biased blocks summary by genotype - mean of ABLs {_abls_title(include_abls)}", fontsize=fs, y=0.995)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    if show:
        plt.show()

    return {
        "summary": {
            "figure": fig,
            "metrics": metrics,
            "abls": include_abls,
            "duration_groups": STIMDUR_SUMMARY_GROUPS,
        }
    }


def plot_biased_block_stimdur_psy_params(
    *,
    bundle: dict[str, Any],
    show: bool = True,
    temporary_slope_options: bool = False,
    abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
) -> dict[str, Any]:
    """Plot ABL-averaged psychometric parameters by genotype panels."""
    style = bundle["style"]
    view_pretty = bundle.get("view_pretty", {})
    param_specs = TEMP_PSY_PARAM_SPECS if temporary_slope_options else PSY_PARAM_SPECS
    include_abls = _normalize_abls(abls)
    metrics = _collect_biased_block_stimdur_psy_params(
        bundle,
        include_abls=include_abls,
        include_temp_slopes=temporary_slope_options,
        include_jnd=temporary_slope_options,
    )
    if metrics.empty:
        raise ValueError("No ABL-averaged biased-block stim-duration psychometric parameters were available.")

    view_names: list[str] = []
    for condition in ("rightward", "leftward"):
        for view in bundle.get("condition_views", {}).get(condition, []):
            if view.name not in view_names:
                view_names.append(view.name)
    if not view_names:
        view_names = sorted(metrics["view"].dropna().astype(str).unique())

    duration_groups = [name for name in STIMDUR_SUMMARY_GROUPS if name in set(metrics["duration_group"].astype(str))]
    condition_order = [name for name in ("leftward", "rightward") if name in set(metrics["block_condition"].astype(str))]
    group_width = len(duration_groups)
    group_gap = 1.0
    condition_bases = {name: i * (group_width + group_gap) for i, name in enumerate(condition_order)}
    condition_centers = {
        name: base + (group_width - 1) / 2.0
        for name, base in condition_bases.items()
    }
    duration_positions = {
        (condition, duration_group): condition_bases[condition] + duration_i
        for condition in condition_order
        for duration_i, duration_group in enumerate(duration_groups)
    }

    fs = style.legend_fs
    rng = np.random.default_rng(6)
    fig, axes = plt.subplots(
        len(param_specs),
        len(view_names),
        figsize=(5.4 * len(view_names), 3.6 * len(param_specs)),
        squeeze=False,
        sharey="row",
    )

    for row_i, (metric, label) in enumerate(param_specs):
        metric_df = metrics[metrics["metric"].astype(str) == metric].copy()
        for col_i, view_name in enumerate(view_names):
            ax = axes[row_i, col_i]
            for condition in condition_order:
                marker = STIMDUR_SUMMARY_MARKERS.get(condition, "o")
                for duration_group in duration_groups:
                    color = STIMDUR_SUMMARY_COLORS.get(duration_group, "0.4")
                    sub = metric_df[
                        (metric_df["view"].astype(str) == view_name)
                        & (metric_df["block_condition"].astype(str) == condition)
                        & (metric_df["duration_group"].astype(str) == duration_group)
                    ].copy()
                    values = pd.to_numeric(sub["value"], errors="coerce").dropna()
                    if values.empty:
                        continue
                    x = duration_positions[(condition, duration_group)]
                    jitter = rng.uniform(-0.018, 0.018, size=len(values))
                    ax.scatter(
                        np.full(len(values), x, dtype=float) + jitter,
                        values.to_numpy(dtype=float),
                        s=24,
                        facecolors="none",
                        edgecolors=color,
                        marker=marker,
                        alpha=0.40,
                        linewidths=0.9,
                        zorder=3,
                    )
                    ax.errorbar(
                        x,
                        float(values.mean()),
                        yerr=sem(values.to_numpy(dtype=float)),
                        color=color,
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=7.5,
                        linestyle="None",
                        linewidth=1.5,
                        elinewidth=1.3,
                        capsize=3,
                        zorder=5,
                    )

            if metric in {"bias_b", "asymmetry_1_minus_d_minus_c"}:
                ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
            if row_i == 0:
                ax.set_title(view_pretty.get(view_name, view_name), fontsize=fs, pad=style.title_pad)
            tick_positions = [
                duration_positions[(condition, duration_group)]
                for condition in condition_order
                for duration_group in duration_groups
            ]
            tick_labels = [
                duration_group
                for condition in condition_order
                for duration_group in duration_groups
            ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=35, ha="center", fontsize=max(8, fs - 3))
            if row_i == len(param_specs) - 1:
                for condition, center in condition_centers.items():
                    group_label = ax.text(
                        center,
                        0,
                        BLOCK_CONDITION_PRETTY.get(condition, condition),
                        transform=ax.get_xaxis_transform()
                        + ScaledTranslation(0, -42 / 72, fig.dpi_scale_trans),
                        ha="center",
                        va="top",
                        fontsize=max(8, fs - 2),
                        clip_on=False,
                    )
                    group_label.set_in_layout(False)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.grid(True, axis="x", linestyle=":", alpha=0.25)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=fs, color="black")
                ax.tick_params(axis="y", labelsize=fs)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
            if col_i != 0:
                ax.spines["left"].set_visible(False)

    duration_handles = [
        Line2D(
            [],
            [],
            color=STIMDUR_SUMMARY_COLORS.get(group, "0.4"),
            marker="o",
            linestyle="None",
            markerfacecolor=STIMDUR_SUMMARY_COLORS.get(group, "0.4"),
            markeredgecolor=STIMDUR_SUMMARY_COLORS.get(group, "0.4"),
            label=group,
        )
        for group in duration_groups
    ]
    fill_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="black", label="Animal"),
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", label="Mean"),
    ]
    fig.legend(
        handles=duration_handles + fill_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=max(1, len(duration_handles) + len(fill_handles)),
        fontsize=fs,
        frameon=False,
    )
    title = f"Biased block psychometric parameters by genotype - mean of ABLs {_abls_title(include_abls)}"
    if temporary_slope_options:
        title = f"TEMPORARY: biased block slope options by genotype - mean of ABLs {_abls_title(include_abls)}"
    fig.suptitle(title, fontsize=fs, y=0.995)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    if show:
        plt.show()

    return {
        "summary": {
            "figure": fig,
            "metrics": metrics,
            "abls": include_abls,
            "duration_groups": STIMDUR_SUMMARY_GROUPS,
            "temporary_slope_options": temporary_slope_options,
        }
    }


def plot_biased_block_stimdur_temp_slope_lines(
    *,
    bundle: dict[str, Any],
    show: bool = True,
    abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
) -> dict[str, Any]:
    """Temporary slope-options layout with genotype mean lines and SEM shading."""
    style = bundle["style"]
    view_pretty = bundle.get("view_pretty", {})
    view_colors = bundle.get("view_colors", {})
    include_abls = _normalize_abls(abls)
    metrics = _collect_biased_block_stimdur_psy_params(
        bundle,
        include_abls=include_abls,
        include_temp_slopes=True,
        include_jnd=True,
    )
    if metrics.empty:
        raise ValueError("No ABL-averaged biased-block stim-duration psychometric parameters were available.")

    view_names: list[str] = []
    for condition in ("rightward", "leftward"):
        for view in bundle.get("condition_views", {}).get(condition, []):
            if view.name not in view_names:
                view_names.append(view.name)
    if not view_names:
        view_names = sorted(metrics["view"].dropna().astype(str).unique())

    duration_groups = [name for name in STIMDUR_SUMMARY_GROUPS if name in set(metrics["duration_group"].astype(str))]
    condition_order = [name for name in ("leftward", "rightward") if name in set(metrics["block_condition"].astype(str))]
    group_width = len(duration_groups)
    group_gap = 1.0
    condition_bases = {name: i * (group_width + group_gap) for i, name in enumerate(condition_order)}
    condition_centers = {
        name: base + (group_width - 1) / 2.0
        for name, base in condition_bases.items()
    }
    duration_positions = {
        (condition, duration_group): condition_bases[condition] + duration_i
        for condition in condition_order
        for duration_i, duration_group in enumerate(duration_groups)
    }
    tick_positions = [
        duration_positions[(condition, duration_group)]
        for condition in condition_order
        for duration_group in duration_groups
    ]
    tick_labels = [
        duration_group
        for condition in condition_order
        for duration_group in duration_groups
    ]

    fs = style.legend_fs
    n_rows = max(len(col_specs) for col_specs in TEMP_PSY_PARAM_LINE_COLUMNS)
    fig, axes = plt.subplots(
        n_rows,
        len(TEMP_PSY_PARAM_LINE_COLUMNS),
        figsize=(12.0, 3.3 * n_rows),
        squeeze=False,
    )

    for col_i, col_specs in enumerate(TEMP_PSY_PARAM_LINE_COLUMNS):
        for row_i in range(n_rows):
            ax = axes[row_i, col_i]
            if row_i >= len(col_specs):
                ax.axis("off")
                continue

            metric, label = col_specs[row_i]
            metric_df = metrics[metrics["metric"].astype(str) == metric].copy()
            for view_name in view_names:
                color = view_colors.get(view_name, GENOTYPE_COLORS.get(view_name, "0.35"))
                for condition_i, condition in enumerate(condition_order):
                    xs = []
                    means = []
                    sems = []
                    for duration_group in duration_groups:
                        sub = metric_df[
                            (metric_df["view"].astype(str) == view_name)
                            & (metric_df["block_condition"].astype(str) == condition)
                            & (metric_df["duration_group"].astype(str) == duration_group)
                        ].copy()
                        values = pd.to_numeric(sub["value"], errors="coerce").dropna().to_numpy(dtype=float)
                        if len(values) == 0:
                            continue
                        xs.append(duration_positions[(condition, duration_group)])
                        means.append(float(np.mean(values)))
                        sems.append(float(sem(values)))

                    if not xs:
                        continue
                    xs_arr = np.asarray(xs, dtype=float)
                    means_arr = np.asarray(means, dtype=float)
                    sems_arr = np.asarray(sems, dtype=float)
                    order = np.argsort(xs_arr)
                    xs_arr = xs_arr[order]
                    means_arr = means_arr[order]
                    sems_arr = sems_arr[order]

                    ax.plot(
                        xs_arr,
                        means_arr,
                        color=color,
                        marker="o",
                        markersize=5.5,
                        linewidth=1.8,
                        label=view_pretty.get(view_name, view_name) if condition_i == 0 else None,
                        zorder=4,
                    )
                    ax.fill_between(
                        xs_arr,
                        means_arr - sems_arr,
                        means_arr + sems_arr,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=2,
                    )

            if metric in {"bias_b", "asymmetry_1_minus_d_minus_c"}:
                ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
            ax.set_ylabel(label, fontsize=fs, color="black")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=35, ha="center", fontsize=max(8, fs - 3))
            is_bottom_visible_panel = row_i == len(col_specs) - 1
            if is_bottom_visible_panel:
                ax.tick_params(axis="x", bottom=True, labelbottom=True)
                for condition, center in condition_centers.items():
                    group_label = ax.text(
                        center,
                        0,
                        BLOCK_CONDITION_PRETTY.get(condition, condition),
                        transform=ax.get_xaxis_transform()
                        + ScaledTranslation(0, -42 / 72, fig.dpi_scale_trans),
                        ha="center",
                        va="top",
                        fontsize=max(8, fs - 2),
                        clip_on=False,
                    )
                    group_label.set_in_layout(False)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.tick_params(axis="y", labelsize=fs)
            ax.grid(True, axis="x", linestyle=":", alpha=0.25)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            color=view_colors.get(view_name, GENOTYPE_COLORS.get(view_name, "0.35")),
            marker="o",
            linestyle="-",
            label=view_pretty.get(view_name, view_name),
        )
        for view_name in view_names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=max(1, len(handles)),
        fontsize=fs,
        frameon=False,
    )
    fig.suptitle(
        f"TEMPORARY: biased block slope options, genotype mean lines - mean of ABLs {_abls_title(include_abls)}",
        fontsize=fs,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    if show:
        plt.show()

    return {
        "summary": {
            "figure": fig,
            "metrics": metrics,
            "abls": include_abls,
            "duration_groups": STIMDUR_SUMMARY_GROUPS,
            "temporary_slope_options": True,
            "line_summary": True,
        }
    }


def plot_biased_block_stimdur_collapsed_temp_lines(
    *,
    bundle: dict[str, Any],
    show: bool = True,
    abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
) -> dict[str, Any]:
    """Temporary left/right-collapsed slope-options layout with genotype mean lines."""
    style = bundle["style"]
    view_pretty = bundle.get("view_pretty", {})
    view_colors = bundle.get("view_colors", {})
    include_abls = _normalize_abls(abls)
    metrics = _collect_biased_block_stimdur_collapsed_temp_params(
        bundle,
        include_abls=include_abls,
    )
    if metrics.empty:
        raise ValueError("No collapsed biased-block stim-duration temporary metrics were available.")

    view_names: list[str] = []
    for condition in ("rightward", "leftward"):
        for view in bundle.get("condition_views", {}).get(condition, []):
            if view.name not in view_names:
                view_names.append(view.name)
    if not view_names:
        view_names = sorted(metrics["view"].dropna().astype(str).unique())

    duration_groups = [name for name in STIMDUR_SUMMARY_GROUPS if name in set(metrics["duration_group"].astype(str))]
    duration_positions = {duration_group: duration_i for duration_i, duration_group in enumerate(duration_groups)}
    tick_positions = [duration_positions[duration_group] for duration_group in duration_groups]

    fs = style.legend_fs
    n_rows = max(len(col_specs) for col_specs in TEMP_PSY_PARAM_COLLAPSED_LINE_COLUMNS)
    fig, axes = plt.subplots(
        n_rows,
        len(TEMP_PSY_PARAM_COLLAPSED_LINE_COLUMNS),
        figsize=(12.0, 3.3 * n_rows),
        squeeze=False,
    )

    for col_i, col_specs in enumerate(TEMP_PSY_PARAM_COLLAPSED_LINE_COLUMNS):
        for row_i in range(n_rows):
            ax = axes[row_i, col_i]
            if row_i >= len(col_specs):
                ax.axis("off")
                continue

            metric, label = col_specs[row_i]
            metric_df = metrics[metrics["metric"].astype(str) == metric].copy()
            for view_name in view_names:
                color = view_colors.get(view_name, GENOTYPE_COLORS.get(view_name, "0.35"))
                xs = []
                means = []
                sems = []
                for duration_group in duration_groups:
                    sub = metric_df[
                        (metric_df["view"].astype(str) == view_name)
                        & (metric_df["duration_group"].astype(str) == duration_group)
                    ].copy()
                    values = pd.to_numeric(sub["value"], errors="coerce").dropna().to_numpy(dtype=float)
                    if len(values) == 0:
                        continue
                    xs.append(duration_positions[duration_group])
                    means.append(float(np.mean(values)))
                    sems.append(float(sem(values)))

                if not xs:
                    continue
                xs_arr = np.asarray(xs, dtype=float)
                means_arr = np.asarray(means, dtype=float)
                sems_arr = np.asarray(sems, dtype=float)
                order = np.argsort(xs_arr)
                xs_arr = xs_arr[order]
                means_arr = means_arr[order]
                sems_arr = sems_arr[order]

                ax.plot(
                    xs_arr,
                    means_arr,
                    color=color,
                    marker="o",
                    markersize=5.5,
                    linewidth=1.8,
                    label=view_pretty.get(view_name, view_name),
                    zorder=4,
                )
                ax.fill_between(
                    xs_arr,
                    means_arr - sems_arr,
                    means_arr + sems_arr,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )

            if metric in {"bias_b_collapsed", "asymmetry_1_minus_d_minus_c_collapsed"}:
                ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
            ax.set_ylabel(label, fontsize=fs, color="black")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(duration_groups, rotation=35, ha="center", fontsize=max(8, fs - 3))
            is_bottom_visible_panel = row_i == len(col_specs) - 1
            if is_bottom_visible_panel:
                ax.tick_params(axis="x", bottom=True, labelbottom=True)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.tick_params(axis="y", labelsize=fs)
            ax.grid(True, axis="x", linestyle=":", alpha=0.25)
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            color=view_colors.get(view_name, GENOTYPE_COLORS.get(view_name, "0.35")),
            marker="o",
            linestyle="-",
            label=view_pretty.get(view_name, view_name),
        )
        for view_name in view_names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=max(1, len(handles)),
        fontsize=fs,
        frameon=False,
    )
    fig.suptitle(
        f"TEMPORARY: left/right-collapsed biased block metrics - mean of ABLs {_abls_title(include_abls)}",
        fontsize=fs,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    if show:
        plt.show()

    return {
        "summary": {
            "figure": fig,
            "metrics": metrics,
            "abls": include_abls,
            "duration_groups": STIMDUR_SUMMARY_GROUPS,
            "temporary_slope_options": True,
            "left_right_collapsed": True,
            "line_summary": True,
        }
    }


def plot_biased_block_stimdur_comparison(
    *,
    bundle: dict[str, Any],
    plot_mode: str = "by_view",
    show: bool = True,
    abls: int | list[int] | tuple[int, ...] = (20, 40, 60),
    absilds: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
    xlim: tuple[float, float] = (0.0, 0.5),
    debug: bool = False,
) -> dict[str, Any]:
    abls = _normalize_abls(abls)
    figures: dict[str, Any] = {}

    if plot_mode in {"block_condition_summary", "bias_pc_jnd", "block_condition_bias_pc_jnd"}:
        fig_payload = plot_biased_block_stimdur_summary_metrics(bundle=bundle, show=show, abls=abls)
        figures["block_condition_summary"] = fig_payload
        return {"figures": figures, **bundle}

    if plot_mode in {"block_condition_params", "psy_params", "block_condition_psy_params"}:
        fig_payload = plot_biased_block_stimdur_psy_params(bundle=bundle, show=show, abls=abls)
        figures["block_condition_params"] = fig_payload
        return {"figures": figures, **bundle}

    if plot_mode in {
        "block_condition_params_temp_slope",
        "temporary_slope_options",
        "block_condition_params_temporary_slope",
    }:
        fig_payload = plot_biased_block_stimdur_psy_params(
            bundle=bundle,
            show=show,
            temporary_slope_options=True,
            abls=abls,
        )
        figures["block_condition_params_temp_slope"] = fig_payload
        return {"figures": figures, **bundle}

    if plot_mode in {
        "block_condition_params_temp_slope_lines",
        "temporary_slope_lines",
        "block_condition_params_temporary_slope_lines",
    }:
        fig_payload = plot_biased_block_stimdur_temp_slope_lines(
            bundle=bundle,
            show=show,
            abls=abls,
        )
        figures["block_condition_params_temp_slope_lines"] = fig_payload
        return {"figures": figures, **bundle}

    if plot_mode in {
        "block_condition_params_temp_collapsed_lines",
        "temporary_slope_collapsed_lines",
        "block_condition_params_temporary_collapsed_lines",
    }:
        fig_payload = plot_biased_block_stimdur_collapsed_temp_lines(
            bundle=bundle,
            show=show,
            abls=abls,
        )
        figures["block_condition_params_temp_collapsed_lines"] = fig_payload
        return {"figures": figures, **bundle}

    if plot_mode == "condition_by_stimdur":
        figures["condition_by_stimdur"] = {}
        all_conditions = bundle["condition_bundles"]
        first_bundle = next(iter(all_conditions.values()))
        stimdur_specs = first_bundle["stimdur_specs"]
        stimdur_names = [spec.name for spec in stimdur_specs]

        view_names: list[str] = []
        for condition in BLOCK_CONDITION_ORDER:
            for view in bundle["condition_views"].get(condition, []):
                if view.name not in view_names:
                    view_names.append(view.name)

        for view_name in view_names:
            figures["condition_by_stimdur"][view_name] = {}
            prepared_by_condition = {}
            for condition in BLOCK_CONDITION_ORDER:
                condition_bundle = bundle["condition_bundles"].get(condition)
                if condition_bundle is None:
                    continue
                view_prepared = condition_bundle["prepared"].get(view_name)
                if view_prepared:
                    prepared_by_condition[condition] = view_prepared

            if not prepared_by_condition:
                continue

            for stimdur_name in stimdur_names:
                has_any = any(
                    stimdur_name in prepared_by_condition.get(condition, {})
                    for condition in prepared_by_condition
                )
                if not has_any:
                    continue
                fig = plot_block_conditions_4x3_for_stimdur_and_view(
                    prepared_by_condition=prepared_by_condition,
                    stimdur_name=stimdur_name,
                    view_name=view_name,
                    condition_order=BLOCK_CONDITION_ORDER,
                    condition_colors=BLOCK_CONDITION_COLORS,
                    cfg=first_bundle["cfg"],
                    style=first_bundle["style"],
                    stimdur_pretty=first_bundle["stimdur_pretty"],
                    view_pretty=first_bundle["view_pretty"],
                    condition_pretty=BLOCK_CONDITION_PRETTY,
                )
                figures["condition_by_stimdur"][view_name][stimdur_name] = fig
                if show:
                    plt.show()

        return {"figures": figures, **bundle}

    if plot_mode == "metric_grid_by_stimdur":
        figures["metric_grid_by_stimdur"] = {}
        all_conditions = bundle["condition_bundles"]
        first_bundle = next(iter(all_conditions.values()))
        stimdur_specs = first_bundle["stimdur_specs"]
        stimdur_names = [spec.name for spec in stimdur_specs]

        view_names: list[str] = []
        for condition in BLOCK_CONDITION_ORDER:
            for view in bundle["condition_views"].get(condition, []):
                if view.name not in view_names:
                    view_names.append(view.name)

        prepared_by_condition_and_view = {
            condition: bundle["condition_bundles"][condition]["prepared"]
            for condition in BLOCK_CONDITION_ORDER
            if condition in bundle["condition_bundles"]
        }

        for stimdur_name in stimdur_names:
            stimdur_figures = {}
            for metric in ("psy", "rt", "mt"):
                fig = plot_biased_metric_grid_for_stimdur(
                    prepared_by_condition_and_view=prepared_by_condition_and_view,
                    stimdur_name=stimdur_name,
                    view_names=view_names,
                    condition_order=BLOCK_CONDITION_ORDER,
                    condition_pretty=BLOCK_CONDITION_PRETTY,
                    view_pretty=first_bundle["view_pretty"],
                    cfg=first_bundle["cfg"],
                    style=first_bundle["style"],
                    metric=metric,
                    stimdur_pretty=first_bundle["stimdur_pretty"],
                    abl_colors=ABL_COLORS,
                )
                stimdur_figures[metric] = fig
                if show:
                    plt.show()
            figures["metric_grid_by_stimdur"][stimdur_name] = stimdur_figures

        return {"figures": figures, **bundle}

    if plot_mode == "psychometric_grid_by_condition":
        figures["psychometric_grid_by_condition"] = {}
        all_conditions = bundle["condition_bundles"]
        first_bundle = next(iter(all_conditions.values()))
        stimdur_specs = first_bundle["stimdur_specs"]

        view_names: list[str] = []
        for condition in BLOCK_CONDITION_ORDER:
            for view in bundle["condition_views"].get(condition, []):
                if view.name not in view_names:
                    view_names.append(view.name)

        for condition in BLOCK_CONDITION_ORDER:
            condition_bundle = bundle["condition_bundles"].get(condition)
            if condition_bundle is None:
                continue
            fig = plot_psychometric_stimdur_grid_for_condition(
                prepared_by_view=condition_bundle["prepared"],
                stimdur_specs=stimdur_specs,
                view_names=view_names,
                condition_name=condition,
                cfg=condition_bundle["cfg"],
                style=condition_bundle["style"],
                view_colors=condition_bundle["view_colors"],
                stimdur_pretty=condition_bundle["stimdur_pretty"],
                view_pretty=condition_bundle["view_pretty"],
                condition_pretty=BLOCK_CONDITION_PRETTY,
            )
            figures["psychometric_grid_by_condition"][condition] = fig
            if show:
                plt.show()

        return {"figures": figures, **bundle}

    if plot_mode == "psychometric_grid_all_conditions":
        all_conditions = bundle["condition_bundles"]
        first_bundle = next(iter(all_conditions.values()))
        stimdur_specs = first_bundle["stimdur_specs"]

        view_names: list[str] = []
        for condition in BLOCK_CONDITION_ORDER:
            for view in bundle["condition_views"].get(condition, []):
                if view.name not in view_names:
                    view_names.append(view.name)

        prepared_by_condition_and_view = {
            condition: bundle["condition_bundles"][condition]["prepared"]
            for condition in BLOCK_CONDITION_ORDER
            if condition in bundle["condition_bundles"]
        }
        stimdur_lookup = {spec.name: spec for spec in stimdur_specs}
        stimdur_groups = {
            "SD_8_16": (["8", "16"], "SD 8 + 16 ms"),
            "SD_32_64": (["32", "64"], "SD 32 + 64 ms"),
            "SD_RT": (["0"], "RT"),
        }
        figures["psychometric_grid_all_conditions"] = {}
        for key, (names, title_suffix) in stimdur_groups.items():
            subset_specs = [stimdur_lookup[name] for name in names if name in stimdur_lookup]
            if not subset_specs:
                continue
            figures["psychometric_grid_all_conditions"][key] = plot_psychometric_stimdur_grid_all_conditions(
                prepared_by_condition_and_view=prepared_by_condition_and_view,
                stimdur_specs=subset_specs,
                view_names=view_names,
                condition_order=BLOCK_CONDITION_ORDER,
                cfg=first_bundle["cfg"],
                style=first_bundle["style"],
                condition_colors=BLOCK_CONDITION_COLORS,
                stimdur_pretty=first_bundle["stimdur_pretty"],
                view_pretty=first_bundle["view_pretty"],
                title_suffix=title_suffix,
                shade_reference_specs=stimdur_specs,
            )
            if show:
                plt.show()
        return {"figures": figures, **bundle}

    for condition in BLOCK_CONDITION_ORDER:
        condition_bundle = bundle["condition_bundles"].get(condition)
        condition_views = bundle["condition_views"].get(condition)
        if condition_bundle is None or not condition_views:
            continue

        figures[condition] = plot_stimdur_comparison(
            bundle=condition_bundle,
            views=condition_views,
            plot_mode=plot_mode,
            show=show,
            abls=abls,
            absilds=absilds,
            xlim=xlim,
            debug=debug,
        )["figures"]

    valid_modes = {
        "by_view",
        "by_stimdur",
        "performance_by_view",
        "performance_all",
        "kreg_by_view",
        "all",
        "condition_by_stimdur",
        "metric_grid_by_stimdur",
        "psychometric_grid_by_condition",
        "psychometric_grid_all_conditions",
        "block_condition_summary",
        "block_condition_params",
        "block_condition_params_temp_slope",
        "block_condition_params_temp_slope_lines",
        "block_condition_params_temp_collapsed_lines",
    }
    if plot_mode not in valid_modes:
        raise ValueError(f"plot_mode must be one of {sorted(valid_modes)}.")

    return {"figures": figures, **bundle}
