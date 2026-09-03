from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

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


def plot_biased_block_stimdur_comparison(
    *,
    bundle: dict[str, Any],
    plot_mode: str = "by_view",
    show: bool = True,
    abls: list[int] | tuple[int, ...] = (20, 40, 60),
    absilds: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
    xlim: tuple[float, float] = (0.0, 0.5),
    debug: bool = False,
) -> dict[str, Any]:
    figures: dict[str, Any] = {}

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
    }
    if plot_mode not in valid_modes:
        raise ValueError(f"plot_mode must be one of {sorted(valid_modes)}.")

    return {"figures": figures, **bundle}
