from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from analysis.datasets import dataset_key, load_dataset_selections
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


def make_selector(*, genotypes=None, lines=None, cohorts=None, dataset_keys=None, animals=None):
    genotype_set = _as_set(genotypes)
    line_set = _as_set(lines)
    cohort_set = _as_set(cohorts)
    dataset_key_set = _as_set(dataset_keys)
    animal_set = _as_set(animals)

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
        elif comparison == "animals":
            animals = sorted(df["animal"].dropna().astype(str).str.strip().unique())
            views.extend(ViewSpec(animal, make_selector(animals=animal)) for animal in animals)
        else:
            raise ValueError("comparison must be one of: genotypes, datasets, lines, cohorts, animals, custom.")

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
