from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from analysis.datasets import dataset_key, load_dataset_selections
from GroupComparison.config import (
    FilterConfig,
    GroupComparisonConfig,
    JNDOverlaySpec,
    OverlaySpec,
    PlotStyle,
    ViewSpec,
)
from GroupComparison.layouts import plot_views_3x3, plot_abls_4x3
from GroupComparison.plots import (
    plot_jnd_comparison_per_view,
    plot_psychometric_animals_plus_average,
    plot_summary_metrics_all_views,
    plot_psychometric_params_all_views,
)
from GroupComparison.prepare import apply_filters, build_prepared, compute_group_jnd_by_view, compute_jnd_individuals_by_view


ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_DIR = ROOT / "DataFiles"


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


def load_groupcomparison_data(
    *,
    lines: list[str] | tuple[str, ...] = ("CNTNAP2",),
    cohorts: list[str] | tuple[str, ...] = ("cohort3",),
    dataset_selections: list[tuple[str, str]] | None = None,
    base_dir: str | Path = BASE_DATA_DIR,
    require_meta: bool = False,
) -> dict[str, Any]:
    selections = normalize_dataset_selections(
        lines=lines,
        cohorts=cohorts,
        dataset_selections=dataset_selections,
    )
    df_all, meta_all, dataset_info = load_dataset_selections(
        selections=selections,
        base_dir=str(base_dir),
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
):
    genotype_set = _as_set(genotypes)
    line_set = _as_set(lines)
    cohort_set = _as_set(cohorts)
    dataset_key_set = _as_set(dataset_keys)

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
        return df[mask].copy()

    return _selector


def build_custom_views(specs: list[dict[str, Any]]) -> list[ViewSpec]:
    views = []
    for spec in specs:
        name = spec["name"]
        views.append(
            ViewSpec(
                name=name,
                selector=make_selector(
                    genotypes=spec.get("genotypes", spec.get("genotype")),
                    lines=spec.get("lines", spec.get("line")),
                    cohorts=spec.get("cohorts", spec.get("cohort")),
                    dataset_keys=spec.get("dataset_keys", spec.get("dataset_key")),
                ),
            )
        )
    return views


def build_group_views(
    df: pd.DataFrame,
    *,
    comparison: str = "genotypes",
    split_by: str = "none",
    genotypes: list[str] | tuple[str, ...] | None = ("wt", "het", "hom"),
    lines: list[str] | tuple[str, ...] | None = None,
    cohorts: list[str] | tuple[str, ...] | None = None,
    custom_specs: list[dict[str, Any]] | None = None,
) -> list[ViewSpec]:
    """
    comparison:
      - "genotypes": compare genotypes, optionally split by dataset/line/cohort.
      - "datasets": compare datasets, optionally restricted to one or more genotypes.
      - "lines": compare lines, optionally restricted to one or more genotypes/cohorts.
      - "cohorts": compare cohorts, optionally restricted to one or more genotypes/lines.
      - "custom": use custom_specs directly.

    split_by for comparison="genotypes":
      - "none": one view per genotype collapsed across selected datasets.
      - "dataset": one view per genotype x line:cohort.
      - "line": one view per genotype x line.
      - "cohort": one view per genotype x cohort.
    """
    if comparison == "custom":
        if not custom_specs:
            raise ValueError("comparison='custom' requires custom_specs.")
        return build_custom_views(custom_specs)

    df_meta = df.dropna(subset=["dataset_key"]).copy()
    available_genotypes = [g for g in ("wt", "het", "hom") if g in set(df_meta.get("genotype", pd.Series(dtype=str)).astype(str))]
    genotype_list = list(genotypes) if genotypes is not None else available_genotypes

    if lines is None:
        line_list = sorted(df_meta["line"].dropna().astype(str).unique())
    else:
        line_list = list(lines)

    if cohorts is None:
        cohort_list = sorted(df_meta["cohort"].dropna().astype(str).unique())
    else:
        cohort_list = list(cohorts)

    dataset_names = sorted(df_meta["dataset_key"].dropna().astype(str).unique(), key=dataset_sort_key)
    views: list[ViewSpec] = []

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
                    views.append(
                        ViewSpec(
                            f"{genotype} {dataset_name}",
                            make_selector(genotypes=genotype, dataset_keys=dataset_name),
                        )
                    )
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

    else:
        raise ValueError("comparison must be one of: genotypes, datasets, lines, cohorts, custom.")

    nonempty_views = []
    for view in views:
        if not view.selector(df).empty:
            nonempty_views.append(view)
    if not nonempty_views:
        raise ValueError("No non-empty views were built. Check selected lines/cohorts/genotypes.")
    return nonempty_views


def load_groupcomparison_overlays(
    *,
    base_dir: str | Path = BASE_DATA_DIR,
    load_rt_psy: bool = True,
    load_jnd: bool = True,
) -> tuple[OverlaySpec, JNDOverlaySpec]:
    base_dir = Path(base_dir)
    overlay = OverlaySpec(overlay_color="black")
    jnd_overlay = JNDOverlaySpec(abl_color_map={20: "C0", 40: "C1", 60: "C3"})

    if load_rt_psy:
        reference_dir = base_dir / "Old Data" / "ILD_task"
        with open(reference_dir / "fig1_plot_data.pkl", "rb") as f:
            makefig1_data = pickle.load(f)
        with open(reference_dir / "fig1_chrono_plot_data.pkl", "rb") as f:
            makefig1_chrono = pickle.load(f)
        overlay = OverlaySpec(
            makefig1_data=makefig1_data,
            makefig1_chrono=makefig1_chrono,
            makefig1_bias=None,
            overlay_color="black",
        )

        bias_path = reference_dir / "fig1_bias_plot_data.pkl"
        if bias_path.exists():
            with open(bias_path, "rb") as f:
                makefig1_bias = pickle.load(f)
            overlay = OverlaySpec(
                makefig1_data=makefig1_data,
                makefig1_chrono=makefig1_chrono,
                makefig1_bias=makefig1_bias,
                overlay_color="black",
            )

    if load_jnd:
        with open(base_dir / "Old Data" / "ILD_task" / "jnd_analysis_data.pkl", "rb") as f:
            old_jnd_data = pickle.load(f)
        jnd_overlay = JNDOverlaySpec(
            old_jnd_data=old_jnd_data,
            abl_color_map={20: "C0", 40: "C1", 60: "C3"},
        )

    return overlay, jnd_overlay


def summarize_views(df: pd.DataFrame, views: list[ViewSpec]) -> pd.DataFrame:
    rows = []
    for view in views:
        sub = view.selector(df)
        rows.append(
            {
                "view": view.name,
                "rows": len(sub),
                "animals": sub["animal"].nunique() if "animal" in sub else 0,
                "lines": ", ".join(sorted(sub["line"].dropna().astype(str).unique())) if "line" in sub else "",
                "cohorts": ", ".join(sorted(sub["cohort"].dropna().astype(str).unique())) if "cohort" in sub else "",
                "genotypes": ", ".join(sorted(sub["genotype"].dropna().astype(str).unique())) if "genotype" in sub else "",
            }
        )
    return pd.DataFrame(rows)


GENOTYPE_COLORS = {
    "wt": "#4D4D4D",
    "het": "#7A8F28",
    "hom": "#C24A7A",
}

COHORT_STYLE_TEMPLATES = [
    {"linestyle": "-", "marker": "o", "markerfacecolor": None},
    {"linestyle": "--", "marker": "o", "markerfacecolor": "white"},
    {"linestyle": ":", "marker": "s", "markerfacecolor": "white"},
    {"linestyle": "-.", "marker": "^", "markerfacecolor": None},
]


def build_view_style_maps(
    df: pd.DataFrame,
    views: list[ViewSpec],
) -> tuple[dict[str, str], dict[str, dict]]:
    cohorts = sorted(df["cohort"].dropna().astype(str).unique())
    cohort_styles = {
        cohort: COHORT_STYLE_TEMPLATES[i % len(COHORT_STYLE_TEMPLATES)].copy()
        for i, cohort in enumerate(cohorts)
    }

    fallback_colors = [f"C{i}" for i in range(10)]
    view_colors: dict[str, str] = {}
    view_styles: dict[str, dict] = {}

    for fallback_i, view in enumerate(views):
        sub = view.selector(df)
        genotypes = sorted(sub["genotype"].dropna().astype(str).unique()) if "genotype" in sub else []
        view_cohorts = sorted(sub["cohort"].dropna().astype(str).unique()) if "cohort" in sub else []

        if len(genotypes) == 1 and genotypes[0] in GENOTYPE_COLORS:
            view_colors[view.name] = GENOTYPE_COLORS[genotypes[0]]
        else:
            view_colors[view.name] = fallback_colors[fallback_i % len(fallback_colors)]

        if len(view_cohorts) == 1:
            view_styles[view.name] = cohort_styles.get(view_cohorts[0], COHORT_STYLE_TEMPLATES[0]).copy()
        else:
            view_styles[view.name] = COHORT_STYLE_TEMPLATES[0].copy()

    return view_colors, view_styles


def run_flexible_groupcomparison(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    layout: str = "abls_4x3",
    load_overlays: bool = True,
    cfg: GroupComparisonConfig | None = None,
    fcfg: FilterConfig | None = None,
    style: PlotStyle | None = None,
    view_colors: dict[str, str] | None = None,
    view_styles: dict[str, dict] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    from GroupComparison.runner import run_groupcomparison

    cfg = cfg or GroupComparisonConfig(
        error_mode="individuals",
        skip_psy_fits=(50,),
        ild_shift_for_abl50=True,
    )
    fcfg = fcfg or FilterConfig(
        training_min=16,
        session_min=0,
        drop_repeat_trials=True,
        session_type_values=[1],
        stim_dur_values=[6000],
        sessiontype_or_stimdur="or",
    )
    style = style or PlotStyle()
    overlay, jnd_overlay = load_groupcomparison_overlays(load_rt_psy=load_overlays, load_jnd=False)
    auto_view_colors, auto_view_styles = build_view_style_maps(df, views)
    view_colors = view_colors or auto_view_colors
    view_styles = view_styles or auto_view_styles

    out = run_groupcomparison(
        df=df,
        views=views,
        cfg=cfg,
        fcfg=fcfg,
        style=style,
        overlay=overlay,
        jnd_overlay=jnd_overlay,
        layout=layout,
        view_colors=view_colors,
        view_styles=view_styles,
        show=show,
    )
    out["view_summary"] = summarize_views(df, views)
    out["view_colors"] = view_colors
    out["view_styles"] = view_styles
    return out


def prepare_flexible_groupcomparison(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    load_overlays: bool = True,
    cfg: GroupComparisonConfig | None = None,
    fcfg: FilterConfig | None = None,
    style: PlotStyle | None = None,
    view_colors: dict[str, str] | None = None,
    view_styles: dict[str, dict] | None = None,
) -> dict[str, Any]:
    cfg = cfg or GroupComparisonConfig(
        error_mode="individuals",
        skip_psy_fits=(50,),
        ild_shift_for_abl50=True,
    )
    fcfg = fcfg or FilterConfig(
        training_min=16,
        session_min=0,
        drop_repeat_trials=True,
        session_type_values=[1],
        stim_dur_values=[6000],
        sessiontype_or_stimdur="or",
    )
    style = style or PlotStyle()
    overlay, jnd_overlay = load_groupcomparison_overlays(load_rt_psy=load_overlays, load_jnd=True)
    auto_view_colors, auto_view_styles = build_view_style_maps(df, views)
    view_colors = view_colors or auto_view_colors
    view_styles = view_styles or auto_view_styles

    df_filtered = apply_filters(df, fcfg)
    prepared = build_prepared(df_filtered, views, cfg)
    jnd_indiv_by_view = compute_jnd_individuals_by_view(prepared, skip_abl=50)
    group_jnd_by_view = compute_group_jnd_by_view(jnd_indiv_by_view)

    return {
        "df_filtered": df_filtered,
        "prepared": prepared,
        "jnd_indiv_by_view": jnd_indiv_by_view,
        "group_jnd_by_view": group_jnd_by_view,
        "overlay": overlay,
        "jnd_overlay": jnd_overlay,
        "cfg": cfg,
        "fcfg": fcfg,
        "style": style,
        "view_colors": view_colors,
        "view_styles": view_styles,
    }


def plot_flexible_groupcomparison(
    *,
    bundle: dict[str, Any],
    views: list[ViewSpec],
    layout: str = "abls_4x3",
    cfg: GroupComparisonConfig | None = None,
    show: bool = True,
) -> dict[str, Any]:
    prepared = bundle["prepared"]
    group_jnd_by_view = bundle["group_jnd_by_view"]
    jnd_indiv_by_view = bundle["jnd_indiv_by_view"]
    overlay = bundle["overlay"]
    jnd_overlay = bundle["jnd_overlay"]
    cfg = cfg or bundle["cfg"]
    style = bundle["style"]
    view_colors = bundle["view_colors"]
    view_styles = bundle["view_styles"]

    if layout == "views_3x3":
        fig_main = plot_views_3x3(
            prepared=prepared,
            views=views,
            cfg=cfg,
            style=style,
            overlay=overlay,
            group_jnd_by_view=group_jnd_by_view,
            view_colors=view_colors,
            add_jnd_inset=True,
        )
        figs = {"main": fig_main}
        view_names = [v.name for v in views]
        fig_jnd, axes = plt.subplots(
            1, len(view_names),
            figsize=(4.2 * len(view_names), 4.0),
            squeeze=False
        )
        plot_jnd_comparison_per_view(
            fig=fig_jnd,
            axes=list(axes[0]),
            view_names=view_names,
            jnd_indiv_by_view=jnd_indiv_by_view,
            jnd_overlay=jnd_overlay,
            style=style,
        )
        fig_jnd.tight_layout()
        figs["jnd_comparison"] = fig_jnd
    elif layout == "abls_4x3":
        from GroupComparison.layouts import plot_abls_4x3

        fig_main = plot_abls_4x3(
            prepared,
            views,
            cfg,
            style,
            overlay,
            view_colors=view_colors,
            group_jnd_by_view=group_jnd_by_view,
            add_inset=True,
            view_styles=view_styles,
        )
        figs = {"main": fig_main}
    elif layout == "animals_plus_mean":
        figs = {"animals_plus_mean": {}}
        for v in views:
            fig_main = plot_psychometric_animals_plus_average(
                tables=prepared[v.name],
                view_name=v.name,
                cfg=cfg,
                style=style,
            )
            figs["animals_plus_mean"][v.name] = fig_main
            if show:
                plt.show()
    elif layout == "psy_params":
        fig_main = plot_psychometric_params_all_views(
            prepared=prepared,
            views=views,
            cfg=cfg,
            style=style,
            overlay_data=overlay.makefig1_data,
        )
        figs = {"psy_params": fig_main}
    elif layout == "summary_metrics":
        fig_main = plot_summary_metrics_all_views(
            df_filtered=bundle["df_filtered"],
            prepared=prepared,
            views=views,
            cfg=cfg,
            style=style,
            overlay=overlay,
            jnd_indiv_by_view=jnd_indiv_by_view,
            jnd_overlay=jnd_overlay,
            mode="core",
        )
        figs = {"summary_metrics": fig_main}
    elif layout == "summary_aborts":
        fig_main = plot_summary_metrics_all_views(
            df_filtered=bundle["df_filtered"],
            prepared=prepared,
            views=views,
            cfg=cfg,
            style=style,
            overlay=overlay,
            jnd_indiv_by_view=jnd_indiv_by_view,
            jnd_overlay=jnd_overlay,
            mode="aborts",
        )
        figs = {"summary_aborts": fig_main}
    else:
        raise ValueError(
            f"Unknown layout='{layout}'. Use 'views_3x3', 'abls_4x3', 'animals_plus_mean', 'psy_params', 'summary_metrics', or 'summary_aborts'."
        )

    if show:
        plt.show()

    return {
        "prepared": prepared,
        "jnd_indiv_by_view": jnd_indiv_by_view,
        "group_jnd_by_view": group_jnd_by_view,
        "figures": figs,
        "view_summary": summarize_views(bundle["df_filtered"], views),
        "view_colors": view_colors,
        "view_styles": view_styles,
    }
