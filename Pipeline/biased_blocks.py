from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import Helpers.DataHelpers as DataHelpers
import Psychometric
from GroupComparison.config import FilterConfig, GroupComparisonConfig, OverlaySpec, PlotStyle, ViewSpec
from GroupComparison.layouts import plot_abls_4x3
from GroupComparison.plots import apply_50_tick_labels
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


def classify_block_direction(
    block_df: pd.DataFrame,
    *,
    rightward_ild_sign: int = 1,
    min_imbalance: float = 0.0,
) -> str | pd._libs.missing.NAType:
    ild = pd.to_numeric(block_df["ILD"], errors="coerce")
    ild = ild[ild.notna() & ild.ne(0)]
    if ild.empty:
        return pd.NA

    signed_imbalance = np.sign(ild).mean()
    if not np.isfinite(signed_imbalance) or abs(signed_imbalance) <= min_imbalance:
        return pd.NA

    block_sign = 1 if signed_imbalance > 0 else -1
    return "rightward" if block_sign == int(rightward_ild_sign) else "leftward"


def add_biased_block_condition(
    df: pd.DataFrame,
    *,
    biased_session_types: tuple[int, ...] = BIASED_SESSION_TYPES,
    unbiased_rt_session_types: tuple[int, ...] = UNBIASED_RT_SESSION_TYPES,
    short_duration_value: Any = SHORT_DURATION_VALUE,
    rightward_ild_sign: int = 1,
    min_direction_imbalance: float = 0.0,
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

        if "block" in session_df.columns and session_df["block"].notna().any():
            block_values = sorted(pd.to_numeric(session_df["block"], errors="coerce").dropna().unique())
            block_series = pd.to_numeric(df.loc[session_df.index, "block"], errors="coerce")
        else:
            block_values = [1]
            block_series = pd.Series(1, index=session_df.index)

        if not block_values:
            continue

        first_block = block_values[0]
        first_idx = session_df.index[block_series.eq(first_block)]
        df.loc[first_idx, "block_condition"] = "unbiased"

        for block_value in block_values[1:]:
            block_idx = session_df.index[block_series.eq(block_value)]
            direction = classify_block_direction(
                df.loc[block_idx],
                rightward_ild_sign=rightward_ild_sign,
                min_imbalance=min_direction_imbalance,
            )
            if pd.notna(direction):
                df.loc[block_idx, "block_condition"] = direction

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
    keep_only_animals_with_biased_sessions: bool = True,
) -> dict[str, Any]:
    cfg = cfg or default_config()
    fcfg = fcfg or default_filter_config()
    style = style or default_style()

    prefilter_session_types = set(unbiased_rt_session_types) | set(biased_session_types)
    prefilter_sess = _numeric_col(df, "session_type")
    prefilter_short = _numeric_col(df, "short_duration")
    short_mask = _match_numeric_filter(prefilter_short, short_duration_value)
    df_prefiltered = df[
        prefilter_sess.isin(prefilter_session_types)
        & short_mask
    ].copy()

    if keep_only_animals_with_biased_sessions and "animal" in df_prefiltered.columns:
        biased_animals = set(
            df_prefiltered.loc[prefilter_sess.loc[df_prefiltered.index].isin(biased_session_types), "animal"]
            .dropna()
            .astype(str)
        )
        df_prefiltered = df_prefiltered[df_prefiltered["animal"].astype(str).isin(biased_animals)].copy()

    df_filtered = apply_filters(df_prefiltered, fcfg)
    df_blocks = add_biased_block_condition(
        df_filtered,
        biased_session_types=biased_session_types,
        unbiased_rt_session_types=unbiased_rt_session_types,
        short_duration_value=short_duration_value,
        rightward_ild_sign=rightward_ild_sign,
        min_direction_imbalance=min_direction_imbalance,
    )

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
        "block_summary": block_summary,
        "views": views,
        "cfg": cfg,
        "fcfg": fcfg,
        "style": style,
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
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for view in selected_views:
        df_v = view.selector(df)
        rt_per_subj, rt_group = prep_rt_signed(df_v)
        mt_per_subj, mt_group = prep_mt(df_v)
        with contextlib.redirect_stdout(io.StringIO()):
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
        ax.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax)


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


def plot_genotype_block_figures(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    df_blocks = bundle["df_blocks"]
    cfg = bundle["cfg"]
    style = bundle["style"]
    views = views or bundle["views"]

    outputs = {}
    for view in views:
        df_view = view.selector(df_blocks)
        if df_view.empty:
            continue
        block_views = make_block_views(df_view)
        prepared = build_prepared_signed_rt(df_view, block_views, cfg)
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
        prepared = build_prepared_signed_rt(df_animal, block_views, cfg)
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

        prepared = build_prepared_signed_rt(df_condition, condition_views, cfg)
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
    rows = []

    for condition in block_conditions:
        df_condition = df_blocks[df_blocks["block_condition"].astype(str) == condition].copy()
        if df_condition.empty:
            continue
        condition_views = [v for v in views if not v.selector(df_condition).empty]
        if not condition_views:
            continue

        prepared = build_prepared_signed_rt(df_condition, condition_views, cfg)
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


def _transition_window_rows(
    df_blocks: pd.DataFrame,
    *,
    window: int = 20,
    from_condition: str = "leftward",
    to_condition: str = "rightward",
) -> pd.DataFrame:
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

            prev_rows = session_df[session_df["_block_num"] == prev_block].sort_values("trial").tail(window).copy()
            next_rows = session_df[session_df["_block_num"] == next_block].sort_values("trial").head(window).copy()
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
    ild = pd.to_numeric(out["ILD"], errors="coerce")
    out["signed_ild_group"] = np.select(
        [
            ild.isin([-2, -1]),
            ild.isin([1, 2]),
            ild.isin([-16, -8]),
            ild.isin([8, 16]),
        ],
        ["hard left", "hard right", "easy left", "easy right"],
        default=pd.NA,
    )
    out = out[out["signed_ild_group"].notna() & out["choice_right"].notna()].copy()
    out["ABL"] = pd.to_numeric(out["ABL"], errors="coerce")
    return out


def _aligned_biased_transition_window_rows(
    df_blocks: pd.DataFrame,
    *,
    window: int = 20,
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

            prev_rows = session_df[session_df["_block_num"] == prev_block].sort_values("trial").tail(window).copy()
            next_rows = session_df[session_df["_block_num"] == next_block].sort_values("trial").head(window).copy()
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
) -> pd.DataFrame:
    """Collect any biased->biased transition, align choices to the new side, and collapse ILDs by difficulty."""
    out = _aligned_biased_transition_window_rows(df_blocks, window=window)
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


def _add_effective_slope(params: pd.DataFrame) -> pd.DataFrame:
    out = params.copy()
    needed = {"slope_a", "lower_c", "upper_d"}
    if not needed.issubset(out.columns):
        return out
    slope_a = pd.to_numeric(out["slope_a"], errors="coerce")
    lower_c = pd.to_numeric(out["lower_c"], errors="coerce")
    upper_d = pd.to_numeric(out["upper_d"], errors="coerce")
    out["effective_slope"] = 0.5 * (upper_d - lower_c) * slope_a
    return out


def _find_suspicious_slope_cases(
    params: pd.DataFrame,
    *,
    condition_col: str,
) -> pd.DataFrame:
    metric_col = "effective_slope" if "effective_slope" in params.columns else "slope_a"
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
) -> dict[str, Any] | None:
    if df_sub.empty:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
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
                pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                    ilds,
                    mean_vals,
                    model="my_psycho",
                    n_trials=n_trials,
                    show_plot=False,
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


def _displayed_curve_slope(xx, yy) -> float | None:
    if xx is None or yy is None:
        return None
    x = np.asarray(xx, dtype=float)
    y = np.asarray(yy, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        return None
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    target = ymin + 0.5 * (ymax - ymin)
    dydx = np.gradient(y, x)
    idx = int(np.argmin(np.abs(y - target)))
    if idx < 0 or idx >= len(dydx) or not np.isfinite(dydx[idx]):
        return None
    return float(dydx[idx])


def _compute_group_curve_slopes(
    history_df: pd.DataFrame,
    *,
    cfg: GroupComparisonConfig,
    condition_col: str,
    robust_mean: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if history_df.empty or condition_col not in history_df.columns:
        return pd.DataFrame()

    for (condition_value, abl, duration_pair), sub in history_df.groupby(
        [condition_col, "ABL", "duration_pair"], dropna=False, sort=False
    ):
        curve = _compute_psy_curve_for_subset(sub, cfg=cfg, robust_mean=robust_mean)
        if curve is None:
            continue
        fit = curve["psy_mean_fits"].get(int(abl)) if curve["psy_mean_fits"] is not None else None
        if fit is None:
            continue
        shown_slope = _displayed_curve_slope(fit["xx"], fit["yy"])
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
        metric_col = row["slope_metric"] if "slope_metric" in row and pd.notna(row["slope_metric"]) else ("effective_slope" if "effective_slope" in params.columns else "slope_a")
        fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.8))
        plotted_any = False
        slope_pairs = []
        for duration_pair in duration_order:
            sub = history_df[
                (history_df[condition_col].astype(str) == str(condition_value))
                & (pd.to_numeric(history_df["ABL"], errors="coerce").eq(abl))
                & (history_df["duration_pair"].astype(str) == duration_pair)
            ].copy()
            curve = _compute_psy_curve_for_subset(sub, cfg=cfg)
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
                slope_summary = float(np.median(slope_vals.to_numpy(dtype=float)))
                slope_pairs.append((duration_pair, slope_summary))
                slope_rows.append(
                    {
                        "condition_type": condition_col,
                        "condition": condition_value,
                        "ABL": abl,
                        "duration_pair": duration_pair,
                        "slope_metric": metric_col,
                        "slope_summary": slope_summary,
                        "slope_summary_type": "median",
                        "n_animals_with_slope": int(len(slope_vals)),
                        "used_trimmed_mean": pd.NA,
                    }
                )
            psy_group = curve["psy_group"]
            fit = curve["psy_mean_fits"].get(abl) if curve["psy_mean_fits"] is not None else None
            color = duration_colors.get(duration_pair, "0.4")
            marker = duration_markers.get(duration_pair, "o")
            shown_slope = _displayed_curve_slope(fit["xx"], fit["yy"]) if fit is not None else None
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
                    lines.append(f"{name}: shown={shown_vals.iloc[0]:.3f}  median={value:.3f}")
                else:
                    lines.append(f"{name}: median={value:.3f}")
            slope_text = "\n".join(lines)
            ax.text(
                0.02,
                0.98,
                f"slope\n{slope_text}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=max(9, fs - 2),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.9),
            )
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
        fig.suptitle(f"{title_prefix} - slope check", fontsize=fs, y=0.98)
        fig.tight_layout()
        figures[f"{condition_value}_ABL_{abl}"] = fig

    return figures, pd.DataFrame(slope_rows)


def _history_transition_rows(
    df_blocks: pd.DataFrame,
    *,
    duration_groups: tuple[tuple[int, ...], ...] = ((8, 16), (32, 64), (120, 0)),
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
    short_duration = pd.to_numeric(out.get("short_duration"), errors="coerce")
    out["duration_pair"] = short_duration.map(lambda x: _duration_pair_label(x, duration_groups))
    out = out[out["duration_pair"].notna()].copy()
    out["ABL"] = pd.to_numeric(out["ABL"], errors="coerce")
    out = out[out["ABL"].notna()].copy()
    out["ABL"] = out["ABL"].astype(int)
    out = out.sort_values(["animal", "session", "block_id", "trial"]).copy()
    out["valid_trial_in_block"] = out.groupby("block_id", dropna=False).cumcount() + 1
    out["n_valid_in_block"] = out.groupby("block_id", dropna=False)["valid_trial_in_block"].transform("max")
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
) -> pd.DataFrame:
    rows = []
    for (transition_type, duration_pair), sub in history_df.groupby(["transition_type", "duration_pair"], dropna=False, sort=False):
        if sub.empty:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
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
    params = _add_effective_slope(params)

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
        ("effective_slope", "Slope"),
        ("JND", "JND"),
    ]
    fs = style.legend_fs
    figs = {}
    rng = np.random.default_rng(1)
    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()
    global_ylims = _global_param_ylim(params, [col for col, _ in specs])
    global_ylims.update(_full_param_ylim(params, ["effective_slope"]))

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
                        edgecolors="black" if col == "effective_slope" else "none",
                        linewidths=0.25 if col == "effective_slope" else 0.0,
                        zorder=3,
                    )
                    if col == "effective_slope":
                        mean, lower_err, upper_err = _median_and_iqr_errors(sub[col].to_numpy(dtype=float))
                        err = np.asarray([[lower_err], [upper_err]], dtype=float)
                    else:
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
) -> pd.DataFrame:
    rows = []
    for (block_name, duration_pair), sub in history_df.groupby(["current_block_condition", "duration_pair"], dropna=False, sort=False):
        if sub.empty:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
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
    params = _add_effective_slope(params)

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
    specs = [("bias_b", "Bias (b)"), ("effective_slope", "Slope"), ("JND", "JND")]
    fs = style.legend_fs
    figs = {}
    rng = np.random.default_rng(3)
    duration_offsets = _duration_pair_offsets(duration_order)
    duration_markers = _duration_pair_markers()
    global_ylims = _global_param_ylim(params, [col for col, _ in specs])
    global_ylims.update(_full_param_ylim(params, ["effective_slope"]))

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
                        edgecolors="black" if col == "effective_slope" else "none",
                        linewidths=0.25 if col == "effective_slope" else 0.0,
                        zorder=3,
                    )
                    if col == "effective_slope":
                        mean, lower_err, upper_err = _median_and_iqr_errors(sub[col].to_numpy(dtype=float))
                        err = np.asarray([[lower_err], [upper_err]], dtype=float)
                    else:
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
    style = bundle["style"]
    cfg = bundle["cfg"]
    views = views or bundle["views"]
    outputs = {}

    for view in views:
        df_view = view.selector(df_blocks)
        history_df = _history_transition_rows(df_view, duration_groups=duration_groups)
        if history_df.empty:
            continue

        performance_fig = _plot_history_performance(
            history_df,
            title=f"{view.name} - performance by previous/current block",
            style=style,
            abls=abls,
        )
        params = _compute_history_psy_params(history_df, cfg=cfg)
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
        current_params = _compute_current_block_psy_params(history_df, cfg=cfg)
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
        if df_view.empty:
            continue
        direction_specs = [
            ("leftward", "rightward", "leftward to rightward"),
            ("rightward", "leftward", "rightward to leftward"),
        ]
        direction_payloads = []
        for from_condition, to_condition, label in direction_specs:
            window_df = _transition_window_rows(
                df_view,
                window=window,
                from_condition=from_condition,
                to_condition=to_condition,
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
        if df_view.empty:
            continue
        window_df = _aligned_biased_transition_window_rows(df_view, window=window)
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
    style = bundle["style"]
    views = views or bundle["views"]
    fs = style.legend_fs
    outputs = {}

    difficulty_order = ["hard", "easy"]
    difficulty_colors = {"hard": "#6A3D9A", "easy": "#E69F00"}
    difficulty_markers = {"hard": "o", "easy": "s"}

    for view in views:
        df_view = view.selector(df_blocks)
        if df_view.empty:
            continue
        window_df = _collapsed_biased_transition_window_rows(df_view, window=window)
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
    else:
        raise ValueError(
            "layout must be one of: genotype_blocks, block_conditions, block_condition_params, block_bias, left_to_right_transition, biased_transition_aligned, biased_transition_collapsed, animal_blocks, all."
        )

    return {"figures": figures}


def save_biased_block_figures(figures: dict[str, Any], out_dir, prefix: str = "biased_blocks") -> None:
    out_dir = pd.io.common.stringify_path(out_dir)
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _save(obj: dict[str, Any], folder: Path, name_prefix: str) -> None:
        for name, value in obj.items():
            if isinstance(value, dict) and "figure" in value:
                value["figure"].savefig(folder / f"{name_prefix}_{_safe_name(name)}.png", dpi=250, bbox_inches="tight")
            elif isinstance(value, dict):
                subdir = folder / _safe_name(name)
                subdir.mkdir(parents=True, exist_ok=True)
                _save(value, subdir, name_prefix)

    _save(figures, out_path, prefix)
