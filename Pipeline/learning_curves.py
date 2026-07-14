from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import Helpers.DataHelpers as DataHelpers
from GroupComparison.config import PlotStyle, ViewSpec
from Pipeline.group_comparison import build_view_style_maps


ROOT = Path(__file__).resolve().parents[1]
SUBJECT_COL = "animal"
TRIAL_COL = "trial"
SESSION_COL = "session"
SUCCESS_COL = "success"
LEVEL_COL = "training_level"
TIME_COL = "tared_trial_start"


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


def build_view_colors(df: pd.DataFrame, views: list[ViewSpec]) -> dict[str, str]:
    colors, _ = build_view_style_maps(df, views)
    return colors


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


def default_style() -> PlotStyle:
    return PlotStyle(title_fs=22, label_fs=18, tick_fs=14, legend_fs=14, title_pad=12)


def _prepare_base_df(
    df: pd.DataFrame,
    *,
    drop_repeat_trials: bool = True,
    valid_success_values: tuple[int, ...] = (1, -1),
) -> pd.DataFrame:
    out = DataHelpers.prepare_data(df.copy(), session_col=SESSION_COL, trial_col=TRIAL_COL)
    if drop_repeat_trials and "trial_is_repeat" in out.columns:
        out = out[out["trial_is_repeat"] == False].copy()
    if SUCCESS_COL in out.columns:
        out = out[out[SUCCESS_COL].isin(valid_success_values)].copy()
    for col in [SUBJECT_COL, SESSION_COL, TRIAL_COL, LEVEL_COL, TIME_COL]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") if col != SUBJECT_COL else out[col].astype(str).str.strip()
    return out


def _compute_session_curves(
    df_view: pd.DataFrame,
    *,
    span: int = 25,
    normalized_points: int = 100,
) -> dict[str, Any]:
    if df_view.empty:
        return {"sessions": {}}

    df_view = df_view.copy()
    df_view["is_correct"] = (pd.to_numeric(df_view[SUCCESS_COL], errors="coerce") == 1).astype(int)
    df_view = df_view.sort_values([SESSION_COL, SUBJECT_COL, TRIAL_COL])

    def ewm_smooth(series: pd.Series) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    output: dict[str, Any] = {"sessions": {}}
    sessions = sorted(pd.to_numeric(df_view[SESSION_COL], errors="coerce").dropna().astype(int).unique())
    for sess in sessions:
        df_sess = df_view[pd.to_numeric(df_view[SESSION_COL], errors="coerce").eq(sess)].copy()
        curves = []
        tl_curves = []
        tl_by_subject: dict[str, np.ndarray] = {}
        curve_by_subject: dict[str, np.ndarray] = {}
        subjects_used: list[str] = []

        for subject in sorted(df_sess[SUBJECT_COL].dropna().astype(str).unique()):
            sub = df_sess[df_sess[SUBJECT_COL].astype(str) == subject].sort_values(TRIAL_COL).copy()
            if len(sub) < 2:
                continue
            smooth = ewm_smooth(sub["is_correct"]).to_numpy(dtype=float)
            levels = pd.to_numeric(sub[LEVEL_COL], errors="coerce").to_numpy(dtype=float)
            if len(smooth) < 2 or not np.isfinite(levels).any():
                continue

            x_raw = np.linspace(0, 1, len(smooth))
            x_target = np.linspace(0, 1, normalized_points)
            smooth_resampled = np.interp(x_target, x_raw, smooth)
            level_resampled = np.interp(x_target, x_raw, levels)

            subjects_used.append(subject)
            curves.append(smooth_resampled)
            tl_curves.append(level_resampled)
            curve_by_subject[subject] = smooth_resampled
            tl_by_subject[subject] = level_resampled

        if not curves:
            continue

        curve_mat = np.vstack(curves)
        tl_mat = np.vstack(tl_curves)
        output["sessions"][int(sess)] = {
            "subjects": subjects_used,
            "n": len(subjects_used),
            "length": int(normalized_points),
            "mean": curve_mat.mean(axis=0),
            "sem": curve_mat.std(axis=0, ddof=0) / np.sqrt(curve_mat.shape[0]),
            "curves": curve_by_subject,
            "training_level_by_subject": tl_by_subject,
            "training_level_mean": tl_mat.mean(axis=0),
            "training_level_var": tl_mat.var(axis=0),
        }

    return output


def _compute_time_in_level(
    df_view: pd.DataFrame,
    *,
    level_max_exclusive: int = 16,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    if df_view.empty:
        return pd.DataFrame()

    out = df_view.copy()
    out = out[pd.to_numeric(out[LEVEL_COL], errors="coerce").lt(level_max_exclusive)].copy()
    if out.empty:
        return pd.DataFrame()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.dropna(subset=[LEVEL_COL, SESSION_COL, time_col]).copy()
    out = out.sort_values([SUBJECT_COL, SESSION_COL, time_col])

    def _session_level_duration(d: pd.DataFrame) -> pd.Series:
        t0 = float(d[time_col].iloc[0])
        t1 = float(d[time_col].iloc[-1])
        return pd.Series(
            {
                "session_duration_sec": float(t1 - t0),
                "n_trials": len(d),
            }
        )

    per_session = (
        out.groupby([SUBJECT_COL, LEVEL_COL, SESSION_COL], dropna=False)
        .apply(_session_level_duration)
        .reset_index()
    )
    per_level = (
        per_session.groupby([SUBJECT_COL, LEVEL_COL], dropna=False)
        .agg(
            total_time_sec=("session_duration_sec", "sum"),
            mean_session_time_sec=("session_duration_sec", "mean"),
            n_sessions=("session_duration_sec", "size"),
            total_trials=("n_trials", "sum"),
        )
        .reset_index()
    )
    per_level["total_time_min"] = per_level["total_time_sec"] / 60.0
    per_level["total_time_hour"] = per_level["total_time_sec"] / 3600.0
    return per_level


def prepare_learning_curve_bundle(
    *,
    df: pd.DataFrame,
    views: list[ViewSpec],
    span: int = 25,
    normalized_points: int = 100,
    drop_repeat_trials: bool = True,
    valid_success_values: tuple[int, ...] = (1, -1),
    time_col: str = TIME_COL,
    level_max_exclusive: int = 16,
    style: PlotStyle | None = None,
    view_labels: dict[str, str] | None = None,
    view_colors: dict[str, str] | None = None,
) -> dict[str, Any]:
    style = style or default_style()
    base_df = _prepare_base_df(
        df,
        drop_repeat_trials=drop_repeat_trials,
        valid_success_values=valid_success_values,
    )
    view_labels = view_labels or build_view_labels(views)
    view_colors = view_colors or build_view_colors(base_df, views)

    curve_results: dict[str, dict[str, Any]] = {}
    time_in_level: dict[str, pd.DataFrame] = {}
    training_level_means: list[np.ndarray] = []
    for view in views:
        df_view = view.selector(base_df).copy()
        curves = _compute_session_curves(df_view, span=span, normalized_points=normalized_points)
        curve_results[view.name] = curves
        for sess_info in curves["sessions"].values():
            training_level_means.append(np.asarray(sess_info["training_level_mean"], dtype=float))
        time_in_level[view.name] = _compute_time_in_level(
            df_view,
            level_max_exclusive=level_max_exclusive,
            time_col=time_col,
        )

    if training_level_means:
        concat_tl = np.concatenate(training_level_means)
        tl_range = (
            float(np.nanmin(concat_tl)),
            float(np.nanmax(concat_tl)),
        )
    else:
        tl_range = (0.0, 16.0)

    return {
        "df": base_df,
        "views": views,
        "curve_results": curve_results,
        "time_in_level": time_in_level,
        "style": style,
        "view_labels": view_labels,
        "view_colors": view_colors,
        "training_level_range": tl_range,
        "span": span,
        "normalized_points": normalized_points,
        "time_col": time_col,
        "level_max_exclusive": level_max_exclusive,
    }


def plot_training_level_colored_curves(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    show: bool = True,
    ylim: tuple[float, float] = (0.45, 1.0),
) -> plt.Figure | None:
    views = views or bundle["views"]
    style: PlotStyle = bundle["style"]
    tl_min, tl_max = bundle["training_level_range"]
    if not views:
        return None

    fig, axes = plt.subplots(len(views), 1, figsize=(13, max(3.5, 3.6 * len(views))), sharey=True)
    if len(views) == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("tab20")

    for ax, view in zip(axes, views):
        res = bundle["curve_results"].get(view.name, {"sessions": {}})
        sessions = sorted(res.get("sessions", {}).keys())
        if not sessions:
            ax.text(0.5, 0.5, f"{bundle['view_labels'].get(view.name, view.name)} - no data", ha="center", va="center")
            ax.axis("off")
            continue

        x_all = []
        mean_all = []
        sem_all = []
        tl_all = []
        subject_payload: dict[str, dict[str, list[np.ndarray]]] = {}
        offset = 0.0

        for sess in sessions:
            info = res["sessions"][sess]
            n = int(info["length"])
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            x_all.append(x_seg)
            mean_all.append(np.asarray(info["mean"], dtype=float))
            sem_all.append(np.asarray(info["sem"], dtype=float))
            tl_all.append(np.asarray(info["training_level_mean"], dtype=float))

            for subject in info["subjects"]:
                curve = np.asarray(info["curves"][subject], dtype=float)
                levels = np.asarray(info["training_level_by_subject"][subject], dtype=float)
                entry = subject_payload.setdefault(subject, {"x": [], "y": [], "level": []})
                entry["x"].append(x_seg)
                entry["y"].append(curve)
                entry["level"].append(levels)

        x = np.concatenate(x_all)
        mean = np.concatenate(mean_all)
        sem_curve = np.concatenate(sem_all)
        tl_mean = np.concatenate(tl_all)
        tl_norm = (tl_mean - tl_min) / max(1e-6, (tl_max - tl_min))

        for subject_data in subject_payload.values():
            xs = np.concatenate(subject_data["x"])
            ys = np.concatenate(subject_data["y"])
            levs = np.concatenate(subject_data["level"])
            lev_norm = (levs - tl_min) / max(1e-6, (tl_max - tl_min))
            for i in range(len(xs) - 1):
                ax.plot(xs[i:i + 2], ys[i:i + 2], color=cmap(lev_norm[i]), linewidth=0.7, alpha=0.45)

        ax.fill_between(x, mean - sem_curve, mean + sem_curve, color="lightgray", alpha=0.18, linewidth=0)
        for i in range(len(x) - 1):
            ax.plot(x[i:i + 2], mean[i:i + 2], color=cmap(tl_norm[i]), linewidth=2.0)

        for sess_i in range(len(sessions)):
            ax.axvline(sess_i, linestyle="--", color="0.6", alpha=0.3, linewidth=1.0)

        ax.set_title(f"{bundle['view_labels'].get(view.name, view.name)} - learning curve by training level", fontsize=style.title_fs, pad=style.title_pad)
        ax.set_xlabel("Normalized session progress (0-1 per session)", fontsize=style.label_fs)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="both", labelsize=style.tick_fs)
        ax.grid(alpha=0.25)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Accuracy (EWMA)", fontsize=style.label_fs)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=tl_min, vmax=tl_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.66])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Training level", fontsize=style.legend_fs)
    cbar.ax.tick_params(labelsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 0.90, 0.98])
    if show:
        plt.show()
    return fig


def _collect_time_level_rows(
    bundle: dict[str, Any],
    views: list[ViewSpec],
    *,
    time_unit: str = "hour",
) -> tuple[pd.DataFrame, str]:
    time_col = {"sec": "total_time_sec", "min": "total_time_min", "hour": "total_time_hour"}[time_unit]
    rows = []
    for view in views:
        sub = bundle["time_in_level"].get(view.name)
        if sub is None or sub.empty:
            continue
        tmp = sub.copy()
        tmp["view"] = view.name
        tmp["view_label"] = bundle["view_labels"].get(view.name, view.name)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(), time_col
    out = pd.concat(rows, ignore_index=True, sort=False)
    out[LEVEL_COL] = pd.to_numeric(out[LEVEL_COL], errors="coerce").astype("Int64")
    return out, time_col


def plot_time_in_level_scatter(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    time_unit: str = "hour",
    show: bool = True,
) -> plt.Figure | None:
    views = views or bundle["views"]
    style: PlotStyle = bundle["style"]
    df_plot, time_col = _collect_time_level_rows(bundle, views, time_unit=time_unit)
    if df_plot.empty:
        return None

    levels = sorted(df_plot[LEVEL_COL].dropna().astype(int).unique())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    rng = np.random.default_rng(0)
    for view in views:
        sub = df_plot[df_plot["view"] == view.name].copy()
        if sub.empty:
            continue
        x = sub[LEVEL_COL].astype(float).to_numpy()
        y = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(
            x + jitter,
            y,
            label=bundle["view_labels"].get(view.name, view.name),
            color=bundle["view_colors"].get(view.name, "0.5"),
            alpha=0.7,
            edgecolors="k",
            linewidths=0.4,
            s=42,
        )

    ax.set_xticks(levels)
    ax.set_xlabel("Training level", fontsize=style.label_fs)
    unit_label = {"sec": "seconds", "min": "minutes", "hour": "hours"}[time_unit]
    ax.set_ylabel(f"Total time at level ({unit_label})", fontsize=style.label_fs)
    ax.set_title("Time in each training level (per animal)", fontsize=style.title_fs, pad=style.title_pad)
    ax.tick_params(axis="both", labelsize=style.tick_fs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=style.legend_fs, frameon=False)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_time_in_level_boxplot(
    bundle: dict[str, Any],
    *,
    views: list[ViewSpec] | None = None,
    time_unit: str = "hour",
    show: bool = True,
) -> plt.Figure | None:
    views = views or bundle["views"]
    style: PlotStyle = bundle["style"]
    df_plot, time_col = _collect_time_level_rows(bundle, views, time_unit=time_unit)
    if df_plot.empty:
        return None

    levels = sorted(df_plot[LEVEL_COL].dropna().astype(int).unique())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_views = max(1, len(views))
    cluster_width = 0.7
    step = cluster_width / n_views

    for view_i, view in enumerate(views):
        color = bundle["view_colors"].get(view.name, f"C{view_i % 10}")
        data = []
        positions = []
        for level_i, level in enumerate(levels):
            vals = df_plot.loc[
                (df_plot["view"] == view.name) & (df_plot[LEVEL_COL] == level),
                time_col,
            ].dropna()
            data.append(vals.to_numpy(dtype=float) if len(vals) else [])
            positions.append(level_i + (view_i - (n_views - 1) / 2) * step)

        ax.boxplot(
            data,
            positions=positions,
            widths=step * 0.85,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=color, alpha=0.95, edgecolor=color),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=4.5,
                linestyle="none",
                alpha=0.85,
            ),
        )

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlabel("Training level", fontsize=style.label_fs)
    unit_label = {"sec": "seconds", "min": "minutes", "hour": "hours"}[time_unit]
    ax.set_ylabel(f"Total time at level ({unit_label})", fontsize=style.label_fs)
    ax.set_title("Time spent at each training level", fontsize=style.title_fs, pad=style.title_pad)
    ax.tick_params(axis="both", labelsize=style.tick_fs)
    ax.grid(axis="y", alpha=0.3)
    handles = [
        Line2D([0], [0], color=bundle["view_colors"].get(view.name, f"C{i % 10}"), lw=4, label=bundle["view_labels"].get(view.name, view.name))
        for i, view in enumerate(views)
    ]
    ax.legend(handles=handles, fontsize=style.legend_fs, frameon=False)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_learning_curve_figures(
    *,
    bundle: dict[str, Any],
    views: list[ViewSpec] | None = None,
    plot_mode: str = "training_level_colormap",
    time_unit: str = "hour",
    show: bool = True,
) -> dict[str, Any]:
    views = views or bundle["views"]
    figures: dict[str, Any] = {}
    mode = plot_mode.lower()

    if mode in {"training_level_colormap", "colorbar", "curves"}:
        figures["training_level_colormap"] = plot_training_level_colored_curves(bundle, views=views, show=show)
    elif mode in {"time_in_level_scatter", "scatter"}:
        figures["time_in_level_scatter"] = plot_time_in_level_scatter(bundle, views=views, time_unit=time_unit, show=show)
    elif mode in {"time_in_level_boxplot", "boxplot", "box"}:
        figures["time_in_level_boxplot"] = plot_time_in_level_boxplot(bundle, views=views, time_unit=time_unit, show=show)
    elif mode == "all":
        figures["training_level_colormap"] = plot_training_level_colored_curves(bundle, views=views, show=show)
        figures["time_in_level_scatter"] = plot_time_in_level_scatter(bundle, views=views, time_unit=time_unit, show=show)
        figures["time_in_level_boxplot"] = plot_time_in_level_boxplot(bundle, views=views, time_unit=time_unit, show=show)
    else:
        raise ValueError("plot_mode must be one of: training_level_colormap, time_in_level_scatter, time_in_level_boxplot, all.")

    return {"figures": figures}

