from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import Helpers.DataHelpers as DataHelpers
from DailyMerge import get_base_dir


def subject_id_from_file(subject_file: str) -> str:
    return Path(subject_file).stem.removeprefix("merged_")


def load_across_sessions_data(
    *,
    subject_file: str,
    line: str,
    cohort: str,
) -> tuple[pd.DataFrame, Path]:
    data_dir = Path(get_base_dir(line, cohort))
    path = data_dir / subject_file
    if not path.exists():
        raise FileNotFoundError(f"Subject file not found: {path}")
    return pd.read_csv(path), data_dir


def prepare_across_sessions_data(
    df: pd.DataFrame,
    *,
    training_level: int | None = 16,
    training_level_max: int | None = None,
) -> dict[str, Any]:
    df = DataHelpers.prepare_data(df.copy(), session_col="session", trial_col="trial")
    if training_level_max is not None:
        df = df[pd.to_numeric(df["training_level"], errors="coerce") < training_level_max].copy()
    elif training_level is not None:
        df = df[df["training_level"] == training_level].copy()

    df_valid = df[df["trial_is_repeat"] == False].copy()

    mean_rt = (
        df_valid[(df_valid["success"] == 1) & (df_valid["timed_rt"] <= 1.2)]
        .groupby(["session", "ABL"])["timed_rt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    mean_mt = (
        df_valid[(df_valid["success"] == 1) & (df_valid["timed_mt"] <= 0.8)]
        .groupby(["session", "ABL"])["timed_mt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    mean_acc = (
        df_valid[df_valid["success"] != 0]
        .groupby(["session", "ABL"])["success"]
        .agg(accuracy=lambda x: (x == 1).mean(), n_trials="count")
        .reset_index()
    )

    mean_fa = (
        df[df["abort_type"] != "CNP"]
        .groupby(["session", "ABL"])["abort_type"]
        .agg(FArate=lambda x: (x == "Fixation").mean(), n_trials="count")
        .reset_index()
    )

    bias_summary = (
        df_valid
        .groupby(["session", "ABL"], observed=False)
        .apply(DataHelpers.compute_bias, include_groups=False)
        .pipe(lambda x: x if isinstance(x, pd.DataFrame) else x.to_frame("bias"))
        .reset_index()
    )

    mean_mta = (
        df[df["abort_type"] != "CNP"]
        .groupby(["session", "ABL"])["abort_type"]
        .agg(MTArate=lambda x: (x == "MT+").mean(), n_trials="count")
        .reset_index()
    )

    mean_rta = (
        df[df["abort_type"] != "CNP"]
        .groupby(["session", "ABL"])["abort_type"]
        .agg(RTArate=lambda x: (x == "RT-").mean(), n_trials="count")
        .reset_index()
    )

    df_reps = df_valid.sort_values(["session", "trial"]).copy()
    df_reps["prev_response"] = df_reps.groupby("session")["response_poke"].shift(1)
    df_reps["repetition"] = (df_reps["response_poke"] == df_reps["prev_response"]).astype(int)

    rep_split = (
        df_reps.groupby(["session", "prev_response", "ABL"])["repetition"]
        .mean()
        .reset_index()
        .pivot(index=["session", "ABL"], columns="prev_response", values="repetition")
        .rename(columns={-1: "after_left", 1: "after_right"})
        .reset_index()
    )

    rep_per_session = (
        df_reps.groupby(["session", "ABL"])["repetition"]
        .mean()
        .reset_index(name="repetition_rate")
    )

    trial_counts = {
        "completed": DataHelpers.count_trials(df_valid, df_valid["success"] != 0, "completed"),
        "cnp": DataHelpers.count_trials(df, df["abort_type"] == "CNP", "cnp"),
        "aborted": DataHelpers.count_trials(
            df,
            (df["abort_type"] != "CNP") & (df["success"] == 0),
            "aborted",
        ),
    }

    return {
        "df": df,
        "df_valid": df_valid,
        "mean_rt": mean_rt,
        "mean_mt": mean_mt,
        "mean_acc": mean_acc,
        "mean_fa": mean_fa,
        "mean_mta": mean_mta,
        "mean_rta": mean_rta,
        "bias_summary": bias_summary,
        "rep_split": rep_split,
        "rep_per_session": rep_per_session,
        "trial_counts": trial_counts,
    }


def load_change_regions(
    *,
    data_dir: Path,
    subject_id: str,
    change_csv: str = "change_points.csv",
    ax=None,
):
    path = data_dir / change_csv
    if not path.exists():
        return None

    if ax is None:
        _, ax = plt.subplots()
        close_fig = True
    else:
        close_fig = False

    regions = DataHelpers.shade_change_regions_from_csv(ax, str(path), subject_id)
    if close_fig:
        plt.close(ax.figure)
    return regions


def _draw_regions(ax, regions):
    if regions is not None:
        DataHelpers.draw_regions(ax, regions, alpha=1)


def plot_trial_counts(
    prepared: dict[str, Any],
    *,
    regions=None,
    figsize=(7, 4),
):
    trial_counts = prepared["trial_counts"]
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        trial_counts["completed"]["session"],
        trial_counts["completed"]["trial_count"],
        linestyle="-",
        marker="o",
        label="Completed Trials",
    )
    ax.plot(
        trial_counts["cnp"]["session"],
        trial_counts["cnp"]["trial_count"],
        linestyle="-",
        marker="o",
        label="CNP Aborts",
    )
    ax.plot(
        trial_counts["aborted"]["session"],
        trial_counts["aborted"]["trial_count"],
        linestyle="-",
        marker="o",
        label="Other Aborts",
    )

    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper center")
    ax.set_xlabel("Session")
    ax.set_ylabel("#trials")
    fig.tight_layout()
    return fig


def plot_across_sessions_summary(
    prepared: dict[str, Any],
    *,
    regions=None,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(28, 23),
):
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    metric_specs = [
        ("mean_rt", "mean", "std", "count", "Reaction Time (s)", "RT progression across sessions", "errorbar"),
        ("mean_mt", "mean", "std", "count", "Movement Time (s)", "MT progression across sessions", "errorbar"),
        ("mean_acc", "accuracy", None, None, "Accuracy (proportion correct)", "Accuracy progression across sessions", "plot"),
        ("mean_fa", "FArate", None, None, "Proportion of Fixation Aborts", "Fixation Aborts across sessions", "plot"),
        ("mean_mta", "MTArate", None, None, "Proportion of MT Aborts", "Movement Time Aborts across sessions", "plot"),
        ("mean_rta", "RTArate", None, None, "Proportion of RT Aborts", "Reaction Time Aborts across sessions", "plot"),
    ]

    for ax, (table_name, y_col, std_col, n_col, ylabel, title, kind) in zip(axes.flat[:6], metric_specs):
        table = prepared[table_name]
        for i, abl in enumerate(sorted(table["ABL"].unique())):
            sub = table[table["ABL"] == abl]
            if kind == "errorbar":
                ax.errorbar(
                    sub["session"],
                    sub[y_col],
                    yerr=sub[std_col] / sub[n_col] ** 0.5,
                    label=f"ABL {abl}",
                    color=colors[i % len(colors)],
                    linestyle="-",
                    marker="o",
                    capsize=5,
                )
            else:
                ax.plot(
                    sub["session"],
                    sub[y_col],
                    label=f"ABL {abl}",
                    color=colors[i % len(colors)],
                    linestyle="-",
                    marker="o",
                )
        _draw_regions(ax, regions)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Session")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0, 0].legend()
    axes[0, 2].set_ylim(0.5, 1)

    bias_summary = prepared["bias_summary"]
    ax = axes[2, 0]
    for i, abl in enumerate(sorted(bias_summary["ABL"].unique())):
        sub = bias_summary[bias_summary["ABL"] == abl]
        ax.errorbar(sub["session"], sub["bias"], color=colors[i % len(colors)], fmt="o-", capsize=4)
    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Session")
    ax.set_ylabel("Bias (mean response)")
    ax.set_title("Bias across sessions")

    rep_per_session = prepared["rep_per_session"]
    ax = axes[2, 1]
    for i, abl in enumerate(sorted(rep_per_session["ABL"].unique())):
        sub = rep_per_session[rep_per_session["ABL"] == abl]
        ax.plot(
            sub["session"],
            sub["repetition_rate"],
            label=f"ABL {abl}",
            color=colors[i % len(colors)],
            linestyle="-",
            marker="o",
        )
    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Session")
    ax.set_ylabel("Response repetition rate")
    ax.set_title("Response repetition across sessions")
    ax.set_ylim(0, 1)

    rep_split = prepared["rep_split"]
    ax = axes[2, 2]
    for i, abl in enumerate(sorted(rep_split["ABL"].unique())):
        sub = rep_split[rep_split["ABL"] == abl]
        color = colors[i % len(colors)]
        if "after_left" in sub:
            ax.plot(sub["session"], sub["after_left"], "<-", color=color)
        if "after_right" in sub:
            ax.plot(sub["session"], sub["after_right"], ">-", color=color)
    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Session")
    ax.set_ylabel("Repetition probability")
    ax.set_title("Response repetition split by previous choice")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


def plot_across_sessions_combined(
    prepared: dict[str, Any],
    *,
    regions=None,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(24, 14),
    title_fs=18,
    label_fs=16,
    tick_fs=13,
    legend_fs=13,
):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 2.0])

    ax_counts = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 3)
    axes = [[fig.add_subplot(right[r, c]) for c in range(3)] for r in range(3)]

    trial_counts = prepared["trial_counts"]
    ax_counts.plot(
        trial_counts["completed"]["session"],
        trial_counts["completed"]["trial_count"],
        linestyle="-",
        marker="o",
        label="Completed Trials",
    )
    ax_counts.plot(
        trial_counts["cnp"]["session"],
        trial_counts["cnp"]["trial_count"],
        linestyle="-",
        marker="o",
        label="CNP Aborts",
    )
    ax_counts.plot(
        trial_counts["aborted"]["session"],
        trial_counts["aborted"]["trial_count"],
        linestyle="-",
        marker="o",
        label="Other Aborts",
    )
    _draw_regions(ax_counts, regions)
    ax_counts.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_counts.legend(loc="upper center", fontsize=legend_fs)
    ax_counts.set_xlabel("Session", fontsize=label_fs)
    ax_counts.set_ylabel("#trials", fontsize=label_fs)
    ax_counts.set_title("Trial counts", fontsize=title_fs)
    ax_counts.tick_params(axis="both", labelsize=tick_fs)
    ax_counts.set_box_aspect(1)

    metric_specs = [
        ("mean_rt", "mean", "std", "count", "Reaction Time (s)", "RT", "errorbar"),
        ("mean_mt", "mean", "std", "count", "Movement Time (s)", "MT", "errorbar"),
        ("mean_acc", "accuracy", None, None, "Accuracy", "Accuracy", "plot"),
        ("mean_fa", "FArate", None, None, "Fixation Aborts", "Fixation aborts", "plot"),
        ("mean_mta", "MTArate", None, None, "MT Aborts", "MT aborts", "plot"),
        ("mean_rta", "RTArate", None, None, "RT Aborts", "RT aborts", "plot"),
    ]

    flat_axes = [ax for row in axes for ax in row]
    for ax, (table_name, y_col, std_col, n_col, ylabel, title, kind) in zip(flat_axes[:6], metric_specs):
        table = prepared[table_name]
        for i, abl in enumerate(sorted(table["ABL"].unique())):
            sub = table[table["ABL"] == abl]
            if kind == "errorbar":
                ax.errorbar(
                    sub["session"],
                    sub[y_col],
                    yerr=sub[std_col] / sub[n_col] ** 0.5,
                    label=f"ABL {abl}",
                    color=colors[i % len(colors)],
                    linestyle="-",
                    marker="o",
                    capsize=4,
                )
            else:
                ax.plot(
                    sub["session"],
                    sub[y_col],
                    color=colors[i % len(colors)],
                    linestyle="-",
                    marker="o",
                )
        _draw_regions(ax, regions)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Session", fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.set_title(title, fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

    axes[0][0].legend(fontsize=legend_fs)
    axes[0][2].set_ylim(0.5, 1)

    bias_summary = prepared["bias_summary"]
    ax = axes[2][0]
    for i, abl in enumerate(sorted(bias_summary["ABL"].unique())):
        sub = bias_summary[bias_summary["ABL"] == abl]
        ax.errorbar(sub["session"], sub["bias"], color=colors[i % len(colors)], fmt="o-", capsize=4)
    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Session", fontsize=label_fs)
    ax.set_ylabel("Bias", fontsize=label_fs)
    ax.set_title("Bias", fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    rep_per_session = prepared["rep_per_session"]
    ax = axes[2][1]
    for i, abl in enumerate(sorted(rep_per_session["ABL"].unique())):
        sub = rep_per_session[rep_per_session["ABL"] == abl]
        ax.plot(
            sub["session"],
            sub["repetition_rate"],
            color=colors[i % len(colors)],
            linestyle="-",
            marker="o",
        )
    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Session", fontsize=label_fs)
    ax.set_ylabel("Repetition rate", fontsize=label_fs)
    ax.set_title("Repetition", fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_ylim(0, 1)

    rep_split = prepared["rep_split"]
    ax = axes[2][2]
    for i, abl in enumerate(sorted(rep_split["ABL"].unique())):
        sub = rep_split[rep_split["ABL"] == abl]
        color = colors[i % len(colors)]
        if "after_left" in sub:
            ax.plot(sub["session"], sub["after_left"], "<-", color=color)
        if "after_right" in sub:
            ax.plot(sub["session"], sub["after_right"], ">-", color=color)
    _draw_regions(ax, regions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Session", fontsize=label_fs)
    ax.set_ylabel("Repetition probability", fontsize=label_fs)
    ax.set_title("Repetition split", fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_ylim(0, 1)

    return fig


def filter_sessions_report(
    prepared: dict[str, Any],
    *,
    min_trials: int = 100,
    trials_use: str = "total",
    history_n: int = 10,
    min_prev: int = 5,
    k: float = 3.0,
    min_perf: float = 0.7,
    require_perf_min_completed: int = 20,
):
    return DataHelpers.filter_sessions_with_history_bias_and_perf(
        df_trials=prepared["df"],
        min_trials=min_trials,
        trials_use=trials_use,
        history_n=history_n,
        min_prev=min_prev,
        k=k,
        min_perf=min_perf,
        require_perf_min_completed=require_perf_min_completed,
    )


def run_across_sessions(
    *,
    subject_file: str,
    line: str,
    cohort: str,
    training_level: int | None = 16,
    training_level_max: int | None = None,
    show: bool = True,
) -> dict[str, Any]:
    df_raw, data_dir = load_across_sessions_data(
        subject_file=subject_file,
        line=line,
        cohort=cohort,
    )
    subject_id = subject_id_from_file(subject_file)
    prepared = prepare_across_sessions_data(
        df_raw,
        training_level=training_level,
        training_level_max=training_level_max,
    )
    regions = load_change_regions(data_dir=data_dir, subject_id=subject_id)

    figs = {
        "trial_counts": plot_trial_counts(prepared, regions=regions),
        "summary": plot_across_sessions_summary(prepared, regions=regions),
        "combined": plot_across_sessions_combined(prepared, regions=regions),
    }

    if show:
        plt.show()

    return {
        "subject_id": subject_id,
        "data_dir": data_dir,
        "raw": df_raw,
        "prepared": prepared,
        "regions": regions,
        "figures": figs,
    }
