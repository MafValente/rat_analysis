from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Helpers.DataHelpers as DataHelpers
import Psychometric
from DailyMerge import get_base_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = ROOT / "DataFiles" / "Old Data" / "ILD_task"


def apply_daily_plot_style() -> None:
    mpl.rcParams.update({
        "savefig.pad_inches": 0.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "axes.linewidth": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def style_axes(
    ax,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    title_fs=22,
    label_fs=22,
    tick_fs=20,
    title_pad=16,
    box_aspect=1,
):
    if title:
        ax.set_title(title, fontsize=title_fs, pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fs, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fs, color="black")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fs,
        colors="black",
        width=1.5,
        length=6,
    )
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_reference_data(
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> dict[str, Any]:
    reference_dir = Path(reference_dir)
    makefig1_data = load_pickle(reference_dir / "fig1_plot_data.pkl")
    makefig1_chrono = load_pickle(reference_dir / "fig1_chrono_plot_data.pkl")
    makefig1_data = DataHelpers.normalize_ABL_labels(makefig1_data)
    return {
        "psychometric": makefig1_data,
        "chrono": makefig1_chrono,
    }


def load_daily_animal_data(
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


def _format_values(values) -> str:
    out = []
    for value in values:
        if pd.isna(value):
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric).is_integer():
            out.append(str(int(numeric)))
        else:
            out.append(str(value))
    return "[" + ", ".join(out) + "]"


def _sessions_by_type(df: pd.DataFrame) -> dict[str, list]:
    if df.empty or "session_type" not in df or "session" not in df:
        return {}

    sessions_by_type = {}
    for session_type, sub in df.groupby("session_type", dropna=True):
        type_label = _format_values([session_type]).strip("[]")
        sessions = sorted(pd.to_numeric(sub["session"], errors="coerce").dropna().astype(int).unique())
        sessions_by_type[type_label] = sessions
    return sessions_by_type


def prepare_daily_animal_data(
    df: pd.DataFrame,
    *,
    training_level: int | None = 16,
    training_level_max: int | None = None,
    include_long_duration: bool = True,
) -> dict[str, Any]:
    df = DataHelpers.prepare_data(df.copy(), session_col="session", trial_col="trial")
    df = df[df["trial_is_repeat"] == False].copy()

    if training_level_max is not None:
        df = df[pd.to_numeric(df["training_level"], errors="coerce") < training_level_max].copy()
    elif training_level is not None:
        df = df[df["training_level"] == training_level].copy()

    if include_long_duration:
        sess = pd.to_numeric(df["session_type"], errors="coerce")
        sd = pd.to_numeric(df["stim_dur"], errors="coerce")
        short_duration = pd.to_numeric(df["short_duration"], errors="coerce")
        df = df[(sess == 1) | (sd == 6000) | (short_duration == 0)].copy()

    df_last = df
    mean_rt = (
        df_last[df_last["success"] == 1]
        .groupby(["ABL", "ILD"])["timed_rt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    mean_mt = (
        df_last[df_last["success"] == 1]
        .groupby(["ABL", "ILD"])["timed_mt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    psychometric_results = Psychometric.compute_psychometrics_by_ABL(df_last)
    jnd = DataHelpers.compute_jnd_by_ABL(psychometric_results, skip_ABL=50)

    info = {
        "animal": df_last["animal"].iloc[0] if not df_last.empty and "animal" in df_last else None,
        "setup": df_last["box"].unique() if not df_last.empty and "box" in df_last else [],
        "session_type": df_last["session_type"].unique() if not df_last.empty and "session_type" in df_last else [],
        "n_sessions": df_last["session"].nunique() if "session" in df_last else 0,
        "sessions_by_type": _sessions_by_type(df_last),
        "n_trials": len(df_last),
    }

    return {
        "df": df,
        "df_last": df_last,
        "info": info,
        "mean_rt": mean_rt,
        "mean_mt": mean_mt,
        "psychometric_results": psychometric_results,
        "jnd": jnd,
    }


def plot_daily_animal_summary(
    prepared: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    overlay_reference_psychometric: bool = True,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(12, 10),
):
    apply_daily_plot_style()
    mean_rt = prepared["mean_rt"]
    mean_mt = prepared["mean_mt"]
    results = prepared["psychometric_results"]
    info_data = prepared["info"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax_text = axes[0, 0]
    ax_text.axis("off")
    info = (
        f"Animal: {info_data['animal']}\n"
        f"Setup number: {info_data['setup']}\n"
        f"Session type: {info_data['session_type']}\n"
        f"Sessions: {info_data['n_sessions']}\n"
        + "\n".join(
            f"  Type {session_type}: {len(sessions)} sessions"
            for session_type, sessions in info_data.get("sessions_by_type", {}).items()
        )
        + "\n"
        f"Number of trials: {info_data['n_trials']}\n"
    )
    ax_text.text(
        0.5,
        0.5,
        info,
        fontsize=16,
        ha="center",
        va="center",
        family="monospace",
    )

    for i, abl in enumerate(sorted(mean_rt["ABL"].unique())):
        sub = mean_rt[mean_rt["ABL"] == abl]
        axes[0, 1].errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {abl}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )

    style_axes(
        axes[0, 1],
        title="Reaction Time",
        xlabel="ILD (dB)",
        ylabel="Mean Reaction Time (s)",
    )

    for i, abl in enumerate(sorted(mean_mt["ABL"].unique())):
        sub = mean_mt[mean_mt["ABL"] == abl]
        axes[1, 0].errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {abl}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )

    style_axes(
        axes[1, 0],
        title="Movement Time",
        xlabel="ILD (dB)",
        ylabel="Mean Movement Time (s)",
    )

    ax_psy = axes[1, 1]
    for color, (abl, res) in zip(colors, results.items()):
        ax_psy.scatter(
            DataHelpers.shift_ILD_for_ABL50(res["ILDs"]),
            res["PropLeft"],
            label=f"ABL={abl}",
            color=color,
            s=70,
            edgecolor=color,
            linewidth=0.6,
            zorder=3,
        )
        if abl != 50:
            ax_psy.plot(
                DataHelpers.shift_ILD_for_ABL50(res["xx"]),
                res["yy"],
                color=color,
                linewidth=1.8,
            )

    if reference is not None and overlay_reference_psychometric:
        DataHelpers.overlay_makefig1_psychometrics(
            ax_psy,
            reference["psychometric"],
            color="black",
            show_individuals=True,
            use_abl_colors=False,
        )

    ax_psy.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    ax_psy.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    style_axes(
        ax_psy,
        title="Psychometric by ABL",
        xlabel="ILD (dB)",
        ylabel="Proportion Left",
    )

    xticks = sorted(set(ax_psy.get_xticks()) | {-18, 18})
    ax_psy.set_xticks(xticks)
    ax_psy.set_xticklabels([
        "-50" if x == -18 else "50" if x == 18 else str(int(x))
        for x in xticks
    ])
    ax_psy.set_xlim(-19, 19)
    ax_psy.legend(fontsize=16, frameon=False)

    fig.tight_layout()
    return fig


def plot_daily_animal_jnd(
    prepared: dict[str, Any],
    *,
    figsize=(2.5, 2),
):
    apply_daily_plot_style()
    jnd = prepared["jnd"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        jnd["ABL"],
        jnd["JND"],
        "-o",
        color="black",
        linewidth=1.2,
        markersize=4,
    )
    style_axes(
        ax,
        xlabel="ABL (dB)",
        ylabel="JND (dB)",
        title_fs=12,
        label_fs=10,
        tick_fs=10,
    )
    ax.set_xticks(sorted(jnd["ABL"].unique()))
    ax.grid(True, linestyle="--", alpha=0.3, zorder=-10)
    fig.tight_layout()
    return fig


def plot_daily_animal_summary_with_jnd(
    prepared: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    overlay_reference_psychometric: bool = True,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(18, 10),
):
    apply_daily_plot_style()
    mean_rt = prepared["mean_rt"]
    mean_mt = prepared["mean_mt"]
    results = prepared["psychometric_results"]
    jnd = prepared["jnd"]
    info_data = prepared["info"]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_mt = fig.add_subplot(gs[1, 0])
    ax_psy = fig.add_subplot(gs[1, 1])

    ax_text.axis("off")
    info = (
        f"Animal: {info_data['animal']}\n"
        f"Setup number: {info_data['setup']}\n"
        f"Session type: {info_data['session_type']}\n"
        f"Sessions: {info_data['n_sessions']}\n"
        + "\n".join(
            f"  Type {session_type}: {len(sessions)} sessions"
            for session_type, sessions in info_data.get("sessions_by_type", {}).items()
        )
        + "\n"
        f"Number of trials: {info_data['n_trials']}\n"
    )
    ax_text.text(
        0.5,
        0.5,
        info,
        fontsize=18,
        ha="center",
        va="center",
        family="monospace",
    )

    for i, abl in enumerate(sorted(mean_rt["ABL"].unique())):
        sub = mean_rt[mean_rt["ABL"] == abl]
        ax_rt.errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {abl}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )
    style_axes(
        ax_rt,
        title="Reaction Time",
        xlabel="ILD (dB)",
        ylabel="Mean RT (s)",
        box_aspect=None,
    )

    for i, abl in enumerate(sorted(mean_mt["ABL"].unique())):
        sub = mean_mt[mean_mt["ABL"] == abl]
        ax_mt.errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]),
            sub["mean"],
            yerr=sub["std"] / np.sqrt(sub["count"]),
            label=f"ABL {abl}",
            color=colors[i % len(colors)],
            linestyle="none",
            marker="o",
            capsize=5,
            linewidth=1.8,
        )
    style_axes(
        ax_mt,
        title="Movement Time",
        xlabel="ILD (dB)",
        ylabel="Mean MT (s)",
        box_aspect=None,
    )

    for color, (abl, res) in zip(colors, results.items()):
        ax_psy.scatter(
            DataHelpers.shift_ILD_for_ABL50(res["ILDs"]),
            res["PropLeft"],
            label=f"ABL={abl}",
            color=color,
            s=70,
            edgecolor=color,
            linewidth=0.6,
            zorder=3,
        )
        if abl != 50:
            ax_psy.plot(
                DataHelpers.shift_ILD_for_ABL50(res["xx"]),
                res["yy"],
                color=color,
                linewidth=1.8,
            )
    if reference is not None and overlay_reference_psychometric:
        DataHelpers.overlay_makefig1_psychometrics(
            ax_psy,
            reference["psychometric"],
            color="black",
            show_individuals=True,
            use_abl_colors=False,
        )
    ax_psy.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    ax_psy.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=-100)
    style_axes(
        ax_psy,
        title="Psychometric",
        xlabel="ILD (dB)",
        ylabel="Proportion Left",
        box_aspect=None,
    )
    xticks = sorted(set(ax_psy.get_xticks()) | {-18, 18})
    ax_psy.set_xticks(xticks)
    ax_psy.set_xticklabels([
        "-50" if x == -18 else "50" if x == 18 else str(int(x))
        for x in xticks
    ])
    ax_psy.set_xlim(-19, 19)
    ax_psy.legend(fontsize=13, frameon=False, loc="upper left")

    ax_jnd = ax_psy.inset_axes([0.7, 0.12, 0.25, 0.25])
    ax_jnd.set_facecolor("white")
    full_abl_order = sorted(mean_rt["ABL"].unique())
    abl_colors = {
        abl: colors[i % len(colors)]
        for i, abl in enumerate(full_abl_order)
    }
    jnd_sorted = jnd.sort_values("ABL").reset_index(drop=True)
    for _, row in jnd_sorted.iterrows():
        ax_jnd.plot(
            row["ABL"],
            row["JND"],
            marker="o",
            linestyle="none",
            color=abl_colors.get(row["ABL"], "black"),
            markersize=7,
        )
    style_axes(
        ax_jnd,
        #title="JND",
        xlabel="ABL",
        ylabel="JND",
        title_fs=13,
        label_fs=12,
        tick_fs=10,
        title_pad=4,
        box_aspect=None,
    )
    ax_jnd.set_xticks(sorted(jnd["ABL"].unique()))
    ax_jnd.set_xticklabels([])
    ax_jnd.grid(True, linestyle="--", alpha=0.3, zorder=-10)

    return fig


def _make_bins(values, bin_width, *, start=None, stop=None):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.array([0, bin_width])
    vmin = values.min() if start is None else start
    vmax = values.max() if stop is None else stop
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmax = vmin + bin_width
    return np.arange(vmin, vmax + bin_width, bin_width)


def plot_timing_histograms(
    prepared: dict[str, Any],
    *,
    subject_id: str | None = None,
    figsize=(16, 9),
    bin_width_ft=20,
    bin_width_cnp=0.5,
    bin_width_rt=0.01,
    bin_width_mt=0.01,
    bin_width_lnp=0.001,
):
    apply_daily_plot_style()

    df = prepared["df"].copy()
    df_valid = df[df["abort_type"] != "CNP"].copy()
    df_aborts = df[df["abort_type"] == "Fixation"].copy()
    subject_id = subject_id or prepared["info"].get("animal") or "Animal"

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    cols_top = ["fix_time", "intended_fix_time", "cnp_time"]
    cols_bottom = ["timed_rt", "timed_mt", "timed_lnp"]
    xlabel_map = {
        "fix_time": "Fixation Time (ms)",
        "intended_fix_time": "Intended Fixation Time (ms)",
        "cnp_time": "CNP Time (s)",
        "timed_rt": "Timed RT (s)",
        "timed_mt": "Timed MT (s)",
        "timed_lnp": "Timed LNP (s)",
    }

    bins_top = {
        "fix_time": _make_bins(df_valid["fix_time"], bin_width_ft, start=0),
        "intended_fix_time": _make_bins(df_valid["intended_fix_time"], bin_width_ft, start=0),
        "cnp_time": _make_bins(df_valid["cnp_time"], bin_width_cnp),
    }
    bins_bottom = {
        "timed_rt": _make_bins(df["timed_rt"], bin_width_rt),
        "timed_mt": _make_bins(df_valid["timed_mt"], bin_width_mt),
        "timed_lnp": _make_bins(df_valid["timed_lnp"], bin_width_lnp),
    }
    bin_labels = {
        "fix_time": bin_width_ft,
        "intended_fix_time": bin_width_ft,
        "cnp_time": bin_width_cnp,
        "timed_rt": bin_width_rt,
        "timed_mt": bin_width_mt,
        "timed_lnp": bin_width_lnp,
    }

    for ax, col in zip(axes[0, :], cols_top):
        ax.hist(
            df_valid[col].dropna(),
            bins=bins_top[col],
            histtype="step",
            color="black",
            linewidth=1.4,
            label="All initiated trials",
        )
        if col == "fix_time":
            ax.hist(
                df_aborts["fix_time"].dropna(),
                bins=bins_top[col],
                histtype="step",
                color="red",
                linewidth=1.4,
                label="Fixation aborts",
            )
            ax.legend(fontsize=11, frameon=False)

    for ax, col in zip(axes[1, :], cols_bottom):
        source = df if col == "timed_rt" else df_valid
        ax.hist(
            source[col].dropna(),
            bins=bins_bottom[col],
            histtype="step",
            color="black",
            linewidth=1.4,
        )

    for ax, col in zip(axes.flat, cols_top + cols_bottom):
        ax.set_xlabel(xlabel_map[col], fontsize=15)
        ax.set_ylabel("Count", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.3, color="black")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")
        ax.text(
            0.72,
            0.94,
            f"bin = {bin_labels[col]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )

    axes[0, 0].set_xlim(-2, 2000)
    axes[0, 2].set_xlim(-1, 10)
    axes[1, 0].set_xlim(-0.5, 0.5)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 2].set_xlim(0, 0.02)

    fig.suptitle(f"{subject_id}: Timing Distributions", fontsize=20, fontweight="bold")
    return fig


def plot_rt_histogram_by_abl(
    prepared: dict[str, Any],
    *,
    subject_id: str | None = None,
    colors=("C0", "C1", "C2", "C3"),
    figsize=(7, 4),
    bin_width=0.01,
    exclude_abls=(50,),
):
    df = prepared["df"].copy()
    df_valid = df[df["abort_type"] != "CNP"].copy()
    df_rt = df_valid.dropna(subset=["timed_rt", "ABL"]).copy()
    if exclude_abls:
        df_rt = df_rt[~df_rt["ABL"].isin(exclude_abls)]
    subject_id = subject_id or prepared["info"].get("animal") or "Animal"

    bins = _make_bins(df_rt["timed_rt"], bin_width)
    fig, ax = plt.subplots(figsize=figsize)
    for i, abl in enumerate(sorted(df_rt["ABL"].unique())):
        subset = df_rt[df_rt["ABL"] == abl]["timed_rt"]
        ax.hist(
            subset,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=colors[i % len(colors)],
            label=f"ABL {abl}",
        )

    ax.set_xlabel("Reaction Time (s)", fontsize=15)
    ax.set_ylabel("Count", fontsize=15)
    ax.set_title(f"{subject_id}: Reaction Times by ABL", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 0.75)
    ax.legend(title="ABL", fontsize=12, frameon=False)
    ax.tick_params(axis="both", labelsize=12, width=1.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")
    fig.tight_layout()
    return fig


def run_daily_animal_plots(
    *,
    subject_file: str,
    line: str,
    cohort: str,
    training_level: int | None = 16,
    training_level_max: int | None = None,
    load_reference: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    df_raw, data_dir = load_daily_animal_data(
        subject_file=subject_file,
        line=line,
        cohort=cohort,
    )
    prepared = prepare_daily_animal_data(
        df_raw,
        training_level=training_level,
        training_level_max=training_level_max,
    )
    reference = load_reference_data() if load_reference else None
    figures = {
        "summary": plot_daily_animal_summary(prepared, reference=reference),
        "jnd": plot_daily_animal_jnd(prepared),
        "summary_with_jnd": plot_daily_animal_summary_with_jnd(prepared, reference=reference),
        "timing_histograms": plot_timing_histograms(prepared),
        "rt_by_abl_histogram": plot_rt_histogram_by_abl(prepared),
    }
    if show:
        plt.show()
    return {
        "data_dir": data_dir,
        "raw": df_raw,
        "prepared": prepared,
        "reference": reference,
        "figures": figures,
    }
