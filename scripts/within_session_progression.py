#%%
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.datasets import load_dataset_selections, load_line_across_cohorts
import Helpers.DataHelpers as DataHelpers


# ==============================================================
# CONFIG
# ==============================================================
LINE = "CNTNAP2"
COHORT_SELECTION = ["cohort2", "cohort3"]  # "all", "cohort2", "cohort3", or ["cohort2", "cohort3"]

DATASET_SELECTIONS = None
# Example:
# DATASET_SELECTIONS = [
#     ("CNTNAP2", "cohort2"),
#     ("CNTNAP2", "cohort3"),
#     ("SHANK3", "cohort1"),
# ]

BASE_DATA_DIR = os.path.join(ROOT, "DataFiles")
OUTPUT_DIR = os.path.join(ROOT, "Figures", "WithinSessionProgression")

TRAINING_LEVEL = 16
WINDOW_TRIALS = 100
RUNNING_WINDOWS = {
    "accuracy": 3,
    "rt": 5,
    "bias": 5,
}
MIN_SESSION_TRIALS = 100
MIN_SESSION_TRIALS_FOR_END = 200

KEEP_ONLY_LONG_SOUND = True
SAVE_FIGURES = False
SHOW_FIGURES = True
PHASE_GAP = 12


mpl.rcParams["savefig.pad_inches"] = 0.4
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]


def _selection_label(dataset_info: dict) -> str:
    if dataset_info.get("dataset_keys"):
        return ", ".join(dataset_info["dataset_keys"])
    return f"{dataset_info['line']}:{','.join(dataset_info['cohorts'])}"


def _cohort_sort_key(name: str):
    name = str(name)
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 10**9, name.lower())


def _subject_label(row: pd.Series, *, multi_dataset: bool) -> str:
    animal = str(row["animal"]).strip()
    if not multi_dataset:
        return animal
    return f"{animal} ({row['line']}, {row['cohort']})"


def _clean_genotype(value) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    return value or None


def _prepare_trials(df: pd.DataFrame) -> pd.DataFrame:
    df = DataHelpers.prepare_data(df.copy(), session_col="session", trial_col="trial")

    if "trial_is_repeat" in df.columns:
        df = df[df["trial_is_repeat"] == False].copy()

    if "training_level" in df.columns:
        df = df[pd.to_numeric(df["training_level"], errors="coerce") == TRAINING_LEVEL].copy()

    if KEEP_ONLY_LONG_SOUND:
        sess = pd.to_numeric(df["session_type"], errors="coerce")
        stim_dur = pd.to_numeric(df["stim_dur"], errors="coerce")
        df = df[(sess == 1) | (stim_dur == 6000)].copy()

    for col in ("animal", "line", "cohort"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "genotype" in df.columns:
        df["genotype"] = df["genotype"].map(_clean_genotype)

    df["session_num"] = pd.to_numeric(df["session"], errors="coerce")
    df["trial_num"] = pd.to_numeric(df["trial"], errors="coerce")
    df = df.dropna(subset=["animal", "session_num", "trial_num"]).copy()
    df["session_num"] = df["session_num"].astype(int)

    return df.sort_values(["line", "cohort", "animal", "session_num", "trial_num"]).copy()


def _summarize_running_window(
    session_df: pd.DataFrame,
    *,
    phase: str,
    windows: dict[str, int],
) -> list[dict]:
    n_trials = len(session_df)
    if n_trials == 0:
        return []

    work = session_df.reset_index(drop=True).copy()
    valid = work["success"].isin([1, -1])

    out = pd.DataFrame({
        "phase": phase,
        "trial_pos": np.arange(1, n_trials + 1, dtype=int),
        "x": np.arange(1, n_trials + 1, dtype=int),
    })

    if "accuracy" in windows:
        w = int(windows["accuracy"])
        accuracy_series = pd.Series(np.where(valid, (work["success"] == 1).astype(float), np.nan))
        out["accuracy"] = accuracy_series.rolling(window=w, min_periods=1).mean().to_numpy()
        out["accuracy_window"] = w

    if "rt" in windows:
        w = int(windows["rt"])
        rt_series = pd.to_numeric(work["timed_rt"], errors="coerce").where(valid, np.nan)
        out["rt"] = rt_series.rolling(window=w, min_periods=1).mean().to_numpy()
        out["rt_window"] = w

    if "bias" in windows:
        w = int(windows["bias"])
        ild = pd.to_numeric(work["ILD"], errors="coerce")
        response = pd.to_numeric(work["response_poke"], errors="coerce")

        valid_f = valid.astype(float)
        n_valid = valid_f.rolling(window=w, min_periods=1).sum()

        neg = ((ild < 0) & valid).astype(float)
        pos = ((ild > 0) & valid).astype(float)
        wrong_right = ((response == 1) & (ild < 0) & valid).astype(float)
        wrong_left = ((response == -1) & (ild > 0) & valid).astype(float)

        neg_roll = neg.rolling(window=w, min_periods=1).sum()
        pos_roll = pos.rolling(window=w, min_periods=1).sum()
        wrong_right_roll = wrong_right.rolling(window=w, min_periods=1).sum()
        wrong_left_roll = wrong_left.rolling(window=w, min_periods=1).sum()

        frac_wrong_right = pd.Series(np.where(neg_roll > 0, wrong_right_roll / neg_roll, 0.0), index=out.index)
        frac_wrong_left = pd.Series(np.where(pos_roll > 0, wrong_left_roll / pos_roll, 0.0), index=out.index)
        bias = frac_wrong_right - frac_wrong_left
        bias = bias.where(n_valid > 0, np.nan)

        out["bias"] = bias.to_numpy()
        out["bias_window"] = w

    return out.to_dict("records")


def build_within_session_summary(
    df: pd.DataFrame,
    *,
    running_windows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    running_windows = dict(running_windows or RUNNING_WINDOWS)
    records: list[dict] = []
    session_records: list[dict] = []
    multi_dataset = df[["line", "cohort"]].drop_duplicates().shape[0] > 1

    group_cols = ["line", "cohort", "genotype", "animal", "session_num"]
    for (line, cohort, genotype, animal, session_num), df_session in df.groupby(group_cols, sort=False):
        df_session = df_session.sort_values("trial_num").copy()
        n_trials = len(df_session)

        session_label = {
            "line": line,
            "cohort": cohort,
            "genotype": genotype,
            "animal": animal,
            "session_num": session_num,
            "subject_label": _subject_label(pd.Series({"animal": animal, "line": line, "cohort": cohort}), multi_dataset=multi_dataset),
            "n_trials_session": n_trials,
            "kept_for_start": n_trials >= MIN_SESSION_TRIALS,
            "kept_for_end": n_trials >= MIN_SESSION_TRIALS_FOR_END,
        }
        session_records.append(session_label)

        if n_trials < MIN_SESSION_TRIALS:
            continue

        start_df = df_session.iloc[:WINDOW_TRIALS].copy()
        start_rows = _summarize_running_window(start_df, phase="start", windows=running_windows)

        for row in start_rows:
            records.append({**session_label, **row})

        if n_trials >= MIN_SESSION_TRIALS_FOR_END:
            end_df = df_session.iloc[-WINDOW_TRIALS:].copy()
            end_rows = _summarize_running_window(end_df, phase="end", windows=running_windows)
            for row in end_rows:
                records.append({**session_label, **row})

    summary = pd.DataFrame.from_records(records)
    sessions = pd.DataFrame.from_records(session_records)

    if summary.empty:
        return summary, sessions

    agg = (
        summary.groupby(
            ["line", "cohort", "genotype", "animal", "subject_label", "phase", "trial_pos", "x"],
            dropna=False,
            observed=False,
        )
        .agg(
            accuracy=("accuracy", "mean"),
            rt=("rt", "mean"),
            bias=("bias", "mean"),
            n_sessions=("session_num", "nunique"),
            accuracy_window=("accuracy_window", "first"),
            rt_window=("rt_window", "first"),
            bias_window=("bias_window", "first"),
        )
        .reset_index()
        .sort_values(["genotype", "subject_label", "x"])
    )

    return agg, sessions


def process_within_session_data(
    *,
    line: str = LINE,
    cohort_selection=COHORT_SELECTION,
    dataset_selections=DATASET_SELECTIONS,
    base_data_dir: str = BASE_DATA_DIR,
    running_windows: dict[str, int] | None = None,
) -> dict:
    if dataset_selections:
        df_all, meta_all, dataset_info = load_dataset_selections(
            selections=dataset_selections,
            base_dir=base_data_dir,
            require_meta=False,
        )
    else:
        df_all, meta_all, dataset_info = load_line_across_cohorts(
            line=line,
            base_dir=base_data_dir,
            cohorts=cohort_selection,
            require_meta=False,
        )

    df_prepared = _prepare_trials(df_all)
    windows_used = dict(running_windows or RUNNING_WINDOWS)
    summary, session_table = build_within_session_summary(df_prepared, running_windows=windows_used)

    if summary.empty:
        raise ValueError(
            "No sessions survived the within-session progression filters. "
            f"Sessions need at least {MIN_SESSION_TRIALS} trials to contribute to the start window."
        )

    kept_start = int(session_table["kept_for_start"].sum())
    kept_end = int(session_table["kept_for_end"].sum())
    excluded_short = int((~session_table["kept_for_start"]).sum())

    return {
        "selection": dataset_info,
        "df_prepared": df_prepared,
        "summary": summary,
        "session_table": session_table,
        "running_windows": windows_used,
        "stats": {
            "subjects": int(session_table[["subject_label"]].drop_duplicates().shape[0]),
            "kept_start": kept_start,
            "kept_end": kept_end,
            "excluded_short": excluded_short,
        },
    }


def _style_metric_axis(ax: plt.Axes, metric: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    if metric == "accuracy":
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color="0.8", linestyle="--", linewidth=1)
    elif metric == "bias":
        ax.axhline(0.0, color="0.8", linestyle="--", linewidth=1)


def plot_genotype_figure(
    genotype: str,
    summary: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    running_windows: dict[str, int] | None = None,
) -> plt.Figure:
    df_g = summary[summary["genotype"] == genotype].copy()
    subject_order = (
        df_g[["subject_label", "line", "cohort", "animal"]]
        .drop_duplicates()
        .sort_values(["line", "cohort", "animal"], key=lambda col: col.map(_cohort_sort_key) if col.name == "cohort" else col)
    )
    subjects = subject_order["subject_label"].tolist()

    n_rows = max(1, len(subjects))
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15, max(2.6 * n_rows, 4.8)),
        sharex=True,
        squeeze=False,
    )

    metrics = [("accuracy", "Accuracy"), ("rt", "RT"), ("bias", "Bias")]
    running_windows = dict(running_windows or RUNNING_WINDOWS)
    phase_styles = {
        "start": {"color": "black", "linestyle": "-", "label": "First 100"},
        "end": {"color": "0.45", "linestyle": "--", "label": "Last 100"},
    }
    end_offset = WINDOW_TRIALS + PHASE_GAP

    for row_idx, subject_label in enumerate(subjects):
        sub = df_g[df_g["subject_label"] == subject_label].copy()
        session_info = sessions[
            (sessions["genotype"] == genotype) & (sessions["subject_label"] == subject_label)
        ].copy()

        n_start = int(session_info["kept_for_start"].sum())
        n_end = int(session_info["kept_for_end"].sum())
        row_label = f"{subject_label}\nfirst100 n={n_start}, last100 n={n_end}"

        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for phase in ("start", "end"):
                phase_df = sub[sub["phase"] == phase].sort_values("x")
                if phase_df.empty:
                    continue
                style = phase_styles[phase]
                xvals = phase_df["x"] if phase == "start" else phase_df["x"] + end_offset
                ax.plot(
                    xvals,
                    phase_df[metric],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    label=style["label"] if row_idx == 0 and col_idx == 0 else None,
                )

            _style_metric_axis(ax, metric)

            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=56)
            else:
                ax.set_ylabel("")

            if metric == "accuracy":
                ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(1, end_offset + WINDOW_TRIALS)
            ax.axvline(WINDOW_TRIALS + (PHASE_GAP / 2), color="0.75", linestyle="--", linewidth=1)

    start_ticks = [1, 25, 50, 75, 100]
    end_ticks = [end_offset + x for x in start_ticks]
    xticks = start_ticks + end_ticks
    xticklabels = [str(x) for x in start_ticks] + [str(x) for x in start_ticks]

    for ax in axes[-1]:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Trial number")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.88, 0.985), ncol=1, frameon=False)

    fig.suptitle(
        f"{genotype.upper()} within-session progression\n"
        f"acc={running_windows['accuracy']}, rt={running_windows['rt']}, bias={running_windows['bias']} trials",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0.06, 0.03, 0.86, 0.95])
    return fig


def plot_within_session_figures(
    processed: dict,
    *,
    save_figures: bool = SAVE_FIGURES,
    show_figures: bool = SHOW_FIGURES,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, plt.Figure]:
    summary = processed["summary"]
    session_table = processed["session_table"]
    dataset_info = processed["selection"]
    running_windows = processed.get("running_windows", RUNNING_WINDOWS)

    genotypes = [g for g in ("wt", "het", "hom") if g in set(summary["genotype"].dropna())]
    if not genotypes:
        genotypes = sorted(summary["genotype"].dropna().unique())

    figures: dict[str, plt.Figure] = {}
    selection_tag = _selection_label(dataset_info).replace(":", "_").replace(",", "__").replace(" ", "")
    save_dir = Path(output_dir) / selection_tag
    if save_figures:
        save_dir.mkdir(parents=True, exist_ok=True)

    for genotype in genotypes:
        fig = plot_genotype_figure(
            genotype,
            summary,
            session_table,
            running_windows=running_windows,
        )
        figures[genotype] = fig
        if save_figures:
            out_path = save_dir / f"within_session_progression_{genotype}.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")

    stats = processed["stats"]
    print(f"Selection: {_selection_label(dataset_info)}")
    print(f"Subjects: {stats['subjects']}")
    print(f"Sessions contributing to first 100 trials: {stats['kept_start']}")
    print(f"Sessions contributing to last 100 trials: {stats['kept_end']}")
    print(f"Sessions excluded for having < {MIN_SESSION_TRIALS} total trials: {stats['excluded_short']}")

    if show_figures:
        plt.show()

    return figures


#%%
processed = process_within_session_data()
dataset_info = processed["selection"]
df_prepared = processed["df_prepared"]
summary = processed["summary"]
session_table = processed["session_table"]

#%%
figures = plot_within_session_figures(processed)

out = {
    **processed,
    "figures": figures,
}

# %%
