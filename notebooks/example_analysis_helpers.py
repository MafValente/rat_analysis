from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import Helpers.DataHelpers as DataHelpers


REPO_ROOT = Path(__file__).resolve().parents[1]
DATAFILES_DIR = REPO_ROOT / "DataFiles"
COLORS = ["C0", "C1", "C2", "C3", "C4", "C5"]


@dataclass(frozen=True)
class DatasetSelection:
    line: str
    cohort: str
    genotype: str = "all"
    animal: str = "all"


def discover_datasets(base_dir: Path = DATAFILES_DIR) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        if path.name == "Old Data":
            continue
        cohort_csv = path / "merged_all_subjects.csv"
        meta_csv = path / "sex_gen.csv"
        if not cohort_csv.exists():
            continue
        line, cohort = _parse_dataset_dir(path.name)
        rows.append(
            {
                "line": line,
                "cohort": cohort,
                "data_dir": str(path),
                "cohort_csv": str(cohort_csv),
                "meta_csv": str(meta_csv) if meta_csv.exists() else None,
            }
        )
    return pd.DataFrame(rows)


def load_dataset(selection: DatasetSelection) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    datasets = discover_datasets()
    match = datasets[
        (datasets["line"].str.lower() == selection.line.lower())
        & (datasets["cohort"].str.lower() == selection.cohort.lower())
    ]
    if match.empty:
        raise ValueError(f"No dataset found for line={selection.line!r}, cohort={selection.cohort!r}.")

    row = match.iloc[0]
    data_dir = Path(row["data_dir"])
    df = pd.read_csv(row["cohort_csv"], low_memory=False)

    meta_path = row["meta_csv"]
    if meta_path:
        meta = pd.read_csv(meta_path, sep=";")
        meta.attrs["meta_csv_path"] = str(meta_path)
    else:
        meta = pd.DataFrame(columns=["animal", "cohort", "line", "sex", "genotype"])

    return df, meta, data_dir


def build_selection_table(meta: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        animals = pd.Series(sorted(df["animal"].dropna().astype(str).unique()), name="animal")
        return pd.DataFrame({"animal": animals, "genotype": "unknown", "line": "unknown"})

    table = meta.copy()
    table.columns = [str(c).strip().lower() for c in table.columns]
    table["animal"] = table["animal"].astype(str)
    if "genotype" not in table.columns:
        table["genotype"] = "unknown"
    if "line" not in table.columns:
        table["line"] = "unknown"
    table["genotype"] = table["genotype"].astype(str).str.lower()
    table["line"] = table["line"].astype(str).str.upper()
    return table.sort_values(["genotype", "animal"]).reset_index(drop=True)


def filter_subjects(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    selection: DatasetSelection,
) -> pd.DataFrame:
    out = df.copy()

    if selection.genotype.lower() != "all" and not meta.empty:
        allowed = (
            meta.loc[meta["genotype"].astype(str).str.lower() == selection.genotype.lower(), "animal"]
            .astype(str)
            .unique()
        )
        out = out[out["animal"].astype(str).isin(allowed)].copy()

    if selection.animal.lower() != "all":
        out = out[out["animal"].astype(str) == selection.animal].copy()

    return out


def prepare_across_sessions_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
    df = df[df["training_level"] < 16].copy()
    df_valid = df[df["trial_is_repeat"] == False].copy()

    summaries = {
        "trial_counts": _build_trial_counts(df, df_valid),
        "rt": _group_mean_sem(
            df_valid[(df_valid["success"] == 1) & (df_valid["timed_rt"] <= 1.2)],
            value_col="timed_rt",
        ),
        "mt": _group_mean_sem(
            df_valid[(df_valid["success"] == 1) & (df_valid["timed_mt"] <= 2)],
            value_col="timed_mt",
        ),
        "accuracy": (
            df_valid[df_valid["success"] != 0]
            .groupby(["session", "ABL"])["success"]
            .agg(accuracy=lambda x: (x == 1).mean(), n_trials="count")
            .reset_index()
        ),
    }
    return summaries


def plot_across_sessions(
    summaries: dict[str, pd.DataFrame],
    title: str,
    change_points_csv: Path | None = None,
    subject: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    trial_counts = summaries["trial_counts"]
    for column, label in [
        ("completed", "Completed Trials"),
        ("cnp", "CNP Aborts"),
        ("aborted", "Other Aborts"),
    ]:
        axes[0].plot(
            trial_counts["session"],
            trial_counts[column],
            marker="o",
            linestyle="-",
            label=label,
        )
    axes[0].set_title("Trial counts")
    axes[0].set_xlabel("Session")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    _plot_by_abl(axes[1], summaries["rt"], "mean", "Reaction Time (s)", "RT progression")
    _plot_by_abl(axes[2], summaries["mt"], "mean", "Movement Time (s)", "MT progression")
    _plot_by_abl(axes[3], summaries["accuracy"], "accuracy", "Accuracy", "Accuracy progression")

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _maybe_shade_change_regions(ax, change_points_csv, subject)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def compute_learning_curve_view(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    selection: DatasetSelection,
    normalized_points: int = 100,
    span: int = 25,
) -> dict:
    df = df[df["success"].isin([1, -1])].copy()

    group_filters = _learning_curve_filters(meta, selection)
    results = {}
    for name, filter_fn in group_filters:
        df_view = filter_fn(df)
        if df_view.empty:
            results[name] = None
            continue
        results[name] = _compute_session_curves(
            name=name,
            df=df_view,
            normalized_points=normalized_points,
            span=span,
        )
    return results


def plot_learning_curves(results: dict, title: str) -> plt.Figure:
    groups = list(results.keys())
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3.8 * max(len(groups), 1)), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        res = results[group]
        if res is None or not res["sessions"]:
            ax.text(0.5, 0.5, f"{group.upper()} - no data", ha="center", va="center")
            ax.set_axis_off()
            continue

        for idx, session in enumerate(sorted(res["sessions"])):
            info = res["sessions"][session]
            x = np.linspace(0, 1, info["length"])
            color = plt.cm.viridis(idx / max(len(res["sessions"]) - 1, 1))
            ax.plot(x, info["mean"], color=color, linewidth=2, label=f"Session {session}")
            ax.fill_between(
                x,
                info["mean"] - info["sem"],
                info["mean"] + info["sem"],
                color=color,
                alpha=0.2,
            )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(correct)")
        ax.set_title(group.upper())
        ax.legend(loc="upper left", ncol=2, fontsize=9)

    axes[-1].set_xlabel("Normalized session progress")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def available_animals(meta: pd.DataFrame, genotype: str = "all") -> list[str]:
    table = build_selection_table(meta, pd.DataFrame(columns=["animal"]))
    if genotype.lower() != "all":
        table = table[table["genotype"].str.lower() == genotype.lower()]
    return ["all", *table["animal"].astype(str).tolist()]


def available_genotypes(meta: pd.DataFrame) -> list[str]:
    if meta.empty or "genotype" not in meta.columns:
        return ["all"]
    vals = sorted(meta["genotype"].dropna().astype(str).str.lower().unique())
    return ["all", *vals]


def _parse_dataset_dir(name: str) -> tuple[str, str]:
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse dataset directory name: {name}")
    return parts[0].upper(), parts[1]


def _build_trial_counts(df: pd.DataFrame, df_valid: pd.DataFrame) -> pd.DataFrame:
    completed = DataHelpers.count_trials(df_valid, df_valid["success"] != 0, "completed").rename(
        columns={"trial_count": "completed"}
    )[["session", "completed"]]
    cnp = DataHelpers.count_trials(df, df["abort_type"] == "CNP", "cnp").rename(
        columns={"trial_count": "cnp"}
    )[["session", "cnp"]]
    aborted = DataHelpers.count_trials(df, (df["abort_type"] != "CNP") & (df["success"] == 0), "aborted").rename(
        columns={"trial_count": "aborted"}
    )[["session", "aborted"]]

    out = completed.merge(cnp, on="session", how="outer").merge(aborted, on="session", how="outer")
    return out.fillna(0).sort_values("session").reset_index(drop=True)


def _group_mean_sem(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["session", "ABL", "mean", "std", "count", "sem"])
    out = (
        df.groupby(["session", "ABL"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def _plot_by_abl(ax: plt.Axes, df: pd.DataFrame, y_col: str, ylabel: str, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    for idx, abl in enumerate(sorted(df["ABL"].dropna().unique())):
        sub = df[df["ABL"] == abl]
        yerr = sub["sem"] if "sem" in sub.columns else None
        ax.errorbar(
            sub["session"],
            sub[y_col],
            yerr=yerr,
            color=COLORS[idx % len(COLORS)],
            marker="o",
            linestyle="-",
            capsize=4,
            label=f"ABL {abl}",
        )

    ax.set_title(title)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.legend()


def _maybe_shade_change_regions(ax: plt.Axes, change_points_csv: Path | None, subject: str | None) -> None:
    if not change_points_csv or not subject or not change_points_csv.exists():
        return
    regions = DataHelpers.shade_change_regions_from_csv(ax, str(change_points_csv), subject)
    DataHelpers.draw_regions(ax, regions, alpha=1)


def _learning_curve_filters(
    meta: pd.DataFrame,
    selection: DatasetSelection,
) -> list[tuple[str, callable]]:
    meta_csv = meta.attrs.get("meta_csv_path")

    if selection.animal.lower() != "all":
        return [(selection.animal, lambda d, animal=selection.animal: d[d["animal"].astype(str) == animal].copy())]

    if selection.genotype.lower() != "all":
        return [
            (
                selection.genotype,
                lambda d, genotype=selection.genotype: DataHelpers.restrict_subjects(
                    d,
                    meta_csv,
                    genotypes=genotype,
                    subject_col="animal",
                    genotype_col="genotype",
                    attach_meta=True,
                ),
            )
        ]

    genotypes = available_genotypes(meta)
    return [
        (
            genotype,
            lambda d, genotype=genotype: DataHelpers.restrict_subjects(
                d,
                meta_csv,
                genotypes=genotype,
                subject_col="animal",
                genotype_col="genotype",
                attach_meta=True,
            ),
        )
        for genotype in genotypes
        if genotype != "all"
    ]


def _compute_session_curves(
    name: str,
    df: pd.DataFrame,
    normalized_points: int,
    span: int,
) -> dict:
    df = df.copy()
    df["is_correct"] = (df["success"] == 1).astype(int)
    sessions = sorted(df["session"].dropna().unique())

    output = {"name": name, "sessions": {}}
    for session in sessions:
        df_session = df[df["session"] == session]
        curves = []
        subjects = []

        for animal in sorted(df_session["animal"].dropna().astype(str).unique()):
            sub = df_session[df_session["animal"].astype(str) == animal].sort_values("trial")
            smooth = sub["is_correct"].ewm(span=span, adjust=False).mean().to_numpy()
            if len(smooth) < 2:
                continue

            x_raw = np.linspace(0, 1, len(smooth))
            x_target = np.linspace(0, 1, normalized_points)
            curves.append(np.interp(x_target, x_raw, smooth))
            subjects.append(animal)

        if not curves:
            continue

        mat = np.vstack(curves)
        output["sessions"][session] = {
            "subjects": subjects,
            "n": len(subjects),
            "length": normalized_points,
            "mean": mat.mean(axis=0),
            "sem": mat.std(axis=0) / np.sqrt(mat.shape[0]),
        }

    return output
