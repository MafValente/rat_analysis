#%%
# run_stimdurComparison.py
# ==============================================================
# StimDur comparison (session_type == 2)
# Layout: 4x3 (rows=ABL; cols=RT/MT/Psychometric)
# Lines: stim_dur
# Views can be pooled genotypes or genotype-by-dataset combinations.
# ==============================================================

import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/Users/mafaldavalente/Documents/Mafalda_analysis"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datasets import dataset_key, load_line_across_cohorts, load_dataset_selections
from StimDur.config import (
    ViewSpec, FilterConfig, PlotStyle, StimDurComparisonConfig,
    make_stimdur_specs,
)
from StimDur.runner import run_stimdur_comparison
from StimDur.layouts import (
    plot_genotypes_4x3_for_stimdur,
    plot_absild_perf_across_stimdur_1x3_for_view,
    plot_absild_perf_3x5_all_genotypes,
    plot_kreg_4x3_by_abl_for_view,
)
from StimDur.prepare import apply_filters


print("USING MULTI-DATASET STIMDUR COMPARISON")

LINE = "CNTNAP2"   # set to None to keep all lines listed in DATASET_SELECTIONS
COHORT_SELECTION = "cohort3"   # "all", "cohort2", or ["cohort2", "cohort3"]
DATASET_SELECTIONS = [
    ("CNTNAP2", "cohort3"),
    ("CNTNAP2", "cohort2"),
    ("SHANK3", "cohort1"),
]
ANIMAL_SELECTION = "ASD0026"  # e.g. "ASD0047"; set to None for no animal restriction
VIEW_MODE = "genotype_dataset"   # "genotype" or "genotype_dataset"

BASE_DATA_DIR = os.path.join(ROOT, "DataFiles")
COHORT_CSV = None   # e.g. "merged_ASD0024.csv" for a specific single-dataset run


def _dataset_sort_key(name: str):
    line, _, cohort = str(name).partition(":")
    match = pd.Series([cohort]).str.extract(r"cohort(\d+)", expand=False).iloc[0]
    cohort_num = int(match) if pd.notna(match) else 10**9
    return (line.lower(), cohort_num, cohort.lower())


def _label_from_dataset_name(dataset_name: str) -> str:
    line, _, cohort = str(dataset_name).partition(":")
    return f"{line} {cohort}"


def make_single_animal_selector(*, animal: str):
    def _selector(df):
        return df[df["animal"].astype(str).str.strip() == animal].copy()
    return _selector


def make_genotype_selector(*, genotype: str):
    def _selector(df):
        return df[df["genotype"].astype(str).str.strip() == genotype].copy()
    return _selector


def make_genotype_dataset_selector(*, genotype: str, dataset_name: str):
    def _selector(df):
        out = df[df["genotype"].astype(str).str.strip() == genotype].copy()
        return out[out["dataset_key"].astype(str) == dataset_name].copy()
    return _selector


def normalize_group_order(df):
    df_meta = df.dropna(subset=["genotype", "dataset_key"]).copy()
    df_meta["genotype"] = df_meta["genotype"].astype(str).str.strip()
    genotypes = [g for g in ("wt", "het", "hom") if g in set(df_meta["genotype"])]
    dataset_names = sorted(df_meta["dataset_key"].astype(str).unique(), key=_dataset_sort_key)
    return genotypes, dataset_names


def build_views(df: pd.DataFrame, view_mode: str) -> list[ViewSpec]:
    animals = sorted(df["animal"].dropna().astype(str).str.strip().unique())
    if len(animals) == 1:
        animal = animals[0]
        return [ViewSpec(animal, make_single_animal_selector(animal=animal))]

    genotypes, dataset_names = normalize_group_order(df)
    if not genotypes:
        raise ValueError("No genotype metadata found in the selected StimDur datasets.")

    if view_mode == "genotype":
        return [ViewSpec(genotype, make_genotype_selector(genotype=genotype)) for genotype in genotypes]

    if view_mode == "genotype_dataset":
        views: list[ViewSpec] = []
        for genotype in genotypes:
            for dataset_name in dataset_names:
                mask = (
                    df["genotype"].astype(str).str.strip().eq(genotype)
                    & df["dataset_key"].astype(str).eq(dataset_name)
                )
                if not mask.any():
                    continue
                views.append(
                    ViewSpec(
                        f"{genotype}_{dataset_name}",
                        make_genotype_dataset_selector(genotype=genotype, dataset_name=dataset_name),
                    )
                )
        if not views:
            raise ValueError("No genotype-by-dataset views were available after loading the selected datasets.")
        return views

    raise ValueError(f"Unsupported VIEW_MODE={view_mode!r}")


def build_view_labels(views: list[ViewSpec]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for view in views:
        parts = str(view.name).split("_", 1)
        if len(parts) == 2 and ":" in parts[1]:
            genotype, dataset_name = parts
            labels[view.name] = f"{genotype.upper()} {_label_from_dataset_name(dataset_name)}"
        else:
            labels[view.name] = str(view.name).upper() if str(view.name) in {"wt", "het", "hom"} else str(view.name)
    return labels


def filter_dataset_selections(
    selections: list[tuple[str, str]] | None,
    *,
    line: str | None,
    cohorts,
) -> list[tuple[str, str]] | None:
    if not selections:
        return selections

    out = list(selections)

    if line is not None:
        out = [(sel_line, sel_cohort) for sel_line, sel_cohort in out if str(sel_line) == str(line)]

    if cohorts is None or cohorts == "all":
        return out

    if isinstance(cohorts, str):
        allowed = {cohorts}
    else:
        allowed = {str(cohort) for cohort in cohorts}

    return [(sel_line, sel_cohort) for sel_line, sel_cohort in out if str(sel_cohort) in allowed]


DATASET_SELECTIONS = filter_dataset_selections(
    DATASET_SELECTIONS,
    line=LINE,
    cohorts=COHORT_SELECTION,
)

if DATASET_SELECTIONS is not None and len(DATASET_SELECTIONS) == 0:
    raise ValueError("DATASET_SELECTIONS became empty after applying LINE/COHORT_SELECTION filters.")


if DATASET_SELECTIONS:
    df_all, meta_all, dataset_info = load_dataset_selections(
        selections=DATASET_SELECTIONS,
        base_dir=BASE_DATA_DIR,
        cohort_file=COHORT_CSV,
        require_meta=False,
    )
else:
    df_all, meta_all, dataset_info = load_line_across_cohorts(
        line=LINE,
        base_dir=BASE_DATA_DIR,
        cohorts=COHORT_SELECTION,
        cohort_file=COHORT_CSV,
        require_meta=False,
    )
    df_all = df_all.copy()
    df_all["dataset_key"] = df_all.apply(lambda row: dataset_key(row["line"], row["cohort"]), axis=1)
    if not meta_all.empty:
        meta_all = meta_all.copy()
        meta_all["dataset_key"] = meta_all.apply(lambda row: dataset_key(row["line"], row["cohort"]), axis=1)

if dataset_info.get("missing_meta_dataset_keys"):
    print("Missing sex/genotype metadata for datasets:", ", ".join(dataset_info["missing_meta_dataset_keys"]))
elif dataset_info.get("missing_meta_cohorts"):
    print("Missing sex/genotype metadata for cohorts:", ", ".join(dataset_info["missing_meta_cohorts"]))

if not meta_all.empty:
    usable_dataset_names = sorted(meta_all["dataset_key"].dropna().astype(str).unique(), key=_dataset_sort_key)
    df_plot = df_all[df_all["dataset_key"].isin(usable_dataset_names)].copy()
else:
    df_plot = df_all.copy()

if ANIMAL_SELECTION is not None:
    animal_id = str(ANIMAL_SELECTION).strip()
    df_plot = df_plot[df_plot["animal"].astype(str).str.strip() == animal_id].copy()
    if df_plot.empty:
        raise ValueError(f"No rows found for ANIMAL_SELECTION={animal_id!r} after dataset filters.")


mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]


# ----------------- views -----------------
views = build_views(df_plot, VIEW_MODE)
view_pretty = build_view_labels(views)
view_colors = {view.name: f"C{i % 10}" for i, view in enumerate(views)}


# ----------------- stim_dur traces -----------------
STIMDUR_COL = "short_duration"
STIM_DURS = [8, 15, 16, 32, 60, 64, 120, 0]
stimdur_pretty = {
    "8": "SD = 8 ms",
    "15": "SD = 15 ms",
    "16": "SD = 16 ms",
    "32": "SD = 32 ms",
    "60": "SD = 60 ms",
    "64": "SD = 64 ms",
    "120": "SD = 120 ms",
    "0": "SD = RT",
}

# ----------------- configs -----------------
cfg = StimDurComparisonConfig(
    error_mode="individuals",
    skip_psy_fits=(50,),
    ild_shift_for_abl50=True,
)
fcfg = FilterConfig(
    training_min=16,
    session_min=13,
    drop_repeat_trials=True,
    session_type_values=[2],
)
style = PlotStyle()

df_plot_filtered = apply_filters(df_plot, fcfg)
available_stim_durs = {
    int(float(x))
    for x in pd.to_numeric(df_plot_filtered[STIMDUR_COL], errors="coerce").dropna().unique()
}
active_stim_durs = [sd for sd in STIM_DURS if int(sd) in available_stim_durs]
if not active_stim_durs:
    raise ValueError(
        f"No stim durations from STIM_DURS={STIM_DURS!r} were present after filtering."
    )

stimdur_specs = make_stimdur_specs(active_stim_durs, stim_dur_col=STIMDUR_COL)

PALETTE = [
    "#B2A706", "#0072B2", "#56B4E9", "#E69F00",
    "#009E73", "#D55E00", "#CC79A7", "#4D4D4D",
]
stimdur_colors = {s.name: PALETTE[i % len(PALETTE)] for i, s in enumerate(stimdur_specs)}


# ==============================================================
# RUN (one 4x3 figure per view)
# ==============================================================
out = run_stimdur_comparison(
    df=df_plot,
    views=views,
    stimdur_specs=stimdur_specs,
    cfg=cfg,
    fcfg=fcfg,
    style=style,
    stimdur_colors=stimdur_colors,
    stimdur_pretty=stimdur_pretty,
    show=True,
)

plt.show()


#%%
# ==============================================================
# RUN (one 4x3 figure per StimDur, lines = views)
# ==============================================================
figs_by_stimdur = {}
for s in stimdur_specs:
    figs_by_stimdur[s.name] = plot_genotypes_4x3_for_stimdur(
        prepared=out["prepared"],
        group_jnd=out["group_jnd"],
        views=views,
        stimdur_name=s.name,
        view_colors=view_colors,
        cfg=cfg,
        style=style,
        stimdur_pretty=stimdur_pretty,
        view_pretty=view_pretty,
    )
    plt.show()


# %%
# ================================================================
# GROUP (one 3x1 figure) performance for each ILD over the StimDurs
# ================================================================
for v in views:
    fig = plot_absild_perf_across_stimdur_1x3_for_view(
        prepared_for_view=out["prepared"][v.name],
        stimdur_specs=stimdur_specs,
        view_name=view_pretty.get(v.name, v.name),
        cfg=cfg,
        style=style,
        stimdur_pretty=stimdur_pretty,
        abls=[20, 40, 60],
    )
    plt.show()


# %%
for v in views:
    fig = plot_kreg_4x3_by_abl_for_view(
        df_view=out["df_by_view"][v.name],
        view_name=view_pretty.get(v.name, v.name),
        stimdur_specs=stimdur_specs,
        stimdur_col=STIMDUR_COL,
        stimdur_colors=stimdur_colors,
        stimdur_pretty=stimdur_pretty,
        abls=(20, 40, 60),
        xlim=(0.0, 0.5),
        debug=True,
    )
    plt.show()


# %%
# ================================================================
# GROUP (3x5): rows are ABLs, columns are |ILD|, lines are views
# ================================================================
fig = plot_absild_perf_3x5_all_genotypes(
    prepared=out["prepared"],
    views=views,
    stimdur_specs=stimdur_specs,
    abls=[20, 40, 60],
    style=style,
    stimdur_pretty=stimdur_pretty,
    view_colors=view_colors,
    view_pretty=view_pretty,
    absilds=[1, 2, 4, 8, 16],
)
plt.show()

# %%
