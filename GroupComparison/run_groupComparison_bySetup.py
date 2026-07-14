#%%
import os
import sys
import pickle
from dataclasses import replace
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/Users/mafaldavalente/Documents/Mafalda_analysis"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datasets import load_dataset_selections, load_line_across_cohorts
from GroupComparison.config import (
    ViewSpec, FilterConfig, PlotStyle, GroupComparisonConfig,
    OverlaySpec,
)
from GroupComparison.prepare import (
    apply_filters,
    build_prepared,
    compute_jnd_individuals_by_view,
    compute_group_jnd_by_view,
)
from GroupComparison.layouts import plot_views_3x3, plot_abls_4x3
from Helpers.DataHelpers import prepare_data


print("USING DOMINANT-SETUP GROUP COMPARISON")

LINE = "SHANK3"
COHORT_SELECTION = "cohort1"   # "all", "cohort2", or ["cohort2", "cohort3"]

DATASET_SELECTIONS = [
    ("CNTNAP2", "cohort3"),
    ("CNTNAP2", "cohort2"),
    ("SHANK3", "cohort1"),
]

BASE_DATA_DIR = os.path.join(ROOT, "DataFiles")
SETUP_COL = "box"
MIN_ANIMALS_PER_SETUP = 1
LAYOUT = "abls_4x3"        # "views_3x3" or "abls_4x3"
MAKE_PER_SETUP_ANIMAL_FIGURES = True


def _normalize_setup_value(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""
    try:
        numeric = float(value)
    except ValueError:
        return value
    return str(int(numeric)) if numeric.is_integer() else value


def assign_dominant_setups(df: pd.DataFrame, setup_col: str = SETUP_COL) -> tuple[pd.DataFrame, pd.DataFrame]:
    if setup_col not in df.columns:
        raise KeyError(f"'{setup_col}' column not found in dataframe.")

    work = df.copy()
    work["animal"] = work["animal"].astype(str).str.strip()
    work[setup_col] = work[setup_col].map(_normalize_setup_value)
    work = work[work[setup_col] != ""].copy()

    usage = (
        work.groupby(["animal", setup_col])
        .size()
        .reset_index(name="n_trials")
    )
    if usage.empty:
        raise ValueError("No valid animal/setup rows were found after filtering.")

    usage["max_trials_for_animal"] = usage.groupby("animal")["n_trials"].transform("max")
    dominant = usage[usage["n_trials"] == usage["max_trials_for_animal"]].copy()
    dominant = dominant.rename(columns={setup_col: "dominant_setup"})

    assigned = work.merge(
        dominant[["animal", "dominant_setup", "n_trials"]],
        left_on=["animal", setup_col],
        right_on=["animal", "dominant_setup"],
        how="inner",
    )
    assigned["setup_group"] = assigned["dominant_setup"]

    return assigned, dominant


def summarize_setup_groups(dominant: pd.DataFrame, min_animals_per_setup: int = MIN_ANIMALS_PER_SETUP) -> list[str]:
    counts = (
        dominant.groupby("dominant_setup")["animal"]
        .nunique()
        .sort_index(key=lambda idx: [float(x) if str(x).replace(".", "", 1).isdigit() else str(x) for x in idx])
    )
    setups = [setup for setup, n in counts.items() if n >= min_animals_per_setup]
    print("Animals per dominant setup:", counts.to_dict())
    if min_animals_per_setup > 1:
        print(f"Keeping setups with at least {min_animals_per_setup} animals:", setups)
    return setups


def print_setup_animal_summary(df_assigned: pd.DataFrame, setup_col: str = "setup_group") -> None:
    summary = (
        df_assigned.groupby(setup_col)["animal"]
        .nunique()
        .sort_index(key=lambda idx: [float(x) if str(x).replace(".", "", 1).isdigit() else str(x) for x in idx])
    )
    print("\nAnimals contributing to each setup:")
    for setup, n_animals in summary.items():
        print(f"  setup {setup}: n={int(n_animals)} animals")


def make_selector(*, setup: str):
    def _selector(df):
        return df[df["setup_group"] == setup].copy()
    return _selector


def build_setup_views(setups: list[str]) -> list[ViewSpec]:
    return [ViewSpec(f"setup {setup}", make_selector(setup=setup)) for setup in setups]


def make_animal_selector(*, animal: str):
    def _selector(df):
        return df[df["animal"].astype(str).str.strip() == animal].copy()
    return _selector


def build_animal_views(animals: list[str]) -> list[ViewSpec]:
    return [ViewSpec(animal, make_animal_selector(animal=animal)) for animal in animals]


def create_animal_figures_by_setup(
    df_assigned: pd.DataFrame,
    setups: list[str],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    overlay: OverlaySpec,
) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    animal_cfg = replace(cfg, error_mode="individuals")

    for setup in setups:
        df_setup = df_assigned[df_assigned["setup_group"] == setup].copy()
        animals = sorted(df_setup["animal"].dropna().astype(str).str.strip().unique())
        if not animals:
            continue

        views = build_animal_views(animals)
        prepared = build_prepared(df_setup, views, animal_cfg)
        jnd_indiv = compute_jnd_individuals_by_view(prepared, skip_abl=50)
        group_jnd = compute_group_jnd_by_view(jnd_indiv)
        view_colors = {view.name: f"C{i % 10}" for i, view in enumerate(views)}

        fig = plot_views_3x3(
            prepared=prepared,
            views=views,
            cfg=animal_cfg,
            style=style,
            overlay=overlay,
            group_jnd_by_view=group_jnd,
            view_colors=view_colors,
            add_jnd_inset=True,
        )
        fig.suptitle(f"Setup {setup} - contributing animals", fontsize=style.title_fs + 2, y=0.995)
        figures[setup] = fig

    return figures


if DATASET_SELECTIONS:
    df_all, meta_all, dataset_info = load_dataset_selections(
        selections=DATASET_SELECTIONS,
        base_dir=BASE_DATA_DIR,
        require_meta=False,
    )
else:
    df_all, meta_all, dataset_info = load_line_across_cohorts(
        line=LINE,
        base_dir=BASE_DATA_DIR,
        cohorts=COHORT_SELECTION,
        require_meta=False,
    )

mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]

with open(os.path.join(BASE_DATA_DIR, "Old Data/ILD_task/fig1_plot_data.pkl"), "rb") as f:
    makefig1_data = pickle.load(f)
with open(os.path.join(BASE_DATA_DIR, "Old Data/ILD_task/fig1_chrono_plot_data.pkl"), "rb") as f:
    makefig1_chrono = pickle.load(f)

overlay = OverlaySpec(
    makefig1_data=makefig1_data,
    makefig1_chrono=makefig1_chrono,
    overlay_color="black",
)

cfg = GroupComparisonConfig(
    error_mode="sem",
    skip_psy_fits=(50,),
    ild_shift_for_abl50=True,
)
fcfg = FilterConfig(
    training_min=16,
    session_min=0,
    drop_repeat_trials=True,
    session_type_values=[1],
    stim_dur_values=[6000],
    sessiontype_or_stimdur="or",
)
style = PlotStyle()

df_prepared = prepare_data(df_all.copy(), session_col="session", trial_col="trial")
if "trial_is_repeat" in df_prepared.columns:
    df_prepared = df_prepared[df_prepared["trial_is_repeat"] == False].copy()
if "training_level" in df_prepared.columns:
    df_prepared = df_prepared[df_prepared["training_level"] == 16].copy()

sess = pd.to_numeric(df_prepared["session_type"], errors="coerce")
sd = pd.to_numeric(df_prepared["stim_dur"], errors="coerce")
df_prepared = df_prepared[(sess == 1) | (sd == 6000)].copy()

df_filtered = apply_filters(
    df_prepared,
    FilterConfig(
        training_min=16,
        session_min=0,
        drop_repeat_trials=False,
        session_type_values=None,
        stim_dur_values=None,
        sessiontype_or_stimdur="or",
    ),
)
df_assigned, dominant_setup_table = assign_dominant_setups(df_filtered, setup_col=SETUP_COL)
setups = summarize_setup_groups(dominant_setup_table, min_animals_per_setup=MIN_ANIMALS_PER_SETUP)

if not setups:
    raise ValueError("No setup groups met the minimum animal count.")

df_assigned = df_assigned[df_assigned["setup_group"].isin(setups)].copy()
print_setup_animal_summary(df_assigned)
views = build_setup_views(setups)
view_colors = {view.name: f"C{i % 10}" for i, view in enumerate(views)}

prepared = build_prepared(df_assigned, views, cfg)
jnd_indiv_by_view = compute_jnd_individuals_by_view(prepared, skip_abl=50)
group_jnd_by_view = compute_group_jnd_by_view(jnd_indiv_by_view)

if LAYOUT == "views_3x3":
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
elif LAYOUT == "abls_4x3":
    fig_main = plot_abls_4x3(
        prepared=prepared,
        views=views,
        cfg=cfg,
        style=style,
        overlay=overlay,
        view_colors=view_colors,
        group_jnd_by_view=group_jnd_by_view,
        add_inset=True,
    )
else:
    raise ValueError(f"Unsupported LAYOUT={LAYOUT!r}")


#%%
setup_animal_figures = {}
if MAKE_PER_SETUP_ANIMAL_FIGURES:
    setup_animal_figures = create_animal_figures_by_setup(
        df_assigned=df_assigned,
        setups=setups,
        cfg=cfg,
        style=style,
        overlay=overlay,
    )

out = {
    "figure": fig_main,
    "prepared": prepared,
    "jnd_indiv_by_view": jnd_indiv_by_view,
    "group_jnd_by_view": group_jnd_by_view,
    "dominant_setup_table": dominant_setup_table,
    "df_assigned": df_assigned,
    "setup_animal_figures": setup_animal_figures,
}

plt.show()

# %%
