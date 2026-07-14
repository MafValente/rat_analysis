#%%
import os
import sys
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/Users/mafaldavalente/Documents/Mafalda_analysis"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from analysis.datasets import dataset_key, load_line_across_cohorts, load_dataset_selections
from GroupComparison.config import (
    ViewSpec, FilterConfig, PlotStyle, GroupComparisonConfig,
    OverlaySpec, JNDOverlaySpec,
)
from GroupComparison.prepare import (
    apply_filters,
    build_prepared,
    compute_jnd_individuals_by_view,
    compute_group_jnd_by_view,
)
from GroupComparison.plots import (
    style_axes,
    apply_50_tick_labels,
    plot_rt_on_ax,
    plot_mt_on_ax,
    plot_psy_on_ax,
)
import Helpers.DataHelpers as DataHelpers


print("USING COHORT OVERLAY PLOTS")

LINE = "CNTNAP2"
COHORT_SELECTION = "cohort3"   # "all", "cohort2", or ["cohort2", "cohort3"]
DATASET_SELECTIONS = None  # e.g. [("CNTNAP2", "cohort3"), ("SHANK3", "cohort1")]
BASE_DATA_DIR = os.path.join(ROOT, "DataFiles")
COHORT_SESSION_MIN = {"cohort2": 13}
SESSION_EQUALIZATION_EXCLUDE = {"cohort3": {"ASD0022"}}
MAKE_VIEWS_FIG = True
MAKE_ABLS_FIG = True
MAKE_JND_FIG = False
USE_PREPARED_CACHE = True
CACHE_DIR = os.path.join(ROOT, "GroupComparison", "_cache")


def _dataset_sort_key(name: str):
    line, _, cohort = str(name).partition(":")
    match = pd.Series([cohort]).str.extract(r"cohort(\d+)", expand=False).iloc[0]
    cohort_num = int(match) if pd.notna(match) else 10**9
    return (line.lower(), cohort_num, cohort.lower())


def make_selector(*, genotype: str, dataset_name: str):
    def _selector(df):
        out = df[df["genotype"] == genotype].copy()
        return out[out["dataset_key"] == dataset_name].copy()

    return _selector


def normalize_group_order(df):
    df_meta = df.dropna(subset=["genotype", "dataset_key"]).copy()
    genotypes = [g for g in ("wt", "het", "hom") if g in set(df_meta["genotype"].astype(str))]
    dataset_names = sorted(df_meta["dataset_key"].astype(str).unique(), key=_dataset_sort_key)
    return genotypes, dataset_names


def build_analysis_views(genotypes, dataset_names):
    return [
        ViewSpec(f"{genotype}_{dataset_name}", make_selector(genotype=genotype, dataset_name=dataset_name))
        for genotype in genotypes
        for dataset_name in dataset_names
    ]


def apply_cohort_specific_session_min(df, cohort_session_min):
    if not cohort_session_min:
        return df

    df = df.copy()
    session_num = pd.to_numeric(df["session"], errors="coerce")
    keep_mask = pd.Series(True, index=df.index)

    for key, session_min in cohort_session_min.items():
        key = str(key)
        if ":" in key:
            match_mask = df["dataset_key"] == key
        else:
            match_mask = df["cohort"] == key
        keep_mask.loc[match_mask] = session_num.loc[match_mask] >= session_min

    return df[keep_mask].copy()


def equalize_sessions_across_cohorts(df, dataset_names, exclude_animals_by_cohort=None):
    if len(dataset_names) <= 1:
        return df

    exclude_animals_by_cohort = exclude_animals_by_cohort or {}
    session_lists = {}
    for dataset_name in dataset_names:
        cohort_df = df[df["dataset_key"] == dataset_name].copy()
        excluded_animals = {str(x).strip() for x in exclude_animals_by_cohort.get(dataset_name, exclude_animals_by_cohort.get(dataset_name.split(":")[-1], set()))}
        if excluded_animals:
            cohort_df = cohort_df[~cohort_df["animal"].astype(str).str.strip().isin(excluded_animals)].copy()

        sessions = sorted(pd.to_numeric(
            cohort_df["session"], errors="coerce"
        ).dropna().astype(int).unique())
        if sessions:
            session_lists[dataset_name] = sessions

    if len(session_lists) <= 1:
        return df

    session_cap = min(len(sessions) for sessions in session_lists.values())
    allowed_sessions = {
        dataset_name: set(sessions[:session_cap])
        for dataset_name, sessions in session_lists.items()
    }

    dataset_keys = df["dataset_key"].astype(str)
    session_num = pd.to_numeric(df["session"], errors="coerce")
    keep_mask = pd.Series(True, index=df.index)

    for dataset_name, sessions in allowed_sessions.items():
        match_mask = dataset_keys == str(dataset_name)
        keep_mask.loc[match_mask] = session_num.loc[match_mask].isin(sessions)

    print(f"Equalizing datasets to first {session_cap} level-16 sessions:", ", ".join(f"{c} -> {session_cap}" for c in sorted(allowed_sessions)))
    return df[keep_mask].copy()


def cohort_style_map(cohorts):
    templates = [
        {"linestyle": "-", "marker": "o", "markerfacecolor": None},
        {"linestyle": "--", "marker": "o", "markerfacecolor": "white"},
        {"linestyle": ":", "marker": "s", "markerfacecolor": "white"},
        {"linestyle": "-.", "marker": "^", "markerfacecolor": None},
    ]
    return {cohort: templates[i % len(templates)].copy() for i, cohort in enumerate(cohorts)}


def _cohort_offsets(cohorts, step):
    if len(cohorts) == 1:
        return {cohorts[0]: 0.0}
    center = (len(cohorts) - 1) / 2
    return {cohort: step * (i - center) for i, cohort in enumerate(cohorts)}


def _title_with_single_cohort(base_title, cohorts):
    if len(cohorts) == 1:
        return f"{base_title} ({cohorts[0]})"
    return base_title


def add_views_jnd_inset(ax_parent, genotype, cohorts, group_jnd_by_view, cohort_styles, abl_colors, style):
    ax_inset = ax_parent.inset_axes([0.70, 0.15, 0.30, 0.30])
    offsets = _cohort_offsets(cohorts, 0.18)
    all_abls = set()

    for cohort in cohorts:
        key = f"{genotype}_{cohort}"
        dfj = group_jnd_by_view.get(key)
        if dfj is None or getattr(dfj, "empty", True):
            continue
        style_cfg = cohort_styles[cohort]

        for _, row in dfj.iterrows():
            abl = int(row["ABL"])
            all_abls.add(abl)
            c = abl_colors.get(abl, "gray")
            mfc = c if style_cfg["markerfacecolor"] is None else style_cfg["markerfacecolor"]
            ax_inset.errorbar(
                abl + offsets[cohort],
                float(row["mean"]),
                yerr=float(row["sem"]),
                fmt=style_cfg["marker"],
                color=c,
                markerfacecolor=mfc,
                markeredgecolor=c,
                markersize=6,
                elinewidth=1.2,
                capsize=3,
            )

    style_axes(ax_inset, style, title=None, xlabel="ABL", ylabel="JND (dB)")
    ax_inset.tick_params(axis="both", labelsize=max(8, style.tick_fs - 6))
    ax_inset.set_box_aspect(1)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)
    ax_inset.grid(False)
    ax_inset.set_xticks(sorted(all_abls))
    return ax_inset


def add_abls_jnd_inset(ax_parent, abl, genotypes, cohorts, group_jnd_by_view, genotype_colors, cohort_styles, style):
    ax_inset = ax_parent.inset_axes([0.70, 0.15, 0.30, 0.30])
    genotype_positions = {g: i for i, g in enumerate(genotypes)}
    offsets = _cohort_offsets(cohorts, 0.14)

    for genotype in genotypes:
        for cohort in cohorts:
            key = f"{genotype}_{cohort}"
            dfj = group_jnd_by_view.get(key)
            if dfj is None or getattr(dfj, "empty", True):
                continue
            sub = dfj[dfj["ABL"] == abl]
            if sub.empty:
                continue

            row = sub.iloc[0]
            c = genotype_colors.get(genotype, "gray")
            style_cfg = cohort_styles[cohort]
            mfc = c if style_cfg["markerfacecolor"] is None else style_cfg["markerfacecolor"]
            ax_inset.errorbar(
                genotype_positions[genotype] + offsets[cohort],
                float(row["mean"]),
                yerr=float(row["sem"]),
                fmt=style_cfg["marker"],
                color=c,
                markerfacecolor=mfc,
                markeredgecolor=c,
                markersize=6,
                elinewidth=1.2,
                capsize=3,
            )

    inset_label_fs = max(8, style.label_fs - 10)
    inset_tick_fs = max(7, style.tick_fs - 10)
    ax_inset.set_ylabel("JND (dB)", fontsize=inset_label_fs, color="black")
    ax_inset.yaxis.set_label_position("right")
    ax_inset.yaxis.tick_right()
    ax_inset.tick_params(axis="both", labelsize=inset_tick_fs)
    ax_inset.set_box_aspect(1)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["left"].set_visible(False)
    ax_inset.grid(False)
    ax_inset.set_xticks(list(genotype_positions.values()))
    ax_inset.set_xticklabels([g.upper() for g in genotypes])
    return ax_inset


def plot_views_overlay(prepared, group_jnd_by_view, genotypes, cohorts, cfg, style, overlay, cohort_styles):
    abls = sorted(set().union(*[
        set(prepared[name]["rt_group"]["ABL"].unique())
        for name in prepared
        if not prepared[name]["rt_group"].empty
    ]))
    abl_colors = {abl: f"C{i % 10}" for i, abl in enumerate(abls)}
    fig, axes = plt.subplots(len(genotypes), 3, figsize=(22, 7 * len(genotypes)), squeeze=False)

    for r, genotype in enumerate(genotypes):
        ax_rt, ax_mt, ax_psy = axes[r]

        for cohort in cohorts:
            key = f"{genotype}_{cohort}"
            tables = prepared.get(key)
            if not tables or tables["rt_group"].empty:
                continue
            style_cfg = cohort_styles[cohort]

            for abl in sorted(tables["rt_group"]["ABL"].unique()):
                c = abl_colors.get(abl, "gray")
                plot_rt_on_ax(ax_rt, tables, abl, c, cfg, **style_cfg)
                plot_mt_on_ax(ax_mt, tables, abl, c, cfg, **style_cfg)
                plot_psy_on_ax(ax_psy, tables, abl, c, cfg, **style_cfg)

        for abl in abls:
            if overlay.makefig1_chrono is not None:
                DataHelpers.overlay_makefig1_rt(ax_rt, abl, overlay.makefig1_chrono, color=overlay.overlay_color, zorder=-1)
        overlay_abls = [abl for abl in abls if abl != 50]
        if overlay.makefig1_data is not None and overlay_abls:
            if len(overlay_abls) > 1:
                DataHelpers.overlay_makefig1_psychometrics(
                    ax_psy, overlay.makefig1_data, abl=None,
                    color="black", show_individuals=False, use_abl_colors=False
                )
            else:
                DataHelpers.overlay_makefig1_psychometrics(
                    ax_psy, overlay.makefig1_data, abl=overlay_abls[0],
                    color="black", show_individuals=False, use_abl_colors=False
                )

        style_axes(ax_rt, style, _title_with_single_cohort(f"{genotype} - RT", cohorts), "ILD (dB)", "Mean RT (s)")
        style_axes(ax_mt, style, _title_with_single_cohort(f"{genotype} - MT", cohorts), "ILD (dB)", "Mean MT (s)")
        style_axes(ax_psy, style, _title_with_single_cohort(f"{genotype} - Psychometric", cohorts), "ILD (dB)", "P(Left)")
        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax_mt)
        apply_50_tick_labels(ax_psy)
        add_views_jnd_inset(ax_psy, genotype, cohorts, group_jnd_by_view, cohort_styles, abl_colors, style)

    abl_handles = [plt.Line2D([], [], color=abl_colors[abl], marker="o", linestyle="None") for abl in abls]
    cohort_handles = []
    for cohort in cohorts:
        style_cfg = cohort_styles[cohort]
        legend_marker = style_cfg["marker"] if len(cohorts) == 1 else None
        cohort_handles.append(
            plt.Line2D(
                [], [], color="black", linestyle=style_cfg["linestyle"], marker=legend_marker,
                markerfacecolor=("black" if style_cfg["markerfacecolor"] is None else style_cfg["markerfacecolor"]),
                markeredgecolor="black",
            )
        )
    overlay_handle = plt.Line2D(
        [], [], color="black", linestyle="-", marker="o",
        markerfacecolor="black", markeredgecolor="black",
    )

    fig.legend(abl_handles, [f"ABL {abl} dB" for abl in abls], loc="upper center", bbox_to_anchor=(0.35, 0.99), ncol=min(6, len(abl_handles)), fontsize=style.legend_fs)
    fig.legend(cohort_handles + [overlay_handle], cohorts + ["Headphone cohorts"], loc="upper center", bbox_to_anchor=(0.82, 0.99), ncol=min(4, len(cohort_handles) + 1), fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_abls_overlay(prepared, group_jnd_by_view, genotypes, cohorts, cfg, style, overlay, cohort_styles):
    genotype_colors = {"wt": "#4d4d4d", "het": "#7a8f28", "hom": "#c24a7a"}
    abl_rows = sorted(set().union(*[
        set(prepared[name]["rt_group"]["ABL"].unique())
        for name in prepared
        if not prepared[name]["rt_group"].empty
    ]))
    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.8 * len(abl_rows)), squeeze=False)

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r]

        for genotype in genotypes:
            for cohort in cohorts:
                key = f"{genotype}_{cohort}"
                tables = prepared.get(key)
                if not tables:
                    continue
                style_cfg = cohort_styles[cohort]
                color = genotype_colors.get(genotype, "gray")
                plot_rt_on_ax(ax_rt, tables, abl, color, cfg, **style_cfg)
                plot_mt_on_ax(ax_mt, tables, abl, color, cfg, **style_cfg)
                plot_psy_on_ax(ax_psy, tables, abl, color, cfg, **style_cfg)

        if overlay.makefig1_chrono is not None:
            DataHelpers.overlay_makefig1_rt(ax_rt, abl, overlay.makefig1_chrono, color=overlay.overlay_color, force_black=True, zorder=-1)
        if overlay.makefig1_data is not None and abl != 50:
            DataHelpers.overlay_makefig1_psychometrics(
                ax_psy, overlay.makefig1_data, abl=abl,
                color="black", show_individuals=False, use_abl_colors=False
            )

        style_axes(ax_rt, style, _title_with_single_cohort(f"ABL {abl} - RT", cohorts), "ILD (dB)", "Mean RT (s)")
        style_axes(ax_mt, style, _title_with_single_cohort(f"ABL {abl} - MT", cohorts), "ILD (dB)", "Mean MT (s)")
        style_axes(ax_psy, style, _title_with_single_cohort(f"ABL {abl} - Psychometric", cohorts), "ILD (dB)", "P(Left)")
        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax_mt)
        apply_50_tick_labels(ax_psy)
        add_abls_jnd_inset(ax_psy, abl, genotypes, cohorts, group_jnd_by_view, genotype_colors, cohort_styles, style)

    genotype_handles = [plt.Line2D([], [], color=genotype_colors[g], marker="o", linestyle="None") for g in genotypes]
    cohort_handles = []
    for cohort in cohorts:
        style_cfg = cohort_styles[cohort]
        legend_marker = style_cfg["marker"] if len(cohorts) == 1 else None
        cohort_handles.append(
            plt.Line2D(
                [], [], color="black", linestyle=style_cfg["linestyle"], marker=legend_marker,
                markerfacecolor=("black" if style_cfg["markerfacecolor"] is None else style_cfg["markerfacecolor"]),
                markeredgecolor="black",
            )
        )
    overlay_handle = plt.Line2D(
        [], [], color="black", linestyle="-", marker="o",
        markerfacecolor="black", markeredgecolor="black",
    )

    fig.legend(genotype_handles, [g.upper() for g in genotypes], loc="upper center", bbox_to_anchor=(0.35, 0.99), ncol=min(3, len(genotype_handles)), fontsize=style.legend_fs)
    fig.legend(cohort_handles + [overlay_handle], cohorts + ["Headphone cohorts"], loc="upper center", bbox_to_anchor=(0.82, 0.99), ncol=min(4, len(cohort_handles) + 1), fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_jnd_overlay_figure(jnd_indiv_by_view, genotypes, cohorts, cohort_styles, jnd_overlay, style):
    fig, axes = plt.subplots(1, len(genotypes), figsize=(4.6 * len(genotypes), 4.2), squeeze=False)
    axes = list(axes[0])

    old = jnd_overlay.old_jnd_data or {}
    old_jnds = old.get("jnds", {})
    old_animals = old.get("animals_with_mean", [])
    old_abls = [int(x) for x in old.get("ABLS", [])]

    new_abls = sorted(set().union(*[
        set(df["ABL"].astype(int).unique())
        for df in jnd_indiv_by_view.values()
        if df is not None and not df.empty
    ]))
    all_abls = sorted(set(old_abls) | set(new_abls))
    abl_colors = jnd_overlay.abl_color_map or {20: "C0", 40: "C1", 60: "C3"}
    cohort_offsets = _cohort_offsets(cohorts, 0.18)

    for ax, genotype in zip(axes, genotypes):
        for abl in all_abls:
            if abl in old_jnds:
                for animal in old_animals:
                    if animal in old_jnds[abl]:
                        ax.scatter(
                            abl + jnd_overlay.old_x_shift,
                            old_jnds[abl][animal],
                            facecolors="none",
                            edgecolors="black",
                            s=60,
                            lw=1,
                            alpha=0.9,
                        )

        for cohort in cohorts:
            key = f"{genotype}_{cohort}"
            dfj = jnd_indiv_by_view.get(key)
            if dfj is None or dfj.empty:
                continue

            style_cfg = cohort_styles[cohort]
            for _, row in dfj.iterrows():
                abl = int(row["ABL"])
                c = abl_colors.get(abl, "gray")
                mfc = c if style_cfg["markerfacecolor"] is None else style_cfg["markerfacecolor"]
                ax.scatter(
                    abl + cohort_offsets[cohort],
                    float(row["JND"]),
                    s=55,
                    marker=style_cfg["marker"],
                    facecolors=mfc,
                    edgecolors=c,
                    linewidth=1,
                    alpha=0.85,
                )

        style_axes(ax, style, title=f"{genotype} - JND", xlabel="ABL (dB)", ylabel="JND (dB)")
        ax.set_xticks(all_abls)
        if all_abls:
            ax.set_xlim(min(all_abls) - 5, max(all_abls) + 5)
        ax.set_box_aspect(1)

    abl_handles = [
        plt.Line2D([], [], color=abl_colors.get(abl, "gray"), marker="o", linestyle="None")
        for abl in all_abls
    ]
    cohort_handles = []
    for cohort in cohorts:
        style_cfg = cohort_styles[cohort]
        cohort_handles.append(
            plt.Line2D(
                [], [],
                color="black",
                linestyle="None",
                marker=style_cfg["marker"],
                markerfacecolor=("black" if style_cfg["markerfacecolor"] is None else style_cfg["markerfacecolor"]),
                markeredgecolor="black",
            )
        )
    old_handle = plt.Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="black")

    fig.legend(abl_handles, [f"ABL {abl} dB" for abl in all_abls], loc="upper center", bbox_to_anchor=(0.28, 0.99), ncol=min(6, len(abl_handles)), fontsize=style.legend_fs)
    fig.legend(cohort_handles + [old_handle], cohorts + ["old data"], loc="upper center", bbox_to_anchor=(0.80, 0.99), ncol=min(4, len(cohort_handles) + 1), fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _cache_path():
    dataset_tag = "mixed" if DATASET_SELECTIONS else f"{LINE}_{str(COHORT_SELECTION).replace(' ', '')}"
    return os.path.join(CACHE_DIR, f"groupcomparison_prepared_{dataset_tag}.pkl")


def load_overlays(base_data_dir, *, need_rt_psy_overlay, need_jnd_overlay):
    overlay = OverlaySpec(overlay_color="black")
    jnd_overlay = JNDOverlaySpec(abl_color_map={20: "C0", 40: "C1", 60: "C3"})

    if need_rt_psy_overlay:
        with open(os.path.join(base_data_dir, "Old Data/ILD_task/fig1_plot_data.pkl"), "rb") as f:
            makefig1_data = pickle.load(f)
        with open(os.path.join(base_data_dir, "Old Data/ILD_task/fig1_chrono_plot_data.pkl"), "rb") as f:
            makefig1_chrono = pickle.load(f)

        overlay = OverlaySpec(
            makefig1_data=makefig1_data,
            makefig1_chrono=makefig1_chrono,
            overlay_color="black",
        )

    if need_jnd_overlay:
        with open(os.path.join(base_data_dir, "Old Data/ILD_task/jnd_analysis_data.pkl"), "rb") as f:
            old_jnd_data = pickle.load(f)
        jnd_overlay = JNDOverlaySpec(
            old_jnd_data=old_jnd_data,
            abl_color_map={20: "C0", 40: "C1", 60: "C3"},
        )

    return overlay, jnd_overlay


def build_or_load_prepared(df_filtered, views, cfg, use_cache):
    cache_path = _cache_path()
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return (
            cached["prepared"],
            cached["jnd_indiv_by_view"],
            cached["group_jnd_by_view"],
        )

    prepared = build_prepared(df_filtered, views, cfg)
    jnd_indiv_by_view = compute_jnd_individuals_by_view(prepared, skip_abl=50)
    group_jnd_by_view = compute_group_jnd_by_view(jnd_indiv_by_view)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "prepared": prepared,
                    "jnd_indiv_by_view": jnd_indiv_by_view,
                    "group_jnd_by_view": group_jnd_by_view,
                },
                f,
            )

    return prepared, jnd_indiv_by_view, group_jnd_by_view


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

genotypes, cohorts = normalize_group_order(df_plot)
views = build_analysis_views(genotypes, cohorts)
cohort_styles = cohort_style_map(cohorts)


mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]

cfg = GroupComparisonConfig(
    error_mode="individuals",
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

df_filtered = apply_filters(df_plot, fcfg)
df_filtered = apply_cohort_specific_session_min(df_filtered, COHORT_SESSION_MIN)
df_filtered = equalize_sessions_across_cohorts(df_filtered, cohorts, SESSION_EQUALIZATION_EXCLUDE)
prepared, jnd_indiv_by_view, group_jnd_by_view = build_or_load_prepared(
    df_filtered, views, cfg, USE_PREPARED_CACHE
)
overlay, jnd_overlay = load_overlays(
    BASE_DATA_DIR,
    need_rt_psy_overlay=(MAKE_VIEWS_FIG or MAKE_ABLS_FIG),
    need_jnd_overlay=MAKE_JND_FIG,
)

fig1 = None
fig2 = None
fig_jnd = None
out1 = None
out2 = None
out_jnd = None

if MAKE_VIEWS_FIG:
    fig1 = plot_views_overlay(prepared, group_jnd_by_view, genotypes, cohorts, cfg, style, overlay, cohort_styles)
    out1 = {"figure": fig1, "prepared": prepared, "jnd_indiv_by_view": jnd_indiv_by_view, "group_jnd_by_view": group_jnd_by_view}

if MAKE_ABLS_FIG:
    fig2 = plot_abls_overlay(prepared, group_jnd_by_view, genotypes, cohorts, cfg, style, overlay, cohort_styles)
    out2 = {"figure": fig2, "prepared": prepared, "jnd_indiv_by_view": jnd_indiv_by_view, "group_jnd_by_view": group_jnd_by_view}

if MAKE_JND_FIG:
    fig_jnd = plot_jnd_overlay_figure(jnd_indiv_by_view, genotypes, cohorts, cohort_styles, jnd_overlay, style)
    out_jnd = {"figure": fig_jnd, "jnd_indiv_by_view": jnd_indiv_by_view}

plt.show()

# %%
