# analysis/plotting/groupcomparison/layouts.py

import matplotlib.pyplot as plt
import Helpers.DataHelpers as DataHelpers


from .style import style_axes, relabel_ticks_minus18_plus18_as_50, _xlim_for_abl
from .traces import plot_rt, plot_mt, plot_psy
from .jnd import compute_group_jnd


from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

def _apply_abl50_axis_format(ax, *, xlim, ticks, ticklabels=None):
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    if ticklabels is None:
        # default: use numeric ticks
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    else:
        mapping = {float(t): str(lbl) for t, lbl in zip(ticks, ticklabels)}
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: mapping.get(float(x), "")))


def plot_views_as_rows_1x3(
    bundle,
    *,
    makefig1_data=None,
    makefig1_chrono=None,
    old_jnd_data=None,
    show_old_overlays=True,
    show_jnd_inset=True,
    show_old_vs_new_jnd_scatter=False,
    SKIP_PSY_FITS=frozenset({50}),
    TITLE_FONTSIZE=24,
    LABEL_FONTSIZE=25,
    TICK_FONTSIZE=24,
    LEGEND_FONTSIZE=16,
    TITLE_PAD=16,
):
    """
    Your 1x3-per-view layout:
    - rows = views (e.g., wt only; or male/female)
    - columns = RT/MT/Psychometric
    - colors = ABL colors within each axis
    - optional old overlays + JND inset + old-vs-new JND scatter
    """

    prepared = bundle["prepared"]
    views = bundle["views"]         # list of ViewSpec
    error_mode = bundle["cfg"].error_mode

    unique_abls = sorted(set().union(*[set(p["rt_group"]["ABL"].unique()) for p in prepared.values()]))
    ABL_COLORS = {abl: f"C{i % 10}" for i, abl in enumerate(unique_abls)}

    jnd_figs = []  

    fig, axes = plt.subplots(len(views), 3, figsize=(22, 7 * len(views)), squeeze=False)

    for r, v in enumerate(views):
        view_name = v.name
        tables = prepared[view_name]
        abls = sorted(tables["rt_group"]["ABL"].unique())
        ax_rt, ax_mt, ax_psy = axes[r]

        for abl in abls:
            color = ABL_COLORS.get(abl, "gray")

            plot_rt(ax_rt, abl, tables, color, error_mode)
            plot_mt(ax_mt, abl, tables, color, error_mode)
            plot_psy(ax_psy, abl, tables, color, error_mode, skip_fit=SKIP_PSY_FITS)

            # overlays
            if show_old_overlays and makefig1_chrono is not None:
                DataHelpers.overlay_makefig1_rt(ax_rt, abl, makefig1_chrono, color="black", zorder=-1)

                # colored squares from old dataset (your code)
                try:
                    out = DataHelpers.extract_rt_points(makefig1_chrono, abl)
                    if out is not None:
                        x_ref, y_ref, sem_ref = out
                        x_ref = DataHelpers.shift_ILD_for_ABL50(x_ref)
                        ax_rt.errorbar(
                            x_ref, y_ref, yerr=sem_ref,
                            fmt="s", color=color, markersize=7,
                            linewidth=0, elinewidth=1.5, capsize=3,
                            alpha=1, zorder=5,
                        )
                except Exception as e:
                    print(f"[warn] Could not overlay colored RT points for ABL {abl}: {e}")

            if show_old_overlays and makefig1_data is not None and abl != 50:
                DataHelpers.overlay_makefig1_psychometrics(
                    ax_psy, makefig1_data, abl, color="black",
                    show_individuals=False, use_abl_colors=False
                )

        # axis styling
        style_axes(ax_rt,  f"{view_name} — RT",          "ILD (dB)", "Mean RT (s)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)
        style_axes(ax_mt,  f"{view_name} — MT",          "ILD (dB)", "Mean MT (s)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)
        style_axes(ax_psy, f"{view_name} — Psychometric","ILD (dB)", "P(Right)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)

        for ax in [ax_rt, ax_mt, ax_psy]:
            relabel_ticks_minus18_plus18_as_50(ax)

        # JND inset (uses df_view saved in prepared)
        if show_jnd_inset:
            df_view = tables.get("df_view", None)
            if df_view is not None and len(df_view) > 0:
                group_jnd, all_jnds_df = compute_group_jnd(df_view, skip_ABL=50)

                ax_inset = ax_psy.inset_axes([0.72, 0.15, 0.33, 0.33])

                for _, row in group_jnd.iterrows():
                    abl = row["ABL"]
                    color = ABL_COLORS.get(abl, "gray")
                    ax_inset.errorbar(
                        abl, row["mean"], yerr=row["sem"],
                        fmt="o", color=color, markersize=7,
                        elinewidth=1.5, capsize=4,
                        markeredgecolor="black", markeredgewidth=1,
                    )

                style_axes(ax_inset, None, "ABL", "JND (dB)",
                           TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE - 6, TITLE_PAD)
                ax_inset.set_xticks(sorted(group_jnd["ABL"].unique()))
                ax_inset.spines["top"].set_visible(False)
                ax_inset.spines["right"].set_visible(False)
                ax_inset.grid(False)

                if show_old_vs_new_jnd_scatter and old_jnd_data is not None:
                    from .jnd import plot_old_vs_new_jnd_scatter
                    jnd_figs.append(   # <--- CHANGE: append returned fig
                        plot_old_vs_new_jnd_scatter(
                            old_jnd_data, all_jnds_df,
                            LABEL_FONTSIZE=LABEL_FONTSIZE,
                            TICK_FONTSIZE=TICK_FONTSIZE
                        )
                    )

        ax_rt.set_xlim(0, 19)
        ax_mt.set_xlim(0, 19)
        ax_psy.set_xlim(-19, 19)

    # global legend = ABL colors
    handles, labels = [], []
    for abl, color in ABL_COLORS.items():
        handles.append(plt.Line2D([], [], color=color, marker="o", linestyle="None"))
        labels.append(f"ABL {abl} dB")

    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig, jnd_figs  


def plot_abls_as_rows_4x3(
    bundle,
    *,
    makefig1_data=None,
    makefig1_chrono=None,
    show_old_overlays=True,
    SKIP_PSY_FITS=frozenset({50}),
    TITLE_FONTSIZE=24,
    LABEL_FONTSIZE=25,
    TICK_FONTSIZE=24,
    LEGEND_FONTSIZE=16,
    TITLE_PAD=16,
    xlim_rt=None,
    xlim_mt=None,
    xlim_psy=None,
):
    """
    Your 4x3-per-ABL layout:
    - rows = ABLs
    - columns = RT/MT/Psychometric
    - colors = views (genotypes/sex)
    """


    prepared = bundle["prepared"]
    views = bundle["views"]
    abl_rows = bundle["abl_rows"]
    error_mode = bundle["cfg"].error_mode

    view_colors = {v.name: (v.color if v.color is not None else "C0") for v in views}

    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.8 * len(abl_rows)), squeeze=False)

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r]

        for v in views:
            name = v.name
            tables = prepared[name]
            color = view_colors[name]

            plot_rt(ax_rt,  abl, tables, color, error_mode)
            plot_mt(ax_mt,  abl, tables, color, error_mode)
            plot_psy(ax_psy, abl, tables, color, error_mode, skip_fit=SKIP_PSY_FITS)

        style_axes(ax_rt,  f"ABL {abl} — RT",          "ILD (dB)", "Mean RT (s)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)
        style_axes(ax_mt,  f"ABL {abl} — MT",          "ILD (dB)", "Mean MT (s)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)
        style_axes(ax_psy, f"ABL {abl} — Psychometric","ILD (dB)", "P(Right)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)

        if show_old_overlays and makefig1_chrono is not None:
            DataHelpers.overlay_makefig1_rt(ax_rt, abl, makefig1_chrono)

        if show_old_overlays and makefig1_data is not None and abl != 50:
            DataHelpers.overlay_makefig1_psychometrics(
                ax_psy, makefig1_data, abl, color="black",
                show_individuals=False, use_abl_colors=False
            )        

        abl_i = int(abl)

        relabel_ticks_minus18_plus18_as_50(ax_rt)
        relabel_ticks_minus18_plus18_as_50(ax_mt)
        relabel_ticks_minus18_plus18_as_50(ax_psy)

        rt_xlim  = _xlim_for_abl(abl_i, xlim_rt,  default=(0, 19))
        mt_xlim  = _xlim_for_abl(abl_i, xlim_mt,  default=(0, 19))
        psy_xlim = _xlim_for_abl(abl_i, xlim_psy, default=(-19, 19))

        ax_rt.set_xlim(*rt_xlim)
        ax_mt.set_xlim(*mt_xlim)
        ax_psy.set_xlim(*psy_xlim)

        ax_rt.set_autoscale_on(False)
        ax_mt.set_autoscale_on(False)


        # reduce tick density when zoomed (especially ABL 50)
        ax_rt.xaxis.set_major_locator(MaxNLocator(nbins=4))
        abl_i = int(abl)

        # ---- SPECIAL FORMATTING FOR ABL 50 ----
        if abl_i == 50:
            # RT: show only the "50" tick at x=18 (or whatever you prefer)
            # Option A (common): only show the 50 location:
            _apply_abl50_axis_format(
                ax_rt,
                xlim=(17, 19),
                ticks=[18],
                ticklabels=["50"],
            )

            # MT: you currently have xlim ~(-10,0) because you passed it;
            # choose what you want for ABL50 MT:
            _apply_abl50_axis_format(
                ax_mt,
                xlim=(17, 19),
                ticks=[18],
                ticklabels=["50"],
            )

            # Psychometric: ABL50 usually is weird (points at ±18).
            # If you want the same global limits, just keep them:
            # (or specify a custom view here too)
            # Example: show full range but label 18 as 50:
            _apply_abl50_axis_format(
                ax_psy,
                xlim=(-19, 19),
                ticks=[-18, 0, 18],
                ticklabels=["-50", "0", "50"],
            )

        else:
            # Normal rows: use your global formatter that maps ±18 -> ±50
            relabel_ticks_minus18_plus18_as_50(ax_rt)
            relabel_ticks_minus18_plus18_as_50(ax_mt)
            relabel_ticks_minus18_plus18_as_50(ax_psy)



    # legend = views
    handles, labels = [], []
    for v in views:
        h = plt.Line2D([], [], color=view_colors[v.name], marker="o", linestyle="None")
        handles.append(h)
        labels.append(v.name)

    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
