import matplotlib.pyplot as plt

from .style import style_axes, relabel_ticks_minus18_plus18_as_50
from .traces import plot_rt, plot_mt, plot_psy


def plot_multibundle_abls_4x3(
    runs,
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
):
    """
    runs = list of dicts with:
      {
        "label": "CNTNAP2 hom",
        "bundle": <bundle>,
        "view":  "<view_name inside bundle>",  # usually "hom"
        "color": "C0",
      }
    """

    # union of ABLs across all runs
    abl_rows = sorted(
        set().union(*[
            set(r["bundle"]["prepared"][r["view"]]["rt_group"]["ABL"].unique())
            for r in runs
        ])
    )

    # assume same cfg/error_mode across runs
    error_mode = runs[0]["bundle"]["cfg"].error_mode

    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.8 * len(abl_rows)), squeeze=False)

    for r_i, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r_i]

        # overlay each dataset/run as a trace color
        for run in runs:
            b = run["bundle"]
            view_name = run["view"]
            color = run["color"]

            tables = b["prepared"][view_name]

            # skip if this dataset doesn't have this ABL
            if abl not in set(tables["rt_group"]["ABL"].unique()):
                continue

            plot_rt(ax_rt,  abl, tables, color, error_mode)
            plot_mt(ax_mt,  abl, tables, color, error_mode)
            plot_psy(ax_psy, abl, tables, color, error_mode, skip_fit=SKIP_PSY_FITS)

        # style
        style_axes(ax_rt,  f"ABL {abl} — RT",          "ILD (dB)", "Mean RT (s)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)
        style_axes(ax_mt,  f"ABL {abl} — MT",          "ILD (dB)", "Mean MT (s)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)
        style_axes(ax_psy, f"ABL {abl} — Psychometric","ILD (dB)", "P(Right)",
                   TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_PAD)

        # optional old overlays (once is enough; do it on top axis)
        if show_old_overlays and makefig1_chrono is not None:
            import Helpers.DataHelpers as DataHelpers
            DataHelpers.overlay_makefig1_rt(ax_rt, abl, makefig1_chrono)

        if show_old_overlays and makefig1_data is not None and int(abl) != 50:
            import Helpers.DataHelpers as DataHelpers
            DataHelpers.overlay_makefig1_psychometrics(
                ax_psy, makefig1_data, abl, color="black",
                show_individuals=False, use_abl_colors=False
            )

        # ticks mapping
        relabel_ticks_minus18_plus18_as_50(ax_rt)
        relabel_ticks_minus18_plus18_as_50(ax_mt)
        relabel_ticks_minus18_plus18_as_50(ax_psy)

    # legend = datasets (runs)
    handles, labels = [], []
    for run in runs:
        handles.append(plt.Line2D([], [], color=run["color"], marker="o", linestyle="None"))
        labels.append(run["label"])
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), fontsize=LEGEND_FONTSIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
