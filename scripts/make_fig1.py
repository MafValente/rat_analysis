# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task")

# Add padding when saving figures to create whitespace around the entire figure
mpl.rcParams["savefig.pad_inches"] = 0.6

# Use Helvetica or its open-source equivalent font throughout the figure.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "TeX Gyre Heros", "Arial", "sans-serif"]

font_path = fm.findfont(mpl.font_manager.FontProperties(family=mpl.rcParams["font.sans-serif"]))
print(f"The font being used is: {font_path}")

# Helper to nudge a list of axes horizontally (dx in figure coordinates)
def shift_axes(ax_list, dx=0, dy=0):
    """Shift axes in ax_list by dx and dy (figure coordinate fractions)."""
    for ax in ax_list:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])

# --- Plotting Configuration ---
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
SUPTITLE_FONTSIZE = 24
plt.rcParams["axes.labelpad"] = 12

# --- Consistent ABL colors everywhere ---
ABL_COLOR = {20: "tab:blue", 40: "tab:orange", 60: "tab:red"}

# --- Sigmoid function (must match the one in the aggregation script) ---
def sigmoid(x, upper, lower, x0, k):
    """Sigmoid function with explicit upper and lower asymptotes."""
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))

# =========================
# LOAD PSYCHOMETRIC DATA
# =========================
with open("fig1_plot_data.pkl", "rb") as f:
    plot_data = pickle.load(f)

ABLS = sorted(plot_data["ABLS"])  # enforce stable order (20,40,60)
black_plot_as = plot_data["black_plot_as"]
ilds_dict = plot_data["ilds_dict"]
mean_params_dict = plot_data["mean_params_dict"]
mean_sigmoid_dict = plot_data["mean_sigmoid_dict"]
x_smooth_dict = plot_data["x_smooth_dict"]
unique_animal_identifiers = plot_data["unique_animal_identifiers"]
merged_valid = plot_data["merged_valid"]
all_sigmoid_curves_dict = plot_data["all_sigmoid_curves_dict"]

# =========================
# FIGURE + GRID
# =========================
fig = plt.figure(figsize=(25, 30))
fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06)

gs = GridSpec(
    5,
    6,
    figure=fig,
    hspace=0.3,
    wspace=0.0,
    width_ratios=[1, 1, 1, 1, 1, 1],
    # IMPORTANT: last row cannot be 0, otherwise it disappears
    height_ratios=[1, 0.5, 0.5, 0.5, 0.5],
)

# =========================
# PSYCHOMETRICS (ROW 1)
# =========================
gs_psych = gs[1, 0:4].subgridspec(1, 4, wspace=0.25)
ax_psych_1 = fig.add_subplot(gs_psych[0, 0])
ax_psych_2 = fig.add_subplot(gs_psych[0, 1], sharey=ax_psych_1)
ax_psych_3 = fig.add_subplot(gs_psych[0, 2], sharey=ax_psych_1)
ax_psych_4 = fig.add_subplot(gs_psych[0, 3], sharey=ax_psych_1)

axes = [ax_psych_1, ax_psych_2, ax_psych_3, ax_psych_4]
shift_axes([ax_psych_4], dx=0.04)

for ax in axes:
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    else:
        ax.set_aspect("equal", adjustable="box")

for ax in [ax_psych_2, ax_psych_3, ax_psych_4]:
    plt.setp(ax.get_yticklabels(), visible=False)

# --- first 3 ABL panels ---
for idx, abl in enumerate(ABLS[:3]):
    color = ABL_COLOR.get(int(abl), "k")
    ax = axes[idx]
    ilds = ilds_dict[abl]
    x_smooth = x_smooth_dict[abl]

    # individual animal fits
    if abl in all_sigmoid_curves_dict:
        for y_fit in all_sigmoid_curves_dict[abl]:
            ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)

    # black average sigmoid
    if black_plot_as == "mean_of_params" and abl in mean_params_dict:
        mean_params = mean_params_dict[abl]
        y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
        ax.plot(x_smooth, y_mean_sigmoid, color="black", linewidth=3, label="Avg sigmoid fit")
    elif black_plot_as == "mean_of_sigmoids" and abl in mean_sigmoid_dict:
        mean_sigmoid = mean_sigmoid_dict[abl]
        ax.plot(x_smooth, mean_sigmoid, color="black", linewidth=3, label="Avg sigmoid fit")

    # points: mean ± SEM across animals
    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[
            (merged_valid["batch_name"] == batch)
            & (merged_valid["animal"] == animal)
            & (merged_valid["ABL"] == abl)
        ]
        psycho_allowed = []
        for ild in ilds:
            sub = animal_df[animal_df["ILD"] == ild]
            psycho_allowed.append(np.mean(sub["choice"] == 1) if len(sub) > 0 else np.nan)
        all_psycho_points.append(np.array(psycho_allowed))

    all_psycho_points = np.array(all_psycho_points, dtype=float)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
    sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(n_points)

    ax.errorbar(
        ilds,
        mean_psycho,
        yerr=sem_psycho,
        fmt="o",
        color=color,
        capsize=0,
        markersize=8.5,
        label="Mean ± SEM",
    )

    ax.set_title(f"ABL = {abl}", fontsize=TITLE_FONTSIZE)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("ILD (dB)", fontsize=LABEL_FONTSIZE)

    if idx == 0:
        ax.set_ylabel("P(Right)", fontsize=LABEL_FONTSIZE)
        ax.spines["left"].set_color("black")
        ax.yaxis.label.set_color("black")
        ax.tick_params(axis="y", colors="black")
    else:
        ax.spines["left"].set_color("#bbbbbb")
        ax.yaxis.label.set_color("#bbbbbb")
        ax.tick_params(axis="y", colors="#bbbbbb")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- 4th psychometric plot: all ABLs together ---
ax4 = axes[3]
for abl in ABLS:
    color = ABL_COLOR.get(int(abl), "k")
    ilds = ilds_dict[abl]

    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[
            (merged_valid["batch_name"] == batch)
            & (merged_valid["animal"] == animal)
            & (merged_valid["ABL"] == abl)
        ]
        psycho = [
            np.mean(animal_df[animal_df["ILD"] == ild]["choice"] == 1)
            if len(animal_df[animal_df["ILD"] == ild]) > 0
            else np.nan
            for ild in ilds
        ]
        all_psycho_points.append(psycho)

    all_psycho_points = np.array(all_psycho_points, dtype=float)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
    sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(n_points)

    ax4.errorbar(
        ilds,
        mean_psycho,
        yerr=sem_psycho,
        fmt="o",
        color=color,
        capsize=0,
        markersize=8.5,
        label=f"ABL={abl} mean ± SEM",
    )

    if black_plot_as == "mean_of_params" and abl in mean_params_dict:
        mean_params = mean_params_dict[abl]
        x_smooth = x_smooth_dict[abl]
        y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
        ax4.plot(x_smooth, y_mean_sigmoid, color=color, linewidth=2, label=f"ABL={abl} curve")
    elif black_plot_as == "mean_of_sigmoids" and abl in mean_sigmoid_dict:
        mean_sigmoid = mean_sigmoid_dict[abl]
        x_smooth = x_smooth_dict[abl]
        if mean_sigmoid is not None and x_smooth is not None:
            ax4.plot(x_smooth, mean_sigmoid, color=color, linewidth=2, label=f"ABL={abl} curve")

ax4.set_title("All ABLs", fontsize=TITLE_FONTSIZE)
ax4.axvline(0, color="gray", linestyle="--", alpha=0.7)
ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
ax4.set_ylim(0, 1)
ax4.set_xticks([-15, -5, 5, 15])
ax4.set_yticks([0, 0.5, 1])
ax4.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
ax4.set_xlabel("ILD (dB)", fontsize=LABEL_FONTSIZE)
ax4.spines["left"].set_color("#bbbbbb")
ax4.yaxis.label.set_color("#bbbbbb")
ax4.tick_params(axis="y", colors="#bbbbbb")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(LEGEND_FONTSIZE)

# =========================
# CHRONOMETRICS (ROW 2)
# =========================
try:
    with open("fig1_chrono_plot_data.pkl", "rb") as f:
        chrono_data = pickle.load(f)

    plot_abls = chrono_data["plot_abls"]
    all_chrono_data_df = chrono_data["all_chrono_data_df"]
    grand_means_data = chrono_data["grand_means_data"]
    abs_ild_ticks = chrono_data["abs_ild_ticks"]

    gs_chrono_main = gs[2, 0:4].subgridspec(1, 4, wspace=0.25)
    ax_chrono_1 = fig.add_subplot(gs_chrono_main[0, 0])
    ax_chrono_2 = fig.add_subplot(gs_chrono_main[0, 1], sharey=ax_chrono_1)
    ax_chrono_3 = fig.add_subplot(gs_chrono_main[0, 2], sharey=ax_chrono_1)
    ax_chrono_4 = fig.add_subplot(gs_chrono_main[0, 3], sharey=ax_chrono_1)
    chrono_axes = [ax_chrono_1, ax_chrono_2, ax_chrono_3, ax_chrono_4]
    shift_axes([ax_chrono_4], dx=0.04)

    for ax in chrono_axes:
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal", adjustable="box")

    for ax in [ax_chrono_2, ax_chrono_3, ax_chrono_4]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # first 3 chronometric panels
    for i, abl in enumerate(list(plot_abls)[:3]):
        ax = chrono_axes[i]
        color = ABL_COLOR.get(int(abl), "k")

        abl_df = all_chrono_data_df[all_chrono_data_df["ABL"] == abl]

        for (batch_name, animal_id), animal_df in abl_df.groupby(["batch_name", "animal_id"]):
            animal_df = animal_df.sort_values("abs_ILD")
            ax.plot(animal_df["abs_ILD"], animal_df["mean"], color=color, alpha=0.4, linewidth=1.5)

        grand_mean_stats = grand_means_data[abl]
        ax.errorbar(
            x=grand_mean_stats["abs_ILD"],
            y=grand_mean_stats["mean"],
            yerr=grand_mean_stats["sem"],
            fmt="o",
            color=color,
            markersize=8.5,
            capsize=0,
            linewidth=0,
            zorder=3,
        )
        ax.plot(grand_mean_stats["abs_ILD"], grand_mean_stats["mean"], color="black", linewidth=2.5, zorder=2)

        ax.set_xlabel("|ILD| (dB)", fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel("Mean RT (s)", fontsize=LABEL_FONTSIZE)
            ax.spines["left"].set_color("black")
            ax.tick_params(axis="y", colors="black")
        else:
            ax.spines["left"].set_color("#bbbbbb")
            ax.tick_params(axis="y", colors="#bbbbbb")

        ax.set_xscale("log")
        ax.set_xticks(abs_ild_ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.set_ylim(0.1, 0.45)
        ax.set_yticks([0.1, 0.2, 0.3, 0.4])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 4th chronometric: all ABLs overlay
    ax4_chrono = chrono_axes[3]
    for abl, stats_data in grand_means_data.items():
        color = ABL_COLOR.get(int(abl), "k")
        ax4_chrono.errorbar(
            x=stats_data["abs_ILD"],
            y=stats_data["mean"],
            yerr=stats_data["sem"],
            fmt="o",
            color=color,
            markersize=8.5,
            capsize=0,
            linewidth=0,
            zorder=3,
        )
        ax4_chrono.plot(stats_data["abs_ILD"], stats_data["mean"], color=color, linewidth=2.5, zorder=2)

    ax4_chrono.set_xlabel("|ILD| (dB)", fontsize=LABEL_FONTSIZE)
    ax4_chrono.set_xscale("log")
    ax4_chrono.set_xticks(abs_ild_ticks)
    ax4_chrono.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax4_chrono.xaxis.set_minor_locator(plt.NullLocator())
    ax4_chrono.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax4_chrono.spines["top"].set_visible(False)
    ax4_chrono.spines["right"].set_visible(False)
    ax4_chrono.spines["left"].set_color("#bbbbbb")
    ax4_chrono.tick_params(axis="y", colors="#bbbbbb")

    # summary plots (last column)
    rt_vs_ild = chrono_data["rt_vs_ild"]
    rt_vs_abl = chrono_data["rt_vs_abl"]

    gs_summary = gs[2, 5].subgridspec(2, 1, hspace=0.05)
    ax_ild = fig.add_subplot(gs_summary[0, 0])
    ax_abl = fig.add_subplot(gs_summary[1, 0])

    ax_ild.errorbar(
        x=rt_vs_ild["abs_ILD"],
        y=rt_vs_ild["mean"],
        yerr=rt_vs_ild["sem"],
        fmt="o",
        color="k",
        capsize=0,
        markersize=6,
        linewidth=2,
    )
    ax_ild.set_xlabel("|ILD|", fontsize=LABEL_FONTSIZE, ha="right", x=1.4)
    ax_ild.xaxis.set_label_coords(1.4, 0.1)
    ax_ild.set_xscale("log")
    ax_ild.set_xticks(abs_ild_ticks)
    ax_ild.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_ild.xaxis.set_minor_locator(plt.NullLocator())
    ax_ild.spines["top"].set_visible(False)
    ax_ild.spines["right"].set_visible(False)
    ax_ild.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    # FIXED LOOP: RT vs ABL (NO SET COLORS)
    for i, row in rt_vs_abl.iterrows():
        abl = int(row["ABL"])
        color = ABL_COLOR.get(abl, "k")
        ax_abl.errorbar(
            x=i,
            y=row["mean"],
            yerr=row["sem"],
            fmt="o",
            linestyle="None",
            color=color,
            capsize=0,
            markersize=8.5,
        )

    ax_abl.set_xticks(range(len(rt_vs_abl)))
    ax_abl.set_xticklabels(rt_vs_abl["ABL"].astype(int))
    ax_abl.set_xlabel("ABL", fontsize=LABEL_FONTSIZE, ha="right", x=1.4)
    ax_abl.xaxis.set_label_coords(1.4, 0.1)
    plt.setp(ax_abl.get_yticklabels(), visible=False)
    ax_abl.spines["top"].set_visible(False)
    ax_abl.spines["right"].set_visible(False)
    ax_abl.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE)

    ax_ild.set_ylim(0.15, 0.26)
    ax_ild.set_yticks([0.15, 0.25])
    ax_ild.set_yticklabels(["0.15", "0.25"])
    ax_ild.tick_params(axis="y", labelleft=True, length=0)

    ax_abl.set_ylim(0.15, 0.30)
    ax_abl.set_yticks([0.15, 0.3])
    ax_abl.set_yticklabels(["0.15", "0.3"])
    ax_abl.tick_params(axis="y", labelleft=True)

    ax_abl.spines["top"].set_visible(False)
    ax_abl.spines["right"].set_visible(False)
    ax_abl.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, length=0)

    # Align summary axes with main chronometric row
    fig.canvas.draw()
    chrono_baseline = ax_chrono_1.get_position().y0
    summary_bottom = ax_abl.get_position().y0
    dy_summary = chrono_baseline - summary_bottom
    shift_axes([ax_ild, ax_abl], dy=dy_summary)

    fig.canvas.draw()
    chrono_top = ax_chrono_1.get_position().y1
    summary_top = ax_ild.get_position().y1
    dh_summary = summary_top - chrono_top
    if dh_summary > 0:
        for ax in (ax_ild, ax_abl):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height - dh_summary])

    fig.canvas.draw()
    gap_now_summ = ax_ild.get_position().y0 - ax_abl.get_position().y1
    gap_desired_summ = 0.02
    delta_summ = gap_now_summ - gap_desired_summ
    if delta_summ > 0:
        shift_axes([ax_ild], dy=-(delta_summ / 2))
        shift_axes([ax_abl], dy=(delta_summ / 2))

    shift_axes([ax_ild, ax_abl], dx=0.05, dy=-0.01)

    width_factor = 0.65
    for ax in (ax_ild, ax_abl):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * width_factor, pos.height])

    fig.canvas.draw()
    left_edge = ax_ild.get_position().x0 - 0.02
    center_y = 0.5 * (ax_ild.get_position().y1 + ax_abl.get_position().y0)
    fig.text(
        left_edge - 0.03,
        center_y,
        "Mean RT (s)",
        rotation="vertical",
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE,
    )

except FileNotFoundError:
    print("\nChronometric data file not found. Skipping chronometric plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting chronometric data: {e}")

# =========================
# JND PLOTS (ROW 1, COL 5)
# =========================
try:
    with open("jnd_analysis_data.pkl", "rb") as f:
        jnd_data = pickle.load(f)

    jnds = jnd_data["jnds"]
    mean_jnd = jnd_data["mean_jnd"]
    grand_mean_jnd = jnd_data["grand_mean_jnd"]
    ABLS_jnd = sorted(jnd_data["ABLS"])
    animals_with_mean = jnd_data["animals_with_mean"]
    mean_jnds = jnd_data["mean_jnds"]
    diff_within = jnd_data["diff_within"]

    gs_nested = gs[1, 5].subgridspec(2, 1, hspace=-0.2)

    gs_jnd_plot = gs_nested[0, 0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax1_main = fig.add_subplot(gs_jnd_plot[0, 0])
    ax1_hist = fig.add_subplot(gs_jnd_plot[0, 1], sharey=ax1_main)

    sorted_animal_indices = np.argsort(mean_jnds)
    sorted_animals = [animals_with_mean[i] for i in sorted_animal_indices]

    for i, animal_id in enumerate(sorted_animals):
        ax1_main.plot(i, mean_jnd[animal_id], "k_", markersize=6, mew=1.5)
        for abl in ABLS_jnd:
            if animal_id in jnds[abl]:
                ax1_main.plot(
                    i,
                    jnds[abl][animal_id],
                    "o",
                    color=ABL_COLOR.get(int(abl), "k"),
                    markersize=4,
                    alpha=0.5,
                    linewidth=2,
                )

    ax1_main.axhline(grand_mean_jnd, color="k", linestyle=":", linewidth=1)
    print(f"grand_mean_jnd: {grand_mean_jnd}")
    ax1_main.set_xticks([])
    ax1_main.set_ylabel("JND", fontsize=LABEL_FONTSIZE)
    ax1_main.spines["top"].set_visible(False)
    ax1_main.spines["right"].set_visible(False)
    ax1_main.spines["bottom"].set_visible(False)
    ax1_main.tick_params(axis="y", labelsize=TICK_FONTSIZE, length=0)
    ax1_main.set_ylim(1, 5)
    ax1_main.set_yticks([1, 5])

    mu_mean = np.mean(mean_jnds)
    sd_mean = np.std(mean_jnds)
    x_bar = 0.05
    ax1_hist.plot([x_bar, x_bar], [mu_mean - sd_mean, mu_mean + sd_mean], color="grey", linewidth=3)
    ax1_hist.set_xlim(0, 1)
    ax1_hist.axis("off")

    gs_var_plot = gs_nested[1, 0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax2_main = fig.add_subplot(gs_var_plot[0, 0])
    ax2_hist = fig.add_subplot(gs_var_plot[0, 1], sharey=ax2_main)

    for i, animal_id in enumerate(sorted_animals):
        jnd0 = mean_jnd[animal_id]
        for abl in ABLS_jnd:
            if animal_id in jnds[abl]:
                diff = jnds[abl][animal_id] - jnd0
                ax2_main.plot(
                    i,
                    diff,
                    "o",
                    color=ABL_COLOR.get(int(abl), "k"),
                    markersize=5,
                    alpha=0.5,
                )

    ax2_main.axhline(0, color="k", linestyle="-", linewidth=1)
    ax2_main.set_xticks([])
    ax2_main.set_ylabel(r"J$_{\text{ABL}}$ - J$_{\mu}$", fontsize=LABEL_FONTSIZE)
    ax2_main.spines["top"].set_visible(False)
    ax2_main.spines["right"].set_visible(False)
    ax2_main.spines["bottom"].set_visible(False)
    ax2_main.tick_params(axis="y", labelsize=TICK_FONTSIZE, length=0)
    ax2_main.set_ylim(-2, 2)
    ax2_main.set_yticks([-2, 0, 2])
    ax2_main.set_yticklabels(["-2", "0", "2"])

    mu_diff = np.mean(diff_within)
    sd_diff = np.std(diff_within)
    ax2_hist.plot([x_bar, x_bar], [mu_diff - sd_diff, mu_diff + sd_diff], color="grey", linewidth=3)
    ax2_hist.set_xlim(0, 1)
    ax2_hist.axis("off")

    # Align bottom JND baseline with psychometric baseline
    fig.canvas.draw()
    psycho_baseline = ax_psych_1.get_position().y0
    jnd_baseline = ax2_main.get_position().y0
    dy_align = psycho_baseline - jnd_baseline
    shift_axes([ax1_main, ax1_hist, ax2_main, ax2_hist], dx=0.05, dy=dy_align)

    # Top-edge alignment
    fig.canvas.draw()
    psycho_top = ax_psych_1.get_position().y1
    jnd_top = ax1_main.get_position().y1
    dh = jnd_top - psycho_top
    if dh > 0:
        for ax in (ax1_main, ax1_hist, ax2_main, ax2_hist):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height - dh])

    # Reduce inter-plot gap
    fig.canvas.draw()
    gap_now = ax1_main.get_position().y0 - ax2_main.get_position().y1
    gap_desired = 0.05 / 2
    delta_gap = gap_now - gap_desired
    if delta_gap > 0:
        shift_axes([ax1_main, ax1_hist], dy=-(delta_gap / 2))
        shift_axes([ax2_main, ax2_hist], dy=(delta_gap / 2))

except FileNotFoundError:
    print("\nJND data file ('jnd_analysis_data.pkl') not found. Skipping these plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting JNDs/histograms: {e}")

# =========================
# QUANTILE PLOTS (ROW 3)
# =========================
try:
    with open("fig1_quantiles_plot_data.pkl", "rb") as f:
        quantile_data = pickle.load(f)

    ABL_arr = list(quantile_data["ABL_arr"])
    abs_ILD_arr = quantile_data["abs_ILD_arr"]
    plotting_quantiles = quantile_data["plotting_quantiles"]
    mean_unscaled = quantile_data["mean_unscaled"]
    sem_unscaled = quantile_data["sem_unscaled"]

    mean_scaled = quantile_data["mean_scaled"]
    sem_scaled = quantile_data["sem_scaled"]

    gs_quant = gs[3, 0:4].subgridspec(1, 4, wspace=0.25)
    ax_quant_1 = fig.add_subplot(gs_quant[0, 0])
    ax_quant_2 = fig.add_subplot(gs_quant[0, 1], sharey=ax_quant_1)
    ax_quant_3 = fig.add_subplot(gs_quant[0, 2], sharey=ax_quant_1)
    quantile_axes = [ax_quant_1, ax_quant_2, ax_quant_3]

    for ax in quantile_axes:
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal", adjustable="box")

    for ax in [ax_quant_2, ax_quant_3]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # unscaled (first 3)
    for col, abl in enumerate(ABL_arr[:3]):
        ax = quantile_axes[col]
        color = ABL_COLOR.get(int(abl), "k")
        q_mat = mean_unscaled[abl]
        sem_mat = sem_unscaled[abl]

        for q_idx, _q_level in enumerate(plotting_quantiles):
            ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker="o", linestyle="-", color=color)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xscale("log")
        ax.set_xticks(abs_ILD_arr)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_xlabel("|ILD| (dB)", fontsize=LABEL_FONTSIZE)
        if col == 0:
            ax.set_ylabel("RT(s)", fontsize=LABEL_FONTSIZE)

    # scaled overlay (4th)
    ax_overlay = fig.add_subplot(gs_quant[0, 3])
    shift_axes([ax_overlay], dx=0.04)

    if hasattr(ax_overlay, "set_box_aspect"):
        ax_overlay.set_box_aspect(1)
    else:
        ax_overlay.set_aspect("equal", adjustable="box")

    for abl in ABL_arr:
        color = ABL_COLOR.get(int(abl), "k")
        q_mat = mean_scaled[abl]
        sem_mat = sem_scaled[abl]
        for q_idx, _q_level in enumerate(plotting_quantiles):
            ax_overlay.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker="o", linestyle="-", color=color)

    ax_overlay.spines["right"].set_visible(False)
    ax_overlay.spines["top"].set_visible(False)
    ax_overlay.set_xscale("log")
    ax_overlay.set_xticks(abs_ILD_arr)
    ax_overlay.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_overlay.xaxis.set_minor_locator(plt.NullLocator())
    ax_overlay.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax_overlay.set_ylim(0, 0.4)
    ax_overlay.set_yticks([0, 0.2, 0.4])
    ax_overlay.set_xlabel("|ILD| (dB)", fontsize=LABEL_FONTSIZE)
    ax_overlay.set_ylabel("Scaled RT (s)", fontsize=LABEL_FONTSIZE)

except FileNotFoundError:
    print("\nQuantile data file not found. Skipping quantile plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting quantile data: {e}")

"""
# =========================
# Q-Q PLOTS (ROW 4)
# =========================
try:
    with open("fig1_qq_plot_data.pkl", "rb") as f:
        qq_data = pickle.load(f)

    abs_ILD_arr_qq = qq_data["abs_ILD_arr"]
    avg_quantiles = qq_data["avg_quantiles"]
    sem_quantiles = qq_data["sem_quantiles"]
    min_RT_cut_by_ILD = qq_data["min_RT_cut_by_ILD"]
    global_min_val = qq_data["global_min_val"]
    global_max_val = qq_data["global_max_val"]

    gs_qq = gs[4, 0:5].subgridspec(1, 5, wspace=0.3)
    qq_axes = [fig.add_subplot(gs_qq[0, i]) for i in range(5)]

    for i, abs_ild in enumerate(abs_ILD_arr_qq):
        ax = qq_axes[i]
        q_20_avg = avg_quantiles[20][:, i]
        q_40_avg = avg_quantiles[40][:, i]
        q_60_avg = avg_quantiles[60][:, i]
        lower_lim = min_RT_cut_by_ILD[abs_ild]

        # ABL 20 vs 40 (blue)
        valid_40_20 = ~np.isnan(q_40_avg) & ~np.isnan(q_20_avg)
        if np.any(valid_40_20):
            x_data, y_data = q_40_avg[valid_40_20], q_20_avg[valid_40_20]
            x_sem = sem_quantiles[40][:, i][valid_40_20]
            y_sem = sem_quantiles[20][:, i][valid_40_20]
            mask = (x_data >= lower_lim) & (y_data >= lower_lim)
            ax.errorbar(
                x_data[mask], y_data[mask],
                xerr=x_sem[mask], yerr=y_sem[mask],
                marker="o", linestyle="none", color=ABL_COLOR[20], capsize=2
            )
            if np.sum(mask) > 1:
                m, c = np.polyfit(x_data[mask], y_data[mask], 1)
                fit_x = np.array([lower_lim, 0.5])
                ax.plot(fit_x, m * fit_x + c, color=ABL_COLOR[20])

        # ABL 60 vs 40 (red)
        valid_40_60 = ~np.isnan(q_40_avg) & ~np.isnan(q_60_avg)
        if np.any(valid_40_60):
            x_data, y_data = q_40_avg[valid_40_60], q_60_avg[valid_40_60]
            x_sem = sem_quantiles[40][:, i][valid_40_60]
            y_sem = sem_quantiles[60][:, i][valid_40_60]
            mask = (x_data >= lower_lim) & (y_data >= lower_lim)
            ax.errorbar(
                x_data[mask], y_data[mask],
                xerr=x_sem[mask], yerr=y_sem[mask],
                marker="o", linestyle="none", color=ABL_COLOR[60], capsize=2
            )
            if np.sum(mask) > 1:
                m, c = np.polyfit(x_data[mask], y_data[mask], 1)
                fit_x = np.array([lower_lim, 0.5])
                ax.plot(fit_x, m * fit_x + c, color=ABL_COLOR[60])

        ax.set_aspect("equal", adjustable="box")
        ax.plot([global_min_val, global_max_val], [global_min_val, global_max_val], "k--", alpha=0.7, zorder=0)
        ax.set_xlim(global_min_val, global_max_val)
        ax.set_ylim(global_min_val, global_max_val)
        ax.set_title(f"|ILD| = {abs_ild}", fontsize=LABEL_FONTSIZE)
        ax.set_xlabel("RT Quantiles (ABL 40)", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([0.08, 0.5])
        ax.set_yticks([0.06, 0.5])

    qq_axes[0].set_ylabel("RT Quantiles (ABL 20/60)", fontsize=LABEL_FONTSIZE)
    for ax in qq_axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

except FileNotFoundError:
    print("\nQ-Q plot data file not found. Skipping Q-Q plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting Q-Q data: {e}")

# =========================
# FINAL SAVE/SHOW
# =========================

# IMPORTANT: tight_layout fights your manual set_position/shift_axes and can hide axes.
# So we DO NOT call plt.tight_layout() here.

fig.suptitle("Figure 1", fontsize=SUPTITLE_FONTSIZE)

plt.savefig("fig1_from_pickle.png", dpi=300, bbox_inches="tight")
plt.savefig("fig1_from_pickle.pdf", bbox_inches="tight", format="pdf")
plt.show()

"""
print("\nFigure saved as fig1_from_pickle.png and fig1_from_pickle.pdf")
# %%
