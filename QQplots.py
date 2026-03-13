#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import Helpers.DataHelpers as DataHelpers

# ==============================================================
# CONFIG: choose which line you're analyzing
# ==============================================================
LINE = "CNTNAP2"   # or "SHANK3"
COHORT = "cohort2" # or "cohort1", etc

BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"

LINE_ROOTS = {
     ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
     ("SHANK3", "cohort1"): "SHANK3_cohort1",
 }

DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[LINE,COHORT])

os.chdir(DATA_DIR)

# ==============================================================
# === CONFIG ===================================================
# ==============================================================
cohort_file = "merged_all_subjects.csv"

error_mode = "individuals"  # "sem" or "individuals"
MASK_59_TO_60 = True
MASK_25_TO_50_WHEN_TL16 = True


TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
TITLE_PAD = 16

plt.rcParams["savefig.pad_inches"] = 0.6
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "sans-serif",
]


# ==============================================================
# === LOAD & FILTER DATA =======================================
# ==============================================================
df = pd.read_csv(cohort_file)
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"] == False].copy()
df = df[df["training_level"] == 16].copy()

sess = pd.to_numeric(df["session_type"], errors="coerce")
sd   = pd.to_numeric(df["stim_dur"], errors="coerce")
df = df[(sess == 1) | (sd == 6000)].copy()

df = df[df["training_level"]==16].copy()
df = df[df["session"]>13].copy()
df = df[df["success"]==1].copy()
# df = df[df["timed_rt"]>.05].copy()

views = [
    (
        "wt",
        lambda d: DataHelpers.restrict_subjects(
            d,
            "sex_gen.csv",
            genotypes="wt",
            subject_col="animal",
            genotype_col="genotype",
            attach_meta=True,
        ),
    ),
    (
        "het",
        lambda d: DataHelpers.restrict_subjects(
            d,
            "sex_gen.csv",
            genotypes="het",
            subject_col="animal",
            genotype_col="genotype",
            attach_meta=True,
        ),
    ),
    (
        "hom",
        lambda d: DataHelpers.restrict_subjects(
            d,
            "sex_gen.csv",
            genotypes="hom",
            subject_col="animal",
            genotype_col="genotype",
            attach_meta=True,
        ),
    ),
]


# ==============================================================
# === ILD / ABL SETTINGS =======================================
# ==============================================================
ILD_LISTS = [[1, 2], [4, 8]]   # like {[1,2],[4,8]} in MATLAB
ABL_VALUES = [20, 40, 60]

# For now no SD; keep structure ready (but it's optional)
SD_VALUES = [None]  # later you can use real SDs


def out_select(
    df_view,
    ild_list,
    abl_value,
    sd_value=None,
    *,
    rt_col="timed_rt",
    out_col="sucess",
    sd_col="SD",   # change when you know SD column name
):
    """
    Python equivalent of:
      outSelect = @(x,y,z,w) x(x.Out==1 & ismember(abs(x.ILD),y) & x.ABL==z & x.RTwrtStim>.001,:);

    Uses:
      - 'timed_rt' instead of 'RTwrtStim'
      - Out filter only if column exists
      - SD filter only if both sd_value and column exist
    """
    mask = pd.Series(True, index=df_view.index)

    mask &= df_view["ILD"].abs().isin(ild_list)
    mask &= (df_view["ABL"] == abl_value)
    mask &= (df_view[rt_col] > 0.001)

    if (sd_value is not None) and (sd_col in df_view.columns):
        mask &= (df_view[sd_col] == sd_value)

    return df_view.loc[mask]

def plot_rt_quantiles_and_hists_for_view(df_view, view_name):
    """
    Rough port of the MATLAB code for a single view (one genotype group).
    Makes a 2 x len(ILD_LISTS) figure:
      - top row: quantile vs quantile curves (20 vs 60, 40 vs 60)
      - bottom row: RT histograms (20, 40, 60)
    """
    # -----------------------------
    # Build outcell + quantiles
    # -----------------------------
    qx = np.arange(0.01, 1.00, 0.01)  # 0.01:0.01:0.99

    outcell = [
        [
            [None for _ in range(len(ILD_LISTS))]
            for _ in range(len(ABL_VALUES))
        ]
        for _ in range(len(SD_VALUES))
    ]

    q_1 = [
        [
            [None for _ in range(len(ILD_LISTS))]
            for _ in range(len(ABL_VALUES))
        ]
        for _ in range(len(SD_VALUES))
    ]

    for i_sd, sd_val in enumerate(SD_VALUES):
        for j_abl, abl_val in enumerate(ABL_VALUES):
            for k_ild, ild_vals in enumerate(ILD_LISTS):
                subset = out_select(df_view, ild_vals, abl_val, sd_val)
                outcell[i_sd][j_abl][k_ild] = subset
                if subset.empty:
                    q_1[i_sd][j_abl][k_ild] = np.full_like(qx, np.nan, dtype=float)
                else:
                    q_1[i_sd][j_abl][k_ild] = np.quantile(
                        subset["timed_rt"].values,
                        qx
                    )

    # -----------------------------
    # Figure + axes
    # -----------------------------
    cm_to_inch = 1 / 2.54
    fig_width = 17 * cm_to_inch
    fig_height = 17 * cm_to_inch

    fig, axes = plt.subplots(
        2,
        len(ILD_LISTS),
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    fig.suptitle(view_name, fontsize=TITLE_FONTSIZE, y=0.98)

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=TICK_FONTSIZE)

    bw = 0.01
    xx = np.arange(-1, 1.5 + bw, bw)

    # -----------------------------
    # Top row: quantile vs quantile
    # -----------------------------
    ymin, ymax = -0.2, 0.6
    for k1 in range(len(ILD_LISTS)):
        ax = axes[0, k1]

        # x = quantiles at ABL "end" (60)
        x_q = q_1[0][-1][k1]
        # y = quantiles at ABL 20 and 40
        y_q_20 = q_1[0][0][k1]
        y_q_40 = q_1[0][1][k1]

        ax.plot(x_q, y_q_20)
        ax.plot(x_q, y_q_40)

        # horizontal ref line: 0
        ax.plot(
            0.5 * np.array([-1, 1]),
            0 * np.array([-1, 1]),
            linestyle=":",
            color="k",
        )

        ax.set_xlim(-0.1, 0.5)
        ax.set_ylim(ymin, ymax)

        ild_label = ILD_LISTS[k1]
        ax.set_title(
            f"ILD = {ild_label}",
            fontsize=LABEL_FONTSIZE - 2,
            pad=TITLE_PAD / 2,
        )

        if k1 == 0:
            ax.set_ylabel("RT difference (s)", fontsize=LABEL_FONTSIZE)
        else:
            ax.set_yticklabels([])

    # -----------------------------
    # Bottom row: histograms
    # -----------------------------
    for k1 in range(len(ILD_LISTS)):
        ax = axes[1, k1]

        for j_abl, abl_val in enumerate(ABL_VALUES):
            subset = outcell[0][j_abl][k1]
            if subset is None or subset.empty:
                continue

            ax.hist(
                subset["timed_rt"],
                bins=xx,
                histtype="step",
                density=True,
                linewidth=0.85,
            )

        ax.set_xlim(-0.1, 0.5)
        ax.set_ylim(0, 8.5)

        if k1 > 0:
            ax.set_yticklabels([])

        ax.set_xlabel("RT (s)", fontsize=LABEL_FONTSIZE)

    # legend in last bottom panel
    axes[1, -1].legend(
        [str(a) for a in ABL_VALUES],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        title=None,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ==============================================================
# === RUN FOR ALL VIEWS ========================================
# ==============================================================
for view_name, view_func in views:
    df_view = view_func(df)
    print(
        f"View '{view_name}': "
        f"{df_view['animal'].nunique()} animals, "
        f"{len(df_view)} trials"
    )

    fig = plot_rt_quantiles_and_hists_for_view(df_view, view_name)
    # Optionally save:
    # fig.savefig(f"RT_quantiles_{view_name}.pdf")
    # plt.close(fig)  # if you don't want all to stay open

# %%
