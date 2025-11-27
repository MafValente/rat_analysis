
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import DataHelpers

# --- CONFIGURATION ---
SUBJECT_COL = 'animal'
TRIAL_COL = 'trial'
SUCCESS_COL = 'success'
SESSION_COL = 'session'
SPAN = 25
META_CSV = "sex_gen.csv"
NORMALIZED_POINTS = 100   # resolution for each session
# ---------------------

# ===============================================================
#   LOAD DATA
# ===============================================================

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file, low_memory=False)

df = df[df["session"] <= 3]

print(f"Loaded file: {cohort_file} ({len(df)} rows)")


# ===============================================================
#   NORMALIZED SESSION-WISE CURVE COMPUTATION
# ===============================================================

def compute_session_curves(name, filter_fn, df):
    """
    Computes *normalized* learning curves per session.
    Each subject's curve runs from 0 → 1 in session time (not trials).
    Per session, curves are resampled to the same number of points.
    """

    df_view = filter_fn(df)
    if len(df_view) == 0:
        print(f"[WARN] View {name} has no subjects.")
        return None

    df_view = df_view.copy()

    # Learning metrics
    df_view["is_correct"] = (df_view[SUCCESS_COL] == 1).astype(int)

    subjects = df_view[SUBJECT_COL].unique()
    sessions = sorted(df_view[SESSION_COL].unique())

    def ewm_smooth(x):
        return x.ewm(span=SPAN, adjust=False).mean()

    output = {"name": name, "sessions": {}}

    # ============================================================
    #   For each session: normalize → resample → aggregate
    # ============================================================
    for sess in sessions:
        df_sess = df_view[df_view[SESSION_COL] == sess]

        sess_curves = []
        sess_subjects = df_sess[SUBJECT_COL].unique()

        for s in sess_subjects:
            sub = df_sess[df_sess[SUBJECT_COL] == s].sort_values(TRIAL_COL)
            smooth = ewm_smooth(sub["is_correct"]).values

            if len(smooth) < 2:
                continue

            # Normalize x to 0–1
            x_raw = np.linspace(0, 1, len(smooth))

            # Resample to uniform length (NORMALIZED_POINTS)
            x_target = np.linspace(0, 1, NORMALIZED_POINTS)
            smooth_resampled = np.interp(x_target, x_raw, smooth)

            sess_curves.append(smooth_resampled)

        if len(sess_curves) == 0:
            continue

        mat = np.vstack(sess_curves)

        mean_curve = mat.mean(axis=0)
        sem_curve = mat.std(axis=0) / np.sqrt(mat.shape[0])

        output["sessions"][sess] = {
            "subjects": sess_subjects,
            "n": len(sess_subjects),
            "length": NORMALIZED_POINTS,
            "mean": mean_curve,
            "sem": sem_curve,
            "curves": sess_curves,
        }

    return output


# ===============================================================
#   DEFINE GENOTYPE VIEWS
# ===============================================================

views = [
    ("females",  lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, sex="female",  subject_col="animal",
        genotype_col="genotype", attach_meta=True)),

    ("males", lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, sex="male", subject_col="animal",
        genotype_col="genotype", attach_meta=True)),

    # ("hom", lambda d: DataHelpers.restrict_subjects(
    #     d, META_CSV, genotypes="hom", subject_col="animal",
    #     genotype_col="genotype", attach_meta=True)),
]


# ===============================================================
#   COMPUTE NORMALIZED SESSION CURVES
# ===============================================================

session_results = {}
for name, fn in views:
    session_results[name] = compute_session_curves(name, fn, df)


#%%
"""
.########.##......##..#######......######...#######..##........#######..########..########.....###....########...######.
....##....##..##..##.##.....##....##....##.##.....##.##.......##.....##.##.....##.##.....##...##.##...##.....##.##....##
....##....##..##..##.##.....##....##.......##.....##.##.......##.....##.##.....##.##.....##..##...##..##.....##.##......
....##....##..##..##.##.....##....##.......##.....##.##.......##.....##.########..########..##.....##.########...######.
....##....##..##..##.##.....##....##.......##.....##.##.......##.....##.##...##...##.....##.#########.##...##.........##
....##....##..##..##.##.....##....##....##.##.....##.##.......##.....##.##....##..##.....##.##.....##.##....##..##....##
....##.....###..###...#######......######...#######..########..#######..##.....##.########..##.....##.##.....##..######.
"""
# ===============================================================
#   FINAL PLOT — Mean Curve Colored by Average Training Level
# ===============================================================

def plot_mean_curve_with_two_colorbars(df, session_results):

    # Automatically extract list of group names
    groups = [name for name, _ in views]    

    # Colormaps
    mean_cmap = plt.cm.get_cmap("Set3")     # continuous for training level
    sem_cmap  = plt.cm.get_cmap("coolwarm")  # diverging for variance

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)

    # Track GLOBAL ranges for colorbars
    global_TL_min = np.inf
    global_TL_max = -np.inf
    global_VAR_min = np.inf
    global_VAR_max = -np.inf

    # --------------------------------------------------------------
    #   First pass — compute global min/max for TL and VAR
    # --------------------------------------------------------------
    precomputed = {}

    for group in groups:

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            continue

        sessions = sorted(res["sessions"].keys())

        X_all, MEAN_all, SEM_all = [], [], []
        TL_mean_segments = []
        TL_var_segments  = []

        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            # training-level data
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            # resample training level per animal
            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) < 2:
                    continue
                x_raw = np.linspace(0, 1, len(levels))
                x_tgt = np.linspace(0, 1, n)
                tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)

            TL_mean_segments.append(tlev_mat.mean(axis=0))
            TL_var_segments.append(tlev_mat.var(axis=0))

        TL_mean = np.concatenate(TL_mean_segments)
        TL_var  = np.concatenate(TL_var_segments)

        precomputed[group] = (TL_mean, TL_var)

        # update global ranges
        global_TL_min = min(global_TL_min, TL_mean.min())
        global_TL_max = max(global_TL_max, TL_mean.max())
        global_VAR_min = min(global_VAR_min, TL_var.min())
        global_VAR_max = max(global_VAR_max, TL_var.max())

    # --------------------------------------------------------------
    #   Second pass — plot everything
    # --------------------------------------------------------------
    for ax, group in zip(axes, groups):

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.text(0.5, 0.5, f"{group.upper()} — No data",
                    ha="center", va="center")
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())

        X_all, MEAN_all, SEM_all = [], [], []
        offset = 0

        TL_mean_segments = []
        TL_var_segments  = []

        # ------------- Build arrays again for plotting -------------
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            # x coord for this session
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

            # training-level data
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            # resample training level
            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) < 2:
                    continue
                x_raw = np.linspace(0, 1, len(levels))
                x_tgt = np.linspace(0, 1, n)
                tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)

            TL_mean_segments.append(tlev_mat.mean(axis=0))
            TL_var_segments.append(tlev_mat.var(axis=0))

        # Concatenate full-trial arrays
        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)
        TL_mean = np.concatenate(TL_mean_segments)
        TL_var  = np.concatenate(TL_var_segments)

        # Normalize using global ranges (important!)
        TL_norm = (TL_mean - global_TL_min) / (global_TL_max - global_TL_min + 1e-6)
        VAR_norm = (TL_var - global_VAR_min) / (global_VAR_max - global_VAR_min + 1e-6)

        # ----------------------------------------------------------
        # 1. Plot SEM shading colored by variance
        # ----------------------------------------------------------
        for i in range(len(X) - 1):
            ax.fill_between(
                X[i:i+2],
                MEAN[i:i+2] - SEM[i:i+2],
                MEAN[i:i+2] + SEM[i:i+2],
                color=sem_cmap(VAR_norm[i]),
                alpha=0.25
            )

        # ----------------------------------------------------------
        # 2. Plot mean curve colored by training level
        # ----------------------------------------------------------
        for i in range(len(X) - 1):
            ax.plot(
                X[i:i+2],
                MEAN[i:i+2],
                color=mean_cmap(TL_norm[i]),
                linewidth=3
            )

        # ----------------------------------------------------------
        # 3. Grey individual traces
        # ----------------------------------------------------------
        per_subject = {}
        offset = 0
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1
            for subj, curve in zip(info["subjects"], info["curves"]):
                per_subject.setdefault(subj, {"x": [], "y": []})
                per_subject[subj]["x"].append(x_seg)
                per_subject[subj]["y"].append(curve)

        for subj, data in per_subject.items():
            ax.plot(
                np.concatenate(data["x"]),
                np.concatenate(data["y"]),
                color="lightgray",
                alpha=0.3
            )

        # Session boundaries
        offset = 0
        for sess in sessions:
            ax.axvline(offset, linestyle="--", color="gray", alpha=0.3)
            offset += 1

        ax.set_title(f"{group.upper()} — Mean Learning Curve")
        ax.set_xlabel("Normalized Session Progress (0–1 per session)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA)")

    # ==============================================================
    # Colorbar 1 — Mean Training Level
    # ==============================================================
    sm_TL = plt.cm.ScalarMappable(
        cmap=mean_cmap,
        norm=plt.Normalize(vmin=global_TL_min, vmax=global_TL_max)
    )
    sm_TL.set_array([])

    cax1 = fig.add_axes([1.05, 0.15, 0.03, 0.7])
    cbar1 = fig.colorbar(sm_TL, cax=cax1)
    cbar1.set_label("Average Training Level", fontsize=12)

    # ==============================================================
    # Colorbar 2 — Training-Level Variance
    # ==============================================================
    sm_VAR = plt.cm.ScalarMappable(
        cmap=sem_cmap,
        norm=plt.Normalize(vmin=global_VAR_min, vmax=global_VAR_max)
    )
    sm_VAR.set_array([])

    cax2 = fig.add_axes([1.12, 0.15, 0.03, 0.7])
    cbar2 = fig.colorbar(sm_VAR, cax=cax2)
    cbar2.set_label("Training-Level Variance", fontsize=12)

    # --------------------------------------------------------------
    plt.suptitle("Mean Curves Colored by Training Level (SEM colored by Variance)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    plt.show()



plot_mean_curve_with_two_colorbars(df, session_results)



#%%

"""
..#######..##....##.########.....######...#######..##........#######..########..########.....###....########.
.##.....##.###...##.##..........##....##.##.....##.##.......##.....##.##.....##.##.....##...##.##...##.....##
.##.....##.####..##.##..........##.......##.....##.##.......##.....##.##.....##.##.....##..##...##..##.....##
.##.....##.##.##.##.######......##.......##.....##.##.......##.....##.########..########..##.....##.########.
.##.....##.##..####.##..........##.......##.....##.##.......##.....##.##...##...##.....##.#########.##...##..
.##.....##.##...###.##..........##....##.##.....##.##.......##.....##.##....##..##.....##.##.....##.##....##.
..#######..##....##.########.....######...#######..########..#######..##.....##.########..##.....##.##.....##
"""

def plot_mean_and_individual_training_colored(df, session_results,
                                              color_sem_by_variance=False):
    """
    Plot:
    - Individual subjects colored by their OWN instantaneous training level
    - Group mean colored by AVERAGE training level
    - Optional SEM colored by training-level variance
    """

    # Automatically extract list of group names
    groups = [name for name, _ in views]

    indiv_cmap = plt.cm.get_cmap("Set3")    # individual traces
    mean_cmap  = plt.cm.get_cmap("Set3")    # mean curve
    sem_cmap   = plt.cm.get_cmap("coolwarm") # SEM (if option enabled)

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 10), sharey=True)

    # global ranges for color normalization
    global_TL_min = np.inf
    global_TL_max = -np.inf
    global_VAR_min = np.inf
    global_VAR_max = -np.inf

    # -----------------------------------------
    # PASS 1 — compute global ranges
    # -----------------------------------------
    for group in groups:
        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            continue

        TL_all = []
        VAR_all = []

        for sess, info in res["sessions"].items():
            n = info["length"]

            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            # training-level per animal
            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) >= 2:
                    x_raw = np.linspace(0, 1, len(levels))
                    x_tgt = np.linspace(0, 1, n)
                    tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)
            if len(tlev_mat) == 0:
                continue

            TL_all.append(tlev_mat.mean(axis=0))
            VAR_all.append(tlev_mat.var(axis=0))

        if len(TL_all) == 0:
            continue

        TL = np.concatenate(TL_all)
        VAR = np.concatenate(VAR_all)

        global_TL_min = min(global_TL_min, TL.min())
        global_TL_max = max(global_TL_max, TL.max())
        global_VAR_min = min(global_VAR_min, VAR.min())
        global_VAR_max = max(global_VAR_max, VAR.max())

    # -----------------------------------------
    # PASS 2 — plotting
    # -----------------------------------------
    for ax, group in zip(axes, groups):

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())

        X_all, MEAN_all, SEM_all = [], [], []
        TL_mean_all, TL_var_all = [], []
        offset = 0

        # ------------------------------
        # Build concatenated structures
        # ------------------------------
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            # session x-range
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

            # training level resampling (group mean + variance)
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) >= 2:
                    x_raw = np.linspace(0, 1, len(levels))
                    x_tgt = np.linspace(0, 1, n)
                    tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)
            TL_mean_all.append(tlev_mat.mean(axis=0))
            TL_var_all.append(tlev_mat.var(axis=0))

        # ------------------------------
        # Concatenate
        # ------------------------------
        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)
        TL_mean = np.concatenate(TL_mean_all)
        TL_var = np.concatenate(TL_var_all)

        # normalize for colorbars
        TL_norm  = (TL_mean - global_TL_min) / (global_TL_max - global_TL_min + 1e-6)
        VAR_norm = (TL_var  - global_VAR_min) / (global_VAR_max - global_VAR_min + 1e-6)

        # -------------------------------------------------------
        # 1. INDIVIDUAL TRACES (COLORED BY EACH OWN TRAINING LVL)
        # -------------------------------------------------------
        per_subject = {}
        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            # get training-level raw data
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            for subj, curve in zip(info["subjects"], info["curves"]):
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values

                if len(levels) < 2:
                    continue

                # resample TL for this subject
                x_raw = np.linspace(0, 1, len(levels))
                x_tgt = np.linspace(0, 1, n)
                TL_subj = np.interp(x_tgt, x_raw, levels)

                per_subject.setdefault(subj, {"x": [], "y": [], "lev": []})
                per_subject[subj]["x"].append(x_seg)
                per_subject[subj]["y"].append(curve)
                per_subject[subj]["lev"].append(TL_subj)

        # plot each subject as a multi-colored line
        for subj, d in per_subject.items():
            Xs = np.concatenate(d["x"])
            Ys = np.concatenate(d["y"])
            Ls = np.concatenate(d["lev"])

            Lnorm = (Ls - global_TL_min) / (global_TL_max - global_TL_min + 1e-6)

            # draw colored fragments
            for i in range(len(Xs)-1):
                ax.plot(
                    Xs[i:i+2],
                    Ys[i:i+2],
                    color=indiv_cmap(Lnorm[i]),
                    linewidth=1.2,
                    alpha=0.5
                )

        # -------------------------------------------------------
        # 2. SEM (optional) — color by variance
        # -------------------------------------------------------
        if color_sem_by_variance:
            for i in range(len(X)-1):
                ax.fill_between(
                    X[i:i+2],
                    MEAN[i:i+2] - SEM[i:i+2],
                    MEAN[i:i+2] + SEM[i:i+2],
                    color=sem_cmap(VAR_norm[i]),
                    alpha=0.30
                )
        else:
            ax.fill_between(X, MEAN-SEM, MEAN+SEM, color="lightgray", alpha=0.15)

        # -------------------------------------------------------
        # 3. Mean curve — colored by AVG training level
        # -------------------------------------------------------
        for i in range(len(X)-1):
            ax.plot(X[i:i+2], MEAN[i:i+2],
                    color=mean_cmap(TL_norm[i]),
                    linewidth=3)

        # session boundaries
        off = 0
        for sess in sessions:
            ax.axvline(off, linestyle="--", color="gray", alpha=0.3)
            off += 1

        ax.set_title(f"{group.upper()} — Mean + Individual Training-Level Coloring")
        ax.set_xlabel("Normalized Session Progress (0–1 per session)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA)")

    # -------------------------------------------------------
    # Colorbar for training level (OUTSIDE the plot)
    # -------------------------------------------------------

    sm1 = plt.cm.ScalarMappable(
        cmap=mean_cmap,
        norm=plt.Normalize(vmin=global_TL_min, vmax=global_TL_max)
    )
    sm1.set_array([])

    # Add new axis to the right side of the figure
    # [left, bottom, width, height] in figure coordinates
    cbar_ax1 = fig.add_axes([1, 0.25, 0.02, 0.5])  
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label("Training Level", fontsize=12)

    # -------------------------------------------------------
    # Colorbar for variance (if used)
    # -------------------------------------------------------
    if color_sem_by_variance:
        sm2 = plt.cm.ScalarMappable(
            cmap=sem_cmap,
            norm=plt.Normalize(vmin=global_VAR_min, vmax=global_VAR_max)
        )
        sm2.set_array([])

        cb2 = fig.colorbar(sm2, ax=axes, location="left", shrink=0.7)
        cb2.set_label("Training-Level Variance")

    plt.tight_layout()
    plt.show()

plot_mean_and_individual_training_colored(df, session_results,
                                          color_sem_by_variance=False)

# %%
