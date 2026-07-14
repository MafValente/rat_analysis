# %%
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.font_manager as fm

# ----------------------------
# STARTING DIR (optional)
# ----------------------------
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis")

# ----------------------------
# IMPORT YOUR HELPERS
# ----------------------------
from Helpers.DataHelpers import restrict_subjects, prepare_data

# Optional sigmoid/JND fitting
try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------------------
# USER PATHS
# ----------------------------
CSV_PATH = "merged_all_subjects.csv"
META_CSV = "sex_gen.csv"   # must contain at least: animal, genotype (sex optional)

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

DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[LINE, COHORT])
os.chdir(DATA_DIR)

# ----------------------------
# FIGURE STYLE (match your old script)
# ----------------------------
mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "TeX Gyre Heros", "Arial", "sans-serif"]
plt.rcParams["axes.labelpad"] = 12

font_path = fm.findfont(mpl.font_manager.FontProperties(family=mpl.rcParams["font.sans-serif"]))
print(f"The font being used is: {font_path}")

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
SUPTITLE_FONTSIZE = 24

def shift_axes(ax_list, dx=0, dy=0):
    """Shift axes in ax_list by dx and dy (figure coordinate fractions)."""
    for ax in ax_list:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])

# ----------------------------
# MATH HELPERS
# ----------------------------
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))

def sem_nan(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))

def _fit_sigmoid(x, y):
    """Fit sigmoid; returns params [upper, lower, x0, k] or None."""
    if not HAVE_SCIPY:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return None

    lb = [0.5, 0.0, -25.0, 0.001]
    ub = [1.0, 0.5,  25.0, 5.0]
    p0 = [min(0.98, np.nanmax(y)), max(0.02, np.nanmin(y)), 0.0, 0.2]

    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=(lb, ub), maxfev=25000)
        return popt
    except Exception:
        return None

# ----------------------------
# CSV ROBUST READERS
# ----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read CSV that might be comma OR semicolon separated.
    Uses a quick delimiter sniff so we can keep the fast C engine (and low_memory=False).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample_lines = []
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            sample_lines.append(line)
    sample = "".join(sample_lines)

    comma = sample.count(",")
    semi = sample.count(";")
    sep = ";" if semi > comma else ","

    df = pd.read_csv(path, sep=sep, engine="c", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if df.shape[1] == 1:
        sep2 = "," if sep == ";" else ";"
        df2 = pd.read_csv(path, sep=sep2, engine="c", low_memory=False)
        if df2.shape[1] > 1:
            df = df2
            df.columns = [c.strip() for c in df.columns]

    return df

def normalize_meta_csv(meta_csv_path: str) -> str:
    """
    Ensures metadata is readable as a normal comma CSV and has an 'animal' column.
    Writes a normalized copy to a temp file and returns that path.
    """
    meta = read_csv_robust(meta_csv_path)

    if "animal" not in meta.columns and "subject" in meta.columns:
        meta = meta.rename(columns={"subject": "animal"})

    if "animal" not in meta.columns:
        raise ValueError(f"Metadata file must contain an 'animal' column. Columns found: {list(meta.columns)}")

    meta["animal"] = meta["animal"].astype(str).str.strip()

    keep_cols = [c for c in ["animal", "genotype", "sex"] if c in meta.columns]
    meta = meta[keep_cols].copy()

    tmp_dir = Path(tempfile.gettempdir())
    out_path = tmp_dir / f"sex_gen__normalized_{os.getpid()}.csv"
    meta.to_csv(out_path, index=False)
    print(f"[meta] normalized metadata written to: {out_path}")
    return str(out_path)

# ----------------------------
# BUILDERS THAT TAKE A DATAFRAME
# ----------------------------
def standardize_trials_df(
    df: pd.DataFrame,
    abls=(20, 40, 60),
    ild_values=(-16, -8, -4, -2, -1, 1, 2, 4, 8, 16),
    stim_dur_label_keep=("RT",),
    exclude_repeats=True,
):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["batch_name"] = df["batch"].astype(str)
    df["animal"] = df["animal"].astype(str)

    df["ABL_plot"] = pd.to_numeric(df["ABL"], errors="coerce").round().astype("Int64")
    df.loc[df["ABL_plot"] >= 58, "ABL_plot"] = 60  # map 58/59/60 -> 60

    df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
    df["choice"] = pd.to_numeric(df["response_poke"], errors="coerce")  # +1 right, -1 left

    valid = (
        df["abort_type"].isna()
        & df["choice"].notna()
        & df["ABL_plot"].notna()
        & df["ILD"].notna()
    )

    if exclude_repeats and "repeated_trial" in df.columns:
        valid = valid & (pd.to_numeric(df["repeated_trial"], errors="coerce").fillna(0) == 0)

    if stim_dur_label_keep is not None and "stim_dur_label" in df.columns:
        valid = valid & df["stim_dur_label"].isin(stim_dur_label_keep)

    out = df.loc[valid].copy()
    out = out[out["ABL_plot"].isin(list(abls))].copy()
    if ild_values is not None:
        out = out[out["ILD"].isin(list(ild_values))].copy()

    out["ABL"] = out["ABL_plot"].astype(int)
    return out

def build_psychometric(df_trials: pd.DataFrame, abls=(20, 40, 60)):
    merged_valid = standardize_trials_df(df_trials, abls=abls)

    ABLS = list(abls)
    color_map = {20: "tab:blue", 40: "tab:orange", 60: "tab:red"}
    COLORS = [color_map.get(int(a), "tab:gray") for a in ABLS]

    unique_animal_identifiers = sorted(
        list({(b, a) for b, a in zip(merged_valid["batch_name"], merged_valid["animal"])})
    )

    ilds_dict = {
        abl: np.array(sorted(merged_valid.loc[merged_valid["ABL"] == abl, "ILD"].unique()), dtype=float)
        for abl in ABLS
    }

    tmp = merged_valid.copy()
    tmp["p_right"] = (tmp["choice"] == 1).astype(float)
    psycho_tbl = (
        tmp.groupby(["batch_name", "animal", "ABL", "ILD"])["p_right"]
           .mean()
           .reset_index()
    )

    x_smooth_dict = {}
    all_sigmoid_curves_dict = {}
    mean_params_dict = {}
    mean_sigmoid_dict = {}

    black_plot_as = "mean_of_params"

    for abl in ABLS:
        ilds = ilds_dict[abl]
        if len(ilds) == 0:
            x_smooth_dict[abl] = None
            all_sigmoid_curves_dict[abl] = []
            mean_params_dict[abl] = None
            mean_sigmoid_dict[abl] = None
            continue

        x_smooth = np.linspace(ilds.min(), ilds.max(), 400)
        x_smooth_dict[abl] = x_smooth

        params_list = []
        yfits = []

        for (batch, animal) in unique_animal_identifiers:
            sub = psycho_tbl[
                (psycho_tbl["batch_name"] == batch) &
                (psycho_tbl["animal"] == animal) &
                (psycho_tbl["ABL"] == abl)
            ]
            if sub.empty:
                continue

            y = []
            for ild in ilds:
                v = sub.loc[sub["ILD"] == ild, "p_right"]
                y.append(float(v.iloc[0]) if len(v) else np.nan)

            popt = _fit_sigmoid(ilds, y)
            if popt is None:
                continue

            params_list.append(popt)
            yfits.append(sigmoid(x_smooth, *popt))

        all_sigmoid_curves_dict[abl] = yfits
        if len(yfits):
            params_arr = np.asarray(params_list, dtype=float)
            mean_params_dict[abl] = np.nanmean(params_arr, axis=0)
            mean_sigmoid_dict[abl] = np.nanmean(np.vstack(yfits), axis=0)
        else:
            mean_params_dict[abl] = None
            mean_sigmoid_dict[abl] = None

    return dict(
        ABLS=ABLS,
        COLORS=COLORS,
        black_plot_as=black_plot_as,
        ilds_dict=ilds_dict,
        mean_params_dict=mean_params_dict,
        mean_sigmoid_dict=mean_sigmoid_dict,
        x_smooth_dict=x_smooth_dict,
        unique_animal_identifiers=unique_animal_identifiers,
        merged_valid=merged_valid,
        all_sigmoid_curves_dict=all_sigmoid_curves_dict,
        psycho_tbl=psycho_tbl,
    )

def build_chronometric(
    df_trials: pd.DataFrame,
    abls=(20, 40, 60),
    abs_ild_ticks=(1, 2, 4, 8, 16),
    rt_col="timed_rt",
    rt_range=(0.05, 1.0),
):
    df = df_trials.copy()
    df["batch_name"] = df["batch"].astype(str)
    df["animal_id"] = df["animal"].astype(str)

    df = prepare_data(df, session_col="session", trial_col="trial")
    df = df[df["trial_is_repeat"] == False].copy()
    df = df[df["training_level"] == 16].copy()

    sess = pd.to_numeric(df["session_type"], errors="coerce")
    sd   = pd.to_numeric(df["stim_dur"], errors="coerce")
    df = df[(sess == 1) | (sd == 6000)].copy()

    df["ABL_plot"] = pd.to_numeric(df["ABL"], errors="coerce").round().astype("Int64")
    df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
    df["abs_ILD"] = df["ILD"].abs()
    df["rt"] = pd.to_numeric(df[rt_col], errors="coerce")

    valid = (
        df["abort_type"].isna()
        & df["ABL_plot"].isin(list(abls))
        & df["abs_ILD"].isin(list(abs_ild_ticks))
        & df["rt"].between(rt_range[0], rt_range[1])
    )

    if "repeated_trial" in df.columns:
        valid = valid & (pd.to_numeric(df["repeated_trial"], errors="coerce").fillna(0) == 0)
    if "stim_dur_label" in df.columns:
        valid = valid & df["stim_dur_label"].isin(["RT"])

    d = df.loc[valid].copy()

    all_chrono_data_df = (
        d.groupby(["ABL_plot", "batch_name", "animal_id", "abs_ILD"])["rt"]
         .mean()
         .reset_index(name="mean")
         .rename(columns={"ABL_plot": "ABL"})
    )

    plot_abls = list(abls)
    abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:red"}  # NO GREEN

    grand_means_data = {}
    for abl in plot_abls:
        sub = all_chrono_data_df[all_chrono_data_df["ABL"] == abl]
        stats = (
            sub.groupby("abs_ILD")["mean"]
               .agg(["mean", sem_nan])
               .reset_index()
               .rename(columns={"sem_nan": "sem"})
        )
        grand_means_data[abl] = stats

    rt_vs_ild = (
        all_chrono_data_df.groupby("abs_ILD")["mean"]
                         .agg(["mean", sem_nan])
                         .reset_index()
                         .rename(columns={"sem_nan": "sem"})
    )

    rt_vs_abl = (
        all_chrono_data_df.groupby("ABL")["mean"]
                         .agg(["mean", sem_nan])
                         .reset_index()
                         .rename(columns={"sem_nan": "sem"})
    )

    return dict(
        plot_abls=plot_abls,
        all_chrono_data_df=all_chrono_data_df,
        grand_means_data=grand_means_data,
        abl_colors=abl_colors,
        abs_ild_ticks=list(abs_ild_ticks),
        rt_vs_ild=rt_vs_ild,
        rt_vs_abl=rt_vs_abl,
    )

def build_jnd(plot_data):
    ABLS = plot_data["ABLS"]
    ilds_dict = plot_data["ilds_dict"]
    psycho_tbl = plot_data["psycho_tbl"]
    unique_ids = plot_data["unique_animal_identifiers"]

    jnds = {abl: {} for abl in ABLS}
    mean_jnd = {}

    if not HAVE_SCIPY:
        return dict(
            jnds=jnds,
            mean_jnd=mean_jnd,
            grand_mean_jnd=np.nan,
            ABLS=ABLS,
            animals_with_mean=[],
            mean_jnds=np.array([]),
            diff_within=np.array([]),
        )

    def sigmoid_inv(p, upper, lower, x0, k):
        eps = 1e-6
        p = np.clip(p, lower + eps, upper - eps)
        return x0 - (1.0 / k) * np.log((upper - lower) / (p - lower) - 1)

    def jnd_from_params(params):
        upper, lower, x0, k = params
        p25 = lower + 0.25 * (upper - lower)
        p75 = lower + 0.75 * (upper - lower)
        x25 = sigmoid_inv(p25, upper, lower, x0, k)
        x75 = sigmoid_inv(p75, upper, lower, x0, k)
        return 0.5 * (x75 - x25)

    for (batch, animal) in unique_ids:
        per_abl = []
        for abl in ABLS:
            ilds = ilds_dict[abl]
            sub = psycho_tbl[
                (psycho_tbl["batch_name"] == batch) &
                (psycho_tbl["animal"] == animal) &
                (psycho_tbl["ABL"] == abl)
            ]
            if sub.empty or len(ilds) < 5:
                continue

            y = []
            for ild in ilds:
                v = sub.loc[sub["ILD"] == ild, "p_right"]
                y.append(float(v.iloc[0]) if len(v) else np.nan)

            popt = _fit_sigmoid(ilds, y)
            if popt is None:
                continue

            j = jnd_from_params(popt)
            jnds[abl][animal] = j
            per_abl.append(j)

        if per_abl:
            mean_jnd[animal] = float(np.mean(per_abl))

    animals_with_mean = list(mean_jnd.keys())
    mean_jnds = np.array([mean_jnd[a] for a in animals_with_mean], dtype=float)
    grand_mean_jnd = float(np.nanmean(mean_jnds)) if len(mean_jnds) else np.nan

    diff_within = []
    for abl in ABLS:
        for animal, j in jnds[abl].items():
            if animal in mean_jnd:
                diff_within.append(j - mean_jnd[animal])
    diff_within = np.array(diff_within, dtype=float)

    return dict(
        jnds=jnds,
        mean_jnd=mean_jnd,
        grand_mean_jnd=grand_mean_jnd,
        ABLS=ABLS,
        animals_with_mean=animals_with_mean,
        mean_jnds=mean_jnds,
        diff_within=diff_within,
    )

def build_quantiles(
    df_trials: pd.DataFrame,
    abls=(20, 40, 60),
    abs_ild_ticks=(1, 2, 4, 8, 16),
    quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
    rt_col="timed_rt",
    rt_range=(0.05, 1.0),
    min_cut_quantile=0.01,
):
    df = df_trials.copy()
    df["batch_name"] = df["batch"].astype(str)
    df["animal_id"] = df["animal"].astype(str)

    df = prepare_data(df, session_col="session", trial_col="trial")
    df = df[df["trial_is_repeat"] == False].copy()
    df = df[df["training_level"] == 16].copy()

    sess = pd.to_numeric(df["session_type"], errors="coerce")
    sd   = pd.to_numeric(df["stim_dur"], errors="coerce")
    df = df[(sess == 1) | (sd == 6000)].copy()

    df["ABL_plot"] = pd.to_numeric(df["ABL"], errors="coerce").round().astype("Int64")
    df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
    df["abs_ILD"] = df["ILD"].abs()
    df["rt"] = pd.to_numeric(df[rt_col], errors="coerce")

    valid = (
        df["abort_type"].isna()
        & df["ABL_plot"].isin(list(abls))
        & df["abs_ILD"].isin(list(abs_ild_ticks))
        & df["rt"].between(rt_range[0], rt_range[1])
    )
    if "repeated_trial" in df.columns:
        valid = valid & (pd.to_numeric(df["repeated_trial"], errors="coerce").fillna(0) == 0)
    if "stim_dur_label" in df.columns:
        valid = valid & df["stim_dur_label"].isin(["RT"])

    d = df.loc[valid].copy()

    ABL_arr = list(abls)
    abs_ILD_arr = np.array(sorted(abs_ild_ticks), dtype=float)
    plotting_quantiles = np.array(list(quantiles), dtype=float)

    min_RT_cut_by_ILD = {}
    for abs_ild in abs_ILD_arr:
        vals = d.loc[d["abs_ILD"] == abs_ild, "rt"].dropna().values
        min_RT_cut_by_ILD[abs_ild] = float(np.quantile(vals, min_cut_quantile)) if len(vals) else np.nan

    def per_animal_quantiles(df_sub: pd.DataFrame, value_col: str, apply_cut: bool):
        mats = []
        for (batch, animal), g in df_sub.groupby(["batch_name", "animal_id"]):
            mat = np.full((len(plotting_quantiles), len(abs_ILD_arr)), np.nan, dtype=float)
            for j, abs_ild in enumerate(abs_ILD_arr):
                vals = g.loc[g["abs_ILD"] == abs_ild, value_col].dropna().values
                if apply_cut:
                    cut = min_RT_cut_by_ILD.get(abs_ild, np.nan)
                    if np.isfinite(cut):
                        vals = vals[vals >= cut]
                if len(vals):
                    mat[:, j] = np.quantile(vals, plotting_quantiles)
            mats.append(mat)
        return np.stack(mats, axis=0) if mats else None

    abl_colors = ["tab:blue", "tab:orange", "tab:red"]

    mean_unscaled, sem_unscaled = {}, {}
    mean_scaled, sem_scaled = {}, {}

    for abl in ABL_arr:
        sub = d[d["ABL_plot"] == abl].copy()

        stack = per_animal_quantiles(sub, "rt", apply_cut=True)
        if stack is None:
            mean_unscaled[abl] = np.full((len(plotting_quantiles), len(abs_ILD_arr)), np.nan)
            sem_unscaled[abl]  = np.full((len(plotting_quantiles), len(abs_ILD_arr)), np.nan)
        else:
            mean_unscaled[abl] = np.nanmean(stack, axis=0)
            n = np.sum(np.isfinite(stack), axis=0)
            sem_unscaled[abl] = np.nanstd(stack, axis=0) / np.sqrt(np.maximum(n, 1))

        sub["rt_scaled"] = sub.apply(lambda r: r["rt"] - min_RT_cut_by_ILD.get(r["abs_ILD"], np.nan), axis=1)
        stack_s = per_animal_quantiles(sub, "rt_scaled", apply_cut=False)
        if stack_s is None:
            mean_scaled[abl] = np.full((len(plotting_quantiles), len(abs_ILD_arr)), np.nan)
            sem_scaled[abl]  = np.full((len(plotting_quantiles), len(abs_ILD_arr)), np.nan)
        else:
            mean_scaled[abl] = np.nanmean(stack_s, axis=0)
            n = np.sum(np.isfinite(stack_s), axis=0)
            sem_scaled[abl] = np.nanstd(stack_s, axis=0) / np.sqrt(np.maximum(n, 1))

    return dict(
        ABL_arr=ABL_arr,
        abs_ILD_arr=abs_ILD_arr,
        plotting_quantiles=plotting_quantiles,
        mean_unscaled=mean_unscaled,
        sem_unscaled=sem_unscaled,
        mean_scaled=mean_scaled,
        sem_scaled=sem_scaled,
        abl_colors=abl_colors,
    )

# ----------------------------
# PLOTTING: ONE FIGURE FOR ONE VIEW
# ----------------------------
def plot_fig1_for_view(view_name: str, df_view: pd.DataFrame, out_prefix: str):
    plot_data = build_psychometric(df_view, abls=(20, 40, 60))
    chrono_data = build_chronometric(df_view, abls=(20, 40, 60))
    jnd_data = build_jnd(plot_data)
    quantile_data = build_quantiles(df_view, abls=(20, 40, 60))

    ABLS = plot_data["ABLS"]
    COLORS = plot_data["COLORS"]
    black_plot_as = plot_data["black_plot_as"]
    ilds_dict = plot_data["ilds_dict"]
    mean_params_dict = plot_data["mean_params_dict"]
    mean_sigmoid_dict = plot_data["mean_sigmoid_dict"]
    x_smooth_dict = plot_data["x_smooth_dict"]
    unique_animal_identifiers = plot_data["unique_animal_identifiers"]
    all_sigmoid_curves_dict = plot_data["all_sigmoid_curves_dict"]
    psycho_tbl = plot_data["psycho_tbl"]

    plot_abls = chrono_data["plot_abls"]
    all_chrono_data_df = chrono_data["all_chrono_data_df"]
    grand_means_data = chrono_data["grand_means_data"]
    abl_colors = chrono_data["abl_colors"]
    abs_ild_ticks = chrono_data["abs_ild_ticks"]
    rt_vs_ild = chrono_data["rt_vs_ild"]
    rt_vs_abl = chrono_data["rt_vs_abl"]

    jnds = jnd_data["jnds"]
    mean_jnd = jnd_data["mean_jnd"]
    grand_mean_jnd = jnd_data["grand_mean_jnd"]
    animals_with_mean = jnd_data["animals_with_mean"]
    mean_jnds = jnd_data["mean_jnds"]
    diff_within = jnd_data["diff_within"]

    ABL_arr = quantile_data["ABL_arr"]
    abs_ILD_arr = quantile_data["abs_ILD_arr"]
    plotting_quantiles = quantile_data["plotting_quantiles"]
    mean_unscaled = quantile_data["mean_unscaled"]
    sem_unscaled = quantile_data["sem_unscaled"]
    mean_scaled = quantile_data["mean_scaled"]
    sem_scaled = quantile_data["sem_scaled"]
    abl_colors_quant = quantile_data["abl_colors"]

    # ---------- Layout (KEEP spacing EXACTLY the same) ----------
    fig = plt.figure(figsize=(25, 30))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06)
    gs = GridSpec(
        5, 6,
        figure=fig,
        hspace=0.3,
        wspace=0.0,
        width_ratios=[1, 1, 1, 1, 1, 1],
        height_ratios=[1, 0.5, 0.5, 0.5, 0],  # KEEP
    )

    # ---------- Psychometric row ----------
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

    # Precompute per-animal psychometric points
    psy_dict = {}
    for abl in ABLS:
        ilds = ilds_dict[abl]
        mat = []
        for (batch, animal) in unique_animal_identifiers:
            sub = psycho_tbl[
                (psycho_tbl["batch_name"] == batch) &
                (psycho_tbl["animal"] == animal) &
                (psycho_tbl["ABL"] == abl)
            ]
            y = []
            for ild in ilds:
                v = sub.loc[sub["ILD"] == ild, "p_right"]
                y.append(float(v.iloc[0]) if len(v) else np.nan)
            mat.append(np.array(y, dtype=float))
        psy_dict[abl] = np.array(mat, dtype=float)

    # First 3 panels
    for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
        ax = axes[idx]
        ilds = ilds_dict[abl]
        x_smooth = x_smooth_dict[abl]

        for y_fit in all_sigmoid_curves_dict.get(abl, []):
            ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)

        if black_plot_as == "mean_of_params" and mean_params_dict.get(abl) is not None:
            ax.plot(x_smooth, sigmoid(x_smooth, *mean_params_dict[abl]),
                    color="black", linewidth=3, label="Avg sigmoid fit")
        elif black_plot_as == "mean_of_sigmoids" and mean_sigmoid_dict.get(abl) is not None:
            ax.plot(x_smooth, mean_sigmoid_dict[abl],
                    color="black", linewidth=3, label="Avg sigmoid fit")

        all_psycho_points = psy_dict[abl]
        mean_psycho = np.nanmean(all_psycho_points, axis=0)
        n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
        sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(np.maximum(n_points, 1))

        ax.errorbar(ilds, mean_psycho, yerr=sem_psycho, fmt="o",
                    color=color, capsize=0, markersize=8.5, label="Mean ± SEM")

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

    # 4th panel: overlay ABLs
    ax4 = axes[3]
    for abl, color in zip(ABLS, COLORS):
        ilds = ilds_dict[abl]
        all_psycho_points = psy_dict[abl]
        mean_psycho = np.nanmean(all_psycho_points, axis=0)
        n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
        sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(np.maximum(n_points, 1))

        ax4.errorbar(ilds, mean_psycho, yerr=sem_psycho, fmt="o", color=color,
                     capsize=0, markersize=8.5, label=f"ABL={abl} mean ± SEM")

        x_smooth = x_smooth_dict[abl]
        if black_plot_as == "mean_of_params" and mean_params_dict.get(abl) is not None:
            ax4.plot(x_smooth, sigmoid(x_smooth, *mean_params_dict[abl]),
                     color=color, linewidth=2, label=f"ABL={abl} curve")
        elif black_plot_as == "mean_of_sigmoids" and mean_sigmoid_dict.get(abl) is not None:
            ax4.plot(x_smooth, mean_sigmoid_dict[abl],
                     color=color, linewidth=2, label=f"ABL={abl} curve")

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
        leg = ax.get_legend()
        if leg:
            leg.prop.set_size(LEGEND_FONTSIZE)

    # ---------- Chronometric row ----------
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

    for i, abl in enumerate(plot_abls):
        ax = chrono_axes[i]
        abl_df = all_chrono_data_df[all_chrono_data_df["ABL"] == abl]

        for (batch_name, animal_id), animal_df in abl_df.groupby(["batch_name", "animal_id"]):
            animal_df = animal_df.sort_values("abs_ILD")
            ax.plot(animal_df["abs_ILD"], animal_df["mean"],
                    color=abl_colors[abl], alpha=0.4, linewidth=1.5)

        grand = grand_means_data[abl]
        ax.errorbar(grand["abs_ILD"], grand["mean"], yerr=grand["sem"],
                    fmt="o", color=abl_colors[abl], markersize=8.5, capsize=0, linewidth=0, zorder=3)
        ax.plot(grand["abs_ILD"], grand["mean"], color="black", linewidth=2.5, zorder=2)

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
        ax.set_yticks([0.1, 0.45])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # overlay chrono
    ax4_chrono = chrono_axes[3]
    for abl, stats_data in grand_means_data.items():
        ax4_chrono.errorbar(stats_data["abs_ILD"], stats_data["mean"], yerr=stats_data["sem"],
                            fmt="o-", color=abl_colors[abl], markersize=8.5, capsize=0, linewidth=0, zorder=3)
        ax4_chrono.plot(stats_data["abs_ILD"], stats_data["mean"],
                        color=abl_colors[abl], linewidth=2.5, zorder=2)
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

    # chrono summary (old look)
    gs_summary = gs[2, 5].subgridspec(2, 1, hspace=0.05)
    ax_ild = fig.add_subplot(gs_summary[0, 0])
    ax_abl = fig.add_subplot(gs_summary[1, 0])

    ax_ild.errorbar(rt_vs_ild["abs_ILD"], rt_vs_ild["mean"], yerr=rt_vs_ild["sem"],
                    fmt="o", color="k", capsize=0, markersize=6, linewidth=2)
    ax_ild.set_xlabel("|ILD|", fontsize=LABEL_FONTSIZE, ha="right", x=1.4)
    ax_ild.xaxis.set_label_coords(1.4, 0.1)
    ax_ild.set_xscale("log")
    ax_ild.set_xticks(abs_ild_ticks)
    ax_ild.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_ild.xaxis.set_minor_locator(plt.NullLocator())
    ax_ild.spines["top"].set_visible(False)
    ax_ild.spines["right"].set_visible(False)
    ax_ild.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    for i, row in rt_vs_abl.iterrows():
        abl = row["ABL"]
        color = abl_colors.get(abl, "k")
        ax_abl.errorbar(x=i, y=row["mean"], yerr=row["sem"],
                        fmt="o", linestyle="None", color=color, capsize=0, markersize=8.5)
    ax_abl.set_xticks(range(len(rt_vs_abl)))
    ax_abl.set_xticklabels(rt_vs_abl["ABL"].astype(int))
    ax_abl.set_xlabel("ABL", fontsize=LABEL_FONTSIZE, ha="right", x=1.4)
    ax_abl.xaxis.set_label_coords(1.4, 0.1)
    plt.setp(ax_abl.get_yticklabels(), visible=False)
    ax_abl.spines["top"].set_visible(False)
    ax_abl.spines["right"].set_visible(False)
    ax_abl.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    ax_ild.set_ylim(0.1, 0.25)
    ax_ild.set_yticks([0.1, 0.25])
    ax_ild.set_yticklabels(["0.1", "0.25"])
    ax_ild.tick_params(axis="y", labelleft=True, length=0)

    ax_abl.set_ylim(0.1, 0.21)
    ax_abl.set_yticks([0.1, 0.2])
    ax_abl.set_yticklabels(["0.1", "0.2"])
    ax_abl.tick_params(axis="y", labelleft=True, length=0)

    fig.canvas.draw()
    chrono_baseline = ax_chrono_1.get_position().y0
    dy = chrono_baseline - ax_abl.get_position().y0
    shift_axes([ax_ild, ax_abl], dy=dy)

    shift_axes([ax_ild, ax_abl], dx=0.05, dy=-0.01)
    width_factor = 0.65
    for ax in (ax_ild, ax_abl):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * width_factor, pos.height])

    fig.canvas.draw()
    left_edge = ax_ild.get_position().x0 - 0.02
    center_y = 0.5 * (ax_ild.get_position().y1 + ax_abl.get_position().y0)
    fig.text(left_edge - 0.03, center_y, "Mean RT (s)", rotation="vertical",
             ha="center", va="center", fontsize=LABEL_FONTSIZE)

    # ---------- JND column (old look) ----------
    gs_nested = gs[1, 5].subgridspec(2, 1, hspace=-0.2)
    gs_jnd_plot = gs_nested[0, 0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax1_main = fig.add_subplot(gs_jnd_plot[0, 0])
    ax1_hist = fig.add_subplot(gs_jnd_plot[0, 1], sharey=ax1_main)

    gs_var_plot = gs_nested[1, 0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax2_main = fig.add_subplot(gs_var_plot[0, 0])
    ax2_hist = fig.add_subplot(gs_var_plot[0, 1], sharey=ax2_main)

    plot_colors = ["tab:blue", "tab:orange", "tab:red"]

    if len(mean_jnds) == 0 or not np.isfinite(grand_mean_jnd):
        ax1_main.text(0.5, 0.5, "JND requires SciPy\n(curve_fit)",
                      ha="center", va="center", fontsize=16)
        ax1_main.axis("off"); ax2_main.axis("off"); ax1_hist.axis("off"); ax2_hist.axis("off")
    else:
        sorted_idx = np.argsort(mean_jnds)
        sorted_animals = [animals_with_mean[i] for i in sorted_idx]

        for i, animal_id in enumerate(sorted_animals):
            ax1_main.plot(i, mean_jnd[animal_id], "k_", markersize=6, mew=1.5)
            for j, abl in enumerate(ABLS):
                if animal_id in jnds[abl]:
                    ax1_main.plot(i, jnds[abl][animal_id], "o",
                                  color=plot_colors[j], markersize=4, alpha=0.5, linewidth=2)

        ax1_main.axhline(grand_mean_jnd, color="k", linestyle=":", linewidth=1)
        ax1_main.set_xticks([])
        ax1_main.set_ylabel("JND", fontsize=LABEL_FONTSIZE)
        ax1_main.spines["top"].set_visible(False)
        ax1_main.spines["right"].set_visible(False)
        ax1_main.spines["bottom"].set_visible(False)
        ax1_main.tick_params(axis="y", labelsize=TICK_FONTSIZE, length=0)
        ax1_main.set_ylim(1, 4)
        ax1_main.set_yticks([1, 4])

        mu = np.mean(mean_jnds); sd = np.std(mean_jnds)
        x_bar = 0.05
        ax1_hist.plot([x_bar, x_bar], [mu - sd, mu + sd], color="grey", linewidth=3)
        ax1_hist.set_xlim(0, 1); ax1_hist.axis("off")

        for i, animal_id in enumerate(sorted_animals):
            j0 = mean_jnd[animal_id]
            for j, abl in enumerate(ABLS):
                if animal_id in jnds[abl]:
                    ax2_main.plot(i, jnds[abl][animal_id] - j0, "o",
                                 color=plot_colors[j], markersize=5, alpha=0.5)

        ax2_main.axhline(0, color="k", linewidth=1)
        ax2_main.set_xticks([])
        ax2_main.set_ylabel(r"J$_{\text{ABL}}$ - J$_{\mu}$", fontsize=LABEL_FONTSIZE)
        ax2_main.spines["top"].set_visible(False)
        ax2_main.spines["right"].set_visible(False)
        ax2_main.spines["bottom"].set_visible(False)
        ax2_main.tick_params(axis="y", labelsize=TICK_FONTSIZE, length=0)
        ax2_main.set_ylim(-1.5, 1.5)
        ax2_main.set_yticks([-1.5, 0, 1.5])
        ax2_main.set_yticklabels(["-1.5", "0", "1.5"])

        mu_d = np.mean(diff_within) if len(diff_within) else 0.0
        sd_d = np.std(diff_within) if len(diff_within) else 0.0
        ax2_hist.plot([x_bar, x_bar], [mu_d - sd_d, mu_d + sd_d], color="grey", linewidth=3)
        ax2_hist.set_xlim(0, 1); ax2_hist.axis("off")

    # Align JND with psychometric baseline (old behavior)
    fig.canvas.draw()
    psycho_baseline = ax_psych_1.get_position().y0
    jnd_baseline = ax2_main.get_position().y0
    shift_axes([ax1_main, ax1_hist, ax2_main, ax2_hist], dx=0.05, dy=(psycho_baseline - jnd_baseline))

    # ---------- Quantile row ----------
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

    for col, abl in enumerate(ABL_arr):
        ax = quantile_axes[col]
        q_mat = mean_unscaled[abl]
        s_mat = sem_unscaled[abl]
        for q_idx in range(len(plotting_quantiles)):
            ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=s_mat[q_idx, :],
                        marker="o", linestyle="-", color=abl_colors_quant[col])

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xscale("log")
        ax.set_xticks(abs_ILD_arr)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.set_ylim(0.15, 0.6)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_xlabel("|ILD| (dB)", fontsize=LABEL_FONTSIZE)
        if col == 0:
            ax.set_ylabel("RT(s)", fontsize=LABEL_FONTSIZE)

    ax_overlay = fig.add_subplot(gs_quant[0, 3])
    shift_axes([ax_overlay], dx=0.04)
    if hasattr(ax_overlay, "set_box_aspect"):
        ax_overlay.set_box_aspect(1)
    else:
        ax_overlay.set_aspect("equal", adjustable="box")

    for col, abl in enumerate(ABL_arr):
        q_mat = mean_scaled[abl]
        s_mat = sem_scaled[abl]
        for q_idx in range(len(plotting_quantiles)):
            ax_overlay.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=s_mat[q_idx, :],
                                marker="o", linestyle="-", color=abl_colors_quant[col])

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

    # ------------------------------------------------------------
    # CUT the unused top GridSpec row WITHOUT changing plot spacing
    # (do NOT change hspace/wspace/height_ratios)
    # ------------------------------------------------------------
    fig.canvas.draw()
    vis_axes = [ax for ax in fig.axes if ax.get_visible()]
    if vis_axes:
        top_now = max(ax.get_position().y1 for ax in vis_axes)
        desired_top = 0.94  # leaves room for the suptitle
        dy_global = max(0.0, desired_top - top_now)
        if dy_global > 0:
            shift_axes(vis_axes, dy=dy_global)
            for t in fig.texts:
                x, y = t.get_position()
                t.set_position((x, y + dy_global))

    # ---------- Final ----------
    fig.suptitle(f"Figure 1 – genotype: {view_name}", fontsize=SUPTITLE_FONTSIZE, y=0.985)

    # IMPORTANT: do NOT call tight_layout here (it fights shift_axes/set_position)
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight", format="pdf")
    plt.show()
    print(f"[saved] {out_prefix}.png and {out_prefix}.pdf")

# ----------------------------
# GENOTYPE VIEWS
# ----------------------------
def get_views(meta_csv_normalized: str):
    return [
        ("wt",  lambda d: restrict_subjects(d, meta_csv_normalized, genotypes="wt",
                                            subject_col="animal", genotype_col="genotype", attach_meta=True)),
        ("het", lambda d: restrict_subjects(d, meta_csv_normalized, genotypes="het",
                                            subject_col="animal", genotype_col="genotype", attach_meta=True)),
        ("hom", lambda d: restrict_subjects(d, meta_csv_normalized, genotypes="hom",
                                            subject_col="animal", genotype_col="genotype", attach_meta=True)),
    ]

# ----------------------------
# MAIN
# ----------------------------
trials_df = read_csv_robust(CSV_PATH)
print(f"[trials] loaded: {trials_df.shape} from {CSV_PATH}")

meta_norm = normalize_meta_csv(META_CSV)

for view_name, view_fn in get_views(meta_norm):
    df_view = view_fn(trials_df)
    n_animals = df_view["animal"].nunique() if "animal" in df_view.columns else 0
    print(f"[view {view_name}] trials={len(df_view)} animals={n_animals}")
    if len(df_view) == 0:
        print(f"[view {view_name}] empty -> skipping")
        continue
    plot_fig1_for_view(view_name, df_view, out_prefix=f"fig1_{view_name}_from_csv")

# %%
