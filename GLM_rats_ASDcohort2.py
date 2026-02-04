#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import Helpers.DataHelpers as DataHelpers
import os

# --- R + rpy2 imports ---
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

print("✅ Imported Python and R libraries successfully.")

# Load R libraries
lme4 = importr("lme4")
stats = importr("stats")
MuMIn = importr("MuMIn")   # for R²

print("✅ Loaded R packages: lme4, stats, MuMIn")

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

#%%
# =======================
# Load and preprocess data
# =======================

print("📂 Loading and preprocessing data...")

cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file)

df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"] == False].copy()
df = df[df["training_level"] == 16].copy()

sess = pd.to_numeric(df["session_type"], errors="coerce")
sd   = pd.to_numeric(df["stim_dur"], errors="coerce")
df = df[(sess == 1) | (sd == 6000)].copy()

print(f"✅ Loaded file: {cohort_file} ({len(df)} rows)")

# ---- MATCH MATLAB FILTERING LOGIC ----
# MATLAB:
# out = m;
# out.Response = (out.Response + 1)/2;
# out.fix_time = out.fix_time/1000;
# out.ABLc = categorical(out.ABL);
# out.subject = categorical(out.subject);
# out.repeated_trial = categorical(out.repeated_trial);
# out.repeated_trial = out.repeated_trial == 'True';
# out = out(out.training_level == 16,:);
# out = out(out.repeated_trial == false,:);
# out = out(out.session >= 13,:);

out = df.copy()

# Unique session ID for each animal
out["session_id"] = out["animal"] + "_S" + out["session"].astype(int).astype(str).str.zfill(3)
# ASD0011_S001, ASD0011_S002, ...

# Rename to match MATLAB
out = out.rename(columns={'animal': 'subject', 'response_poke': 'Response'})
out['Response'] = (out['Response'] + 1) / 2
out['fix_time'] = out['fix_time'] / 1000

# Categoricals
out['ABLc'] = out['ABL'].astype('category')
out['subject'] = out['subject'].astype('category')
out['repeated_trial'] = out['repeated_trial'].astype('category')


out = out[out['session'] >= 13]

print(f"✅ After filtering: {len(out)} rows, {out['subject'].nunique()} subjects")

# subject groups
wt_subj = ['ASD0007','ASD0011','ASD0014','ASD0015','ASD0017']
het_subj = ['ASD0009','ASD0012','ASD0016','ASD0018','ASD0021']
hom_subj = ['ASD0008','ASD0010','ASD0013','ASD0019','ASD0020', 'ASD0022']

out_type = [
    out[out['subject'].isin(wt_subj)],
    out[out['subject'].isin(het_subj)],
    out[out['subject'].isin(hom_subj)]
]
print("✅ Prepared data subsets (WT/HET/HOM).")

# ==========================
# Helpers
# ==========================

# MATLAB: zscore_fun = @(tab,var) (tab.(var) - nanmean(tab.(var)))./nanstd(tab.(var));
def zscore_fun(series: pd.Series):
    # ddof=1 → same as MATLAB nanstd default
    return (series - series.mean()) / series.std(ddof=1)

def join_slopes(slopes):
    if slopes is None:
        return ""

    # If it's a single string, just return it
    if isinstance(slopes, str):
        return slopes

    # Filter out empty / None
    slopes = [s for s in slopes if s]

    if not slopes:
        return ""

    # If we got a list of single characters, assume it's a broken-up string
    if all(isinstance(s, str) and len(s) == 1 for s in slopes):
        return "".join(slopes)

    # Normal case: list/tuple of full variable names
    return " + ".join(slopes)
# ==========================
# Model configuration
# ==========================

response_var = 'Response'

use_current = True
use_pre = True

preds_grouped = []
var_group_names = []

if use_current:
    var_group_names.append('current')
    preds_grouped.append(['ABL','ILD', 'fix_time_long', 'ILD:ABL','trial','ILD:trial','ILD:fix_time_long', 'trial:fix_time_long'])
if use_pre:
    var_group_names.append('pre')
    preds_grouped.append(['Pre_choice', 'Pre_success','ILD:Pre_success','Pre_success:Pre_choice'])

all_fixed_predictors = sum(preds_grouped, [])
pred_to_zscore = all_fixed_predictors

print(f"✅ Using predictors: {all_fixed_predictors}")

# Full list of random slopes for subject (from MATLAB)
subject_random_slopes = [
 'ABL','ILD','ILD:ABL',
    'trial','ILD:trial',
    'Pre_choice','Pre_success','ILD:Pre_success','Pre_success:Pre_choice'
]
sessionID_random_slopes = 'ILD:ABL'
# ==========================
# Fit models for WT / HET / HOM
# ==========================

mdle_all = []
out_glm3 = []

for ktype, df_group in enumerate(out_type, start=1):
    print(f"\n🔹 Starting group {ktype} ({len(df_group)} rows)")
    df_group = df_group.copy()
    subjects = df_group['subject'].unique()
    print(f"   Found {len(subjects)} subjects.")

    df_out = []

    # === replicate MATLAB per-subject/per-session loop ===
    for subj_i, subj in enumerate(subjects, start=1):
        subj_df = df_group[df_group['subject'] == subj].copy()

        # Reindex session ids within subject to 1..N (similar to MATLAB renumbering)
        sess_unique = np.sort(subj_df['session'].unique())
        sess_map = {s: i+1 for i, s in enumerate(sess_unique)}
        subj_df['session'] = subj_df['session'].map(sess_map)

        sessions = np.sort(subj_df['session'].unique())
        print(f"     [{subj_i}/{len(subjects)}] Processing subject {subj} ({len(sessions)} sessions)...")

        for sess in sessions:
            s_df = subj_df[subj_df['session'] == sess].copy()
            s_df = s_df.sort_values('trial').reset_index(drop=True)

            # Recenter trial index within this subject+session
            # (MATLAB subtracts nanmean(trial); here we mimic exactly)
            s_df['trial'] = s_df['trial'] - s_df['trial'].mean()

            # # Turn ILD into sign and difficulty
            # s_df['abs_ILD'] = s_df['ILD'].abs()
            # s_df['sign_ILD'] = np.sign(s_df['ILD'])

            # Pre-trial vars (MATLAB: shift by 1; first row stays NaN initially)
            s_df['Pre_ILD'] = s_df['ILD'].shift(1)
            s_df['Pre_ABL'] = s_df['ABL'].shift(1)
            s_df['Pre_choice'] = s_df['Response'].shift(1)
            s_df['Pre_success'] = s_df['success'].shift(1)

            # Avoid NaNs as in MATLAB (for some predictors)
            s_df['Pre_ILD'] = s_df['Pre_ILD'].fillna(0)
            s_df['Pre_ABL'] = s_df['Pre_ABL'].fillna(0)
            s_df['Pre_choice'] = s_df['Pre_choice'].fillna(0)
            # Pre_success can be NaN on first trial; NaNs get turned to 0 later

            # interactions (subset needed for this model)
            s_df['ILD:ABL'] = s_df['ILD'] * s_df['ABL']
            s_df['ILD:trial'] = s_df['ILD'] * s_df['trial']

            # fix_time_long & interaction
            s_df['fix_time_long'] = (s_df['fix_time'] > s_df['fix_time'].median()).astype(float)
            s_df['ILD:fix_time_long'] = s_df['ILD'] * s_df['fix_time_long']

            s_df['trial:fix_time_long'] = s_df['trial'] * s_df['fix_time_long']

            s_df['ILD:Pre_success'] = s_df['ILD'] * s_df['Pre_success']
            s_df['Pre_success:Pre_choice'] = s_df['Pre_success'] * s_df['Pre_choice']

            # z-score predictors per session & subject (same set as MATLAB pred_to_zscore)
            for col in pred_to_zscore:
                if col in s_df.columns:
                    s_df[col] = zscore_fun(s_df[col])

            df_out.append(s_df)

    df_out = pd.concat(df_out, ignore_index=True)
    print(f"   → Combined subject data: {len(df_out)} rows")

    # Drop NaNs in response (MATLAB: out1_X = out1_X(~isnan(Response),:))
    df_out = df_out.dropna(subset=[response_var])

    # Fill NaNs in numeric columns with 0 (like MATLAB numeric-loop)
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].fillna(0)

    out_glm3.append(df_out)

    # ===========================================================
    # Clean conversion helper for R GLMM
    # ===========================================================
    def prep_for_r_glmm(df):
        fixed = all_fixed_predictors
        random_groups = ['session_id', 'subject']
        response = 'Response'
        keep = [response] + fixed + random_groups
        df = df.loc[:, [c for c in keep if c in df.columns]].copy()

        # Response 0/1
        df[response] = df[response].astype(float).round().astype(int)

        # Numeric predictors
        for c in fixed:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)

        # Grouping vars as strings (R factors)
        for g in random_groups:
            if g in df.columns:
                df[g] = df[g].astype(str)

        # Drop rows with missing data in key vars
        df = df.dropna(subset=[response] + fixed + random_groups)

        return df

    # ===========================================================
    # MATLAB-Equivalent GLMM (R glmer)
    # ===========================================================
    # Fixed effects part
    fixed_part = "1 + " + " + ".join(all_fixed_predictors)

    # RANDOM EFFECTS:
    # (1|session) + (1 + ABL + ILD + ILD:ABL + ILD:Pre_success + ILD:trial +
    #                Pre_choice + Pre_success:Pre_choice + trial | subject)
    # Using || to enforce diagonal covariance (similar to MATLAB CovariancePattern='Diagonal').
    # subject_slopes_str = " + ".join(subject_random_slopes)
    # sessionID_slopes_str = " + ".join(sessionID_random_slopes)

    subject_slopes_str = join_slopes(subject_random_slopes)
    sessionID_slopes_str = join_slopes(sessionID_random_slopes)
    # then in formula:
    #random_part = f"(1 + {sessionID_slopes_str} | session_id)"

    
    random_part = (
        f"(1 + {sessionID_slopes_str} || session_id) + "
        f"(1 + {subject_slopes_str} || subject)"
    )

    formula_str = f"Response ~ {fixed_part} + {random_part}"
    formula = Formula(formula_str)

    print(f"\n=== 🚀 Fitting R GLMM for group {ktype} ===")
    print(f"Formula: {formula_str}")
    print(f"Preparing dataframe for R...")

    df_out = df_out[df_out["trial_is_repeat"]==False].copy()

    df_clean = prep_for_r_glmm(df_out)
    print(f"Dataframe ready: {len(df_clean)} rows, {df_clean['subject'].nunique()} subjects")

    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df_clean)

    print(f"Calling lme4::glmer() in R... this may take a few minutes ⏳")
    glmer_fit = lme4.glmer(
        formula,
        data=df_r,
        family=stats.binomial(link="logit")
    )
    print(f"✅ Finished fitting GLMM for group {ktype}.")

    mdle_all.append([glmer_fit])

#%%
# ==========================
# Save fitted GLMMs and data for later reuse
# ==========================
import pickle

save_path = "glmm_results_cntnap2.pkl"   # choose any filename you like

to_save = {
    "mdle_all": mdle_all,
    "out_glm3": out_glm3,
    "all_fixed_predictors": all_fixed_predictors,
    "preds_grouped": preds_grouped,
    "var_group_names": var_group_names,
    "subject_random_slopes": subject_random_slopes,
    "sessionID_random_slopes": sessionID_random_slopes,
}

with open(save_path, "wb") as f:
    pickle.dump(to_save, f)

print(f"💾 Saved GLMM objects to {save_path}")

#%%
# # ==========================
# # Reduced models (R² drop) using | (correlated RE) ONLY for R²
# # ==========================

# print("\n📊 Starting reduced-model (R² drop) analysis...")
# frac_R2_all = []

# r2_func = ro.r('MuMIn::r.squaredGLMM')

# def as_str_list(x):
#     if x is None:
#         return []
#     if isinstance(x, str):
#         return [x]
#     return list(x)

# for ktype in range(3):   # WT, HET, HOM
#     print(f"\n=== Computing R² for group {ktype+1} ===")

#     # Data for this group (same as used for GLMMs)
#     df_clean = prep_for_r_glmm(out_glm3[ktype])
#     with localconverter(ro.default_converter + pandas2ri.converter):
#         df_r = ro.conversion.py2rpy(df_clean)

#     # ---------- FULL MODEL FOR R² (with | instead of ||) ----------
#     fixed_full_str = "1 + " + " + ".join(all_fixed_predictors)

#     sess_slopes = as_str_list(sessionID_random_slopes)
#     sess_slopes_str = join_slopes(sess_slopes)
#     subj_slopes_str = join_slopes(subject_random_slopes)

#     # Build random part with | (correlated)
#     re_sess_full = f"(1 + {sess_slopes_str} | session_id)" if sess_slopes_str else "(1 | session_id)"
#     re_subj_full = f"(1 + {subj_slopes_str} | subject)"    if subj_slopes_str else "(1 | subject)"

#     random_full_r2 = f"{re_sess_full} + {re_subj_full}"
#     formula_full_r2_str = f"Response ~ {fixed_full_str} + {random_full_r2}"
#     print("   Full R² formula:", formula_full_r2_str)
#     formula_full_r2 = Formula(formula_full_r2_str)

#     full_model_r2 = lme4.glmer(
#         formula_full_r2,
#         data=df_r,
#         family=stats.binomial(link="logit")
#     )

#     r2 = r2_func(full_model_r2)
#     R2_full = float(r2[0])   # marginal R²
#     print(f"   Marginal R² (fixed effects): {R2_full:.4f}")

#     R2_vec = [R2_full]

#     # ---------- REDUCED MODELS (drop predictor groups) ----------
#     for g_i, preds_to_remove in enumerate(preds_grouped, start=1):
#         kept_fixed = [p for p in all_fixed_predictors if p not in preds_to_remove]
#         kept_fixed_str = "1 + " + join_slopes(kept_fixed) if kept_fixed else "1"

#         # Random slopes for session_id, minus removed predictors
#         kept_sess_slopes = [s for s in sess_slopes if s not in preds_to_remove]
#         sess_str = join_slopes(kept_sess_slopes)
#         re_sess = f"(1 + {sess_str} | session_id)" if sess_str else "(1 | session_id)"

#         # Random slopes for subject, minus removed predictors
#         kept_subj_slopes = [s for s in subject_random_slopes if s not in preds_to_remove]
#         subj_str = join_slopes(kept_subj_slopes)
#         re_subj = f"(1 + {subj_str} | subject)" if subj_str else "(1 | subject)"

#         random_red_r2 = f"{re_sess} + {re_subj}"
#         formula_red_r2_str = f"Response ~ {kept_fixed_str} + {random_red_r2}"
#         print(f"   → [{g_i}/{len(preds_grouped)}] Reduced R² formula:", formula_red_r2_str)

#         reduced_formula_r2 = Formula(formula_red_r2_str)
#         reduced_fit_r2 = lme4.glmer(
#             reduced_formula_r2,
#             data=df_r,
#             family=stats.binomial(link="logit")
#         )

#         r2_red = r2_func(reduced_fit_r2)
#         R2_reduced = float(r2_red[0])
#         delta = (R2_full - R2_reduced) / R2_full
#         print(f"      Reduced R²={R2_reduced:.4f}, ΔR² fraction={delta:.4f}")
#         R2_vec.append(R2_reduced)

#     # Store fractional R² drops for this group
#     R2_full_val = R2_vec[0]
#     frac_R2 = [(R2_full_val - r) / R2_full_val for r in R2_vec[1:]]
#     frac_R2_all.append(frac_R2)

# print("\n✅ Done fitting R² models (with |).")

# ==========================
# Reduced models (R² drop) using simple RE: (1|session_id) + (1|subject)
# ==========================

print("\n📊 Starting reduced-model (R² drop) analysis...")
frac_R2_all = []

r2_func = ro.r('MuMIn::r.squaredGLMM')

def join_slopes_safe(xs):
    # helper just to reuse your join but be safe on empty lists
    if not xs:
        return ""
    return join_slopes(xs)

for ktype in range(3):   # WT, HET, HOM
    print(f"\n=== Computing R² for group {ktype+1} ===")

    # Data for this group (same as used for GLMMs)
    df_clean = prep_for_r_glmm(out_glm3[ktype])
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df_clean)

    # ---------- FULL MODEL FOR R² (simple RE) ----------
    fixed_full_str = "1 + " + " + ".join(all_fixed_predictors)

    # random part: ONLY intercepts
    random_full_r2 = "(1 | session_id) + (1 | subject)"
    formula_full_r2_str = f"Response ~ {fixed_full_str} + {random_full_r2}"
    print("   Full R² formula:", formula_full_r2_str)
    formula_full_r2 = Formula(formula_full_r2_str)

    full_model_r2 = lme4.glmer(
        formula_full_r2,
        data=df_r,
        family=stats.binomial(link="logit")
    )

    r2 = r2_func(full_model_r2)
    R2_full = float(r2[0])   # marginal R²
    print(f"   Marginal R² (fixed effects): {R2_full:.4f}")

    R2_vec = [R2_full]

    # ---------- REDUCED MODELS (dropping predictor groups) ----------
    for g_i, preds_to_remove in enumerate(preds_grouped, start=1):
        kept_fixed = [p for p in all_fixed_predictors if p not in preds_to_remove]
        kept_fixed_core = join_slopes_safe(kept_fixed)
        kept_fixed_str = "1 + " + kept_fixed_core if kept_fixed_core else "1"

        # same simple random part for ALL reduced models
        random_red_r2 = "(1 | session_id) + (1 | subject)"
        formula_red_r2_str = f"Response ~ {kept_fixed_str} + {random_red_r2}"
        print(f"   → [{g_i}/{len(preds_grouped)}] Reduced R² formula:", formula_red_r2_str)

        reduced_formula_r2 = Formula(formula_red_r2_str)
        reduced_fit_r2 = lme4.glmer(
            reduced_formula_r2,
            data=df_r,
            family=stats.binomial(link="logit")
        )

        r2_red = r2_func(reduced_fit_r2)
        R2_reduced = float(r2_red[0])
        delta = (R2_full - R2_reduced) / R2_full
        print(f"      Reduced R²={R2_reduced:.4f}, ΔR² fraction={delta:.4f}")
        R2_vec.append(R2_reduced)

    # Store fractional R² drops for this group
    R2_full_val = R2_vec[0]
    frac_R2 = [(R2_full_val - r) / R2_full_val for r in R2_vec[1:]]
    frac_R2_all.append(frac_R2)

print("\n✅ Done fitting R² models (with simple random intercepts).")


#%%
print("📦 Extracting fixed effects and SE from GLMMs...")

fixef = ro.r['fixef']
vcov = ro.r['vcov']
diag = ro.r['diag']
sqrt = ro.r['sqrt']

model_results = []  # reset

for k in range(3):  # WT, HET, HOM
    fitres = mdle_all[k][0]

    # fixed effects
    betas_R = fixef(fitres)
    betas = np.array(betas_R)
    names = list(betas_R.names)

    # standard errors from variance-covariance matrix
    vc = vcov(fitres)
    se_R = sqrt(diag(vc))
    se = np.array(se_R)

    # t-values (beta / se)
    tvals = betas / se

    model_results.append({
        "betas": betas,
        "se": se,
        "t": tvals,
        "names": names,
        "frac_R2": frac_R2_all[k]  # from earlier computation
    })

print("✅ model_results built for WT, HET, HOM")


#%%
# Save also R² and model_results (optional)
extra_save_path = "glmm_results_cntnap2_full.pkl"

extra = {
    "mdle_all": mdle_all,
    "out_glm3": out_glm3,
    "frac_R2_all": frac_R2_all,
    "model_results": model_results,
    "all_fixed_predictors": all_fixed_predictors,
    "preds_grouped": preds_grouped,
    "var_group_names": var_group_names,
}

with open(extra_save_path, "wb") as f:
    pickle.dump(extra, f)

print(f"💾 Saved extended results to {extra_save_path}")

#%%
################################
#   MATLAB-STYLE PLOTS (with jitter)
###############################

types_to_plot = [0, 1, 2]
group_names = ["WT", "HET", "HOM"]
colors = sns.color_palette("Set1", 3)

# reorder predictors top→bottom
common_betanames = model_results[0]["names"]
n_pred = len(common_betanames)
y_positions = np.arange(n_pred)
# fixed vertical offsets (top to bottom: WT, HET, HOM)
jitter_offsets = [-0.15, 0, 0.15]   # one per group
jitter_offsets_var = [-0.015, 0, 0.015]   # one per group


# offsets = np.linspace(-0.25, 0.25, len(types_to_plot))

fig, axes = plt.subplots(1, 3, figsize=(18, 10))
plt.subplots_adjust(wspace=0.3)

#################################
# 1. Betas ± SE
#################################
ax = axes[0]
ax.axvline(0, color="k", ls="--")

for gi, res in enumerate(model_results):
    betas = res["betas"]
    se = res["se"]

    for i in range(n_pred):
        y_jitter = y_positions[i] + jitter_offsets[gi]

        ax.errorbar(
            betas[i], #+ offsets[gi],     # x
            y_jitter,                  # y (jittered)
            xerr=se[i],
            fmt='o', color=colors[gi],
            label=group_names[gi] if i == 0 else ""
        )

ax.set_yticks(y_positions)
ax.set_yticklabels(common_betanames)
ax.set_xlabel("Beta ± SE")
ax.set_title("Fixed Effects")
ax.invert_yaxis()
ax.legend()
ax.legend(frameon=False)

##############################
# 2. t-values
##############################
ax = axes[1]
ax.axvline(0, color="k", ls="--")
ax.axvline(2, color="k", ls=":")
ax.axvline(-2, color="k", ls=":")

for gi, res in enumerate(model_results):
    tvals = res["t"]

    for i in range(n_pred):
        y_jitter = y_positions[i] + jitter_offsets[gi]

        ax.plot(
            tvals[i], #+ offsets[gi],   # x
            y_jitter,                 # y
            'o', color=colors[gi]
        )

ax.set_yticks(y_positions)
ax.set_yticklabels(common_betanames)
ax.set_xlabel("t-statistic")
ax.set_title("t-values")
ax.invert_yaxis()

###################################
# 3. Fraction of R²
###################################
group_x = np.arange(len(var_group_names))
ax = axes[2]

for gi, res in enumerate(model_results):
    vals = np.array(res["frac_R2"])

    for j in range(len(vals)):
        #y_jitter = group_x[j] + np.random.uniform(-0.1, 0.1)
        y_jitter =  group_x[j] + jitter_offsets_var[gi]

        ax.plot(
            vals[j], #+ offsets[gi],   # x
            y_jitter,                # y
            'o-', color=colors[gi],
            label=group_names[gi] if j == 0 else ""
        )

ax.set_yticks(group_x)
ax.set_yticklabels(var_group_names)
ax.set_xlabel("Fraction of R²")
ax.set_title("Variance Explained by Predictor Group")
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# %%
