#%%
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore
import DataHelpers

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

# =======================
# Load and preprocess data
# =======================

print("📂 Loading and preprocessing data...")
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file)

df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
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
hom_subj = ['ASD0008','ASD0010','ASD0013','ASD0019','ASD0020']

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
    preds_grouped.append(['ABL','abs_ILD','sign_ILD', 'ILD__ABL','trial','ILD__trial','ILD__fix_time_long'])
if use_pre:
    var_group_names.append('pre')
    preds_grouped.append(['Pre_choice','ILD__Pre_success','Pre_success__Pre_choice'])

all_fixed_predictors = sum(preds_grouped, [])
pred_to_zscore = all_fixed_predictors

print(f"✅ Using predictors: {all_fixed_predictors}")

# Full list of random slopes for subject (from MATLAB)
subject_random_slopes = [
    'ABL','abs_ILD','sign_ILD','ILD__ABL',
    'trial','ILD__trial',
    'Pre_choice','ILD__Pre_success','Pre_success__Pre_choice'
]

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

            # Turn ILD into sign and difficulty
            s_df['abs_ILD'] = s_df['ILD'].abs()
            s_df['sign_ILD'] = np.sign(s_df['ILD'])

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
            s_df['ILD__ABL'] = s_df['ILD'] * s_df['ABL']
            s_df['ILD__trial'] = s_df['ILD'] * s_df['trial']

            # fix_time_long & interaction
            s_df['fix_time_long'] = (s_df['fix_time'] > s_df['fix_time'].median()).astype(float)
            s_df['ILD__fix_time_long'] = s_df['ILD'] * s_df['fix_time_long']

            s_df['ILD__Pre_success'] = s_df['ILD'] * s_df['Pre_success']
            s_df['Pre_success__Pre_choice'] = s_df['Pre_success'] * s_df['Pre_choice']

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
        fixed = [
            'ABL','abs_ILD','sign_ILD','ILD__ABL','trial','ILD__trial','ILD__fix_time_long',
            'Pre_choice','ILD__Pre_success','Pre_success__Pre_choice'
        ]
        random_groups = ['session', 'subject']
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
    # (1|session) + (1 + ABL + ILD + ILD__ABL + ILD__Pre_success + ILD__trial +
    #                Pre_choice + Pre_success__Pre_choice + trial | subject)
    # Using || to enforce diagonal covariance (similar to MATLAB CovariancePattern='Diagonal').
    subject_slopes_str = " + ".join(subject_random_slopes)
    random_part = (
        "(1 | session) + "
        f"(1 + {subject_slopes_str} || subject)"
    )

    formula_str = f"Response ~ {fixed_part} + {random_part}"
    formula = Formula(formula_str)

    print(f"\n=== 🚀 Fitting R GLMM for group {ktype} ===")
    print(f"Formula: {formula_str}")
    print(f"Preparing dataframe for R...")

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

# ==========================
# Reduced models (R² drop)
# ==========================

print("\n📊 Starting reduced-model (R² drop) analysis...")
frac_R2_all = []

ro.r('library(MuMIn)')
r2_func = ro.r('r.squaredGLMM')

for ktype, model_list in enumerate(mdle_all):
    print(f"\n=== Computing R² for group {ktype+1} ===")
    full_model = model_list[0]

    print("→ Calculating full-model R² (MuMIn::r.squaredGLMM)...")
    r2 = r2_func(full_model)
    R2_full = float(r2[0])
    print(f"   Marginal R² (fixed effects): {R2_full:.4f}")

    R2_vec = [R2_full]

    # For each group ("current", "pre") remove that group's predictors
    for g_i, preds_to_remove in enumerate(preds_grouped, start=1):
        kept_fixed = [p for p in all_fixed_predictors if p not in preds_to_remove]

        kept_slopes = [s for s in subject_random_slopes if s not in preds_to_remove]
        slopes_str = " + ".join(kept_slopes)

        kept_fixed_str = "1 + " + " + ".join(kept_fixed)

        reduced_random_part = f"(1 | session) + (1 + {slopes_str} || subject)"
        reduced_formula_str = f"Response ~ {kept_fixed_str} + {reduced_random_part}"
        reduced_formula = Formula(reduced_formula_str)

        print(f"   → [{g_i}/{len(preds_grouped)}] Fitting reduced model without {preds_to_remove} ...")
        df_clean_reduced = prep_for_r_glmm(out_glm3[ktype])
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df_clean_reduced)

        reduced_fit = lme4.glmer(
            reduced_formula,
            data=df_r,
            family=stats.binomial(link="logit")
        )

        r2_red = r2_func(reduced_fit)
        R2_reduced = float(r2_red[0])
        delta = (R2_full - R2_reduced) / R2_full
        print(f"      Reduced R²={R2_reduced:.4f}, ΔR² fraction={delta:.4f}")
        R2_vec.append(R2_reduced)

    R2_full_val = R2_vec[0]
    frac_R2 = [(R2_full_val - r) / R2_full_val for r in R2_vec[1:]]
    frac_R2_all.append(frac_R2)

print("\n✅ Done fitting GLMMs with R and computing R² comparisons.")
print("Preparing multi-group plot...")

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
        y_jitter = y_positions[i] + np.random.uniform(-0.2, 0.2)

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
        y_jitter = group_x[j] + np.random.uniform(-0.1, 0.1)

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
ax.legend()

plt.tight_layout()
plt.show()

# %%
