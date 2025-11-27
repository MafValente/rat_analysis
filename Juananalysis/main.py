#%%
import sys
sys.path.append("/Users/mafaldavalente/Documents/Mafalda_analysis")

import DataHelpers
from load_data import load_behavior_csv
import DataHelpers
from kernel_regression import kreg_for_aggregate
from plot_results import shaded_curve
import os
import numpy as np
import matplotlib.pyplot as plt

from kernel_regression import (
    kreg_for_aggregate,
    build_hierarchical_data_MT,
    hierarchical_bootstrap_RT_MT,
)



os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
TRIALS_CSV = "merged_all_subjects.csv"
META_CSV   = "sex_gen.csv"   # <-- metadata file is called

def main():
    # 1) Load trial data
    df = load_behavior_csv(TRIALS_CSV)

    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")

    # 2) Attach genotype from metadata CSV
    #    - this will add a 'genotype' column with values 'wt'/'het'/'hom'
    #    - you can also filter here by sex/genotype if you want
    df = DataHelpers.restrict_subjects(
        df_trials=df,
        meta_csv=META_CSV,
        sex=None,          # e.g. "female" if you want only females
        genotypes=None,    # e.g. ["wt","het","hom"] or None for all
        attach_meta=True,  # ⬅ important: this adds "genotype" column
    )

    # sanity check
    print("Genotype values:", df["genotype"].unique())

    print("Out min/max:", df["Out"].min(), df["Out"].max())
    print(df["Out"].value_counts().head())


    # 3) Define absILD and Easy/Hard (optional)
    df["absILD"] = df["ILD"].abs()
    # EASY_ABS_ILD = 8
    # df["Easy"]   = (df["absILD"] == EASY_ABS_ILD).astype(int)

    # 4) RT grid and kernel params
    xxi = np.linspace(0.0, 1.0, 200)
    h = 0.015 #bandwidth for kernels
    B = 1000 #bootstrapping samples

    # 5) Group by genotype: hom vs wt vs het (all ILDs for now)
    group_col    = "genotype"
    group_values = ["wt", "het", "hom"]   # change order if you like

    RTD_all, TCM_all, CDF_all = kreg_for_aggregate(
        df,
        xxi=xxi,
        h=h,
        B=B,
        group_col=group_col,
        group_values=group_values,
        easy_value=None,   # 🔹 None = all ILDs
        abl_value=None,    # 🔹 None = all ABLs
    )

    # 6) Plot tachometric curves by genotype (all ILDs)
    colors = {
        "wt":  "tab:blue",
        "het": "tab:orange",
        "hom": "tab:green",
    }

    fig, ax = plt.subplots(figsize=(7,4))
    for g in group_values:
        shaded_curve(xxi, *TCM_all[g], color=colors.get(g, "k"), label=g)

    ax.set_xlabel("RT (s)")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0, 1)
    ax.set_title("Tachometric curves by genotype (all ILDs)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    colors = {
        "wt":  "tab:blue",
        "het": "tab:orange",
        "hom": "tab:green",
    }

    fig, ax = plt.subplots(figsize=(7,4))

    for g in group_values:
        data_MT = build_hierarchical_data_MT(
            df,
            group_col=group_col,
            group_value=g,
            easy_value=None,   # all ILDs for now
            abl_value=None,    # all ABLs
        )
        MT_curve_g, RTD_g, CDF_g = hierarchical_bootstrap_RT_MT(
            data_MT, xxi, h, B
        )

        shaded_curve(xxi, *MT_curve_g, color=colors.get(g, "k"), label=g)

    ax.set_xlabel("RT (s)")
    ax.set_ylabel("MT (s)")
    ax.set_title("MT vs RT by genotype (all ILDs)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # RT distribution (pdf-like, up to scale)
    fig, ax = plt.subplots(figsize=(7,4))
    for g in group_values:
        rtd_med, rtd_up, rtd_dn = RTD_all[g]
        shaded_curve(xxi, rtd_med, rtd_up, rtd_dn,
                    color=colors.get(g, "k"), label=g)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("RT density (arb. units)")
    ax.set_title("RTD by genotype (all ILDs)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # CDF
    fig, ax = plt.subplots(figsize=(7,4))
    for g in group_values:
        cdf_med, cdf_up, cdf_dn = CDF_all[g]
        shaded_curve(xxi, cdf_med, cdf_up, cdf_dn,
                    color=colors.get(g, "k"), label=g)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1)
    ax.set_title("RT CDF by genotype (all ILDs)")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()


# %%
