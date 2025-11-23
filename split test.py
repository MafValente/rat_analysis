#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Psychometric 
import OldDatasets



path = "/Users/mafaldavalente/Documents/Mafalda_analysis/Old Data/batch_LED8_valid_and_aborts (1).csv"


OldDatasets.plot_cohort_from_table(
    df,                     # <- your full df (after your filters/masks)
    abls=abls,              # match the ABLs you’re showing on this row
    level=None,             # or your chosen level filter
    average_across_abls=False,
    axes=(axes[0], axes[1], axes[2]),
    color="k",
    zorder=6,
    show_titles=False,
    show_legend=False,
)
#%%

# (a) Specific ABLs, keep separate
OldDatasets.plot_cohort_from_table(
    path,
    abls=[20, 40, 60],
    average_across_abls=False,
    level=16,         # set None if you don't want a level filter
    cap_rt=1.0,       # optional caps for plotting only
    cap_mt=1.5,
    title_prefix="LED8 cohort",
    save_to=None      # e.g., "LED8_plots.pdf" for a multi-page PDF
)
#%%
# (b) Average across ABLs
OldDatasets.plot_cohort_from_table(
    path,
    abls="all",
    average_across_abls=True,
    level=None,
    cap_rt=1.0,
    save_to=None
)

# %%
