# kernel_regression.py
#
# Implements hierarchical (multilevel) bootstrap kernel regression for:
#   - Tachometric curves:  P(correct | RT) as a function of reaction time
#   - RT vs MT curves:     E[MT | RT]      as a function of reaction time
#   - RTD and CDF:         RT density and its cumulative
#
# Key ideas:
#   - Data are hierarchical: animal → session → stimulus (ILD) → trials
#   - We want population-level curves by group (e.g. genotype), with honest
#     uncertainty that reflects variability across animals, sessions, stimuli,
#     and trials.
#   - To do this, we use a hierarchical bootstrap:
#       For each bootstrap replicate:
#         1) Sample animals with replacement
#         2) Within each sampled animal, sample sessions with replacement
#         3) Within each sampled session, sample stimuli (ILDs) with replacement
#         4) Within each stimulus, sample trials with replacement
#       Then we pool all these resampled trials and run kernel regression
#       to estimate:
#         - P(correct | RT)
#         - E[MT | RT]
#         - RTD and CDF
#     Repeating this B times gives a distribution of curves at each RT bin,
#     from which we derive pointwise bootstrap confidence intervals (2.5–97.5%).
#
# IMPORTANT:
#   - The shaded error regions you plot are NOT SEM over animals.
#     They are hierarchical bootstrap confidence bands: they include
#     variability from animals, sessions, stimuli, and trial noise.
#

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz  # SciPy ≥1.14


# ---------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------

def epanechnikov_kernel(z):
    """
    Epanechnikov kernel K(z) = 0.75 * (1 - z^2) for |z| <= 1, else 0.
    Used for smoothing in RT space.

    z can be a scalar or array; output has same shape.
    """
    return 0.75 * (1 - z**2) * (np.abs(z) <= 1)


# ---------------------------------------------------------------------
# Data structuring: build hierarchical nested lists
# ---------------------------------------------------------------------

def build_hierarchical_data_full(df, group_col, group_value,
                                 easy_value=None, abl_value=None):
    """
    Build nested list structure for *joint* analysis of:
      - tachometric curve (RT vs correctness Out)
      - RT vs MT curve (RT vs movement time MT)

    Structure returned:
        data[animal_index][session_index][stim_index] = (RT_array, Out_array, MT_array)

    Where:
      - RT_array : 1D array of RTs for that (animal, session, ILD)
      - Out_array: 1D array of correctness (0/1) for those trials
      - MT_array : 1D array of movement times for those trials

    Inputs
    ------
    df : pandas.DataFrame
        Must contain columns:
          - 'animal'   : subject ID
          - 'session'  : session/day index
          - 'ABL'      : sound level (optional filter)
          - 'ILD'      : stimulus level (we aggregate per ILD)
          - 'RT'       : reaction time (float)
          - 'Out'      : correctness (0 or 1; NaNs already removed in mask)
          - 'MT'       : movement time (float)
          - group_col  : e.g. 'genotype'
          - 'Easy'     : optional 0/1 flag for easy/hard trials

    group_col : str
        Column name used to define groups (e.g. 'genotype').
    group_value : any
        Specific group value to include (e.g. 'wt', 'het', 'hom').
    easy_value : {0, 1, None}
        If None, use all ILDs.
        If 1, restrict to "easy" ILDs (df['Easy'] == 1).
        If 0, restrict to "hard" ILDs (df['Easy'] == 0).
    abl_value : {float, int, None}
        If not None, restrict to a single ABL.

    Returns
    -------
    animals_data : list
        Nested list as described above. Possibly empty if no data after filters.
    """
    # Basic inclusion mask: correct group, finite RT/Out/MT
    mask = (
        df[group_col].notna() &
        (df[group_col] == group_value) &
        df["RT"].notna() &
        df["Out"].notna() &
        df["MT"].notna()
    )

    # Optional filters
    if easy_value is not None:
        mask &= (df["Easy"] == easy_value)

    if abl_value is not None:
        mask &= (df["ABL"] == abl_value)

    sub = df.loc[mask].copy()
    if sub.empty:
        return []

    animals = sub["animal"].unique()
    animals_data = []

    # Build data[animal][session][stimulus]
    for animal in animals:
        sub_a = sub[sub["animal"] == animal]
        sessions = sub_a["session"].unique()
        sessions_data = []

        for session in sessions:
            sub_s = sub_a[sub_a["session"] == session]
            ilds = sub_s["ILD"].unique()
            stimuli_data = []

            for ild in ilds:
                sub_stim = sub_s[sub_s["ILD"] == ild]
                RT_arr  = sub_stim["RT"].to_numpy()
                Out_arr = sub_stim["Out"].to_numpy()
                MT_arr  = sub_stim["MT"].to_numpy()
                if len(RT_arr) == 0:
                    continue
                stimuli_data.append((RT_arr, Out_arr, MT_arr))

            if len(stimuli_data) > 0:
                sessions_data.append(stimuli_data)

        if len(sessions_data) > 0:
            animals_data.append(sessions_data)

    # Shape: [animal][session][stimulus] = (RTs, Outs, MTs)
    return animals_data


# ---------------------------------------------------------------------
# Unified hierarchical bootstrap: TCM, MT vs RT, RTD, CDF
# ---------------------------------------------------------------------

def hierarchical_bootstrap_joint(data_nested, xxi, h, B):
    """
    Hierarchical bootstrap that simultaneously estimates, for each RT grid x:

      - TCM(x)    = P(correct | RT = x)
      - MT(x)     = E[MT | RT = x]
      - RTD(x)    = RT density (shape, up to scale)
      - CDF(x)    = cumulative RT distribution

    using the same resampled trials and the same kernel weights.

    Hierarchical resampling:
      - animals with replacement
      - sessions within animals with replacement
      - stimuli (ILDs) within sessions with replacement
      - trials within each stimulus with replacement

    Inputs
    ------
    data_nested : list
        [animal][session][stimulus] = (RT_array, Out_array, MT_array)
    xxi : 1D array
        RT grid on which to evaluate the curves.
    h : float
        Kernel bandwidth (in seconds).
    B : int
        Number of bootstrap samples.

    Returns
    -------
    RTD   : (med, up, dn)
        Summary of RT density (kernel denominator) across bootstraps.
    TCM   : (med, up, dn)
        Summary of tachometric curve P(correct | RT).
    CDF   : (med, up, dn)
        Summary of cumulative RT distribution (integral of RTD).
    MTcur : (med, up, dn)
        Summary of E[MT | RT] curve.
    """
    n_grid = len(xxi)
    ZTCM = np.zeros((B, n_grid))  # P(correct | RT) per bootstrap
    ZMT  = np.zeros((B, n_grid))  # E[MT | RT] per bootstrap
    Zden = np.zeros((B, n_grid))  # RT density per bootstrap

    K = epanechnikov_kernel
    n_animals = len(data_nested)

    if n_animals == 0:
        nan = np.full(n_grid, np.nan)
        nan_triplet = (nan, nan, nan)
        return nan_triplet, nan_triplet, nan_triplet, nan_triplet

    for b in range(B):
        RTs_all  = []
        Outs_all = []
        MTs_all  = []

        # 1) Sample animals with replacement
        animal_indices = np.random.randint(0, n_animals, size=n_animals)

        for ai in animal_indices:
            sessions = data_nested[ai]
            if not sessions:
                continue
            n_sess = len(sessions)

            # 2) Sample sessions with replacement
            sess_indices = np.random.randint(0, n_sess, size=n_sess)

            for si in sess_indices:
                stimuli = sessions[si]
                if not stimuli:
                    continue
                n_stim = len(stimuli)

                # 3) Sample stimuli with replacement
                stim_indices = np.random.randint(0, n_stim, size=n_stim)

                for gi in stim_indices:
                    RT_arr, Out_arr, MT_arr = stimuli[gi]
                    n_trials = len(RT_arr)
                    if n_trials == 0:
                        continue

                    # 4) Sample trials with replacement within each stimulus
                    trial_idx = np.random.randint(0, n_trials, size=n_trials)
                    RTs_all.append(RT_arr[trial_idx])
                    Outs_all.append(Out_arr[trial_idx])
                    MTs_all.append(MT_arr[trial_idx])

        if len(RTs_all) == 0:
            # No data in this bootstrap replicate (very unlikely if data exist)
            ZTCM[b, :] = np.nan
            ZMT[b, :]  = np.nan
            Zden[b, :] = np.nan
            continue

        # Flatten resampled data
        x     = np.concatenate(RTs_all)
        y_out = np.concatenate(Outs_all)  # 0/1 correctness
        y_mt  = np.concatenate(MTs_all)   # MT (continuous)

        # Kernel smoothing: evaluate kernels for each RT grid point
        # z shape: (n_trials, n_grid)
        z  = (xxi[None, :] - x[:, None]) / h
        Kz = K(z)

        # Weighted sums:
        #   den      ~ Σ K( (x - RT_i)/h ) / (n*h)
        #   num_out  ~ Σ K( (x - RT_i)/h ) * Out_i / (n*h)
        #   num_mt   ~ Σ K( (x - RT_i)/h ) * MT_i  / (n*h)
        #
        # TCM(x)   = num_out / den = P(correct | RT = x)
        # MTcur(x) = num_mt  / den = E[MT | RT = x]
        n       = len(x)
        den     = Kz.sum(axis=0) / (n * h) + 1e-12
        num_out = (Kz * y_out[:, None]).sum(axis=0) / (n * h)
        num_mt  = (Kz * y_mt[:, None]).sum(axis=0)  / (n * h)

        with np.errstate(invalid="ignore", divide="ignore"):
            tcm      = np.full_like(den, np.nan, dtype=float)
            mt_curve = np.full_like(den, np.nan, dtype=float)
            valid    = den > 0
            tcm[valid]      = num_out[valid] / den[valid]  # P(correct | RT)
            mt_curve[valid] = num_mt[valid]  / den[valid]  # E[MT | RT]

        ZTCM[b, :] = tcm
        ZMT[b, :]  = mt_curve
        Zden[b, :] = den

    # CDF from density: integrate along xxi
    ZCDF = cumtrapz(Zden, xxi, axis=1, initial=0)

    def summarize(arr):
        """
        Given an array of shape (B, n_grid), compute:
          - median across bootstraps
          - upper and lower error (97.5% and 2.5% quantiles around median).
        """
        med = np.nanmedian(arr, axis=0)
        q_hi = np.nanquantile(arr, 0.975, axis=0)
        q_lo = np.nanquantile(arr, 0.025, axis=0)
        up = q_hi - med
        dn = med - q_lo
        return med, up, dn

    # Hierarchical bootstrap summaries for each curve
    TCM   = summarize(ZTCM)  # P(correct | RT)
    MTcur = summarize(ZMT)   # E[MT | RT]
    RTD   = summarize(Zden)  # RT density
    CDF   = summarize(ZCDF)  # cumulative RT

    return RTD, TCM, CDF, MTcur


# ---------------------------------------------------------------------
# High-level wrapper: run joint bootstrap by group (e.g. genotype)
# ---------------------------------------------------------------------

def kreg_for_aggregate(df, xxi, h, B,
                       group_col, group_values,
                       easy_value=None, abl_value=None):
    """
    High-level wrapper (keeps the "kreg_for_aggregate" name) that performs
    the *joint* hierarchical bootstrap analysis for each group:

      For each group g in group_values:
        - builds hierarchical data with RT, Out, MT
        - runs hierarchical_bootstrap_joint
        - returns RTD[g], TCM[g], CDF[g], MT[g]

    Example
    -------
        RTD, TCM, CDF, MT = kreg_for_aggregate(
            df,
            xxi=xxi,
            h=h,
            B=1000,
            group_col="genotype",
            group_values=["wt", "het", "hom"],
            easy_value=None,
            abl_value=None,
        )

    Inputs
    ------
    df : pandas.DataFrame
        Trial-level data with required columns:
          - animal, session, ILD, ABL
          - RT, Out, MT
          - group_col (e.g. genotype)
          - optionally Easy
    xxi : 1D array
        RT grid for evaluation.
    h : float
        Kernel bandwidth.
    B : int
        Number of bootstrap replicates.
    group_col : str
        Column used to define groups (e.g. "genotype").
    group_values : list
        Values in group_col for which to compute curves (e.g. ["wt","het","hom"]).
    easy_value : {0, 1, None}
        If None, all ILDs.
        If 1, only Easy ILDs.
        If 0, only Hard ILDs.
    abl_value : {float, int, None}
        If provided, curves are restricted to that ABL.

    Returns
    -------
    RTD, TCM, CDF, MT : dict
        Each dict maps group_value → (med, up, dn), i.e. the curve and its
        hierarchical bootstrap confidence band for that group.
          - RTD[g] : RT density vs RT
          - TCM[g] : P(correct | RT)
          - CDF[g] : cumulative RT
          - MT[g]  : E[MT | RT]
    """
    RTD = {}
    TCM = {}
    CDF = {}
    MT  = {}

    for g in group_values:
        data_nested = build_hierarchical_data_full(
            df,
            group_col=group_col,
            group_value=g,
            easy_value=easy_value,
            abl_value=abl_value,
        )

        RTD_g, TCM_g, CDF_g, MT_g = hierarchical_bootstrap_joint(
            data_nested, xxi, h, B
        )

        RTD[g] = RTD_g
        TCM[g] = TCM_g
        CDF[g] = CDF_g
        MT[g]  = MT_g

    return RTD, TCM, CDF, MT
