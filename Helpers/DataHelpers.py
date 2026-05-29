import numpy as np
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import os
import Psychometric
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from collections.abc import Mapping


"""
.########..########..########.########.....########.....###....########....###...
.##.....##.##.....##.##.......##.....##....##.....##...##.##......##......##.##..
.##.....##.##.....##.##.......##.....##....##.....##..##...##.....##.....##...##.
.########..########..######...########.....##.....##.##.....##....##....##.....##
.##........##...##...##.......##...........##.....##.#########....##....#########
.##........##....##..##.......##...........##.....##.##.....##....##....##.....##
.##........##.....##.########.##...........########..##.....##....##....##.....##
"""
def prepare_data(
    df,
    #training_level_filter=16,
    session_col="session",   # <-- change to your real session column name
    trial_col="trial",       # <-- or trial_index if different
):
    """
    Full data preparation pipeline:
    1. Fix ABL values according to Mafalda's rules
    2. Filter to a specific training level
    3. Add a 'trial_is_repeat' column:
       True  = current trial is itself a repetition trial
       False = otherwise
    """

    df = df.copy()

    # ----------------------------
    # 1. ---- ABL FIXES ---------
    # ----------------------------
    mask1 = df["training_level"] < 7
    df.loc[mask1, "ABL"] = pd.to_numeric(df.loc[mask1, "ABL"], errors="coerce") * 2

    mask2 = df["ABL"] == 59
    df.loc[mask2, "ABL"] = 60

    mask3 = df["ABL"] == 58
    df.loc[mask3, "ABL"] = 60

    mask4 = (df["training_level"] == 16) & (df["ABL"] == 25)
    df.loc[mask4, "ABL"] = 50

    # ----------------------------
    # 2. ---- FILTER BY LEVEL ----
    # ----------------------------
    # if training_level_filter is not None:
    #     df = df[df["training_level"] == training_level_filter].copy()

    # ----------------------------
    # 3. ---- MARK REPEATED TRIALS ----
    #
    # repeated_trial == True flags the *failed* trial that SHOULD be repeated.
    # We want to mark the NEXT trial (within the same session)
    # as "trial_is_repeat" *if and only if* the previous trial
    #   - had repeated_trial == True
    #   - abort_type NOT in {"Fixation", "CNP"}
    # ----------------------------

    df = df.sort_values([session_col, trial_col]).copy()
    df["trial_is_repeat"] = False

    non_repeat_abort_types = {"Fixation", "CNP"}

    for session_id, df_sess in df.groupby(session_col):
        idx = df_sess.index.to_list()

        for i in range(1, len(idx)):
            prev_idx = idx[i - 1]
            this_idx = idx[i]

            prev_row = df.loc[prev_idx]

            prev_triggers_repeat = (
                bool(prev_row["repeated_trial"]) and
                str(prev_row["abort_type"]) not in non_repeat_abort_types
            )

            if prev_triggers_repeat:
                df.loc[this_idx, "trial_is_repeat"] = True

    return df




#prep data for the short durations

DUR_RULES = [(12, 16, 15), (17, 21, 60), (22, 26, 120)]
RT_MS = 6000

def add_stim_dur(df, sound_col="sound_index", session_col="session_type",
                 out_col="stim_dur", type1_value=RT_MS):  # set to pd.NA for NaN behavior
    df = df.copy()

    # default for everything (type1 -> RT_MS, or pd.NA if you prefer)
    df[out_col] = pd.Series([type1_value] * len(df), index=df.index, dtype="Int64")

    # only compute mapping for session_type == 2 (and only if column exists)
    if session_col in df.columns:
        mask = pd.to_numeric(df[session_col], errors="coerce") == 2
    else:
        mask = pd.Series(False, index=df.index)

    if mask.any():
        s = pd.to_numeric(df.loc[mask, sound_col], errors="coerce")
        conds = [s.between(lo, hi) for lo, hi, _ in DUR_RULES]
        choices = [dur for _, _, dur in DUR_RULES]
        df.loc[mask, out_col] = pd.Series(np.select(conds, choices, default=RT_MS), index=df.loc[mask].index).astype("Int64")

    # optional label column
    df["stim_dur_label"] = np.where(df[out_col] == RT_MS, "RT", df[out_col].astype(str) + "ms")
    return df

"""
..######..##.....##..#######..########..########....########..##.....##.########.....###....########.####..#######..##....##..######.
.##....##.##.....##.##.....##.##.....##....##.......##.....##.##.....##.##.....##...##.##......##.....##..##.....##.###...##.##....##
.##.......##.....##.##.....##.##.....##....##.......##.....##.##.....##.##.....##..##...##.....##.....##..##.....##.####..##.##......
..######..#########.##.....##.########.....##.......##.....##.##.....##.########..##.....##....##.....##..##.....##.##.##.##..######.
.......##.##.....##.##.....##.##...##......##.......##.....##.##.....##.##...##...#########....##.....##..##.....##.##..####.......##
.##....##.##.....##.##.....##.##....##.....##.......##.....##.##.....##.##....##..##.....##....##.....##..##.....##.##...###.##....##
..######..##.....##..#######..##.....##....##.......########...#######..##.....##.##.....##....##....####..#######..##....##..######.
"""



def _empty_to_na(s: pd.Series) -> pd.Series:
    """Treat '', ' ', 'nan', 'none' as missing."""
    if s is None:
        return s
    out = s.copy()
    # strings that should be NA
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    out = out.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "none": pd.NA})
    return out

def _extract_first_number_as_numeric(s: pd.Series) -> pd.Series:
    """Extract first numeric token from strings; return Float64 series."""
    # If already numeric, keep it; if string, extract digits
    as_num = pd.to_numeric(s, errors="coerce")
    if as_num.notna().any():
        # still also try extracting for the non-numeric rows
        pass
    extracted = s.astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    extracted_num = pd.to_numeric(extracted, errors="coerce")
    return as_num.fillna(extracted_num).astype("Float64")

def normalize_short_sound_fields(
    df: pd.DataFrame,
    session_col: str = "session_type",
    stimdur_col: str = "stim_dur",
    shortdur_col: str = "short_duration",
    isshort_col: str = "is_short_sound",
    long_value: float = 6000,
) -> pd.DataFrame:
    """
    1) Ensure `is_short_sound` exists for all sessions:
       - If missing/empty: False/0.0 for session_type==1
       - If missing/empty: False/0.0 for session_type==2 AND stim_dur==6000
       - Otherwise (session_type==2 AND stim_dur!=6000): True/1.0

    2) Harmonize `short_duration`:
       - If short_duration empty:
            * 0 if stim_dur == 6000
            * else numeric portion of stim_dur (string -> extracted number; numeric -> itself)
       - Result stored as numeric (Float64) in `short_duration`.
    """
    df = df.copy()

    # --- session_type numeric (tolerant) ---
    if session_col not in df.columns:
        raise KeyError(f"Missing required column: {session_col}")
    sess = pd.to_numeric(df[session_col], errors="coerce")

    # --- stim_dur numeric (tolerant: numeric or string with number) ---
    if stimdur_col not in df.columns:
        df[stimdur_col] = pd.NA
    stim_raw = _empty_to_na(df[stimdur_col])
    stim_num = _extract_first_number_as_numeric(stim_raw)

    # --- short_duration numeric ---
    if shortdur_col not in df.columns:
        df[shortdur_col] = pd.NA
    sd_raw = _empty_to_na(df[shortdur_col])
    sd_num = _extract_first_number_as_numeric(sd_raw)

    sd_missing = sd_num.isna()
    # fill empty short_duration using stim_dur rule
    is_long = stim_num.eq(long_value)
    sd_num = sd_num.copy()
    sd_num.loc[sd_missing & is_long] = 0
    # "numerical portion of the string otherwise" -> use stim_num
    sd_num.loc[sd_missing & (~is_long)] = stim_num.loc[sd_missing & (~is_long)]
    df[shortdur_col] = sd_num.astype("Float64")

    # --- is_short_sound ---
    if isshort_col not in df.columns:
        df[isshort_col] = pd.NA

    iss_raw = _empty_to_na(df[isshort_col])

    # Coerce various encodings to numeric if present
    iss_norm = iss_raw.astype("string").str.strip()
    iss_norm = iss_norm.replace({
        "TRUE": "1", "True": "1", "true": "1",
        "FALSE": "0", "False": "0", "false": "0",
    })
    iss_num = pd.to_numeric(iss_norm, errors="coerce").astype("Float64")

    iss_missing = iss_num.isna()

    # Infer for missing values:
    # session_type 1 => False
    # session_type 2 => True iff stim_dur != 6000 (i.e., short)
    inferred = ((sess.eq(2)) & (~stim_num.eq(long_value))).astype("Float64")

    # Your explicit conditions are a subset of this inference, but this also fills the "otherwise" case.
    iss_num.loc[iss_missing] = inferred.loc[iss_missing]

    df[isshort_col] = iss_num.astype("Float64")

    return df

"""
..######..####..######...##.....##..#######..####.########.
.##....##..##..##....##..###...###.##.....##..##..##.....##
.##........##..##........####.####.##.....##..##..##.....##
..######...##..##...####.##.###.##.##.....##..##..##.....##
.......##..##..##....##..##.....##.##.....##..##..##.....##
.##....##..##..##....##..##.....##.##.....##..##..##.....##
..######..####..######...##.....##..#######..####.########.
"""

def sigmoid(x, upper, lower, x0, k):
    """Sigmoid function with explicit upper and lower asymptotes."""
    import numpy as np
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))

"""
..######..########..######...######..####..#######..##....##....########.....###....########.########
.##....##.##.......##....##.##....##..##..##.....##.###...##....##.....##...##.##......##....##......
.##.......##.......##.......##........##..##.....##.####..##....##.....##..##...##.....##....##......
..######..######....######...######...##..##.....##.##.##.##....##.....##.##.....##....##....######..
.......##.##.............##.......##..##..##.....##.##..####....##.....##.#########....##....##......
.##....##.##.......##....##.##....##..##..##.....##.##...###....##.....##.##.....##....##....##......
..######..########..######...######..####..#######..##....##....########..##.....##....##....########
"""

def _infer_file_date(filepath):
    """
    Extract date from filename patterns like: ...out_YYMMDD....csv
    Falls back to the first 6-digit group if out_YYMMDD is missing,
    and finally to file mtime.
    """
    name = os.path.basename(filepath)

    # Strict: out_YYMMDD
    m = re.search(r'out_(\d{6})', name)
    if not m:
        # Relaxed: any YYMMDD in name
        m = re.search(r'(\d{6})', name)

    if m:
        y, mo, d = m.group(1)[:2], m.group(1)[2:4], m.group(1)[4:6]
        try:
            return pd.Timestamp(year=2000 + int(y), month=int(mo), day=int(d))
        except ValueError:
            pass

    # Last resort: modification time
    return pd.Timestamp(os.path.getmtime(filepath), unit="s").normalize()



"""
.########..########.##.....##..#######..##.....##.########....########.########..####....###....##........######.
.##.....##.##.......###...###.##.....##.##.....##.##.............##....##.....##..##....##.##...##.......##....##
.##.....##.##.......####.####.##.....##.##.....##.##.............##....##.....##..##...##...##..##.......##......
.########..######...##.###.##.##.....##.##.....##.######.........##....########...##..##.....##.##........######.
.##...##...##.......##.....##.##.....##..##...##..##.............##....##...##....##..#########.##.............##
.##....##..##.......##.....##.##.....##...##.##...##.............##....##....##...##..##.....##.##.......##....##
.##.....##.########.##.....##..#######.....###....########.......##....##.....##.####.##.....##.########..######.
"""
# to remove problematic trials - hardware/user error

def mark_repeated_from(csv_path, start_trial, trial_col=None, output_path=None, make_backup=True):
    """
    Set repeated_trial = TRUE for all rows with trial >= start_trial in one session CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to the session CSV to edit.
    start_trial : int
        First trial number to mark as repeated.
    trial_col : str or None
        Name of the trial-number column. If None, auto-detects common names.
    output_path : str or Path or None
        Where to save the edited CSV. If None, overwrites the input file.
    make_backup : bool
        If overwriting, write a .bak next to the original first.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # 1) pick trial column
    if trial_col is None:
        candidates = ["trial", "trial_index", "trial_number", "trial_num", "trialID"]
        trial_col = next((c for c in candidates if c in df.columns), None)
        if trial_col is None:
            raise ValueError(f"Couldn't find a trial column. Looked for: {candidates}. "
                             "Pass trial_col='your_column'.")

    # 2) ensure numeric trials
    trial_num = pd.to_numeric(df[trial_col], errors="coerce")

    # 3) ensure repeated_trial exists and is nullable boolean
    if "repeated_trial" not in df.columns:
        df["repeated_trial"] = pd.Series(pd.array([pd.NA]*len(df), dtype="boolean"))
    else:
        # coerce common encodings to nullable boolean (True/False/<NA>)
        if not pd.api.types.is_bool_dtype(df["repeated_trial"]):
            mapped = (df["repeated_trial"].astype("string")
                      .str.strip().str.upper()
                      .map({"TRUE": True, "FALSE": False, "1": True, "0": False,
                            "YES": True, "NO": False}))
            df["repeated_trial"] = pd.Series(pd.array(mapped, dtype="boolean"))

    # 4) apply change
    mask = trial_num >= int(start_trial)
    changed = int(mask.sum())
    if changed:
        df.loc[mask, "repeated_trial"] = True
        df["repeated_trial"] = df["repeated_trial"].astype("boolean")

    # 5) save (with optional backup)
    out_path = Path(output_path) if output_path else csv_path
    if output_path is None and make_backup:
        csv_path.with_suffix(csv_path.suffix + ".bak").write_text(csv_path.read_text())
    df.to_csv(out_path, index=False)

    print(f"Marked {changed} rows (trial >= {start_trial}) in '{out_path.name}'.")


"""
..######..########.##.......########..######..########.....######..########..######...######..####..#######..##....##..######.
.##....##.##.......##.......##.......##....##....##.......##....##.##.......##....##.##....##..##..##.....##.###...##.##....##
.##.......##.......##.......##.......##..........##.......##.......##.......##.......##........##..##.....##.####..##.##......
..######..######...##.......######...##..........##........######..######....######...######...##..##.....##.##.##.##..######.
.......##.##.......##.......##.......##..........##.............##.##.............##.......##..##..##.....##.##..####.......##
.##....##.##.......##.......##.......##....##....##.......##....##.##.......##....##.##....##..##..##.....##.##...###.##....##
..######..########.########.########..######.....##........######..########..######...######..####..#######..##....##..######.
"""
# to select a certain number of sessions

def get_last_n_sessions(df, n, session_col="session"):
    unique_sessions = sorted(df[session_col].unique())
    last_sessions = unique_sessions[-n:]
    return df[df[session_col].isin(last_sessions)], last_sessions

"""
.########..####....###.....######.....########..#######..########.....########.########..########...#######..########...######.
.##.....##..##....##.##...##....##....##.......##.....##.##.....##....##.......##.....##.##.....##.##.....##.##.....##.##....##
.##.....##..##...##...##..##..........##.......##.....##.##.....##....##.......##.....##.##.....##.##.....##.##.....##.##......
.########...##..##.....##..######.....######...##.....##.########.....######...########..########..##.....##.########...######.
.##.....##..##..#########.......##....##.......##.....##.##...##......##.......##...##...##...##...##.....##.##...##.........##
.##.....##..##..##.....##.##....##....##.......##.....##.##....##.....##.......##....##..##....##..##.....##.##....##..##....##
.########..####.##.....##..######.....##........#######..##.....##....########.##.....##.##.....##..#######..##.....##..######.
"""
# computed bias for errors

def compute_bias(df):
    valid_group = df[df["success"] != 0]
    if valid_group.empty: 
        return np.nan
    n_neg = (valid_group["ILD"] < 0).sum()
    n_pos = (valid_group["ILD"] > 0).sum()
    if n_neg == 0 or n_pos == 0:
        return np.nan
    resp = pd.to_numeric(valid_group["response_poke"], errors="coerce")
    if resp.dropna().isin([2, 3]).any():
        choice = pd.Series(np.where(resp == 3, 1, np.where(resp == 2, -1, np.nan)), index=valid_group.index)
    elif resp.dropna().isin([-1, 1]).any():
        choice = resp
    else:
        choice = resp
    wrong_right = ((choice == 1) & (valid_group["ILD"] < 0)).sum()
    wrong_left = ((choice == -1) & (valid_group["ILD"] > 0)).sum()
    frac_wrong_right = wrong_right / n_neg if n_neg > 0 else 0
    frac_wrong_left = wrong_left / n_pos if n_pos > 0 else 0
    
    return frac_wrong_right - frac_wrong_left


"""
.########.########..####....###....##........######.....####.##....##.......###........######..########..######...######..####..#######..##....##
....##....##.....##..##....##.##...##.......##....##.....##..###...##......##.##......##....##.##.......##....##.##....##..##..##.....##.###...##
....##....##.....##..##...##...##..##.......##...........##..####..##.....##...##.....##.......##.......##.......##........##..##.....##.####..##
....##....########...##..##.....##.##........######......##..##.##.##....##.....##.....######..######....######...######...##..##.....##.##.##.##
....##....##...##....##..#########.##.............##.....##..##..####....#########..........##.##.............##.......##..##..##.....##.##..####
....##....##....##...##..##.....##.##.......##....##.....##..##...###....##.....##....##....##.##.......##....##.##....##..##..##.....##.##...###
....##....##.....##.####.##.....##.########..######.....####.##....##....##.....##.....######..########..######...######..####..#######..##....##
"""

#count trials in a session

def count_trials(df, mask, name):
    return (
        df[mask]
        .groupby("session", observed=False)
        .size()
        .reset_index(name="trial_count")
        .assign(type=name)
    )


"""
.########..####....###.....######.....####.##....##.....######...#######..##....##.########.########.##.....##.########
.##.....##..##....##.##...##....##.....##..###...##....##....##.##.....##.###...##....##....##........##...##.....##...
.##.....##..##...##...##..##...........##..####..##....##.......##.....##.####..##....##....##.........##.##......##...
.########...##..##.....##..######......##..##.##.##....##.......##.....##.##.##.##....##....######......###.......##...
.##.....##..##..#########.......##.....##..##..####....##.......##.....##.##..####....##....##.........##.##......##...
.##.....##..##..##.....##.##....##.....##..##...###....##....##.##.....##.##...###....##....##........##...##.....##...
.########..####.##.....##..######.....####.##....##.....######...#######..##....##....##....########.##.....##....##...
"""
#compute the cutoff for the bias according to previous sessions

def _mad_cutoff(abs_bias, k=3.0):
    """
    Robust cutoff using Median Absolute Deviation (MAD).
    With bias ideally 0, use median(abs_bias) as center of spread.
    Cutoff = k * 1.4826 * MAD  (1.4826 makes MAD ~ std if normal)
    """
    median_abs = np.median(abs_bias)
    mad = np.median(np.abs(abs_bias - median_abs))
    robust_scale = 1.4826 * mad
    # If variation is basically zero, return 0 to avoid excluding everything for tiny noise
    if np.isclose(robust_scale, 0):
        return 0.0
    return k * robust_scale

# assumes you already have:
# - compute_bias(df)           # from your message
# - _mad_cutoff(abs_bias, k)   # from your message

"""
..######..########..######...######..####..#######..##....##.....######..##.....##.##.....##.##.....##....###....########..##....##
.##....##.##.......##....##.##....##..##..##.....##.###...##....##....##.##.....##.###...###.###...###...##.##...##.....##..##..##.
.##.......##.......##.......##........##..##.....##.####..##....##.......##.....##.####.####.####.####..##...##..##.....##...####..
..######..######....######...######...##..##.....##.##.##.##.....######..##.....##.##.###.##.##.###.##.##.....##.########.....##...
.......##.##.............##.......##..##..##.....##.##..####..........##.##.....##.##.....##.##.....##.#########.##...##......##...
.##....##.##.......##....##.##....##..##..##.....##.##...###....##....##.##.....##.##.....##.##.....##.##.....##.##....##.....##...
..######..########..######...######..####..#######..##....##.....######...#######..##.....##.##.....##.##.....##.##.....##....##...
"""

def build_session_summary(
    df,
    trials_use="total",           # "total" or "valid" (valid := success != 0)
    valid_mask=lambda d: d["success"] != 0,
):
    """
    From trial-level df -> per-session summary.
    Adds: n_trials_total, n_trials_valid, n_completed, n_correct, performance, bias
    performance = n_correct / n_completed (with n_completed = success != 0)
    """
    # total trials per (subject, session)
    counts_total = (
        df.groupby(["animal", "session"], observed=False)
          .size().rename("n_trials_total")
          .reset_index()
    )

    # completed (valid) trials
    valid_df = df[valid_mask(df)]
    counts_valid = (
        valid_df.groupby(["animal", "session"], observed=False)
                .size().rename("n_trials_valid")
                .reset_index()
    )

    # correct trials (assumes success == 1 means correct)
    n_correct = (
        valid_df[valid_df["success"] == 1]
        .groupby(["animal", "session"], observed=False)
        .size().rename("n_correct")
        .reset_index()
    )

    # bias per session using your helper (computed over valid trials inside compute_bias)
    bias_per_session = (
        df.groupby(["animal", "session"], observed=False)
          .apply(compute_bias, include_groups=False)
          .rename("bias")
          .reset_index()
    )

    # merge
    ses = (counts_total
           .merge(counts_valid, on=["animal","session"], how="left")
           .merge(n_correct, on=["animal","session"], how="left")
           .merge(bias_per_session, on=["animal","session"], how="left")
           .fillna({"n_trials_valid": 0, "n_correct": 0})
           .sort_values(["animal","session"])
           .reset_index(drop=True))

    ses["n_completed"] = ses["n_trials_valid"].astype(int)
    ses["n_correct"]   = ses["n_correct"].astype(int)
    # safe performance (avoid div/0)
    ses["performance"] = np.where(ses["n_completed"] > 0,
                                  ses["n_correct"] / ses["n_completed"],
                                  np.nan)

    # choose which trial count to compare to min_trials
    ses["n_trials"] = ses["n_trials_valid"] if trials_use == "valid" else ses["n_trials_total"]

    # absolute bias for cutoff logic
    ses["abs_bias"] = ses["bias"].abs().astype(float)

    return ses[[
        "animal","session",
        "n_trials_total","n_trials_valid","n_trials","n_completed","n_correct",
        "performance","bias","abs_bias"
    ]]

"""
.########.####.##.......########.########.########......######..########..######...######..####..#######..##....##..######.
.##........##..##..........##....##.......##.....##....##....##.##.......##....##.##....##..##..##.....##.###...##.##....##
.##........##..##..........##....##.......##.....##....##.......##.......##.......##........##..##.....##.####..##.##......
.######....##..##..........##....######...########......######..######....######...######...##..##.....##.##.##.##..######.
.##........##..##..........##....##.......##...##............##.##.............##.......##..##..##.....##.##..####.......##
.##........##..##..........##....##.......##....##.....##....##.##.......##....##.##....##..##..##.....##.##...###.##....##
.##.......####.########....##....########.##.....##.....######..########..######...######..####..#######..##....##..######.
"""

def filter_sessions_with_history_bias_and_perf(
    df_trials,
    min_trials=100,
    trials_use="total",     # "total" or "valid"
    history_n=10,           # look-back window size (per subject) for bias cutoff
    min_prev=5,             # minimum previous sessions to enforce bias cutoff
    k=3.0,                  # MAD multiplier for bias cutoff
    min_perf=0.6,           # performance threshold: correct/completed must be >= this
    require_perf_min_completed=20,  # only enforce perf if we have at least this many completed trials in session
):
    """
    Adds a performance filter in addition to trials and history-based bias.

    Returns:
        sessions_kept     : DataFrame of kept sessions with diagnostics
        report_lines      : list[str] human-readable reasons for exclusions
        excluded_table    : DataFrame of excluded sessions + reasons
        df_trials_clean   : trial-level df with excluded sessions removed
    """
    # 1) Build per-session summary
    ses = build_session_summary(df_trials, trials_use=trials_use).copy()
    ses = ses.sort_values(["animal","session"]).reset_index(drop=True)

    # 2) History-based bias cutoff (previous sessions only, per subject)
    ses["bias_cutoff"]  = np.inf
    ses["enforce_bias"] = False
    def _cutoffs(group):
        g = group.sort_values("session").copy()
        absb = g["abs_bias"].to_numpy()
        cut  = np.full(len(g), np.inf, dtype=float)
        enf  = np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            start = max(0, i - history_n)
            prev  = absb[start:i]
            if len(prev) >= min_prev:
                cut[i] = _mad_cutoff(prev, k=k)
                enf[i] = True
        g["bias_cutoff"]  = cut
        g["enforce_bias"] = enf
        return g
    ses = ses.groupby("animal", group_keys=True).apply(_cutoffs, include_groups=False).reset_index(level=0).reset_index(drop=True)
    

    # 3) Failure conditions
    fail_trials = ses["n_trials"] < min_trials
    fail_bias   = ses["enforce_bias"] & (ses["abs_bias"] > ses["bias_cutoff"])

    # Performance: only enforce if session has enough completed trials
    enough_completed = ses["n_completed"] >= require_perf_min_completed
    fail_perf = enough_completed & (ses["performance"] < min_perf)

    ses["exclude"] = fail_trials | fail_bias | fail_perf

    # 4) Human-readable report
    report = []
    reasons_col = []
    for r in ses[ses["exclude"]].itertuples(index=False):
        reasons = []
        if getattr(r, "n_trials") < min_trials:
            reasons.append(f"too few {trials_use} trials ({int(getattr(r,'n_trials'))}) < {min_trials}")
        if getattr(r, "enforce_bias") and getattr(r, "abs_bias") > getattr(r, "bias_cutoff"):
            reasons.append(f"extreme bias |{getattr(r,'bias'):.3f}| > cutoff {getattr(r,'bias_cutoff'):.3f}")
        # performance reason (only if we enforced it)
        if getattr(r, "n_completed") >= require_perf_min_completed and getattr(r, "performance") < min_perf:
            reasons.append(f"low performance {getattr(r,'performance'):.3f} < {min_perf} "
                           f"(correct {int(getattr(r,'n_correct'))} / completed {int(getattr(r,'n_completed'))})")
        line = f"Rat {getattr(r,'animal')}, Session {getattr(r,'session')} excluded: " + "; ".join(reasons)
        report.append(line)
        reasons_col.append("; ".join(reasons))

    excluded_cols = ["animal","session","n_trials","n_completed","n_correct",
                     "performance","bias","bias_cutoff","enforce_bias"]
    excluded_table = ses.loc[ses["exclude"], excluded_cols].copy()
    if len(excluded_table):
        excluded_table["reasons"] = reasons_col

    kept_cols = ["animal","session","n_trials","n_completed","n_correct",
                 "performance","bias","bias_cutoff","enforce_bias"]
    sessions_kept = ses.loc[~ses["exclude"], kept_cols].reset_index(drop=True)

    # 5) Filter trial-level df accordingly
    bad_idx = set(map(tuple, ses.loc[ses["exclude"], ["animal","session"]].to_numpy()))
    df_trials_clean = df_trials[~df_trials.set_index(["animal","session"]).index.isin(bad_idx)].copy()

    return sessions_kept, report, excluded_table, df_trials_clean

"""
..######..##.....##....###....########..########.....######..########..######...######..####..#######..##....##..######.
.##....##.##.....##...##.##...##.....##.##..........##....##.##.......##....##.##....##..##..##.....##.###...##.##....##
.##.......##.....##..##...##..##.....##.##..........##.......##.......##.......##........##..##.....##.####..##.##......
..######..#########.##.....##.##.....##.######.......######..######....######...######...##..##.....##.##.##.##..######.
.......##.##.....##.#########.##.....##.##................##.##.............##.......##..##..##.....##.##..####.......##
.##....##.##.....##.##.....##.##.....##.##..........##....##.##.......##....##.##....##..##..##.....##.##...###.##....##
..######..##.....##.##.....##.########..########.....######..########..######...######..####..#######..##....##..######.
"""

#used in across session plots for changes that affect the whole session

def shade_change_regions_from_csv(
    ax,
    csv_path,
    rat_id,
    session_min=None,
    session_max=None,
    alpha=1,
    default_colors=("#FFF3CD", "#D1ECF1", "#000000"),  # gentle alternating fills
    draw_edge_between_sessions=True,
):
    """
    Shades background bands from session 1 → first change, and between change points.
    Leaves the region after the last change unshaded.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    csv_path : str
        Path to the change-points CSV with columns: subject_id, change_session, [label], [color]
    subject_id : str
        Current subject ID to filter in the CSV.
    session_min, session_max : int or None
        Range of sessions to consider. If None, inferred from current x-data on the Axes.
    alpha : float
        Fill transparency for bands.
    default_colors : tuple[str]
        Colors to use when CSV doesn't specify a color.
    draw_edge_between_sessions : bool
        If True, bands align between integer sessions using 0.5 offsets.
    """

    # --- Load and filter CSV ---
 
    cp = pd.read_csv(
    csv_path,
    sep=None, engine='python', encoding='utf-8-sig', skipinitialspace=True
    )
    cp.columns = cp.columns.str.replace('\ufeff','', regex=False).str.strip()
    # print(cp.columns)  # uncomment once to verify what pandas sees
    cp = cp.loc[cp["Rat"] == rat_id].copy()  # works now even if header had BOM/space
   
    if cp.empty:
        # Nothing to shade for this subject
        return
    
    # --- Validate/prepare ---
    if "change_session" not in cp.columns:
        raise ValueError("CSV must contain a 'change_session' column (int).")

    cp["change_session"] = cp["change_session"].astype(int)
    cp.sort_values("change_session", inplace=True)

    # --- Infer x-range from plotted data if not provided ---
    if session_min is None or session_max is None:
        # Collect all x data already plotted on the Axes
        x_all = []
        for line in ax.get_lines():
            x = line.get_xdata(orig=False)
            if len(x) > 0:
                x_all.extend(x.tolist())
        if len(x_all) == 0:
            raise ValueError("Could not infer session range from Axes; please set session_min/session_max.")
        if session_min is None:
            session_min = int(min(x_all))
        if session_max is None:
            session_max = int(max(x_all))


    # --- choose visible changes and align metadata ---
    # changes used for shading/lines must match the same rows used to get labels/colors/styles
    mask_vis = (cp["change_session"] >= session_min) & (cp["change_session"] <= session_max)
    cp_vis = cp.loc[mask_vis].reset_index(drop=True)

    # Use these visible changes for edges
    changes = cp_vis["change_session"].astype(int).tolist()

    # Safety: ensure all change points are within displayed range
    changes = [c for c in cp["change_session"].tolist() if session_min <= c <= session_max]
    # If all change points lie beyond the plot, nothing to do
    if len(changes) == 0 and not (min(cp["change_session"]) > session_max):
        # If first change > session_max, we still shade session_min → (first_change-1),
        # but that would exceed the visible range. Here we just proceed with visible span.
        pass

    # Use between-session edges so the shaded blocks align with discrete sessions nicely
    # Example: sessions 1..N → edges at 0.5, 1.5, ..., N+0.5
    if draw_edge_between_sessions:
        to_edge = lambda s: s - 0.5
        left_edge  = to_edge(session_min)
        right_edge = session_max + 0.5
        change_edges = [to_edge(c) for c in changes]
    else:
        left_edge  = session_min
        right_edge = session_max
        change_edges = changes

    # Build intervals to shade:
    # [session_min → first_change), [first_change → second_change), ...,
    # Stop before the last (i.e., do NOT shade after the last change).
    edges = [left_edge] + change_edges
    # Nothing to shade if no changes in range
    if len(edges) == 1:
        # No change within view: shade from session_min → (first_change) if a future change exists?
        # But per your spec, if there are zero change points for the subject, we shade 1→first_change?
        # You said: table marks changes; if none, then no shaded areas. So do nothing.
        return

    # Determine labels/colors per segment from the CSV rows:
    labels = cp_vis["label"].tolist() if "label" in cp_vis.columns else [None] * len(cp_vis)
    colors = cp_vis["color"].tolist() if "color" in cp_vis.columns else [None] * len(cp_vis)
    styles = (
        [str(s).strip().lower() if pd.notna(s) else "shade" for s in cp_vis["style"].tolist()]
        if "style" in cp_vis.columns else ["shade"] * len(cp_vis)
    )

    regions = []
    used_labels = set()

    # We'll shade each region using color from the *next* change row (so the segment preceding that change
    # can be annotated with the phase it represents). If you prefer other logic, flip the indexing.
    for i in range(len(edges)):
        x0 = edges[i]
        # Determine x1: the next edge if it exists, else stop (we don't shade after last change)
        if i + 1 < len(edges):
            x1 = edges[i + 1]
        else:
            # Last interval would be [last_change → end]; we must NOT shade it per your spec.
            break

        # Choose color/label for this segment
        seg_color = colors[i] if (i < len(colors) and pd.notna(colors[i])) else default_colors[i % len(default_colors)]
        seg_label = labels[i] if (i < len(labels) and pd.notna(labels[i])) else None
        seg_style = styles[i] if i < len(styles) else "shade"

        if seg_style in {"line", "vline", "border", "separator"}:
            # draw a vertical line at the boundary BETWEEN sessions
            boundary_x = x1  # right edge is the change boundary
            ax.axvline(
                boundary_x,
                ymin=0.0, ymax=1.0,
                #transform=ax.get_xaxis_transform(),
                color=(seg_color if seg_color else "0.3"),
                linestyle="--",
                linewidth=1.5,
                alpha=alpha,
                zorder=1
            )
            # STRICT: only the 'x' key for lines; no x0/x1
            regions.append({
                "mode": "line",
                "x": float(boundary_x),
                "color": seg_color,
                "label": seg_label,
                "alpha": float(alpha),
            })

            # optional legend token for line labels
            if seg_label and seg_label not in used_labels:
                ax.plot([], [], linestyle="--",
                color=(seg_color if seg_color else "0.3"),
                label=seg_label, alpha=alpha)
                used_labels.add(seg_label)
        else:
            # default: shade band

            # Shade full height (using axis transform so it always spans y-range)
            ax.axvspan(
                x0, x1,
                ymin=0.0, ymax=1.0,
                transform=ax.get_xaxis_transform(),
                facecolor=seg_color,
                alpha=alpha,
                zorder=0
            )

            # store the span so we can reuse later
            regions.append({"mode": "shade", "x0": float(x0), "x1": float(x1), "color": seg_color, "label": seg_label, "alpha": float(alpha)})

        # Optional: If you want one legend entry per *type* (not per segment), you can add a proxy.

        if seg_label and seg_label not in used_labels:
            ax.plot([], [], linestyle="none", marker="s", markersize=10,
                    markerfacecolor=seg_color, markeredgecolor=seg_color,
                    label=seg_label, alpha=alpha, zorder=0)
            used_labels.add(seg_label)

    return regions



# to help use the regions from the previous function without having to recalculate

def draw_regions(ax, regions, alpha=None, zorder=0):
    """
    Re-draw precomputed regions that mix 'shade' bands and 'line' separators.
    Strict: never shades for line entries, never lines for shade entries.
    """
    if not regions:
        return

    for r in regions:
        mode = r.get("mode", None)

        if mode == "shade":
            # required keys for a band
            if "x0" not in r or "x1" not in r:
                # skip malformed shade entries
                continue
            ax.axvspan(
                float(r["x0"]), float(r["x1"]),
                ymin=0.0, ymax=1.0,
                facecolor=r.get("color", None),
                alpha=(alpha if alpha is not None else float(r.get("alpha", 1.0))),
                zorder=zorder,
            )

        elif mode == "line":
            # required key for a line
            if "x" not in r:
                # skip malformed line entries
                continue
            ax.axvline(
                float(r["x"]),
                ymin=0.0, ymax=1.0,
                color=(r.get("color") or "0.3"),
                linestyle="--",
                linewidth=1.5,
                alpha=(alpha if alpha is not None else float(r.get("alpha", 1.0))),
                zorder=zorder + 1,
            )

        else:
            # Unknown/legacy entry → skip to avoid accidental fills
            continue


"""
..######..########.##.......########..######..########....########.....###....########..######....
.##....##.##.......##.......##.......##....##....##.......##.....##...##.##......##....##....##...
.##.......##.......##.......##.......##..........##.......##.....##..##...##.....##....##.........
..######..######...##.......######...##..........##.......########..##.....##....##.....######....
.......##.##.......##.......##.......##..........##.......##...##...#########....##..........##...
.##....##.##.......##.......##.......##....##....##.......##....##..##.....##....##....##....##...
..######..########.########.########..######.....##.......##.....##.##.....##....##.....######....
"""

_POSSIBLE_SUBJECT_COLS = ["animal", "subject", "rat"]

def _find_subject_col(df):
    for c in _POSSIBLE_SUBJECT_COLS:
        if c in df.columns:
            return c
    raise KeyError(f"Subject column not found. Tried: {_POSSIBLE_SUBJECT_COLS}. Pass subject_col=...")

# --- Normalizers --------------------------------------------------------------

def _norm_sex(x: str) -> str:
    if pd.isna(x): return x
    s = str(x).strip().lower()
    if s in {"m", "male", "man"}: return "male"
    if s in {"f", "female", "woman"}: return "female"
    return s  # leave as-is for unexpected values

def _norm_genotype(x: str) -> str:
    if pd.isna(x): return x
    g = str(x).strip().lower().replace(" ", "").replace("-", "")
    if g in {"het", "hetero", "heterozygous"}: return "het"
    if g in {"hom", "homo", "homozygous"}:   return "hom"
    if g in {"wt", "wildtype"}:              return "wt"
    return g

def _to_set(x):
    if x is None: return None
    if isinstance(x, (list, tuple, set)): return {str(v) for v in x}
    return {str(x)}

# --- Metadata I/O -------------------------------------------------------------

def _load_subject_metadata(csv_path, subject_col=None, sex_col="sex", genotype_col="genotype"):
    """
    Reads the subject metadata CSV and standardizes columns to:
      - subject (string)
      - sex: 'male' / 'female'
      - genotype: 'het' / 'wt' / 'hom'
    """
    meta = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)
    subject_col = subject_col or _find_subject_col(meta)

    rename_map = {subject_col: "subject"}
    if sex_col in meta.columns:      rename_map[sex_col] = "sex"
    if genotype_col in meta.columns: rename_map[genotype_col] = "genotype"
    meta = meta.rename(columns=rename_map)

    meta["subject"] = meta["subject"].astype(str).str.strip()

    if "sex" in meta.columns:
        meta["sex"] = meta["sex"].map(_norm_sex)

    if "genotype" in meta.columns:
        meta["genotype"] = meta["genotype"].map(_norm_genotype)

    keep = [c for c in ["subject", "sex", "genotype"] if c in meta.columns]
    return meta[keep].drop_duplicates(subset=["subject"])

# --- Selection core -----------------------------------------------------------

def _select_subjects(meta, *, sex=None, genotypes=None, include_subjects=None, exclude_subjects=None):
    """
    sex: 'male'/'female' (accepts also 'M'/'F', mixtures, case-insensitive)
    genotypes: 'het'/'wt'/'hom' (accepts synonyms like 'heterozygous','wild-type','homozygous')
    """
    df = meta.copy()

    # normalize filter inputs
    sex = _to_set(sex)
    if sex is not None:
        sex = {_norm_sex(s) for s in sex}

    genotypes = _to_set(genotypes)
    if genotypes is not None:
        genotypes = {_norm_genotype(g) for g in genotypes}

    include_subjects = _to_set(include_subjects)
    exclude_subjects = _to_set(exclude_subjects)

    if sex is not None and "sex" in df.columns:
        df = df[df["sex"].isin(sex)]
    if genotypes is not None and "genotype" in df.columns:
        df = df[df["genotype"].isin(genotypes)]

    selected = set(df["subject"].astype(str))
    if include_subjects is not None:
        selected = selected.intersection(include_subjects) if selected else include_subjects
    if exclude_subjects is not None:
        selected = selected.difference(exclude_subjects)

    return selected

# --- Public API ---------------------------------------------------------------

def restrict_subjects(
    df_trials,
    meta_csv,
    *,
    sex=None,                 # 'male' / 'female' (or ['male','female'])
    genotypes=None,           # 'het' / 'wt' / 'hom' (or list)
    include_subjects=None,
    exclude_subjects=None,
    subject_col=None,
    attach_meta=True,
    sex_col="sex",
    genotype_col="genotype",
):
    """
    One-call filter:
      - Loads metadata from `meta_csv`
      - Filters by sex/genotype and/or explicit include/exclude lists
      - Returns trials restricted to matching subjects
      - Optionally attaches `sex` and `genotype` columns
    """
    subject_col = subject_col or _find_subject_col(df_trials)
    meta = _load_subject_metadata(
        meta_csv, subject_col=subject_col, sex_col=sex_col, genotype_col=genotype_col
    )
    wanted = _select_subjects(
        meta,
        sex=sex,
        genotypes=genotypes,
        include_subjects=include_subjects,
        exclude_subjects=exclude_subjects,
    )

    if not wanted:
        return df_trials.iloc[0:0].copy()  # empty but valid

    df = df_trials.copy()
    df[subject_col] = df[subject_col].astype(str).str.strip()
    out = df[df[subject_col].isin(wanted)].copy()

    if attach_meta:
        out = out.merge(meta.rename(columns={"subject": subject_col}), on=subject_col, how="left")

    return out

def restrict_to_subject_list(
    df_trials,
    subjects,
    *,
    subject_col=None,
    attach_meta=False,
    meta_csv=None,
    sex_col="sex",
    genotype_col="genotype",
):
    """Quick filter by an explicit subject list; optionally attach metadata."""
    subject_col = subject_col or _find_subject_col(df_trials)
    df = df_trials.copy()
    subjects = {str(s) for s in (subjects if isinstance(subjects, (list, tuple, set)) else [subjects])}
    df[subject_col] = df[subject_col].astype(str).str.strip()
    out = df[df[subject_col].isin(subjects)].copy()
    if attach_meta and meta_csv is not None:
        meta = _load_subject_metadata(meta_csv, subject_col=subject_col, sex_col=sex_col, genotype_col=genotype_col)
        out = out.merge(meta.rename(columns={"subject": subject_col}), on=subject_col, how="left")
    return out


"""
..#######..##.......########.....########.....###....########....###...
.##.....##.##.......##.....##....##.....##...##.##......##......##.##..
.##.....##.##.......##.....##....##.....##..##...##.....##.....##...##.
.##.....##.##.......##.....##....##.....##.##.....##....##....##.....##
.##.....##.##.......##.....##....##.....##.#########....##....#########
.##.....##.##.......##.....##....##.....##.##.....##....##....##.....##
..#######..########.########.....########..##.....##....##....##.....##
"""

import numpy as np
import pandas as pd

def normalize_ABL_labels(data):
    """Fix ABL label inconsistencies in a make_fig1 pickle."""
    if "merged_valid" in data:
        mv = data["merged_valid"]
        mv["ABL"] = pd.to_numeric(mv["ABL"], errors="coerce")
        mv.loc[mv["ABL"] == 30, "ABL"] = 20
        mv.loc[mv["ABL"] == 35, "ABL"] = 40
        mv.loc[mv["ABL"] == 59, "ABL"] = 60
    if "ABLS" in data:
        for x in [20, 40, 60]:
            if x not in data["ABLS"]:
                data["ABLS"].append(x)
    return data

def extract_psychometric_points(plot_data, abl):
    """
    Compute mean and SEM psychometric points for a given ABL 
    from the make_fig1 pickle structure.
    """
    ilds = plot_data["ilds_dict"][abl]
    merged_valid = plot_data["merged_valid"]
    unique_animal_identifiers = plot_data["unique_animal_identifiers"]

    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        df_sub = merged_valid[
            (merged_valid["batch_name"] == batch)
            & (merged_valid["animal"] == animal)
            & (merged_valid["ABL"] == abl)
        ]
        psycho = [
            np.mean(df_sub[df_sub["ILD"] == ild]["choice"] == 1)
            if len(df_sub[df_sub["ILD"] == ild]) > 0 else np.nan
            for ild in ilds
        ]
        all_psycho_points.append(psycho)

    all_psycho_points = np.array(all_psycho_points, dtype=float)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
    sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(n_points)
    return ilds, mean_psycho, sem_psycho

"""
def overlay_makefig1_psychometrics(ax, abl, plot_data, color="black"):
    
    #Overlay mean psychometric (and SEM) from make_fig1 on an existing axis.
    
    ABLS = plot_data["ABLS"]
    mean_params_dict = plot_data["mean_params_dict"]
    mean_sigmoid_dict = plot_data["mean_sigmoid_dict"]
    x_smooth_dict = plot_data["x_smooth_dict"]

    if abl is None:
        # Average across all ABLs for “All ABLs” plot
        ilds_all, means_all, sems_all = [], [], []
        for abl_ in ABLS:
            ilds, mean_psycho, sem_psycho= extract_psychometric_points(plot_data, abl_)
            ilds_all.append(ilds)
            means_all.append(mean_psycho)
            sems_all.append(sem_psycho)
        ilds = ilds_all[0]
        mean = np.nanmean(np.stack(means_all), axis=0)
        sem = np.nanmean(np.stack(sems_all), axis=0)
        ax.errorbar(ilds, mean, yerr=sem, fmt='o', color=color, capsize=3, markersize=8.5, linewidth=0, elinewidth=1.5, zorder=-1)
        return

    if abl not in ABLS:
        return

    ilds, mean_psycho, sem_psycho = extract_psychometric_points(plot_data, abl)
    x_smooth = x_smooth_dict[abl]

    # Overlay black curve
    if abl in mean_sigmoid_dict and mean_sigmoid_dict[abl] is not None:
        ax.plot(x_smooth, mean_sigmoid_dict[abl], color=color, linewidth=3, zorder=-1)
    elif abl in mean_params_dict:
        ax.plot(x_smooth, sigmoid(x_smooth, *mean_params_dict[abl]), color=color, linewidth=3, zorder=-1)

    # Overlay black points ± SEM
    ax.errorbar(ilds, mean_psycho, yerr=sem_psycho,
                fmt='o', color=color, capsize=3, markersize=8.5, linewidth=-1,
                elinewidth=1.5, zorder=0)

"""

def overlay_makefig1_psychometrics(ax, plot_data, abl=None, color="black", show_individuals=True, use_abl_colors=True):
    """
    Overlay psychometric data from make_fig1 on an existing axis.

    Parameters
    ----------
    ax : matplotlib axis
    plot_data : dict
        Loaded fig1_plot_data.pkl
    abl : int or None
        - If None → grand mean across all ABLs
        - If a value (e.g., 50) → specific ABL only
    color : str
        Color for mean curve/points (usually 'black')
    show_individuals : bool
        Whether to plot individual animal traces.
    use_abl_colors : bool
        If True, color individual traces by ABL (e.g. 20→C0, 35→C1...);
        if False, draw all individual traces in gray.
    """

    ABLS = plot_data["ABLS"]
    mean_sigmoid_dict = plot_data["mean_sigmoid_dict"]
    mean_params_dict = plot_data["mean_params_dict"]
    x_smooth_dict = plot_data["x_smooth_dict"]
    merged_valid = plot_data["merged_valid"]
    unique_animals = plot_data["unique_animal_identifiers"]
    ilds_dict = plot_data["ilds_dict"]

    # Color map by ABL
    abl_colors = {20: "C0", 40: "C1", 50: "C2", 60: "C3"}

    # ===============================================================
    # === 1. Which ABLs to include? ================================
    # ===============================================================
    abls_to_use = ABLS if abl is None else [abl]

    # ===============================================================
    # === 2. Plot individual animal traces =========================
    # ===============================================================
    if show_individuals:
        for abl_ in abls_to_use:
            if abl_ not in ilds_dict:
                continue
            # choose color based on flag
            c = abl_colors.get(abl_, "gray") if use_abl_colors else "gray"

            ilds = ilds_dict[abl_]
            for batch, animal in unique_animals:
                df_sub = merged_valid[
                    (merged_valid["batch_name"] == batch)
                    & (merged_valid["animal"] == animal)
                    & (merged_valid["ABL"] == abl_)
                ]
                if len(df_sub) == 0:
                    continue

                y = [
                    np.mean(df_sub[df_sub["ILD"] == ild]["choice"] == 1)
                    if len(df_sub[df_sub["ILD"] == ild]) > 0 else np.nan
                    for ild in ilds
                ]

                ax.plot(
                    ilds, y, "o-",
                    color=c,
                    alpha=0.2,
                    markersize=4,
                    linewidth=.5,
                    zorder=-2,
                )
    # ===============================================================
    # === 3. Compute mean(s) =======================================
    # ===============================================================
    ilds_all, means_all, sems_all, curves_all, xx_ref = [], [], [], [], None

    for abl_ in abls_to_use:
        ilds, mean_psycho, sem_psycho = extract_psychometric_points(plot_data, abl_)
        ilds_all.append(ilds)
        means_all.append(mean_psycho)
        sems_all.append(sem_psycho)

        # Get sigmoid curve for this ABL
        if abl_ in mean_sigmoid_dict and mean_sigmoid_dict[abl_] is not None:
            xx = x_smooth_dict[abl_]
            yy = mean_sigmoid_dict[abl_]
        elif abl_ in mean_params_dict:
            xx = x_smooth_dict[abl_]
            yy = sigmoid(xx, *mean_params_dict[abl_])
        else:
            continue

        xx_ref = xx if xx_ref is None else xx_ref
        curves_all.append(yy)

        

    # ===============================================================
    # === 4. Aggregate if multiple ABLs (abl=None) =================
    # ===============================================================
    if len(abls_to_use) > 1:
        ilds = ilds_all[0]
        mean = np.nanmean(np.stack(means_all), axis=0)
        sem = np.nanmean(np.stack(sems_all), axis=0)
        mean_curve = np.nanmean(np.stack(curves_all), axis=0)
    else:
        ilds = ilds_all[0]
        mean = means_all[0]
        sem = sems_all[0]
        mean_curve = curves_all[0]

    # ===============================================================
    # === 5. Plot mean points ± SEM and curve =======================
    # ===============================================================
    ax.errorbar(
        ilds, mean, yerr=sem,
        fmt="o", color=color, capsize=3, markersize=8,
        linewidth=0, elinewidth=.5, zorder=-1,
    )

    ax.plot(xx_ref, mean_curve, color=color, linewidth=3, zorder=-2)

    x_smooth = x_smooth_dict[abls_to_use[0]]
    if abls_to_use[0] in mean_sigmoid_dict and mean_sigmoid_dict[abls_to_use[0]] is not None:
        ax.plot(x_smooth, mean_sigmoid_dict[abls_to_use[0]], color=color, linewidth=3, zorder=2)
    elif abls_to_use[0] in mean_params_dict:
        ax.plot(x_smooth, sigmoid(x_smooth, *mean_params_dict[abls_to_use[0]]), color=color, linewidth=3, zorder=2)
    
def extract_rt_points(chrono_data, abl):
    grand_means_data = chrono_data["grand_means_data"]
    if abl not in grand_means_data:
        return None

    stats = grand_means_data[abl]  # in your pickle this is a DataFrame
    x = np.array(stats["abs_ILD"], dtype=float)
    y = np.array(stats["mean"], dtype=float)
    sem = np.array(stats["sem"], dtype=float)

    # Sort by x (prevents weird zig-zag lines if order ever differs)
    order = np.argsort(x)
    return x[order], y[order], sem[order]

_CANON_ABL_COLORS = {
    20: "C0",
    40: "C1",
    50: "C2",
    60: "C3",
}

def _resolve_color(color, abl, default="black", force_black=False):
    if force_black:
        return "black"
    # If user passed a single string like "black" / "C3", honor it
    if isinstance(color, str):
        return color
    # If dict and has this ABL, use it
    if isinstance(color, Mapping) and abl in color:
        return color[abl]
    # Otherwise fall back to canonical
    return _CANON_ABL_COLORS.get(abl, default)

def overlay_makefig1_rt(ax, abl, chrono_data, color="black",
                        zorder=-1, y_scale=1.0, force_black=False):
    out = extract_rt_points(chrono_data, abl)
    if out is None:
        return

    x, y, sem = out
    y *= y_scale
    sem *= y_scale

    marker_color = _resolve_color(color, abl, default="black", force_black=force_black)

    ax.errorbar(
        x, y, yerr=sem,
        fmt="o",
        linestyle="none",
        color=marker_color,          # NOTE: no quotes here
        markerfacecolor=marker_color,
        markeredgecolor=marker_color,
        ecolor=marker_color,
        markersize=8.5,
        linewidth=3,
        elinewidth=1.5,
        capsize=0,
        zorder=zorder,
    )
    ax.plot(x, y, color="black", linewidth=2.5, zorder=zorder)

def overlay_makefig1_rt_individuals(
    ax,
    chrono_data,
    abl=None,
    show_individuals=True,
    use_abl_colors=True,
    alpha=0.35,
    linewidth=1.5,
):
    """
    Overlay individual-subject chronometric (RT) traces from make_fig1 data.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    chrono_data : dict
        Loaded fig1_chrono_plot_data.pkl.
    abl : int or None
        Specific ABL to plot, or None to plot all available.
    show_individuals : bool
        Whether to plot individual animal traces.
    use_abl_colors : bool
        If True, color individual traces by ABL (C0–C3);
        if False, draw all individual traces in gray.
    alpha : float
        Transparency for individual lines.
    linewidth : float
        Line width for individual traces.
    """
    import pandas as pd
    if "all_chrono_data_df" not in chrono_data or not show_individuals:
        return

    df = chrono_data["all_chrono_data_df"].copy()
    df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
    df["abs_ILD"] = pd.to_numeric(df["abs_ILD"], errors="coerce")

    abl_colors = {20: "C0", 40: "C1", 50: "C2", 60: "C3"}
    abls_to_use = sorted(df["ABL"].unique()) if abl is None else [abl]

    for abl_ in abls_to_use:
        df_abl = df[df["ABL"] == abl_]
        if df_abl.empty:
            continue

        color = abl_colors.get(abl_, "gray") if use_abl_colors else "gray"

        # One line per animal
        for (batch_name, animal_id), df_sub in df_abl.groupby(["batch_name", "animal_id"]):
            df_sub = df_sub.sort_values("abs_ILD")
            if "mean" not in df_sub.columns:
                continue
            ax.plot(
                df_sub["abs_ILD"],
                df_sub["mean"],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=-2,
            )

"""
..######..##.....##.####.########.########....########...#####..
.##....##.##.....##..##..##..........##.......##........##...##.
.##.......##.....##..##..##..........##.......##.......##.....##
..######..#########..##..######......##.......#######..##.....##
.......##.##.....##..##..##..........##.............##.##.....##
.##....##.##.....##..##..##..........##.......##....##..##...##.
..######..##.....##.####.##..........##........######....#####..
"""
def shift_ILD_for_ABL50(x):
    """Shift ILD values so ±50 appear at ±18 instead."""
    x = np.asarray(x, dtype=float)
    x = np.where(x == 50, 18, x)
    x = np.where(x == -50, -18, x)
    return x


"""
.......##.##....##.########.
.......##.###...##.##.....##
.......##.####..##.##.....##
.......##.##.##.##.##.....##
.##....##.##..####.##.....##
.##....##.##...###.##.....##
..######..##....##.########.
"""
# ==============================================================
# === COMPUTE JNDs FROM PSYCHOMETRIC FITS (FOR pars=[a,b,c,d]) ==
# ==============================================================

def compute_jnd_by_ABL(psychometric_results, skip_ABL=50):
    """
    Compute JND (mean of 25% and 75% thresholds) from logistic psychometric fits.
    Assumes parameters pars = [a, b, c, d].

    Parameters
    ----------
    psychometric_results : dict
        Each entry {ABL: {"pars": [a, b, c, d], ...}}
        a: slope
        b: bias
        c: lower asymptote
        d: upper asymptote
    skip_ABL : int or list
        ABL(s) to skip (default=50).

    Returns
    -------
    jnd_df : pd.DataFrame
        Columns: ["ABL", "a", "b", "c", "d", "JND"]
    """

    skip_ABL = [skip_ABL] if isinstance(skip_ABL, (int, float)) else list(skip_ABL)
    jnd_list = []

    for abl, res in psychometric_results.items():
        if abl in skip_ABL:
            continue

        pars = res.get("pars")
        if pars is None or len(pars) < 4:
            continue

        a, b, c, d = pars[:4]
        #a = abs(a)  # make sure slope sign doesn't flip direction

        # Clamp asymptotes
        c = np.clip(c, 0, 1)
        d = np.clip(d, 0, 1)
        if d <= c:
            print(f"[compute_jnd_by_ABL] Skipping ABL {abl}: invalid asymptotes (c={c:.3f}, d={d:.3f})")
            continue

        # Define 25% and 75% target probabilities within [c, d]
        upper = 0.75 * (d)
        lower = 1-((1-c) * 0.75)

        try:
            term75 = ((d - c) / (upper - c)) - 1
            term25 = ((d - c) / (lower - c)) - 1
            if term75 <= 0 or term25 <= 0:
                raise ValueError("invalid term for log")
            JND75 = b - (1 / (2 * a)) * np.log(term75)
            JND25 = b - (1 / (2 * a)) * np.log(term25)
            jnd = np.nanmean([abs(JND75 - b), abs(JND25 - b)])
        except Exception:
            jnd = np.nan

        jnd_list.append({"ABL": abl, "a": a, "b": b, "c": c, "d": d, "JND": jnd})

    if len(jnd_list) == 0:
        print("[compute_jnd_by_ABL] Warning: no valid fits found.")
        return pd.DataFrame(columns=["ABL", "a", "b", "c", "d", "JND"])

    df = pd.DataFrame(jnd_list).sort_values("ABL").reset_index(drop=True)
    print(df[["ABL", "JND"]])
    return df
