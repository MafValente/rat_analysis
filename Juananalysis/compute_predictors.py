# compute_predictors.py
import numpy as np
import pandas as pd

def compute_prev_trial_predictors(df):
    df = df.sort_values(["animal", "session", "trial"])

    # Shifted columns
    df["Pre_Rsp"] = df.groupby(["animal", "session"])["Rsp"].shift(1)
    df["Pre_Sign"] = df.groupby(["animal", "session"])["Sign"].shift(1)
    df["Pre_Out"] = df.groupby(["animal", "session"])["Out"].shift(1)
    df["Pre_ILD"] = df.groupby(["animal", "session"])["ILD"].shift(1)
    df["Pre_LED"] = df.groupby(["animal", "session"])["LED"].shift(1)

    # Binary versions
    df["Pre_Cor"] = (df["Pre_Out"] == 1).astype(float)
    df["Pre_Err"] = (df["Pre_Out"] == 0).astype(float)

    return df


def compute_consecutive_streaks(df):
    """Consecutive correct/error streaks as in MATLAB."""
    df = df.sort_values(["animal", "session", "trial"])

    df["Con_Cor"] = 0
    df["Con_Err"] = 0

    for (animal, session), g in df.groupby(["animal", "session"]):
        streak_cor = 0
        streak_err = 0
        for idx, row in g.iterrows():
            if row["Out"] == 1:
                streak_cor += 1
                streak_err = 0
            elif row["Out"] == 0:
                streak_err += 1
                streak_cor = 0
            df.loc[idx, "Con_Cor"] = streak_cor
            df.loc[idx, "Con_Err"] = streak_err

    # sqrt transform to match MATLAB
    df["Con_Cor"] = np.sqrt(df["Con_Cor"])
    df["Con_Err"] = np.sqrt(df["Con_Err"])
    return df


def compute_reward_rate(df, window=20):
    """Reward rate (like MATLAB movmean of correct trials)."""
    df = df.sort_values(["animal", "session", "trial"])
    df["RewRate"] = (
        df.groupby(["animal", "session"])["Pre_Cor"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )
    df["RewRate"] = df["RewRate"].fillna(0)
    return df


def compute_delta_accuracy(df, window=20):
    """Difference in accuracy between right- and left-side responses."""
    df = df.sort_values(["animal", "session", "trial"])

    # accuracy for right (Rsp=1)
    maskR = df["Pre_Rsp"] == 1
    df["accR"] = (
        maskR & (df["Pre_Out"] == 1)
    ).groupby([df["animal"], df["session"]]).rolling(window, min_periods=1).mean().values
    df["accR"] = df["accR"].fillna(0.5)

    maskL = df["Pre_Rsp"] == -1
    df["accL"] = (
        maskL & (df["Pre_Out"] == 1)
    ).groupby([df["animal"], df["session"]]).rolling(window, min_periods=1).mean().values
    df["accL"] = df["accL"].fillna(0.5)

    df["DeltaAcc"] = df["accR"] - df["accL"]
    df.loc[df["DeltaAcc"].abs() >= 0.4, "DeltaAcc"] = 0

    return df
