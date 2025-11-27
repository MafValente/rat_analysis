# load_data.py
import pandas as pd
import numpy as np

def load_behavior_csv(path):
    df = pd.read_csv(path, low_memory=False)

    # --- Basic renaming for readability ---
    rename_pairs = {
        "poke_response": "response_poke",
        "correct_side": "Sign",
        "stim_ild": "ILD",
        "stim_abl": "ABL",
        "timed_rt": "RT",   # your actual RT column
        "timed_mt": "MT",   # your actual MT column
    }

    rename_map = {old: new for old, new in rename_pairs.items() if old in df.columns}
    df = df.rename(columns=rename_map)

    # Sanity checks
    print("Rename map used:", rename_map)
    print("Columns with 'timed':", [c for c in df.columns if "timed" in c.lower()])
    print("RT in columns after rename?", "RT" in df.columns)
    print("MT in columns after rename?", "MT" in df.columns)

    # --- Derived variables ---
    if "ILD" in df.columns:
        df["Sign"] = np.sign(df["ILD"])
        df["absILD"] = df["ILD"].abs()
        
        # mark easy vs hard ILDs
        EASY_THRESHOLD = 6  # or whatever cutoff you want
        df["Easy"] = (df["absILD"] >= EASY_THRESHOLD).astype(int)  # 1 = easy, 0 = hard

    if "response_poke" in df.columns:
        df["Rsp"] = (df["response_poke"] - 2) * 2 - 1
        df["Rsp_R"] = df["response_poke"] - 2
    else:
        df["Rsp"] = np.nan
        df["Rsp_R"] = np.nan

    if "success" in df.columns:
        # success:  1 = correct, -1 = error, 0 / others = abort / ignore
        s = df["success"].astype(float)

        # map -1 → 0, 1 → 1, others → NaN
        df["Out"] = np.where(
            s == 1, 1.0,
            np.where(s == -1, 0.0, np.nan)
        )
    else:
        raise RuntimeError("No 'success' column found to define correctness ('Out').")

    if "led0_power" in df.columns and "led1_power" in df.columns:
        df["LED"] = ((df["led0_power"] > 0) | (df["led1_power"] > 0)).astype(int)
    else:
        df["LED"] = 0

    return df
