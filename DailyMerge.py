#%%
import os
import pandas as pd
import DataHelpers


"""
.##.....##.########.########...######...########.....######..########..######...######..####..#######..##....##..######.
.###...###.##.......##.....##.##....##..##..........##....##.##.......##....##.##....##..##..##.....##.###...##.##....##
.####.####.##.......##.....##.##........##..........##.......##.......##.......##........##..##.....##.####..##.##......
.##.###.##.######...########..##...####.######.......######..######....######...######...##..##.....##.##.##.##..######.
.##.....##.##.......##...##...##....##..##................##.##.............##.......##..##..##.....##.##..####.......##
.##.....##.##.......##....##..##....##..##..........##....##.##.......##....##.##....##..##..##.....##.##...###.##....##
.##.....##.########.##.....##..######...########.....######..########..######...######..####..#######..##....##..######.
"""
def merge_session_files(input_rat, output_dir=None, output_file=None):
    base_dir = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/"
    input_dir = os.path.join(base_dir, input_rat)

    folder_name = os.path.basename(os.path.normpath(input_dir))
    output_dir = base_dir
    output_file = os.path.join(output_dir, f"merged_{folder_name}.csv")

    if folder_name == "ASD0013":
        DataHelpers.mark_repeated_from(
        f"/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/ASD0013/out_ASD0013_251014.csv",
            start_trial=6690
    )
        
    if folder_name == "ASD0018":
        DataHelpers.mark_repeated_from(
        f"/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/ASD0018/ASD0018_out_251014.csv",
            start_trial=7370)
        
        DataHelpers.mark_repeated_from(
        f"/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/ASD0018/ASD0018_out_251015.csv",
            start_trial=8000)

        DataHelpers.mark_repeated_from(
        f"/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/ASD0018/out_ASD0018_251028.csv",
            start_trial=10900

    )

    all_files = list_csv_files(input_dir)
    if not all_files:
        print("No CSV files found in the directory.")
        return

    # Sort files by modification time (latest last)
    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))

    # Latest file defines reference columns and order
    latest_file = all_files[-1]
    ref_path = os.path.join(input_dir, latest_file)
    ref_df = pd.read_csv(ref_path)
    reference_columns = list(ref_df.columns)
    print(f"Using latest file '{latest_file}' as column reference ({len(reference_columns)} columns).")

    merged_dataframes = []
    all_columns = list(reference_columns)  # maintain order of reference columns
    new_columns = set(reference_columns)

    # Collect any extra columns appearing in other files (added at the end)
    for f in all_files[:-1]:  # skip the reference file since already handled
        try:
            cols = pd.read_csv(os.path.join(input_dir, f), nrows=0).columns
            for c in cols:
                if c not in new_columns:
                    all_columns.append(c)
                    new_columns.add(c)
        except Exception as e:
            print(f"⚠️ Skipping {f}: could not read header ({e})")

    print(f"Total unique columns across all files: {len(all_columns)}")

    # Merge all data
    for file in all_files:
        filepath = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(filepath)
            # reindex to ensure all columns exist, fill missing with NaN
            df = df.reindex(columns=all_columns)
            df["__source_file"] = file
            df["__session_sort_key"] = DataHelpers._infer_file_date(filepath)  # <-- now from name out_YYMMDD
            df["__session_mtime"] = os.path.getmtime(filepath)
            merged_dataframes.append(df)
            print(f"✅ Added {file} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Skipping {file}, could not read CSV: {e}")
            continue

    if merged_dataframes:
        merged_df = pd.concat(merged_dataframes, ignore_index=True)
        session_order = (
            merged_df[["__source_file", "__session_sort_key", "__session_mtime"]]
            .drop_duplicates()
            .sort_values(["__session_sort_key", "__session_mtime"])
            .reset_index(drop=True)
        )
        session_order["__new_session"] = range(1, len(session_order) + 1)
        map_new_session = dict(zip(session_order["__source_file"], session_order["__new_session"]))

        merged_df["session"] = merged_df["__source_file"].map(map_new_session)

        sort_cols = ["session"]
        if "trial" in merged_df.columns:
            sort_cols.append("trial")
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)

        merged_df = merged_df.drop(columns=["__source_file", "__session_sort_key", "__session_mtime"], errors="ignore")

        merged_df.to_csv(output_file, index=False)
        print(f"🎉 Merged {len(merged_dataframes)} files into {output_file}")
        print(f"Final shape: {merged_df.shape}")

        print("\nSession order check:")
        print(session_order[["__new_session", "__source_file", "__session_sort_key"]])

    else:
        print("❌ No files were successfully merged.")
        return  # <-- ensure function exits if nothing merged
"""
   # --- NEW PART: merge into setup-level file ---
    # Split by setup and save per-subject per-setup merged files
    if "box" in merged_df.columns:
        for setup_id, subdf in merged_df.groupby("box"):
            setup_str = str(setup_id).strip()
            setup_file = os.path.join(output_dir, f"merged_{folder_name}_setup{setup_str}.csv")
            subdf.to_csv(setup_file, index=False)
            print(f"📁 Saved per-setup file for subject: {setup_file}")
    else:
        print("⚠️ No 'box' column found — could not split by setup.")


"""
        
def list_csv_files(input_dir):
    """Return list of .csv files in a directory."""
    return [f for f in os.listdir(input_dir) if f.endswith(".csv")]


"""def merge_session_files(input_rat, output_dir=None, output_file=None):
    base_dir = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/"
    input_dir = os.path.join(base_dir, input_rat)

    folder_name = os.path.basename(os.path.normpath(input_dir))
    output_dir = base_dir
    output_file = output_dir + f"merged_{folder_name}.csv"

    all_files = list_csv_files(input_dir)

    if not all_files:
        print("No CSV files found in the directory.")
        return

    merged_dataframes = []
    reference_columns = None

    for file in all_files:
        filepath = os.path.join(input_dir, file)
        print(f"in {file}")
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Skipping {file}, could not read CSV: {e}")
            continue

        # Initialize reference columns
        if reference_columns is None:
            reference_columns = list(df.columns)
            merged_dataframes.append(df)
            print(f"Using {file} as reference with columns: {reference_columns}")
        else:
            if list(df.columns) == reference_columns:
                merged_dataframes.append(df)
                print(f"Merged {file}")
            else:
                print(f"Skipping {file}, columns don't match reference.")

    if merged_dataframes:
        merged_df = pd.concat(merged_dataframes, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged {len(merged_dataframes)} files into {output_file}")
    else:
        print("No files with matching columns were merged.")


# Reusable function of the coe to list the .csv files in one folder
def list_csv_files(input_dir):
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")] 
    return all_files
    """



#%%

"""
.##.....##.########.########...######...########.......###....##....##.####.##.....##....###....##........######.
.###...###.##.......##.....##.##....##..##............##.##...###...##..##..###...###...##.##...##.......##....##
.####.####.##.......##.....##.##........##...........##...##..####..##..##..####.####..##...##..##.......##......
.##.###.##.######...########..##...####.######......##.....##.##.##.##..##..##.###.##.##.....##.##........######.
.##.....##.##.......##...##...##....##..##..........#########.##..####..##..##.....##.#########.##.............##
.##.....##.##.......##....##..##....##..##..........##.....##.##...###..##..##.....##.##.....##.##.......##....##
.##.....##.########.##.....##..######...########....##.....##.##....##.####.##.....##.##.....##.########..######.
"""

import os
import pandas as pd

base_dir = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2/"

model_file = "merged_ASD0007.csv"

def merge_subject_files_with_model(base_dir, model_file, output_file=None):
    """
    Merge all subject-level merged CSVs in a directory using a reference model file
    to define column order. Columns not in the model are appended at the end,
    and missing columns are filled with NaN.

    Parameters
    ----------
    base_dir : str
        Directory containing merged subject files (e.g., 'merged_ASD0001.csv', etc.).
    model_file : str
        Filename or path of the reference file to use for column order (e.g., 'merged_ASD0007.csv').
    output_file : str, optional
        Path for the final merged file (default: 'merged_all_subjects.csv' in base_dir).

    Returns
    -------
    merged_df : pd.DataFrame
        The combined dataframe.
    """

    # --- find the model file ---
    model_path = model_file if os.path.isabs(model_file) else os.path.join(base_dir, model_file)
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    model_df = pd.read_csv(model_path, nrows=0)
    model_columns = list(model_df.columns)
    print(f"📘 Using model file '{os.path.basename(model_path)}' with {len(model_columns)} columns.")

    # --- find other subject merged files ---
    all_files = [
        f for f in os.listdir(base_dir)
        if f.startswith("merged_AS") and f.endswith(".csv") and "setup" not in f
    ]
    if not all_files:
        print("⚠️ No merged subject files found.")
        return None

    # --- gather all unique columns ---
    all_columns = list(model_columns)
    known_cols = set(model_columns)

    for f in all_files:
        try:
            cols = pd.read_csv(os.path.join(base_dir, f), nrows=0).columns
            for c in cols:
                if c not in known_cols:
                    all_columns.append(c)
                    known_cols.add(c)
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")

    print(f"🧾 Total unique columns across all subjects: {len(all_columns)}")

    # --- merge all dataframes ---
    dfs = []
    for f in all_files:
        try:
            path = os.path.join(base_dir, f)
            df = pd.read_csv(path)
            df = df.reindex(columns=all_columns)
            dfs.append(df)
            print(f"✅ Added {f} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")

    if not dfs:
        print("❌ No valid subject files to merge.")
        return None

    merged_df = pd.concat(dfs, ignore_index=True)

    if output_file is None:
        output_file = os.path.join(base_dir, "merged_all_subjects.csv")

    merged_df.to_csv(output_file, index=False)
    print(f"🎉 Saved merged dataset: {output_file}")
    print(f"Final shape: {merged_df.shape}")

    return merged_df

# %%
