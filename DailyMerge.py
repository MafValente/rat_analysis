#%%
import os
import pandas as pd
import Helpers.DataHelpers as DataHelpers
import argparse
from pathlib import Path



"""
.##.....##.########.########...######...########.....######..########..######...######..####..#######..##....##..######.
.###...###.##.......##.....##.##....##..##..........##....##.##.......##....##.##....##..##..##.....##.###...##.##....##
.####.####.##.......##.....##.##........##..........##.......##.......##.......##........##..##.....##.####..##.##......
.##.###.##.######...########..##...####.######.......######..######....######...######...##..##.....##.##.##.##..######.
.##.....##.##.......##...##...##....##..##................##.##.............##.......##..##..##.....##.##..####.......##
.##.....##.##.......##....##..##....##..##..........##....##.##.......##....##.##....##..##..##.....##.##...###.##....##
.##.....##.########.##.....##..######...########.....######..########..######...######..####..#######..##....##..######.
"""


# ==========================================================
# CONFIG: where are the data folders for each line/cohort
# ==========================================================
BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"

LINE_ROOTS = {
    ("CNTNAP2", "cohort1"): "CNTNAP2_cohort1",
    ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
    ("CNTNAP2", "cohort3"): "CNTNAP2_cohort3",
    ("CNTNAP2", "cohort4"): "CNTNAP2_cohort4",
    ("SHANK3",  "cohort1"): "SHANK3_cohort1",
    # add more as needed
}

def get_base_dir(line="CNTNAP2", cohort="cohort2"):
    """
    Return the full base_dir for a given line + cohort.
    """
    key = (line, cohort)
    folder = LINE_ROOTS.get(key, f"{line}_{cohort}")
    base_dir = os.path.join(BASE_DATA_DIR, folder)
    if not os.path.isdir(base_dir):
        known = ", ".join(f"{known_line} {known_cohort}" for known_line, known_cohort in sorted(LINE_ROOTS))
        raise FileNotFoundError(
            f"Data folder not found for {line} {cohort}: {base_dir}\n"
            f"Known configured datasets: {known}"
        )
    return base_dir


def get_animals_for_cohort(line="CNTNAP2", cohort="cohort2", rat=None):
    """
    Return the animals for a given line + cohort.
    If rat is provided, return only that rat after validating it exists.
    Prefer sex_gen.csv when available; otherwise fall back to animal folders.
    """
    base_dir = get_base_dir(line, cohort)

    if rat:
        rat_dir = os.path.join(base_dir, rat)
        if not os.path.isdir(rat_dir):
            raise FileNotFoundError(f"Rat folder not found: {rat_dir}")
        return [rat]

    sex_gen_path = os.path.join(base_dir, "sex_gen.csv")
    if os.path.exists(sex_gen_path):
        try:
            sex_gen_df = pd.read_csv(sex_gen_path, sep=";")
            if "animal" in sex_gen_df.columns:
                animals = (
                    sex_gen_df["animal"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                animals = [animal for animal in animals if os.path.isdir(os.path.join(base_dir, animal))]
                if animals:
                    return animals
        except Exception as e:
            print(f"⚠️ Could not read {sex_gen_path}: {e}")

    animals = [
        entry for entry in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, entry)) and entry.startswith("ASD")
    ]
    return animals


def _normalize_source_label(filename: str) -> str:
    return Path(filename).name


def _validate_one_session_per_source(merged_df: pd.DataFrame, *, source_col: str, session_col: str = "session") -> None:
    source_to_sessions = (
        merged_df[[source_col, session_col]]
        .drop_duplicates()
        .groupby(source_col)[session_col]
        .nunique()
    )
    bad_sources = source_to_sessions[source_to_sessions != 1]
    if not bad_sources.empty:
        raise ValueError(
            "Each daily CSV must map to exactly one merged session. "
            f"Violations: {bad_sources.to_dict()}"
        )

    session_to_sources = (
        merged_df[[source_col, session_col]]
        .drop_duplicates()
        .groupby(session_col)[source_col]
        .nunique()
    )
    bad_sessions = session_to_sources[session_to_sources != 1]
    if not bad_sessions.empty:
        raise ValueError(
            "Each merged session must come from exactly one daily CSV. "
            f"Violations: {bad_sessions.to_dict()}"
        )

def _get_session_rules(session_edits, animal):
    if not session_edits:
        return []

    rules = []
    for key in (animal, str(animal), "all", "*"):
        if key in session_edits:
            value = session_edits[key]
            rules.extend(value if isinstance(value, list) else [value])
    return rules


def _matching_session_rules(rules, file):
    matches = []
    for rule in rules:
        rule_file = rule.get("file") or rule.get("source_file")
        if rule_file is None:
            matches.append(rule)
            continue
        if _normalize_source_label(rule_file) == _normalize_source_label(file):
            matches.append(rule)
    return matches


def _find_trial_col(df, trial_col=None):
    if trial_col and trial_col in df.columns:
        return trial_col
    candidates = ["trial", "trial_index", "trial_number", "trial_num", "trialID"]
    return next((col for col in candidates if col in df.columns), None)


def _coerce_repeated_trial(df):
    if "repeated_trial" not in df.columns:
        df["repeated_trial"] = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"))
        return df

    if not pd.api.types.is_bool_dtype(df["repeated_trial"]):
        mapped = (
            df["repeated_trial"]
            .astype("string")
            .str.strip()
            .str.upper()
            .map({
                "TRUE": True,
                "FALSE": False,
                "1": True,
                "0": False,
                "YES": True,
                "NO": False,
            })
        )
        df["repeated_trial"] = pd.Series(pd.array(mapped, dtype="boolean"))
    return df


def _apply_session_rules(df, file, rules):
    original_rows = len(df)
    for rule in rules:
        action = rule.get("action", "drop_from_trial")
        trial_col = _find_trial_col(df, rule.get("trial_col"))

        if action == "drop_entire_session":
            print(f"🧹 Removing entire session file {file} ({len(df)} rows)")
            return df.iloc[0:0].copy()

        if action in {"drop_from_trial", "mark_repeated_from"}:
            if trial_col is None:
                raise ValueError(
                    f"Cannot apply {action} to {file}: no trial column found. "
                    "Pass trial_col in the rule."
                )
            start_trial = int(rule["start_trial"])
            trial_num = pd.to_numeric(df[trial_col], errors="coerce")
            mask = trial_num >= start_trial

            if action == "drop_from_trial":
                removed = int(mask.sum())
                df = df.loc[~mask].copy()
                print(f"🧹 Removed {removed} rows from {file} where {trial_col} >= {start_trial}")
            else:
                changed = int(mask.sum())
                df = _coerce_repeated_trial(df)
                df.loc[mask, "repeated_trial"] = True
                df["repeated_trial"] = df["repeated_trial"].astype("boolean")
                print(f"🧹 Marked {changed} rows repeated in {file} where {trial_col} >= {start_trial}")
            continue

        if action == "drop_trial_range":
            if trial_col is None:
                raise ValueError(
                    f"Cannot apply {action} to {file}: no trial column found. "
                    "Pass trial_col in the rule."
                )
            start_trial = rule.get("start_trial")
            end_trial = rule.get("end_trial")
            trial_num = pd.to_numeric(df[trial_col], errors="coerce")
            mask = pd.Series(True, index=df.index)
            if start_trial is not None:
                mask &= trial_num >= int(start_trial)
            if end_trial is not None:
                mask &= trial_num <= int(end_trial)
            removed = int(mask.sum())
            df = df.loc[~mask].copy()
            print(
                f"🧹 Removed {removed} rows from {file} where "
                f"{trial_col} is in [{start_trial}, {end_trial}]"
            )
            continue

        if action == "drop_block":
            block_col = rule.get("block_col", "block")
            if block_col not in df.columns:
                raise ValueError(
                    f"Cannot apply {action} to {file}: no block column named {block_col!r} found. "
                    "Pass block_col in the rule if needed."
                )

            blocks = rule.get("block", rule.get("blocks"))
            if blocks is None:
                raise ValueError(f"Cannot apply {action} to {file}: pass block=... or blocks=[...].")
            if isinstance(blocks, (str, int, float)):
                blocks = [blocks]

            observed_num = pd.to_numeric(df[block_col], errors="coerce")
            wanted_num = pd.to_numeric(pd.Series(blocks), errors="coerce")
            if wanted_num.notna().all():
                mask = observed_num.isin(wanted_num.astype(float).tolist())
            else:
                wanted = {str(block).strip() for block in blocks}
                mask = df[block_col].astype("string").str.strip().isin(wanted)

            removed = int(mask.sum())
            df = df.loc[~mask].copy()
            print(f"🧹 Removed {removed} rows from {file} where {block_col} is in {list(blocks)}")
            continue

        raise ValueError(f"Unknown session edit action for {file}: {action}")

    if len(df) != original_rows:
        df = df.reset_index(drop=True)
    return df


def _apply_rt_value_rules(merged_df, rt_value_edits):
    if not rt_value_edits:
        return merged_df

    out = merged_df.copy()
    if "rt_value_valid" not in out.columns:
        out["rt_value_valid"] = True
    else:
        out["rt_value_valid"] = out["rt_value_valid"].fillna(True).astype(bool)
    if "rt_value_note" not in out.columns:
        out["rt_value_note"] = pd.NA

    for rule in rt_value_edits:
        setup_col = rule.get("setup_col", "box")
        date_col = rule.get("date_col", "source_date")
        rt_col = rule.get("rt_col", "timed_rt")
        missing = [col for col in [setup_col, date_col, rt_col] if col not in out.columns]
        if missing:
            raise KeyError(f"RT value edit cannot be applied; missing columns: {missing}")

        dates = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
        start = pd.to_datetime(rule["start_date"]).normalize()
        end = pd.to_datetime(rule["end_date"]).normalize()
        mask = dates.ge(start) & dates.le(end)

        if "setup" in rule:
            setup = pd.to_numeric(out[setup_col], errors="coerce")
            mask &= setup.eq(float(rule["setup"]))

        if "animal" in rule and "animal" in out.columns:
            animals = rule["animal"]
            if isinstance(animals, str):
                animals = [animals]
            wanted = {str(animal).strip() for animal in animals}
            mask &= out["animal"].astype("string").str.strip().isin(wanted)

        if "source_file" in rule and "source_file" in out.columns:
            source_files = rule["source_file"]
            if isinstance(source_files, str):
                source_files = [source_files]
            wanted = {_normalize_source_label(source_file) for source_file in source_files}
            mask &= out["source_file"].astype("string").map(_normalize_source_label).isin(wanted)

        if "block" in rule or "blocks" in rule:
            block_col = rule.get("block_col", "block")
            if block_col not in out.columns:
                raise KeyError(f"RT value edit cannot be applied; missing block column: {block_col}")

            blocks = rule.get("block", rule.get("blocks"))
            if isinstance(blocks, (str, int, float)):
                blocks = [blocks]

            observed_num = pd.to_numeric(out[block_col], errors="coerce")
            wanted_num = pd.to_numeric(pd.Series(blocks), errors="coerce")
            if wanted_num.notna().all():
                mask &= observed_num.isin(wanted_num.astype(float).tolist())
            else:
                wanted = {str(block).strip() for block in blocks}
                mask &= out[block_col].astype("string").str.strip().isin(wanted)

        affected = int(mask.sum())
        out.loc[mask, "rt_value_valid"] = False
        out.loc[mask, "rt_value_note"] = rule.get("reason", "invalid numeric timed_rt")
        out.loc[mask, rt_col] = pd.NA
        print(
            f"🧹 Set {rt_col}=NaN for {affected} rows "
            f"from {start.date()} to {end.date()}"
            + (f" on setup {rule['setup']}" if "setup" in rule else "")
        )

    return out


def _merge_single_rat_session_files(input_rat, line="CNTNAP2", cohort="cohort2", session_edits=None, rt_value_edits=None):

    """
    Merge all raw session CSVs for one subject into merged_<rat>.csv
    for a given line+cohort.
    """
        
    base_dir = get_base_dir(line, cohort)
    input_dir = os.path.join(base_dir, input_rat)

    folder_name = os.path.basename(os.path.normpath(input_dir))
    output_dir = base_dir
    output_file = os.path.join(output_dir, f"merged_{folder_name}.csv")
    session_rules = _get_session_rules(session_edits, folder_name)

# def merge_session_files(input_rat, output_dir=None, output_file=None):
#     base_dir = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2/"
#     input_dir = os.path.join(base_dir, input_rat)

#     folder_name = os.path.basename(os.path.normpath(input_dir))
#     output_dir = base_dir
#     output_file = os.path.join(output_dir, f"merged_{folder_name}.csv")

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
    if any(rule.get("action") == "mark_repeated_from" for rule in session_rules) and "repeated_trial" not in new_columns:
        all_columns.append("repeated_trial")
        new_columns.add("repeated_trial")
        print("Added 'repeated_trial' to merged columns because a session edit uses mark_repeated_from.")

    # Merge all data
    for file in all_files:
        filepath = os.path.join(input_dir, file)
        try:
            matching_rules = _matching_session_rules(session_rules, file)
            if any(rule.get("action") == "drop_entire_session" for rule in matching_rules):
                print(f"🧹 Skipping {file}: configured as drop_entire_session")
                continue

            df = pd.read_csv(filepath)
            if matching_rules:
                df = _apply_session_rules(df, file, matching_rules)
                if df.empty:
                    print(f"🧹 No rows left after edits for {file}; skipping.")
                    continue

            # reindex to ensure all columns exist, fill missing with NaN
            df = df.reindex(columns=all_columns)

            # add stim duration annotation (safe to do here)
            df = DataHelpers.add_stim_dur(
                df,
                sound_col="sound_index",
                session_col="session_type",
                out_col="stim_dur",
                type1_value=6000,  # or pd.NA
            )

            # ✅ NEW: normalize is_short_sound + short_duration across old/new sessions
            df = DataHelpers.normalize_short_sound_fields(
                df,
                session_col="session_type",
                stimdur_col="stim_dur",
                shortdur_col="short_duration",
                isshort_col="is_short_sound",
                long_value=6000,
            )

            df["__source_file"] = _normalize_source_label(file)
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
        merged_df["source_file"] = merged_df["__source_file"]
        merged_df["source_date"] = merged_df["__session_sort_key"]
        merged_df = _apply_rt_value_rules(merged_df, rt_value_edits)

        sort_cols = ["session"]
        if "trial" in merged_df.columns:
            sort_cols.append("trial")
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)

        _validate_one_session_per_source(merged_df, source_col="source_file", session_col="session")

        merged_df = merged_df.drop(columns=["__source_file", "__session_sort_key", "__session_mtime"], errors="ignore")

        merged_df.to_csv(output_file, index=False)
        print(f"🎉 Merged {len(merged_dataframes)} files into {output_file}")
        print(f"Final shape: {merged_df.shape}")

        print("\nSession order check:")
        print(session_order[["__new_session", "__source_file", "__session_sort_key"]])

    else:
        print("❌ No files were successfully merged.")
        return  # <-- ensure function exits if nothing merged


def merge_session_files(line="CNTNAP2", cohort="cohort2", rat=None, session_edits=None, rt_value_edits=None):
    """
    Merge raw session CSVs for either:
    - one rat, if rat is provided
    - all rats in the given line + cohort, if rat is omitted

    Examples
    --------
    merge_session_files("SHANK3", "cohort1")
    merge_session_files("CNTNAP2", "cohort2", rat="ASD0013")
    """
    animals = get_animals_for_cohort(line, cohort, rat=rat)
    if not animals:
        print(f"❌ No animals found for {line} {cohort}")
        return

    print(f"Processing {len(animals)} animal(s) for {line} {cohort}: {', '.join(animals)}")
    for animal in animals:
        _merge_single_rat_session_files(
            animal,
            line=line,
            cohort=cohort,
            session_edits=session_edits,
            rt_value_edits=rt_value_edits,
        )
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

#base_dir = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2/"

#merge_subject_files_with_model("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2", "merged_ASD0007.csv")

model_file = None

def merge_subject_files_with_model(line="CNTNAP2", cohort="cohort2",
                                   model_file=None,
                                   output_file=None):
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

    base_dir = get_base_dir(line, cohort)

    # --- find the model file ---
    if model_file is None:
        available_merged_files = sorted(
            f for f in os.listdir(base_dir)
            if f.startswith("merged_AS") and f.endswith(".csv") and "setup" not in f
        )
        if not available_merged_files:
            print("❌ No merged subject files found to use as model.")
            return None
        model_path = os.path.join(base_dir, available_merged_files[0])
    else:
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
    sort_cols = [col for col in ["animal", "session", "trial"] if col in merged_df.columns]
    if sort_cols:
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)

    if output_file is None:
        output_file = os.path.join(base_dir, "merged_all_subjects.csv")

    merged_df.to_csv(output_file, index=False)
    print(f"🎉 Saved merged dataset: {output_file}")
    print(f"Final shape: {merged_df.shape}")

    return merged_df

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line",   choices=["CNTNAP2", "SHANK3"], default="CNTNAP2")
    parser.add_argument("--cohort", default="cohort2")
    parser.add_argument("--rat", help="Optional subject ID, e.g. ASD0007. If omitted, process all rats in the cohort.")
    parser.add_argument(
        "--mode",
        choices=["session", "animals", "both"],
        default="both",
        help=(
            "session = merge one rat's raw files\n"
            "animals     = merge all merged_ASDXXXX into merged_all_subjects\n"
            "both    = do both steps"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model file used for all-subject merge",
    )
    args = parser.parse_args()
    animals = get_animals_for_cohort(args.line, args.cohort, rat=args.rat)
    if not animals:
        raise SystemExit(f"No animals found for {args.line} {args.cohort}")

    # --- run only what you asked for ---
    if args.mode in ("session", "both"):
        merge_session_files(args.line, args.cohort, rat=args.rat)

    if args.mode in ("animals", "both"):
        merge_subject_files_with_model(line=args.line, cohort=args.cohort, model_file=args.model)
