#%%
import argparse
import hashlib
import json
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
conda_r_home = Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "R"
if conda_r_home.is_dir():
    os.environ["R_HOME"] = str(conda_r_home)
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

import numpy as np
import pandas as pd

import Helpers.DataHelpers as DataHelpers


# ==============================================================
# CONFIG: choose which line you're analyzing
# ==============================================================
LINE = "CNTNAP2"   # or "SHANK3"
COHORT = "cohort2" # or "cohort1", etc

BASE_DATA_DIR = str(SCRIPT_DIR / "DataFiles")

LINE_ROOTS = {
     ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
     ("CNTNAP2", "cohort4"): "CNTNAP2_cohort4",
     ("SHANK3", "cohort1"): "SHANK3_cohort1",
 }

GROUP_NAMES = ["WT", "HET", "HOM"]
GROUP_NAME_TO_NUMBER = {name: idx for idx, name in enumerate(GROUP_NAMES, start=1)}
GROUP_COLUMN_CANDIDATES = [
    "genotype",
    "Genotype",
    "geno",
    "Geno",
    "subject_group",
    "animal_group",
    "group",
    "Group",
]
SUBJECT_METADATA_FILES = [
    "subject_metadata.csv",
    "subjects.csv",
    "sex_gen.csv",
    "animal_metadata.csv",
    "animals.csv",
    "metadata.csv",
]
SUBJECT_COLUMN_CANDIDATES = ["subject", "animal", "animal_id", "subject_id", "rat", "rat_id"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the cohort model with glmmTMB for comparison with lme4::glmer."
    )
    parser.add_argument(
        "--groups",
        default="1",
        help="Groups to fit: '1', '1,2,3', or 'all'. Default: 1.",
    )
    parser.add_argument(
        "--line",
        default=LINE,
        choices=sorted({key[0] for key in LINE_ROOTS}),
        help=f"Genetic line to analyze. Default: {LINE}.",
    )
    parser.add_argument(
        "--cohort",
        default=COHORT,
        help=f"Cohort to analyze. Default: {COHORT}.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated LINE:COHORT selections to pool, e.g. CNTNAP2:cohort2,CNTNAP2:cohort4.",
    )
    parser.add_argument(
        "--base-data-dir",
        default=BASE_DATA_DIR,
        help=f"Directory containing cohort data folders. Default: {BASE_DATA_DIR}.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=10000,
        help="Maximum optimizer iterations/evaluations passed to glmmTMB.",
    )
    parser.add_argument(
        "--outcome",
        choices=["choice", "accuracy"],
        default="choice",
        help="Model outcome: right/left choice or correct/error accuracy. Default: choice.",
    )
    parser.add_argument(
        "--exclude-repeat-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude repeated trials. Default: true.",
    )
    parser.add_argument(
        "--training-level",
        type=int,
        default=16,
        help="Keep only this training level. Default: 16.",
    )
    parser.add_argument(
        "--apply-rt-stim-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply (session_type == rt_session_type) OR (stim_dur == stim_dur_ms). Default: true.",
    )
    parser.add_argument(
        "--rt-session-type",
        type=int,
        default=1,
        help="Session type used by the RT/stimulus-duration filter. Default: 1.",
    )
    parser.add_argument(
        "--stim-dur-ms",
        type=int,
        default=6000,
        help="Stimulus duration in ms used by the RT/stimulus-duration filter. Default: 6000.",
    )
    parser.add_argument(
        "--min-session",
        type=int,
        default=None,
        help="Minimum session number to keep. Default: 13 for CNTNAP2 cohort2, otherwise no cutoff.",
    )
    parser.add_argument(
        "--dataset-filter-overrides",
        default=None,
        help=(
            "Optional JSON string or path to JSON file mapping LINE:COHORT to filter overrides, "
            "for example {'CNTNAP2:cohort2': {'min_session': 13}}."
        ),
    )
    parser.add_argument(
        "--cohort-random-intercept",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include a cohort-level random intercept. Default: true for pooled multi-dataset runs, false otherwise.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown command-line arguments: {unknown}")
    return args


def dataset_key(line, cohort):
    return f"{line}:{cohort}"


def parse_dataset_selections(value):
    selections = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid dataset selection '{item}'. Use LINE:COHORT, e.g. CNTNAP2:cohort2."
            )
        line, cohort = [piece.strip() for piece in item.split(":", 1)]
        if (line, cohort) not in LINE_ROOTS:
            raise ValueError(
                f"Unknown dataset selection {line}:{cohort}. "
                f"Valid combinations: {', '.join(dataset_key(*key) for key in LINE_ROOTS)}."
            )
        selections.append((line, cohort))
    if not selections:
        raise ValueError("No valid dataset selections were provided.")
    return selections


def parse_groups(value):
    if value.lower() == "all":
        return None
    groups = [int(part.strip()) for part in value.split(",") if part.strip()]
    bad = [group for group in groups if group not in (1, 2, 3)]
    if bad:
        raise ValueError(f"Unknown group numbers: {bad}. Use 1, 2, 3, or all.")
    return groups


def canonical_group_name(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    key = text.lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "wt": "WT",
        "wildtype": "WT",
        "wild": "WT",
        "control": "WT",
        "het": "HET",
        "hetero": "HET",
        "heterozygous": "HET",
        "hom": "HOM",
        "homo": "HOM",
        "homozygous": "HOM",
        "ko": "HOM",
        "knockout": "HOM",
    }
    return aliases.get(key, text.upper())


def first_present_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    lower_to_col = {str(col).lower(): col for col in df.columns}
    for col in candidates:
        found = lower_to_col.get(col.lower())
        if found is not None:
            return found
    return None


def load_subject_group_metadata(data_dir):
    for filename in SUBJECT_METADATA_FILES:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue

        meta = pd.read_csv(path, sep=None, engine="python", dtype=str)
        subject_col = first_present_column(meta, SUBJECT_COLUMN_CANDIDATES)
        group_col = first_present_column(meta, GROUP_COLUMN_CANDIDATES)
        if subject_col is None or group_col is None:
            raise ValueError(
                f"Found metadata file {path}, but could not identify both a subject "
                f"column and a genotype/group column. Columns: {list(meta.columns)}"
            )

        meta = meta[[subject_col, group_col]].rename(
            columns={subject_col: "subject", group_col: "group_name"}
        )
        meta["subject"] = meta["subject"].astype(str).str.strip()
        meta["group_name"] = meta["group_name"].map(canonical_group_name)
        print(f"Loaded subject group metadata from: {path}")
        return meta.dropna(subset=["subject", "group_name"]).drop_duplicates()

    return None


def infer_subject_groups(df: pd.DataFrame, data_dir):
    group_col = first_present_column(df, GROUP_COLUMN_CANDIDATES)
    if group_col is not None:
        meta = df[["subject", group_col]].rename(columns={group_col: "group_name"})
        meta["subject"] = meta["subject"].astype(str).str.strip()
        meta["group_name"] = meta["group_name"].map(canonical_group_name)
        source = f"column '{group_col}'"
    else:
        meta = load_subject_group_metadata(data_dir)
        source = "metadata file"

    if meta is None:
        raise ValueError(
            "No subject genotype/group information was found. Add a genotype column "
            "to merged_all_subjects.csv, or place a metadata CSV in the cohort folder "
            f"({data_dir}) named one of {SUBJECT_METADATA_FILES}, with columns like "
            "'subject,genotype'."
        )

    meta = meta.dropna(subset=["subject", "group_name"]).copy()
    meta["group_name"] = meta["group_name"].map(canonical_group_name)
    known_subjects = set(df["subject"].astype(str))
    meta = meta[meta["subject"].isin(known_subjects)]

    conflicts = (
        meta.drop_duplicates(["subject", "group_name"])
        .groupby("subject")["group_name"]
        .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting group assignments found for subjects: "
            + ", ".join(conflicts.index.astype(str))
        )

    meta = meta.drop_duplicates(subset=["subject"]).copy()
    unknown_groups = sorted(set(meta["group_name"]) - set(GROUP_NAMES))
    if unknown_groups:
        raise ValueError(
            "Unrecognized group labels: "
            + ", ".join(unknown_groups)
            + f". Expected labels compatible with {GROUP_NAMES}."
        )

    missing_subjects = sorted(known_subjects - set(meta["subject"]))
    if missing_subjects:
        raise ValueError(
            "Missing group assignments for subjects: "
            + ", ".join(missing_subjects)
            + f". Source used: {source}."
        )

    group_info = []
    for group_name in GROUP_NAMES:
        subjects = sorted(meta.loc[meta["group_name"] == group_name, "subject"].unique())
        if not subjects:
            continue
        group_info.append({
            "group": GROUP_NAME_TO_NUMBER[group_name],
            "group_name": group_name,
            "subjects": subjects,
        })

    if not group_info:
        raise ValueError(f"No WT/HET/HOM groups were found from {source}.")

    print("Inferred subject groups:")
    for info in group_info:
        print(
            f"  Group {info['group']} ({info['group_name']}): "
            + ", ".join(info["subjects"])
        )

    return group_info


def qualify_dataset_ids(df: pd.DataFrame, line, cohort):
    df = df.copy()
    key = dataset_key(line, cohort)
    df["subject_raw"] = df["subject"].astype(str).str.strip()
    df["session_id_raw"] = df["session_id"].astype(str)
    df["subject"] = key + "__" + df["subject_raw"]
    df["session_id"] = key + "__" + df["session_id_raw"]
    df["cohort_id"] = key
    df["subject"] = df["subject"].astype("category")
    return df


def qualify_group_info(group_info, line, cohort):
    qualified = []
    key = dataset_key(line, cohort)
    for info in group_info:
        updated = deepcopy(info)
        updated["dataset_key"] = key
        updated["line"] = line
        updated["cohort"] = cohort
        updated["subjects"] = [key + "__" + str(subject).strip() for subject in info["subjects"]]
        qualified.append(updated)
    return qualified


def merge_group_info(group_info_list):
    merged = {}
    for dataset_groups in group_info_list:
        for info in dataset_groups:
            slot = merged.setdefault(
                info["group_name"],
                {
                    "group": info["group"],
                    "group_name": info["group_name"],
                    "subjects": [],
                    "datasets": [],
                },
            )
            slot["subjects"].extend(info["subjects"])
            slot["datasets"].append(info["dataset_key"])
    ordered = []
    for group_name in GROUP_NAMES:
        if group_name not in merged:
            continue
        info = merged[group_name]
        info["subjects"] = sorted(set(info["subjects"]))
        info["datasets"] = sorted(set(info["datasets"]))
        ordered.append(info)
    return ordered


def zscore_fun(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return series * 0
    return (series - series.mean()) / std


def missing_counts(df: pd.DataFrame, columns):
    present_columns = [col for col in columns if col in df.columns]
    return {
        col: int(df[col].isna().sum())
        for col in present_columns
        if df[col].isna().sum() > 0
    }


def format_missing_counts(counts):
    if not counts:
        return ""
    return "; ".join(f"{key}={value}" for key, value in counts.items())


def formula_variables(terms):
    variables = []
    for term in terms:
        for part in term.split(":"):
            if part not in variables:
                variables.append(part)
    return variables


def infer_predictor_types(df: pd.DataFrame, formula_terms):
    predictors = formula_variables(formula_terms)
    binary = []
    continuous = []

    for col in predictors:
        if col not in df.columns:
            raise KeyError(f"Predictor '{col}' is not present in the dataframe.")

        values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        value_set = set(values)
        is_binary_01 = len(value_set) <= 2 and value_set.issubset({0, 1, False, True})
        is_signed_binary = value_set.issubset({-1, 0, 1}) and bool(value_set & {-1, 1})

        if is_binary_01 or is_signed_binary:
            binary.append(col)
        else:
            continuous.append(col)

    return continuous, binary, predictors


def add_model_predictors(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(["subject", "session", "trial"]).reset_index(drop=True)

    group_cols = ["subject", "session_id"]
    df["Pre_choice"] = df.groupby(group_cols, observed=True)["Response_signed"].shift(1)
    df["Pre_success"] = df.groupby(group_cols, observed=True)["success"].shift(1)

    pre_choice_filled = int(df["Pre_choice"].isna().sum())
    pre_success_filled = int(df["Pre_success"].isna().sum())
    df["Pre_choice"] = df["Pre_choice"].fillna(0)
    df["Pre_success"] = df["Pre_success"].fillna(0)
    print(
        "Filled previous-trial predictors for first trials: "
        f"Pre_choice={pre_choice_filled}, Pre_success={pre_success_filled}"
    )

    df["fix_time_long"] = np.where(df["fix_time"] > df["fix_time"].median(), 1, -1)
    ild = pd.to_numeric(df["ILD"], errors="coerce")
    df["abs_ILD"] = ild.abs()
    df["ILD_side"] = np.sign(ild).fillna(0)
    df["Pre_choice_aligned"] = df["Pre_choice"] * df["ILD_side"]
    return df


def normalize_model_predictors(df: pd.DataFrame, continuous, binary):
    df = df.copy()
    for col in continuous:
        df[col] = zscore_fun(df[col])

    for col in binary:
        df[col] = pd.to_numeric(df[col], errors="coerce").round()

    return df


def short_config_hash(config):
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def default_min_session(line, cohort):
    if (line, cohort) == ("CNTNAP2", "cohort2"):
        return 13
    return None


def load_dataset_filter_overrides(value):
    if not value:
        return {}

    candidate = Path(os.path.expanduser(value))
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = value

    overrides = json.loads(text)
    if not isinstance(overrides, dict):
        raise ValueError("--dataset-filter-overrides must decode to a JSON object.")
    return overrides


def resolve_dataset_filters(args, line, cohort, overrides):
    filters = {
        "exclude_repeat_trials": args.exclude_repeat_trials,
        "training_level": args.training_level,
        "apply_rt_stim_filter": args.apply_rt_stim_filter,
        "rt_session_type": args.rt_session_type,
        "stim_dur_ms": args.stim_dur_ms,
        "min_session": args.min_session,
    }
    dataset_overrides = overrides.get(dataset_key(line, cohort), {})
    if not isinstance(dataset_overrides, dict):
        raise ValueError(
            f"Dataset filter override for {dataset_key(line, cohort)} must be a JSON object."
        )
    filters.update(dataset_overrides)
    if filters["min_session"] is None:
        filters["min_session"] = default_min_session(line, cohort)
    return filters


def make_run_dir(base_dir, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{short_config_hash(config)}"
    run_dir = os.path.join(base_dir, "runs_glmmTMB", run_id)
    os.makedirs(run_dir, exist_ok=False)
    return run_id, run_dir


def load_and_prepare_data(data_dir, base_data_dir, filters, line, cohort):
    print("Loading and preprocessing data...")

    cohort_file_candidates = [
        os.path.join(data_dir, "merged_all_subjects.csv"),
        os.path.join(base_data_dir, "merged_all_subjects.csv"),
        str(SCRIPT_DIR / "merged_all_subjects.csv"),
    ]
    cohort_file = next(
        (path for path in cohort_file_candidates if os.path.exists(path)),
        None,
    )
    if cohort_file is None:
        raise FileNotFoundError(
            "Could not find merged_all_subjects.csv in: "
            + ", ".join(cohort_file_candidates)
        )

    df = pd.read_csv(cohort_file)
    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
    rows_start = len(df)

    if filters["exclude_repeat_trials"]:
        df = df[df["trial_is_repeat"] == False].copy()

    if filters["training_level"] is not None:
        df = df[df["training_level"] == filters["training_level"]].copy()

    if filters["apply_rt_stim_filter"]:
        sess = pd.to_numeric(df["session_type"], errors="coerce")
        sd = pd.to_numeric(df["stim_dur"], errors="coerce")
        df = df[(sess == filters["rt_session_type"]) | (sd == filters["stim_dur_ms"])].copy()

    min_session = filters["min_session"]

    print(f"Loaded file: {cohort_file} ({rows_start} rows before filtering)")

    out = df.copy()
    out["session_id"] = (
        out["animal"] + "_S" + out["session"].astype(int).astype(str).str.zfill(3)
    )
    out = out.rename(columns={"animal": "subject", "response_poke": "Response"})
    out["Response_signed"] = out["Response"]
    out["Response"] = (out["Response"] + 1) / 2
    out["Accuracy"] = out["success"].map({1: 1, -1: 0})
    out["fix_time"] = out["fix_time"] / 1000

    out["ABLc"] = out["ABL"].astype("category")
    out["repeated_trial"] = out["repeated_trial"].astype("category")
    if min_session is not None:
        out = out[out["session"] >= min_session]

    out["line"] = line
    out["cohort"] = cohort
    out["dataset_key"] = dataset_key(line, cohort)

    print(
        "Applied filters: "
        f"exclude_repeat_trials={filters['exclude_repeat_trials']}, "
        f"training_level={filters['training_level']}, "
        f"apply_rt_stim_filter={filters['apply_rt_stim_filter']}, "
        f"rt_session_type={filters['rt_session_type']}, "
        f"stim_dur_ms={filters['stim_dur_ms']}, "
        f"min_session={min_session}"
    )
    print("Response coding: original -1/1 kept in Response_signed; Response recoded to 0/1.")
    print("Timing conversion: fix_time converted from milliseconds to seconds.")
    print(f"After filtering: {len(out)} rows, {out['subject'].nunique()} subjects")
    return out, cohort_file


def prep_for_glmmtmb(df, response, fixed, group, missing_reports):
    random_groups = ["session_id", "subject"]
    keep = [response] + fixed + random_groups
    df = df.loc[:, [c for c in keep if c in df.columns]].copy()

    rows_before = len(df)
    df[response] = pd.to_numeric(df[response], errors="coerce")

    for c in fixed:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for g in random_groups:
        df[g] = df[g].astype(str)

    key_vars = [response] + fixed + random_groups
    missing_before_drop = missing_counts(df, key_vars)
    if missing_before_drop:
        print(f"   Missing model inputs before glmmTMB fit: {missing_before_drop}")

    df = df.dropna(subset=key_vars)
    rows_dropped = rows_before - len(df)
    if rows_dropped:
        print(f"   Dropped {rows_dropped} rows with missing model inputs.")

    df[response] = df[response].round().astype(int)
    missing_reports.append({
        "group": group,
        "rows_before_model_drop": rows_before,
        "rows_after_model_drop": len(df),
        "rows_dropped_missing_model_inputs": rows_dropped,
        "missing_before_model_drop": format_missing_counts(missing_before_drop),
    })
    return df


def random_formula_terms(group_var, slopes):
    terms = [f"(1 | {group_var})"]
    terms.extend(f"(0 + {slope} | {group_var})" for slope in slopes)
    return terms


def model_spec(outcome):
    if outcome == "choice":
        response_var = "Response"
        preds_grouped = [
            [
                "ABL",
                "ILD",
                "ABL:ILD",
                "ILD:fix_time_long",
                "fix_time_long",
            ],
            [
                "Pre_choice",
                "Pre_success",
                "ILD:Pre_success",
                "Pre_choice:Pre_success",
            ],
        ]
        subject_random_slopes = [
            "ABL",
            "ILD",
            "ABL:ILD",
            "Pre_choice",
            "Pre_success",
            "ILD:Pre_success",
            "ILD:fix_time_long",
            "fix_time_long",
            "Pre_choice:Pre_success",
        ]
        session_random_slopes = ["ILD", "ABL:ILD", "Pre_choice"]
    else:
        response_var = "Accuracy"
        preds_grouped = [
            [
                "ABL",
                "abs_ILD",
                "ILD_side",
                "ABL:abs_ILD",
                "abs_ILD:fix_time_long",
                "fix_time_long",
                "trial",
            ],
            [
                "Pre_success",
                "Pre_choice_aligned",
            ],
        ]
        subject_random_slopes = [
            "ABL",
            "abs_ILD",
            "ILD_side",
            "ABL:abs_ILD",
            "Pre_success",
            "Pre_choice_aligned",
            "abs_ILD:Pre_success",
            "fix_time_long",
            "trial",
        ]
        session_random_slopes = ["abs_ILD", "ABL", "Pre_choice_aligned"]

    return {
        "response_var": response_var,
        "preds_grouped": preds_grouped,
        "var_group_names": ["current", "pre"],
        "subject_random_slopes": subject_random_slopes,
        "sessionID_random_slopes": session_random_slopes,
    }


def import_r_backend():
    import rpy2.robjects as ro
    from rpy2.robjects import Formula, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    installed = bool(ro.r("function() requireNamespace('glmmTMB', quietly = TRUE)")()[0])
    if not installed:
        raise ImportError(
            "R package glmmTMB is not installed in this environment. "
            "Install it with: conda install -n datasci -c conda-forge r-glmmtmb"
        )

    glmmTMB = importr("glmmTMB")
    stats = importr("stats")
    return ro, Formula, pandas2ri, localconverter, glmmTMB, stats


def make_r_helpers(ro):
    fixed_effects = ro.r(
        """
        function(model) {
          cf <- summary(model)$coefficients$cond
          data.frame(
            term = rownames(cf),
            beta = unname(cf[, "Estimate"]),
            se = unname(cf[, "Std. Error"]),
            z = unname(cf[, "z value"]),
            p = unname(cf[, "Pr(>|z|)"]),
            row.names = NULL,
            check.names = FALSE
          )
        }
        """
    )
    random_effects = ro.r(
        """
        function(model) {
          vc <- VarCorr(model)$cond
          rows <- list()
          idx <- 1L
          for (grp in names(vc)) {
            mat <- vc[[grp]]
            sds <- attr(mat, "stddev")
            terms <- names(sds)
            if (is.null(terms)) {
              terms <- rownames(mat)
            }
            clean_grp <- sub("\\\\.[0-9]+$", "", grp)
            for (i in seq_along(sds)) {
              rows[[idx]] <- data.frame(
                random_group = clean_grp,
                term = terms[[i]],
                sd = unname(sds[[i]]),
                vc_name = grp,
                stringsAsFactors = FALSE
              )
              idx <- idx + 1L
            }
          }
          if (length(rows) == 0L) {
            return(data.frame(
              random_group = character(),
              term = character(),
              sd = numeric(),
              vc_name = character(),
              stringsAsFactors = FALSE
            ))
          }
          do.call(rbind, rows)
        }
        """
    )
    diagnostics = ro.r(
        """
        function(model) {
          data.frame(
            convergence = model$fit$convergence,
            optimizer_message = if (!is.null(model$fit$message)) model$fit$message else "",
            pdHess = if (!is.null(model$sdr$pdHess)) model$sdr$pdHess else NA,
            iterations = if (!is.null(model$fit$iterations)) model$fit$iterations else NA,
            evaluations_function = if (!is.null(model$fit$evaluations)) model$fit$evaluations[["function"]] else NA,
            evaluations_gradient = if (!is.null(model$fit$evaluations)) model$fit$evaluations[["gradient"]] else NA,
            logLik = as.numeric(logLik(model)),
            AIC = AIC(model),
            stringsAsFactors = FALSE
          )
        }
        """
    )
    sanitize_call = ro.r(
        """
        function(model, data_name = "model_data") {
          model$call[[1]] <- quote(glmmTMB::glmmTMB)
          model$call$data <- as.name(data_name)
          model
        }
        """
    )
    return fixed_effects, random_effects, diagnostics, sanitize_call


def main():
    args = parse_args()
    selected_groups = parse_groups(args.groups)
    base_data_dir = os.path.abspath(os.path.expanduser(args.base_data_dir))
    dataset_selections = (
        parse_dataset_selections(args.datasets)
        if args.datasets
        else [(args.line, args.cohort)]
    )
    filter_overrides = load_dataset_filter_overrides(args.dataset_filter_overrides)
    pooled_run = len(dataset_selections) > 1
    include_cohort_random_intercept = (
        pooled_run if args.cohort_random_intercept is None else args.cohort_random_intercept
    )

    ro, Formula, pandas2ri, localconverter, glmmTMB, stats = import_r_backend()
    fixed_effects_r, random_effects_r, diagnostics_r, sanitize_call_r = make_r_helpers(ro)

    dataset_frames = []
    cohort_files = []
    all_group_info = []
    dataset_filter_summary = {}
    for line, cohort in dataset_selections:
        data_dir = os.path.join(base_data_dir, LINE_ROOTS[line, cohort])
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Expected data directory does not exist: {data_dir}. "
                "Check --datasets, --line, --cohort, or --base-data-dir."
            )

        dataset_filters = resolve_dataset_filters(args, line, cohort, filter_overrides)
        dataset_filter_summary[dataset_key(line, cohort)] = dataset_filters
        out_one, cohort_file = load_and_prepare_data(
            data_dir, base_data_dir, dataset_filters, line, cohort
        )
        group_info_one = infer_subject_groups(out_one, data_dir)
        out_one = qualify_dataset_ids(out_one, line, cohort)
        group_info_one = qualify_group_info(group_info_one, line, cohort)

        dataset_frames.append(out_one)
        cohort_files.append(cohort_file)
        all_group_info.append(group_info_one)

    out = pd.concat(dataset_frames, ignore_index=True)
    group_info = merge_group_info(all_group_info)
    cohort_file = cohort_files[0] if len(cohort_files) == 1 else cohort_files

    spec = model_spec(args.outcome)
    response_var = spec["response_var"]
    preds_grouped = spec["preds_grouped"]
    var_group_names = spec["var_group_names"]

    all_fixed_predictors = sum(preds_grouped, [])
    print(f"Using predictors: {all_fixed_predictors}")

    out = add_model_predictors(out)
    continuous_predictors, binary_predictors, model_predictors = infer_predictor_types(
        out, all_fixed_predictors
    )
    out = normalize_model_predictors(out, continuous_predictors, binary_predictors)
    print(f"Inferred continuous predictors: {continuous_predictors}")
    print(f"Inferred binary predictors: {binary_predictors}")

    available_groups = [info["group"] for info in group_info]
    if selected_groups is None:
        selected_groups = available_groups
    else:
        missing_groups = sorted(set(selected_groups) - set(available_groups))
        if missing_groups:
            raise ValueError(
                f"Requested groups {missing_groups}, but only groups "
                f"{available_groups} are available in this cohort."
            )

    subject_random_slopes = spec["subject_random_slopes"]
    sessionID_random_slopes = spec["sessionID_random_slopes"]

    fixed_formula = f"{response_var} ~ 1 + " + " + ".join(all_fixed_predictors)
    random_terms = (
        random_formula_terms("session_id", sessionID_random_slopes)
        + random_formula_terms("subject", subject_random_slopes)
    )
    if include_cohort_random_intercept:
        random_terms = ["(1 | cohort_id)"] + random_terms
    random_part = " + ".join(random_terms)
    formula_str = f"{fixed_formula} + {random_part}"

    if pooled_run:
        run_base_dir = os.path.join(base_data_dir, "runs_glmmTMB_pooled")
        os.makedirs(run_base_dir, exist_ok=True)
    else:
        only_line, only_cohort = dataset_selections[0]
        run_base_dir = os.path.join(base_data_dir, LINE_ROOTS[only_line, only_cohort])

    run_config = {
        "backend": "glmmTMB",
        "outcome": args.outcome,
        "response_var": response_var,
        "line": args.line,
        "cohort": args.cohort,
        "dataset_selections": dataset_selections,
        "pooled_run": pooled_run,
        "cohort_random_intercept": include_cohort_random_intercept,
        "base_data_dir": base_data_dir,
        "dataset_filter_summary": dataset_filter_summary,
        "cohort_file": cohort_file,
        "available_groups": group_info,
        "selected_groups": selected_groups,
        "all_fixed_predictors": all_fixed_predictors,
        "model_predictors": model_predictors,
        "continuous_predictors": continuous_predictors,
        "binary_predictors": binary_predictors,
        "preds_grouped": preds_grouped,
        "var_group_names": var_group_names,
        "subject_random_slopes": subject_random_slopes,
        "sessionID_random_slopes": sessionID_random_slopes,
        "fixed_formula": fixed_formula,
        "random_terms": random_terms,
        "formula": formula_str,
        "family": "binomial(logit)",
        "maxiter": args.maxiter,
    }
    run_id, run_dir = make_run_dir(run_base_dir, run_config)
    print(f"Saving this glmmTMB run under: {run_dir}")

    control = ro.r("glmmTMB::glmmTMBControl")(
        optCtrl=ro.ListVector({"iter.max": args.maxiter, "eval.max": args.maxiter})
    )

    model_paths = []
    fixed_effect_tables = []
    random_effect_tables = []
    model_diagnostics = []
    missing_reports = []
    out_glm3 = []

    for info in group_info:
        group_number = info["group"]
        if group_number not in selected_groups:
            continue

        group_name = info["group_name"]
        df_group = out[out["subject"].astype(str).isin(info["subjects"])]
        print(f"\nStarting group {group_number} ({group_name}): {len(df_group)} rows")
        df_out = df_group.copy()
        df_out = df_out.dropna(subset=[response_var])
        out_glm3.append(df_out)

        df_clean = prep_for_glmmtmb(
            df_out,
            response_var,
            model_predictors,
            group_number,
            missing_reports,
        )
        print(
            f"Dataframe ready: {len(df_clean)} rows, "
            f"{df_clean['subject'].nunique()} subjects, "
            f"{df_clean['session_id'].nunique()} sessions"
        )
        print(f"Formula: {formula_str}")

        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df_clean)

        tic = time.perf_counter()
        fit = glmmTMB.glmmTMB(
            Formula(formula_str),
            data=df_r,
            family=stats.binomial(link="logit"),
            control=control,
        )
        elapsed_s = time.perf_counter() - tic
        fit = sanitize_call_r(fit)
        print(f"Finished glmmTMB fit for group {group_number} in {elapsed_s / 60:.2f} min.")

        model_path = os.path.join(run_dir, f"glmmTMB_group_{group_number}.rds")
        summary_path = os.path.join(run_dir, f"glmmTMB_group_{group_number}_summary.txt")
        ro.r["saveRDS"](fit, file=model_path.replace("\\", "/"))
        summary_text = "\n".join(str(x) for x in ro.r["capture.output"](ro.r["summary"](fit)))
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        with localconverter(ro.default_converter + pandas2ri.converter):
            fixed_table = ro.conversion.rpy2py(fixed_effects_r(fit))
            random_table = ro.conversion.rpy2py(random_effects_r(fit))
            diagnostic = ro.conversion.rpy2py(diagnostics_r(fit))

        fixed_table.insert(0, "group", group_number)
        random_table.insert(0, "group", group_number)
        fixed_effect_tables.append(fixed_table)
        random_effect_tables.append(random_table)
        model_paths.append(model_path)

        diagnostic = diagnostic.iloc[0].to_dict()
        diagnostic.update({
            "group": group_number,
            "group_name": group_name,
            "n_rows": len(df_clean),
            "n_subjects": df_clean["subject"].nunique(),
            "n_sessions": df_clean["session_id"].nunique(),
            "elapsed_s": elapsed_s,
            "formula": formula_str,
            "model_path": model_path,
            "summary_path": summary_path,
        })
        model_diagnostics.append(diagnostic)

        if diagnostic.get("convergence", 1) == 0 and bool(diagnostic.get("pdHess", False)):
            print("glmmTMB diagnostics OK: convergence code 0 and positive-definite Hessian.")
        else:
            print(
                "glmmTMB diagnostics warning: "
                f"convergence={diagnostic.get('convergence')}, "
                f"pdHess={diagnostic.get('pdHess')}, "
                f"message={diagnostic.get('optimizer_message')}"
            )

    fixed_effects = (
        pd.concat(fixed_effect_tables, ignore_index=True)
        if fixed_effect_tables
        else pd.DataFrame()
    )
    random_effects = (
        pd.concat(random_effect_tables, ignore_index=True)
        if random_effect_tables
        else pd.DataFrame()
    )

    fixed_effects_path = os.path.join(run_dir, "glmmTMB_fixed_effects.csv")
    random_effects_path = os.path.join(run_dir, "glmmTMB_random_effects.csv")
    diagnostics_path = os.path.join(run_dir, "glmmTMB_fit_diagnostics.csv")
    missing_report_path = os.path.join(run_dir, "glmmTMB_missingness_report.csv")
    run_config_path = os.path.join(run_dir, "run_config.json")
    save_path = os.path.join(run_dir, "glmmTMB_results_cntnap2.pkl")

    fixed_effects.to_csv(fixed_effects_path, index=False)
    random_effects.to_csv(random_effects_path, index=False)
    pd.DataFrame(model_diagnostics).to_csv(diagnostics_path, index=False)
    pd.DataFrame(missing_reports).to_csv(missing_report_path, index=False)
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True, default=str)

    to_save = {
        "run_id": run_id,
        "run_dir": run_dir,
        "run_config": run_config,
        "outcome": args.outcome,
        "response_var": response_var,
        "run_config_path": run_config_path,
        "available_groups": group_info,
        "model_paths": model_paths,
        "model_diagnostics": model_diagnostics,
        "diagnostics_path": diagnostics_path,
        "missing_reports": missing_reports,
        "missing_report_path": missing_report_path,
        "out_glm3": out_glm3,
        "fixed_effects_path": fixed_effects_path,
        "random_effects_path": random_effects_path,
        "all_fixed_predictors": all_fixed_predictors,
        "model_predictors": model_predictors,
        "continuous_predictors": continuous_predictors,
        "binary_predictors": binary_predictors,
        "preds_grouped": preds_grouped,
        "var_group_names": var_group_names,
        "subject_random_slopes": subject_random_slopes,
        "sessionID_random_slopes": sessionID_random_slopes,
    }
    with open(save_path, "wb") as f:
        pickle.dump(to_save, f)

    print(f"Saved run config to {run_config_path}")
    print(f"Saved fixed effects to {fixed_effects_path}")
    print(f"Saved random effects to {random_effects_path}")
    print(f"Saved fit diagnostics to {diagnostics_path}")
    print(f"Saved missingness report to {missing_report_path}")
    print(f"Saved glmmTMB metadata to {save_path}")


if __name__ == "__main__":
    main()

# %%
