import argparse
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from GLM_rats_ASDcohort2_glmmTMB import (
    BASE_DATA_DIR,
    LINE,
    COHORT,
    add_model_predictors,
    dataset_key,
    import_r_backend,
    infer_predictor_types,
    infer_subject_groups,
    load_and_prepare_data,
    load_dataset_filter_overrides,
    model_spec,
    normalize_model_predictors,
    parse_dataset_selections,
    qualify_dataset_ids,
    qualify_group_info,
    resolve_dataset_filters,
    resolve_dataset_dir,
)

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
conda_r_home = Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "R"
if conda_r_home.is_dir():
    os.environ["R_HOME"] = str(conda_r_home)
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit pooled genotype-by-cohort inference GLMMs with glmmTMB."
    )
    parser.add_argument("--line", default=LINE)
    parser.add_argument("--cohort", default=COHORT)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated LINE:COHORT selections to pool, e.g. CNTNAP2:cohort2,CNTNAP2:cohort4.",
    )
    parser.add_argument("--base-data-dir", default=BASE_DATA_DIR)
    parser.add_argument(
        "--outcome",
        choices=["choice", "accuracy"],
        default="choice",
        help="Model outcome for the pooled inference model.",
    )
    parser.add_argument(
        "--exclude-repeat-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--training-level", type=int, default=16)
    parser.add_argument(
        "--apply-rt-stim-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rt-session-type", type=int, default=1)
    parser.add_argument("--stim-dur-ms", type=int, default=6000)
    parser.add_argument(
        "--session-types-first-block-only",
        default="3,23",
        help=(
            "Comma-separated session_type values for which only the first block of each session "
            "should be kept. Use an empty string to disable. Default: 3,23."
        ),
    )
    parser.add_argument("--min-session", type=int, default=None)
    parser.add_argument(
        "--dataset-filter-overrides",
        default=None,
        help="Optional JSON string or path to JSON file mapping LINE:COHORT to filter overrides.",
    )
    parser.add_argument("--maxiter", type=int, default=10000)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown command-line arguments: {unknown}")
    return args


def make_run_dir(base_dir, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{abs(hash(json.dumps(config, sort_keys=True, default=str))) % (10 ** 8):08d}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def build_subject_genotype_map(group_info_lists):
    mapping = {}
    for dataset_groups in group_info_lists:
        for info in dataset_groups:
            for subject in info["subjects"]:
                mapping[subject] = info["group_name"]
    return mapping


def prep_for_model(df, response, fixed_terms):
    group_terms = ["subject", "session_id", "cohort_id", "genotype"]
    keep = [response] + fixed_terms + group_terms
    model_df = df.loc[:, [col for col in keep if col in df.columns]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce").round().astype("Int64")
    for col in fixed_terms:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    for col in group_terms:
        model_df[col] = model_df[col].astype(str)
    model_df = model_df.dropna(subset=[response] + fixed_terms + group_terms)
    model_df[response] = model_df[response].astype(int)
    return model_df


def extract_lrt_row(anova_df):
    columns = list(anova_df.columns)
    p_col = next((col for col in columns if "Pr(" in col or "p-value" in col.lower()), None)
    chisq_col = next((col for col in columns if "Chisq" in col or "LRT" in col), None)
    df_col = next((col for col in columns if col.lower() == "df"), None)
    row = anova_df.iloc[-1].to_dict()
    return {
        "df": row.get(df_col),
        "chisq": row.get(chisq_col),
        "p_value": row.get(p_col),
    }


def main():
    args = parse_args()
    dataset_selections = (
        parse_dataset_selections(args.datasets)
        if args.datasets
        else [(args.line, args.cohort)]
    )
    if len(dataset_selections) < 2:
        raise ValueError("Use at least two datasets for pooled genotype-by-cohort inference.")

    base_data_dir = os.path.abspath(os.path.expanduser(args.base_data_dir))
    filter_overrides = load_dataset_filter_overrides(args.dataset_filter_overrides)

    dataset_frames = []
    cohort_files = []
    group_info_lists = []
    dataset_filter_summary = {}
    for line, cohort in dataset_selections:
        data_dir = resolve_dataset_dir(base_data_dir, line, cohort)
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
        group_info_lists.append(group_info_one)

    out = pd.concat(dataset_frames, ignore_index=True)
    genotype_map = build_subject_genotype_map(group_info_lists)
    out["genotype"] = out["subject"].astype(str).map(genotype_map)
    out["genotype"] = pd.Categorical(out["genotype"], categories=["WT", "HET", "HOM"])

    spec = model_spec(args.outcome)
    response_var = spec["response_var"]
    all_fixed_predictors = sum(spec["preds_grouped"], [])

    out = add_model_predictors(out)
    continuous_predictors, binary_predictors, model_predictors = infer_predictor_types(
        out, all_fixed_predictors
    )
    out = normalize_model_predictors(out, continuous_predictors, binary_predictors)

    model_df = prep_for_model(out, response_var, model_predictors)
    if model_df.empty:
        raise ValueError("No rows remain after preprocessing for the pooled inference model.")

    ro, Formula, pandas2ri, localconverter, glmmTMB, stats = import_r_backend()
    fixed_effects_r = ro.r(
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
    anova_helper = ro.r(
        """
        function(reduced_model, full_model) {
          as.data.frame(stats::anova(reduced_model, full_model))
        }
        """
    )

    core_predictors = " + ".join(model_predictors)
    random_part = "(1 | session_id) + (1 | subject)"
    full_formula = (
        f"{response_var} ~ 1 + genotype * cohort_id"
        + (f" + {core_predictors}" if core_predictors else "")
        + f" + {random_part}"
    )
    no_interaction_formula = (
        f"{response_var} ~ 1 + genotype + cohort_id"
        + (f" + {core_predictors}" if core_predictors else "")
        + f" + {random_part}"
    )
    no_genotype_formula = (
        f"{response_var} ~ 1 + cohort_id"
        + (f" + {core_predictors}" if core_predictors else "")
        + f" + {random_part}"
    )
    no_cohort_formula = (
        f"{response_var} ~ 1 + genotype"
        + (f" + {core_predictors}" if core_predictors else "")
        + f" + {random_part}"
    )

    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(model_df)

    control = ro.r("glmmTMB::glmmTMBControl")(
        optCtrl=ro.ListVector({"iter.max": args.maxiter, "eval.max": args.maxiter})
    )

    tic = time.perf_counter()
    fit_full = glmmTMB.glmmTMB(
        Formula(full_formula), data=df_r, family=stats.binomial(link="logit"), control=control
    )
    fit_no_interaction = glmmTMB.glmmTMB(
        Formula(no_interaction_formula), data=df_r, family=stats.binomial(link="logit"), control=control
    )
    fit_no_genotype = glmmTMB.glmmTMB(
        Formula(no_genotype_formula), data=df_r, family=stats.binomial(link="logit"), control=control
    )
    fit_no_cohort = glmmTMB.glmmTMB(
        Formula(no_cohort_formula), data=df_r, family=stats.binomial(link="logit"), control=control
    )
    elapsed_s = time.perf_counter() - tic

    with localconverter(ro.default_converter + pandas2ri.converter):
        fixed_effects = ro.conversion.rpy2py(fixed_effects_r(fit_full))
        interaction_lrt = ro.conversion.rpy2py(anova_helper(fit_no_interaction, fit_full))
        genotype_lrt = ro.conversion.rpy2py(anova_helper(fit_no_genotype, fit_full))
        cohort_lrt = ro.conversion.rpy2py(anova_helper(fit_no_cohort, fit_full))

    effect_tests = pd.DataFrame(
        [
            {"effect": "genotype_by_cohort", **extract_lrt_row(interaction_lrt)},
            {"effect": "genotype", **extract_lrt_row(genotype_lrt)},
            {"effect": "cohort", **extract_lrt_row(cohort_lrt)},
        ]
    )

    run_config = {
        "backend": "glmmTMB",
        "model_type": "genotype_by_cohort_inference",
        "dataset_selections": dataset_selections,
        "dataset_filter_summary": dataset_filter_summary,
        "outcome": args.outcome,
        "response_var": response_var,
        "core_predictors": model_predictors,
        "full_formula": full_formula,
        "no_interaction_formula": no_interaction_formula,
        "no_genotype_formula": no_genotype_formula,
        "no_cohort_formula": no_cohort_formula,
        "n_rows": len(model_df),
        "n_subjects": model_df["subject"].nunique(),
        "n_sessions": model_df["session_id"].nunique(),
        "n_cohorts": model_df["cohort_id"].nunique(),
        "elapsed_s": elapsed_s,
        "cohort_files": cohort_files,
    }
    run_base_dir = Path(base_data_dir) / "runs_glmmTMB_genotype_cohort_inference"
    run_id, run_dir = make_run_dir(run_base_dir, run_config)

    summary_path = run_dir / "glmmTMB_genotype_cohort_summary.txt"
    fixed_effects_path = run_dir / "glmmTMB_genotype_cohort_fixed_effects.csv"
    effect_tests_path = run_dir / "glmmTMB_genotype_cohort_effect_tests.csv"
    run_config_path = run_dir / "run_config.json"
    save_path = run_dir / "glmmTMB_genotype_cohort_results.pkl"

    summary_text = "\n".join(str(x) for x in ro.r["capture.output"](ro.r["summary"](fit_full)))
    summary_path.write_text(summary_text, encoding="utf-8")
    fixed_effects.to_csv(fixed_effects_path, index=False)
    effect_tests.to_csv(effect_tests_path, index=False)
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str), encoding="utf-8")

    to_save = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_config": run_config,
        "fixed_effects_path": str(fixed_effects_path),
        "effect_tests_path": str(effect_tests_path),
        "summary_path": str(summary_path),
    }
    with save_path.open("wb") as f:
        pickle.dump(to_save, f)

    print(f"Saved pooled inference summary to {summary_path}")
    print(f"Saved pooled inference fixed effects to {fixed_effects_path}")
    print(f"Saved pooled inference effect tests to {effect_tests_path}")
    print(effect_tests)


if __name__ == "__main__":
    main()
