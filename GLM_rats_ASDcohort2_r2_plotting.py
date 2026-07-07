#%%
import argparse
from argparse import ArgumentParser
import os
from pathlib import Path
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
conda_r_home = Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "R"
if conda_r_home.is_dir():
    os.environ["R_HOME"] = str(conda_r_home)
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


GROUP_NAMES = ["WT", "HET", "HOM"]
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "DataFiles" / "CNTNAP2_cohort2"


def outcome_label(bundle):
    outcome = bundle.get("outcome") or bundle.get("run_config", {}).get("outcome", "choice")
    response = bundle.get("response_var") or bundle.get("run_config", {}).get("response_var", "Response")
    if outcome == "accuracy":
        return "Accuracy model (correct/error)"
    if outcome == "choice":
        return "Choice model (right/left)"
    return f"{outcome} model ({response})"


def find_latest_results(base_dir=DEFAULT_RESULTS_ROOT):
    candidates = []
    search_specs = [
        ("runs_glmmTMB", "glmmTMB_results_cntnap2.pkl"),
        ("runs_statsmodels", "statsmodels_results_cntnap2.pkl"),
        ("runs", "glmm_results_cntnap2.pkl"),
    ]
    for runs_dir_name, result_name in search_specs:
        runs_dir = base_dir / runs_dir_name
        if not runs_dir.exists():
            continue
        candidates.extend(runs_dir.glob(f"*/{result_name}"))

    if not candidates:
        raise FileNotFoundError(
            f"No result bundles found under {base_dir}. "
            "Pass --results path\\to\\results.pkl explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args():
    parser = ArgumentParser(
        description="Run optional R2-drop analysis and plotting for ASD cohort GLMMs."
    )
    parser.add_argument(
        "--results",
        default=None,
        type=Path,
        help=f"Run-specific result pickle. Default: latest run under {DEFAULT_RESULTS_ROOT}.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Where to save R2 drops and fixed-effect summaries. Default: next to --results.",
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        type=Path,
        help="Fixed-effect figure path. Default: <outcome>_fixed_effects_with_random.png next to --results.",
    )
    parser.add_argument(
        "--skip-r2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip reduced-model R2 fits. Default: true.",
    )
    parser.add_argument(
        "--plot-blups",
        action="store_true",
        help="Also plot subject/session conditional random effects from saved models.",
    )
    parser.add_argument(
        "--save-blup-plot",
        default=None,
        type=Path,
        help="Optional BLUP figure path. Default: subject_blups.png next to --results.",
    )
    parser.add_argument(
        "--plot-conditional-effects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also plot fixed effects plus subject/session BLUPs. Default: true.",
    )
    parser.add_argument(
        "--save-conditional-plot",
        default=None,
        type=Path,
        help="Optional conditional-effects figure path. Default: conditional_effects.png next to --results.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown command-line arguments: {unknown}")
    return args


def join_slopes(slopes):
    if slopes is None:
        return ""
    if isinstance(slopes, str):
        return slopes
    slopes = [s for s in slopes if s]
    return " + ".join(slopes)


def formula_variables(terms):
    variables = []
    for term in terms:
        for part in term.split(":"):
            if part not in variables:
                variables.append(part)
    return variables


def prep_for_r_glmm(df, model_predictors):
    random_groups = ["session_id", "subject"]
    response = "Response"
    keep = [response] + model_predictors + random_groups
    df = df.loc[:, [c for c in keep if c in df.columns]].copy()

    df[response] = df[response].astype(float).round().astype(int)
    for c in model_predictors:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for g in random_groups:
        df[g] = df[g].astype(str)

    return df.dropna(subset=[response] + model_predictors + random_groups)


def compute_r2_drops(bundle, r_api, lme4, stats):
    ro, Formula, pandas2ri, localconverter = r_api
    all_fixed_predictors = bundle["all_fixed_predictors"]
    model_predictors = bundle.get(
        "model_predictors", formula_variables(all_fixed_predictors)
    )
    preds_grouped = bundle["preds_grouped"]
    out_glm3 = bundle["out_glm3"]
    r2_func = ro.r("MuMIn::r.squaredGLMM")
    frac_r2_all = []

    print("\nStarting reduced-model R2-drop analysis.")
    for ktype, df_group in enumerate(out_glm3, start=1):
        print(f"\n=== Computing R2 for group {ktype} ===")
        df_clean = prep_for_r_glmm(df_group, model_predictors)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df_clean)

        fixed_full_str = "1 + " + " + ".join(all_fixed_predictors)
        random_part = "(1 | session_id) + (1 | subject)"
        formula_full_str = f"Response ~ {fixed_full_str} + {random_part}"
        print("Full R2 formula:", formula_full_str)

        full_model = lme4.glmer(
            Formula(formula_full_str),
            data=df_r,
            family=stats.binomial(link="logit"),
        )
        r2_full = float(r2_func(full_model)[0])
        print(f"Marginal R2: {r2_full:.4f}")

        r2_vec = [r2_full]
        for group_i, preds_to_remove in enumerate(preds_grouped, start=1):
            kept_fixed = [
                p for p in all_fixed_predictors if p not in preds_to_remove
            ]
            kept_fixed_core = join_slopes(kept_fixed)
            kept_fixed_str = "1 + " + kept_fixed_core if kept_fixed_core else "1"
            formula_red_str = f"Response ~ {kept_fixed_str} + {random_part}"
            print(f"Reduced R2 formula [{group_i}]:", formula_red_str)

            reduced_model = lme4.glmer(
                Formula(formula_red_str),
                data=df_r,
                family=stats.binomial(link="logit"),
            )
            r2_reduced = float(r2_func(reduced_model)[0])
            r2_vec.append(r2_reduced)
            print(f"Reduced R2={r2_reduced:.4f}")

        frac_r2 = [(r2_full - r) / r2_full for r in r2_vec[1:]]
        frac_r2_all.append(frac_r2)

    return frac_r2_all


def extract_lme4_model_results(bundle, frac_r2_all, ro):
    if "model_paths" not in bundle:
        raise KeyError(
            "Results bundle has no 'model_paths'. Re-run GLM_rats_ASDcohort2.py "
            "so models are saved as .rds files."
        )

    fixef = ro.r["fixef"]
    vcov = ro.r["vcov"]
    diag = ro.r["diag"]
    sqrt = ro.r["sqrt"]
    read_rds = ro.r["readRDS"]
    get_me = ro.r["getME"]

    model_results = []
    for k, model_path in enumerate(bundle["model_paths"]):
        fitres = read_rds(str(model_path).replace("\\", "/"))
        betas_r = fixef(fitres)
        betas = np.array(betas_r)
        se = np.array(sqrt(diag(vcov(fitres))))
        theta_r = get_me(fitres, "theta")
        random_effects = []
        for name, sd in zip(list(theta_r.names), np.array(theta_r)):
            group, _, term = name.partition(".")
            if not term:
                continue
            random_effects.append(
                {
                    "group": group,
                    "term": term,
                    "sd": float(sd),
                    "name": name,
                }
            )
        model_results.append(
            {
                "betas": betas,
                "se": se,
                "t": betas / se,
                "names": list(betas_r.names),
                "random_effects": random_effects,
                "frac_R2": [] if frac_r2_all is None else frac_r2_all[k],
            }
        )

    return model_results


def extract_glmmTMB_model_results(bundle, frac_r2_all, ro):
    if "model_paths" not in bundle:
        raise KeyError(
            "Results bundle has no 'model_paths'. Re-run GLM_rats_ASDcohort2_glmmTMB.py "
            "so models are saved as .rds files."
        )

    read_rds = ro.r["readRDS"]
    fixed_effects = ro.r(
        """
        function(model) {
          cf <- summary(model)$coefficients$cond
          data.frame(
            term = rownames(cf),
            beta = unname(cf[, "Estimate"]),
            se = unname(cf[, "Std. Error"]),
            stringsAsFactors = FALSE,
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
                group = clean_grp,
                term = terms[[i]],
                sd = unname(sds[[i]]),
                name = grp,
                stringsAsFactors = FALSE
              )
              idx <- idx + 1L
            }
          }
          if (length(rows) == 0L) {
            return(data.frame(
              group = character(),
              term = character(),
              sd = numeric(),
              name = character(),
              stringsAsFactors = FALSE
            ))
          }
          do.call(rbind, rows)
        }
        """
    )

    model_results = []
    for k, model_path in enumerate(bundle["model_paths"]):
        fitres = read_rds(str(model_path).replace("\\", "/"))
        fixed_df = fixed_effects(fitres)
        random_df = random_effects(fitres)

        names = list(fixed_df.rx2("term"))
        betas = np.array(fixed_df.rx2("beta"))
        se = np.array(fixed_df.rx2("se"))
        random_rows = []
        for group, term, sd, name in zip(
            list(random_df.rx2("group")),
            list(random_df.rx2("term")),
            np.array(random_df.rx2("sd")),
            list(random_df.rx2("name")),
        ):
            random_rows.append(
                {
                    "group": group,
                    "term": term,
                    "sd": float(sd),
                    "name": name,
                }
            )

        model_results.append(
            {
                "betas": betas,
                "se": se,
                "t": betas / se,
                "names": names,
                "random_effects": random_rows,
                "frac_R2": [] if frac_r2_all is None else frac_r2_all[k],
            }
        )

    return model_results


def extract_model_results(bundle, frac_r2_all, ro, backend):
    if backend == "glmmTMB":
        return extract_glmmTMB_model_results(bundle, frac_r2_all, ro)
    return extract_lme4_model_results(bundle, frac_r2_all, ro)


def extract_blups(bundle, ro):
    if "model_paths" not in bundle:
        raise KeyError("Results bundle has no 'model_paths'.")

    read_rds = ro.r["readRDS"]
    ranef_rows = ro.r(
        """
        function(model) {
          re <- ranef(model)
          if (!is.null(re$cond)) {
            re <- re$cond
          }

          rows <- list()
          idx <- 1L
          for (grp in names(re)) {
            tab <- as.data.frame(re[[grp]])
            if (nrow(tab) == 0L || ncol(tab) == 0L) next
            units <- rownames(tab)
            for (term in colnames(tab)) {
              subjects <- if (grp == "session_id") {
                sub("_S[0-9]+$", "", units)
              } else {
                units
              }
              rows[[idx]] <- data.frame(
                random_group = grp,
                unit = units,
                subject = subjects,
                term = term,
                blup = as.numeric(tab[[term]]),
                stringsAsFactors = FALSE
              )
              idx <- idx + 1L
            }
          }

          if (length(rows) == 0L) {
            return(data.frame(
              random_group = character(),
              unit = character(),
              subject = character(),
              term = character(),
              blup = numeric(),
              stringsAsFactors = FALSE
            ))
          }
          do.call(rbind, rows)
        }
        """
    )

    diagnostics = bundle.get("model_diagnostics", [])
    selected_groups = bundle.get("run_config", {}).get("selected_groups")
    rows = []
    for idx, model_path in enumerate(bundle["model_paths"]):
        if idx < len(diagnostics) and "group" in diagnostics[idx]:
            group_number = int(diagnostics[idx]["group"])
        elif selected_groups and idx < len(selected_groups):
            group_number = int(selected_groups[idx])
        else:
            group_number = idx + 1
        group_name = (
            GROUP_NAMES[group_number - 1]
            if 1 <= group_number <= len(GROUP_NAMES)
            else f"group {group_number}"
        )

        fitres = read_rds(str(model_path).replace("\\", "/"))
        r_df = ranef_rows(fitres)
        df = pd.DataFrame({
            "random_group": list(r_df.rx2("random_group")),
            "unit": list(r_df.rx2("unit")),
            "subject": list(r_df.rx2("subject")),
            "term": list(r_df.rx2("term")),
            "blup": np.array(r_df.rx2("blup")),
        })
        df.insert(0, "group", group_number)
        df.insert(1, "group_name", group_name)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def ordered_blup_terms(blups, bundle):
    present = set(blups["term"])
    order = ["(Intercept)"]
    order.extend(bundle.get("all_fixed_predictors", []))
    order.extend(bundle.get("subject_random_slopes", []))
    order.extend(bundle.get("sessionID_random_slopes", []))

    unique_order = []
    for term in order:
        if term in present and term not in unique_order:
            unique_order.append(term)
    unique_order.extend(sorted(present - set(unique_order)))
    return unique_order


def subject_group_layout(blups):
    subject_groups = (
        blups[["group", "subject"]]
        .drop_duplicates()
        .sort_values(["group", "subject"])
    )
    x_by_subject = {}
    group_centers = {}
    group_boundaries = []
    x_pos = 0.0
    within_step = 0.7
    group_gap = 1.4
    for group_number, group_df in subject_groups.groupby("group", sort=True):
        xs = []
        for subject in group_df["subject"]:
            x_by_subject[subject] = x_pos
            xs.append(x_pos)
            x_pos += within_step
        group_centers[int(group_number)] = float(np.mean(xs))
        group_boundaries.append(x_pos - within_step / 2 + group_gap / 2)
        x_pos += group_gap
    x_min = -within_step
    x_max = x_pos - group_gap + within_step
    return x_by_subject, group_centers, group_boundaries, x_min, x_max


def format_grouped_subject_axis(ax, group_centers, group_boundaries, x_min, x_max):
    center_items = sorted(group_centers.items())
    ax.set_xticks([center for _, center in center_items])
    ax.set_xticklabels(
        [
            GROUP_NAMES[group_number - 1]
            if 1 <= group_number <= len(GROUP_NAMES)
            else f"group {group_number}"
            for group_number, _ in center_items
        ]
    )
    ax.tick_params(axis="x", length=2)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Subjects sorted by ID within group")
    for boundary in group_boundaries[:-1]:
        ax.axvline(boundary, color="0.85", lw=0.8, zorder=0)


def plot_blups(blups, bundle, title=None):
    terms = ordered_blup_terms(blups, bundle)
    x_by_subject, group_centers, group_boundaries, x_min, x_max = subject_group_layout(blups)
    n_terms = len(terms)
    n_cols = min(3, n_terms)
    n_rows = int(np.ceil(n_terms / n_cols))
    colors = sns.color_palette("Set1", max(3, blups["group"].nunique()))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.8 * n_cols, max(3.2 * n_rows, 5)),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.default_rng(0)
    for ax, term in zip(axes, terms):
        term_df = blups[blups["term"] == term].copy()
        ax.axhline(0, color="0.2", lw=0.8, ls="--")
        ax.set_title(term)

        session_df = term_df[term_df["random_group"] == "session_id"]
        for group_number, group_df in session_df.groupby("group"):
            color = colors[int(group_number) - 1]
            x = group_df["subject"].map(x_by_subject).to_numpy(dtype=float)
            x = x + rng.uniform(-0.06, 0.06, size=len(group_df))
            ax.scatter(
                x,
                group_df["blup"],
                marker=".",
                color=color,
                alpha=0.28,
                s=18,
                linewidths=0,
            )

        subject_df = term_df[term_df["random_group"] == "subject"]
        for group_number, group_df in subject_df.groupby("group"):
            color = colors[int(group_number) - 1]
            x = group_df["subject"].map(x_by_subject).to_numpy(dtype=float)
            ax.scatter(
                x,
                group_df["blup"],
                marker="D",
                facecolors="none",
                edgecolors=color,
                s=56,
                linewidths=1.5,
                zorder=4,
            )

        ax.set_ylabel("BLUP / conditional mode")

    for ax in axes[n_terms:]:
        ax.axis("off")

    for ax in axes[:n_terms]:
        format_grouped_subject_axis(ax, group_centers, group_boundaries, x_min, x_max)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="0.25",
            markerfacecolor="none",
            linestyle="None",
            label="subject BLUP",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="0.25",
            linestyle="None",
            label="session BLUP",
        ),
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[gi],
            linestyle="None",
            label=GROUP_NAMES[gi],
        )
        for gi in range(min(len(GROUP_NAMES), blups["group"].nunique()))
    ]
    axes[0].legend(handles=marker_handles, title="Level", frameon=False, loc="best")
    if n_terms > 1:
        axes[1].legend(handles=color_handles, title="Group", frameon=False, loc="best")

    if title:
        fig.suptitle(f"{title}: random-effect BLUPs", y=1.01)
    fig.tight_layout()
    return fig


def conditional_effects_from_blups(blups, model_results, bundle):
    diagnostics = bundle.get("model_diagnostics", [])
    selected_groups = bundle.get("run_config", {}).get("selected_groups")
    fixed_rows = []
    for idx, res in enumerate(model_results):
        if idx < len(diagnostics) and "group" in diagnostics[idx]:
            group_number = int(diagnostics[idx]["group"])
        elif selected_groups and idx < len(selected_groups):
            group_number = int(selected_groups[idx])
        else:
            group_number = idx + 1

        for term, beta, se in zip(res["names"], res["betas"], res["se"]):
            fixed_rows.append({
                "group": group_number,
                "term": term,
                "fixed_beta": float(beta),
                "fixed_se": float(se),
            })

    fixed = pd.DataFrame(fixed_rows)
    conditional = blups.merge(fixed, on=["group", "term"], how="inner")
    conditional["conditional_effect"] = conditional["fixed_beta"] + conditional["blup"]

    fixed_only = fixed.copy()
    fixed_only["group_name"] = fixed_only["group"].map(
        lambda group: GROUP_NAMES[int(group) - 1]
        if 1 <= int(group) <= len(GROUP_NAMES)
        else f"group {int(group)}"
    )
    fixed_only["random_group"] = "fixed"
    fixed_only["unit"] = ""
    fixed_only["subject"] = ""
    fixed_only["blup"] = 0.0
    fixed_only["conditional_effect"] = fixed_only["fixed_beta"]

    return pd.concat([conditional, fixed_only], ignore_index=True, sort=False)


def plot_conditional_effects(conditional, bundle, title=None):
    fixed_rows = conditional[conditional["random_group"] == "fixed"].copy()
    terms = ordered_blup_terms(fixed_rows, bundle)
    x_by_subject, group_centers, group_boundaries, x_min, x_max = subject_group_layout(
        conditional[conditional["random_group"] != "fixed"]
    )
    n_terms = len(terms)
    n_cols = min(3, n_terms)
    n_rows = int(np.ceil(n_terms / n_cols))
    colors = sns.color_palette("Set1", max(3, conditional["group"].nunique()))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.8 * n_cols, max(3.2 * n_rows, 5)),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)

    subject_extents = (
        conditional[conditional["random_group"] != "fixed"][["group", "subject"]]
        .drop_duplicates()
        .assign(x=lambda d: d["subject"].map(x_by_subject))
        .groupby("group")["x"]
        .agg(["min", "max"])
        .to_dict("index")
    )

    for ax, term in zip(axes, terms):
        term_df = conditional[conditional["term"] == term].copy()
        fixed_term_df = term_df[term_df["random_group"] == "fixed"].copy()
        blup_term_df = term_df[term_df["random_group"] != "fixed"].copy()
        blup_term_df = blup_term_df.dropna(subset=["conditional_effect", "fixed_beta"])
        ax.axhline(0, color="0.2", lw=0.8, ls="--")
        ax.set_title(term)

        for group_number, group_df in fixed_term_df.groupby("group"):
            color = colors[int(group_number) - 1]
            fixed_beta = group_df["fixed_beta"].iloc[0]
            fixed_se = group_df["fixed_se"].iloc[0]
            extent = subject_extents.get(group_number)
            if extent is None:
                continue
            ax.hlines(
                fixed_beta,
                extent["min"] - 0.25,
                extent["max"] + 0.25,
                color=color,
                lw=1.6,
                alpha=0.85,
                zorder=1,
            )
            ax.errorbar(
                group_centers[int(group_number)],
                fixed_beta,
                yerr=fixed_se,
                fmt="none",
                ecolor=color,
                elinewidth=1.6,
                capsize=4,
                capthick=1.4,
                zorder=3,
            )

        session_df = blup_term_df[blup_term_df["random_group"] == "session_id"]
        for group_number, group_df in session_df.groupby("group"):
            color = colors[int(group_number) - 1]
            x = group_df["subject"].map(x_by_subject).to_numpy(dtype=float)
            x = x + rng.uniform(-0.06, 0.06, size=len(group_df))
            ax.scatter(
                x,
                group_df["conditional_effect"],
                marker=".",
                color=color,
                alpha=0.28,
                s=18,
                linewidths=0,
                zorder=2,
            )

        subject_df = blup_term_df[blup_term_df["random_group"] == "subject"]
        for group_number, group_df in subject_df.groupby("group"):
            color = colors[int(group_number) - 1]
            x = group_df["subject"].map(x_by_subject).to_numpy(dtype=float)
            ax.scatter(
                x,
                group_df["conditional_effect"],
                marker="D",
                facecolors="none",
                edgecolors=color,
                s=56,
                linewidths=1.5,
                zorder=4,
            )

        ax.set_ylabel("Fixed effect + BLUP")

    for ax in axes[n_terms:]:
        ax.axis("off")

    for ax in axes[:n_terms]:
        format_grouped_subject_axis(ax, group_centers, group_boundaries, x_min, x_max)

    marker_handles = [
        Line2D([0], [0], color="0.25", lw=1.6, label="fixed effect +/- SE"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="0.25",
            markerfacecolor="none",
            linestyle="None",
            label="subject fixed + BLUP",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="0.25",
            linestyle="None",
            label="session fixed + BLUP",
        ),
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[gi],
            linestyle="None",
            label=GROUP_NAMES[gi],
        )
        for gi in range(min(len(GROUP_NAMES), conditional["group"].nunique()))
    ]
    axes[0].legend(handles=marker_handles, title="Estimate", frameon=False, loc="best")
    if n_terms > 1:
        axes[1].legend(handles=color_handles, title="Group", frameon=False, loc="best")

    if title:
        fig.suptitle(f"{title}: fixed effects plus BLUPs", y=1.01)
    fig.tight_layout()
    return fig


def plot_model_results(model_results, var_group_names, include_r2=True, title=None):
    n_models = len(model_results)
    colors = sns.color_palette("Set1", max(3, n_models))
    common_betanames = model_results[0]["names"]
    y_by_name = {name: i for i, name in enumerate(common_betanames)}
    n_pred = len(common_betanames)
    y_positions = np.arange(n_pred)
    jitter_offsets = np.linspace(-0.15, 0.15, n_models) if n_models > 1 else [0]
    jitter_offsets_var = np.linspace(-0.015, 0.015, n_models) if n_models > 1 else [0]

    n_cols = 3 if include_r2 else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(16 if include_r2 else 12, 9))
    plt.subplots_adjust(wspace=0.3)

    ax = axes[0]
    ax.axvline(0, color="k", ls="--")
    for gi, res in enumerate(model_results):
        for i in range(n_pred):
            ax.errorbar(
                res["betas"][i],
                y_positions[i] + jitter_offsets[gi],
                xerr=res["se"][i],
                fmt="o",
                color=colors[gi],
            )
        for re in res["random_effects"]:
            if re["term"] not in y_by_name:
                continue
            fixed_beta = res["betas"][y_by_name[re["term"]]]
            signed_sd = np.sign(fixed_beta) * re["sd"]
            y = y_by_name[re["term"]] + jitter_offsets[gi]
            if re["group"] == "subject":
                ax.scatter(
                    signed_sd,
                    y + 0.06,
                    marker="D",
                    facecolors="none",
                    edgecolors=colors[gi],
                    s=46,
                    linewidths=1.4,
                )
            elif re["group"] == "session_id":
                ax.scatter(
                    signed_sd,
                    y - 0.06,
                    marker="x",
                    color=colors[gi],
                    s=52,
                    linewidths=1.6,
                )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(common_betanames)
    ax.set_xlabel("Fixed beta +/- SE; random-effect SD")
    ax.set_title("Fixed Effects and Random SDs")
    ax.invert_yaxis()
    marker_handles = [
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", label="fixed effect"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="0.25",
            markerfacecolor="none",
            linestyle="None",
            label="subject random SD",
        ),
        Line2D([0], [0], marker="x", color="0.25", linestyle="None", label="session random SD"),
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[gi],
            linestyle="None",
            label=GROUP_NAMES[gi] if gi < len(GROUP_NAMES) else f"group {gi + 1}",
        )
        for gi in range(n_models)
    ]
    marker_legend = ax.legend(
        handles=marker_handles,
        title="Estimate",
        frameon=False,
        loc="lower right",
    )
    ax.add_artist(marker_legend)
    ax.legend(
        handles=color_handles,
        title="Group",
        frameon=False,
        loc="upper right",
    )

    ax = axes[1]
    ax.axvline(0, color="k", ls="--")
    ax.axvline(2, color="k", ls=":")
    ax.axvline(-2, color="k", ls=":")
    for gi, res in enumerate(model_results):
        for i in range(n_pred):
            ax.plot(
                res["t"][i],
                y_positions[i] + jitter_offsets[gi],
                "o",
                color=colors[gi],
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(common_betanames)
    ax.set_xlabel("t-statistic")
    ax.set_title("t-values")
    ax.invert_yaxis()

    if include_r2:
        ax = axes[2]
        group_x = np.arange(len(var_group_names))
        for gi, res in enumerate(model_results):
            vals = np.array(res["frac_R2"])
            for j, val in enumerate(vals):
                ax.plot(
                    val,
                    group_x[j] + jitter_offsets_var[gi],
                    "o-",
                    color=colors[gi],
                    label=GROUP_NAMES[gi] if gi < len(GROUP_NAMES) and j == 0 else "",
                )
        ax.set_yticks(group_x)
        ax.set_yticklabels(var_group_names)
        ax.set_xlabel("Fraction of R2")
        ax.set_title("Variance Explained by Predictor Group")
        ax.invert_yaxis()

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    results_path = args.results or find_latest_results()
    results_path = results_path.expanduser()
    print(f"Using results bundle: {results_path}")

    with results_path.open("rb") as f:
        bundle = pickle.load(f)
    title = outcome_label(bundle)
    outcome = bundle.get("outcome") or bundle.get("run_config", {}).get("outcome", "choice")
    output_path = args.output or results_path.with_name(f"glmm_results_cntnap2_{outcome}_full.pkl")
    backend = bundle.get("run_config", {}).get("backend", "lme4")

    import rpy2.robjects as ro
    from rpy2.robjects import Formula, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    r_api = (ro, Formula, pandas2ri, localconverter)
    if backend == "glmmTMB":
        importr("glmmTMB")
        lme4 = None
    else:
        lme4 = importr("lme4")

    if args.skip_r2:
        frac_r2_all = None
    else:
        if backend == "glmmTMB":
            raise ValueError(
                "R2-drop refits are only implemented for the lme4 path. "
                "Use --skip-r2 when plotting glmmTMB runs."
            )
        stats = importr("stats")
        importr("MuMIn")
        frac_r2_all = compute_r2_drops(bundle, r_api, lme4, stats)
    model_results = extract_model_results(bundle, frac_r2_all, ro, backend)

    extra = {k: v for k, v in bundle.items() if k != "mdle_all"}
    if frac_r2_all is not None:
        extra["frac_R2_all"] = frac_r2_all
    extra["model_results"] = model_results
    with output_path.open("wb") as f:
        pickle.dump(extra, f)
    print(f"Saved extended results to {output_path}")

    fig = plot_model_results(
        model_results,
        bundle["var_group_names"],
        include_r2=frac_r2_all is not None,
        title=title,
    )
    save_plot = args.save_plot or results_path.with_name(
        f"{outcome}_fixed_effects_with_random.png"
    )
    fig.savefig(save_plot, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_plot}")

    if args.plot_blups or args.plot_conditional_effects:
        blups = extract_blups(bundle, ro)
        blups_path = results_path.with_name(f"glmm_{outcome}_blups.csv")
        blups.to_csv(blups_path, index=False)
        print(f"Saved BLUP table to {blups_path}")

    if args.plot_blups:
        blup_fig = plot_blups(blups, bundle, title=title)
        save_blup_plot = args.save_blup_plot or results_path.with_name(f"{outcome}_subject_blups.png")
        blup_fig.savefig(save_blup_plot, dpi=300, bbox_inches="tight")
        print(f"Saved BLUP plot to {save_blup_plot}")

    if args.plot_conditional_effects:
        conditional = conditional_effects_from_blups(blups, model_results, bundle)
        conditional_path = results_path.with_name(f"glmm_{outcome}_conditional_effects.csv")
        conditional.to_csv(conditional_path, index=False)
        print(f"Saved conditional-effects table to {conditional_path}")

        conditional_fig = plot_conditional_effects(conditional, bundle, title=title)
        save_conditional_plot = (
            args.save_conditional_plot
            or results_path.with_name(f"{outcome}_conditional_effects.png")
        )
        conditional_fig.savefig(save_conditional_plot, dpi=300, bbox_inches="tight")
        print(f"Saved conditional-effects plot to {save_conditional_plot}")


if __name__ == "__main__":
    main()

# %%
