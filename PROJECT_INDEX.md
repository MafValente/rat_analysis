# Mafalda_analysis – reusable code map

## Core reusable modules (import these)
- datasets.py
  - DatasetSpec(...)
  - resolve_data_dir(spec)

- pipeline.py
  - ViewSpec(name, color, selector)
  - AnalysisConfig(...)
  - build_prepared(spec, views, cfg, profile=...) -> bundle

- views.py (repo root)
  - (general code for group selection, i use similar code in other (still) non-modular scripts so tis should become universal)

- session_profiles.py
  - SessionProfile definitions / per-session-type settings (if used)

- metrics.py
  - shared metrics computations (if used)

- Helpers/ (package)
  - DataHelpers.py (prepare_data, restrict_subjects, shift_ILD_for_ABL50, overlays helpers, etc.)

- Psychometric.py
  - compute_psychometrics_by_ABL
  - fit_and_plot_psychometric

## GroupComparison plotting package (reusable)
Folder: plotting/GroupComparison/
- api.py
  - plot_groupcomparison(...)

- layouts.py
  - plot_views_as_rows_1x3(...)
  - plot_abls_as_rows_4x3(...)

- layouts_multi.py
  - multi-bundle overlay layouts (compare across line/cohort)

- traces.py
  - plot_rt(...)
  - plot_mt(...)
  - plot_psy(...)

- overlays.py
  - neurotypical overlays / make_fig1 overlays (if separated here)

- style.py
  - style_axes(...)
  - relabel_ticks_minus18_plus18_as_50(...) (formatter version)

- jnd.py
  - compute_group_jnd(...)
  - plot_old_vs_new_jnd_scatter(...)  (returns fig; no plt.show)

## “Runner” scripts (don’t reuse directly; they call the modules above and some will be legacy soon after being improved)
Folder: scripts/
- Across_sessions.py
- DailyPlots.py
- GLM_rats_ASDcohort2.py
- GLM_rats_ASDcohort2_glmmTMB.py
- GLM_rats_ASDcohort2_r2_plotting.py
- GLM_rats_genotype_cohort_inference_glmmTMB.py
- make_fig1.py
- histograms_timings.py, QQplots.py, psy_param_all.py, etc.

## Data / outputs (not imported)
- DataFiles/
- ASDcohort2GLM.png
