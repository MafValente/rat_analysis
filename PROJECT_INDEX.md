# Mafalda_analysis – project map

## Start here

- [notebooks/ASD/](notebooks/ASD/): ASD merge, daily review, group comparison, stimulus duration, biased blocks, learning curves, trial selection, and GLM workflows.
- [notebooks/Stakes/](notebooks/Stakes/): Stakes merge, daily review, group analysis, kernel timing, and trial selection workflows.
- [scripts/](scripts/): runnable analyses and command-line entrypoints; see [scripts/README.md](scripts/README.md).

## Reusable Python code

| Location | Purpose |
| --- | --- |
| [analysis/](analysis/) | Dataset resolution, daily merging, and psychometric analysis |
| [Pipeline/](Pipeline/) | Workflow modules for daily plots/reports, group comparisons, learning curves, across-session analysis, biased blocks, and stimulus duration |
| [Helpers/](Helpers/) | Shared data preparation helpers |
| [GroupComparison/](GroupComparison/) | Group comparison configuration, preparation, plotting, layouts, and runners |
| [StimDur/](StimDur/) | Stimulus duration configuration, preparation, plotting, layouts, and runners |
| [Juananalysis/](Juananalysis/) | Kernel regression and related analysis; imported by current workflows |
| [basic_ramp.py](basic_ramp.py) | Sound ramp functions used by the Stakes kernel timing notebook |

Keep these package locations stable unless their imports and path assumptions are updated together.

## Data and results

- [DataFiles/](DataFiles/): datasets organized by genetic line/cohort and animal, with cohort metadata and merged files.
- [outputs/](outputs/): generated figures, tables, daily reports, and executed notebooks.
- [outputs/data_corrections/](outputs/data_corrections/): preserve data correction artifacts.
- Existing GLM run folders remain under their DataFiles cohort directories; relocating them requires updating script paths.

Duration metadata conventions differ by line and cohort. Inspect the schema before filtering or rebuilding datasets; do not assume a universal RT duration rule.

## Notes and history

- [docs/](docs/): schema notes and notebook follow-ups.
- [docs/data_corrections/](docs/data_corrections/): dated records of session-type corrections. Original filenames and contents are preserved.
- [archive/legacy_code/](archive/legacy_code/): historical scripts formerly under History/; retained for reference.
- [archive/snapshots/](archive/snapshots/): historical source archives, including GroupComparison.zip, which differs from the current code.
- [notebooks/legacy/](notebooks/legacy/): historical notebooks.
- [allfiles.txt](allfiles.txt): historical file inventory; paths describe the old layout.

## Environment

- [environment.yml](environment.yml): Conda environment specification.
- [python-version.txt](python-version.txt): recorded Python version.
- `.conda/`: local environment; do not treat it as disposable cache.
