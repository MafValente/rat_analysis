# Scripts

This folder holds runnable analysis scripts and older one-off entrypoints that do not need to live at the repo root.

- Shared modules still used across notebooks stay at the repo root or in package folders such as `Helpers/` and `Pipeline/`.
- Notebook wrappers should point here when they launch standalone scripts.
- Older scripts can stay here while you decide whether they belong in `legacy/`, should be modularized, or can be removed.
- `check_notebook_markdown_drift.py` compares one or more notebooks against `git HEAD` and reports markdown/title/comment cells that changed, were added, or were removed. This is useful as a post-edit guardrail when we only meant to touch code cells.

Both daily animal review terminal commands (`run_daily_animal_review.py` and
`run_daily_animal_review_notebook_batch.py`) also create a PDF automatically:

```text
outputs/daily_animal_review/<line>/<cohort>/daily_animal_review.pdf
```

The report contains one landscape page per animal processed in the current run,
with all its generated plots arranged in two columns. Individual PNGs remain in
each animal's folder. `--out-dir` changes the output root for both PNGs and the PDF.
Each successful run replaces the cohort PDF; `--animals` limits the report to those
animals, and `--figures` in the standalone runner limits the included plots.
No additional command-line option is needed, for example:

```bash
python scripts/run_daily_animal_review_notebook_batch.py --line CNTNAP2 --cohort cohort4
```
