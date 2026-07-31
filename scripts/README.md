# Scripts

This folder holds runnable analysis scripts and older one-off entrypoints that do not need to live at the repo root.

- Shared modules still used across notebooks stay at the repo root or in package folders such as `Helpers/` and `Pipeline/`.
- Notebook wrappers should point here when they launch standalone scripts.
- Older scripts can stay here while you decide whether they belong in `legacy/`, should be modularized, or can be removed.
- `check_notebook_markdown_drift.py` compares one or more notebooks against `git HEAD` and reports markdown/title/comment cells that changed, were added, or were removed. This is useful as a post-edit guardrail when we only meant to touch code cells.
