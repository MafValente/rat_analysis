from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".jupyter"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter_data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_runtime"))

from analysis.daily_merge import get_animals_for_cohort, get_base_dir


DEFAULT_NOTEBOOK = ROOT / "notebooks" / "ASD" / "02_daily_animal_review.ipynb"


def subject_file_for_animal(animal: str) -> str:
    return f"merged_{animal}.csv"


def latest_training_level_for_animal(line: str, cohort: str, animal: str) -> int:
    path = Path(get_base_dir(line, cohort)) / subject_file_for_animal(animal)
    if not path.exists():
        raise FileNotFoundError(f"Subject file not found: {path}")

    try:
        df = pd.read_csv(path, usecols=["training_level"])
    except ValueError as exc:
        raise ValueError(f"{path} does not contain a training_level column") from exc

    levels = pd.to_numeric(df["training_level"], errors="coerce").dropna()
    if levels.empty:
        raise ValueError(f"No numeric training_level values found in {path}")

    return int(levels.max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Daily Animal Review notebook once per animal in a cohort "
            "using papermill, saving the figures and optionally the executed notebooks."
        )
    )
    parser.add_argument("--line", default="CNTNAP2")
    parser.add_argument("--cohort", default="cohort4")
    parser.add_argument(
        "--animals",
        nargs="+",
        help="Animal IDs to run, for example ASD0053 ASD0054. Omit to run the whole cohort.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="Notebook to execute.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Kernel name to use for papermill execution. Default: python3",
    )
    parser.add_argument("--training-level", type=int, default=None)
    parser.add_argument("--training-level-min", type=int, default=None)
    parser.add_argument("--training-level-max", type=int, default=None)
    parser.add_argument(
        "--latest-training-level",
        action="store_true",
        help=(
            "For each animal, use its highest numeric training_level and analyze "
            "all sessions at that level."
        ),
    )
    parser.add_argument(
        "--abl-filter",
        type=float,
        nargs="+",
        default=None,
        help="Optional ABL filter. Pass one value or several values.",
    )
    parser.add_argument(
        "--sound-ramp-filter",
        type=float,
        nargs="+",
        default=None,
        help="Optional sound ramp filter in seconds. Pass one value or several values.",
    )
    parser.add_argument(
        "--sound-ramp-exclude",
        type=float,
        nargs="+",
        default=None,
        help="Optional sound ramp values to exclude, in seconds.",
    )
    parser.add_argument("--figure-dpi", type=int, default=250)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "daily_animal_review",
        help="Root folder for saved figures.",
    )
    parser.add_argument(
        "--executed-notebook-dir",
        type=Path,
        default=ROOT / "outputs" / "executed_notebooks" / "daily_animal_review",
        help="Where to save the executed notebook for each animal.",
    )
    parser.add_argument(
        "--no-save-executed",
        action="store_true",
        help="Execute the notebook without keeping the executed .ipynb outputs.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip loading reference data inside the notebook.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import papermill as pm
    except ImportError as exc:
        raise SystemExit(
            "papermill is not installed in this environment. "
            "Install it in the notebook kernel environment to use notebook batch execution."
        ) from exc

    args = parse_args()
    notebook_path = args.notebook.resolve()
    animals = args.animals or get_animals_for_cohort(args.line, args.cohort)

    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    if not animals:
        raise SystemExit(f"No animals found for {args.line} {args.cohort}")

    if args.latest_training_level and any(
        value is not None
        for value in (args.training_level, args.training_level_min, args.training_level_max)
    ):
        raise SystemExit(
            "--latest-training-level cannot be combined with --training-level, "
            "--training-level-min, or --training-level-max."
        )

    total = 0
    for animal in animals:
        subject_id = animal
        training_level = (
            latest_training_level_for_animal(args.line, args.cohort, subject_id)
            if args.latest_training_level
            else args.training_level
        )
        figure_dir = (args.out_dir / args.line / args.cohort / subject_id).resolve()
        figure_dir.mkdir(parents=True, exist_ok=True)

        temp_output_path: Path | None = None
        if args.no_save_executed:
            with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
                temp_output_path = Path(tmp.name)
            output_notebook = temp_output_path
        else:
            output_notebook = (
                args.executed_notebook_dir / args.line / args.cohort / f"{subject_id}.ipynb"
            ).resolve()
            output_notebook.parent.mkdir(parents=True, exist_ok=True)

        parameters = {
            "LINE": args.line,
            "COHORT": args.cohort,
            "SUBJECT_FILE": subject_file_for_animal(subject_id),
            "TRAINING_LEVEL": training_level,
            "TRAINING_LEVEL_MIN": args.training_level_min,
            "TRAINING_LEVEL_MAX": args.training_level_max,
            "USE_LATEST_TRAINING_LEVEL": args.latest_training_level,
            "ABL_FILTER": args.abl_filter[0] if args.abl_filter and len(args.abl_filter) == 1 else args.abl_filter,
            "SOUND_RAMP_FILTER": (
                args.sound_ramp_filter[0]
                if args.sound_ramp_filter and len(args.sound_ramp_filter) == 1
                else args.sound_ramp_filter
            ),
            "SOUND_RAMP_EXCLUDE": (
                args.sound_ramp_exclude[0]
                if args.sound_ramp_exclude and len(args.sound_ramp_exclude) == 1
                else args.sound_ramp_exclude
            ),
            "LOAD_REFERENCE": not args.no_reference,
            "SAVE_FIGURES": True,
            "OUTPUT_DIR": str(figure_dir),
            "FIGURE_DPI": args.figure_dpi,
        }

        print(f"Running notebook for {subject_id}")
        print(
            "  filters:",
            {
                "TRAINING_LEVEL": parameters["TRAINING_LEVEL"],
                "TRAINING_LEVEL_MIN": parameters["TRAINING_LEVEL_MIN"],
                "TRAINING_LEVEL_MAX": parameters["TRAINING_LEVEL_MAX"],
                "ABL_FILTER": parameters["ABL_FILTER"],
                "SOUND_RAMP_FILTER": parameters["SOUND_RAMP_FILTER"],
                "SOUND_RAMP_EXCLUDE": parameters["SOUND_RAMP_EXCLUDE"],
            },
        )
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_notebook),
            parameters=parameters,
            cwd=str(ROOT),
            kernel_name=args.kernel,
            request_save_on_cell_execute=not args.no_save_executed,
            progress_bar=False,
        )
        print(f"  figures: {figure_dir}")
        figure_files = sorted(figure_dir.glob("*.png"))
        if not figure_files:
            print(f"  WARNING: no PNG figures were saved to {figure_dir}")
        if not args.no_save_executed:
            print(f"  notebook: {output_notebook}")
        elif temp_output_path and temp_output_path.exists():
            temp_output_path.unlink()
        total += 1

    print(f"Completed {total} animal(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
