from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".jupyter"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter_data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_runtime"))

from analysis.daily_merge import get_animals_for_cohort


DEFAULT_NOTEBOOK = ROOT / "notebooks" / "ASD" / "02_daily_animal_review.ipynb"


def subject_file_for_animal(animal: str) -> str:
    return f"merged_{animal}.csv"


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
        "--abl-filter",
        type=float,
        nargs="+",
        default=None,
        help="Optional ABL filter. Pass one value or several values.",
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

    total = 0
    for animal in animals:
        subject_id = animal
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
            "TRAINING_LEVEL": args.training_level,
            "TRAINING_LEVEL_MIN": args.training_level_min,
            "TRAINING_LEVEL_MAX": args.training_level_max,
            "ABL_FILTER": args.abl_filter[0] if args.abl_filter and len(args.abl_filter) == 1 else args.abl_filter,
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
