from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import Pipeline.biased_blocks as bb
import Pipeline.daily_plots as daily_plots
from Pipeline.across_sessions import (
    load_change_regions,
    plot_across_sessions_combined,
    prepare_across_sessions_data,
    subject_id_from_file,
)
from Pipeline.daily_plots import (
    load_daily_animal_data,
    load_reference_data,
    prepare_daily_animal_data,
)
from analysis.daily_merge import get_animals_for_cohort

FIGURE_CHOICES = ("across", "daily", "hist", "rt_by_abl")


def filter_daily_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    session_type = pd.to_numeric(out.get("session_type"), errors="coerce")
    short_duration = pd.to_numeric(out.get("short_duration"), errors="coerce")

    keep_non_23 = ~session_type.eq(23)
    keep_23 = pd.Series(False, index=out.index, dtype=bool)
    type23 = out[session_type.eq(23)].copy()
    if not type23.empty:
        labeled_23 = bb.add_biased_block_condition(
            type23,
            biased_session_types=(23,),
            unbiased_rt_session_types=(),
            short_duration_value=None,
        )
        keep_23 = out.index.isin(
            labeled_23.index[labeled_23["block_condition"].eq("unbiased")]
        )
    keep_23_like_rt_unbiased = keep_23 & short_duration.eq(0)
    return out[keep_non_23 | keep_23_like_rt_unbiased].copy()


def subject_file_for_animal(animal: str) -> str:
    return f"merged_{animal}.csv"


def save_requested_figures(
    *,
    out_dir: Path,
    subject_id: str,
    figures: dict[str, plt.Figure],
    dpi: int,
) -> list[Path]:
    saved_paths: list[Path] = []
    filename_map = {
        "across": f"{subject_id}_01_across_sessions.png",
        "daily": f"{subject_id}_02_daily_summary_jnd.png",
        "hist": f"{subject_id}_03_timing_histograms.png",
        "rt_by_abl": f"{subject_id}_04_rt_by_abl.png",
    }
    for key, fig in figures.items():
        path = out_dir / filename_map[key]
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def run_one_subject(
    *,
    line: str,
    cohort: str,
    animal: str,
    training_level: int | None,
    training_level_max: int | None,
    load_reference: bool,
    figures_to_make: tuple[str, ...],
    out_root: Path,
    dpi: int,
) -> list[Path]:
    subject_file = subject_file_for_animal(animal)
    df_raw, data_dir = load_daily_animal_data(
        subject_file=subject_file,
        line=line,
        cohort=cohort,
    )
    df_daily_input = filter_daily_review_rows(df_raw)

    subject_id = subject_id_from_file(subject_file)
    across_prepared = prepare_across_sessions_data(
        df_raw,
        training_level=training_level,
        training_level_max=training_level_max,
    )
    daily_prepared = prepare_daily_animal_data(
        df_daily_input,
        training_level=training_level,
        training_level_max=training_level_max,
    )
    regions = load_change_regions(data_dir=data_dir, subject_id=subject_id)
    reference = load_reference_data() if load_reference else None

    figures: dict[str, plt.Figure] = {}
    if "across" in figures_to_make:
        figures["across"] = plot_across_sessions_combined(
            across_prepared,
            regions=regions,
            figsize=(24, 14),
        )
    if "daily" in figures_to_make:
        figures["daily"] = daily_plots.plot_daily_animal_summary_with_jnd(
            daily_prepared,
            reference=reference,
            figsize=(10, 10),
        )
    if "hist" in figures_to_make:
        figures["hist"] = daily_plots.plot_timing_histograms(
            daily_prepared,
            subject_id=subject_id,
            figsize=(16, 9),
        )
    if "rt_by_abl" in figures_to_make:
        figures["rt_by_abl"] = daily_plots.plot_rt_histogram_by_abl(
            daily_prepared,
            subject_id=subject_id,
            figsize=(7, 4),
        )

    out_dir = out_root / line / cohort / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = save_requested_figures(
        out_dir=out_dir,
        subject_id=subject_id,
        figures=figures,
        dpi=dpi,
    )
    plt.close("all")
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Daily Animal Review workflow for one animal or an entire cohort "
            "and save the generated figures."
        )
    )
    parser.add_argument("--line", default="CNTNAP2")
    parser.add_argument("--cohort", default="cohort4")
    parser.add_argument(
        "--animals",
        nargs="+",
        help="Animal IDs to run, for example ASD0053 ASD0054. Omit to run the full cohort.",
    )
    parser.add_argument("--training-level", type=int, default=16)
    parser.add_argument("--training-level-max", type=int, default=None)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=FIGURE_CHOICES,
        default=FIGURE_CHOICES,
        help="Choose which logical notebook sections to run.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip loading reference data for the daily summary figure.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "daily_animal_review",
    )
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    animals = args.animals or get_animals_for_cohort(args.line, args.cohort)

    total_saved = 0
    for animal in animals:
        saved_paths = run_one_subject(
            line=args.line,
            cohort=args.cohort,
            animal=animal,
            training_level=args.training_level,
            training_level_max=args.training_level_max,
            load_reference=not args.no_reference,
            figures_to_make=tuple(args.figures),
            out_root=args.out_dir,
            dpi=args.dpi,
        )
        total_saved += len(saved_paths)
        print(f"{animal}: saved {len(saved_paths)} figure(s)")
        for path in saved_paths:
            print(f"  {path}")

    print(f"Completed {len(animals)} animal(s); saved {total_saved} figure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
