"""Assemble saved daily review plots into a cohort PDF."""
from __future__ import annotations

import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.image as mpimg


def save_cohort_report(
    path: Path,
    pages: list[tuple[str, list[Path]]],
    *,
    line: str,
    cohort: str,
) -> Path:
    """Write one landscape page per animal, using only this run's saved plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.infodict()["Title"] = f"Daily animal review — {line} / {cohort}"
        for animal, plots in pages:
            rows = max(1, math.ceil(len(plots) / 2))
            fig = Figure(figsize=(24, 18), facecolor="white")
            FigureCanvasAgg(fig)
            fig.suptitle(f"{line} / {cohort} — {animal}", fontsize=24, y=0.98)
            axes = fig.subplots(rows, 2, squeeze=False).flat
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94,
                                wspace=0.03, hspace=0.04)
            for index, ax in enumerate(axes):
                ax.set_axis_off()
                if index < len(plots):
                    ax.imshow(mpimg.imread(plots[index]))
            if not plots:
                fig.text(0.5, 0.5, "No plots generated for this animal.", ha="center")
            pdf.savefig(fig, dpi=200)
            fig.clear()
    return path
