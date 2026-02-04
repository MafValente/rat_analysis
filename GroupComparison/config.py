# analysis/groupcomparison/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Any
import pandas as pd


Selector = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ViewSpec:
    name: str
    selector: Selector


@dataclass(frozen=True)
class FilterConfig:
    training_min: int = 16
    session_min: int = 13
    drop_repeat_trials: bool = True

    # NEW: optional restriction by session_type / stim_dur
    session_type_values: Optional[Sequence[int]] = None   # e.g. [1]
    stim_dur_values: Optional[Sequence[int]] = None       # e.g. [6000]
    sessiontype_or_stimdur: str = "or"  # "or" or "and"


@dataclass(frozen=True)
class PlotStyle:
    title_fs: int = 24
    label_fs: int = 25
    tick_fs: int = 24
    legend_fs: int = 16
    title_pad: int = 16


@dataclass(frozen=True)
class GroupComparisonConfig:
    error_mode: str = "individuals"   # "sem" or "individuals"
    skip_psy_fits: Tuple[int, ...] = (50,)
    xlim_sym: Tuple[float, float] = (-18.5, 18.5)
    xlim_abs: Tuple[float, float] = (0, 19)
    ylim_rt: Tuple[float, float] = (0, 0.35)
    ylim_mt: Tuple[float, float] = (0, 0.55)
    ild_shift_for_abl50: bool = True


@dataclass(frozen=True)
class OverlaySpec:
    """Old neurotypical overlays for RT + psychometric (from make_fig1 pickles)."""
    makefig1_data: Optional[dict] = None
    makefig1_chrono: Optional[dict] = None
    overlay_color: str = "black"


@dataclass(frozen=True)
class JNDOverlaySpec:
    """
    Old JND overlay data (your jnd_analysis_data.pkl).
    Expected keys like your old file:
      - "ABLS"
      - "jnds" (dict: abl -> {animal: jnd})
      - "animals_with_mean"
    """
    old_jnd_data: Optional[dict] = None

    # How much to shift old vs new along x (like your example)
    old_x_shift: float = -0.5
    new_x_shift: float = +0.5

    # Optional hardcoded ABL->color (matches your example)
    # If None, it will auto-assign using matplotlib default cycle.
    abl_color_map: Optional[Dict[int, str]] = None
