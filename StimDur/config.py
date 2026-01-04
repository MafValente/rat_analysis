from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
import pandas as pd

Selector = Callable[[pd.DataFrame], pd.DataFrame]

@dataclass(frozen=True)
class ViewSpec:
    """Like your GroupComparison views: genotype selectors etc."""
    name: str
    selector: Selector

@dataclass(frozen=True)
class StimDurSpec:
    """One trace per stim duration."""
    name: str
    stim_dur: int
    selector: Selector

def make_stimdur_specs(stim_durs: Sequence[int], stim_dur_col: str = "stim_dur") -> List[StimDurSpec]:
    out: List[StimDurSpec] = []
    for sd in stim_durs:
        out.append(
            StimDurSpec(
                name=str(int(sd)),
                stim_dur=int(sd),
                selector=lambda d, _sd=int(sd), _col=stim_dur_col: d[d[_col] == _sd].copy(),
            )
        )
    return out

@dataclass(frozen=True)
class FilterConfig:
    training_min: int = 16
    session_min: int = 13
    drop_repeat_trials: bool = True
    session_type_values: Optional[Sequence[int]] = (2,)  # << restrict to session_type==2

@dataclass(frozen=True)
class PlotStyle:
    title_fs: int = 24
    label_fs: int = 25
    tick_fs: int = 24
    legend_fs: int = 16
    title_pad: int = 16

@dataclass(frozen=True)
class StimDurComparisonConfig:
    error_mode: str = "sem"     # "individuals" or "sem"
    skip_psy_fits: tuple[int, ...] = (50,)
    xlim_sym: tuple[float, float] = (-18.5, 18.5)
    xlim_abs: tuple[float, float] = (0, 19)
    ild_shift_for_abl50: bool = True
