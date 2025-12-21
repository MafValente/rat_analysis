# analysis/session_profiles.py
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass(frozen=True)
class SessionProfile:
    name: str
    # If you later want “one trace per stim_duration”, set facet_col="stim_duration"
    facet_col: Optional[str] = None
    facet_values: Optional[Sequence] = None

NORMAL = SessionProfile(name="normal", facet_col=None)

# Example for your future session type:
# STIM_SAMPLED = SessionProfile(name="stim_sampled", facet_col="stim_duration", facet_values=[...])

normal = SessionProfile(name="normal")
short_dur = SessionProfile(name="short_dur", facet_col="stim_dur")
