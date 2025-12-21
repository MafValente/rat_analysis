# analysis/datasets.py
from dataclasses import dataclass
import os
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    line: str          # "CNTNAP2"
    cohort: str        # "cohort2"
    base_dir: str      # "/Users/.../DataFiles"
    cohort_file: str = "merged_all_subjects.csv"
    meta_file: str = "sex_gen.csv"

LINE_ROOTS = {
    ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
    ("SHANK3",  "cohort1"): "SHANK3_cohort1",
}

def resolve_data_dir(spec: DatasetSpec) -> str:
    root = LINE_ROOTS[(spec.line, spec.cohort)]
    return os.path.join(spec.base_dir, root)

def load_cohort_df(spec: DatasetSpec) -> pd.DataFrame:
    data_dir = resolve_data_dir(spec)
    path = os.path.join(data_dir, spec.cohort_file)
    return pd.read_csv(path), data_dir
