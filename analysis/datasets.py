# analysis/datasets.py
from dataclasses import dataclass
import os
import re
from typing import Iterable, Optional
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

DEFAULT_COHORT_FILES = (
    "merged_all_subjects.csv",
    "merged_ASD_cohort1.csv",
)


def dataset_key(line: str, cohort: str) -> str:
    return f"{line}:{cohort}"

def resolve_data_dir(spec: DatasetSpec) -> str:
    root = LINE_ROOTS.get((spec.line, spec.cohort), f"{spec.line}_{spec.cohort}")
    return os.path.join(spec.base_dir, root)

def load_cohort_df(spec: DatasetSpec) -> pd.DataFrame:
    data_dir = resolve_data_dir(spec)
    path = os.path.join(data_dir, spec.cohort_file)
    return pd.read_csv(path), data_dir


def discover_line_cohorts(line: str, base_dir: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(line)}_(cohort\d+)$", re.IGNORECASE)
    cohorts = []

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue

        match = pattern.match(entry)
        if match:
            cohorts.append(match.group(1))

    def cohort_key(name: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", name)
        return (int(match.group(1)) if match else 10**9, name.lower())

    return sorted(set(cohorts), key=cohort_key)


def _resolve_cohort_csv(data_dir: str, cohort_file: Optional[str]) -> str:
    candidates = [cohort_file] if cohort_file is not None else list(DEFAULT_COHORT_FILES)

    for candidate in candidates:
        path = os.path.join(data_dir, candidate)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No cohort CSV found in {data_dir!r}. Tried: {candidates!r}")


def _normalize_meta(meta: pd.DataFrame, cohort: str, line: str) -> pd.DataFrame:
    meta = meta.copy()

    if "animal" not in meta.columns:
        if "subject" in meta.columns:
            meta = meta.rename(columns={"subject": "animal"})
        else:
            raise KeyError("Metadata CSV must contain either 'animal' or 'subject'.")

    meta["animal"] = meta["animal"].astype(str).str.strip()
    meta["cohort"] = cohort
    meta["line"] = line

    keep = [c for c in ["animal", "cohort", "line", "sex", "genotype"] if c in meta.columns]
    return meta[keep].drop_duplicates(subset=["animal", "cohort"])


def _normalize_experimenter_value(value):
    if pd.isna(value):
        return value
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    if text.endswith("MV"):
        return "MV"
    if text.endswith("HY"):
        return "HY"
    return text


def load_line_across_cohorts(
    *,
    line: str,
    base_dir: str,
    cohorts: Optional[Iterable[str] | str] = "all",
    cohort_file: Optional[str] = None,
    meta_file: str = "sex_gen.csv",
    require_meta: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if cohorts is None or cohorts == "all":
        cohort_list = discover_line_cohorts(line, base_dir)
    elif isinstance(cohorts, str):
        cohort_list = [cohorts]
    else:
        cohort_list = list(cohorts)

    if not cohort_list:
        raise FileNotFoundError(f"No cohorts found for line={line!r} in {base_dir!r}.")

    df_parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []
    missing_meta: list[str] = []
    used_csvs: dict[str, str] = {}

    for cohort in cohort_list:
        spec = DatasetSpec(line=line, cohort=cohort, base_dir=base_dir, cohort_file=cohort_file or DEFAULT_COHORT_FILES[0])
        data_dir = resolve_data_dir(spec)
        cohort_csv = _resolve_cohort_csv(data_dir, cohort_file)
        used_csvs[cohort] = cohort_csv

        df = pd.read_csv(cohort_csv, low_memory=False)
        df["animal"] = df["animal"].astype(str).str.strip()
        df["cohort"] = cohort
        df["line"] = line
        if "experimenter" in df.columns:
            df["experimenter"] = df["experimenter"].map(_normalize_experimenter_value)
        df_parts.append(df)

        meta_path = os.path.join(data_dir, meta_file)
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, sep=None, engine="python", dtype=str)
            meta_parts.append(_normalize_meta(meta, cohort=cohort, line=line))
        else:
            missing_meta.append(cohort)
            if require_meta:
                raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    df_all = pd.concat(df_parts, ignore_index=True, sort=False)

    if meta_parts:
        meta_all = pd.concat(meta_parts, ignore_index=True, sort=False)
        df_all = df_all.merge(
            meta_all.drop(columns=["line"], errors="ignore"),
            on=["animal", "cohort"],
            how="left",
            suffixes=("", "_meta"),
        )
        for col in ("sex", "genotype"):
            meta_col = f"{col}_meta"
            if meta_col not in df_all.columns:
                continue
            if col in df_all.columns:
                df_all[col] = df_all[col].fillna(df_all[meta_col])
            else:
                df_all[col] = df_all[meta_col]
            df_all = df_all.drop(columns=[meta_col])
    else:
        meta_all = pd.DataFrame(columns=["animal", "cohort", "line", "sex", "genotype"])

    info = {
        "line": line,
        "cohorts": cohort_list,
        "used_csvs": used_csvs,
        "missing_meta_cohorts": missing_meta,
    }
    return df_all, meta_all, info


def load_dataset_selections(
    *,
    selections: Iterable[tuple[str, str] | DatasetSpec],
    base_dir: str,
    cohort_file: Optional[str] = None,
    meta_file: str = "sex_gen.csv",
    require_meta: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []
    selection_info: list[dict] = []
    used_csvs: dict[str, str] = {}
    missing_meta_keys: list[str] = []

    for item in selections:
        if isinstance(item, DatasetSpec):
            line = item.line
            cohort = item.cohort
            selected_cohort_file = item.cohort_file if item.cohort_file != DatasetSpec.__dataclass_fields__["cohort_file"].default else cohort_file
            selected_meta_file = item.meta_file if item.meta_file != DatasetSpec.__dataclass_fields__["meta_file"].default else meta_file
        else:
            line, cohort = item
            selected_cohort_file = cohort_file
            selected_meta_file = meta_file

        df_one, meta_one, info_one = load_line_across_cohorts(
            line=line,
            base_dir=base_dir,
            cohorts=[cohort],
            cohort_file=selected_cohort_file,
            meta_file=selected_meta_file,
            require_meta=require_meta,
        )

        key = dataset_key(line, cohort)
        df_one = df_one.copy()
        df_one["dataset_key"] = key
        df_parts.append(df_one)

        if not meta_one.empty:
            meta_one = meta_one.copy()
            meta_one["dataset_key"] = key
            meta_parts.append(meta_one)

        used_csvs[key] = next(iter(info_one["used_csvs"].values()))
        missing_meta_keys.extend(dataset_key(line, c) for c in info_one["missing_meta_cohorts"])
        selection_info.append({"line": line, "cohort": cohort, "dataset_key": key})

    if not df_parts:
        raise FileNotFoundError("No datasets were loaded from the provided selections.")

    df_all = pd.concat(df_parts, ignore_index=True, sort=False)
    meta_all = (
        pd.concat(meta_parts, ignore_index=True, sort=False)
        if meta_parts
        else pd.DataFrame(columns=["animal", "cohort", "line", "dataset_key", "sex", "genotype"])
    )

    info = {
        "selections": selection_info,
        "dataset_keys": [item["dataset_key"] for item in selection_info],
        "used_csvs": used_csvs,
        "missing_meta_dataset_keys": missing_meta_keys,
    }
    return df_all, meta_all, info
