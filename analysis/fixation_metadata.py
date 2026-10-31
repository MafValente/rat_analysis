"""Apply YAML fixation configuration to local merged Stakes data."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis import daily_merge as DailyMerge


def _session_date(source_file: str) -> str | None:
    match = re.search(r"_(\d{6})(?:_problematic)?\.csv$", str(source_file))
    return match.group(1) if match else None


def _read_yaml_settings(config_dir: Path):
    files = sorted(config_dir.glob("*.yml")) + sorted(config_dir.glob("*.yaml"))
    if not files:
        return None, "yaml_missing"

    constants = set()
    exponentials = set()
    for path in files:
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as error:
            return None, f"yaml_read_error: {error}"

        for setting in config.get("fixation_time", {}).get("task", []) or []:
            if not isinstance(setting, dict):
                continue
            distribution = str(setting.get("distribution", "")).lower()
            if distribution == "constant" and setting.get("value") is not None:
                constants.add(float(setting["value"]))
            elif distribution == "exponential" and setting.get("mean") is not None:
                exponentials.add(float(setting["mean"]))

    if len(constants) != 1 or len(exponentials) != 1:
        return {
            "constant_values": sorted(constants),
            "exponential_values": sorted(exponentials),
            "yaml_files": [str(path) for path in files],
        }, "yaml_ambiguous_or_incomplete"

    return {
        "fixation_constant": next(iter(constants)),
        "ft_exp_mean": next(iter(exponentials)),
        "yaml_files": [str(path) for path in files],
    }, "ok"


def _value_text(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    return json.dumps(sorted(float(value) for value in values))


def _apply_to_dataframe(df, animal, drive_animal_dir, audit_rows):
    out = df.copy()
    if "fixation_constant" not in out.columns:
        out["fixation_constant"] = pd.NA

    if "source_file" not in out.columns:
        audit_rows.append({"animal": animal, "status": "missing_source_file_column"})
        return out

    for source_file in out["source_file"].dropna().astype(str).unique():
        mask = out["source_file"].astype(str).eq(source_file)
        date = _session_date(source_file)
        base_audit = {
            "animal": animal,
            "source_file": source_file,
            "session_date": date or "",
            "old_ft_exp_mean": _value_text(out.loc[mask, "ft_exp_mean"]),
            "old_fixation_constant": _value_text(out.loc[mask, "fixation_constant"]),
        }

        if date is None:
            audit_rows.append({**base_audit, "status": "source_date_unresolved"})
            continue

        settings, status = _read_yaml_settings(drive_animal_dir / date / "config")
        if status != "ok":
            audit_rows.append({
                **base_audit,
                "status": status,
                "yaml_settings": json.dumps(settings, default=str),
            })
            continue

        old_exp = pd.to_numeric(out.loc[mask, "ft_exp_mean"], errors="coerce")
        old_constant = pd.to_numeric(
            out.loc[mask, "fixation_constant"], errors="coerce"
        )
        already_correct = (
            old_exp.notna().all()
            and np.isclose(old_exp, settings["ft_exp_mean"]).all()
            and old_constant.notna().all()
            and np.isclose(old_constant, settings["fixation_constant"]).all()
        )

        # Only configuration columns are replaced.
        out.loc[mask, "ft_exp_mean"] = settings["ft_exp_mean"]
        out.loc[mask, "fixation_constant"] = settings["fixation_constant"]
        audit_rows.append({
            **base_audit,
            "status": "already_matching" if already_correct else "corrected",
            "new_ft_exp_mean": settings["ft_exp_mean"],
            "new_fixation_constant": settings["fixation_constant"],
            "yaml_files": json.dumps(settings["yaml_files"]),
        })

    return out


def _replace_local_csv(path, animal, drive_animal_dir, audit_rows):
    if not path.exists():
        return
    corrected = _apply_to_dataframe(
        pd.read_csv(path), animal, drive_animal_dir, audit_rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    corrected.to_csv(temporary, index=False)
    temporary.replace(path)


def apply_yaml_fixation_metadata(
    line="Stakes",
    cohort="cohort2",
    exclude_animals=None,
    audit_path=None,
):
    """Correct local merged fixation configuration from session YAML files.

    Raw daily CSVs and Google Drive are read-only. The function updates local
    per-animal and cohort merged CSVs, adding ``fixation_constant`` if needed,
    and returns the local audit-report path.
    """
    excluded = {str(animal) for animal in (exclude_animals or set())}
    base_dir = Path(DailyMerge.get_base_dir(line, cohort))
    animals = [
        str(animal)
        for animal in DailyMerge.get_animals_for_cohort(line, cohort)
        if str(animal) not in excluded
    ]
    audit_rows = []
    drive_dirs = {}

    for animal in animals:
        drive_dirs[animal] = Path(
            DailyMerge.get_raw_animal_dir(line, cohort, animal)
        )
        _replace_local_csv(
            base_dir / f"merged_{animal}.csv",
            animal,
            drive_dirs[animal],
            audit_rows,
        )

    cohort_path = base_dir / "merged_all_subjects.csv"
    if cohort_path.exists():
        cohort_df = pd.read_csv(cohort_path)
        if "animal" in cohort_df.columns:
            if "fixation_constant" not in cohort_df.columns:
                cohort_df["fixation_constant"] = pd.NA
            for animal in animals:
                mask = cohort_df["animal"].astype(str).eq(animal)
                if mask.any():
                    cohort_df.loc[mask, :] = _apply_to_dataframe(
                        cohort_df.loc[mask].copy(),
                        animal,
                        drive_dirs[animal],
                        audit_rows,
                    )
            temporary = cohort_path.with_suffix(cohort_path.suffix + ".tmp")
            cohort_df.to_csv(temporary, index=False)
            temporary.replace(cohort_path)

    if audit_path is None:
        audit_path = base_dir.parent.parent / "outputs" / "stakes_fixation_yaml_audit.csv"
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    return audit_path
