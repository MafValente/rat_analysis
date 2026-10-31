#!/usr/bin/env python3
"""Safely place corrected daily CSVs in the RatBehaviorRoom Google Drive.

The script is dry-run by default. In apply mode it never deletes or overwrites
files: it uploads to a new staging name, verifies the checksum, renames the
existing Drive CSV with ``_problematic``, then renames the staging file to the
original name. Any item that fails validation is skipped independently.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "docs"
    / "data_corrections"
    / "session_type_daily_file_changes_2026-04-24.txt"
)
RAW_ROOT_RELATIVE = Path("RatBehaviorRoom/Rats/ILDtask/ASD_MV animals")
LOCAL_TO_DRIVE_ROOT = {
    "CNTNAP2_cohort3": Path("cntnap2/Cohort 3"),
    "SHANK3_cohort1": Path("Shank3/Cohort 1"),
    "SHANK3_cohort2": Path("Shank3"),
}
MANIFEST_PATTERN = re.compile(r"^- (DataFiles/.+?/out_.+?\.csv)(?: \(.*\))?$")


@dataclass(frozen=True)
class Transfer:
    local: Path
    drive_current: Path
    drive_backup: Path
    drive_staging: Path


@dataclass
class Result:
    local: str
    drive_current: str
    drive_backup: str
    status: str
    detail: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_drive_root() -> Path:
    configured = os.environ.get("MAFALDA_RAW_DATA_DIR")
    if configured:
        root = Path(configured).expanduser()
        if root.is_dir():
            return root
        raise FileNotFoundError(f"MAFALDA_RAW_DATA_DIR does not exist: {root}")

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    for mount in sorted(cloud_storage.glob("GoogleDrive-renartlab@gmail.com*")):
        candidates = [
            mount / RAW_ROOT_RELATIVE,
            mount / "My Drive" / RAW_ROOT_RELATIVE,
            *sorted(
                (mount / ".shortcut-targets-by-id").glob(
                    f"*/{RAW_ROOT_RELATIVE.as_posix()}"
                )
            ),
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Could not locate {RAW_ROOT_RELATIVE} in Google Drive")


def load_transfers(manifest: Path, drive_root: Path) -> list[Transfer]:
    relative_paths = []
    for line in manifest.read_text().splitlines():
        match = MANIFEST_PATTERN.match(line.strip())
        if match:
            relative_paths.append(Path(match.group(1)))

    transfers = []
    for relative in sorted(set(relative_paths)):
        dataset, animal, filename = relative.parts[1:4]
        if dataset not in LOCAL_TO_DRIVE_ROOT:
            raise ValueError(f"No Drive mapping configured for {dataset}")
        date_match = re.search(r"_(\d{6})\.csv$", filename)
        if not date_match:
            raise ValueError(f"Cannot infer session date from {filename}")
        session_date = date_match.group(1)
        drive_dir = drive_root / LOCAL_TO_DRIVE_ROOT[dataset] / animal / session_date
        current = drive_dir / filename
        backup = drive_dir / f"{current.stem}_problematic{current.suffix}"
        staging = drive_dir / f"{current.stem}_corrected_upload_pending{current.suffix}"
        transfers.append(
            Transfer(
                local=PROJECT_ROOT / relative,
                drive_current=current,
                drive_backup=backup,
                drive_staging=staging,
            )
        )
    return transfers


def compare_expected_change(local: Path, drive_current: Path) -> tuple[bool, str]:
    try:
        local_df = read_csv_strings(local)
        drive_df = read_csv_strings(drive_current)
    except Exception as exc:
        return False, f"CSV read failed: {exc}"

    if local_df.shape != drive_df.shape:
        return False, f"shape differs: local={local_df.shape}, Drive={drive_df.shape}"
    if list(local_df.columns) != list(drive_df.columns):
        return False, "column names or order differ"
    if "session_type" not in local_df.columns:
        return False, "session_type column is missing"

    differing_columns = [
        column
        for column in local_df.columns
        if not local_df[column].equals(drive_df[column])
    ]
    if differing_columns != ["session_type"]:
        return False, f"unexpected differing columns: {differing_columns}"

    changed = local_df["session_type"] != drive_df["session_type"]
    if not changed.any():
        return False, "no session_type values differ"

    old_numeric = pd.to_numeric(
        drive_df.loc[changed, "session_type"], errors="coerce"
    )
    new_numeric = pd.to_numeric(
        local_df.loc[changed, "session_type"], errors="coerce"
    )
    if old_numeric.isna().any() or new_numeric.isna().any():
        return False, "changed session_type values are not numeric"
    directions = set(zip(old_numeric.astype(int), new_numeric.astype(int)))
    if directions not in ({(1, 2)}, {(2, 1)}):
        return False, f"unexpected or mixed session_type changes: {sorted(directions)}"

    old_value, new_value = next(iter(directions))
    return True, (
        f"validated {len(local_df)} rows; {int(changed.sum())} rows change "
        f"{old_value} -> {new_value}"
    )


def read_csv_strings(path: Path) -> pd.DataFrame:
    """Read a local or cloud-backed CSV, retrying transient Drive read errors."""
    last_error = None
    for attempt in range(3):
        try:
            content = path.read_bytes()
            return pd.read_csv(
                io.BytesIO(content),
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
        except (OSError, pd.errors.ParserError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1)
    raise last_error


def preflight(transfer: Transfer) -> tuple[bool, str]:
    if not transfer.local.is_file():
        return False, "local corrected file is missing"
    if not transfer.drive_current.is_file():
        return False, "current Drive file is missing"
    if transfer.drive_backup.exists():
        return False, "_problematic destination already exists"
    if transfer.drive_staging.exists():
        return False, "staging destination already exists"
    if transfer.local.name != transfer.drive_current.name:
        return False, "local and Drive filenames differ"
    return compare_expected_change(transfer.local, transfer.drive_current)


def copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as dest_handle:
        shutil.copyfileobj(source_handle, dest_handle, length=1024 * 1024)


def apply_one(transfer: Transfer) -> Result:
    ok, detail = preflight(transfer)
    if not ok:
        return make_result(transfer, "SKIPPED", detail)

    try:
        copy_exclusive(transfer.local, transfer.drive_staging)
        if sha256(transfer.local) != sha256(transfer.drive_staging):
            return make_result(transfer, "SKIPPED", "staging checksum mismatch")

        # Both destinations were checked absent. These are renames, not deletes.
        transfer.drive_current.rename(transfer.drive_backup)
        try:
            transfer.drive_staging.rename(transfer.drive_current)
        except Exception:
            if not transfer.drive_current.exists() and transfer.drive_backup.exists():
                transfer.drive_backup.rename(transfer.drive_current)
            raise

        if sha256(transfer.local) != sha256(transfer.drive_current):
            return make_result(transfer, "ERROR", "final checksum mismatch")
        return make_result(transfer, "UPDATED", detail)
    except Exception as exc:
        return make_result(transfer, "ERROR", f"apply failed: {exc}")


def make_result(transfer: Transfer, status: str, detail: str) -> Result:
    return Result(
        local=str(transfer.local.relative_to(PROJECT_ROOT)),
        drive_current=str(transfer.drive_current),
        drive_backup=str(transfer.drive_backup),
        status=status,
        detail=detail,
    )


def write_report(results: list[Result], mode: str) -> Path:
    report_dir = PROJECT_ROOT / "outputs" / "data_corrections"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"drive_daily_csv_sync_{mode}_{timestamp}.json"
    report_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n"
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-no-delete",
        action="store_true",
        help="Required with --apply; confirms the rename-and-copy workflow.",
    )
    args = parser.parse_args()
    if args.apply and not args.confirm_no_delete:
        parser.error("--apply requires --confirm-no-delete")

    drive_root = find_drive_root()
    transfers = load_transfers(args.manifest, drive_root)
    print(f"Drive root: {drive_root}")
    print(f"Manifest entries: {len(transfers)}")

    results = []
    for index, transfer in enumerate(transfers, start=1):
        if args.apply:
            result = apply_one(transfer)
        else:
            ok, detail = preflight(transfer)
            result = make_result(transfer, "READY" if ok else "SKIPPED", detail)
        results.append(result)
        print(
            f"[{index:03d}/{len(transfers):03d}] {result.status}: "
            f"{result.local} — {result.detail}",
            flush=True,
        )

    mode = "apply" if args.apply else "preflight"
    report_path = write_report(results, mode)
    counts = pd.Series([result.status for result in results]).value_counts().to_dict()
    print(f"Summary: {counts}")
    print(f"Report: {report_path}")
    return 1 if counts.get("ERROR", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
