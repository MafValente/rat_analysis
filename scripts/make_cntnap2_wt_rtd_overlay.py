from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Juananalysis.kernel_regression import build_hierarchical_data_full

DEFAULT_OUTPUT = ROOT / "DataFiles" / "Old Data" / "ILD_task" / "cntnap2_wt_cohort2_cohort4_kernel_timing_overlay.pkl"


def epanechnikov_kernel(z: np.ndarray) -> np.ndarray:
    return 0.75 * (1 - z**2) * (np.abs(z) <= 1)


def load_meta(data_dir: Path, cohort: str) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "sex_gen.csv", sep=None, engine="python", dtype=str)
    meta = meta.rename(columns={"subject": "animal"})
    meta["animal"] = meta["animal"].astype(str).str.strip()
    meta["genotype"] = meta["genotype"].astype(str).str.strip().str.lower()
    meta["cohort"] = cohort
    return meta[["animal", "cohort", "sex", "genotype"]]


def load_filtered_trials(base_dir: Path, cohorts: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    used_csvs = {}

    for cohort in cohorts:
        data_dir = base_dir / f"CNTNAP2_{cohort}"
        cohort_csv = data_dir / "merged_all_subjects.csv"
        used_csvs[cohort] = str(cohort_csv.relative_to(ROOT))

        df = pd.read_csv(cohort_csv, low_memory=False)
        df["animal"] = df["animal"].astype(str).str.strip()
        df["cohort"] = cohort

        meta = load_meta(data_dir, cohort)
        df = df.merge(meta[["animal", "cohort", "genotype", "sex"]], on=["animal", "cohort"], how="left")

        df["session_type_num"] = pd.to_numeric(df["session_type"], errors="coerce")
        df["short_duration_num"] = pd.to_numeric(df["short_duration"], errors="coerce")
        df["RT"] = pd.to_numeric(df["timed_rt"], errors="coerce")
        df["MT"] = pd.to_numeric(df.get("timed_mt"), errors="coerce")
        df["ILD"] = pd.to_numeric(df.get("ILD"), errors="coerce")
        df["ABL"] = pd.to_numeric(df.get("ABL"), errors="coerce")
        df["session"] = pd.to_numeric(df.get("session"), errors="coerce")
        df["success_num"] = pd.to_numeric(df.get("success"), errors="coerce")
        df["genotype"] = df["genotype"].astype("string").str.lower()
        df["Out"] = np.where(
            df["success_num"].eq(1),
            1.0,
            np.where(df["success_num"].eq(-1), 0.0, np.nan),
        )
        df["absILD"] = df["ILD"].abs()
        df["Easy"] = (df["absILD"] >= 6).astype(float)

        keep = (
            df["genotype"].eq("wt")
            & df["session_type_num"].isin([1, 2])
            & df["short_duration_num"].eq(0)
            & df["RT"].notna()
        )
        frames.append(df.loc[keep].copy())

    filtered = pd.concat(frames, ignore_index=True, sort=False)
    return filtered, used_csvs


def build_hierarchical_rtd_data(df_in: pd.DataFrame, group_col: str, group_value) -> list[list[list[np.ndarray]]]:
    mask = df_in[group_col].notna() & (df_in[group_col] == group_value) & df_in["RT"].notna()
    sub = df_in.loc[mask].copy()
    if sub.empty:
        return []

    animals_data = []
    for animal in sub["animal"].dropna().unique():
        sub_a = sub[sub["animal"] == animal]
        sessions_data = []
        for session in sub_a["session"].dropna().unique():
            sub_s = sub_a[sub_a["session"] == session].copy()
            stim_key = sub_s["ILD"].where(sub_s["ILD"].notna(), "__nan__")
            sub_s = sub_s.assign(_stim_key=stim_key)

            stimuli_data = []
            for stim in sub_s["_stim_key"].unique():
                rt_arr = sub_s.loc[sub_s["_stim_key"] == stim, "RT"].to_numpy(dtype=float)
                rt_arr = rt_arr[np.isfinite(rt_arr)]
                if rt_arr.size:
                    stimuli_data.append(rt_arr)

            if stimuli_data:
                sessions_data.append(stimuli_data)

        if sessions_data:
            animals_data.append(sessions_data)
    return animals_data


def summarize_bootstrap(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(arr, axis=0)
    q_hi = np.nanquantile(arr, 0.975, axis=0)
    q_lo = np.nanquantile(arr, 0.025, axis=0)
    return median, q_hi - median, median - q_lo


def optimized_hierarchical_bootstrap_tcm(
    data_nested,
    x_grid: np.ndarray,
    bandwidth: float,
    bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same animal -> session -> ILD -> trial bootstrap as notebook 04, optimized
    for plotting on a fixed RT grid by dropping RTs outside kernel support before
    building the kernel matrix.
    """
    rng = np.random.default_rng(seed)
    n_grid = len(x_grid)
    tcm_curves = np.full((bootstraps, n_grid), np.nan, dtype=float)
    n_animals = len(data_nested)

    if n_animals == 0:
        raise ValueError("No animals available for tachometric bootstrap after filtering.")

    for b in range(bootstraps):
        rts_all = []
        outs_all = []
        animal_indices = rng.integers(0, n_animals, size=n_animals)

        for animal_idx in animal_indices:
            sessions = data_nested[animal_idx]
            if not sessions:
                continue
            session_indices = rng.integers(0, len(sessions), size=len(sessions))

            for session_idx in session_indices:
                stimuli = sessions[session_idx]
                if not stimuli:
                    continue
                stim_indices = rng.integers(0, len(stimuli), size=len(stimuli))

                for stim_idx in stim_indices:
                    rt_arr, out_arr, _mt_arr = stimuli[stim_idx]
                    if len(rt_arr) == 0:
                        continue
                    trial_idx = rng.integers(0, len(rt_arr), size=len(rt_arr))
                    rts_all.append(rt_arr[trial_idx])
                    outs_all.append(out_arr[trial_idx])

        if not rts_all:
            continue

        rt = np.concatenate(rts_all).astype(float)
        out = np.concatenate(outs_all).astype(float)
        support = (rt >= x_grid[0] - bandwidth) & (rt <= x_grid[-1] + bandwidth)
        if not support.any():
            continue

        rt_support = rt[support]
        out_support = out[support]
        z = (x_grid[None, :] - rt_support[:, None]) / bandwidth
        weights = epanechnikov_kernel(z)
        den = weights.sum(axis=0) + 1e-12
        num = (weights * out_support[:, None]).sum(axis=0)
        tcm_curves[b, :] = num / den

    return summarize_bootstrap(tcm_curves)


def hierarchical_bootstrap_rtd(data_nested: list[list[list[np.ndarray]]], x_grid: np.ndarray, bandwidth: float, bootstraps: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Notebook-04-style RTD bootstrap:
    animals -> sessions -> ILDs -> trials, then kernel density over RT.

    RTs outside the plotted support are excluded only from the kernel-weight
    matrix, but kept in the density denominator.
    """
    rng = np.random.default_rng(seed)
    n_grid = len(x_grid)
    densities = np.full((bootstraps, n_grid), np.nan, dtype=float)
    n_animals = len(data_nested)

    if n_animals == 0:
        raise ValueError("No animals available for RTD bootstrap after filtering.")

    for b in range(bootstraps):
        sampled_rts = []
        animal_indices = rng.integers(0, n_animals, size=n_animals)

        for animal_idx in animal_indices:
            sessions = data_nested[animal_idx]
            if not sessions:
                continue
            session_indices = rng.integers(0, len(sessions), size=len(sessions))

            for session_idx in session_indices:
                stimuli = sessions[session_idx]
                if not stimuli:
                    continue
                stim_indices = rng.integers(0, len(stimuli), size=len(stimuli))

                for stim_idx in stim_indices:
                    rt_arr = stimuli[stim_idx]
                    if rt_arr.size == 0:
                        continue
                    trial_idx = rng.integers(0, rt_arr.size, size=rt_arr.size)
                    sampled_rts.append(rt_arr[trial_idx])

        if not sampled_rts:
            continue

        x = np.concatenate(sampled_rts)
        n_total = len(x)
        x_support = x[(x >= x_grid[0] - bandwidth) & (x <= x_grid[-1] + bandwidth)]
        if x_support.size == 0:
            densities[b, :] = 1e-12
            continue

        z = (x_grid[None, :] - x_support[:, None]) / bandwidth
        weights = epanechnikov_kernel(z)
        densities[b, :] = weights.sum(axis=0) / (n_total * bandwidth) + 1e-12

    return summarize_bootstrap(densities)


def scale_to_peak(bundle: tuple[np.ndarray, np.ndarray, np.ndarray], peak_value: float = 0.5) -> dict:
    median, up, dn = bundle
    peak = float(np.nanmax(median))
    scale = float(peak_value / peak) if np.isfinite(peak) and peak > 0 else np.nan
    return {
        "median": median * scale,
        "up": up * scale,
        "dn": dn * scale,
        "scale": scale,
        "original_peak": peak,
        "y_unit": f"scaled to median peak {peak_value:g}",
    }


def make_overlay(
    *,
    base_dir: Path,
    cohorts: list[str],
    x_min: float,
    x_max: float,
    n_grid: int,
    bandwidth: float,
    bootstraps: int,
    seed: int,
) -> dict:
    trials, used_csvs = load_filtered_trials(base_dir, cohorts)
    x_grid = np.linspace(x_min, x_max, n_grid)

    trials = trials.copy()
    trials["analysis_group"] = "cntnap2_wt"

    valid_for_tcm = trials[
        trials["RT"].notna()
        & trials["Out"].notna()
        & trials["MT"].notna()
        & trials["ILD"].notna()
        & trials["session"].notna()
    ].copy()

    tcm_nested = build_hierarchical_data_full(
        valid_for_tcm,
        group_col="analysis_group",
        group_value="cntnap2_wt",
        easy_value=None,
        abl_value=None,
    )
    tcm = optimized_hierarchical_bootstrap_tcm(
        tcm_nested,
        x_grid,
        bandwidth,
        bootstraps,
        seed,
    )

    rtd_nested = build_hierarchical_rtd_data(
        trials,
        group_col="analysis_group",
        group_value="cntnap2_wt",
    )
    rtd = hierarchical_bootstrap_rtd(
        rtd_nested,
        x_grid,
        bandwidth,
        bootstraps,
        seed + 1,
    )

    animal_counts = (
        trials.groupby(["cohort", "animal", "sex", "genotype"], dropna=False)
        .agg(
            n_rtd_trials=("RT", "size"),
            n_tcm_trials=("Out", lambda s: int(s.notna().sum())),
            n_sessions=("session", "nunique"),
            rt_median=("RT", "median"),
        )
        .reset_index()
        .sort_values(["cohort", "animal"])
    )

    session_type_counts = (
        trials.groupby(["cohort", "session_type_num"], dropna=False)
        .size()
        .reset_index(name="n_rtd_trials")
        .sort_values(["cohort", "session_type_num"])
    )

    return {
        "kind": "kernel_timing_overlay",
        "name": "CNTNAP2 WT cohorts 2+4 kernel timing overlay",
        "description": (
            "Notebook-04-style kernel timing overlay for WT CNTNAP2 cohorts 2 and 4. "
            "Filters: session_type in {1, 2}, short_duration == 0. "
            "Tachometric uses valid correct/error trials with finite RT and MT. "
            "RTD uses all filtered trials with finite timed_rt."
        ),
        "x": x_grid,
        "x_unit": "s",
        "tcm": {
            "median": tcm[0],
            "up": tcm[1],
            "dn": tcm[2],
            "y_unit": "P(correct | RT)",
        },
        "rtd": {
            "median": rtd[0],
            "up": rtd[1],
            "dn": rtd[2],
            "y_unit": "density",
        },
        "rtd_peak_0p5": scale_to_peak(rtd, peak_value=0.5),
        "filters": {
            "line": "CNTNAP2",
            "cohorts": cohorts,
            "genotype": "wt",
            "session_type": [1, 2],
            "short_duration": 0,
            "rt_column": "timed_rt",
            "tcm_success_filter": [-1, 1],
            "rtd_success_filter": None,
        },
        "bootstrap": {
            "method": "notebook_04 kernel timing: hierarchical animal -> session -> ILD -> trial bootstrap",
            "bandwidth": bandwidth,
            "bootstraps": bootstraps,
            "seed": seed,
            "x_min": x_min,
            "x_max": x_max,
            "n_grid": n_grid,
        },
        "summary": {
            "n_rtd_trials": int(len(trials)),
            "n_tcm_trials": int(len(valid_for_tcm)),
            "n_animals": int(trials["animal"].nunique()),
            "n_sessions": int(trials[["cohort", "animal", "session"]].drop_duplicates().shape[0]),
            "animals": sorted(trials["animal"].dropna().unique().tolist()),
            "used_csvs": used_csvs,
            "animal_counts": animal_counts,
            "session_type_counts": session_type_counts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CNTNAP2 WT notebook-04-style tachometric + RTD overlay pickle.")
    parser.add_argument("--base-dir", type=Path, default=ROOT / "DataFiles")
    parser.add_argument("--cohorts", nargs="+", default=["cohort2", "cohort4"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--x-min", type=float, default=-0.3)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--n-grid", type=int, default=260)
    parser.add_argument("--bandwidth", type=float, default=0.015)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    overlay = make_overlay(
        base_dir=args.base_dir,
        cohorts=args.cohorts,
        x_min=args.x_min,
        x_max=args.x_max,
        n_grid=args.n_grid,
        bandwidth=args.bandwidth,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(overlay, f)

    summary = overlay["summary"]
    print(f"Saved {args.output}")
    print(f"RTD trials: {summary['n_rtd_trials']}")
    print(f"Tachometric trials: {summary['n_tcm_trials']}")
    print(f"Animals: {summary['n_animals']} {summary['animals']}")
    print(f"Sessions: {summary['n_sessions']}")
    print("Session type counts:")
    print(summary["session_type_counts"].to_string(index=False))


if __name__ == "__main__":
    main()
