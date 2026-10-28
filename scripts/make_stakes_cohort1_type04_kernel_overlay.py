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

DEFAULT_INPUT = ROOT / "DataFiles" / "Stakes_cohort1" / "out_type_04.csv"
DEFAULT_OUTPUT = ROOT / "DataFiles" / "Old Data" / "ILD_task" / "stakes_cohort1_out_type_04_kernel_timing_overlay.pkl"


def epanechnikov_kernel(z: np.ndarray) -> np.ndarray:
    return 0.75 * (1 - z**2) * (np.abs(z) <= 1)


def summarize_bootstrap(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(arr, axis=0)
    q_hi = np.nanquantile(arr, 0.975, axis=0)
    q_lo = np.nanquantile(arr, 0.025, axis=0)
    return median, q_hi - median, median - q_lo


def load_stakes_type04(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["animal"] = df["animal"].astype(str).str.strip()
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["MT"] = pd.to_numeric(df["MT"], errors="coerce")
    df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
    df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
    df["session"] = pd.to_numeric(df["session"], errors="coerce")
    df["session_type_num"] = pd.to_numeric(df["session_type"], errors="coerce")
    df["success_num"] = pd.to_numeric(df["success"], errors="coerce")
    df["Out"] = np.where(
        df["success_num"].eq(1),
        1.0,
        np.where(df["success_num"].eq(-1), 0.0, np.nan),
    )
    df["absILD"] = df["ILD"].abs()
    df["Easy"] = (df["absILD"] >= 6).astype(float)
    df["analysis_group"] = "stakes_cohort1_type04"
    return df[df["RT"].notna()].copy()


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


def optimized_hierarchical_bootstrap_tcm(
    data_nested,
    x_grid: np.ndarray,
    bandwidth: float,
    bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    curves = np.full((bootstraps, len(x_grid)), np.nan, dtype=float)
    n_animals = len(data_nested)
    if n_animals == 0:
        raise ValueError("No animals available for tachometric bootstrap after filtering.")

    for b in range(bootstraps):
        rts_all = []
        outs_all = []
        for animal_idx in rng.integers(0, n_animals, size=n_animals):
            sessions = data_nested[animal_idx]
            for session_idx in rng.integers(0, len(sessions), size=len(sessions)):
                stimuli = sessions[session_idx]
                for stim_idx in rng.integers(0, len(stimuli), size=len(stimuli)):
                    rt_arr, out_arr, _mt_arr = stimuli[stim_idx]
                    trial_idx = rng.integers(0, len(rt_arr), size=len(rt_arr))
                    rts_all.append(rt_arr[trial_idx])
                    outs_all.append(out_arr[trial_idx])

        rt = np.concatenate(rts_all).astype(float)
        out = np.concatenate(outs_all).astype(float)
        support = (rt >= x_grid[0] - bandwidth) & (rt <= x_grid[-1] + bandwidth)
        if not support.any():
            continue
        rt_support = rt[support]
        out_support = out[support]
        weights = epanechnikov_kernel((x_grid[None, :] - rt_support[:, None]) / bandwidth)
        den = weights.sum(axis=0) + 1e-12
        num = (weights * out_support[:, None]).sum(axis=0)
        curves[b, :] = num / den

    return summarize_bootstrap(curves)


def hierarchical_bootstrap_rtd(
    data_nested: list[list[list[np.ndarray]]],
    x_grid: np.ndarray,
    bandwidth: float,
    bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    densities = np.full((bootstraps, len(x_grid)), np.nan, dtype=float)
    n_animals = len(data_nested)
    if n_animals == 0:
        raise ValueError("No animals available for RTD bootstrap after filtering.")

    for b in range(bootstraps):
        sampled_rts = []
        for animal_idx in rng.integers(0, n_animals, size=n_animals):
            sessions = data_nested[animal_idx]
            for session_idx in rng.integers(0, len(sessions), size=len(sessions)):
                stimuli = sessions[session_idx]
                for stim_idx in rng.integers(0, len(stimuli), size=len(stimuli)):
                    rt_arr = stimuli[stim_idx]
                    trial_idx = rng.integers(0, len(rt_arr), size=len(rt_arr))
                    sampled_rts.append(rt_arr[trial_idx])

        rt = np.concatenate(sampled_rts).astype(float)
        n_total = len(rt)
        support = (rt >= x_grid[0] - bandwidth) & (rt <= x_grid[-1] + bandwidth)
        rt_support = rt[support]
        if rt_support.size == 0:
            densities[b, :] = 1e-12
            continue
        weights = epanechnikov_kernel((x_grid[None, :] - rt_support[:, None]) / bandwidth)
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
    input_csv: Path,
    x_min: float,
    x_max: float,
    n_grid: int,
    bandwidth: float,
    bootstraps: int,
    seed: int,
) -> dict:
    trials = load_stakes_type04(input_csv)
    x_grid = np.linspace(x_min, x_max, n_grid)

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
        group_value="stakes_cohort1_type04",
        easy_value=None,
        abl_value=None,
    )
    tcm = optimized_hierarchical_bootstrap_tcm(tcm_nested, x_grid, bandwidth, bootstraps, seed)

    rtd_nested = build_hierarchical_rtd_data(
        trials,
        group_col="analysis_group",
        group_value="stakes_cohort1_type04",
    )
    rtd = hierarchical_bootstrap_rtd(rtd_nested, x_grid, bandwidth, bootstraps, seed + 1)

    animal_counts = (
        trials.groupby(["animal"], dropna=False)
        .agg(
            n_rtd_trials=("RT", "size"),
            n_tcm_trials=("Out", lambda s: int(s.notna().sum())),
            n_sessions=("session", "nunique"),
            rt_median=("RT", "median"),
        )
        .reset_index()
        .sort_values("animal")
    )

    return {
        "kind": "kernel_timing_overlay",
        "name": "Stakes cohort1 out_type_04 kernel timing overlay",
        "description": (
            "Notebook-04-style kernel timing overlay generated from "
            "DataFiles/Stakes_cohort1/out_type_04.csv."
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
            "source": str(input_csv.relative_to(ROOT)),
            "session_type_values_in_file": sorted(pd.to_numeric(trials["session_type_num"], errors="coerce").dropna().unique().tolist()),
            "ABL_values_in_file": sorted(pd.to_numeric(trials["ABL"], errors="coerce").dropna().unique().tolist()),
            "rt_column": "RT",
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
            "n_sessions": int(trials[["animal", "session"]].drop_duplicates().shape[0]),
            "animals": sorted(trials["animal"].dropna().unique().tolist()),
            "animal_counts": animal_counts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stakes cohort1 type04 tachometric + RTD overlay pickle.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--x-min", type=float, default=-0.3)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--n-grid", type=int, default=260)
    parser.add_argument("--bandwidth", type=float, default=0.015)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    overlay = make_overlay(
        input_csv=args.input,
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
    print("Animal counts:")
    print(summary["animal_counts"].to_string(index=False))
    print("Filters:", overlay["filters"])


if __name__ == "__main__":
    main()
