#!/usr/bin/env python3
"""Evaluate a trained NPE run: calibration suite + exact-posterior comparison.

Usage:
    python scripts/evaluate_npe.py --run-dir out/run_XXXX --num-sbc 300 --num-exact 20

Loads the NPE model saved under ``--run-dir`` (``model_config.yaml`` +
``weights/best_model.pt``) and runs the section-4 diagnostics from the NPE
design note:

    * replication test (X vs X(+)X; SD ratio -> 1/sqrt(2))
    * SBC rank uniformity (overall + stratified by K)
    * central credible-interval coverage (50% / 90%)
    * posterior-SD vs K shrinkage slope (-> -1/2)
    * NPE vs exact-posterior mean/SD agreement on a few fresh datasets

Writes a JSON report to ``--out`` (default ``<run-dir>/npe_eval.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ctmc_surrogate.data.targets import z_to_nu
from ctmc_surrogate.data_generation.config import TransitionRateConfig
from ctmc_surrogate.data_generation.transition_rate import DiagonalTransitionRateMatrixGenerator
from ctmc_surrogate.eval import (
    coverage_test,
    exact_posterior,
    generate_sbc_datasets,
    make_fixed_truth_sampler,
    replication_test,
    run_sbc,
    sbc_ks_test,
    sbc_stratified,
    shrinkage_curve,
)
from ctmc_surrogate.inference import load_npe_predictor


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NPE run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with model_config.yaml + weights")
    parser.add_argument("--out", type=Path, default=None, help="Report path (default <run-dir>/npe_eval.json)")
    parser.add_argument("--num-sbc", type=int, default=300, help="Datasets for SBC / coverage")
    parser.add_argument("--num-draws", type=int, default=1000, help="Posterior draws per SBC rank")
    parser.add_argument("--num-exact", type=int, default=15, help="Datasets for NPE-vs-exact comparison")
    parser.add_argument("--num-states", type=int, default=4)
    parser.add_argument("--lifetime-upper", type=float, default=100.0)
    parser.add_argument("--k-min", type=int, default=200)
    parser.add_argument("--k-max", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predictor = load_npe_predictor(args.run_dir, device=args.device)
    model = predictor.model
    n_states = predictor.n_states
    upper = float(args.lifetime_upper)

    report: dict = {"run_dir": str(args.run_dir), "head": predictor.model_config.get("head")}

    # Fresh calibration datasets (no MLE -> cheap).
    datasets = generate_sbc_datasets(
        num_datasets=int(args.num_sbc),
        num_states=int(args.num_states),
        lifetime_upper=upper,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        base_seed=int(args.seed),
    )

    # 1) Replication test (averaged over a handful of datasets).
    rep_ratios = []
    for samples, _ in datasets[: min(10, len(datasets))]:
        rep_ratios.append(replication_test(model, samples).ratio)
    rep_ratios = np.stack(rep_ratios, axis=0)
    report["replication"] = {
        "expected_ratio": float(1.0 / np.sqrt(2.0)),
        "mean_ratio_per_dim": rep_ratios.mean(axis=0),
        "mean_ratio_overall": float(rep_ratios.mean()),
    }

    # 2) SBC (overall + stratified by K).
    sbc = run_sbc(model, datasets, num_draws=int(args.num_draws), seed=int(args.seed))
    ks = sbc_ks_test(sbc)
    k_edges = [args.k_min, 500, 1000, 2000, args.k_max + 1]
    report["sbc"] = {
        "ks_statistic": ks["ks_statistic"],
        "p_value": ks["p_value"],
        "stratified": sbc_stratified(sbc, k_edges),
    }

    # 3) Coverage of central credible intervals.
    cov = coverage_test(model, datasets, levels=(0.5, 0.9), seed=int(args.seed))
    report["coverage"] = {
        "per_dim": cov.coverage_per_dim,
        "overall": cov.coverage_overall,
        "n_datasets": cov.n_datasets,
    }

    # 4) Shrinkage curve for a fixed ground truth.
    q_fixed = DiagonalTransitionRateMatrixGenerator(
        TransitionRateConfig(num_states=int(args.num_states), lifetime_upper=upper)
    ).generate(np.random.default_rng(int(args.seed) + 99))
    sampler = make_fixed_truth_sampler(q_fixed, seed=int(args.seed) + 7)
    k_values = [200, 400, 800, 1600, 3200, min(5000, int(args.k_max))]
    shrink = shrinkage_curve(model, sampler, k_values)
    report["shrinkage"] = {
        "expected_slope": -0.5,
        "k_values": shrink.k_values,
        "slopes_per_dim": shrink.slopes,
    }

    # 5) NPE vs exact posterior on a few fresh datasets.
    exact_rows = []
    for samples, z_true in datasets[: int(args.num_exact)]:
        pred = predictor.predict(samples, num_samples=2000, return_samples=True, seed=int(args.seed))
        ex = exact_posterior(samples, n_states=n_states, lifetime_upper=upper, n_importance=2000, seed=int(args.seed))
        exact_rows.append(
            {
                "K": int(samples.shape[0]),
                "z_true": z_true,
                "npe_mean_z": pred.z_mean,
                "exact_mean_z": ex.mean_z,
                "npe_std_z": pred.z_std,
                "exact_std_z": ex.std_z,
                "npe_mean_nu": pred.nu_mean,
                "exact_mean_nu": z_to_nu(ex.mean_z),
                "exact_ess": ex.ess,
            }
        )
    if exact_rows:
        dz_mean = np.array([r["npe_mean_z"] - r["exact_mean_z"] for r in exact_rows])
        dz_std = np.array([r["npe_std_z"] - r["exact_std_z"] for r in exact_rows])
        report["exact_comparison"] = {
            "mean_abs_error_mean_z": float(np.abs(dz_mean).mean()),
            "mean_abs_error_std_z": float(np.abs(dz_std).mean()),
            "rows": exact_rows,
        }

    out_path = args.out if args.out is not None else (args.run_dir / "npe_eval.json")
    Path(out_path).write_text(json.dumps(_to_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"NPE evaluation report written to: {out_path}")
    print(f"  replication mean ratio (target {1/np.sqrt(2):.3f}): {report['replication']['mean_ratio_overall']:.3f}")
    print(f"  SBC KS per dim: {np.round(report['sbc']['ks_statistic'], 3)}")
    print(f"  coverage 50/90: {report['coverage']['overall']}")
    print(f"  shrinkage slopes (target -0.5): {np.round(report['shrinkage']['slopes_per_dim'], 3)}")
    if exact_rows:
        print(f"  NPE-vs-exact |Δmean z|: {report['exact_comparison']['mean_abs_error_mean_z']:.4f}")


if __name__ == "__main__":
    main()
