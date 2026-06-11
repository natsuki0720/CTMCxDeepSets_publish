"""Evaluation and calibration tools for the NPE posterior."""

from .exact_posterior import (
    AggregatedCells,
    ExactPosteriorResult,
    aggregate_cells,
    exact_loglik_z,
    exact_posterior,
    find_map,
    laplace_covariance,
    log_posterior_z,
    log_prior_z,
)
from .sbc import (
    CoverageResult,
    ReplicationResult,
    SBCResult,
    ShrinkageResult,
    coverage_test,
    generate_sbc_datasets,
    make_fixed_truth_sampler,
    posterior_rank,
    replication_test,
    run_sbc,
    sbc_ks_test,
    sbc_stratified,
    shrinkage_curve,
)

__all__ = [
    # exact posterior
    "AggregatedCells",
    "ExactPosteriorResult",
    "aggregate_cells",
    "exact_loglik_z",
    "exact_posterior",
    "find_map",
    "laplace_covariance",
    "log_posterior_z",
    "log_prior_z",
    # calibration
    "ReplicationResult",
    "SBCResult",
    "CoverageResult",
    "ShrinkageResult",
    "replication_test",
    "posterior_rank",
    "run_sbc",
    "sbc_ks_test",
    "sbc_stratified",
    "coverage_test",
    "shrinkage_curve",
    "generate_sbc_datasets",
]
