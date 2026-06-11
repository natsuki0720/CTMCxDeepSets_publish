"""Module providing datasets and batching utilities."""

from .collate import ctmc_collate_fn
from .dataset_csv_loader import ParsedCTMCDataset, as_filewise, as_samplewise, load_dir, parse_ctmc_csv, parse_ctmc_csv_header
from .dataset_screening import (
    ScreeningConfig,
    ScreeningResult,
    extract_lambdas_from_Q,
    screen_datasets,
    screen_datasets_npe,
    screen_dir_fast,
    validate_samples,
)
from .dataset import CTMCSurrogateDataset
from .targets import (
    build_npe_target_from_Q,
    lambdas_to_z,
    nu_to_z,
    unbounded_to_z,
    z_to_lambdas,
    z_to_nu,
    z_to_unbounded,
)

__all__ = [
    "CTMCSurrogateDataset",
    "ctmc_collate_fn",
    "ParsedCTMCDataset",
    "parse_ctmc_csv",
    "load_dir",
    "parse_ctmc_csv_header",
    "as_filewise",
    "as_samplewise",
    "ScreeningConfig",
    "ScreeningResult",
    "extract_lambdas_from_Q",
    "screen_datasets",
    "screen_datasets_npe",
    "screen_dir_fast",
    "validate_samples",
    # NPE target-space transforms
    "build_npe_target_from_Q",
    "lambdas_to_z",
    "z_to_lambdas",
    "z_to_nu",
    "nu_to_z",
    "z_to_unbounded",
    "unbounded_to_z",
]
