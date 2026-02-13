"""互換用エントリポイント。data_generation配下の実装を再エクスポートする。"""

from src.ctmc_surrogate.data_generation.rng_utils import dirichlet_ones, make_rng

__all__ = ["make_rng", "dirichlet_ones"]
