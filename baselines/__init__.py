"""Local ARIA experiment-baseline implementations.

The package lives outside :mod:`src` deliberately: baselines are experiment
artifacts with their own provenance, not dependencies of ARIA itself.
"""

from baselines.registry import BASELINE_METHODS, build_method

__all__ = ["BASELINE_METHODS", "build_method"]
