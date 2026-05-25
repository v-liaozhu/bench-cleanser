"""Legacy entry point — delegates to bench_cleanser.cli:trajectory_main.

Prefer the `bench-cleanser-trajectory` console script (installed via pyproject.toml).
"""
import warnings

from bench_cleanser.cli import trajectory_main

if __name__ == "__main__":
    warnings.warn(
        "run_trajectory_analysis.py is deprecated; use the "
        "`bench-cleanser-trajectory` console script.",
        DeprecationWarning,
        stacklevel=2,
    )
    trajectory_main()
