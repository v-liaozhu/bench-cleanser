"""Legacy entry point — delegates to bench_cleanser.cli:deep_dive_main.

Prefer the `bench-cleanser-deep-dive` console script (installed via pyproject.toml).
"""
import warnings

from bench_cleanser.cli import deep_dive_main

if __name__ == "__main__":
    warnings.warn(
        "run_deep_dive.py is deprecated; use the "
        "`bench-cleanser-deep-dive` console script.",
        DeprecationWarning,
        stacklevel=2,
    )
    deep_dive_main()
