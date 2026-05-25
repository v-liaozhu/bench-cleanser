"""Legacy entry point — delegates to bench_cleanser.cli:main.

Prefer the `bench-cleanser` console script (installed via pyproject.toml).
"""
import warnings

from bench_cleanser.cli import main

if __name__ == "__main__":
    warnings.warn(
        "run_pipeline.py is deprecated; use the `bench-cleanser` console script.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
