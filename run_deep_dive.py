"""Legacy entry point — delegates to bench_cleanser.cli:deep_dive_main.

Prefer the `bench-cleanser-deep-dive` console script (installed via pyproject.toml).
"""
from bench_cleanser.cli import deep_dive_main

if __name__ == "__main__":
    deep_dive_main()
