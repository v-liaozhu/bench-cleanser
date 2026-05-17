"""Legacy entry point — delegates to bench_cleanser.cli:trajectory_main.

Prefer the `bench-cleanser-trajectory` console script (installed via pyproject.toml).
"""
from bench_cleanser.cli import trajectory_main

if __name__ == "__main__":
    trajectory_main()
