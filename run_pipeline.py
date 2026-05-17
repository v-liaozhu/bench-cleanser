"""Legacy entry point — delegates to bench_cleanser.cli:main.

Prefer the `bench-cleanser` console script (installed via pyproject.toml).
"""
from bench_cleanser.cli import main

if __name__ == "__main__":
    main()
