"""Project entrypoint.

Allows running from repository root with:
    python -m src.main ...
without requiring PYTHONPATH hacks.
"""

from .rl_airfoil.training.cli import main

if __name__ == "__main__":
    main()
