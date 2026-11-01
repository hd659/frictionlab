"""`python -m frictionlab ...` -> the same entry point as the `frictionlab` console script."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
