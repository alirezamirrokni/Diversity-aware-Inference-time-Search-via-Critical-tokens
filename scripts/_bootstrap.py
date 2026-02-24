import os
import sys


def bootstrap() -> None:
    # Ensure /path/to/DISC/src is on sys.path so imports like 'utils' work.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
