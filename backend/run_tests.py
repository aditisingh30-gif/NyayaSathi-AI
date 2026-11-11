import sys
import os
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else [str(backend_dir / "tests")]
    cmd = [
        "-v",
        "-s",
        "-p", "no:cacheprovider",
        "-o", "norecursedirs=frontend node_modules .venv* .git data",
        "-W", "ignore::DeprecationWarning",
        "-W", "ignore::FutureWarning",
        *args
    ]
    print(f"Running pytest with args: {cmd}")
    exit_code = pytest.main(cmd)
    sys.exit(exit_code)
