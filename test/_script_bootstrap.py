from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_project(*, allow_omp_duplicate: bool = False) -> Path:
    if allow_omp_duplicate:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return PROJECT_ROOT
