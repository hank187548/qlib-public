#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.research.workflow_by_code_tw import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
