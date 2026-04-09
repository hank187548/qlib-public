#!/usr/bin/env python3
"""Search model params, backtest top candidates, optionally promote best run."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tw_workflow.search import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    raise SystemExit(main())
