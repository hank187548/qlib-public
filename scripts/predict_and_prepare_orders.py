#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.trade.predict_and_prepare_orders import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
