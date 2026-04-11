from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperTradingPaths:
    profile_root: Path
    latest_dir: Path
    daily_root: Path

    def daily_dir(self, trade_date: str) -> Path:
        directory = self.daily_root / trade_date
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def build_paths(profile_root: Path) -> PaperTradingPaths:
    profile_root.mkdir(parents=True, exist_ok=True)
    latest_dir = profile_root / "latest"
    daily_root = profile_root / "daily"
    latest_dir.mkdir(parents=True, exist_ok=True)
    daily_root.mkdir(parents=True, exist_ok=True)
    return PaperTradingPaths(profile_root=profile_root, latest_dir=latest_dir, daily_root=daily_root)
