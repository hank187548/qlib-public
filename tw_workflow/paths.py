from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tw_workflow.settings import WORK_DIR


@dataclass(frozen=True)
class WorkflowPaths:
    output_root: Path
    report_dir: Path
    fig_dir: Path


def set_output_dirs(combo_name: str) -> WorkflowPaths:
    root = WORK_DIR / "outputs" / "tw_workflow" / combo_name
    report_dir = root / "reports"
    fig_dir = root / "figures"
    for directory in (root, report_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return WorkflowPaths(output_root=root, report_dir=report_dir, fig_dir=fig_dir)


def build_effective_name(
    combo_name: str,
    *,
    n_drop_override: int | None = None,
    topk_override: int | None = None,
    rebalance: str = "day",
    strategy_choice: str = "bucket",
    deal_price: str = "close",
    simulate_limit: bool = False,
    limit_slippage: float = 0.01,
    limit_tplus: bool = False,
) -> str:
    effective_name = combo_name
    if n_drop_override is not None:
        effective_name = f"{combo_name}_ndrop{n_drop_override}"
    if topk_override is not None:
        effective_name = f"{effective_name}_topk{topk_override}"
    if rebalance != "day":
        effective_name = f"{effective_name}_{rebalance}"
    if strategy_choice != "bucket":
        effective_name = f"{effective_name}_{strategy_choice}"
    actual_deal_price = "open" if limit_tplus else deal_price
    if actual_deal_price != "close" or simulate_limit or limit_tplus:
        effective_name = f"{effective_name}_{actual_deal_price}"
    if simulate_limit:
        slippage_tag = str(limit_slippage).replace(".", "p").replace("-", "m")
        effective_name = f"{effective_name}_simlimit{slippage_tag}"
    if limit_tplus:
        effective_name = f"{effective_name}_tplus"
    return effective_name
