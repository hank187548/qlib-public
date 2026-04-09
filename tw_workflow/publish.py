from __future__ import annotations

import shutil
from pathlib import Path

from tw_workflow.settings import WORK_DIR


SOURCE_ROOT = WORK_DIR / "outputs" / "tw_workflow"
DEST_ROOT = WORK_DIR / "outputs" / "best_run"
SKIP_NAMES = {"Thumbs.db"}
SKIP_PARTS = {"@eaDir"}
SUMMARY_LABELS = {
    "Universe 檔數": "Universe size",
    "資料期間": "Data period",
    "訓練區間": "Train range",
    "驗證區間": "Validation range",
    "測試區間": "Test range",
    "回測期間": "Backtest period",
    "策略累積報酬": "Strategy cumulative return",
    "基準累積報酬": "Benchmark cumulative return",
    "有交易日數": "Trading days",
}


def should_skip(path: Path) -> bool:
    return path.name in SKIP_NAMES or any(part in SKIP_PARTS for part in path.parts)


def translate_summary(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        normalized = line.replace("：", ":")
        if ":" not in normalized:
            out_lines.append(line)
            continue
        key, value = normalized.split(":", 1)
        mapped = SUMMARY_LABELS.get(key.strip())
        if mapped is None:
            out_lines.append(line)
            continue
        out_lines.append(f"{mapped}: {value.strip()}")
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def copy_tree(src_dir: Path, dst_dir: Path, translate_summary_file: bool) -> None:
    if not src_dir.exists():
        return
    for path in src_dir.rglob("*"):
        if should_skip(path):
            continue
        rel = path.relative_to(src_dir)
        target = dst_dir / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if translate_summary_file and path.name == "summary.txt":
            target.write_text(translate_summary(path.read_text(encoding="utf-8")), encoding="utf-8")
            continue
        shutil.copy2(path, target)


def promote_output(output_name: str, *, dest: Path = DEST_ROOT, clean: bool = False, translate_summary_file: bool = True) -> Path:
    src = SOURCE_ROOT / output_name
    if not src.exists():
        raise FileNotFoundError(f"Source run not found: {src}")

    reports_src = src / "reports"
    figures_src = src / "figures"
    reports_dst = dest / "reports"
    figures_dst = dest / "figures"

    if clean:
        shutil.rmtree(reports_dst, ignore_errors=True)
        shutil.rmtree(figures_dst, ignore_errors=True)

    dest.mkdir(parents=True, exist_ok=True)
    copy_tree(reports_src, reports_dst, translate_summary_file=translate_summary_file)
    copy_tree(figures_src, figures_dst, translate_summary_file=False)
    return dest
