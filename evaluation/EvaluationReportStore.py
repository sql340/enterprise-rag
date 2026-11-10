from __future__ import annotations

from pathlib import Path

from .EvaluationReport import EvaluationReport


class EvaluationReportStore:
    """评测报告 JSON 持久化。"""

    def save(self, report: EvaluationReport, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_text(report.to_json(), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, path: str | Path) -> dict:
        import json

        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


__all__ = ["EvaluationReportStore"]
