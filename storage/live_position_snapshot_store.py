import json
import time
from pathlib import Path
from typing import Any, Dict

from config.settings import settings


class LivePositionSnapshotStore:
    def __init__(self) -> None:
        p = Path(settings.state_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.file_path = p / "live_position_snapshots.json"

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not self.file_path.exists():
            return {}
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def save(self, payload: Dict[str, Dict[str, Any]]) -> None:
        self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def replace(self, rows: Dict[str, Dict[str, Any]]) -> None:
        now = time.time()
        payload: Dict[str, Dict[str, Any]] = {}
        for key, row in (rows or {}).items():
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item.setdefault("last_seen_ts", now)
            payload[key] = item
        self.save(payload)
