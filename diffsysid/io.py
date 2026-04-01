from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_result_json(output_json: str | None, result: dict[str, Any]) -> Path | None:
    if not output_json:
        return None
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out
