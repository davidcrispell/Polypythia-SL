from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    config["_config_path"] = str(config_path)
    return config


def output_dir(config: dict[str, Any]) -> Path:
    return Path(config["run"]["output_dir"]).resolve()

