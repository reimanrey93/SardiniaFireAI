"""Configuration loading utilities."""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load and validate a YAML configuration file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    return config
