
from importlib.metadata import version

import sardinia_fire


def test_package_import():
    assert sardinia_fire is not None


def test_project_version():
    assert version("sardinia-fire-ai") == "0.1.0"


def test_core_modules_import():
    from sardinia_fire import (
        clustering,
        evaluation,
        features,
        ingestion,
        preprocessing,
        ranking,
        tracking,
        visualization,
    )

    modules = [
        clustering,
        evaluation,
        features,
        ingestion,
        preprocessing,
        ranking,
        tracking,
        visualization,
    ]

    assert all(module is not None for module in modules)
