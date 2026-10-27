"""Keep archived low-level tests opt-in while the default suite uses models."""

from pathlib import Path

import pytest


LEGACY = Path(__file__).parent / "legacy"


def pytest_addoption(parser):
    parser.addoption("--run-legacy", action="store_true", default=False,
                     help="also collect the preserved pre-API test modules")


def pytest_ignore_collect(collection_path, config):
    if collection_path == LEGACY and not config.getoption("--run-legacy"):
        return True
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if LEGACY in item.path.parents:
            item.add_marker(pytest.mark.legacy)
            if not config.getoption("--run-legacy"):
                item.add_marker(pytest.mark.skip(reason="use --run-legacy for archived tests"))
