"""Keep runnable examples aligned with the current checkout and public API."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def example_sources():
    return sorted(EXAMPLES.rglob("*.py"))


def test_example_sources_compile_and_have_no_checkout_paths():
    forbidden = ("claude-tests", "ai-tests", "wsl.localhost", "/home/",
                 "PYTHONPATH=.", "from examples.")
    for path in example_sources():
        text = path.read_text()
        compile(text, str(path), "exec")
        assert not any(value in text for value in forbidden), path


def test_example_garnet_imports_resolve_against_current_api():
    for path in example_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "garnet" or alias.name.startswith("garnet."):
                        importlib.import_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name == "garnet" or module_name.startswith("garnet."):
                    module = importlib.import_module(module_name)
                    for alias in node.names:
                        if alias.name != "*":
                            assert not alias.name.startswith("_"), (
                                path, module_name, alias.name
                            )
                            assert hasattr(module, alias.name), (
                                path, module_name, alias.name
                            )


def test_cross_example_import_works_from_documented_entry_point():
    result = subprocess.run(
        [sys.executable, "examples/hpmr_adaptive_coupled_benchmark.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hpmr_example_metadata_is_public_and_immutable():
    from garnet.benchmarks import (HPMR_BE_SITES, HPMR_DRUM_SITES,
                                  HPMR_FUEL_SITES,
                                  hpmr_placeholder_materials)

    assert isinstance(HPMR_FUEL_SITES, tuple) and len(HPMR_FUEL_SITES) == 30
    assert isinstance(HPMR_BE_SITES, tuple) and len(HPMR_BE_SITES) == 12
    assert isinstance(HPMR_DRUM_SITES, tuple) and len(HPMR_DRUM_SITES) == 12
    assert len(hpmr_placeholder_materials()) == 6


def test_colab_source_archive_matches_the_checkout():
    """Notebook uploads must carry the same implementation as the checkout."""
    from zipfile import ZipFile
    package = ROOT / "garnet"
    with ZipFile(ROOT / "dist" / "garnet-src.zip") as archive:
        paths = [ROOT / "README.md", ROOT / "pyproject.toml"]
        paths.extend(sorted(package.rglob("*.py")))
        paths.extend(sorted(package.rglob("*.npz")))
        for path in paths:
            member = path.relative_to(ROOT).as_posix()
            assert member in archive.namelist(), (
                member, "missing from Colab archive; run python tools/build_src_zip.py")
            assert archive.read(member) == path.read_bytes(), (
                member, "stale Colab archive; run python tools/build_src_zip.py")
