"""Keeps the typing_extensions minimum tested in CI in step with the declared one."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_ci_tests_the_declared_typing_extensions_minimum():
    pyproject = ROOT / "pyproject.toml"
    workflow = ROOT / ".github" / "workflows" / "ci.yaml"
    if not (pyproject.exists() and workflow.exists()):
        pytest.skip("not running from a source checkout")
    (declared,) = re.findall(r'"typing_extensions>=([0-9.]+)"', pyproject.read_text())
    (tested,) = re.findall(r'"typing_extensions==([0-9.]+)"', workflow.read_text())
    assert "-c min-constraints.txt" in workflow.read_text()
    assert tested == declared or tested == f"{declared}.0"
