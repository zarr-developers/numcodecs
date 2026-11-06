"""Checks on packages/c-blosc2-blosc1-compat, which only exist in a source checkout."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "c-blosc2-blosc1-compat"
WRAP = ROOT / "subprojects" / "c-blosc2.wrap"

pytestmark = pytest.mark.skipif(not PACKAGE.exists(), reason="not a source checkout")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(PACKAGE / "tools" / script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_wrap_copies_are_in_sync():
    result = _run("sync_wrap.py", "--check")
    assert result.returncode == 0, result.stderr


def test_wrap_matches_upstream_pin():
    pin = dict(
        line.split(" = ")
        for line in (PACKAGE / "UPSTREAM").read_text().splitlines()
        if " = " in line
    )
    wrap = WRAP.read_text()
    assert f"url = {pin['repository']}.git" in wrap
    assert f"revision = {pin['commit']}" in wrap


def test_sync_wrap_detects_a_stale_copy(tmp_path, monkeypatch):
    sync_wrap = _load("sync_wrap")
    package = tmp_path / "packages" / "pkg"
    (package / "meson").mkdir(parents=True)
    (package / "meson" / "meson.build").write_text("project('x')\n")
    dest = tmp_path / "subprojects" / "packagefiles" / "c-blosc2"
    monkeypatch.setattr(sync_wrap, "PKG", package)
    monkeypatch.setattr(sync_wrap, "DEST", dest)

    monkeypatch.setattr(sys, "argv", ["sync_wrap.py", "--check"])
    assert sync_wrap.main() == 1  # nothing copied yet
    monkeypatch.setattr(sys, "argv", ["sync_wrap.py"])
    assert sync_wrap.main() == 0
    monkeypatch.setattr(sys, "argv", ["sync_wrap.py", "--check"])
    assert sync_wrap.main() == 0

    (dest / "meson.build").write_text("changed\n")
    (dest / "stale.txt").write_text("left over\n")
    assert sync_wrap.main() == 1
    monkeypatch.setattr(sys, "argv", ["sync_wrap.py"])
    assert sync_wrap.main() == 0
    assert not (dest / "stale.txt").exists()
