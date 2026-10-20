#!/usr/bin/env python3
"""Print the `diff_files` line for subprojects/c-blosc2.wrap from the patch series."""

from pathlib import Path

pkg = Path(__file__).resolve().parents[1]
print(
    "diff_files = "
    + ", ".join(f"c-blosc2/patches/{p.name}" for p in sorted((pkg / "patches").glob("*.patch")))
)
