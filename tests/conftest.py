from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ellegon import config


@pytest.fixture()
def temp_campaigns_root(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "campaigns"
    dest = tmp_path / "campaigns"
    shutil.copytree(source, dest)
    original = config.CAMPAIGNS_ROOT
    config.set_campaigns_root(dest)
    try:
        yield dest
    finally:
        config.set_campaigns_root(original)


@pytest.fixture()
def temp_saves_root(tmp_path: Path) -> Path:
    dest = tmp_path / "saves"
    original = config.SAVES_ROOT
    config.set_saves_root(dest)
    try:
        yield dest
    finally:
        config.set_saves_root(original)
