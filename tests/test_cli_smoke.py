from __future__ import annotations

from pathlib import Path

import pytest

from ellegon import cli


def test_cli_smoke(
    temp_campaigns_root: Path, temp_saves_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "ellegon",
            "--campaign",
            "test_campaign",
            "--instance",
            "cli_instance",
            "--players",
            "1",
            "--fake-gateway",
        ],
    )

    cli.main()
