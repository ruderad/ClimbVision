"""Tests never read the user's own climber profiles (data/climbers.yaml): results must
be the same on every computer. Tests that need profiles point to their own file."""

import pytest


@pytest.fixture(autouse=True)
def _no_personal_climbers(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("CLIMBVISION_CLIMBERS",
                       str(tmp_path_factory.getbasetemp() / "no_climbers.yaml"))
