"""Tests never read the user's own climber profiles (data/climbers.yaml) or models
(models/: e.g. a hold detector would change the regression results): results must be
the same on every computer. Tests that need them point to their own."""

import pytest


@pytest.fixture(autouse=True)
def _no_personal_climbers(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("CLIMBVISION_CLIMBERS",
                       str(tmp_path_factory.getbasetemp() / "no_climbers.yaml"))
    monkeypatch.setenv("CLIMBVISION_MODELS", str(tmp_path_factory.getbasetemp() / "no_models"))
