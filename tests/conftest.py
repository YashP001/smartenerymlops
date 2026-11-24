"""Keep default monitoring writes in tests away from application databases."""

import pytest


@pytest.fixture(autouse=True)
def isolated_default_monitoring_database(tmp_path, monkeypatch):
    """Use a fresh store per test; tests may still override the path explicitly."""
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "default_monitoring.db"))
