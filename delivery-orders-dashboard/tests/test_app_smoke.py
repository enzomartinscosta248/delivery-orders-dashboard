"""Roda o app inteiro com o framework de testes oficial do Streamlit (AppTest)."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_runs_without_exceptions():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
