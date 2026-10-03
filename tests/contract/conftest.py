"""Fixtures for the contract tests.

Nothing in this directory may import from ``amtrak_status`` or patch its
internals: see ``_harness.py`` for why and for the seams that are used instead.
"""

import pytest

from _harness import App


@pytest.fixture
def app() -> App:
    """A CLI runner with an empty fake API, frozen at 2026-02-08 11:00 EST."""
    return App()
