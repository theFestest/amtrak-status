"""Fixtures for the contract tests.

Nothing in this directory may import from ``amtrak_status`` or patch its
internals: see ``_harness.py`` for why and for the seams that are used instead.
"""

import socket

import pytest

from _harness import App


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything bypasses the fake API and opens a real connection."""

    def refuse(self, address):
        raise AssertionError(f"contract tests must not open network connections (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture
def app() -> App:
    """A CLI runner with an empty fake API, frozen at 2026-02-08 11:00 EST."""
    return App()
