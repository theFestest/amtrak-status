"""Fixtures for the white-box tests in this directory.

Helpers live in ``legacy_helpers.py``: importing from ``conftest`` breaks as soon as
another test directory has its own conftest (both are imported as ``conftest``).
"""

from unittest.mock import patch

import pytest

import amtrak_status.tracker as tracker
from legacy_helpers import FIXED_NOW


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset all module-level globals between tests."""
    tracker.COMPACT_MODE = False
    tracker.STATION_FROM = None
    tracker.STATION_TO = None
    tracker.FOCUS_CURRENT = True
    tracker.NOTIFY_STATIONS = set()
    tracker.NOTIFY_ALL = False
    tracker._notified_stations = set()
    tracker._notifications_initialized = False
    tracker.CONNECTION_STATION = None
    tracker._last_successful_data = None
    tracker._last_fetch_time = None
    tracker._last_error = None
    tracker._train_caches = {}
    tracker.REFRESH_INTERVAL = 30
    yield


@pytest.fixture(autouse=True)
def freeze_time():
    """Patch tracker._now to return FIXED_NOW for deterministic tests."""
    with patch("amtrak_status.tracker._now", return_value=FIXED_NOW):
        yield
