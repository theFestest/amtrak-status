"""Shared test fixtures and helpers for amtrak-status tests."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console

import amtrak_status.tracker as tracker


# =============================================================================
# Constants
# =============================================================================


# A fixed "now" for deterministic time-based tests
FIXED_NOW = datetime(2025, 3, 15, 14, 30, 0)


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


# =============================================================================
# Test data helpers
# =============================================================================


def make_station(
    code="TST",
    name="Test Station",
    status="",
    sch_arr=None,
    sch_dep=None,
    arr=None,
    dep=None,
    platform="",
):
    """Build a station dict matching the API shape."""
    return {
        "code": code,
        "name": name,
        "status": status,
        "schArr": sch_arr,
        "schDep": sch_dep,
        "arr": arr,
        "dep": dep,
        "platform": platform,
    }


def make_train(
    train_num="42",
    route_name="Pennsylvanian",
    train_id="42-1",
    stations=None,
    velocity=45,
    heading="E",
    train_state="Active",
    status_msg="On Time",
    dest_name="New York Penn",
):
    """Build a train dict matching the API shape."""
    return {
        "trainNum": train_num,
        "routeName": route_name,
        "trainID": train_id,
        "stations": stations or [],
        "velocity": velocity,
        "heading": heading,
        "trainState": train_state,
        "statusMsg": status_msg,
        "destName": dest_name,
    }


def ts_ms(dt: datetime) -> int:
    """Convert datetime to Unix timestamp in milliseconds (API format)."""
    return int(dt.timestamp() * 1000)


def sample_journey_stations():
    """A realistic 5-stop journey, train between stations 2 and 3."""
    base = FIXED_NOW - timedelta(hours=3)
    return [
        make_station(
            code="PGH", name="Pittsburgh", status="Departed",
            sch_dep=ts_ms(base), dep=ts_ms(base + timedelta(minutes=2)),
        ),
        make_station(
            code="GBG", name="Greensburg", status="Departed",
            sch_arr=ts_ms(base + timedelta(hours=1)),
            sch_dep=ts_ms(base + timedelta(hours=1, minutes=2)),
            arr=ts_ms(base + timedelta(hours=1, minutes=5)),
            dep=ts_ms(base + timedelta(hours=1, minutes=7)),
        ),
        make_station(
            code="HBG", name="Harrisburg", status="Enroute",
            sch_arr=ts_ms(base + timedelta(hours=3, minutes=30)),
            sch_dep=ts_ms(base + timedelta(hours=3, minutes=35)),
            arr=ts_ms(base + timedelta(hours=3, minutes=40)),
            dep=ts_ms(base + timedelta(hours=3, minutes=45)),
        ),
        make_station(
            code="PHL", name="Philadelphia", status="",
            sch_arr=ts_ms(base + timedelta(hours=5)),
            sch_dep=ts_ms(base + timedelta(hours=5, minutes=5)),
        ),
        make_station(
            code="NYP", name="New York Penn", status="",
            sch_arr=ts_ms(base + timedelta(hours=6, minutes=30)),
        ),
    ]


def render_to_text(renderable, width=120) -> str:
    """Capture a Rich renderable as plain text for assertion."""
    console = Console(record=True, width=width, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def load_fixture(name: str):
    """Load a JSON fixture file from tests/fixtures/."""
    fixture_path = Path(__file__).parent / "fixtures" / name
    with open(fixture_path) as f:
        return json.load(f)


def make_mock_httpx_client(json_response):
    """Create a mock httpx.Client that returns the given JSON from .get()."""
    mock_response = MagicMock()
    mock_response.json.return_value = json_response
    mock_response.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_response
    return mock_client


def journey_at_phase(phase: str, now=None):
    """
    Build a realistic train #42 (Pennsylvanian) at different journey phases.
    5 stations: PGH -> GBG -> HBG -> PHL -> NYP.

    Phases:
        "predeparture" -- all stations scheduled, none departed
        "early"        -- PGH departed, heading to GBG
        "mid"          -- PGH+GBG departed, en route to HBG
        "mid_late"     -- same as mid but HBG arrival delayed +25 min
        "arriving"     -- PGH+GBG+HBG departed, at station PHL
        "final_leg"    -- PGH..PHL departed, en route to NYP
    """
    if now is None:
        now = FIXED_NOW
    base = now - timedelta(hours=2)

    sch_times = {
        "PGH_dep": base,
        "GBG_arr": base + timedelta(hours=1),
        "GBG_dep": base + timedelta(hours=1, minutes=2),
        "HBG_arr": base + timedelta(hours=2, minutes=30),
        "HBG_dep": base + timedelta(hours=2, minutes=35),
        "PHL_arr": base + timedelta(hours=4),
        "PHL_dep": base + timedelta(hours=4, minutes=5),
        "NYP_arr": base + timedelta(hours=5, minutes=30),
    }

    if phase == "predeparture":
        return make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Predeparture", status_msg="", velocity=0,
            dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="",
                             sch_dep=ts_ms(sch_times["PGH_dep"])),
                make_station(code="GBG", name="Greensburg", status="",
                             sch_arr=ts_ms(sch_times["GBG_arr"]),
                             sch_dep=ts_ms(sch_times["GBG_dep"])),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(sch_times["HBG_arr"]),
                             sch_dep=ts_ms(sch_times["HBG_dep"])),
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_arr=ts_ms(sch_times["PHL_arr"]),
                             sch_dep=ts_ms(sch_times["PHL_dep"])),
                make_station(code="NYP", name="New York Penn", status="",
                             sch_arr=ts_ms(sch_times["NYP_arr"])),
            ],
        )

    elif phase == "early":
        return make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Active", status_msg="On Time", velocity=55,
            heading="E", dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(sch_times["PGH_dep"]),
                             dep=ts_ms(sch_times["PGH_dep"] + timedelta(minutes=1))),
                make_station(code="GBG", name="Greensburg", status="Enroute",
                             sch_arr=ts_ms(sch_times["GBG_arr"]),
                             sch_dep=ts_ms(sch_times["GBG_dep"]),
                             arr=ts_ms(sch_times["GBG_arr"] + timedelta(minutes=2))),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(sch_times["HBG_arr"]),
                             sch_dep=ts_ms(sch_times["HBG_dep"])),
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_arr=ts_ms(sch_times["PHL_arr"]),
                             sch_dep=ts_ms(sch_times["PHL_dep"])),
                make_station(code="NYP", name="New York Penn", status="",
                             sch_arr=ts_ms(sch_times["NYP_arr"])),
            ],
        )

    elif phase == "mid":
        return make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Active", status_msg="On Time", velocity=62,
            heading="E", dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(sch_times["PGH_dep"]),
                             dep=ts_ms(sch_times["PGH_dep"] + timedelta(minutes=1))),
                make_station(code="GBG", name="Greensburg", status="Departed",
                             sch_arr=ts_ms(sch_times["GBG_arr"]),
                             sch_dep=ts_ms(sch_times["GBG_dep"]),
                             arr=ts_ms(sch_times["GBG_arr"] + timedelta(minutes=2)),
                             dep=ts_ms(sch_times["GBG_dep"] + timedelta(minutes=3))),
                make_station(code="HBG", name="Harrisburg", status="Enroute",
                             sch_arr=ts_ms(sch_times["HBG_arr"]),
                             sch_dep=ts_ms(sch_times["HBG_dep"]),
                             arr=ts_ms(sch_times["HBG_arr"] + timedelta(minutes=5))),
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_arr=ts_ms(sch_times["PHL_arr"]),
                             sch_dep=ts_ms(sch_times["PHL_dep"])),
                make_station(code="NYP", name="New York Penn", status="",
                             sch_arr=ts_ms(sch_times["NYP_arr"])),
            ],
        )

    elif phase == "mid_late":
        # Same as mid but HBG arrival delayed +25 min
        return make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Active", status_msg="25 Minutes Late", velocity=45,
            heading="E", dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(sch_times["PGH_dep"]),
                             dep=ts_ms(sch_times["PGH_dep"] + timedelta(minutes=1))),
                make_station(code="GBG", name="Greensburg", status="Departed",
                             sch_arr=ts_ms(sch_times["GBG_arr"]),
                             sch_dep=ts_ms(sch_times["GBG_dep"]),
                             arr=ts_ms(sch_times["GBG_arr"] + timedelta(minutes=15)),
                             dep=ts_ms(sch_times["GBG_dep"] + timedelta(minutes=18))),
                make_station(code="HBG", name="Harrisburg", status="Enroute",
                             sch_arr=ts_ms(sch_times["HBG_arr"]),
                             sch_dep=ts_ms(sch_times["HBG_dep"]),
                             arr=ts_ms(sch_times["HBG_arr"] + timedelta(minutes=25))),
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_arr=ts_ms(sch_times["PHL_arr"]),
                             sch_dep=ts_ms(sch_times["PHL_dep"]),
                             arr=ts_ms(sch_times["PHL_arr"] + timedelta(minutes=22))),
                make_station(code="NYP", name="New York Penn", status="",
                             sch_arr=ts_ms(sch_times["NYP_arr"]),
                             arr=ts_ms(sch_times["NYP_arr"] + timedelta(minutes=20))),
            ],
        )

    elif phase == "arriving":
        return make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Active", status_msg="On Time", velocity=0,
            heading="E", dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(sch_times["PGH_dep"]),
                             dep=ts_ms(sch_times["PGH_dep"] + timedelta(minutes=1))),
                make_station(code="GBG", name="Greensburg", status="Departed",
                             sch_arr=ts_ms(sch_times["GBG_arr"]),
                             sch_dep=ts_ms(sch_times["GBG_dep"]),
                             arr=ts_ms(sch_times["GBG_arr"]),
                             dep=ts_ms(sch_times["GBG_dep"])),
                make_station(code="HBG", name="Harrisburg", status="Departed",
                             sch_arr=ts_ms(sch_times["HBG_arr"]),
                             sch_dep=ts_ms(sch_times["HBG_dep"]),
                             arr=ts_ms(sch_times["HBG_arr"]),
                             dep=ts_ms(sch_times["HBG_dep"])),
                make_station(code="PHL", name="Philadelphia", status="Station",
                             sch_arr=ts_ms(sch_times["PHL_arr"]),
                             sch_dep=ts_ms(sch_times["PHL_dep"]),
                             arr=ts_ms(sch_times["PHL_arr"] - timedelta(minutes=2))),
                make_station(code="NYP", name="New York Penn", status="",
                             sch_arr=ts_ms(sch_times["NYP_arr"]),
                             arr=ts_ms(sch_times["NYP_arr"])),
            ],
        )

    elif phase == "final_leg":
        return make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Active", status_msg="On Time", velocity=70,
            heading="NE", dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(sch_times["PGH_dep"]),
                             dep=ts_ms(sch_times["PGH_dep"])),
                make_station(code="GBG", name="Greensburg", status="Departed",
                             sch_arr=ts_ms(sch_times["GBG_arr"]),
                             sch_dep=ts_ms(sch_times["GBG_dep"]),
                             arr=ts_ms(sch_times["GBG_arr"]),
                             dep=ts_ms(sch_times["GBG_dep"])),
                make_station(code="HBG", name="Harrisburg", status="Departed",
                             sch_arr=ts_ms(sch_times["HBG_arr"]),
                             sch_dep=ts_ms(sch_times["HBG_dep"]),
                             arr=ts_ms(sch_times["HBG_arr"]),
                             dep=ts_ms(sch_times["HBG_dep"])),
                make_station(code="PHL", name="Philadelphia", status="Departed",
                             sch_arr=ts_ms(sch_times["PHL_arr"]),
                             sch_dep=ts_ms(sch_times["PHL_dep"]),
                             arr=ts_ms(sch_times["PHL_arr"]),
                             dep=ts_ms(sch_times["PHL_dep"])),
                make_station(code="NYP", name="New York Penn", status="Enroute",
                             sch_arr=ts_ms(sch_times["NYP_arr"]),
                             arr=ts_ms(sch_times["NYP_arr"])),
            ],
        )

    raise ValueError(f"Unknown phase: {phase}")
