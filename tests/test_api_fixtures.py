"""Tests using real API response fixtures to validate handling of actual data shapes.

These tests complement the synthetic-data tests in test_amtrak_status.py by
exercising code paths with realistic API responses, including:
- ISO 8601 timestamps (the real API format)
- Empty array `[]` not-found responses
- Single-space statusMsg edge case
- Missing/null station fields
- Extra fields the code doesn't use
- Station endpoint metadata shape mismatch
"""

from unittest.mock import patch

import pytest
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

import amtrak_status.tracker as tracker
import amtrak_status.models as models
import amtrak_status.connection as connection
from conftest import (
    load_fixture,
    make_mock_httpx_client,
    render_to_text,
)


# =============================================================================
# Shared fixtures for fixture-based test classes
# =============================================================================


@pytest.fixture
def missing_fields_train():
    """Load train_missing_fields.json and return the parsed train dict."""
    fixture = load_fixture("train_missing_fields.json")
    with patch("amtrak_status.api.httpx.Client") as mock_cls:
        mock_cls.return_value = make_mock_httpx_client(fixture)
        result = tracker.fetch_train_data("42")
        assert result is not None
        yield result


@pytest.fixture
def cancelled_stops_train():
    """Load train_cancelled_stops.json and return the parsed train dict."""
    fixture = load_fixture("train_cancelled_stops.json")
    with patch("amtrak_status.api.httpx.Client") as mock_cls:
        mock_cls.return_value = make_mock_httpx_client(fixture)
        result = tracker.fetch_train_data("42")
        assert result is not None
        yield result


# =============================================================================
# TestFixtureTrainParsing
# =============================================================================


class TestFixtureTrainParsing:
    """Feed fixtures through fetch_train_data via mocked httpx."""

    @patch("amtrak_status.api.httpx.Client")
    def test_midjourney_returns_train_data(self, mock_client_cls):
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert result["trainNum"] == "42"
        assert result["routeName"] == "Pennsylvanian"
        assert len(result["stations"]) == 10

    @patch("amtrak_status.api.httpx.Client")
    def test_midjourney_preserves_all_api_fields(self, mock_client_cls):
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert result["trainTimely"] == "On Time"
        assert result["iconColor"] == "#00FF00"
        assert result["lat"] == 40.3456
        assert result["lon"] == -76.8912
        assert result["alerts"] == []
        assert result["trainNumRaw"] == "42"
        assert result["onlyOfTrainNum"] is True

    @patch("amtrak_status.api.httpx.Client")
    def test_midjourney_stations_have_all_fields(self, mock_client_cls):
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        station = result["stations"][0]  # PGH
        assert station["tz"] == "America/New_York"
        assert station["bus"] is False
        assert station["arrCmnt"] == ""
        assert station["depCmnt"] == ""
        assert station["stopIconColor"] == "#2a893d"

    @patch("amtrak_status.api.httpx.Client")
    def test_multi_day_returns_first_entry(self, mock_client_cls):
        """API returns multiple entries for same train number across days.

        fetch_train_data always takes data[key][0] — it trusts the API to
        return entries with the most recent/active train first. This test
        documents that assumption. If the API ever changes sort order,
        the code would silently return stale data.
        """
        fixture = load_fixture("train_multi_day.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert result["trainID"] == "42-8"  # today's train
        assert result["trainState"] == "Active"

        # Verify the fixture has both entries and the stale one is second
        assert len(fixture["42"]) == 2
        assert fixture["42"][1]["trainID"] == "42-7"  # yesterday's completed train
        assert fixture["42"][1]["trainState"] == "Completed"

    @patch("amtrak_status.api.httpx.Client")
    def test_predeparture_returns_data(self, mock_client_cls):
        fixture = load_fixture("train_predeparture.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert result["trainState"] == "Predeparture"


# =============================================================================
# TestFixtureISOTimestamps
# =============================================================================


class TestFixtureISOTimestamps:
    """Verify ISO 8601 timestamps (the real API format) parse and display correctly."""

    def test_iso_with_offset_parses_to_aware_datetime(self):
        dt = tracker.parse_time("2026-02-08T14:30:00-05:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.hour == 14
        assert dt.minute == 30

    @patch("amtrak_status.api.httpx.Client")
    def test_iso_timestamps_display_as_ampm(self, mock_client_cls):
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        text = render_to_text(tracker.build_stations_table(result, focus=False))
        # PGH departed at 7:32 AM (actual dep from fixture)
        assert "7:32 AM" in text
        # GBG departed at 8:20 AM
        assert "8:20 AM" in text

    @patch("amtrak_status.api.httpx.Client")
    def test_iso_timestamps_in_header_eta(self, mock_client_cls):
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        text = render_to_text(tracker.build_header(result))
        # Next station is HUN (Enroute), with arr = 11:20 AM
        assert "Huntingdon" in text
        assert "11:20 AM" in text

    @patch("amtrak_status.api.httpx.Client")
    def test_iso_timestamps_in_compact_display(self, mock_client_cls):
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        text = render_to_text(tracker.build_compact_display(result))
        # HUN is the next station with arr = 11:20 AM
        assert "11:20 AM" in text

    @patch("amtrak_status.api.httpx.Client")
    def test_position_calculation_with_iso_times(self, mock_client_cls):
        """calculate_position_between_stations works with aware datetimes from ISO strings."""
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        position = tracker.calculate_position_between_stations(result)
        # ALT is last Departed, HUN is Enroute
        assert position is not None
        last_code, next_code, progress, mins_remaining = position
        assert last_code == "ALT"
        assert next_code == "HUN"
        assert 0.0 <= progress <= 1.0
        assert mins_remaining >= 0


# =============================================================================
# TestFixtureMissingFields
# =============================================================================


class TestFixtureMissingFields:
    """Station dicts with absent/null fields."""

    def test_missing_dep_key_no_crash(self, missing_fields_train):
        """Station without `dep` key renders fine (.get() returns None)."""
        gbg = missing_fields_train["stations"][1]
        assert "dep" not in gbg
        text = render_to_text(tracker.build_stations_table(missing_fields_train, focus=False))
        assert "Greensburg" in text

    def test_null_arr_renders_blank(self, missing_fields_train):
        """Station with arr: null shows blank in table."""
        gbg = missing_fields_train["stations"][1]
        assert gbg["arr"] is None
        text = render_to_text(tracker.build_stations_table(missing_fields_train, focus=False))
        assert "Greensburg" in text

    def test_bus_flag_ignored_gracefully(self, missing_fields_train):
        """Station with bus: true renders normally."""
        hbg = missing_fields_train["stations"][2]
        assert hbg["bus"] is True
        text = render_to_text(tracker.build_stations_table(missing_fields_train, focus=False))
        assert "Harrisburg" in text

    def test_platform_displayed_for_enroute_station(self, missing_fields_train):
        """Enroute station with non-empty platform shows '(Plt 5B)' in status."""
        hbg = missing_fields_train["stations"][2]
        assert hbg["platform"] == "5B"
        assert hbg["status"] == "Enroute"
        # Use wide render to avoid column wrapping splitting "Plt 5B"
        text = render_to_text(tracker.build_stations_table(missing_fields_train, focus=False), width=200)
        assert "Plt 5B" in text

    def test_platform_on_future_station_not_displayed(self, missing_fields_train):
        """Future station (status='') with platform '3A' does NOT show platform."""
        phl = missing_fields_train["stations"][3]
        assert phl["platform"] == "3A"
        assert phl["status"] == ""
        text = render_to_text(tracker.build_stations_table(missing_fields_train, focus=False))
        assert "Plt 3A" not in text
        assert "Philadelphia" in text  # station still renders

    def test_extra_train_fields_ignored(self, missing_fields_train):
        """Extra fields (lat, lon, objectID, etc.) don't cause errors."""
        assert "objectID" in missing_fields_train
        assert "iconColor" in missing_fields_train
        text = render_to_text(tracker.build_header(missing_fields_train))
        assert "Pennsylvanian" in text

    def test_extra_station_fields_ignored(self, missing_fields_train):
        """Extra fields (tz, bus, arrCmnt, stopIconColor) don't cause errors."""
        station = missing_fields_train["stations"][0]
        assert "tz" in station
        assert "bus" in station
        text = render_to_text(tracker.build_stations_table(missing_fields_train, focus=False))
        assert "Pittsburgh" in text


# =============================================================================
# TestFixtureStatusMsg
# =============================================================================


class TestFixtureStatusMsg:
    """The single-space statusMsg edge case."""

    @patch("amtrak_status.api.httpx.Client")
    def test_space_statusmsg_used_in_header(self, mock_client_cls):
        """Header renders ' ' as status instead of 'Active'.

        This is a cosmetic bug: single space is truthy in Python, so
        `if status_msg:` is True, and the header shows a blank area
        instead of "Active". The correct result (no crash) happens for
        the wrong reason (space is truthy).
        """
        fixture = load_fixture("train_space_statusmsg.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert result["statusMsg"] == " "
        text = render_to_text(tracker.build_header(result))
        # " " is truthy, so build_header uses it as display_status
        # "Active" should NOT appear since statusMsg is truthy
        assert "Active" not in text
        assert "Pennsylvanian" in text

    @patch("amtrak_status.api.httpx.Client")
    def test_space_statusmsg_in_compact(self, mock_client_cls):
        """Compact display includes ' ' in output."""
        fixture = load_fixture("train_space_statusmsg.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        assert result is not None
        text = render_to_text(tracker.build_compact_display(result))
        # The compact display should render without error
        assert "Pennsylvanian" in text
        assert "#42" in text


# =============================================================================
# TestFixtureCancelledStops
# =============================================================================


class TestFixtureCancelledStops:
    """Cancelled station handling with realistic data."""

    def test_explicit_cancelled_status_detected(self, cancelled_stops_train):
        # GBG has status: "Cancelled"
        gbg = cancelled_stops_train["stations"][1]
        assert tracker.is_station_cancelled(gbg) is True

    def test_no_times_cancelled_detected(self, cancelled_stops_train):
        """Detected for station with no schArr/schDep (heuristic detection)."""
        lat = cancelled_stops_train["stations"][2]
        assert tracker.is_station_cancelled(lat) is True

    def test_cancelled_excluded_from_progress(self, cancelled_stops_train):
        completed, current_idx, total = tracker.calculate_progress(cancelled_stops_train["stations"])
        # 6 stations total, 2 cancelled (GBG + LAT) = 4 active
        assert total == 4

    def test_cancelled_rendered_in_table(self, cancelled_stops_train):
        text = render_to_text(tracker.build_stations_table(cancelled_stops_train, focus=False))
        assert "Cancelled" in text
        assert "✗" in text

    def test_cancelled_skipped_in_position_calc(self, cancelled_stops_train):
        position = tracker.calculate_position_between_stations(cancelled_stops_train)
        # JST is last Departed (skipping cancelled GBG/LAT), HBG is Enroute
        assert position is not None
        last_code, next_code, _, _ = position
        assert last_code == "JST"
        assert next_code == "HBG"


# =============================================================================
# TestFixtureStationEndpoint
# =============================================================================


class TestFixtureStationEndpoint:
    """Real station API response shape vs. get_train_schedule_from_station."""

    @pytest.mark.xfail(
        reason="BUG: get_train_schedule_from_station expects {code: [train_dicts]} "
               "but real API returns {code: {metadata_dict}}. The isinstance(trains, list) "
               "check skips the metadata dict, so it silently returns None."
    )
    def test_real_station_api_returns_none(self):
        """Real station response returns None because value is metadata dict, not train list."""
        fixture = load_fixture("station_schedule.json")
        result = tracker.get_train_schedule_from_station(fixture, "42")
        # This should find train 42, but it won't because the value is a dict, not a list
        assert result is not None

    def test_station_not_found_empty_array(self):
        """[] response: `not station_data` is True -> returns None gracefully."""
        fixture = load_fixture("station_not_found.json")
        # not [] is True, so the early return catches it
        result = tracker.get_train_schedule_from_station(fixture, "42")
        assert result is None

    @pytest.mark.coincidence
    def test_station_not_found_caught_by_falsy_check_not_error_check(self):
        """[] is caught by `not station_data`, never reaching
        `"error" in station_data` which would work on [] (returns False)
        but is semantically wrong for a list.

        The `not data` check is load-bearing — it masks the type confusion
        in the `"error" in data` check that follows it.
        """
        fixture = load_fixture("station_not_found.json")

        # The function returns None correctly
        result = tracker.get_train_schedule_from_station(fixture, "42")
        assert result is None

        # Prove the coincidence: `not []` is True, so we never reach
        # the `"error" in data` check
        assert not fixture  # This is what saves us


# =============================================================================
# TestFixtureNotFoundEdgeCases
# =============================================================================


class TestFixtureNotFoundEdgeCases:
    """The [] not-found response that works by coincidence."""

    @pytest.mark.coincidence
    @patch("amtrak_status.api.httpx.Client")
    def test_empty_array_returns_none_via_wrong_code_path(self, mock_client_cls):
        """fetch_train_data returns None for [], but via the wrong logic.

        The code does `key in data` expecting dict-key lookup, but `[]` is a
        list so `in` does value-search instead. Both `"42" in []` and
        `len([]) == 1` are False, so the function falls through to return None.

        This is the correct RESULT but the wrong REASON. If the API ever returns
        a list with values (e.g. `["error"]`), the key-in-list check would still
        return False and mask a real error.
        """
        fixture = load_fixture("train_not_found.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)

        result = tracker.fetch_train_data("42")
        # Correct result, wrong reason:
        assert result is None

        # Prove the coincidence: the code treats [] like a dict
        raw = fixture  # This is []
        assert isinstance(raw, list), "API returns list, not dict — code assumes dict"
        assert "42" not in raw  # list-value search, not dict-key lookup
        assert len(raw) != 1   # empty list doesn't trigger single-key fallback

    @pytest.mark.coincidence
    def test_error_in_empty_array_would_raise(self):
        """If `not []` didn't catch it first, `"error" in []` would be fine
        (returns False), but a non-empty list like `[{"error": "msg"}]` would
        bypass both the `not data` and `key in data` checks incorrectly.

        Documents that the code's error handling is order-dependent on the
        `not data` check happening before `"error" in data`.
        """
        # `not []` is True — early return catches empty lists before
        # the `"error" in data` check that assumes dict
        assert not []

        # But this is fine — "error" in [] returns False, no TypeError
        assert "error" not in []


# =============================================================================
# TestFixtureFullPipeline
# =============================================================================


class TestFixtureFullPipeline:
    """End-to-end: mock httpx -> fetch_train_data -> display functions -> rendered text."""

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_midjourney_build_display(self, mock_fetch):
        fixture = load_fixture("train_active_midjourney.json")
        mock_fetch.return_value = fixture["42"][0]

        result = tracker.build_display("42")
        assert isinstance(result, Layout)
        text = render_to_text(result)
        # Should contain station names from the fixture
        assert "Pittsburgh" in text
        assert "New York Penn" in text
        assert "Pennsylvanian" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_midjourney_compact_display(self, mock_fetch):
        tracker.COMPACT_MODE = True
        fixture = load_fixture("train_active_midjourney.json")
        mock_fetch.return_value = fixture["42"][0]

        result = tracker.build_display("42")
        assert isinstance(result, Text)
        text = render_to_text(result)
        assert "Pennsylvanian" in text
        assert "#42" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_predeparture_build_display(self, mock_fetch):
        fixture = load_fixture("train_predeparture.json")
        mock_fetch.return_value = fixture["42"][0]

        result = tracker.build_display("42")
        text = render_to_text(result)
        assert "Predeparture" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_cancelled_stops_build_display(self, mock_fetch):
        fixture = load_fixture("train_cancelled_stops.json")
        mock_fetch.return_value = fixture["42"][0]

        result = tracker.build_display("42")
        text = render_to_text(result)
        assert "Cancelled" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_alerts_present_but_not_rendered(self, mock_fetch):
        """Train with alerts[] renders without error.

        Note: the current UI does not display alert text anywhere.
        The alerts field is preserved in data but ignored by build_display,
        build_header, and build_stations_table. This test documents that
        behavior — when alert rendering is added, update this test.
        """
        fixture = load_fixture("train_with_alerts.json")
        mock_fetch.return_value = fixture["42"][0]

        result = tracker.build_display("42")
        assert isinstance(result, Layout)
        text = render_to_text(result)
        assert "Pennsylvanian" in text
        # Alerts exist in data but are NOT shown in display
        assert "track work" not in text
        assert "15 minutes" not in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_missing_fields_full_render(self, mock_fetch):
        """Train with missing/null fields renders completely."""
        fixture = load_fixture("train_missing_fields.json")
        mock_fetch.return_value = fixture["42"][0]

        result = tracker.build_display("42")
        assert isinstance(result, Layout)
        text = render_to_text(result)
        assert "Pittsburgh" in text
        assert "Greensburg" in text
        assert "Harrisburg" in text
        assert "New York Penn" in text


# =============================================================================
# TestFixtureMultiTrain
# =============================================================================


class TestFixtureMultiTrain:
    """Multi-train display with fixture data."""

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_both_trains_from_fixtures(self, mock_fetch):
        """Two fixture trains at connection, build_multi_train_display returns Layout."""
        midjourney = load_fixture("train_active_midjourney.json")["42"][0]
        # Use a second fixture as the second train
        with_alerts = load_fixture("train_with_alerts.json")["42"][0]
        # Modify to be a different train number for the second train
        train2 = dict(with_alerts)
        train2["trainNum"] = "178"
        train2["routeName"] = "Keystone"
        # Add PHL station to train2 for connection
        train2["stations"] = [
            {
                "code": "PHL",
                "name": "Philadelphia 30th Street",
                "status": "",
                "schDep": "2026-02-08T15:30:00-05:00",
                "schArr": "2026-02-08T15:25:00-05:00",
                "arr": None,
                "dep": None,
                "platform": "",
            },
            {
                "code": "HBG",
                "name": "Harrisburg",
                "status": "",
                "schArr": "2026-02-08T17:00:00-05:00",
                "schDep": None,
                "arr": None,
                "dep": None,
                "platform": "",
            },
        ]

        mock_fetch.side_effect = [midjourney, train2]
        result = tracker.build_multi_train_display(["42", "178"], "PHL")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_connection_panel_with_iso_times(self, mock_fetch):
        """build_connection_panel shows formatted layover with ISO timestamps."""
        midjourney = load_fixture("train_active_midjourney.json")["42"][0]
        train2 = {
            "trainNum": "178",
            "routeName": "Keystone",
            "trainID": "178-8",
            "trainState": "Active",
            "statusMsg": "On Time",
            "destName": "Harrisburg",
            "heading": "W",
            "velocity": 45,
            "stations": [
                {
                    "code": "PHL",
                    "name": "Philadelphia 30th Street",
                    "status": "",
                    "schDep": "2026-02-08T15:30:00-05:00",
                    "schArr": "2026-02-08T15:25:00-05:00",
                    "arr": None,
                    "dep": None,
                    "platform": "",
                },
            ],
        }

        panel = tracker.build_connection_panel(midjourney, train2, "PHL")
        assert isinstance(panel, Panel)
        text = render_to_text(panel)
        # Should show formatted arrival/departure times
        assert "Pennsylvanian" in text
        assert "Keystone" in text
        assert "Philadelphia" in text


# =============================================================================
# TestFixtureDelayedISO
# =============================================================================


class TestFixtureDelayedISO:
    """Delay display with ISO timestamp fixtures."""

    @pytest.fixture
    def delayed_train(self):
        fixture = load_fixture("train_delayed_iso.json")
        with patch("amtrak_status.api.httpx.Client") as mock_cls:
            mock_cls.return_value = make_mock_httpx_client(fixture)
            result = tracker.fetch_train_data("42")
            assert result is not None
            yield result

    def test_header_shows_delay_indicator(self, delayed_train):
        text = render_to_text(tracker.build_header(delayed_train))
        assert "(+15m)" in text
        assert "11:30 AM" in text

    def test_compact_shows_delay(self, delayed_train):
        text = render_to_text(tracker.build_compact_display(delayed_train))
        assert "+15m" in text

    def test_status_msg_rendered(self, delayed_train):
        text = render_to_text(tracker.build_header(delayed_train))
        assert "15 Minutes Late" in text


# =============================================================================
# TestFixtureCompletedTrain
# =============================================================================


class TestFixtureCompletedTrain:
    """Completed train where all stations are Departed."""

    @pytest.fixture
    def completed_train(self):
        fixture = load_fixture("train_completed.json")
        with patch("amtrak_status.api.httpx.Client") as mock_cls:
            mock_cls.return_value = make_mock_httpx_client(fixture)
            result = tracker.fetch_train_data("42")
            assert result is not None
            yield result

    def test_all_departed(self, completed_train):
        statuses = [s["status"] for s in completed_train["stations"]]
        assert all(s == "Departed" for s in statuses)

    def test_progress_100_percent(self, completed_train):
        completed, _, total = tracker.calculate_progress(
            completed_train["stations"]
        )
        assert completed == total

    def test_position_returns_none(self, completed_train):
        """No next station, so position calc returns None."""
        assert tracker.calculate_position_between_stations(completed_train) is None

    def test_header_renders_without_position_bar(self, completed_train):
        text = render_to_text(tracker.build_header(completed_train))
        assert "Position:" not in text

    def test_full_display_no_crash(self, completed_train):
        text = render_to_text(
            tracker.build_stations_table(completed_train, focus=False)
        )
        assert "Pittsburgh" in text
        assert "New York Penn" in text


# =============================================================================
# TestFixtureFocusedDisplay
# =============================================================================


class TestFixtureFocusedDisplay:
    """Focus mode with real fixture data exceeding 10 stations."""

    @patch("amtrak_status.api.httpx.Client")
    def test_focus_hides_old_departed_with_iso_data(self, mock_client_cls):
        """Midjourney fixture + extra stations triggers focus elision."""
        fixture = load_fixture("train_active_midjourney.json")
        mock_client_cls.return_value = make_mock_httpx_client(fixture)
        result = tracker.fetch_train_data("42")
        assert result is not None

        # Duplicate early departed stations to exceed 10 total
        extra = dict(result["stations"][1])  # copy GBG
        extra["code"] = "EX1"
        extra["name"] = "Extra Station 1"
        result["stations"].insert(1, extra)

        extra2 = dict(result["stations"][2])
        extra2["code"] = "EX2"
        extra2["name"] = "Extra Station 2"
        result["stations"].insert(2, extra2)

        assert len(result["stations"]) > 10

        text = render_to_text(
            tracker.build_stations_table(result, focus=True)
        )
        assert "departed stops hidden" in text
        # Current/future stations still visible
        assert "Huntingdon" in text
        assert "New York Penn" in text
