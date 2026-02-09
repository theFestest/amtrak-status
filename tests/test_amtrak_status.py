"""Comprehensive tests for amtrak_status.tracker module."""

import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

import amtrak_status.tracker as tracker

# Shared helpers from conftest (imported explicitly for use in test code)
from conftest import (
    make_station, make_train, ts_ms, FIXED_NOW,
    sample_journey_stations, render_to_text, journey_at_phase,
)


# =============================================================================
# TestParseTime
# =============================================================================


class TestParseTime:
    def test_none(self):
        assert tracker.parse_time(None) is None

    def test_int_ms_timestamp(self):
        # 2025-07-01 00:00:00 UTC — safely 2025 in all US timezones
        ms = 1751328000000
        dt = tracker.parse_time(ms)
        assert dt is not None
        assert dt.year == 2025
        assert dt.month in (6, 7)  # June 30 or July 1 depending on local tz

    def test_float_ms_timestamp(self):
        dt = tracker.parse_time(1735689600000.0)
        assert dt is not None

    def test_string_digit_timestamp(self):
        dt = tracker.parse_time("1735689600000")
        assert dt is not None

    def test_iso_string(self):
        dt = tracker.parse_time("2025-03-15T14:30:00")
        assert dt is not None
        assert dt.hour == 14
        assert dt.minute == 30

    def test_iso_string_with_z(self):
        dt = tracker.parse_time("2025-03-15T14:30:00Z")
        assert dt is not None
        assert dt.tzinfo is not None  # Z -> +00:00

    def test_iso_string_with_offset(self):
        dt = tracker.parse_time("2025-03-15T14:30:00-05:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_invalid_string(self):
        assert tracker.parse_time("not-a-time") is None

    def test_empty_string(self):
        # empty string is not digits, not ISO — should return None
        assert tracker.parse_time("") is None

    def test_negative_timestamp(self):
        # Negative timestamps are unusual but shouldn't crash
        result = tracker.parse_time(-1000)
        # May return a datetime or None depending on platform; just don't crash
        assert result is None or isinstance(result, datetime)


# =============================================================================
# TestFormatTime
# =============================================================================


class TestFormatTime:
    def test_none(self):
        assert tracker.format_time(None) == "—"

    def test_afternoon(self):
        dt = datetime(2025, 3, 15, 14, 30, 0)
        result = tracker.format_time(dt)
        assert result == "2:30 PM"

    def test_morning(self):
        dt = datetime(2025, 3, 15, 9, 5, 0)
        result = tracker.format_time(dt)
        assert result == "9:05 AM"

    def test_midnight(self):
        dt = datetime(2025, 3, 15, 0, 0, 0)
        result = tracker.format_time(dt)
        # 12:00 AM — strftime gives "12:00 AM", lstrip("0") won't strip "1"
        assert "12:00 AM" in result

    def test_noon(self):
        dt = datetime(2025, 3, 15, 12, 0, 0)
        result = tracker.format_time(dt)
        assert "12:00 PM" in result

    def test_leading_zero_stripped(self):
        dt = datetime(2025, 3, 15, 8, 0, 0)
        result = tracker.format_time(dt)
        # Should be "8:00 AM" not "08:00 AM"
        assert result.startswith("8:")


# =============================================================================
# TestIsStationCancelled
# =============================================================================


class TestIsStationCancelled:
    def test_normal_station_not_cancelled(self):
        station = make_station(status="Departed", sch_arr=100, sch_dep=200)
        assert tracker.is_station_cancelled(station) is False

    def test_enroute_station_not_cancelled(self):
        station = make_station(status="Enroute", sch_arr=100, sch_dep=200)
        assert tracker.is_station_cancelled(station) is False

    def test_cancel_in_status(self):
        station = make_station(status="Cancelled", sch_arr=100, sch_dep=200)
        assert tracker.is_station_cancelled(station) is True

    def test_skip_in_status(self):
        station = make_station(status="Skipped", sch_arr=100, sch_dep=200)
        assert tracker.is_station_cancelled(station) is True

    def test_cancel_case_insensitive(self):
        station = make_station(status="CANCELLED", sch_arr=100, sch_dep=200)
        assert tracker.is_station_cancelled(station) is True

    def test_no_scheduled_times_is_cancelled(self):
        station = make_station(status="", sch_arr=None, sch_dep=None)
        assert tracker.is_station_cancelled(station) is True

    def test_station_status_no_actual_times_is_cancelled(self):
        # Station status but no arr/dep means train never actually stopped
        station = make_station(
            status="Station", sch_arr=100, sch_dep=200, arr=None, dep=None
        )
        assert tracker.is_station_cancelled(station) is True

    def test_station_status_with_actual_times_not_cancelled(self):
        station = make_station(
            status="Station", sch_arr=100, sch_dep=200, arr=300, dep=None
        )
        assert tracker.is_station_cancelled(station) is False

    def test_origin_with_only_sch_dep(self):
        # Origin station: has schDep but no schArr — should NOT be cancelled
        station = make_station(status="Departed", sch_arr=None, sch_dep=100)
        assert tracker.is_station_cancelled(station) is False

    def test_empty_string_times_treated_as_falsy(self):
        # Empty strings are falsy in Python — same as None for this check
        station = make_station(status="", sch_arr="", sch_dep="")
        assert tracker.is_station_cancelled(station) is True


# =============================================================================
# TestGetStatusStyle
# =============================================================================


class TestGetStatusStyle:
    def test_departed(self):
        style, icon = tracker.get_status_style({"status": "Departed"})
        assert style == "green"
        assert icon == "✓"

    def test_enroute(self):
        style, icon = tracker.get_status_style({"status": "Enroute"})
        assert style == "yellow bold"
        assert icon == "→"

    def test_station(self):
        style, icon = tracker.get_status_style({"status": "Station"})
        assert style == "cyan bold"
        assert icon == "●"

    def test_empty_status(self):
        style, icon = tracker.get_status_style({"status": ""})
        assert style == "dim"
        assert icon == "○"

    def test_unknown_status(self):
        style, icon = tracker.get_status_style({"status": "SomethingElse"})
        assert style == "dim"
        assert icon == "○"

    def test_missing_status_key(self):
        style, icon = tracker.get_status_style({})
        assert style == "dim"
        assert icon == "○"


# =============================================================================
# TestFindStationIndex
# =============================================================================


class TestFindStationIndex:
    def test_found(self):
        stations = [make_station(code="AAA"), make_station(code="BBB"), make_station(code="CCC")]
        assert tracker.find_station_index(stations, "BBB") == 1

    def test_not_found(self):
        stations = [make_station(code="AAA"), make_station(code="BBB")]
        assert tracker.find_station_index(stations, "ZZZ") is None

    def test_case_insensitive(self):
        stations = [make_station(code="PGH"), make_station(code="NYP")]
        assert tracker.find_station_index(stations, "pgh") == 0

    def test_none_code(self):
        stations = [make_station(code="AAA")]
        assert tracker.find_station_index(stations, None) is None

    def test_empty_list(self):
        assert tracker.find_station_index([], "AAA") is None


# =============================================================================
# TestFindCurrentStationIndex
# =============================================================================


class TestFindCurrentStationIndex:
    def test_mid_journey(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="Departed", sch_arr=200, sch_dep=300),
            make_station(status="Enroute", sch_arr=400, sch_dep=500),
            make_station(status="", sch_arr=600),
        ]
        assert tracker.find_current_station_index(stations) == 2

    def test_all_departed(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="Departed", sch_arr=200, sch_dep=300),
        ]
        # Default to last station
        assert tracker.find_current_station_index(stations) == len(stations) - 1

    def test_all_future(self):
        stations = [
            make_station(status="", sch_arr=100, sch_dep=200),
            make_station(status="", sch_arr=300),
        ]
        assert tracker.find_current_station_index(stations) == 0

    def test_at_station(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="Station", sch_arr=200, arr=250),
            make_station(status="", sch_arr=400),
        ]
        assert tracker.find_current_station_index(stations) == 1

    def test_skips_cancelled(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            # Cancelled: no times
            make_station(status="", sch_arr=None, sch_dep=None),
            make_station(status="Enroute", sch_arr=200, sch_dep=300),
        ]
        # Should skip the cancelled station and find Enroute at index 2
        assert tracker.find_current_station_index(stations) == 2

    def test_empty_stations(self):
        assert tracker.find_current_station_index([]) == -1


# =============================================================================
# TestFilterStations
# =============================================================================


class TestFilterStations:
    def setup_method(self):
        self.stations = [
            make_station(code="A"),
            make_station(code="B"),
            make_station(code="C"),
            make_station(code="D"),
            make_station(code="E"),
        ]

    def test_no_filter(self):
        result, before, after = tracker.filter_stations(self.stations, None, None)
        assert len(result) == 5
        assert before == 0
        assert after == 0

    def test_from_only(self):
        result, before, after = tracker.filter_stations(self.stations, "B", None)
        assert len(result) == 4  # B, C, D, E
        assert result[0]["code"] == "B"
        assert before == 1
        assert after == 0

    def test_to_only(self):
        result, before, after = tracker.filter_stations(self.stations, None, "C")
        assert len(result) == 3  # A, B, C
        assert result[-1]["code"] == "C"
        assert before == 0
        assert after == 2

    def test_from_and_to(self):
        result, before, after = tracker.filter_stations(self.stations, "B", "D")
        assert len(result) == 3  # B, C, D
        assert result[0]["code"] == "B"
        assert result[-1]["code"] == "D"
        assert before == 1
        assert after == 1

    def test_swapped_from_to_corrected(self):
        # from > to should be auto-corrected
        result, before, after = tracker.filter_stations(self.stations, "D", "B")
        assert len(result) == 3  # B, C, D
        assert before == 1
        assert after == 1

    def test_not_found_from(self):
        result, before, after = tracker.filter_stations(self.stations, "ZZZ", "C")
        # ZZZ not found → start_idx defaults to 0
        assert result[0]["code"] == "A"
        assert result[-1]["code"] == "C"

    def test_not_found_to(self):
        result, before, after = tracker.filter_stations(self.stations, "B", "ZZZ")
        # ZZZ not found → end_idx defaults to last
        assert result[0]["code"] == "B"
        assert result[-1]["code"] == "E"

    def test_single_station_range(self):
        result, before, after = tracker.filter_stations(self.stations, "C", "C")
        assert len(result) == 1
        assert result[0]["code"] == "C"


# =============================================================================
# TestCalculateProgress
# =============================================================================


class TestCalculateProgress:
    def test_mid_journey(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="Departed", sch_arr=100, sch_dep=200),
            make_station(status="Enroute", sch_arr=200, sch_dep=300),
            make_station(status="", sch_arr=400),
        ]
        completed, current_idx, total = tracker.calculate_progress(stations)
        assert completed == 2
        assert current_idx == 2
        assert total == 4

    def test_all_departed(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="Departed", sch_arr=200, sch_dep=300),
        ]
        completed, current_idx, total = tracker.calculate_progress(stations)
        assert completed == 2
        assert total == 2

    def test_all_future(self):
        stations = [
            make_station(status="", sch_dep=100),
            make_station(status="", sch_arr=200),
        ]
        completed, current_idx, total = tracker.calculate_progress(stations)
        assert completed == 0
        assert total == 2

    def test_at_station(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="Station", sch_arr=200, arr=250),
            make_station(status="", sch_arr=400),
        ]
        completed, current_idx, total = tracker.calculate_progress(stations)
        assert completed == 1
        assert current_idx == 1  # Station is the "current"

    def test_cancelled_stations_excluded(self):
        stations = [
            make_station(status="Departed", sch_dep=100),
            make_station(status="", sch_arr=None, sch_dep=None),  # cancelled
            make_station(status="Enroute", sch_arr=300, sch_dep=400),
            make_station(status="", sch_arr=500),
        ]
        completed, current_idx, total = tracker.calculate_progress(stations)
        assert total == 3  # cancelled station excluded

    def test_empty_stations(self):
        completed, current_idx, total = tracker.calculate_progress([])
        assert total == 0
        assert completed == 0


# =============================================================================
# TestCalculatePositionBetweenStations
# =============================================================================


class TestCalculatePositionBetweenStations:
    def test_normal_position(self):
        """Train departed station A, heading to station B."""
        now = FIXED_NOW
        dep_time = ts_ms(now - timedelta(minutes=30))
        arr_time = ts_ms(now + timedelta(minutes=30))

        train = make_train(stations=[
            make_station(code="A", status="Departed", dep=dep_time, sch_dep=dep_time),
            make_station(code="B", status="Enroute", arr=arr_time, sch_arr=arr_time),
        ])

        result = tracker.calculate_position_between_stations(train)
        assert result is not None
        last_code, next_code, progress, mins_remaining = result
        assert last_code == "A"
        assert next_code == "B"
        assert progress == pytest.approx(0.5, abs=0.01)
        assert mins_remaining == 30

    def test_no_departed_station(self):
        train = make_train(stations=[
            make_station(code="A", status="Enroute", sch_arr=100),
            make_station(code="B", status="", sch_arr=200),
        ])
        assert tracker.calculate_position_between_stations(train) is None

    def test_no_future_station(self):
        train = make_train(stations=[
            make_station(code="A", status="Departed", dep=100, sch_dep=100),
        ])
        assert tracker.calculate_position_between_stations(train) is None

    def test_skips_cancelled_stations(self):
        now = FIXED_NOW
        dep_time = ts_ms(now - timedelta(minutes=20))
        arr_time = ts_ms(now + timedelta(minutes=40))

        train = make_train(stations=[
            make_station(code="A", status="Departed", dep=dep_time, sch_dep=dep_time),
            # Cancelled stop - should be skipped
            make_station(code="X", status="", sch_arr=None, sch_dep=None),
            make_station(code="B", status="Enroute", arr=arr_time, sch_arr=arr_time),
        ])

        result = tracker.calculate_position_between_stations(train)
        assert result is not None
        last_code, next_code, _, _ = result
        assert last_code == "A"
        assert next_code == "B"

    def test_zero_duration(self):
        """When dep and arr times are the same, progress should be 1.0."""
        now = FIXED_NOW
        same_time = ts_ms(now - timedelta(minutes=5))

        train = make_train(stations=[
            make_station(code="A", status="Departed", dep=same_time, sch_dep=same_time),
            make_station(code="B", status="Enroute", arr=same_time, sch_arr=same_time),
        ])

        result = tracker.calculate_position_between_stations(train)
        assert result is not None
        _, _, progress, mins = result
        assert progress == 1.0
        assert mins == 0


# =============================================================================
# TestFindOverlappingStations
# =============================================================================


class TestFindOverlappingStations:
    def test_overlap_exists(self):
        train1 = make_train(stations=[
            make_station(code="A"), make_station(code="B"), make_station(code="C"),
        ])
        train2 = make_train(stations=[
            make_station(code="B"), make_station(code="C"), make_station(code="D"),
        ])
        result = tracker.find_overlapping_stations(train1, train2)
        assert result == ["B", "C"]

    def test_no_overlap(self):
        train1 = make_train(stations=[make_station(code="A"), make_station(code="B")])
        train2 = make_train(stations=[make_station(code="C"), make_station(code="D")])
        assert tracker.find_overlapping_stations(train1, train2) == []

    def test_empty_stations(self):
        train1 = make_train(stations=[])
        train2 = make_train(stations=[make_station(code="A")])
        assert tracker.find_overlapping_stations(train1, train2) == []

    def test_preserves_train1_order(self):
        train1 = make_train(stations=[
            make_station(code="C"), make_station(code="B"), make_station(code="A"),
        ])
        train2 = make_train(stations=[
            make_station(code="A"), make_station(code="B"), make_station(code="C"),
        ])
        result = tracker.find_overlapping_stations(train1, train2)
        assert result == ["C", "B", "A"]


# =============================================================================
# TestGetStationTimes
# =============================================================================


class TestGetStationTimes:
    def test_found(self):
        train = make_train(stations=[
            make_station(code="PHL", sch_arr=1000, sch_dep=2000, arr=1500, dep=2500),
        ])
        sch_arr, sch_dep, arr, dep = tracker.get_station_times(train, "PHL")
        assert sch_arr is not None
        assert sch_dep is not None
        assert arr is not None
        assert dep is not None

    def test_not_found(self):
        train = make_train(stations=[make_station(code="PHL")])
        result = tracker.get_station_times(train, "ZZZ")
        assert result == (None, None, None, None)

    def test_case_insensitive(self):
        train = make_train(stations=[
            make_station(code="PHL", sch_arr=1000),
        ])
        sch_arr, _, _, _ = tracker.get_station_times(train, "phl")
        assert sch_arr is not None


# =============================================================================
# TestGetStationStatus
# =============================================================================


class TestGetStationStatus:
    def test_found(self):
        train = make_train(stations=[
            make_station(code="PHL", status="Departed"),
        ])
        assert tracker.get_station_status(train, "PHL") == "Departed"

    def test_not_found(self):
        train = make_train(stations=[make_station(code="PHL")])
        assert tracker.get_station_status(train, "ZZZ") == ""

    def test_case_insensitive(self):
        train = make_train(stations=[
            make_station(code="PHL", status="Enroute"),
        ])
        assert tracker.get_station_status(train, "phl") == "Enroute"


# =============================================================================
# TestCalculateLayover
# =============================================================================


class TestCalculateLayover:
    def _make_trains(self, arr_time, dep_time):
        """Helper: train1 arrives at PHL at arr_time, train2 departs PHL at dep_time."""
        train1 = make_train(
            train_num="42",
            stations=[make_station(code="PHL", name="Philadelphia", arr=arr_time, sch_arr=arr_time)],
        )
        train2 = make_train(
            train_num="178",
            stations=[make_station(code="PHL", name="Philadelphia", dep=dep_time, sch_dep=dep_time)],
        )
        return train1, train2

    def test_comfortable_layover(self):
        """Layover >= 60 min should be 'comfortable'."""
        now = datetime.now()
        arr = ts_ms(now)
        dep = ts_ms(now + timedelta(minutes=90))
        train1, train2 = self._make_trains(arr, dep)

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_minutes"] == 90
        assert result["layover_status"] == "comfortable"
        assert result["is_valid"] is True

    def test_risky_layover(self):
        """Layover < 30 min should be 'risky'."""
        now = datetime.now()
        arr = ts_ms(now)
        dep = ts_ms(now + timedelta(minutes=20))
        train1, train2 = self._make_trains(arr, dep)

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_minutes"] == 20
        assert result["layover_status"] == "risky"

    def test_missed_connection(self):
        """Negative layover (train2 departs before train1 arrives)."""
        now = datetime.now()
        arr = ts_ms(now + timedelta(minutes=30))
        dep = ts_ms(now)
        train1, train2 = self._make_trains(arr, dep)

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_minutes"] < 0
        assert result["layover_status"] == "missed"
        assert result["is_valid"] is False

    def test_tight_layover_30_to_44(self):
        """Layover 30-44 min should be 'tight'."""
        now = datetime.now()
        arr = ts_ms(now)
        dep = ts_ms(now + timedelta(minutes=35))
        train1, train2 = self._make_trains(arr, dep)

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_minutes"] == 35
        assert result["layover_status"] == "tight"

    @pytest.mark.xfail(
        reason="BUG: 45-59 min layover returns 'tight' instead of a distinct status. "
               "Lines 858-859 duplicate the 'tight' branch for the LAYOVER_TIGHT..LAYOVER_COMFORTABLE range."
    )
    def test_layover_45_to_59_should_not_be_tight(self):
        """Layover 45-59 min should be distinct from 30-44 min (currently both 'tight')."""
        now = datetime.now()
        arr = ts_ms(now)
        dep = ts_ms(now + timedelta(minutes=50))
        train1, train2 = self._make_trains(arr, dep)

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_minutes"] == 50
        # This SHOULD be "comfortable" or a middle category, not "tight"
        assert result["layover_status"] != "tight"

    def test_train2_already_departed_is_missed(self):
        """If train2 departed and train1 hasn't arrived, connection is missed."""
        now = datetime.now()
        arr = ts_ms(now + timedelta(minutes=30))
        dep = ts_ms(now - timedelta(minutes=10))

        train1 = make_train(stations=[
            make_station(code="PHL", name="Philadelphia", status="Enroute",
                         arr=arr, sch_arr=arr),
        ])
        train2 = make_train(stations=[
            make_station(code="PHL", name="Philadelphia", status="Departed",
                         dep=dep, sch_dep=dep),
        ])

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_status"] == "missed"
        assert result["is_valid"] is False

    def test_no_times_available(self):
        train1 = make_train(stations=[make_station(code="PHL")])
        train2 = make_train(stations=[make_station(code="PHL")])

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["layover_minutes"] is None
        assert result["layover_status"] == "unknown"

    def test_station_name_resolved(self):
        train1 = make_train(stations=[
            make_station(code="PHL", name="Philadelphia 30th Street"),
        ])
        train2 = make_train(stations=[make_station(code="PHL")])

        result = tracker.calculate_layover(train1, train2, "PHL")
        assert result["station_name"] == "Philadelphia 30th Street"


# =============================================================================
# TestGetTrainScheduleFromStation
# =============================================================================


class TestGetTrainScheduleFromStation:
    def test_found(self):
        station_data = {
            "PHL": [
                {"trainNum": "42", "schArr": 1000, "schDep": 2000, "arr": None, "dep": None, "status": ""},
                {"trainNum": "178", "schArr": 3000, "schDep": 4000, "arr": None, "dep": None, "status": ""},
            ]
        }
        result = tracker.get_train_schedule_from_station(station_data, "42")
        assert result is not None
        assert result["trainNum"] == "42"
        assert result["schArr"] == 1000

    def test_not_found(self):
        station_data = {
            "PHL": [{"trainNum": "42", "schArr": 1000}]
        }
        result = tracker.get_train_schedule_from_station(station_data, "999")
        assert result is None

    def test_error_in_data(self):
        result = tracker.get_train_schedule_from_station({"error": "fail"}, "42")
        assert result is None

    def test_none_data(self):
        result = tracker.get_train_schedule_from_station(None, "42")
        assert result is None

    def test_train_number_with_dash(self):
        """Train number like '42-26' should match train '42'."""
        station_data = {
            "PHL": [
                {"trainNum": "42-26", "schArr": 1000, "schDep": 2000},
            ]
        }
        result = tracker.get_train_schedule_from_station(station_data, "42")
        assert result is not None
        assert result["trainNum"] == "42-26"

    def test_non_list_value_skipped(self):
        station_data = {
            "metadata": "some string",
            "PHL": [{"trainNum": "42", "schArr": 1000}],
        }
        result = tracker.get_train_schedule_from_station(station_data, "42")
        assert result is not None


# =============================================================================
# TestBuildPredepartureTrainData
# =============================================================================


class TestBuildPredepartureTrainData:
    def test_with_schedule(self):
        schedule = {"schArr": 1000, "schDep": 2000, "arr": None, "dep": None}
        result = tracker.build_predeparture_train_data("42", "PHL", schedule)
        assert result["trainNum"] == "42"
        assert result["trainState"] == "Predeparture"
        assert result["_predeparture"] is True
        assert len(result["stations"]) == 1
        assert result["stations"][0]["code"] == "PHL"
        assert result["stations"][0]["schArr"] == 1000

    def test_without_schedule(self):
        result = tracker.build_predeparture_train_data("42", "PHL", None)
        assert result["trainNum"] == "42"
        assert result["stations"] == []


# =============================================================================
# TestInitializeNotificationState
# =============================================================================


class TestInitializeNotificationState:
    def test_marks_departed_as_seen(self):
        train = make_train(stations=[
            make_station(code="A", status="Departed", sch_dep=100),
            make_station(code="B", status="Departed", sch_arr=200, sch_dep=300),
            make_station(code="C", status="Enroute", sch_arr=400, sch_dep=500),
            make_station(code="D", status="", sch_arr=600),
        ])
        tracker.initialize_notification_state(train)
        assert "A" in tracker._notified_stations
        assert "B" in tracker._notified_stations
        assert "C" not in tracker._notified_stations
        assert "D" not in tracker._notified_stations
        assert tracker._notifications_initialized is True

    def test_idempotent(self):
        train = make_train(stations=[
            make_station(code="A", status="Departed", sch_dep=100),
        ])
        tracker.initialize_notification_state(train)
        first = tracker._notified_stations.copy()

        # Call again with different data — should not change anything
        train2 = make_train(stations=[
            make_station(code="A", status="Departed", sch_dep=100),
            make_station(code="B", status="Departed", sch_arr=200, sch_dep=300),
        ])
        tracker.initialize_notification_state(train2)
        assert tracker._notified_stations == first


# =============================================================================
# TestCheckAndNotify
# =============================================================================


class TestCheckAndNotify:
    def test_no_notify_config_returns_empty(self):
        train = make_train(stations=[
            make_station(code="A", status="Station", sch_arr=100, arr=200),
        ])
        result = tracker.check_and_notify(train)
        assert result == []

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_notifies_on_new_arrival(self, mock_notify):
        """Simulate: init with PHL as Enroute, then it becomes Station → notify."""
        tracker.NOTIFY_STATIONS = {"PHL"}

        # Step 1: Initialize with PHL still en route
        train_before = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="PHL", status="Enroute", sch_arr=200),
        ])
        tracker.check_and_notify(train_before)
        mock_notify.reset_mock()

        # Step 2: PHL is now at Station — should trigger notification
        train_after = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="PHL", status="Station", sch_arr=200, arr=300),
        ])
        result = tracker.check_and_notify(train_after)
        assert "PHL" in result
        assert mock_notify.called

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_does_not_renotify(self, mock_notify):
        """After notifying once, subsequent calls should not re-notify."""
        tracker.NOTIFY_STATIONS = {"PHL"}

        # Init with PHL enroute
        train_before = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="PHL", status="Enroute", sch_arr=200),
        ])
        tracker.check_and_notify(train_before)

        # PHL arrives — first notification
        train_after = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="PHL", status="Station", sch_arr=200, arr=300),
        ])
        tracker.check_and_notify(train_after)
        mock_notify.reset_mock()

        # Third call — should not re-notify
        result = tracker.check_and_notify(train_after)
        assert result == []
        assert not mock_notify.called

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_notify_all(self, mock_notify):
        """With NOTIFY_ALL, new arrivals should trigger notifications."""
        tracker.NOTIFY_ALL = True

        # Init with HBG enroute
        train_before = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="HBG", status="Enroute", sch_arr=200),
            make_station(code="PHL", status="", sch_arr=400),
        ])
        tracker.check_and_notify(train_before)
        mock_notify.reset_mock()

        # HBG now at Station
        train_after = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="HBG", status="Station", sch_arr=200, arr=300),
            make_station(code="PHL", status="", sch_arr=400),
        ])
        result = tracker.check_and_notify(train_after)
        assert "HBG" in result

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_does_not_notify_for_unspecified_station(self, mock_notify):
        tracker.NOTIFY_STATIONS = {"NYP"}
        train = make_train(stations=[
            make_station(code="PGH", status="Departed", sch_dep=100),
            make_station(code="PHL", status="Station", sch_arr=200, arr=300),
        ])
        result = tracker.check_and_notify(train)
        assert "PHL" not in result


# =============================================================================
# TestSendNotification
# =============================================================================


class TestSendNotification:
    @patch("amtrak_status.tracker.sys")
    @patch("amtrak_status.tracker.subprocess.run")
    def test_macos(self, mock_run, mock_sys):
        mock_sys.platform = "darwin"
        result = tracker.send_notification("Title", "Message")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert "osascript" in args[0][0]

    @patch("amtrak_status.tracker.sys")
    @patch("amtrak_status.tracker.subprocess.run")
    def test_linux(self, mock_run, mock_sys):
        mock_sys.platform = "linux"
        result = tracker.send_notification("Title", "Message")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert "notify-send" in args[0][0]

    @patch("amtrak_status.tracker.sys")
    @patch("amtrak_status.tracker.subprocess.run", side_effect=FileNotFoundError)
    def test_fallback_to_bell(self, mock_run, mock_sys):
        mock_sys.platform = "darwin"
        result = tracker.send_notification("Title", "Message")
        assert result is False


# =============================================================================
# TestFetchTrainData
# =============================================================================


class TestFetchTrainData:
    @patch("amtrak_status.tracker.httpx.Client")
    def test_success_direct_key(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "42": [{"trainNum": "42", "routeName": "Pennsylvanian", "stations": []}]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert result["trainNum"] == "42"

    @patch("amtrak_status.tracker.httpx.Client")
    def test_success_single_key_response(self, mock_client_cls):
        """API sometimes returns a single key that doesn't match the query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "42-1": [{"trainNum": "42", "routeName": "Pennsylvanian", "stations": []}]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_train_data("42")
        assert result is not None

    @patch("amtrak_status.tracker.sleep")
    @patch("amtrak_status.tracker.httpx.Client")
    def test_retry_on_http_error(self, mock_client_cls, mock_sleep):
        """Should retry up to MAX_RETRIES times on HTTP errors."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        error_response = MagicMock()
        error_response.status_code = 500
        mock_client.get.side_effect = httpx_status_error(500)
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_train_data("42")
        assert result is not None
        assert "error" in result
        assert mock_client.get.call_count == tracker.MAX_RETRIES

    @patch("amtrak_status.tracker.httpx.Client")
    def test_not_found_returns_none(self, mock_client_cls):
        """Empty API response → None."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_train_data("999")
        assert result is None

    @patch("amtrak_status.tracker.sleep")
    @patch("amtrak_status.tracker.httpx.Client")
    def test_cache_fallback_on_failure(self, mock_client_cls, mock_sleep):
        """Should use cached data when API fails and cache is fresh."""
        cached_data = {"trainNum": "42", "stations": []}
        tracker._train_caches["42"] = {
            "data": cached_data,
            "fetch_time": FIXED_NOW,
            "error": None,
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx_http_error()
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_train_data("42")
        assert result == cached_data


def httpx_status_error(status_code):
    """Create an httpx.HTTPStatusError side_effect for mocking."""
    import httpx

    response = httpx.Response(status_code, request=httpx.Request("GET", "http://test"))
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=response.request, response=response
    )


def httpx_http_error():
    """Create a generic httpx.HTTPError for mocking."""
    import httpx

    return httpx.HTTPError("Connection failed")


# =============================================================================
# TestFetchStationSchedule
# =============================================================================


class TestFetchStationSchedule:
    @patch("amtrak_status.tracker.httpx.Client")
    def test_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "PHL": [{"trainNum": "42", "schArr": 1000}]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_station_schedule("phl")
        assert result is not None
        assert "PHL" in result

    @patch("amtrak_status.tracker.httpx.Client")
    def test_error(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.HTTPError("fail")
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_station_schedule("PHL")
        assert result is not None
        assert "error" in result


# =============================================================================
# Rich Display Tests
# =============================================================================


class TestBuildHeader:
    def test_predeparture_header_content(self):
        train = make_train(
            train_state="Predeparture", status_msg="", velocity=0,
            stations=[make_station(code="PGH", status="", sch_dep=100)],
        )
        panel = tracker.build_header(train)
        text = render_to_text(panel)
        assert "Predeparture" in text
        assert "Position:" not in text  # no position bar for predeparture


class TestBuildProgressBar:
    def test_normal_shows_origin_and_dest(self):
        train = make_train(stations=sample_journey_stations())
        panel = tracker.build_progress_bar(train)
        text = render_to_text(panel)
        assert "Pittsburgh" in text
        assert "Journey Progress" in text

    def test_empty_stations_shows_no_data(self):
        train = make_train(stations=[])
        panel = tracker.build_progress_bar(train)
        text = render_to_text(panel)
        assert "No station data" in text


class TestBuildStationsTable:
    def test_normal(self):
        train = make_train(stations=sample_journey_stations())
        result = tracker.build_stations_table(train)
        assert isinstance(result, Panel)

    def test_with_cancelled_station(self):
        stations = sample_journey_stations()
        # Add a cancelled station
        stations.insert(2, make_station(code="XXX", name="Cancelled Stop"))
        train = make_train(stations=stations)
        result = tracker.build_stations_table(train)
        assert isinstance(result, Panel)

    def test_with_station_filter(self):
        tracker.STATION_FROM = "GBG"
        tracker.STATION_TO = "PHL"
        train = make_train(stations=sample_journey_stations())
        result = tracker.build_stations_table(train)
        assert isinstance(result, Panel)

    def test_focus_mode_hides_old_departed(self):
        """With many stations, focus mode should hide old departed stops."""
        # Build a long station list
        stations = []
        for i in range(15):
            if i < 10:
                stations.append(make_station(
                    code=f"S{i:02d}", status="Departed", sch_dep=i * 100,
                ))
            elif i == 10:
                stations.append(make_station(
                    code=f"S{i:02d}", status="Enroute", sch_arr=i * 100, sch_dep=i * 100 + 50,
                ))
            else:
                stations.append(make_station(
                    code=f"S{i:02d}", status="", sch_arr=i * 100,
                ))
        train = make_train(stations=stations)
        result = tracker.build_stations_table(train, focus=True)
        assert isinstance(result, Panel)

    def test_no_focus_mode(self):
        tracker.FOCUS_CURRENT = False
        train = make_train(stations=sample_journey_stations())
        result = tracker.build_stations_table(train, focus=False)
        assert isinstance(result, Panel)


class TestBuildCompactDisplay:
    def test_normal(self):
        train = make_train(stations=sample_journey_stations())
        result = tracker.build_compact_display(train)
        assert isinstance(result, Text)
        assert "Pennsylvanian" in result.plain

    def test_no_stations(self):
        train = make_train(stations=[])
        result = tracker.build_compact_display(train)
        assert isinstance(result, Text)


class TestBuildPredeparturePanel:
    def test_content(self):
        panel = tracker.build_predeparture_panel("42")
        text = render_to_text(panel)
        assert "#42" in text
        assert "Awaiting Departure" in text
        assert "Predeparture" in text

class TestBuildPredepartureHeader:
    def test_content(self):
        panel = tracker.build_predeparture_header("42")
        text = render_to_text(panel)
        assert "#42" in text
        assert "Awaiting Departure" in text


class TestBuildCompactTrainHeader:
    def test_active_train(self):
        train = make_train(stations=sample_journey_stations())
        result = tracker.build_compact_train_header(train)
        assert isinstance(result, Panel)

    def test_predeparture_synthetic(self):
        train = make_train(
            train_state="Predeparture",
            stations=[make_station(code="PHL", sch_dep=ts_ms(FIXED_NOW))],
        )
        train["_predeparture"] = True
        result = tracker.build_compact_train_header(train)
        assert isinstance(result, Panel)

    def test_predeparture_with_sch_arr(self):
        """Predeparture train with only schArr, not schDep."""
        train = make_train(
            train_state="Predeparture",
            stations=[make_station(code="PHL", sch_arr=ts_ms(FIXED_NOW))],
        )
        train["_predeparture"] = True
        result = tracker.build_compact_train_header(train)
        assert isinstance(result, Panel)


class TestBuildDisplay:
    @patch("amtrak_status.tracker.fetch_train_data")
    def test_normal(self, mock_fetch):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        result = tracker.build_display("42")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_not_found(self, mock_fetch):
        mock_fetch.return_value = None
        result = tracker.build_display("999")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_error(self, mock_fetch):
        mock_fetch.return_value = {"error": "HTTP 500"}
        result = tracker.build_display("42")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_compact_mode(self, mock_fetch):
        tracker.COMPACT_MODE = True
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        result = tracker.build_display("42")
        assert isinstance(result, Text)

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_compact_not_found(self, mock_fetch):
        tracker.COMPACT_MODE = True
        mock_fetch.return_value = None
        result = tracker.build_display("42")
        assert isinstance(result, Text)

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_compact_error(self, mock_fetch):
        tracker.COMPACT_MODE = True
        mock_fetch.return_value = {"error": "fail"}
        result = tracker.build_display("42")
        assert isinstance(result, Text)


# =============================================================================
# TestMultiTrainDisplay
# =============================================================================


class TestBuildMultiTrainDisplay:
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_both_trains_valid(self, mock_fetch):
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", status="Departed", sch_dep=ts_ms(now - timedelta(hours=3)),
                             dep=ts_ms(now - timedelta(hours=3))),
                make_station(code="PHL", status="Enroute",
                             sch_arr=ts_ms(now + timedelta(hours=1)),
                             arr=ts_ms(now + timedelta(hours=1))),
                make_station(code="NYP", status="", sch_arr=ts_ms(now + timedelta(hours=3))),
            ],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", status="",
                             sch_dep=ts_ms(now + timedelta(hours=2))),
                make_station(code="HBG", status="", sch_arr=ts_ms(now + timedelta(hours=4))),
            ],
        )
        mock_fetch.side_effect = [train1, train2]

        result = tracker.build_multi_train_display(["42", "178"], "PHL")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_neither_train_valid(self, mock_fetch):
        mock_fetch.return_value = None
        result = tracker.build_multi_train_display(["42", "178"], "PHL")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_only_train1_valid(self, mock_fetch):
        train1 = make_train(
            train_num="42",
            stations=[
                make_station(code="PGH", status="Departed", sch_dep=100),
                make_station(code="PHL", status="Enroute", sch_arr=200, arr=300),
            ],
        )
        mock_fetch.side_effect = [train1, None]

        result = tracker.build_multi_train_display(["42", "178"], "PHL")
        assert isinstance(result, Layout)

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_only_train2_valid(self, mock_fetch):
        train2 = make_train(
            train_num="178",
            stations=[
                make_station(code="PHL", status="", sch_dep=200),
                make_station(code="HBG", status="", sch_arr=300),
            ],
        )
        mock_fetch.side_effect = [None, train2]

        result = tracker.build_multi_train_display(["42", "178"], "PHL")
        assert isinstance(result, Layout)


# =============================================================================
# TestMainArgParsing
# =============================================================================


class TestMainArgParsing:
    """Test that main() sets globals correctly from CLI args."""

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_single_train_once(self, mock_console_cls, mock_fetch, mock_live):
        """--once mode should display and return without looping."""
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--once"]):
            tracker.main()

        mock_console.print.assert_called()
        mock_live.assert_not_called()

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_compact_once(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--compact", "--once"]):
            tracker.main()

        assert tracker.COMPACT_MODE is True

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_from_to_args(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--from", "PGH", "--to", "NYP", "--once"]):
            tracker.main()

        assert tracker.STATION_FROM == "PGH"
        assert tracker.STATION_TO == "NYP"

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_notify_at_arg(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--notify-at", "PGH,NYP", "--once"]):
            tracker.main()

        assert tracker.NOTIFY_STATIONS == {"PGH", "NYP"}

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_notify_all_arg(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--notify-all", "--once"]):
            tracker.main()

        assert tracker.NOTIFY_ALL is True

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_refresh_interval(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "-r", "60", "--once"]):
            tracker.main()

        assert tracker.REFRESH_INTERVAL == 60

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_all_flag_disables_focus(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--all", "--once"]):
            tracker.main()

        assert tracker.FOCUS_CURRENT is False

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_no_focus_flag(self, mock_console_cls, mock_fetch, mock_live):
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--no-focus", "--once"]):
            tracker.main()

        assert tracker.FOCUS_CURRENT is False


# =============================================================================
# Edge case / integration tests
# =============================================================================


class TestTimezoneEdgeCases:
    """Test that mixed timezone scenarios are handled."""

    def test_parse_time_unix_is_naive(self):
        """Unix timestamps produce naive datetimes (no tzinfo)."""
        dt = tracker.parse_time(1735689600000)
        assert dt is not None
        assert dt.tzinfo is None

    def test_parse_time_iso_z_is_aware(self):
        """ISO with Z produces timezone-aware datetimes."""
        dt = tracker.parse_time("2025-03-15T14:30:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    @pytest.mark.xfail(
        reason="BUG: parse_time returns naive dt for unix timestamps but aware dt "
               "for ISO+Z strings. Comparing these downstream raises TypeError."
    )
    def test_comparing_naive_and_aware_raises(self):
        """Mixing naive (unix) and aware (ISO+Z) datetimes should not happen."""
        naive = tracker.parse_time(1735689600000)
        aware = tracker.parse_time("2025-03-15T14:30:00Z")
        assert naive is not None
        assert aware is not None
        # This comparison would raise TypeError in real code
        # The test documents that this is a bug — both should be consistent
        _ = naive < aware  # Should not raise if bug is fixed


# =============================================================================
# Cache and fetch coverage
# =============================================================================


class TestCacheExpiry:
    @patch("amtrak_status.tracker.sleep")
    @patch("amtrak_status.tracker.httpx.Client")
    def test_stale_cache_not_used(self, mock_client_cls, mock_sleep):
        """Cache older than 300 seconds should NOT be used as fallback."""
        cached_data = {"trainNum": "42", "stations": []}
        tracker._train_caches["42"] = {
            "data": cached_data,
            "fetch_time": FIXED_NOW - timedelta(seconds=301),
            "error": None,
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx_http_error()
        mock_client_cls.return_value = mock_client

        result = tracker.fetch_train_data("42")
        # Should NOT return cached data since it's stale
        assert result != cached_data
        assert result is not None and "error" in result


class TestFetchTrainDataCachedErrorPath:
    @patch("amtrak_status.tracker.fetch_train_data")
    def test_cached_returns_cache_on_error_result(self, mock_fetch):
        """When fetch returns an error dict but cache is fresh, use cache."""
        cached = {"trainNum": "42", "stations": [], "routeName": "Pennsylvanian"}
        tracker._train_caches["42"] = {
            "data": cached,
            "fetch_time": FIXED_NOW,
            "error": None,
        }
        mock_fetch.return_value = {"error": "HTTP 500"}

        result = tracker.fetch_train_data_cached("42")
        assert result == cached  # should use cache, not error


# =============================================================================
# CLI / main() coverage
# =============================================================================


class TestMultiTrainArgParsing:
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.Console")
    def test_multi_train_connection_arg(
        self, mock_console_cls, mock_fetch_cached, mock_fetch, mock_live
    ):
        """--connection sets CONNECTION_STATION."""
        train1 = make_train(
            train_num="42",
            stations=[make_station(code="PHL", status="Enroute", sch_arr=100, arr=200)],
        )
        train2 = make_train(
            train_num="178",
            stations=[make_station(code="PHL", status="", sch_dep=300)],
        )
        mock_fetch.side_effect = [train1, train2]
        mock_fetch_cached.side_effect = [train1, train2]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "178", "--connection", "PHL", "--once"]):
            tracker.main()

        assert tracker.CONNECTION_STATION == "PHL"

    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Console")
    def test_two_train_numbers_triggers_multi_mode(
        self, mock_console_cls, mock_fetch, mock_live
    ):
        """Providing two train numbers should use multi-train display."""
        mock_fetch.return_value = None
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with (
            patch("sys.argv", ["amtrak-status", "42", "178", "--connection", "PHL", "--once"]),
            patch("amtrak_status.tracker.fetch_train_data_cached", return_value=None),
        ):
            tracker.main()

        assert tracker.CONNECTION_STATION == "PHL"


class TestSelectConnectionStation:
    def test_numeric_selection(self):
        from rich.console import Console
        console = Console(file=open("/dev/null", "w"))
        with patch("amtrak_status.tracker.Prompt.ask", return_value="2"):
            result = tracker.select_connection_station(
                console, ["PHL", "NYP", "WAS"],
                make_train(stations=[
                    make_station(code="PHL", name="Philadelphia"),
                    make_station(code="NYP", name="New York Penn"),
                    make_station(code="WAS", name="Washington"),
                ]),
                make_train(stations=[]),
            )
        assert result == "NYP"

    def test_code_selection(self):
        from rich.console import Console
        console = Console(file=open("/dev/null", "w"))
        with patch("amtrak_status.tracker.Prompt.ask", return_value="WAS"):
            result = tracker.select_connection_station(
                console, ["PHL", "NYP", "WAS"],
                make_train(stations=[
                    make_station(code="PHL", name="Philadelphia"),
                    make_station(code="NYP", name="New York Penn"),
                    make_station(code="WAS", name="Washington"),
                ]),
                make_train(stations=[]),
            )
        assert result == "WAS"


# =============================================================================
# Edge cases
# =============================================================================


class TestFilterStationsEmptyString:
    def test_empty_string_from(self):
        """Empty string should behave like None (no filter)."""
        stations = [
            make_station(code="A"),
            make_station(code="B"),
            make_station(code="C"),
        ]
        result, before, after = tracker.filter_stations(stations, "", None)
        # Empty string is falsy, so it should behave like None (no filter)
        assert len(result) == 3
        assert before == 0
        assert after == 0


# =============================================================================
# Journey Phase Rendering Tests
# =============================================================================


class TestJourneyPhaseHeaderRendering:
    """Test that the header panel renders correct content at each journey phase."""

    def test_predeparture_shows_status_and_no_position(self):
        train = journey_at_phase("predeparture")
        text = render_to_text(tracker.build_header(train))
        assert "Predeparture" in text
        assert "Position:" not in text

    def test_predeparture_shows_first_station_as_next(self):
        train = journey_at_phase("predeparture")
        text = render_to_text(tracker.build_header(train))
        assert "Next: Pittsburgh" in text

    def test_early_shows_next_and_position(self):
        train = journey_at_phase("early")
        text = render_to_text(tracker.build_header(train))
        assert "On Time" in text
        assert "Next: Greensburg" in text
        assert "Position:" in text
        assert "PGH" in text

    def test_early_shows_speed(self):
        train = journey_at_phase("early")
        text = render_to_text(tracker.build_header(train))
        assert "55 mph" in text

    def test_mid_shows_next_station_and_speed(self):
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_header(train))
        assert "Next: Harrisburg" in text
        assert "62 mph" in text

    def test_mid_shows_position_between_gbg_and_hbg(self):
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_header(train))
        assert "Position:" in text
        assert "GBG" in text
        assert "HBG" in text

    def test_mid_position_shows_minutes_remaining(self):
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_header(train))
        assert "(35 min)" in text

    def test_mid_late_shows_delay_status(self):
        train = journey_at_phase("mid_late")
        text = render_to_text(tracker.build_header(train))
        assert "25 Minutes Late" in text

    def test_mid_late_shows_delay_in_eta(self):
        train = journey_at_phase("mid_late")
        text = render_to_text(tracker.build_header(train))
        assert "(+25m)" in text

    def test_arriving_shows_philadelphia_as_next(self):
        train = journey_at_phase("arriving")
        text = render_to_text(tracker.build_header(train))
        assert "Next: Philadelphia" in text

    def test_arriving_shows_early_arrival_indicator(self):
        train = journey_at_phase("arriving")
        text = render_to_text(tracker.build_header(train))
        # PHL arrived 2 min early
        assert "(-2m)" in text

    def test_arriving_no_speed_displayed(self):
        """velocity=0 renders speed as dash, not 'mph'."""
        train = journey_at_phase("arriving")
        text = render_to_text(tracker.build_header(train))
        assert "mph" not in text

    def test_final_leg_shows_nyp_and_speed(self):
        train = journey_at_phase("final_leg")
        text = render_to_text(tracker.build_header(train))
        assert "Next: New York Penn" in text
        assert "70 mph" in text

    def test_final_leg_position_shows_phl_nyp(self):
        train = journey_at_phase("final_leg")
        text = render_to_text(tracker.build_header(train))
        assert "Position:" in text
        assert "PHL" in text
        assert "NYP" in text

    @pytest.mark.parametrize("phase", [
        "predeparture", "early", "mid", "mid_late", "arriving", "final_leg",
    ])
    def test_all_phases_show_route_and_destination(self, phase):
        train = journey_at_phase(phase)
        text = render_to_text(tracker.build_header(train))
        assert "Pennsylvanian" in text
        assert "New York Penn" in text


class TestJourneyPhaseProgressBar:
    """Test that progress bar shows correct completion at each phase."""

    @pytest.mark.parametrize("phase,expected_pct", [
        ("predeparture", "0%"),
        ("early", "20%"),
        ("mid", "40%"),
        ("mid_late", "40%"),
        ("arriving", "60%"),
        ("final_leg", "80%"),
    ])
    def test_progress_percentage(self, phase, expected_pct):
        train = journey_at_phase(phase)
        text = render_to_text(tracker.build_progress_bar(train))
        assert expected_pct in text
        assert "Journey Progress" in text

    @pytest.mark.parametrize("phase", [
        "predeparture", "early", "mid", "mid_late", "arriving", "final_leg",
    ])
    def test_progress_shows_origin(self, phase):
        train = journey_at_phase(phase)
        text = render_to_text(tracker.build_progress_bar(train))
        assert "Pittsburgh" in text


class TestJourneyPhaseStationsTable:
    """Test that the stations table shows correct statuses and icons at each phase."""

    def test_predeparture_all_scheduled(self):
        train = journey_at_phase("predeparture")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert text.count("Scheduled") == 5
        assert "Departed" not in text
        assert "Enroute" not in text

    def test_predeparture_only_open_circle_icons(self):
        train = journey_at_phase("predeparture")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert "○" in text
        assert "✓" not in text
        assert "→" not in text
        assert "●" not in text

    def test_early_one_departed_one_enroute(self):
        train = journey_at_phase("early")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert text.count("Departed") == 1
        assert text.count("Enroute") == 1
        assert text.count("Scheduled") == 3
        assert "✓" in text
        assert "→" in text

    def test_mid_two_departed_one_enroute(self):
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert text.count("Departed") == 2
        assert text.count("Enroute") == 1
        assert text.count("Scheduled") == 2

    def test_arriving_three_departed_at_station(self):
        train = journey_at_phase("arriving")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert text.count("Departed") == 3
        assert "●" in text  # PHL at station
        assert text.count("Scheduled") == 1  # NYP only

    def test_final_leg_four_departed_one_enroute(self):
        train = journey_at_phase("final_leg")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert text.count("Departed") == 4
        assert text.count("Enroute") == 1

    @pytest.mark.parametrize("phase", [
        "predeparture", "early", "mid", "mid_late", "arriving", "final_leg",
    ])
    def test_all_phases_list_all_stations(self, phase):
        """Every phase should list all 5 station names and codes when focus=False."""
        train = journey_at_phase(phase)
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        for name in ("Pittsburgh", "Greensburg", "Harrisburg", "Philadelphia", "New York Penn"):
            assert name in text, f"{name} missing in {phase}"
        for code in ("PGH", "GBG", "HBG", "PHL", "NYP"):
            assert code in text, f"{code} missing in {phase}"


class TestJourneyPhaseCompactDisplay:
    """Test compact single-line display at key phases."""

    @pytest.mark.parametrize("phase", [
        "predeparture", "early", "mid", "mid_late", "arriving", "final_leg",
    ])
    def test_compact_shows_route_and_train(self, phase):
        train = journey_at_phase(phase)
        text = render_to_text(tracker.build_compact_display(train))
        assert "Pennsylvanian" in text
        assert "#42" in text

    @pytest.mark.parametrize("phase,expected_pct", [
        ("predeparture", "0%"),
        ("early", "20%"),
        ("mid", "40%"),
        ("arriving", "60%"),
        ("final_leg", "80%"),
    ])
    def test_compact_progress_percentage(self, phase, expected_pct):
        train = journey_at_phase(phase)
        text = render_to_text(tracker.build_compact_display(train))
        assert expected_pct in text

    def test_compact_mid_late_shows_delay(self):
        train = journey_at_phase("mid_late")
        text = render_to_text(tracker.build_compact_display(train))
        assert "25 Minutes Late" in text
        assert "+25m" in text

    def test_compact_early_shows_speed(self):
        train = journey_at_phase("early")
        text = render_to_text(tracker.build_compact_display(train))
        assert "55mph" in text

    def test_compact_final_leg_shows_speed(self):
        train = journey_at_phase("final_leg")
        text = render_to_text(tracker.build_compact_display(train))
        assert "70mph" in text

    def test_compact_mid_shows_position_between_stations(self):
        """Mid phase should show inter-station position with GBG and HBG."""
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_compact_display(train))
        assert "GBG" in text
        assert "HBG" in text

    def test_compact_predeparture_shows_next_station(self):
        """Predeparture has no position info, so should fall back to 'Next:'."""
        train = journey_at_phase("predeparture")
        text = render_to_text(tracker.build_compact_display(train))
        assert "Next:" in text


class TestRenderedContentDetails:
    """Test content details and relative positioning in rendered output."""

    def test_stations_appear_in_route_order(self):
        """Station rows should appear top-to-bottom in route order."""
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        names = ["Pittsburgh", "Greensburg", "Harrisburg", "Philadelphia", "New York Penn"]
        positions = [text.index(name) for name in names]
        assert positions == sorted(positions)

    def test_departed_stations_show_formatted_times(self):
        """Departed stations should show human-readable AM/PM times, not raw timestamps."""
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert "AM" in text or "PM" in text

    def test_cancelled_station_renders_with_label_and_icon(self):
        """A cancelled station should show 'Cancelled' text and ✗ icon."""
        stations = sample_journey_stations()
        stations.insert(2, make_station(code="XXX", name="Cancelled Stop"))
        train = make_train(stations=stations)
        text = render_to_text(tracker.build_stations_table(train, focus=False))
        assert "Cancelled" in text
        assert "✗" in text
        assert "Cancelled Stop" in text

    def test_focus_mode_shows_elision_message(self):
        """With >10 stations and focus mode, old departed stops should be hidden."""
        stations = []
        for i in range(15):
            if i < 10:
                stations.append(make_station(
                    code=f"S{i:02d}", name=f"Station {i:02d}",
                    status="Departed", sch_dep=i * 100,
                ))
            elif i == 10:
                stations.append(make_station(
                    code=f"S{i:02d}", name=f"Station {i:02d}",
                    status="Enroute", sch_arr=i * 100, sch_dep=i * 100 + 50,
                ))
            else:
                stations.append(make_station(
                    code=f"S{i:02d}", name=f"Station {i:02d}",
                    status="", sch_arr=i * 100,
                ))
        train = make_train(stations=stations)
        text = render_to_text(tracker.build_stations_table(train, focus=True))
        assert "departed stops hidden" in text

    def test_station_filter_hides_omitted_and_shows_indicators(self):
        """Station filter should show omission indicators and hide filtered-out stations."""
        tracker.STATION_FROM = "GBG"
        tracker.STATION_TO = "PHL"
        train = journey_at_phase("mid")
        text = render_to_text(tracker.build_stations_table(train))
        assert "earlier stops omitted" in text
        assert "later stops omitted" in text
        assert "Greensburg" in text
        assert "Philadelphia" in text
        assert "Pittsburgh (PGH)" not in text
        assert "New York Penn (NYP)" not in text


# =============================================================================
# TestBuildConnectionPanel
# =============================================================================


class TestBuildConnectionPanel:
    """Test rendering of build_connection_panel for various layover scenarios."""

    @staticmethod
    def _make_connection_trains(arr_time, dep_time, train1_status="Enroute", train2_status=""):
        """Build train1 arriving at PHL and train2 departing PHL."""
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(FIXED_NOW - timedelta(hours=3)),
                             dep=ts_ms(FIXED_NOW - timedelta(hours=3))),
                make_station(code="PHL", name="Philadelphia", status=train1_status,
                             sch_arr=arr_time, arr=arr_time),
            ],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", name="Philadelphia", status=train2_status,
                             sch_dep=dep_time, dep=dep_time if train2_status == "Departed" else None),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(FIXED_NOW + timedelta(hours=4))),
            ],
        )
        return train1, train2

    def test_comfortable_layover_rendering(self):
        arr = ts_ms(FIXED_NOW)
        dep = ts_ms(FIXED_NOW + timedelta(minutes=90))
        train1, train2 = self._make_connection_trains(arr, dep)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "1h 30m layover" in text
        assert "#42" in text
        assert "#178" in text
        assert "Connection at Philadelphia (PHL)" in text

    def test_risky_layover_rendering(self):
        arr = ts_ms(FIXED_NOW)
        dep = ts_ms(FIXED_NOW + timedelta(minutes=20))
        train1, train2 = self._make_connection_trains(arr, dep)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "20 min layover" in text
        assert "risky!" in text
        assert "⚠" in text

    def test_tight_layover_rendering(self):
        arr = ts_ms(FIXED_NOW)
        dep = ts_ms(FIXED_NOW + timedelta(minutes=35))
        train1, train2 = self._make_connection_trains(arr, dep)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "35 min layover" in text
        assert "tight" in text
        assert "⚡" in text

    def test_missed_layover_rendering(self):
        arr = ts_ms(FIXED_NOW + timedelta(minutes=30))
        dep = ts_ms(FIXED_NOW)
        train1, train2 = self._make_connection_trains(arr, dep)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "MISSED" in text
        assert "✗" in text

    def test_unknown_layover_rendering(self):
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[make_station(code="PHL", name="Philadelphia", status="Enroute")],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[make_station(code="PHL", name="Philadelphia", status="")],
        )
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Layover unknown" in text

    def test_train1_arrived_shows_icon(self):
        arr = ts_ms(FIXED_NOW - timedelta(minutes=10))
        dep = ts_ms(FIXED_NOW + timedelta(minutes=90))
        train1, train2 = self._make_connection_trains(arr, dep, train1_status="Departed")
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Arrived" in text

    def test_train1_at_station_shows_icon(self):
        arr = ts_ms(FIXED_NOW)
        dep = ts_ms(FIXED_NOW + timedelta(minutes=90))
        train1, train2 = self._make_connection_trains(arr, dep, train1_status="Station")
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "At station" in text
        assert "●" in text

    def test_train1_expected_shows_icon(self):
        arr = ts_ms(FIXED_NOW + timedelta(minutes=30))
        dep = ts_ms(FIXED_NOW + timedelta(minutes=120))
        train1, train2 = self._make_connection_trains(arr, dep, train1_status="Enroute")
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Expected" in text

    def test_train2_scheduled_shows_label(self):
        arr = ts_ms(FIXED_NOW)
        dep = ts_ms(FIXED_NOW + timedelta(minutes=90))
        train1, train2 = self._make_connection_trains(arr, dep, train2_status="")
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Scheduled" in text

    def test_train2_boarding_shows_label(self):
        arr = ts_ms(FIXED_NOW - timedelta(minutes=10))
        dep = ts_ms(FIXED_NOW + timedelta(minutes=10))
        train1, train2 = self._make_connection_trains(arr, dep, train1_status="Departed", train2_status="Station")
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Boarding" in text

    def test_train2_departed_missed_shows_x(self):
        arr = ts_ms(FIXED_NOW + timedelta(minutes=30))
        dep = ts_ms(FIXED_NOW - timedelta(minutes=10))
        train1, train2 = self._make_connection_trains(arr, dep, train1_status="Enroute", train2_status="Departed")
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "MISSED" in text or "Departed" in text
        assert "✗" in text

    def test_panel_title_has_station(self):
        arr = ts_ms(FIXED_NOW)
        dep = ts_ms(FIXED_NOW + timedelta(minutes=60))
        train1, train2 = self._make_connection_trains(arr, dep)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Connection at Philadelphia (PHL)" in text


# =============================================================================
# TestBuildErrorPanel and TestBuildNotFoundPanel
# =============================================================================


class TestBuildErrorPanel:
    """Test build_error_panel rendering."""

    def test_error_message_rendered(self):
        panel = tracker.build_error_panel("HTTP 500 Server Error")
        text = render_to_text(panel)
        assert "Error: HTTP 500 Server Error" in text

    def test_error_panel_title(self):
        panel = tracker.build_error_panel("something broke")
        text = render_to_text(panel)
        assert "Error" in text

    def test_returns_panel(self):
        result = tracker.build_error_panel("fail")
        assert isinstance(result, Panel)


class TestBuildNotFoundPanel:
    """Test build_not_found_panel rendering."""

    def test_shows_train_number(self):
        panel = tracker.build_not_found_panel("42")
        text = render_to_text(panel)
        assert "Train #42" in text

    def test_shows_not_found_text(self):
        panel = tracker.build_not_found_panel("42")
        text = render_to_text(panel)
        assert "not found" in text

    def test_shows_possible_causes(self):
        panel = tracker.build_not_found_panel("42")
        text = render_to_text(panel)
        assert "hasn't started" in text
        assert "incorrect" in text
        assert "completed its journey" in text

    def test_panel_title(self):
        panel = tracker.build_not_found_panel("42")
        text = render_to_text(panel)
        assert "Train Not Found" in text

    def test_returns_panel(self):
        result = tracker.build_not_found_panel("42")
        assert isinstance(result, Panel)


# =============================================================================
# TestBuildHeaderEdgeCases
# =============================================================================


class TestBuildHeaderEdgeCases:
    """Test build_header ETA edge cases."""

    def test_eta_exact_on_schedule_no_diff(self):
        """When arr == schArr, no +/- diff should appear and no '(sched)' label."""
        same_time = ts_ms(FIXED_NOW + timedelta(hours=1))
        train = make_train(
            train_state="Active", status_msg="On Time",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(FIXED_NOW - timedelta(hours=2)),
                             dep=ts_ms(FIXED_NOW - timedelta(hours=2))),
                make_station(code="PHL", name="Philadelphia", status="Enroute",
                             sch_arr=same_time, arr=same_time),
            ],
        )
        text = render_to_text(tracker.build_header(train))
        # Time should appear but no delay/early indicators
        assert "+" not in text or "+0m" not in text
        assert "-" not in text.split("@")[1] if "@" in text else True
        assert "(sched)" not in text

    def test_eta_no_times_shows_dash(self):
        """Station with only schDep, no schArr/arr -> ETA shows dash."""
        train = make_train(
            train_state="Active", status_msg="On Time",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(FIXED_NOW - timedelta(hours=2)),
                             dep=ts_ms(FIXED_NOW - timedelta(hours=2))),
                make_station(code="GBG", name="Greensburg", status="Enroute",
                             sch_dep=ts_ms(FIXED_NOW + timedelta(hours=1))),
            ],
        )
        text = render_to_text(tracker.build_header(train))
        # The ETA after "@ " should be a dash since there's no arrival time
        assert "@ —" in text

    def test_active_status_when_no_status_msg(self):
        """With train_state='Active' and empty status_msg, header should show 'Active'."""
        train = make_train(
            train_state="Active", status_msg="",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(FIXED_NOW - timedelta(hours=2)),
                             dep=ts_ms(FIXED_NOW - timedelta(hours=2))),
                make_station(code="PHL", name="Philadelphia", status="Enroute",
                             sch_arr=ts_ms(FIXED_NOW + timedelta(hours=1)),
                             arr=ts_ms(FIXED_NOW + timedelta(hours=1))),
            ],
        )
        text = render_to_text(tracker.build_header(train))
        assert "Active" in text


# =============================================================================
# TestMainLiveRefreshLoop
# =============================================================================


class TestMainLiveRefreshLoop:
    """Test main() live-refresh loop entry/exit."""

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep", side_effect=KeyboardInterrupt)
    def test_single_full_mode_uses_live(self, mock_sleep, mock_live_cls, mock_fetch, mock_console_cls):
        """Full mode uses Live context manager; KeyboardInterrupt triggers sys.exit(0)."""
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        # Live context manager setup
        mock_live_instance = MagicMock()
        mock_live_cls.return_value = mock_live_instance
        mock_live_instance.__enter__ = MagicMock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = MagicMock(return_value=False)

        with patch("sys.argv", ["amtrak-status", "42"]):
            with pytest.raises(SystemExit):
                tracker.main()

        mock_live_cls.assert_called_once()

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep", side_effect=KeyboardInterrupt)
    def test_single_compact_mode_no_live(self, mock_sleep, mock_live_cls, mock_fetch, mock_console_cls):
        """Compact mode does NOT use Live; uses console.print and console.clear."""
        mock_fetch.return_value = make_train(stations=sample_journey_stations())
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "--compact"]):
            tracker.main()

        mock_live_cls.assert_not_called()
        mock_console.print.assert_called()

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep", side_effect=[None, KeyboardInterrupt])
    def test_multi_full_mode_uses_live(self, mock_sleep, mock_live_cls, mock_fetch_cached, mock_fetch, mock_console_cls):
        """Multi-train full mode uses Live; sleep(1) during setup then KeyboardInterrupt in loop."""
        train1 = make_train(
            train_num="42",
            stations=[
                make_station(code="PHL", name="Philadelphia", status="Enroute",
                             sch_arr=ts_ms(FIXED_NOW + timedelta(hours=1)),
                             arr=ts_ms(FIXED_NOW + timedelta(hours=1))),
            ],
        )
        train2 = make_train(
            train_num="178",
            stations=[
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_dep=ts_ms(FIXED_NOW + timedelta(hours=2))),
            ],
        )
        mock_fetch.side_effect = [train1, train2]
        mock_fetch_cached.side_effect = [train1, train2]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        # Live context manager setup
        mock_live_instance = MagicMock()
        mock_live_cls.return_value = mock_live_instance
        mock_live_instance.__enter__ = MagicMock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = MagicMock(return_value=False)

        with patch("sys.argv", ["amtrak-status", "42", "178", "--connection", "PHL"]):
            with pytest.raises(SystemExit):
                tracker.main()

        mock_live_cls.assert_called_once()


# =============================================================================
# TestMainMultiTrainOrchestration
# =============================================================================


class TestMainMultiTrainOrchestration:
    """Test main() multi-train orchestration branches using --once to avoid loop."""

    def _make_train_with_stations(self, train_num, station_codes, route_name="TestRoute"):
        """Build a train with stations at the given codes."""
        stations = [
            make_station(code=code, name=f"Station {code}", status="Enroute" if i == 0 else "",
                         sch_arr=ts_ms(FIXED_NOW + timedelta(hours=i)),
                         sch_dep=ts_ms(FIXED_NOW + timedelta(hours=i, minutes=5)))
            for i, code in enumerate(station_codes)
        ]
        return make_train(train_num=train_num, route_name=route_name, stations=stations)

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.fetch_station_schedule")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_conn_provided_neither_valid_fetches_schedule(
        self, mock_sleep, mock_live, mock_fetch_station, mock_fetch_cached,
        mock_fetch, mock_console_cls
    ):
        """Branch 1a: --connection PHL, neither train valid -> fetch station schedule."""
        mock_fetch.return_value = None
        mock_fetch_cached.return_value = None
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        schedule_data = {
            "PHL": [
                {"trainNum": "42", "schArr": 1000, "schDep": 2000, "arr": None, "dep": None, "status": ""},
                {"trainNum": "178", "schArr": 3000, "schDep": 4000, "arr": None, "dep": None, "status": ""},
            ]
        }
        mock_fetch_station.return_value = schedule_data

        with patch("sys.argv", ["amtrak-status", "42", "178", "--connection", "PHL", "--once"]):
            tracker.main()

        mock_fetch_station.assert_called_with("PHL")
        # Caches should be populated with predeparture data
        assert "42" in tracker._train_caches
        assert tracker._train_caches["42"]["data"]["_predeparture"] is True
        assert "178" in tracker._train_caches
        assert tracker._train_caches["178"]["data"]["_predeparture"] is True

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.fetch_station_schedule")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_conn_provided_one_invalid_fetches_for_missing(
        self, mock_sleep, mock_live, mock_fetch_station, mock_fetch_cached,
        mock_fetch, mock_console_cls
    ):
        """Branch 1b: --connection PHL, train1 valid, train2 None -> fetch schedule for train2."""
        train1 = self._make_train_with_stations("42", ["PGH", "PHL", "NYP"])
        mock_fetch.side_effect = [train1, None]
        mock_fetch_cached.side_effect = [train1, None]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        schedule_data = {
            "PHL": [
                {"trainNum": "178", "schArr": 3000, "schDep": 4000, "arr": None, "dep": None, "status": ""},
            ]
        }
        mock_fetch_station.return_value = schedule_data

        with patch("sys.argv", ["amtrak-status", "42", "178", "--connection", "PHL", "--once"]):
            tracker.main()

        mock_fetch_station.assert_called_with("PHL")
        assert "178" in tracker._train_caches
        assert tracker._train_caches["178"]["data"]["_predeparture"] is True

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_no_conn_single_overlap_autodetects(
        self, mock_sleep, mock_live, mock_fetch_cached, mock_fetch, mock_console_cls
    ):
        """Branch 2a-ii: No --connection, both valid, single overlap -> auto-detect."""
        train1 = self._make_train_with_stations("42", ["PGH", "PHL", "NYP"])
        train2 = self._make_train_with_stations("178", ["PHL", "HBG"])
        mock_fetch.side_effect = [train1, train2]
        mock_fetch_cached.side_effect = [train1, train2]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "178", "--once"]):
            tracker.main()

        assert tracker.CONNECTION_STATION == "PHL"

    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_no_conn_no_overlap_exits(
        self, mock_sleep, mock_live, mock_fetch_cached, mock_fetch, mock_console_cls
    ):
        """Branch 2a-i: No --connection, both valid, no shared stations -> sys.exit(1)."""
        train1 = self._make_train_with_stations("42", ["PGH", "HBG"])
        train2 = self._make_train_with_stations("178", ["WAS", "BOS"])
        mock_fetch.side_effect = [train1, train2]
        mock_fetch_cached.side_effect = [train1, train2]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "178", "--once"]):
            with pytest.raises(SystemExit):
                tracker.main()

    @patch("amtrak_status.tracker.Prompt.ask", return_value="1")
    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_no_conn_multiple_overlaps_prompts(
        self, mock_sleep, mock_live, mock_fetch_cached, mock_fetch,
        mock_console_cls, mock_prompt
    ):
        """Branch 2a-iii: No --connection, both valid, multiple overlaps -> prompt user."""
        train1 = self._make_train_with_stations("42", ["PGH", "PHL", "NYP"])
        train2 = self._make_train_with_stations("178", ["PHL", "NYP", "BOS"])
        mock_fetch.side_effect = [train1, train2]
        mock_fetch_cached.side_effect = [train1, train2]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with patch("sys.argv", ["amtrak-status", "42", "178", "--once"]):
            tracker.main()

        assert tracker.CONNECTION_STATION == "PHL"

    @patch("amtrak_status.tracker.Prompt.ask", return_value="PHL")
    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.fetch_station_schedule")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_no_conn_one_valid_prompts_user(
        self, mock_sleep, mock_live, mock_fetch_station, mock_fetch_cached,
        mock_fetch, mock_console_cls, mock_prompt
    ):
        """Branch 2b: No --connection, one valid, one None -> prompt for station code."""
        train1 = self._make_train_with_stations("42", ["PGH", "PHL", "NYP"])
        mock_fetch.side_effect = [train1, None]
        mock_fetch_cached.side_effect = [train1, None]
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        mock_fetch_station.return_value = {"PHL": []}

        with patch("sys.argv", ["amtrak-status", "42", "178", "--once"]):
            tracker.main()

        assert tracker.CONNECTION_STATION == "PHL"
        mock_fetch_station.assert_called_with("PHL")

    @patch("amtrak_status.tracker.Prompt.ask", return_value="PHL")
    @patch("amtrak_status.tracker.Console")
    @patch("amtrak_status.tracker.fetch_train_data")
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    @patch("amtrak_status.tracker.fetch_station_schedule")
    @patch("amtrak_status.tracker.Live")
    @patch("amtrak_status.tracker.sleep")
    def test_no_conn_neither_valid_prompts_user(
        self, mock_sleep, mock_live, mock_fetch_station, mock_fetch_cached,
        mock_fetch, mock_console_cls, mock_prompt
    ):
        """Branch 2c: No --connection, neither valid -> prompt for station code."""
        mock_fetch.return_value = None
        mock_fetch_cached.return_value = None
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        mock_fetch_station.return_value = {"PHL": []}

        with patch("sys.argv", ["amtrak-status", "42", "178", "--once"]):
            tracker.main()

        assert tracker.CONNECTION_STATION == "PHL"
