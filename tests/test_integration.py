"""Integration tests for amtrak-status: rendered output verification and lifecycle tests."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

import pytest
from rich.text import Text

import amtrak_status.tracker as tracker

# Shared helpers from conftest (imported explicitly for use in test code)
from conftest import (
    make_station, make_train, ts_ms, FIXED_NOW,
    render_to_text, journey_at_phase,
)


# =============================================================================
# TestRenderedStationTable
# =============================================================================


class TestRenderedStationTable:
    def test_station_names_and_codes_appear(self):
        train = journey_at_phase("mid")
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        assert "Pittsburgh (PGH)" in text
        assert "Greensburg (GBG)" in text
        assert "Harrisburg (HBG)" in text
        assert "Philadelphia (PHL)" in text
        assert "New York Penn (NYP)" in text

    def test_status_indicators_appear(self):
        train = journey_at_phase("mid")
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        # Departed stations should show checkmark, Enroute should show arrow
        assert "✓" in text  # Departed (PGH, GBG)
        assert "→" in text  # Enroute (HBG)
        assert "○" in text  # Future (PHL, NYP)

    def test_arriving_station_shows_bullet(self):
        train = journey_at_phase("arriving")
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        assert "●" in text  # Station (PHL)

    def test_cancelled_station_shows_cancelled(self):
        train = journey_at_phase("mid")
        # Insert a cancelled station
        train["stations"].insert(2, make_station(
            code="LAT", name="Latrobe", status="", sch_arr=None, sch_dep=None
        ))
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        assert "Latrobe" in text
        assert "Cancelled" in text
        assert "✗" in text

    def test_filter_shows_omitted_messages(self):
        tracker.STATION_FROM = "GBG"
        tracker.STATION_TO = "PHL"
        train = journey_at_phase("mid")
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        assert "earlier stops omitted" in text
        assert "later stops omitted" in text
        # The filtered stations should still appear
        assert "Greensburg (GBG)" in text
        assert "Philadelphia (PHL)" in text

    def test_focus_hides_old_departed(self):
        """With >10 stations and many departed, focus should hide old ones."""
        stations = []
        for i in range(15):
            if i < 10:
                stations.append(make_station(
                    code=f"S{i:02d}", name=f"Station {i}", status="Departed",
                    sch_dep=i * 1000,
                    dep=(i + 1) * 1000,
                ))
            elif i == 10:
                stations.append(make_station(
                    code=f"S{i:02d}", name=f"Station {i}", status="Enroute",
                    sch_arr=i * 1000, sch_dep=i * 1000 + 500,
                    arr=(i + 1) * 1000,
                ))
            else:
                stations.append(make_station(
                    code=f"S{i:02d}", name=f"Station {i}", status="",
                    sch_arr=i * 1000,
                ))
        train = make_train(stations=stations)
        panel = tracker.build_stations_table(train, focus=True)
        text = render_to_text(panel)

        assert "departed stops hidden" in text

    def test_scheduled_column_headers(self):
        train = journey_at_phase("mid")
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        assert "Sch Arr" in text
        assert "Sch Dep" in text
        assert "Act/Est Arr" in text
        assert "Act/Est Dep" in text
        assert "Status" in text

    def test_status_column_values(self):
        train = journey_at_phase("mid")
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        assert "Departed" in text
        assert "Enroute" in text
        assert "Scheduled" in text  # Future stations get "Scheduled"


# =============================================================================
# TestRenderedHeader
# =============================================================================


class TestRenderedHeader:
    def test_train_info_appears(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "Pennsylvanian" in text
        assert "#42" in text
        assert "42-1" in text

    def test_next_station_and_eta(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "Harrisburg" in text
        assert "Next:" in text

    def test_speed_and_heading(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "62 mph" in text
        assert "Heading: E" in text

    def test_on_time_status(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "On Time" in text

    def test_late_status(self):
        train = journey_at_phase("mid_late")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "25 Minutes Late" in text

    def test_delay_indicator_in_eta(self):
        """When the estimated arrival differs from scheduled, show +Xm."""
        train = journey_at_phase("mid_late")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        # Header shows delay next to the ETA
        assert "+25m" in text

    def test_destination_shown(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "New York Penn" in text

    def test_position_bar_shows_station_codes(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        # Position bar: GBG █████░░░░░ HBG
        assert "GBG" in text
        assert "HBG" in text
        assert "Position:" in text

    def test_predeparture_no_position_bar(self):
        train = journey_at_phase("predeparture")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "Position:" not in text

    def test_amtrak_status_title(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "Amtrak Status" in text

    def test_refresh_info_in_subtitle(self):
        train = journey_at_phase("mid")
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "Refresh:" in text
        assert "Ctrl+C" in text


# =============================================================================
# TestRenderedCompactOutput
# =============================================================================


class TestRenderedCompactOutput:
    def test_contains_train_name_and_number(self):
        train = journey_at_phase("mid")
        result = tracker.build_compact_display(train)
        text = result.plain

        assert "Pennsylvanian" in text
        assert "#42" in text

    def test_contains_next_station_and_eta(self):
        train = journey_at_phase("mid")
        result = tracker.build_compact_display(train)
        text = result.plain

        # Should contain next station (HBG) and a time
        assert "HBG" in text
        assert "AM" in text or "PM" in text

    def test_contains_speed(self):
        train = journey_at_phase("mid")
        result = tracker.build_compact_display(train)
        text = result.plain

        assert "62mph" in text

    def test_contains_progress_percentage(self):
        train = journey_at_phase("mid")
        result = tracker.build_compact_display(train)
        text = result.plain

        assert "%" in text

    def test_position_between_stations(self):
        train = journey_at_phase("mid")
        result = tracker.build_compact_display(train)
        text = result.plain

        # Position: GBG→X%→HBG
        assert "GBG" in text
        assert "HBG" in text
        assert "→" in text

    def test_delay_shown_when_late(self):
        train = journey_at_phase("mid_late")
        result = tracker.build_compact_display(train)
        text = result.plain

        assert "+25m" in text or "25 Minutes Late" in text

    def test_status_message_included(self):
        train = journey_at_phase("mid")
        result = tracker.build_compact_display(train)
        text = result.plain

        assert "On Time" in text

    def test_no_crash_with_zero_velocity(self):
        train = journey_at_phase("arriving")  # velocity=0
        result = tracker.build_compact_display(train)
        assert result.plain  # Just verify it renders without crashing


# =============================================================================
# TestRenderedConnectionPanel
# =============================================================================


class TestRenderedConnectionPanel:
    def _make_connection_trains(self, layover_minutes):
        """Build two trains connecting at PHL with specified layover."""
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(now - timedelta(hours=3))),
                make_station(code="PHL", name="Philadelphia", status="Enroute",
                             sch_arr=ts_ms(now + timedelta(minutes=30)),
                             arr=ts_ms(now + timedelta(minutes=30))),
            ],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_dep=ts_ms(now + timedelta(minutes=30 + layover_minutes))),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(now + timedelta(hours=3))),
            ],
        )
        return train1, train2

    def test_shows_train_names(self):
        train1, train2 = self._make_connection_trains(90)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "Pennsylvanian" in text
        assert "#42" in text
        assert "Keystone" in text
        assert "#178" in text

    def test_shows_station_name(self):
        train1, train2 = self._make_connection_trains(90)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "Philadelphia" in text
        assert "PHL" in text

    def test_comfortable_layover_text(self):
        train1, train2 = self._make_connection_trains(90)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "1h 30m layover" in text

    def test_tight_layover_text(self):
        train1, train2 = self._make_connection_trains(35)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "35 min layover" in text
        assert "tight" in text

    def test_risky_layover_text(self):
        train1, train2 = self._make_connection_trains(15)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "15 min layover" in text
        assert "risky" in text

    def test_missed_connection_text(self):
        now = FIXED_NOW
        # Train1 arrives AFTER train2 departs
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Enroute",
                sch_arr=ts_ms(now + timedelta(hours=1)),
                arr=ts_ms(now + timedelta(hours=1)),
            )],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="",
                sch_dep=ts_ms(now + timedelta(minutes=20)),
            )],
        )
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "MISSED" in text

    def test_arrival_and_departure_times_shown(self):
        train1, train2 = self._make_connection_trains(90)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)

        assert "Arrives:" in text
        assert "Departs:" in text
        # Should contain AM/PM time strings
        assert "AM" in text or "PM" in text


# =============================================================================
# TestJourneyLifecycle
# =============================================================================


class TestJourneyLifecycle:
    """Simulate a train progressing through its journey across multiple API refreshes."""

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_not_found_then_active(self, mock_fetch):
        """First refresh: train not found. Second refresh: train active."""
        # First refresh — not found
        mock_fetch.return_value = None
        result1 = tracker.build_display("42")
        text1 = render_to_text(result1)
        assert "not found" in text1.lower()

        # Second refresh — train appeared
        mock_fetch.return_value = journey_at_phase("early")
        result2 = tracker.build_display("42")
        text2 = render_to_text(result2)
        assert "Pennsylvanian" in text2
        assert "Pittsburgh" in text2

    def test_station_progression(self):
        """Verify station indicators change as train progresses."""
        # Phase: early — PGH departed, GBG enroute
        train_early = journey_at_phase("early")
        text_early = render_to_text(tracker.build_stations_table(train_early))

        # PGH should show as departed (checkmark), GBG as enroute (arrow)
        lines_early = text_early.split("\n")
        pgh_line = [l for l in lines_early if "Pittsburgh" in l]
        gbg_line = [l for l in lines_early if "Greensburg" in l]
        assert any("✓" in l for l in pgh_line)
        assert any("→" in l for l in gbg_line)

        # Phase: mid — PGH+GBG departed, HBG enroute
        train_mid = journey_at_phase("mid")
        text_mid = render_to_text(tracker.build_stations_table(train_mid))

        lines_mid = text_mid.split("\n")
        gbg_line_mid = [l for l in lines_mid if "Greensburg" in l]
        hbg_line_mid = [l for l in lines_mid if "Harrisburg" in l]
        assert any("✓" in l for l in gbg_line_mid)  # GBG now departed
        assert any("→" in l for l in hbg_line_mid)  # HBG now enroute

        # Phase: arriving — at PHL station
        train_arriving = journey_at_phase("arriving")
        text_arriving = render_to_text(tracker.build_stations_table(train_arriving))

        lines_arriving = text_arriving.split("\n")
        phl_line = [l for l in lines_arriving if "Philadelphia" in l]
        assert any("●" in l for l in phl_line)  # PHL at station

    def test_delay_appears_in_header(self):
        """When train becomes late, header reflects the delay."""
        # On time
        train_ontime = journey_at_phase("mid")
        text_ontime = render_to_text(tracker.build_header(train_ontime))
        assert "On Time" in text_ontime

        # Now late
        train_late = journey_at_phase("mid_late")
        text_late = render_to_text(tracker.build_header(train_late))
        assert "25 Minutes Late" in text_late
        assert "+25m" in text_late

    def test_progress_increases(self):
        """Progress bar should advance as more stations are departed."""
        train_early = journey_at_phase("early")
        _, _, total_early = tracker.calculate_progress(train_early["stations"])
        completed_early, _, _ = tracker.calculate_progress(train_early["stations"])

        train_final = journey_at_phase("final_leg")
        completed_final, _, total_final = tracker.calculate_progress(train_final["stations"])

        assert completed_final > completed_early
        assert total_early == total_final  # Same route, same total


# =============================================================================
# TestMultiTrainLifecycle
# =============================================================================


class TestMultiTrainLifecycle:
    def _make_two_trains(self, train1_delay_mins=0):
        """Build train1 (PGH→PHL→NYP) and train2 (PHL→HBG) connecting at PHL."""
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(now - timedelta(hours=3)),
                             dep=ts_ms(now - timedelta(hours=3))),
                make_station(code="PHL", name="Philadelphia", status="Enroute",
                             sch_arr=ts_ms(now + timedelta(minutes=30)),
                             arr=ts_ms(now + timedelta(minutes=30 + train1_delay_mins))),
                make_station(code="NYP", name="New York Penn", status="",
                             sch_arr=ts_ms(now + timedelta(hours=3))),
            ],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_dep=ts_ms(now + timedelta(hours=2))),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(now + timedelta(hours=4))),
            ],
        )
        return train1, train2

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_comfortable_connection_display(self, mock_fetch):
        train1, train2 = self._make_two_trains(train1_delay_mins=0)
        mock_fetch.side_effect = [train1, train2]

        layout = tracker.build_multi_train_display(["42", "178"], "PHL")
        text = render_to_text(layout)

        assert "Pennsylvanian" in text
        assert "Keystone" in text
        assert "Philadelphia" in text
        assert "layover" in text.lower()

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_connection_tightens_with_delay(self, mock_fetch):
        """When train1 is delayed, the layover shrinks."""
        # On time → comfortable
        train1_ontime, train2 = self._make_two_trains(train1_delay_mins=0)
        layover_ontime = tracker.calculate_layover(train1_ontime, train2, "PHL")

        # Delayed → tighter
        train1_late, train2 = self._make_two_trains(train1_delay_mins=45)
        layover_late = tracker.calculate_layover(train1_late, train2, "PHL")

        assert layover_late["layover_minutes"] < layover_ontime["layover_minutes"]

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_one_train_predeparture(self, mock_fetch):
        """When train2 is not found, show predeparture panel."""
        train1, _ = self._make_two_trains()
        mock_fetch.side_effect = [train1, None]

        layout = tracker.build_multi_train_display(["42", "178"], "PHL")
        text = render_to_text(layout)

        assert "Pennsylvanian" in text
        assert "Awaiting Departure" in text
        assert "#178" in text

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_both_trains_not_found(self, mock_fetch):
        mock_fetch.return_value = None

        layout = tracker.build_multi_train_display(["42", "178"], "PHL")
        text = render_to_text(layout)

        assert "not found" in text.lower() or "not active" in text.lower() or "awaiting" in text.lower()

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_multi_train_display_has_top_header(self, mock_fetch):
        """Multi-train display should include the top-level Amtrak Status header
        with update time, refresh interval, and quit hint — just like single-train mode."""
        train1, train2 = self._make_two_trains()
        mock_fetch.side_effect = [train1, train2]

        layout = tracker.build_multi_train_display(["42", "178"], "PHL")
        text = render_to_text(layout)

        assert "Amtrak Status" in text
        assert "Refresh:" in text
        assert "Ctrl+C" in text


# =============================================================================
# TestErrorAndRecovery
# =============================================================================


class TestErrorAndRecovery:
    @patch("amtrak_status.tracker.fetch_train_data")
    def test_error_panel_shows_message(self, mock_fetch):
        mock_fetch.return_value = {"error": "HTTP 503"}
        result = tracker.build_display("42")
        text = render_to_text(result)

        assert "HTTP 503" in text
        assert "Error" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_not_found_panel_content(self, mock_fetch):
        mock_fetch.return_value = None
        result = tracker.build_display("42")
        text = render_to_text(result)

        assert "#42" in text
        assert "not found" in text.lower()
        assert "hasn't started" in text.lower() or "hasn" in text
        assert "incorrect" in text.lower()

    def test_cached_data_with_warning_indicator(self):
        """When API fails but cache is fresh, data still renders and warning appears."""
        train = journey_at_phase("mid")
        tracker._last_successful_data = train
        tracker._last_fetch_time = FIXED_NOW
        tracker._last_error = "Train not in API response (using cached data)"

        panel = tracker.build_header(train)
        text = render_to_text(panel)

        # Warning should appear in the subtitle
        assert "using cached data" in text.lower() or "⚠" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_compact_error(self, mock_fetch):
        tracker.COMPACT_MODE = True
        mock_fetch.return_value = {"error": "timeout"}
        result = tracker.build_display("42")
        text = result.plain

        assert "error" in text.lower()
        assert "42" in text


# =============================================================================
# TestNotificationIntegration
# =============================================================================


class TestNotificationIntegration:
    """Full notification lifecycle: init → progress → notify."""

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_full_notification_lifecycle(self, mock_notify):
        tracker.NOTIFY_STATIONS = {"PHL"}

        # Step 1: Train en route — initialize notification state
        train_enroute = journey_at_phase("mid")
        result1 = tracker.check_and_notify(train_enroute)
        assert result1 == []  # No notifications on init
        assert tracker._notifications_initialized is True

        # Step 2: Train arrives at PHL — should notify
        train_at_phl = journey_at_phase("arriving")
        result2 = tracker.check_and_notify(train_at_phl)
        assert "PHL" in result2
        assert mock_notify.call_count == 1

        # Verify notification content
        title_arg = mock_notify.call_args[0][0]
        message_arg = mock_notify.call_args[0][1]
        assert "Pennsylvanian" in title_arg
        assert "#42" in title_arg
        assert "Philadelphia" in message_arg
        assert "PHL" in message_arg

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_notify_all_tracks_each_station(self, mock_notify):
        tracker.NOTIFY_ALL = True

        # Init: PGH+GBG departed, HBG enroute
        train_mid = journey_at_phase("mid")
        tracker.check_and_notify(train_mid)
        mock_notify.reset_mock()

        # Progress: HBG now departed, PHL is Station
        train_arriving = journey_at_phase("arriving")
        result = tracker.check_and_notify(train_arriving)

        # HBG transitioned to Departed, PHL to Station → both should notify
        assert "HBG" in result
        assert "PHL" in result
        assert mock_notify.call_count == 2

    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_departed_before_init_not_notified(self, mock_notify):
        """Stations that were already departed when we started should not trigger."""
        tracker.NOTIFY_ALL = True

        # Init at mid journey — PGH and GBG already departed
        train_mid = journey_at_phase("mid")
        tracker.check_and_notify(train_mid)

        # PGH and GBG should be in _notified_stations (pre-marked)
        assert "PGH" in tracker._notified_stations
        assert "GBG" in tracker._notified_stations

        # They should not have triggered actual notifications
        # (only pre-existing departures are marked, not new ones)
        calls = mock_notify.call_args_list
        for c in calls:
            title = c[0][0]
            assert "Pittsburgh" not in title or "PGH" not in title

    @patch("amtrak_status.tracker.check_and_notify", return_value=[])
    @patch("amtrak_status.tracker.sleep")
    @patch("amtrak_status.tracker.Console")
    def test_notifications_fire_in_multi_train_mode(
        self, mock_console_cls, mock_sleep, mock_check_notify
    ):
        """check_and_notify should be called in multi-train --once mode."""
        now = FIXED_NOW
        tracker.NOTIFY_STATIONS = {"PHL"}

        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(now - timedelta(hours=3))),
                make_station(code="PHL", name="Philadelphia", status="Station",
                             sch_arr=ts_ms(now), arr=ts_ms(now),
                             sch_dep=ts_ms(now + timedelta(minutes=5))),
            ],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_dep=ts_ms(now + timedelta(hours=1))),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(now + timedelta(hours=3))),
            ],
        )

        # Mock fetches that also populate _train_caches (as real functions would)
        def fake_fetch(num):
            data = {"42": train1, "178": train2}.get(num)
            if data:
                tracker._train_caches[num] = {
                    "data": data, "fetch_time": FIXED_NOW, "error": None
                }
            return data

        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with (
            patch("amtrak_status.tracker.fetch_train_data", side_effect=fake_fetch),
            patch("amtrak_status.tracker.fetch_train_data_cached", side_effect=fake_fetch),
            patch("sys.argv", ["amtrak-status", "42", "178", "--connection", "PHL", "--once"]),
        ):
            tracker.main()

        # check_and_notify should have been called for each train's cached data
        assert mock_check_notify.called, (
            "check_and_notify was never called in multi-train --once mode"
        )


# =============================================================================
# TestFullBuildDisplay
# =============================================================================


class TestFullBuildDisplay:
    """End-to-end: mock fetch → build_display → verify rendered output."""

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_full_display_contains_all_sections(self, mock_fetch):
        """The full display should have header, progress bar, and stations."""
        mock_fetch.return_value = journey_at_phase("mid")
        result = tracker.build_display("42")
        text = render_to_text(result)

        # Header section
        assert "Pennsylvanian" in text
        assert "Amtrak Status" in text

        # Progress bar section
        assert "Journey Progress" in text
        assert "Pittsburgh" in text  # Origin in progress bar

        # Stations section
        assert "Sch Arr" in text
        assert "Harrisburg" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_compact_full_display(self, mock_fetch):
        """Compact mode returns a single-line Text."""
        tracker.COMPACT_MODE = True
        mock_fetch.return_value = journey_at_phase("mid")
        result = tracker.build_display("42")

        assert isinstance(result, Text)  # it's a Text
        text = result.plain
        assert "Pennsylvanian" in text
        assert "#42" in text
        assert "%" in text

    @patch("amtrak_status.tracker.fetch_train_data")
    def test_display_with_station_filter(self, mock_fetch):
        """--from/--to filter should restrict visible stations."""
        tracker.STATION_FROM = "HBG"
        tracker.STATION_TO = "NYP"
        mock_fetch.return_value = journey_at_phase("mid")
        result = tracker.build_display("42")
        text = render_to_text(result)

        # Filtered range
        assert "Harrisburg" in text
        assert "New York Penn" in text
        assert "earlier stops omitted" in text


# =============================================================================
# Multi-train display coverage
# =============================================================================


class TestRenderedCompactTrainHeader:
    def test_active_train_shows_info(self):
        train = journey_at_phase("mid")
        panel = tracker.build_compact_train_header(train)
        text = render_to_text(panel)

        assert "Pennsylvanian" in text
        assert "#42" in text
        assert "On Time" in text
        assert "Harrisburg" in text  # next station
        assert "62 mph" in text or "62mph" in text

    def test_position_bar_present(self):
        train = journey_at_phase("mid")
        panel = tracker.build_compact_train_header(train)
        text = render_to_text(panel)

        assert "GBG" in text  # last departed
        assert "HBG" in text  # next station

    def test_zero_velocity_shows_dash(self):
        train = journey_at_phase("arriving")  # velocity=0
        panel = tracker.build_compact_train_header(train)
        text = render_to_text(panel)

        assert "0 mph" not in text

    def test_predeparture_synthetic_shows_schedule(self):
        """Synthetic predeparture data should show scheduled time."""
        now = FIXED_NOW
        train = make_train(
            train_state="Predeparture",
            stations=[make_station(code="PHL", sch_dep=ts_ms(now))],
        )
        train["_predeparture"] = True
        panel = tracker.build_compact_train_header(train)
        text = render_to_text(panel)

        assert "Predeparture" in text
        assert "PHL" in text
        assert "Live tracking begins" in text

    def test_late_train_shows_delay(self):
        train = journey_at_phase("mid_late")
        panel = tracker.build_compact_train_header(train)
        text = render_to_text(panel)

        assert "25 Minutes Late" in text


class TestRenderedPredepartureDisplay:
    def test_predeparture_panel_content(self):
        panel = tracker.build_predeparture_panel("42")
        text = render_to_text(panel)

        assert "#42" in text
        assert "Awaiting Departure" in text
        assert "Predeparture" in text

    def test_predeparture_header_content(self):
        panel = tracker.build_predeparture_header("42")
        text = render_to_text(panel)

        assert "#42" in text
        assert "Awaiting Departure" in text
        assert "Live tracking" in text


class TestMultiTrainTitlePresence:
    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_title_present_when_only_train1_valid(self, mock_fetch):
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", status="Departed",
                             sch_dep=ts_ms(now - timedelta(hours=3)),
                             dep=ts_ms(now - timedelta(hours=3))),
                make_station(code="PHL", status="Enroute",
                             sch_arr=ts_ms(now + timedelta(hours=1)),
                             arr=ts_ms(now + timedelta(hours=1))),
            ],
        )
        mock_fetch.side_effect = [train1, None]

        layout = tracker.build_multi_train_display(["42", "178"], "PHL")
        text = render_to_text(layout)

        assert "Amtrak Status" in text
        assert "Refresh:" in text
        assert "Ctrl+C" in text

    @patch("amtrak_status.tracker.fetch_train_data_cached")
    def test_title_present_when_only_train2_valid(self, mock_fetch):
        now = FIXED_NOW
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", status="",
                             sch_dep=ts_ms(now + timedelta(hours=2))),
                make_station(code="HBG", status="",
                             sch_arr=ts_ms(now + timedelta(hours=4))),
            ],
        )
        mock_fetch.side_effect = [None, train2]

        layout = tracker.build_multi_train_display(["42", "178"], "PHL")
        text = render_to_text(layout)

        assert "Amtrak Status" in text
        assert "Refresh:" in text
        assert "Ctrl+C" in text


# =============================================================================
# Connection, notification, and cache coverage
# =============================================================================


class TestConnectionPanelStatusTransitions:
    def test_train1_arrived_shows_checkmark(self):
        """When train1 has departed the connection station, show 'Arrived'."""
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Departed",
                sch_arr=ts_ms(now - timedelta(minutes=30)),
                arr=ts_ms(now - timedelta(minutes=30)),
                dep=ts_ms(now - timedelta(minutes=25)),
            )],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="",
                sch_dep=ts_ms(now + timedelta(minutes=60)),
            )],
        )
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Arrived" in text

    def test_train1_at_station_shows_bullet(self):
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Station",
                sch_arr=ts_ms(now - timedelta(minutes=5)),
                arr=ts_ms(now - timedelta(minutes=3)),
            )],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="",
                sch_dep=ts_ms(now + timedelta(minutes=60)),
            )],
        )
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "At station" in text

    def test_train2_boarding_shows_bullet(self):
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Departed",
                sch_arr=ts_ms(now - timedelta(hours=1)),
                arr=ts_ms(now - timedelta(hours=1)),
            )],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Station",
                sch_dep=ts_ms(now + timedelta(minutes=5)),
            )],
        )
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Boarding" in text

    def test_train2_already_departed_shows_departed(self):
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Enroute",
                sch_arr=ts_ms(now + timedelta(minutes=30)),
                arr=ts_ms(now + timedelta(minutes=30)),
            )],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[make_station(
                code="PHL", name="Philadelphia", status="Departed",
                sch_dep=ts_ms(now - timedelta(minutes=10)),
                dep=ts_ms(now - timedelta(minutes=10)),
            )],
        )
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "Departed" in text


class TestLayoverBoundary:
    def _make_connection_trains(self, layover_minutes):
        now = FIXED_NOW
        train1 = make_train(
            train_num="42", route_name="Pennsylvanian",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(now - timedelta(hours=3))),
                make_station(code="PHL", name="Philadelphia", status="Enroute",
                             sch_arr=ts_ms(now + timedelta(minutes=30)),
                             arr=ts_ms(now + timedelta(minutes=30))),
            ],
        )
        train2 = make_train(
            train_num="178", route_name="Keystone",
            stations=[
                make_station(code="PHL", name="Philadelphia", status="",
                             sch_dep=ts_ms(now + timedelta(minutes=30 + layover_minutes))),
                make_station(code="HBG", name="Harrisburg", status="",
                             sch_arr=ts_ms(now + timedelta(hours=3))),
            ],
        )
        return train1, train2

    def test_exactly_60_min_layover(self):
        train1, train2 = self._make_connection_trains(60)
        panel = tracker.build_connection_panel(train1, train2, "PHL")
        text = render_to_text(panel)
        assert "1h 0m layover" in text


class TestNotificationObservable:
    @patch("amtrak_status.tracker.send_notification", return_value=True)
    def test_departed_before_init_not_notified(self, mock_notify):
        """Only check observable behavior: no notifications sent for pre-departed stations."""
        tracker.NOTIFY_ALL = True

        # Init at mid journey — PGH and GBG already departed
        train_mid = journey_at_phase("mid")
        tracker.check_and_notify(train_mid)

        # No notification should have been sent for pre-departed stations
        for c in mock_notify.call_args_list:
            message = c[0][1]
            assert "Pittsburgh" not in message
            assert "Greensburg" not in message


# =============================================================================
# Edge cases
# =============================================================================


class TestCompletedJourney:
    def test_completed_journey_all_checkmarks(self):
        """A completed journey should show all stations as departed."""
        now = FIXED_NOW
        base = now - timedelta(hours=6)
        train = make_train(
            train_num="42", route_name="Pennsylvanian", train_id="42-1",
            train_state="Active", status_msg="On Time", velocity=0,
            heading="", dest_name="New York Penn",
            stations=[
                make_station(code="PGH", name="Pittsburgh", status="Departed",
                             sch_dep=ts_ms(base),
                             dep=ts_ms(base)),
                make_station(code="GBG", name="Greensburg", status="Departed",
                             sch_arr=ts_ms(base + timedelta(hours=1)),
                             sch_dep=ts_ms(base + timedelta(hours=1, minutes=2)),
                             arr=ts_ms(base + timedelta(hours=1)),
                             dep=ts_ms(base + timedelta(hours=1, minutes=2))),
                make_station(code="HBG", name="Harrisburg", status="Departed",
                             sch_arr=ts_ms(base + timedelta(hours=3)),
                             sch_dep=ts_ms(base + timedelta(hours=3, minutes=5)),
                             arr=ts_ms(base + timedelta(hours=3)),
                             dep=ts_ms(base + timedelta(hours=3, minutes=5))),
                make_station(code="NYP", name="New York Penn", status="Station",
                             sch_arr=ts_ms(base + timedelta(hours=5, minutes=30)),
                             arr=ts_ms(base + timedelta(hours=5, minutes=30))),
            ],
        )
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)

        # All stations should appear
        assert "Pittsburgh" in text
        assert "Greensburg" in text
        assert "Harrisburg" in text
        assert "New York Penn" in text

        # Final station should show bullet (Station status)
        lines = text.split("\n")
        nyp_lines = [l for l in lines if "New York Penn" in l]
        assert any("●" in l for l in nyp_lines)


class TestPlatformDisplay:
    def test_platform_shown_for_enroute_station(self):
        train = make_train(stations=[
            make_station(code="PGH", name="Pittsburgh", status="Departed",
                         sch_dep=100, dep=101),
            make_station(code="HBG", name="Harrisburg", status="Enroute",
                         sch_arr=200, sch_dep=300, arr=250, platform="3"),
        ])
        panel = tracker.build_stations_table(train)
        text = render_to_text(panel)
        assert "Plt 3" in text


class TestZeroVelocityHeader:
    def test_zero_velocity_shows_dash(self):
        train = journey_at_phase("arriving")  # velocity=0
        panel = tracker.build_header(train)
        text = render_to_text(panel)

        assert "0 mph" not in text
