"""Known bugs, expressed as strict xfail tests.

Each test asserts the *correct* behaviour and is expected to fail today.  IDs
match docs/known-bugs.md.  ``xfail_strict`` is on, so fixing a bug makes its
test XPASS and fail the run: delete the marker in the same change (and expect
the related golden files in ``golden/`` to change too).

Like the rest of ``tests/contract`` these run through the CLI only, so they
survive the refactor unchanged.
"""

from __future__ import annotations

import copy
import importlib.metadata

import pytest

from _harness import App, RawBody, train_payload
from _payloads import LIVE_CAPTURED_AT, NER_171, UPSTREAM_42, journey, live_train, long_route_payload, with_stop


def bug(bug_id: str, summary: str):
    return pytest.mark.xfail(strict=True, reason=f"{bug_id}: {summary} (docs/known-bugs.md)")


# -----------------------------------------------------------------------------
# Connection (multi-train) mode
# -----------------------------------------------------------------------------


@bug("B01", "notification state is global, so train 2's already-passed stops fire at startup")
def test_b01_connection_mode_does_not_notify_for_stops_passed_before_start(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", train_payload(NER_171))  # BOS and PVD already departed
    result = app.run("42", "171", "--connection", "PHL", "--notify-all", polls=1)
    assert result.exception is None
    assert result.notifications == []


@bug("B02", "notifications are de-duplicated by station code across trains")
def test_b02_both_trains_notify_at_shared_connection_station(app: App) -> None:
    t42 = train_payload(UPSTREAM_42)
    t42_at_phl = with_stop(t42, "PHL", status="Station", arr="2026-02-08T14:50:00-05:00")
    t171 = train_payload(NER_171)
    t171_at_phl = with_stop(t171, "PHL", status="Station", arr="2026-02-08T15:22:00-05:00")
    app.api.train("42", t42, t42, t42_at_phl)
    app.api.train("171", t171, t171, t171, t171_at_phl)
    result = app.run("42", "171", "--connection", "PHL", "--notify-at", "PHL", polls=2)
    assert result.exception is None
    assert [n.argv[-2] for n in result.notifications] == [
        "🚂 Pennsylvanian #42 Arriving",
        "🚂 Northeast Regional #171 Arriving",
    ]


@bug("B06", "--compact is ignored by connection mode when combined with --once")
def test_b06_connection_compact_once_prints_compact_lines(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", train_payload(NER_171))
    result = app.run("42", "171", "--connection", "PHL", "--compact", "--once")
    assert "╭" not in result.plain  # no panels
    assert "Pennsylvanian #42" in result.plain and "Northeast Regional #171" in result.plain


@bug("B07", "connection mode re-stamps cached data on every poll, so stale data never expires")
def test_b07_connection_mode_stops_showing_stale_data_after_five_minutes(app: App) -> None:
    app.color = False
    app.api.train("42", train_payload(UPSTREAM_42), [])  # disappears from the feed after setup
    app.api.train("171", train_payload(NER_171))
    result = app.run("42", "171", "--connection", "PHL", "--compact", "-r", "60", polls=10)
    last_screen = result.plain.strip().splitlines()[-2:]
    assert "Train #42: Error or not found" in last_screen[0]


@bug("B08", "the 'using cached data' warning is a single global that the other train's success clears")
def test_b08_cached_data_warning_survives_other_trains_fetch(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42), [])  # good during setup, then cached
    app.api.train("171", train_payload(NER_171))
    result = app.run("42", "171", "--connection", "PHL", "--once")
    assert "using cached data" in result.plain


@bug("B15", "'MISSED by N min' reports a positive layover when train 2 departed early per the feed")
def test_b15_missed_connection_does_not_report_positive_minutes(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train(
        "171",
        with_stop(train_payload(NER_171), "PHL", status="Departed", dep="2026-02-08T15:30:00-05:00"),
    )
    result = app.run("42", "171", "--connection", "PHL", "--once")
    assert "MISSED" in result.plain
    assert "MISSED by 40 min" not in result.plain  # 15:30 dep - 14:50 arr is +40, not a miss margin


@bug("B17", "only the first two train numbers are used; extras are silently ignored")
def test_b17_more_than_two_trains_is_rejected(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", train_payload(NER_171))
    result = app.run("42", "171", "650", "--connection", "PHL", "--once")
    assert result.exit_code == 2


@bug("B21", "the connection-station prompt rejects lower-case codes")
def test_b21_connection_prompt_accepts_lowercase_code(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", train_payload(NER_171))
    result = app.run("42", "171", "--once", stdin="phl\n")
    assert result.exception is None
    assert "Connection set to PHL" in result.plain


@bug("B26", "the feed copies a future stop's arrival estimate into 'dep', so layovers use train 2's arrival")
def test_b26_layover_uses_scheduled_departure_when_no_departure_estimate(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", train_payload(NER_171))  # PHL: schDep 15:30, arr == dep == 15:22 (estimate)
    result = app.run("42", "171", "--connection", "PHL", "--once")
    assert "Departs: 3:30 PM" in result.plain
    assert "40 min layover" in result.plain


# -----------------------------------------------------------------------------
# API robustness
# -----------------------------------------------------------------------------


@bug("B03", "a non-JSON 200 response (e.g. a proxy error page) crashes the app")
def test_b03_non_json_response_shows_error_instead_of_crashing(app: App) -> None:
    app.api.train("42", RawBody("<html>502 Bad Gateway</html>"))
    result = app.run("42", "--once")
    assert result.exception is None
    assert "Error" in result.plain


@bug("B04", "a non-empty JSON list response crashes with AttributeError")
def test_b04_unexpected_json_list_does_not_crash(app: App) -> None:
    app.api.train("42", [{"error": "upstream failure"}])
    result = app.run("42", "--once")
    assert result.exception is None


@bug("B11", "API strings are interpolated into Rich markup; '[/]' in a name raises MarkupError")
def test_b11_markup_characters_in_api_strings_are_shown_literally(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42, routeName="Keystone [/] Service"))
    result = app.run("42", "--once")
    assert result.exception is None
    assert "Keystone [/] Service" in result.plain


@bug("B20", "notification text is spliced into the osascript source without escaping")
def test_b20_notification_text_is_escaped_for_applescript(app: App) -> None:
    j = journey()
    quoted = 'Huntingdon "Juniata" Station'
    app.platform = "darwin"
    app.api.train(
        "42",
        with_stop(j["enroute"], "HUN", name=quoted),
        with_stop(j["at_hun"], "HUN", name=quoted),
    )
    result = app.run("42", "--compact", "--notify-at", "HUN", polls=2)
    (note,) = result.notifications
    script = note.argv[2]
    assert '"Juniata"' not in script.replace('\\"', "")


# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------


@bug("B09", "--once prints a full-screen Layout, so long routes are cut off at terminal height")
def test_b09_once_output_is_not_clipped_to_terminal_height(app: App) -> None:
    app.height = 25
    app.api.train("42", long_route_payload())
    result = app.run("42", "--once", "--all")
    for code in ["PGH", "XT0", "XT3", "GBG", "HUN", "PHL", "NYP"]:
        assert f"({code})" in result.plain


@bug("B10", "a whitespace-only statusMsg (sent for non-Amtrak trains) renders as a blank status")
def test_b10_blank_status_message_falls_back_to_active(app: App) -> None:
    app.api.train("42", train_payload("train_space_statusmsg.json"))
    result = app.run("42", "--once")
    assert "Active" in result.plain


@bug("B16", "focus mode hides the first N rows by count, so cancelled stops shift the window")
def test_b16_focus_mode_keeps_exactly_two_departed_stops(app: App) -> None:
    payload = long_route_payload()
    stations = payload["42"][0]["stations"]
    for s in stations[1:3]:
        s.update(status="Cancelled", arr=None, dep=None)
    app.api.train("42", payload)
    result = app.run("42", "--once")
    departed_rows = [line for line in result.plain.splitlines() if "Departed" in line]
    assert len(departed_rows) == 2


@bug("B18", "compact mode treats a cancelled stop as the next stop, losing the ETA")
def test_b18_compact_mode_skips_cancelled_stops(app: App) -> None:
    app.api.train("42", train_payload("train_cancelled_stops.json"))
    result = app.run("42", "--once", "--compact")
    assert "@ 12:50 PM" in result.plain


@bug("B22", "an unknown --from/--to code is silently ignored but still shown in the title")
def test_b22_unknown_filter_station_is_not_claimed_in_title(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    result = app.run("42", "--once", "--from", "ZZZ")
    assert "ZZZ → end" not in result.plain


@bug("B24", "the Status column is too narrow for 'Enroute (Plt N)', which wraps onto two lines")
def test_b24_platform_status_fits_on_one_line(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))  # HBG has platform 3
    result = app.run("42", "--once")
    assert "Enroute (Plt 3)" in result.plain


@bug("B25", "the live feed marks every future stop 'Enroute', so every row is styled as the current stop")
def test_b25_only_the_next_stop_is_marked_enroute(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    result = app.run("42", "--once")
    table_rows = [line for line in result.plain.splitlines() if line.startswith("│ │")]
    assert sum("Enroute" in row for row in table_rows) == 1


# -----------------------------------------------------------------------------
# Real API data (tests/fixtures/live/)
# -----------------------------------------------------------------------------


@bug("B27", "several runs share a train number and the API lists them oldest first; the code shows [0]")
def test_b27_train_number_shows_most_recent_run(app: App) -> None:
    app.now = LIVE_CAPTURED_AT
    app.api.train("5", live_train("5"))  # runs 5-30, 5-1, 5-2 are all on the road
    result = app.run("5", "--once")
    assert "#5 (5-2)" in result.plain  # the code comment's stated intent: "the most recent one"


@bug("B33", "a Completed train keeps its terminus 'Enroute', so it is shown as still arriving")
def test_b33_completed_train_is_shown_as_arrived(app: App) -> None:
    app.now = LIVE_CAPTURED_AT
    app.api.train("660", live_train("660"))  # trainState Completed, arrived NYP 10:57
    result = app.run("660", "--once")
    assert "100%" in result.plain
    assert "Next: New York Penn" not in result.plain


@bug("B35", "UTC ('...Z') times are displayed in UTC instead of the station's local time")
def test_b35_utc_times_are_shown_in_station_local_time(app: App) -> None:
    app.now = LIVE_CAPTURED_AT
    app.api.train("b5712", live_train("b5712"))  # Miami schDep 2026-10-03T11:42:00.000Z = 7:42 AM EDT
    result = app.run("b5712", "--once")
    assert "7:42 AM" in result.plain


# -----------------------------------------------------------------------------
# CLI and packaging
# -----------------------------------------------------------------------------


@bug("B05", "a negative --refresh is accepted and crashes in sleep()")
def test_b05_negative_refresh_is_rejected_by_argument_parser(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    result = app.run("42", "--compact", "-r", "-5", polls=1)
    assert result.exception is None
    assert result.exit_code == 2


@bug("B12", "amtrak_status.__version__ is hard-coded and stale")
def test_b12_dunder_version_matches_installed_metadata() -> None:
    import amtrak_status

    assert amtrak_status.__version__ == importlib.metadata.version("amtrak-status")


def test_bug_tests_do_not_mutate_shared_fixtures() -> None:
    """Guard: helpers above must deep-copy, or one xfail could mask another."""
    a = train_payload(UPSTREAM_42)
    b = copy.deepcopy(a)
    with_stop(a, "HUN", status="Station")
    assert a == b
