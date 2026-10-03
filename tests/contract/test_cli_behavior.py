"""Contract tests for refreshing modes: polling, caching, retries, notifications.

These drive the real refresh loops: each ``sleep`` advances the frozen clock,
and the run stops (as if Ctrl+C were pressed) at the N-th refresh sleep.  Live
(full-screen) frames depend on Rich's refresh thread timing, so full-screen
runs assert only on side effects; screen content is checked in compact mode,
which prints deterministically.
"""

from __future__ import annotations

import pytest

from _harness import App, HTTPStatus, NetworkError, train_payload
from _payloads import NER_171, UPSTREAM_42, journey



# -----------------------------------------------------------------------------
# Polling
# -----------------------------------------------------------------------------


def test_compact_mode_reprints_each_refresh(app: App) -> None:
    j = journey()
    app.color = False
    app.api.train("42", j["enroute"], j["at_hun"], j["left_hun"])
    result = app.run("42", "--compact", polls=3)

    assert result.exit_code == 0
    assert result.requests == ["/v3/trains/42"] * 3
    assert result.sleeps == [30, 30, 30]
    lines = [line for line in result.plain.splitlines() if line.strip()]
    assert len(lines) == 3
    assert "Updated 11:00:00" in lines[0] and "→HUN" in lines[0]
    assert "Updated 11:00:30" in lines[1]
    assert "→HBG" in lines[2] and "Updated 11:01:00" in lines[2]


def test_refresh_interval_is_used_between_polls(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    result = app.run("42", "--compact", "-r", "45", polls=2)
    assert result.sleeps == [45, 45]
    assert "Updated 11:00:45" in result.plain


def test_full_screen_mode_polls_until_ctrl_c(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    result = app.run("42", polls=3)
    assert result.exception is None
    assert result.exit_code == 0
    assert result.requests == ["/v3/trains/42"] * 3
    assert result.plain.rstrip().endswith("Tracking stopped.")


def test_connection_mode_polls_both_trains(app: App) -> None:
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", train_payload(NER_171))
    result = app.run("42", "171", "--connection", "PHL", polls=2)
    assert result.exception is None
    assert result.exit_code == 0
    # two lookups during setup, then both trains on every refresh
    assert result.requests == ["/v3/trains/42", "/v3/trains/171"] * 3
    assert result.sleeps == [1, 30, 30]


def test_connection_compact_mode_prints_one_line_per_train(app: App) -> None:
    app.color = False
    app.api.train("42", train_payload(UPSTREAM_42))
    app.api.train("171", [])
    result = app.run("42", "171", "--connection", "PHL", "--compact", polls=1)
    assert result.exit_code == 0
    assert "Pennsylvanian #42" in result.plain
    assert "Train #171: Error or not found" in result.plain


# -----------------------------------------------------------------------------
# Failures and cached data
# -----------------------------------------------------------------------------


def test_transient_failure_falls_back_to_cached_data(app: App) -> None:
    app.color = False
    app.api.train("42", train_payload(UPSTREAM_42), HTTPStatus(503))
    result = app.run("42", "--compact", polls=2)
    # first poll succeeds; second poll retries 3 times (sleeping 2s then 4s) then uses the cache
    assert result.requests == ["/v3/trains/42"] * 4
    assert result.sleeps == [30, 2, 4, 30]
    lines = [line for line in result.plain.splitlines() if line.strip()]
    assert all("Pennsylvanian #42" in line for line in lines)
    assert "Updated 11:00:00" in lines[-1]  # time of the last *successful* fetch


def test_cached_data_expires_after_five_minutes_in_single_train_mode(app: App) -> None:
    app.color = False
    app.api.train("42", train_payload(UPSTREAM_42), [])
    result = app.run("42", "--compact", "-r", "60", polls=7)
    lines = [line for line in result.plain.splitlines() if line.strip()]
    assert "Pennsylvanian #42" in lines[4]  # 4 min after the last good fetch
    assert lines[5] == "🚂 Train #42 not found"  # 5 min: cache is stale


def test_network_errors_retry_with_increasing_backoff(app: App) -> None:
    app.api.train("42", NetworkError())
    result = app.run("42", "--once")
    assert result.requests == ["/v3/trains/42"] * 3
    assert result.sleeps == [2, 4]
    assert "Connection refused" in result.plain


# -----------------------------------------------------------------------------
# Notifications
# -----------------------------------------------------------------------------


def test_notify_at_fires_once_when_train_reaches_station(app: App) -> None:
    j = journey()
    app.api.train("42", j["enroute"], j["at_hun"], j["left_hun"])
    result = app.run("42", "--notify-at", "hun", polls=3)
    assert result.exit_code == 0
    assert [n.argv for n in result.notifications] == [
        ["notify-send", "-a", "Amtrak Tracker", "🚂 Pennsylvanian #42 Arriving", "Now arriving at Huntingdon (HUN)"],
    ]
    assert "Notifications enabled for: HUN" in result.plain


def test_notify_at_reports_departure_if_arrival_was_missed_between_polls(app: App) -> None:
    j = journey()
    app.api.train("42", j["enroute"], j["left_hun"])
    result = app.run("42", "--notify-at", "HUN", polls=2)
    assert [n.argv[-2:] for n in result.notifications] == [
        ["🚂 Pennsylvanian #42 Departed", "Departed from Huntingdon (HUN)"],
    ]


def test_notify_all_skips_stops_already_passed_at_startup(app: App) -> None:
    j = journey()
    app.api.train("42", j["enroute"], j["at_hun"], j["left_hun"])
    result = app.run("42", "--notify-all", polls=3)
    assert [n.argv[-1] for n in result.notifications] == ["Now arriving at Huntingdon (HUN)"]


def test_no_notifications_without_notify_flags(app: App) -> None:
    j = journey()
    app.api.train("42", j["enroute"], j["at_hun"], j["left_hun"])
    result = app.run("42", polls=3)
    assert result.notifications == []


@pytest.mark.parametrize(
    "platform, program",
    [("linux", "notify-send"), ("darwin", "osascript"), ("win32", "powershell")],
)
def test_notification_command_per_platform(app: App, platform: str, program: str) -> None:
    j = journey()
    app.platform = platform
    app.api.train("42", j["enroute"], j["at_hun"])
    result = app.run("42", "--compact", "--notify-at", "HUN", polls=2)
    assert len(result.notifications) == 1
    assert result.notifications[0].argv[0] == program
    assert "Huntingdon" in " ".join(result.notifications[0].argv)


def test_notification_falls_back_to_terminal_bell(app: App) -> None:
    j = journey()
    app.api.train("42", j["enroute"], j["at_hun"])
    result = app.run("42", "--compact", "--notify-at", "HUN", polls=2, notify_ok=False)
    assert len(result.notifications) == 1  # attempted
    assert "\a" in result.stdout


def test_unsupported_platform_rings_bell_without_subprocess(app: App) -> None:
    j = journey()
    app.platform = "freebsd14"
    app.api.train("42", j["enroute"], j["at_hun"])
    result = app.run("42", "--compact", "--notify-at", "HUN", polls=2)
    assert result.notifications == []
    assert "\a" in result.stdout


# -----------------------------------------------------------------------------
# CLI surface
# -----------------------------------------------------------------------------


def test_help_lists_every_option(app: App) -> None:
    result = app.run("--help")
    assert result.exit_code == 0
    for option in [
        "--refresh", "--once", "--compact", "--connection", "--from", "--to",
        "--all", "--no-focus", "--notify-at", "--notify-all",
    ]:
        assert option in result.plain


def test_train_number_is_required(app: App, capsys: pytest.CaptureFixture[str]) -> None:
    result = app.run()
    assert result.exit_code == 2
    assert "train_numbers" in capsys.readouterr().err
