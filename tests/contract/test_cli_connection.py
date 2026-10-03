"""Contract tests for connection (layover) classification at its boundaries.

Train 42 is due at PHL at 14:50 in the upstream-shaped fixture; train 171's PHL
departure is moved to produce each layover length.  Assertions include the Rich
style of the text, so colour regressions fail too.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from _harness import App, style_at, train_payload
from _payloads import NER_171, UPSTREAM_42, with_stop

ARRIVAL = datetime(2026, 2, 8, 14, 50, tzinfo=ZoneInfo("America/New_York"))


def iso(dt: datetime) -> str:
    return dt.isoformat()


def run_connection(app: App, t42: dict, t171: dict) -> str:
    """Raw (ANSI) output of a one-shot connection run."""
    app.api.train("42", t42)
    app.api.train("171", t171)
    result = app.run("42", "171", "--connection", "PHL", "--once")
    assert result.exception is None
    assert result.exit_code == 0
    return result.stdout


def departing_after(minutes: int, **fields) -> dict:
    dep = iso(ARRIVAL + timedelta(minutes=minutes))
    return with_stop(train_payload(NER_171), "PHL", schArr=dep, schDep=dep, arr=dep, dep=dep, **fields)


@pytest.mark.parametrize(
    "minutes, text, style",
    [
        (-1, "✗ MISSED by 1 min", "bold red"),
        (0, "⚠ 0 min layover (risky!)", "red"),
        (29, "⚠ 29 min layover (risky!)", "red"),
        (30, "⚡ 30 min layover (tight)", "yellow"),
        (44, "⚡ 44 min layover (tight)", "yellow"),
        (45, "⚡ 45 min layover (tight)", "yellow"),  # B29: same band as 30-44 today
        (59, "⚡ 59 min layover (tight)", "yellow"),
        (60, "✓ 1h 0m layover", "green"),
        (61, "✓ 1h 1m layover", "green"),
        (135, "✓ 2h 15m layover", "green"),
    ],
)
def test_layover_classification(app: App, minutes: int, text: str, style: str) -> None:
    screen = run_connection(app, train_payload(UPSTREAM_42), departing_after(minutes))
    assert style_at(screen, text) == style


@pytest.mark.parametrize(
    "minutes, title_style, border_style",
    [
        (-1, "bold red", "red"),
        (29, "bold yellow", "red"),
        (45, "bold yellow", "yellow"),
        (60, "bold green", "green"),
    ],
)
def test_connection_panel_colours_follow_layover(app: App, minutes: int, title_style: str, border_style: str) -> None:
    screen = run_connection(app, train_payload(UPSTREAM_42), departing_after(minutes))
    assert style_at(screen, "🔗 Connection at Philadelphia 30th Street (PHL)") == title_style
    assert style_at(screen, "╭", on_line_with="🔗 Connection") == border_style


def test_unknown_layover_when_departure_time_missing(app: App) -> None:
    t171 = with_stop(train_payload(NER_171), "PHL", schArr=None, schDep=None, arr=None, dep=None)
    screen = run_connection(app, train_payload(UPSTREAM_42), t171)
    assert style_at(screen, "— Layover unknown") == "dim"


def test_train2_already_departed_marks_connection_missed(app: App) -> None:
    t171 = departing_after(-30, status="Departed")
    screen = run_connection(app, train_payload(UPSTREAM_42), t171)
    assert "MISSED" in screen
    assert style_at(screen, "Departs: 2:20 PM") == "red"
    assert style_at(screen, "✗ Departed") == "red"


def test_train2_departed_after_train1_arrived_is_a_made_connection(app: App) -> None:
    t42 = with_stop(train_payload(UPSTREAM_42), "PHL", status="Departed", arr=iso(ARRIVAL), dep=iso(ARRIVAL))
    t171 = departing_after(40, status="Departed")
    screen = run_connection(app, t42, t171)
    assert "MISSED" not in screen
    assert style_at(screen, "✓ Arrived") == "green"
    assert style_at(screen, "✓ Departed") == "green"


def test_both_trains_at_connection_station(app: App) -> None:
    t42 = with_stop(train_payload(UPSTREAM_42), "PHL", status="Station", arr=iso(ARRIVAL))
    t171 = departing_after(20, status="Station")
    screen = run_connection(app, t42, t171)
    assert style_at(screen, "● At station") == "bold cyan"
    assert style_at(screen, "● Boarding") == "bold cyan"
