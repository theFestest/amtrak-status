"""Golden-master tests: the full observable output of one-shot (``--once``) runs.

Every scenario records argv, exit code, HTTP requests, sleeps, notifications and
the styled screen output in ``golden/<id>.txt``.  These pin *current*
behaviour, bugs included (see docs/known-bugs.md); a refactor must leave every
golden file byte-identical.  When a bug is fixed on purpose, regenerate with
``UPDATE_GOLDEN=1 pytest tests/contract`` and review the diff.
"""

from __future__ import annotations

import pytest

from _harness import App, HTTPStatus, NetworkError, RawBody, assert_golden, golden_document, load_fixture, train_payload
from _payloads import MIDJOURNEY, NER_171, UPSTREAM_42, depart_through, drop_stations, long_route_payload


# -----------------------------------------------------------------------------
# Single train
# -----------------------------------------------------------------------------

SINGLE_TRAIN = {
    # id: (fixture or payload factory, extra argv)
    "single-midjourney": (MIDJOURNEY, []),
    "single-midjourney-all": (MIDJOURNEY, ["--all"]),
    "single-midjourney-compact": (MIDJOURNEY, ["--compact"]),
    "single-midjourney-from-to": (MIDJOURNEY, ["--from", "ALT", "--to", "PHL"]),
    "single-midjourney-from-to-reversed": (MIDJOURNEY, ["--from", "PHL", "--to", "ALT"]),
    "single-midjourney-from-lowercase": (MIDJOURNEY, ["--from", "hun"]),
    "single-midjourney-from-unknown": (MIDJOURNEY, ["--from", "ZZZ"]),
    "single-midjourney-refresh-60": (MIDJOURNEY, ["-r", "60"]),
    "single-long-route-focus": (long_route_payload, []),
    "single-long-route-all": (long_route_payload, ["--all"]),
    "single-long-route-no-focus": (long_route_payload, ["--no-focus"]),
    "single-long-route-filtered-focus": (long_route_payload, ["--from", "XT1"]),
    "single-upstream-shape": (UPSTREAM_42, []),
    "single-upstream-shape-compact": (UPSTREAM_42, ["--compact"]),
    "single-cancelled-stops": ("train_cancelled_stops.json", []),
    "single-cancelled-stops-compact": ("train_cancelled_stops.json", ["--compact"]),
    "single-completed": ("train_completed.json", []),
    "single-completed-compact": ("train_completed.json", ["--compact"]),
    "single-delayed": ("train_delayed_iso.json", []),
    "single-delayed-compact": ("train_delayed_iso.json", ["--compact"]),
    "single-missing-fields": ("train_missing_fields.json", []),
    "single-multi-day": ("train_multi_day.json", []),
    "single-predeparture": ("train_predeparture.json", []),
    "single-predeparture-compact": ("train_predeparture.json", ["--compact"]),
    "single-space-statusmsg": ("train_space_statusmsg.json", []),
    "single-with-alerts": ("train_with_alerts.json", []),
    "single-notify-at-once": (MIDJOURNEY, ["--notify-at", "PHL, nyp"]),
    "single-notify-all-once": (MIDJOURNEY, ["--notify-all"]),
}


@pytest.mark.parametrize("scenario", SINGLE_TRAIN)
def test_single_train_once(app: App, scenario: str) -> None:
    source, extra = SINGLE_TRAIN[scenario]
    app.api.train("42", source() if callable(source) else train_payload(source))
    argv = ["42", "--once", *extra]
    result = app.run(*argv)
    assert result.exception is None
    assert_golden(scenario, golden_document(argv, result))


SINGLE_TRAIN_API_FAILURES = {
    "single-not-found": [],
    "single-http-503": HTTPStatus(503, "Service Unavailable"),
    "single-network-error": NetworkError("[Errno 111] Connection refused"),
}


@pytest.mark.parametrize("scenario", SINGLE_TRAIN_API_FAILURES)
@pytest.mark.parametrize("compact", [False, True], ids=["full", "compact"])
def test_single_train_api_failures(app: App, scenario: str, compact: bool) -> None:
    app.api.train("42", SINGLE_TRAIN_API_FAILURES[scenario])
    argv = ["42", "--once", *(["--compact"] if compact else [])]
    result = app.run(*argv)
    assert result.exception is None
    assert_golden(f"{scenario}{'-compact' if compact else ''}", golden_document(argv, result))


def test_day_specific_train_id(app: App) -> None:
    """``42-8`` is requested verbatim; the real API answers keyed by ``"42"``."""
    app.api.train("42-8", train_payload(MIDJOURNEY))
    argv = ["42-8", "--once"]
    result = app.run(*argv)
    assert result.requests == ["/v3/trains/42-8"]
    assert_golden("single-day-specific-id", golden_document(argv, result))


def test_plain_output_when_not_a_terminal(app: App) -> None:
    """Without a TTY (e.g. piped into a status bar) the output carries no ANSI codes."""
    app.color = False
    app.api.train("42", train_payload(MIDJOURNEY))
    argv = ["42", "--once", "--compact"]
    result = app.run(*argv)
    assert "\x1b[" not in result.stdout
    assert_golden("single-midjourney-compact-no-tty", golden_document(argv, result))


# -----------------------------------------------------------------------------
# Two trains with a connection
# -----------------------------------------------------------------------------

def _connection_api(app: App, train1, train2, station=None) -> None:
    app.api.train("42", train1)
    app.api.train("171", train2)
    if station is not None:
        app.api.station("PHL", station)


CONNECTIONS = {
    # id: (train 42 payload, train 171 payload, station payload, extra argv, stdin)
    "connection-explicit": (
        lambda: train_payload(MIDJOURNEY), lambda: train_payload(NER_171), None, ["--connection", "PHL"], ""),
    "connection-explicit-lowercase": (
        lambda: train_payload(MIDJOURNEY), lambda: train_payload(NER_171), None, ["--connection", "phl"], ""),
    "connection-explicit-upstream-shape": (
        lambda: train_payload(UPSTREAM_42), lambda: train_payload(NER_171), None, ["--connection", "PHL"], ""),
    "connection-explicit-all": (
        lambda: train_payload(UPSTREAM_42), lambda: train_payload(NER_171), None, ["--connection", "PHL", "--all"], ""),
    "connection-train1-arrived": (
        lambda: depart_through(train_payload(UPSTREAM_42), "PHL"), lambda: train_payload(NER_171), None,
        ["--connection", "PHL"], ""),
    "connection-autodetect-prompt-number": (
        lambda: train_payload(UPSTREAM_42), lambda: train_payload(NER_171), None, [], "1\n"),
    "connection-autodetect-prompt-code": (
        lambda: train_payload(UPSTREAM_42), lambda: train_payload(NER_171), None, [], "NYP\n"),
    "connection-autodetect-single-overlap": (
        lambda: train_payload(UPSTREAM_42), lambda: drop_stations(train_payload(NER_171), "NYP"), None, [], ""),
    "connection-no-overlap": (
        lambda: train_payload(UPSTREAM_42), lambda: drop_stations(train_payload(NER_171), "NYP", "PHL"), None, [], ""),
    "connection-train2-missing-explicit": (
        lambda: train_payload(UPSTREAM_42), lambda: [], lambda: _station_phl(), ["--connection", "PHL"], ""),
    "connection-train1-missing-explicit": (
        lambda: [], lambda: train_payload(NER_171), lambda: _station_phl(), ["--connection", "PHL"], ""),
    "connection-both-missing-explicit": (
        lambda: [], lambda: [], lambda: _station_phl(), ["--connection", "PHL"], ""),
    "connection-train2-missing-prompt": (
        lambda: train_payload(UPSTREAM_42), lambda: [], lambda: _station_phl(), [], "PHL\n"),
    "connection-train1-missing-prompt": (
        lambda: [], lambda: train_payload(NER_171), lambda: _station_phl(), [], "\n"),
    "connection-both-missing-prompt": (
        lambda: [], lambda: [], lambda: _station_phl(), [], "phl\n"),
    "connection-compact-once": (
        lambda: train_payload(UPSTREAM_42), lambda: train_payload(NER_171), None,
        ["--connection", "PHL", "--compact"], ""),
}


def _station_phl():
    return load_fixture("station_schedule.json")


@pytest.mark.parametrize("scenario", CONNECTIONS)
def test_connection_once(app: App, scenario: str) -> None:
    t1, t2, station, extra, stdin = CONNECTIONS[scenario]
    _connection_api(app, t1(), t2(), station() if station else None)
    argv = ["42", "171", "--once", *extra]
    result = app.run(*argv, stdin=stdin)
    assert result.exception is None
    assert_golden(scenario, golden_document(argv, result, stdin))


def test_connection_station_api_error(app: App) -> None:
    _connection_api(app, train_payload(UPSTREAM_42), [], HTTPStatus(500))
    argv = ["42", "171", "--once", "--connection", "PHL"]
    result = app.run(*argv)
    assert result.exception is None
    assert_golden("connection-station-api-error", golden_document(argv, result))


def test_non_json_body_is_recorded(app: App) -> None:
    """Pins today's crash on a non-JSON 200 response (bug B03) so a fix is a visible golden change."""
    app.api.train("42", RawBody("<html>Bad gateway</html>"))
    argv = ["42", "--once"]
    result = app.run(*argv)
    assert result.exception is not None
    assert_golden("single-non-json-body", golden_document(argv, result))
