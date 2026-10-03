"""Builders that derive scenario payloads from the JSON fixtures.

All helpers return new objects and never mutate their input.
"""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

from _harness import load_fixture, train_payload

MIDJOURNEY = "train_active_midjourney.json"
UPSTREAM_42 = "train_42_upstream_shape.json"
NER_171 = "train_171_northeast_regional.json"

# tests/fixtures/live/ holds real responses captured at this instant (see tests/fixtures/README.md).
LIVE_CAPTURED_AT = datetime(2026, 10, 3, 11, 4, 30, tzinfo=ZoneInfo("America/New_York"))


def live_train(number: str, train_id: str | None = None) -> dict:
    """A captured ``/v3/trains/{number}`` body, optionally narrowed to one run like ``/v3/trains/5-2``."""
    payload = load_fixture(f"live/train_{number}.json")
    if train_id is not None:
        payload = {number: [t for t in payload[number] if t["trainID"] == train_id]}
    return payload


def _train(payload: dict) -> dict:
    return next(iter(payload.values()))[0]


def with_stop(payload: dict, code: str, **fields) -> dict:
    """Copy of ``payload`` with ``fields`` updated on the stop ``code``."""
    payload = copy.deepcopy(payload)
    for s in _train(payload)["stations"]:
        if s["code"] == code:
            s.update(fields)
    return payload


def drop_stations(payload: dict, *codes: str) -> dict:
    payload = copy.deepcopy(payload)
    train = _train(payload)
    train["stations"] = [s for s in train["stations"] if s["code"] not in codes]
    return payload


def depart_through(payload: dict, code: str) -> dict:
    """Mark every stop up to and including ``code`` as departed on schedule."""
    payload = copy.deepcopy(payload)
    for s in _train(payload)["stations"]:
        s.update(status="Departed", arr=s["schArr"], dep=s["schDep"])
        if s["code"] == code:
            break
    return payload


def long_route_payload() -> dict:
    """Midjourney train with four extra departed stops (14 total) so focus mode engages."""
    payload = train_payload(MIDJOURNEY)
    stations = _train(payload)["stations"]
    for n in range(4):
        extra = copy.deepcopy(stations[1])
        extra.update(code=f"XT{n}", name=f"Extra Stop {n}")
        stations.insert(1 + n, extra)
    return payload


def journey() -> dict[str, dict]:
    """Train 42 approaching, at, then leaving Huntingdon (upstream API semantics)."""
    enroute = train_payload(UPSTREAM_42)
    at_hun = with_stop(
        enroute, "HUN", status="Station", arr="2026-02-08T11:20:00-05:00", dep="2026-02-08T11:22:00-05:00"
    )
    left_hun = with_stop(at_hun, "HUN", status="Departed", dep="2026-02-08T11:23:00-05:00")
    return {"enroute": enroute, "at_hun": at_hun, "left_hun": left_hun}
