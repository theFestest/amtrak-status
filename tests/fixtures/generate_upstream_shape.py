"""Regenerate the fixtures that follow the live API's rules.

    uv run python tests/fixtures/generate_upstream_shape.py

Rules mirror ``parseRawStation()`` in amtraker-v3 ``index.ts`` @ 74789d0
(https://github.com/piemadd/amtraker-v3/blob/74789d05790d6b2f34b33044b6c9b5c3bba7041e/index.ts):

- origin: ``schArr = schDep`` (``scharr ?? schdep``); once departed, ``arr = dep = postdep``
- departed intermediate stop: ``arr = postarr``, ``dep = postdep``, status ``Departed``
- every stop not yet reached: status ``Enroute``, ``arr = estarr``, ``dep = arr`` (``dep ?? arr``)
- terminus: ``schDep = schArr`` (``schdep ?? scharr``)
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
D = "2026-02-08T{}:00-05:00"


def stop(code, name, sch_arr, sch_dep, status, arr=None, dep=None, platform=""):
    sch_arr = sch_arr or sch_dep
    sch_dep = sch_dep or sch_arr
    if status == "Enroute":
        dep = arr  # upstream copies the arrival estimate into dep
    arr = arr or dep
    dep = dep or arr
    return {
        "name": name, "code": code, "tz": "America/New_York", "bus": False,
        "schArr": D.format(sch_arr), "schDep": D.format(sch_dep),
        "arr": D.format(arr) if arr else None, "dep": D.format(dep) if dep else None,
        "arrCmnt": "", "depCmnt": "", "status": status,
        "stopIconColor": "#212529", "platform": platform,
    }


def train(num, route, tid, stations, **kw):
    base = {
        "routeName": route, "trainNum": num, "trainNumRaw": num, "trainID": tid,
        "lat": 0.0, "lon": 0.0, "trainTimely": "", "iconColor": "#212529", "textColor": "#ffffff",
        "stations": stations, "heading": "E",
        "eventCode": "", "eventTZ": "America/New_York", "eventName": "",
        "origCode": stations[0]["code"], "originTZ": "America/New_York", "origName": stations[0]["name"],
        "destCode": stations[-1]["code"], "destTZ": "America/New_York", "destName": stations[-1]["name"],
        "trainState": "Active", "velocity": 0, "statusMsg": "On Time",
        "createdAt": D.format("07:00"), "updatedAt": D.format("10:58"), "lastValTS": D.format("10:58"),
        "objectID": 0, "provider": "Amtrak", "providerShort": "AMTK", "onlyOfTrainNum": True, "alerts": [],
    }
    base.update(kw)
    return base


TRAIN_42 = train("42", "Pennsylvanian", "42-8", [
    stop("PGH", "Pittsburgh", None, "07:30", "Departed", dep="07:32"),
    stop("GBG", "Greensburg", "08:15", "08:17", "Departed", arr="08:18", dep="08:20"),
    stop("LAT", "Latrobe", "08:35", "08:36", "Departed", arr="08:38", dep="08:40"),
    stop("JST", "Johnstown", "09:25", "09:27", "Departed", arr="09:30", dep="09:32"),
    stop("ALT", "Altoona", "10:20", "10:30", "Departed", arr="10:25", dep="10:35"),
    stop("HUN", "Huntingdon", "11:15", "11:16", "Enroute", arr="11:20"),
    stop("HBG", "Harrisburg", "12:45", "12:55", "Enroute", arr="12:50", platform="3"),
    stop("LNC", "Lancaster", "13:30", "13:32", "Enroute", arr="13:35"),
    stop("PHL", "Philadelphia 30th Street", "14:45", "14:55", "Enroute", arr="14:50"),
    stop("NYP", "New York Penn", "16:30", None, "Enroute", arr="16:35"),
], heading="E", velocity=62.3, eventCode="ALT", eventName="Altoona", statusMsg="On Time")

TRAIN_171 = train("171", "Northeast Regional", "171-8", [
    stop("BOS", "Boston South Station", None, "09:15", "Departed", dep="09:17"),
    stop("PVD", "Providence", "09:55", "09:57", "Departed", arr="10:00", dep="10:02"),
    stop("NHV", "New Haven Union Station", "11:30", "11:35", "Enroute", arr="11:36"),
    stop("NYP", "New York Penn", "13:20", "13:40", "Enroute", arr="13:25"),
    stop("NWK", "Newark Penn Station", "13:55", "13:57", "Enroute", arr="14:00"),
    stop("TRE", "Trenton", "14:40", "14:41", "Enroute", arr="14:44"),
    stop("PHL", "Philadelphia 30th Street", "15:20", "15:30", "Enroute", arr="15:22"),
    stop("WIL", "Wilmington", "15:55", "15:56", "Enroute", arr="15:57"),
    stop("BAL", "Baltimore Penn Station", "16:45", "16:47", "Enroute", arr="16:47"),
    stop("WAS", "Washington Union Station", "17:25", None, "Enroute", arr="17:25"),
], heading="SW", velocity=79.0, eventCode="PVD", eventName="Providence", statusMsg="On Time")


if __name__ == "__main__":
    for name, t in [("train_42_upstream_shape.json", TRAIN_42), ("train_171_northeast_regional.json", TRAIN_171)]:
        with open(HERE / name, "w") as f:
            json.dump({t["trainNum"]: [t]}, f, indent=2)
            f.write("\n")
