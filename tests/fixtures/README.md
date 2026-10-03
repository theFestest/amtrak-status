# Test fixtures

JSON bodies as returned by `GET https://api-v3.amtraker.com/v3/trains/{id}` (`{"<number>": [train, …]}`)
or `GET …/v3/stations/{code}`.

## Real responses: `live/`

Captured from the live API on 2026-10-03 at 11:04:30 EDT (15:04:30Z), from one snapshot of
`/v3/trains`, so all trains share a single instant. Tests that use them freeze the clock at that
instant (`LIVE_CAPTURED_AT` in `tests/contract/_payloads.py`).

| File | Why it's here |
|---|---|
| `live/train_42.json` | Pennsylvanian 42-3 mid-route, 17 stops |
| `live/train_5.json` | California Zephyr: three runs (5-30, 5-1, 5-2) on the road at once, listed oldest first (B27) |
| `live/train_757.json` | Pacific Surfliner 757-3: last four stops are `Station` with no times (no data / skipped) |
| `live/train_660.json` | Keystone 660-3, `trainState: Completed`, final stop still `Enroute` (B33) |
| `live/train_b5712.json` | Brightline: UTC (`…Z`) times, every stop `Enroute` (B35) |
| `live/station_PHL.json` | `/v3/stations/PHL`: metadata plus train IDs, no times (B23) |

## Rules the live API follows

Write new hand-made fixtures to match what the live snapshot showed (see also
[`amtraker-v3/index.ts`](https://github.com/piemadd/amtraker-v3/blob/74789d05790d6b2f34b33044b6c9b5c3bba7041e/index.ts),
`parseRawStation`/`parseDate`):

- times are ISO-8601 strings with an offset (`2026-02-08T07:25:00-05:00`); Brightline uses UTC `…:00.000Z`;
- stop `status` is `Departed`, `Station` or `Enroute`, and **every** stop not yet reached is `Enroute`;
- for a stop not yet reached, `dep` usually equals the arrival estimate (`dep ?? arr`);
- `schArr`/`schDep` fall back to each other (origin and terminus have both);
- `statusMsg` is `" "` and `trainTimely` is `""` for every train;
- a predeparture train's origin is usually `Enroute`, with `arr`/`dep` equal to the schedule;
- unknown train or station → `[]`.

## Hand-made and generated fixtures

| File | Matches the live rules? | Scenario |
|---|---|---|
| `train_42_upstream_shape.json` | yes (generated) | Train 42 between Altoona and Huntingdon at 11:00; HBG has platform 3 |
| `train_171_northeast_regional.json` | yes (generated) | Train 171 BOS→WAS, departs PHL 15:30 (schedule), shares PHL and NYP with train 42 |
| `train_active_midjourney.json` | no: future stops have `status: ""` and `dep: null` | Train 42 between Altoona and Huntingdon |
| `train_cancelled_stops.json` | no | GBG `Cancelled`, LAT without times (cancelled-stop heuristic) |
| `train_completed.json` | no: real completed trains keep the last stop `Enroute` | All stops departed, `trainState: Completed` |
| `train_delayed_iso.json` | no: the live feed has no status text | 15 min late, `statusMsg: "15 Minutes Late"` |
| `train_missing_fields.json` | no | `dep` key absent, `arr: null`, `bus: true`, platforms `5B`/`3A` |
| `train_multi_day.json` | no: real runs are listed oldest first | Two trains under key `42` (today active first, yesterday completed second) |
| `train_predeparture.json` | no: future stops have `status: ""` | `trainState: Predeparture` |
| `train_space_statusmsg.json` | yes | `statusMsg: " "` |
| `train_with_alerts.json` | no | Two `alerts` entries |
| `train_not_found.json` | yes | `[]` |
| `station_schedule.json` | yes | `/v3/stations/PHL`: metadata plus a list of train IDs, no times |
| `station_not_found.json` | yes | `[]` |

`generate_upstream_shape.py` regenerates the two generated files.
