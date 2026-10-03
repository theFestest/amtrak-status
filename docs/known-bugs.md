# Known bugs

Bugs found while preparing the refactor (October 2026). Each one was reproduced by running the
code, not just by reading it. Most have a strict-xfail test in
[`tests/contract/test_known_bugs.py`](../tests/contract/test_known_bugs.py) that asserts the
correct behaviour. When you fix a bug, its test starts passing and fails the run (`xfail_strict`);
delete the marker and regenerate any affected golden files in the same change.

Facts about the live API come from the Amtraker v3 server source,
[`piemadd/amtraker-v3` `index.ts` @ 74789d0](https://github.com/piemadd/amtraker-v3/blob/74789d05790d6b2f34b33044b6c9b5c3bba7041e/index.ts)
(`parseDate`, `parseRawStation`, and the `/v3/trains` and `/v3/stations` handlers). The API host
itself could not be reached from the environment where this was written, so these facts come
from reading the source, not from captured responses.

Severity: **High** = wrong information or a crash in normal use · **Med** = wrong in
plausible situations · **Low** = cosmetic or unlikely · **Decide** = the right behaviour is a
product decision.

## Connection (two-train) mode

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B01 | High | All notification state is global (`_notified_stations`, `_notifications_initialized`). The first train initialises it, so the second train's *already-passed* stops are reported as new at startup (`--notify-all` with train 171 already past BOS/PVD fires two notifications immediately). This is the "notifications don't work in connection mode" item from the CHANGELOG. | `test_b01_…` |
| B02 | High | Notifications are de-duplicated by station code across trains. After train 1 arrives at the connection station, train 2's arrival/boarding there is never announced. | `test_b02_…` |
| B06 | Low | `--compact --once` with two trains prints the full panel layout; `--once` is checked before `--compact`. | `test_b06_…` |
| B07 | Med | Two-train mode never expires cached data. `fetch_train_data_cached` re-stamps `fetch_time` whenever `fetch_train_data` hands back cached data, so with any refresh shorter than 5 min a train that vanished from the feed is shown indefinitely. In a probe the data was served for 20 of 20 minutes, versus 4.5 minutes in single-train mode. Synthetic predeparture data has the same lifetime problem. | `test_b07_…` |
| B08 | Med | `_last_error` and `_last_fetch_time` are single globals shared by both trains. Train 2's successful fetch clears train 1's "using cached data" warning, and the `Updated HH:MM:SS` stamp reflects whichever train was fetched last (a train whose data is 8 minutes stale shows `Updated 11:08:01`). | `test_b08_…` |
| B15 | Low | When the connection counts as missed because train 2 shows `Departed`, the panel prints `MISSED by {abs(layover)} min` even if the computed layover is positive (`MISSED by 40 min`). | `test_b15_…` |
| B17 | Low | `nargs="+"` accepts any number of trains, but only the first two are used; the rest are silently ignored. | `test_b17_…` |
| B21 | Low | The "Select connection station" prompt lists choices in upper case and Rich matches them case-sensitively, so typing `phl` is rejected. (`--connection phl` and the free-text prompts upper-case their input correctly.) | `test_b21_…` |
| B26 | Med | The live API copies a future stop's arrival estimate into `dep` (`dep: dep ?? arr`). The layover therefore uses train 2's *arrival* at the connection station as its departure, underestimating the layover by the dwell time (37 min shown where the scheduled departure gives 45). The same copy fills the "Act/Est Dep" column for every future stop. | `test_b26_…` |
| B23 | Med | The station-schedule fallback (≈150 lines in `main()` plus three helpers) can never work. `/v3/stations/{code}` returns station metadata with `trains: ["42-8", …]` (IDs only, no times), and the API only lists trains that are active or less than 1 h from departure. There is no endpoint with a timetable for trains not yet in the live feed. **Decide:** delete the feature, or source timetables elsewhere (e.g. Amtrak's GTFS feed). | legacy `TestFixtureStationEndpoint` (xfail) |
| B29 | Decide | `LAYOVER_TIGHT = 45` is never used for its own band: 30–44 and 45–59 minutes both classify as `tight`. Either drop the constant or introduce a fourth band. | legacy `test_layover_45_to_59_should_not_be_tight` (xfail) |

## Fetching and caching

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B03 | High | A 200 response that is not JSON (a captive portal or a proxy error page) raises `JSONDecodeError` straight out of `fetch_train_data`, which kills the live display. | `test_b03_…`, golden `single-non-json-body` |
| B04 | Low | A non-empty JSON list response (`[{…}]`) crashes with `AttributeError: 'list' object has no attribute 'keys'`. | `test_b04_…` |
| B27 | Med (unverified) | When the API returns several trains under one number (e.g. yesterday's still-running 42 and today's), the code takes element `[0]` on the assumption that it is "the most recent". Upstream pushes trains in raw feed order, so nothing guarantees that. Prefer `Active` trains, or the day given in `42-26` syntax. | — |
| B32 | Low | The retry comment says "exponential backoff", but the code sleeps 2 s, then 4 s (linear). Worst case the UI freezes for about 36 s per train (3 × 10 s timeouts plus sleeps), and in two-train mode the trains are fetched one after the other. | — |

## Display

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B09 | Med | `--once` prints a `Layout`, which always renders at exactly the terminal height. Long routes are cut off, and so is any output piped somewhere (Rich assumes 25 lines). | `test_b09_…` |
| B10 | Low | A whitespace-only `statusMsg` (upstream sends `" "` for every VIA, Brightline and CPKC train) is truthy, so the status shows blank instead of "Active". The legacy test `test_space_statusmsg_used_in_header` asserts this buggy behaviour. | `test_b10_…` |
| B11 | Low | Route, station and train names are interpolated into Rich markup strings. A `[/]` in API data raises `MarkupError`, and a `[bold]` is silently applied as a style. | `test_b11_…` |
| B16 | Low | Focus mode hides "all but the last two departed" stops by slicing off the first *N* rows. Cancelled stops among the departed ones shift the window, so 4 departed rows show instead of 2. | `test_b16_…` |
| B18 | Med | The compact one-liner doesn't skip cancelled stops when looking for the next stop (the full header does). It shows `@ —` instead of the real ETA. | `test_b18_…` |
| B19 | Decide | The delay threshold differs between views: the header shows `(+1m)`, but the compact line only shows delays greater than 1 minute. | — |
| B22 | Low | An unknown `--from`/`--to` code is silently ignored, yet the table title still claims `(ZZZ → end)`. | `test_b22_…` |
| B24 | Low | The Status column (width 14) is too narrow for `Enroute (Plt 3)`, which wraps onto two lines. | `test_b24_…` |
| B25 | Med | The live API marks **every** stop not yet reached as `Enroute` (never `""`). The table therefore styles every future stop as the current one (bold yellow `→`), and the `○ Scheduled` state never appears with real data. Most existing fixtures use `""` for future stops, which the real API never sends. | `test_b25_…` |
| B31 | Low | Two-train compact mode prints "Compact mode with connections – showing basic info", then immediately clears the screen. | — |
| B33 | Low (unverified) | Upstream marks the final stop `Station` (or leaves it without a status) after arrival, never `Departed`, so the progress bar may never reach 100 %. Verify against live data. | — |

## Notifications

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B20 | Med (security) | Notification text is spliced into AppleScript (`display notification "…"`) and a PowerShell script without escaping. A `"` in a station or route name breaks the notification on macOS. Worse, text from the API can break out of the string literal and inject commands (AppleScript `do shell script`, PowerShell `$(...)`). Pass the text as separate arguments instead. | `test_b20_…` |

## CLI, packaging, docs

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B05 | Low | `-r/--refresh` accepts negative numbers (crashes with `ValueError` from `sleep`) and 0 (polls the API in a tight loop). | `test_b05_…` |
| B12 | Low | `amtrak_status.__version__` is hard-coded as `0.1.0`, while the package is `0.1.3`. The release workflow only bumps `pyproject.toml`. Use `importlib.metadata.version`. | `test_b12_…` |
| B13 | Low | The README options table lists 3 of the 11 flags and doesn't mention connection mode, `--compact`, `--from/--to` or notifications. (Its development instructions, which used a non-existent `.[test]` extra, were fixed alongside this document.) | — |

## Latent and test-only issues

| ID | Sev | Issue |
|----|-----|-------|
| B14 | Low | `int(seconds / 60)` truncates toward zero, so a layover of −30 s becomes 0 min, which counts as "risky" rather than "missed". This is theoretical while upstream times have whole-minute precision. |
| B28 | Test | `_now()` exists "for test patching", but `calculate_position_between_stations` only uses it for naive datetimes. For real (timezone-aware ISO) data it calls `datetime.now(timezone.utc)` directly, so position tests on fixture data run against the real clock and can only assert `0 ≤ progress ≤ 1`. |
| B30 | Test | `parse_time` returns naive datetimes for epoch-millisecond input but aware ones for ISO strings (legacy xfail `test_comparing_naive_and_aware_raises`). `calculate_layover` "fixes" a mismatch by labelling the naive side as UTC, but it is actually local time. The live API only sends ISO strings, so the epoch-ms path only exists for the synthetic test data. |

## Dead code (delete during the refactor)

- `build_predeparture_panel()`: only called from tests.
- `_train_caches[n]["error"]`: written in three places, never read.
- `calculate_progress()` returns `current_idx`, but both callers ignore it. `build_stations_table` calls `find_current_station_index()` and never uses the result.
- `select_connection_station()`: the "Invalid selection" retry loop and the `None` return are unreachable, because `Prompt.ask(choices=…)` already validates. The caller's `if not CONNECTION_STATION: sys.exit(1)` is unreachable for the same reason.
- `getattr(args, 'all', False)`: `args.all` always exists.
- `--all` and `--no-focus` do the same thing through two different mechanisms (`FOCUS_CURRENT` and `show_all`).
