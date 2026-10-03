# Known bugs

Bugs found while preparing the refactor (October 2026). Each one was reproduced by running the
code, not just by reading it. Most have a strict-xfail test in
[`tests/contract/test_known_bugs.py`](../tests/contract/test_known_bugs.py) that asserts the
correct behaviour. When you fix a bug, its test starts passing and fails the run (`xfail_strict`);
delete the marker and regenerate any affected golden files in the same change.

Facts about the API come from two places:

* **The live API.** A snapshot of `/v3/trains` (180 trains: 140 Amtrak, 31 VIA, 9 Brightline),
  `/v3/stations` and `/v3/stations/{,expanded/}PHL`, taken 2026-10-03 11:04:30 EDT. Six of the
  responses are kept in [`tests/fixtures/live/`](../tests/fixtures/live/).
* **The Amtraker v3 server source.** [`piemadd/amtraker-v3`](https://github.com/piemadd/amtraker-v3)
  (`index.ts` at 74789d0, plus its history for B23).

Where the two disagreed, the live data won and the entry says so.

Severity: **High** = wrong information or a crash in normal use · **Med** = wrong in
plausible situations · **Low** = cosmetic or unlikely · **Decide** = the right behaviour is a
product decision.

## Connection (two-train) mode

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B01 | High | All notification state is global (`_notified_stations`, `_notifications_initialized`). The first train initialises it, so the second train's *already-passed* stops are reported as new at startup (`--notify-all` with train 171 already past BOS/PVD fires two notifications immediately). This is the "notifications don't work in connection mode" item from the CHANGELOG. | `test_b01_…` |
| B02 | High | Notifications are de-duplicated by station code across trains. After train 1 arrives at the connection station, train 2's arrival/boarding there is never announced. | `test_b02_…` |
| B06 | Low | `--compact --once` with two trains prints the full panel layout; `--once` is checked before `--compact`. | `test_b06_…` |
| B07 | Med | Two-train mode never expires cached data. `fetch_train_data_cached` re-stamps `fetch_time` whenever `fetch_train_data` hands back cached data, so with any refresh shorter than 5 min a train that vanished from the feed is shown indefinitely. In a probe the data was served for 20 of 20 minutes, versus 4.5 minutes in single-train mode. | `test_b07_…` |
| B08 | Med | `_last_error` and `_last_fetch_time` are single globals shared by both trains. Train 2's successful fetch clears train 1's "using cached data" warning, and the `Updated HH:MM:SS` stamp reflects whichever train was fetched last (a train whose data is 8 minutes stale shows `Updated 11:08:01`). | `test_b08_…` |
| B15 | Low | When the connection counts as missed because train 2 shows `Departed`, the panel prints `MISSED by {abs(layover)} min` even if the computed layover is positive (`MISSED by 40 min`). | `test_b15_…` |
| B17 | Low | `nargs="+"` accepts any number of trains, but only the first two are used; the rest are silently ignored. | `test_b17_…` |
| B21 | Low | The "Select connection station" prompt lists choices in upper case and Rich matches them case-sensitively, so typing `phl` is rejected. (`--connection phl` and the free-text prompts upper-case their input correctly.) | `test_b21_…` |
| B23 | Med | The station-schedule fallback can never work; see [below](#b23-the-station-schedule-fallback). **Decide:** delete it and document the limitation, or add a timetable source. | legacy `TestFixtureStationEndpoint` (xfail) |
| B26 | Med | For a stop not yet reached, the API usually copies the arrival estimate into `dep` (`dep: dep ?? arr`): live, `dep == arr` on 1,427 of 1,704 `Enroute` stops. The layover therefore uses train 2's *arrival* at the connection station as its departure, underestimating it by the dwell time (37 min shown where the scheduled departure gives 45). The same copy fills the "Act/Est Dep" column. | `test_b26_…` |
| B29 | Decide | `LAYOVER_TIGHT = 45` is never used for its own band: 30–44 and 45–59 minutes both classify as `tight`. Either drop the constant or introduce a fourth band. | legacy `test_layover_45_to_59_should_not_be_tight` (xfail); pinned in `test_cli_connection.py` |

### B23: the station-schedule fallback

**The gap it tries to cover.** In connection mode, the second train may not be in the live feed
yet, so the screen shows "Awaiting Departure" and the layover is unknown. Since upstream commit
[`db9d91c`](https://github.com/piemadd/amtraker-v3/commit/db9d91c) (2026-04-18, "Only put train
on api when within an hour of first departure"), a train appears in `/v3/trains` only from about
an hour before it leaves its **origin**. In the live snapshot, every `Predeparture` train was at
most 56 minutes from scheduled departure (two were a few minutes past it). This bites hardest when the connecting train starts at or near the
connection station and leaves more than an hour after you start tracking, e.g. a Keystone that
originates at Philadelphia, or a long trip where you start tracking in the morning.

**Why the fallback doesn't fill it.** `main()` calls `fetch_station_schedule(code)`, and
`get_train_schedule_from_station()` looks for a list of train objects with `schArr`/`schDep`.
`/v3/stations/{code}` has never returned that. Since upstream commit `989458e` (2022-10-22) it has
returned station metadata plus `trains: ["42-3", …]`, which are IDs only, so the function always
returns `None`. The live PHL response has the same shape: its 51 IDs are 46 active, 3
predeparture and 2 completed trains, all of which are already in `/v3/trains`.

**Why it may have seemed to work.** Before 2026-04-18, upstream published predeparture trains
earlier than one hour ahead (exactly how much earlier isn't recorded), so the second train usually
came back from `/v3/trains` itself. The connection view then worked through the normal path, not
through the fallback. That is a plausible explanation, not a verified one.

**Newer endpoint, same limitation.** `/v3/stations/expanded/{code}` (added 2026-04-19, upstream
`bc9d189`) includes per-train times at the station (`thisStationMin`), but only for trains already
in the feed, so it doesn't close the gap either.

**Options.** (a) Delete the fallback (about 150 lines of `main()` plus three helpers). Replace it
with a message saying when the train will appear, e.g. "Train 171 will appear about an hour
before it leaves Boston". (b) Close the gap with a static timetable, such as Amtrak's GTFS feed;
that is a new feature for after the refactor.

## Fetching and caching

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B03 | High | A 200 response that is not JSON (a captive portal or a proxy error page) raises `JSONDecodeError` straight out of `fetch_train_data`, which kills the live display. | `test_b03_…`, golden `single-non-json-body` |
| B04 | Low | A non-empty JSON list response (`[{…}]`) crashes with `AttributeError: 'list' object has no attribute 'keys'`. | `test_b04_…` |
| B27 | High | Several runs of a long-distance train can be on the road at once, and the API lists them **oldest first**. The code takes element `[0]`, while its comment says "return the most recent one". In the live snapshot, `/v3/trains/5` returned runs 5-30, 5-1 and 5-2, so `amtrak-status 5` showed the train that left Chicago on Sept 30 and was 16 hours late (`+970m`). Nine train numbers had more than one run. `42-26` syntax selects a run explicitly. **Decide:** which run to show by default (the most recent active one? ask?). | `test_b27_…`, golden `live-5-three-runs` |
| B32 | Low | The retry comment says "exponential backoff", but the code sleeps 2 s, then 4 s (linear). Worst case the UI freezes for about 36 s per train (3 × 10 s timeouts plus sleeps), and in two-train mode the trains are fetched one after the other. | — |

## Display

| ID | Sev | Bug | Test |
|----|-----|-----|------|
| B09 | Med | `--once` prints a `Layout`, which always renders at exactly the terminal height. Long routes are cut off, and so is any output piped somewhere (Rich assumes 25 lines). | `test_b09_…` |
| B10 | High | `statusMsg` is a single space for **every** train in the live feed (all 180, Amtrak included), and `trainTimely` is always empty. `" "` is truthy, so the header status is blank instead of "Active", and the compact line ends in a stray `\|   \|`. The status colouring ("green for on-time, red for late" in the README) therefore never appears with real data. The feed carries no on-time/late text at all, so a fix has to derive the status from the computed delay. The legacy test `test_space_statusmsg_used_in_header` asserts the buggy behaviour. | `test_b10_…` |
| B11 | Low | Route, station and train names are interpolated into Rich markup strings. A `[/]` in API data raises `MarkupError`, and a `[bold]` is silently applied as a style. | `test_b11_…` |
| B16 | Low | Focus mode hides "all but the last two departed" stops by slicing off the first *N* rows. Cancelled stops among the departed ones shift the window, so 4 departed rows show instead of 2. | `test_b16_…` |
| B18 | Med | The compact one-liner doesn't skip cancelled stops when looking for the next stop (the full header does). It shows `@ —` instead of the real ETA. | `test_b18_…` |
| B19 | Decide | The delay threshold differs between views: the header shows `(+1m)`, but the compact line only shows delays greater than 1 minute. | — |
| B22 | Low | An unknown `--from`/`--to` code is silently ignored, yet the table title still claims `(ZZZ → end)`. | `test_b22_…` |
| B24 | Low | The Status column (width 14) is too narrow for `Enroute (Plt 3)`, which wraps onto two lines. Live platforms are short (`1`–`9`), so every platform shown on an `Enroute` stop wraps. | `test_b24_…` |
| B25 | Med | The API marks **every** stop not yet reached as `Enroute` (never `""`): live, all 128 active trains whose next stop is `Enroute` have only `Enroute` stops after it. The table therefore styles every future stop as the current one (bold yellow `→`), and `○ Scheduled` never appears with real data. Most hand-written fixtures use `""` for future stops. | `test_b25_…` |
| B31 | Low | Two-train compact mode prints "Compact mode with connections – showing basic info", then immediately clears the screen. | — |
| B33 | Med | Trains keep appearing with `trainState: Completed` after they arrive, and their final stop stays `Enroute` (all 5 completed trains in the snapshot). The code ignores `Completed`, so a finished train is shown as still arriving: Keystone 660 shows "Next: New York Penn … (arriving)" and 94 % progress after it arrived. | `test_b33_…`, golden `live-660-completed` |
| B35 | Low | Brightline trains (and some VIA stops) send UTC times (`2026-10-03T11:42:00.000Z`) even though the station's `tz` is `America/New_York`. Times are displayed in whatever offset the string carries, so these show 4 hours off. Brightline trains also keep every stop `Enroute`, so progress stays at 0 %. The tool is for Amtrak, so the fix may just be to say these trains are unsupported. | `test_b35_…`, golden `live-b5712-brightline` |

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
| B28 | Test | `_now()` exists "for test patching", but `calculate_position_between_stations` only uses it for naive datetimes. Real API times always carry an offset (or `Z`), so for real data it calls `datetime.now(timezone.utc)` directly. Tests that patch `_now()` therefore only control the clock for the offset-less ISO strings the legacy builders produce. |
| B30 | Fixed | `parse_time` used to return naive datetimes for epoch-ms input and aware ones for ISO strings. #2 made it ISO-only and removed the xfail. Mixing is still possible in principle, because offset-less ISO strings parse to naive datetimes, but the live feed never sends them. |
| B34 | Test | `tests/legacy/test_amtrak_status.py::TestMultiTrainArgParsing::test_two_train_numbers_triggers_multi_mode` doesn't mock `fetch_station_schedule` or `sleep`. On every run it makes a real request to `api-v3.amtraker.com/v3/stations/PHL` and sleeps for 2 s. It passes whether or not the request succeeds. `test_notify_at_arg`, `test_notify_all_arg` and `test_multi_train_connection_arg` each really sleep for 1 s. |

## Confirmed working

- **Cancelled-stop heuristic.** `is_station_cancelled()` treats a `Station` stop with no `arr`/`dep` as cancelled. That is what upstream emits when Amtrak has no data for a stop. Live, Pacific Surfliner 757 had its last four stops in that state, consistent with a run truncated at Goleta, and the table shows them as cancelled (golden `live-757-no-data-stops`).

## Dead code (delete during the refactor)

- `build_predeparture_panel()`: only called from tests.
- `_train_caches[n]["error"]`: written in three places, never read.
- `calculate_progress()` returns `current_idx`, but both callers ignore it. `build_stations_table` calls `find_current_station_index()` and never uses the result.
- `select_connection_station()`: the "Invalid selection" retry loop and the `None` return are unreachable, because `Prompt.ask(choices=…)` already validates. The caller's `if not CONNECTION_STATION: sys.exit(1)` is unreachable for the same reason.
- `getattr(args, 'all', False)`: `args.all` always exists.
- `--all` and `--no-focus` do the same thing through two different mechanisms (`FOCUS_CURRENT` and `show_all`).
- The station-schedule fallback (B23) and `build_predeparture_train_data()`: no API response can reach them.
