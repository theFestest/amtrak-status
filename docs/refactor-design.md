# Refactor design: from one 2,056-line module to a maintainable package

Status: proposal · October 2026 · Companion documents: [known-bugs.md](known-bugs.md), and the
contract test harness in [`tests/contract/`](../tests/contract/).

## TL;DR

* `amtrak_status/tracker.py` does everything: HTTP, caching, domain rules, notifications, Rich
  rendering, argument parsing, interactive setup and the refresh loops. It shares 14 mutable
  module globals between those concerns. Six of the ~30 bugs found while preparing this design
  ([known-bugs.md](known-bugs.md): B01, B02, B07, B08, B18, B19) come directly from that shared
  state or from logic duplicated between views.
* The existing suite has 89 % branch coverage but a mutation score of only ~50 %: about half of
  all deliberate small code changes go undetected. A quarter of the tests patch
  `amtrak_status.tracker.*` internals, so they would break on *any* module split even when
  behaviour is preserved, which makes them unusable as a migration safety net.
* **Phase 0 (done in this PR)** adds that safety net. Contract tests drive the real CLI through
  seams that survive refactoring (console-script entry point, httpx transport, wall clock, sleep,
  stdin/stdout, `subprocess.run`). They pin every observable effect in 54 golden files that
  include colours, plus 19 behavioural tests and 22 strict-xfail tests for known bugs. Nothing
  under `tests/contract/` imports the package.
* The target is a small package with a typed domain model, no module globals, pure render
  functions and injected I/O (HTTP client, clock, sleep, notifier, prompt). The migration has
  seven behaviour-preserving phases (zero golden diffs allowed), then one PR per bug fix.

---

## 1. Where things stand

### 1.1 Code

| Concern | Where it lives today | Problems |
|---|---|---|
| Config | 8 module globals assigned in `main()` (`COMPACT_MODE`, `STATION_FROM`, `NOTIFY_ALL`, …) | read deep inside render functions; tests must reset 14 globals before every test |
| State | `_train_caches`, `_last_successful_data`, `_last_fetch_time`, `_last_error`, `_notified_stations`, `_notifications_initialized` | shared between the two trains in connection mode → B01, B02, B07, B08 |
| API data | raw `dict`s with `.get()`, magic strings (`"Departed"`, `""`), sentinel dicts `{"error": …}` mixed with train dicts, a synthetic `_predeparture` flag | every consumer re-interprets the data; the error sentinel can be mistaken for a train |
| Domain logic | spread across render functions | "next stop + ETA + delay" is implemented 3 times and the copies have drifted (B18, B19); station lookup by code is a hand-written loop 8 times; the panel subtitle is built twice |
| Rendering | `build_*` functions read config globals; the two top-level screen builders (`build_display`, `build_multi_train_display`) also **fetch** | full screens can't be rendered without mocking the network; API strings are interpolated into Rich markup (B11) |
| CLI / orchestration | `main()`: 380 lines, 4 nested levels of connection-setup branches | the station-schedule fallback is copy-pasted 4 times and can never succeed against the real API (B23) |
| Time | `parse_time` returns naive datetimes for epoch-ms input and aware ones for ISO input; `_now()` only covers the naive path | the real API only sends ISO strings, so the epoch-ms path (which most tests exercise) is dead in production, and real-data paths can't be clock-controlled (B28) |

### 1.2 What the live API actually sends

Confirmed by reading the server source
([`amtraker-v3/index.ts`](https://github.com/piemadd/amtraker-v3/blob/74789d05790d6b2f34b33044b6c9b5c3bba7041e/index.ts));
the API host was blocked from this environment, so no live capture was possible. The domain model
must be built around these facts, not around the current fixtures:

* Times are ISO-8601 strings with a fixed offset, e.g. `2026-02-08T07:25:00-05:00`. Epoch
  milliseconds are never sent.
* Stop `status` is `Departed`, `Station`, `Enroute` or (rarely) missing / `Unknown`. **Every stop
  not yet reached is `Enroute`**, not just the next one. `""` is never sent (B25).
* For stops not yet reached, `dep` is a copy of the arrival estimate (`dep ?? arr`), and
  `schArr`/`schDep` back-fill each other at the origin and terminus (B26).
* A predeparture train's origin has status `Station`. Trains appear in the feed only within about
  1 h of departure.
* Unknown train or station → `[]`. `/v3/trains/42-26` answers keyed by `"42"`.
* `/v3/stations/{code}` is metadata plus `trains: ["42-8", …]`. There are no times in it (B23).
* `statusMsg` is `" "` for VIA, Brightline and CPKC trains (B10). It can also be the literal string
  `"SERVICE DISRUPTION"`.

Two new fixtures follow these rules (`train_42_upstream_shape.json`,
`train_171_northeast_regional.json`); see [`tests/fixtures/README.md`](../tests/fixtures/README.md).

### 1.3 Tests

| Measure | Value |
|---|---|
| Tests (before this PR) | 361 passing + 3 xfail |
| Branch coverage of `tracker.py` | 89 % |
| Mutation score (mutmut 3.8; 4,310 of 4,478 mutants evaluated) | **48.6 %** (2,096 killed, 2,214 survived) |
| Tests that patch `amtrak_status.tracker.*` | 88 of 335 class-based tests (170 `patch(...)` calls) |
| Tests whose only assertions are `isinstance(...)` | 22 |
| Direct reads/writes of module globals in tests | 76 |

Why the score is low even though coverage is high:

* **No style assertions.** `render_to_text` exports plain text, so every colour and emphasis
  decision is untested. That includes the README's headline "green for on-time, red for late".
* **Unrealistic inputs.** The synthetic builders use epoch-ms timestamps and `status=""` for
  future stops. The live API sends neither, so the code paths that real data takes are only
  lightly exercised.
* **Weak or misleading assertions.** For example, `test_midjourney_build_display` checks
  `"New York Penn" in text`, which the *header* satisfies even though the stations table is
  clipped before the NYP row (B09). `test_departed_before_init_not_notified` checks the
  notification *title* for a station name that titles never contain.
  `test_eta_exact_on_schedule_no_diff` asserts `"+" not in text or "+0m" not in text`.
  `test_space_statusmsg_used_in_header` asserts the buggy behaviour (B10). The `coincidence`
  tests assert facts about Python (`assert not []`), not about the code.
* **Duplication.** Several scenarios are tested 2–3 times across files (predeparture panels,
  layover rendering, notification init).

Mutation score by area (existing suite, first pass):

| Area | Functions | Score |
|---|---|---|
| Small pure helpers | `filter_stations` 97 %, `get_status_style` 91 %, `format_time` 89 %, `get_station_times` 87 %, `is_station_cancelled` 86 %, `parse_time` 83 % | high |
| Domain calculations | `calculate_layover` 71 %, `calculate_position_between_stations` 71 %, `check_and_notify` 71 %, `calculate_progress` 61 % | medium |
| Rendering | `build_stations_table` 33 %, `build_compact_train_header` 38 %, `build_multi_train_display` 46 %, `build_progress_bar` 47 %, `build_header` 55 % | low |
| I/O and orchestration | `send_notification` 24 %, `select_connection_station` 30 %, `fetch_train_data_cached` 37 %, `fetch_train_data` 38 %, `main` 39 % | low |

Some survivors are equivalent mutants (e.g. changing the default of a `.get()` whose key is always
present), but many are real gaps in exactly the code a refactor moves. Examples:

* the network-error branch can retry 2 times instead of 3 (`attempt < MAX_RETRIES - 2`);
* the cached-data fallback after an HTTP 5xx can read a wrong key (`cache["XXdataXX"]`);
* the cache age limit can change from `< 300` to `<= 300`;
* `"Station"` → `"station"` in `check_and_notify` turns every "Arriving" notification into "Departed";
* `if train1_arrives and train2_departs` → `or` crashes on a connection with one unknown time;
* any colour in the stations table can be changed or removed.

## 2. Goals and non-goals

Goals:

1. Each concern lives in its own module with a narrow interface, and dependencies point one way.
2. No module-level mutable state. Configuration is an immutable value, and runtime state is owned
   by objects created per run (and per train where it is per-train).
3. API JSON is parsed once, at the boundary, into typed values. Nothing downstream calls `.get()`
   on API dicts.
4. Rendering is a pure function of a view model, and all I/O is injected. That makes every layer
   testable without patching.
5. The refactor itself changes no behaviour, as proven by the contract tests. Bug fixes come
   afterwards as small, separately reviewable changes.

Non-goals: new features, a different UI library, async I/O, supporting other APIs.

## 3. Target architecture

```
amtrak_status/
├── __init__.py        __version__ from importlib.metadata (fixes B12); re-exports main
├── __main__.py        python -m amtrak_status
├── cli.py             argparse → Config; validation (B05, B17); main(argv=None) -> int
├── config.py          @dataclass(frozen=True) Config
├── models.py          Train, Stop, StopStatus; from_api() parsing; parse_time (aware datetimes only)
├── journey.py         pure: next_stop, progress, position_between_stops(now), delay, visible_rows(filter+focus)
├── connection.py      pure: shared_stops, Layover + LayoverStatus classification
├── api.py             AmtrakerClient(http: httpx.Client, sleep): get_train() -> Found | NotFound | Failed
├── feed.py            TrainFeed (one per train): cache, staleness, last success time, warning
├── notify.py          ArrivalWatcher (one per train) → events; Notifier protocol + platform backends
├── setup.py           resolve the connection station (auto-detect / prompt), with an injected ask()
├── app.py             Tracker.tick() → Screen; run_once / run_compact / run_live with injected console, clock, sleep
└── render/
    ├── theme.py       every style, icon and label (status → icon/style lives here only)
    ├── views.py       view models built from Train + now (one place for "next stop / ETA / delay")
    ├── single.py      header, progress bar, stations table
    ├── compact.py     one-line status
    ├── connection.py  connection panel, train summaries, two-train layout
    └── messages.py    error / not found / awaiting-departure panels, status bar
```

Dependency direction: `cli → app → {setup, feed, notify, render} → {journey, connection} → models`.
`api` is used only by `feed`, and nothing under `render` imports `api`, `feed` or `app`.
`models` imports nothing from the package.

### 3.1 Key types (sketch)

```python
class StopStatus(Enum):
    DEPARTED = "Departed"
    AT_STATION = "Station"
    ENROUTE = "Enroute"
    UNKNOWN = "Unknown"          # upstream enum value, or field missing

@dataclass(frozen=True)
class Stop:
    code: str                    # always upper-case
    name: str
    status: StopStatus
    scheduled_arrival: datetime | None      # all datetimes are timezone-aware
    scheduled_departure: datetime | None
    arrival: datetime | None                # actual once passed, otherwise an estimate
    departure: datetime | None
    platform: str
    cancelled: bool                         # today's is_station_cancelled heuristic, computed once

@dataclass(frozen=True)
class Train:
    number: str
    train_id: str
    route: str
    state: str                  # "Active" | "Predeparture" | other upstream values
    status_message: str         # stripped; "" means none (fixes B10 once render falls back)
    heading: str
    speed_mph: float | None
    destination: str
    alerts: tuple[str, ...]
    stops: tuple[Stop, ...]
    def stop(self, code: str) -> Stop | None: ...

FetchResult = Found | NotFound | Failed     # Found(train), Failed(reason) — no {"error": ...} dicts

class Clock(Protocol):
    def now(self) -> datetime: ...          # aware; SystemClock returns datetime.now().astimezone()

class Notifier(Protocol):
    def notify(self, title: str, message: str) -> bool: ...
```

### 3.2 State ownership

| Today (global) | Target owner |
|---|---|
| `REFRESH_INTERVAL`, `COMPACT_MODE`, `STATION_FROM/TO`, `FOCUS_CURRENT`, `NOTIFY_*`, `CONNECTION_STATION` | `Config` (frozen), passed explicitly |
| `_train_caches`, `_last_successful_data`, `_last_fetch_time`, `_last_error` | one `TrainFeed` per train: `snapshot()` → `(train, fetched_at, warning)` |
| `_notified_stations`, `_notifications_initialized` | one `ArrivalWatcher` per train |

`fetch_train_data` and `fetch_train_data_cached` currently implement two overlapping caches with
different expiry rules (B07). `TrainFeed` is the single cache.

### 3.3 Rendering

* View models (`render/views.py`) compute everything a panel shows: next stop, ETA, delay
  minutes, position fraction, visible rows with elision counts, and status-bar fields. They do
  it once, from a `Train` and `now`. The three drifting copies of next-stop/ETA logic collapse
  into one. Fixing B18/B19 then becomes a one-line change.
* Panels are built with `Text.assemble` / `Text.append` instead of f-string markup, so API
  strings are never parsed as markup (B11). This changes nothing visible, and the goldens prove it.
* The status bar (`Updated … | ⚠ … | Refresh: 30s | Press Ctrl+C to quit`) is one function taking
  a `StatusLine` value, replacing `build_header`'s copy and `_apply_main_title`.
* All styles, icons and labels live in `theme.py`.

### 3.4 Application loop

`Tracker.tick()` refreshes the feeds, builds view models, renders, and runs the watchers.
`run_once`, `run_compact` and `run_live` differ only in how they put a `Screen` on the terminal.
They receive `console`, `clock`, `sleep` and `notifier`, so the loop logic is testable directly,
and the contract tests keep covering the real wiring. `main(argv=None) -> int` returns an exit
code, and the console script does `sys.exit(main())` (the harness already supports this).

### 3.5 Where each current function goes

| `tracker.py` today | Target |
|---|---|
| `_now` | `Clock` / `SystemClock` (app wiring); domain functions take `now` as a parameter |
| `parse_time` | `models.parse_time` (ISO → aware; the epoch-ms branch can go once the legacy tests are migrated) |
| `format_time` | `render/theme.py` (or `render/format.py`) |
| `is_station_cancelled` | `models` (computed into `Stop.cancelled` at parse time) |
| `find_station_index`, `get_station_times`, `get_station_status` | `Train.stop(code)` / `Train.index_of(code)` |
| `find_current_station_index`, `calculate_progress`, `calculate_position_between_stations` | `journey` |
| `filter_stations` + focus logic inside `build_stations_table` | `journey.visible_rows()` (fixes the slicing in B16 later) |
| `find_overlapping_stations`, `calculate_layover` | `connection` (returns a `Layover` with a `LayoverStatus` enum) |
| `get_status_style`, status colours inside table and connection panel | `render/theme.py` |
| `build_header`, `build_progress_bar`, `build_stations_table` | `render/single.py` |
| `build_compact_display` | `render/compact.py` |
| `build_connection_panel`, `build_compact_train_header`, `build_predeparture_header`, the panel assembly inside `build_multi_train_display` | `render/connection.py` |
| `build_error_panel`, `build_not_found_panel`, `_apply_main_title` | `render/messages.py` |
| `build_predeparture_panel` | delete (unused) |
| `fetch_train_data` | `api.AmtrakerClient.get_train` (HTTP, retries, JSON validation) |
| `fetch_train_data_cached` + cache parts of `fetch_train_data` | `feed.TrainFeed` |
| `initialize_notification_state`, `check_and_notify` | `notify.ArrivalWatcher` |
| `send_notification` | `notify.DesktopNotifier` (per-platform backends that pass text as arguments: fixes B20 later) |
| `select_connection_station`, connection branches of `main` | `setup.resolve_connection(...)` with an injected `ask` |
| `fetch_station_schedule`, `get_train_schedule_from_station`, `build_predeparture_train_data` | delete, or replace (decision on B23) |
| `build_display`, `build_multi_train_display`, loops in `main` | `app` |
| argument parsing in `main` | `cli` + `config` |

## 4. Phase 0: the safety net (this PR)

`tests/contract/_harness.py` runs the installed `amtrak-status` entry point in-process and
controls only these seams:

| Seam | Mechanism | Survives refactor because… |
|---|---|---|
| entry point | `importlib.metadata.entry_points()["amtrak-status"]` | moving `main` only changes `pyproject.toml` |
| network | `httpx.HTTPTransport.handle_request` → `FakeAmtrakerAPI` | any `httpx.Client` anywhere uses it |
| clock | `time-machine` (also freezes aware `datetime.now(tz)`), `TZ=America/New_York` | not tied to a helper like `_now` |
| sleep | `time.sleep` *and every alias of it found in `amtrak_status.*` modules* | works however `sleep` is imported |
| notifications | `subprocess.run` (same alias scan) and `sys.platform` | same |
| terminal | `COLUMNS`, `LINES`, `FORCE_COLOR`; stdin/stdout redirected; Rich's global console reset | — |
| module state | the package is re-imported fresh for every run, then the previous modules are restored | globals can't leak between runs or into `tests/legacy` |

What it pins:

* `test_cli_golden.py`: 54 `--once` scenarios (single train × flags × fixtures, API failures,
  connection mode with every setup branch including prompts). Each golden file records argv,
  exit code, HTTP requests, sleeps, notifications, and the screen **with styles**
  (`[green]✓[/] Pittsburgh`). Today's bugs are pinned too, so a refactor that accidentally
  "fixes" something is also caught.
* `test_cli_behavior.py`: 19 tests for the refresh loops (re-polling, intervals, retry backoff,
  cache fallback and expiry, notifications per platform, bell fallback, `--help`).
* `test_known_bugs.py`: 22 strict-xfail tests, one per bug that the CLI can reach.

Rules for refactoring PRs:

1. `tests/contract/` must pass with **no golden diffs and no edits to `tests/contract/`**. The only
   allowed edit is the entry-point path, if `pyproject.toml` changes it.
2. The set of xfails must not change. With `xfail_strict`, an accidental fix shows up as a failure.
3. Legacy tests that break because code moved are migrated in the same PR (see §6), not patched
   to follow the new layout.

To intentionally change output: `UPDATE_GOLDEN=1 uv run pytest tests/contract`, then review
`git diff tests/contract/golden` as part of the PR.

## 5. Migration plan

Each phase is one PR (or a few small ones). Phases 1–7 must satisfy the rules in §4.

| Phase | Change | Notes |
|---|---|---|
| 1 | **Mechanical hygiene.** Adopt ruff format and ruff lint in CI; remove unused imports and trailing whitespace; delete the dead code listed in known-bugs.md. | Pure formatting first, so later diffs are readable. |
| 2 | **Models.** Add `models.py` (`Train`, `Stop`, `StopStatus`, `parse_time` → aware). Parse in `fetch_train_data` and pass models to the existing functions, converting each function body. Domain functions take `now` explicitly (fixes the B28 testability gap). | Largest step; can be split per function group. Port the matching legacy unit tests to `tests/unit/test_models.py` using ISO strings. |
| 3 | **Pure domain.** Move progress, position, next-stop, filter/focus and layover logic into `journey.py` / `connection.py`. Collapse the three next-stop/ETA copies into one function, keeping each view's current behaviour through parameters, e.g. `skip_cancelled=False` for the compact line (B18) and a `min_delay` threshold (B19). | Behaviour-preserving duplication removal, made safe by the goldens. |
| 4 | **API + feed.** `AmtrakerClient` (injected `httpx.Client` and `sleep`) and `TrainFeed`. Keep today's cache semantics, including the B07 re-stamping quirk, behind a clearly named method. Remove the `{"error": …}` dicts. | Unit tests with `httpx.MockTransport`. |
| 5 | **Rendering.** Create `render/` with view models, theme and `StatusLine`; switch to `Text.assemble`. Render functions take view models only and never fetch. | The goldens are the main guard here. |
| 6 | **Notifications.** `ArrivalWatcher` + `Notifier`. Initially **share one watcher between both trains**, so B01/B02 are preserved; fixing them is then a one-line change in phase 8. | |
| 7 | **CLI, setup, app.** `Config`, `cli.main(argv) -> int`, `setup.resolve_connection` (merging the four copy-pasted branches), and `app` loops. Point the entry point at `amtrak_status.cli:main`, delete `tracker.py`, delete `tests/legacy/` (everything migrated or superseded). | Optionally keep `amtrak_status/tracker.py` as a one-line re-export for a release. |
| 8 | **Bug fixes**, one PR each: flip the xfail, update goldens, add a CHANGELOG entry. Suggested order: B03/B04 (crashes) → B01/B02 (notifications) → B07/B08 (caching) → B25/B26 (real-API semantics) → B18 → B09 → B20 → the rest; decisions (§7) as they are made. | |
| 9 | **Tooling follow-ups.** Type-check `amtrak_status/` (pyright or mypy, strict on the new modules); coverage report in CI; run mutmut on `journey`/`connection`/`render/views` occasionally; relax the `rich<14` pin, letting the goldens show what Rich 14 changes. | |

## 6. Test strategy after the refactor

| Layer | Location | What | Style |
|---|---|---|---|
| Contract | `tests/contract/` | the CLI end to end (unchanged by the refactor) | goldens + behaviour + known bugs |
| Unit: domain | `tests/unit/test_journey.py`, `test_connection.py`, `test_models.py` | pure functions on typed inputs | table-driven `parametrize`; explicit boundaries (layover at 29/30/59/60 min; focus at 10/11 stops); invariants (0 ≤ progress ≤ 1, completed ≤ total) |
| Unit: I/O | `test_api.py`, `test_feed.py`, `test_notify.py` | retries, JSON validation, cache expiry, per-train watcher, per-platform commands | `httpx.MockTransport`; fake clock / sleep / notifier objects, no `patch()` |
| Unit: views | `test_views.py` | view-model fields (next stop, delay, rows, elisions) | assert on data, not rendered text; leave rendering details to the goldens |

Triage of the legacy suite during migration:

| Legacy tests | Disposition |
|---|---|
| Pure-function classes (`TestParseTime`, `TestFormatTime`, `TestIsStationCancelled`, `TestFilterStations`, `TestCalculateProgress`, `TestCalculatePositionBetweenStations`, `TestFindOverlappingStations`, `TestCalculateLayover`, `TestGetStatusStyle`, …) | port to `tests/unit/` with typed inputs and ISO times |
| Fetch/cache (`TestFetchTrainData`, `TestCacheExpiry`, `TestFetchTrainDataCachedErrorPath`, `TestFixtureTrainParsing`) | port to `test_api.py`/`test_feed.py` with `MockTransport` |
| Notification classes | port to `test_notify.py` |
| Rendered-text checks (`TestRendered*`, `TestJourneyPhase*`, `TestBuild*Panel`, `TestFixtureFullPipeline`, `TestBuildConnectionPanel`) | mostly superseded by goldens; keep the few that test a decision (e.g. elision counts) as view-model tests |
| `main()`/orchestration (`TestMainArgParsing`, `TestMainMultiTrainOrchestration`, `TestMainLiveRefreshLoop`, `TestMultiTrainArgParsing`) | superseded by contract tests; delete in phase 7 |
| `isinstance`-only tests and `@pytest.mark.coincidence` tests | delete |
| Existing xfails | B23/B29/B30, tracked in known-bugs.md; resolve with the decisions below |

## 7. Decisions needed

1. **B23, station-schedule fallback:** delete it (recommended; it cannot work with Amtraker v3), or
   add a timetable source such as Amtrak's GTFS feed (a new feature, so after the refactor).
2. **B29, layover bands:** keep 3 bands (risky < 30 ≤ tight < 60 ≤ comfortable) and drop
   `LAYOVER_TIGHT`, or add a fourth band at 45.
3. **B25/B26, presenting real API data:** only highlight the next stop, and show later stops as
   "Scheduled"/"Expected"? Treat a future stop's `dep == arr` as "no departure estimate" and use
   `max(estimate, scheduled departure)` for layovers?
4. **B19:** one delay threshold everywhere (suggested: show any non-zero delay, as the header does).
5. **Python floor:** Python 3.10 reaches end of life this month (October 2026). Moving to `>=3.11`
   simplifies typing (`Self`, `StrEnum`) and lets tests use `tomllib`.
6. **Import compatibility:** keep `amtrak_status.tracker` as a deprecated re-export for one
   release, or drop it. It is a CLI, so dropping it is probably fine.
