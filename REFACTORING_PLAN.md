# Refactoring Plan: amtrak-status

## Current State

All application logic lives in a single 2,043-line file (`amtrak_status/tracker.py`) with 18 module-level global variables. The tests (360 passing, ~4,300 lines across 3 files) are comprehensive and provide a safety net for refactoring.

### Problems

1. **Monolithic file** — `tracker.py` handles CLI parsing, API calls, caching, notifications, business logic, and all display rendering in one file. Finding and understanding any single concern requires scrolling through 2,000+ lines.

2. **Global state everywhere** — 18 module-level globals control configuration, caching, and notification state. Functions read and mutate these implicitly via `global` declarations, making data flow invisible and testing awkward (requires `reset_globals()` fixture).

3. **Duplicated display logic** — `build_header()` (lines 581-700) and `build_compact_train_header()` (lines 972-1093) repeat the same patterns for finding the next station, calculating ETA with delay coloring, determining status style, and rendering the position bar.

4. **382-line `main()` function** — Mixes argument parsing, global configuration, multi-train connection setup with interactive prompts, predeparture schedule fetching, and three separate display loop implementations (full TUI, compact, once-and-exit).

5. **Likely bug** — `calculate_layover()` lines 848-851: layovers between 45-60 minutes are categorized as "tight" twice (the 45-60 range should probably be "comfortable" or at least distinct from the 30-45 range).

---

## Proposed Module Structure

Break `tracker.py` into focused modules inside the existing `amtrak_status/` package:

```
amtrak_status/
├── __init__.py              # Package exports (main, __version__)
├── config.py                # Config dataclass + constants
├── api.py                   # API fetching, caching, retry logic
├── models.py                # Shared types/helpers (parse_time, format_time, station utils)
├── notifications.py         # Notification state + send logic
├── display/
│   ├── __init__.py          # Re-exports for convenience
│   ├── header.py            # build_header, build_compact_train_header (deduplicated)
│   ├── stations.py          # build_stations_table
│   ├── progress.py          # build_progress_bar
│   ├── compact.py           # build_compact_display
│   ├── connection.py        # build_connection_panel, build_multi_train_display
│   ├── errors.py            # build_error_panel, build_not_found_panel
│   └── predeparture.py      # build_predeparture_panel, build_predeparture_header
└── main.py                  # CLI parsing + display loop orchestration
```

---

## Detailed Breakdown

### 1. `config.py` — Configuration & Constants

**What moves here:**
- `API_BASE`, `MAX_RETRIES`, `RETRY_DELAY` (lines 43-46)
- `LAYOVER_COMFORTABLE`, `LAYOVER_TIGHT`, `LAYOVER_RISKY` (lines 55-57)
- Default `REFRESH_INTERVAL` (line 44)

**New additions:**
- A `Config` dataclass replacing the 8 configuration globals:
  ```python
  @dataclass
  class Config:
      compact_mode: bool = False
      station_from: str | None = None
      station_to: str | None = None
      focus_current: bool = True
      refresh_interval: int = 30
      notify_stations: set[str] = field(default_factory=set)
      notify_all: bool = False
      connection_station: str | None = None
  ```

**Why:** Replaces implicit global state with an explicit object that can be passed to functions. Tests construct a `Config()` instead of resetting 8 separate globals. Functions declare their dependencies clearly through parameters.

### 2. `models.py` — Time/Station Utilities

**What moves here:**
- `_now()` (line 38)
- `parse_time()`, `format_time()` (lines 481-496)
- `get_status_style()` (lines 499-510)
- `is_station_cancelled()` (lines 77-111)
- `find_station_index()`, `find_current_station_index()` (lines 513-532)
- `filter_stations()` (lines 535-557)
- `calculate_progress()` (lines 560-578)
- `calculate_position_between_stations()` (lines 114-181)
- Station time/status helpers: `get_station_times()`, `get_station_status()` (lines 755-777)
- `find_overlapping_stations()` (lines 737-752)
- `calculate_layover()` (lines 780-862)

**Why:** These are pure functions operating on dicts/datetimes with no side effects. They're the core business logic that everything else depends on. Grouping them makes them easy to find, test in isolation, and reuse.

### 3. `api.py` — API Communication & Caching

**What moves here:**
- `fetch_train_data()` (lines 315-404)
- `fetch_train_data_cached()` (lines 1096-1119)
- `fetch_station_schedule()` (lines 407-421)
- `get_train_schedule_from_station()` (lines 424-450)
- `build_predeparture_train_data()` (lines 453-478)

**Changes:**
- Replace the 5 cache/error globals with a `TrainCache` class:
  ```python
  class TrainCache:
      def __init__(self):
          self.last_successful_data: dict | None = None
          self.last_fetch_time: datetime | None = None
          self.last_error: str | None = None
          self.per_train: dict[str, dict] = {}
  ```
- Functions take `cache: TrainCache` as a parameter instead of using globals.

**Why:** Isolates all network I/O and caching into one module. The `TrainCache` class makes cache state explicit and testable without global resets.

### 4. `notifications.py` — Notification System

**What moves here:**
- `initialize_notification_state()` (lines 184-215)
- `send_notification()` (lines 218-265)
- `check_and_notify()` (lines 268-312)

**Changes:**
- Replace the 4 notification globals with a `NotificationState` class:
  ```python
  class NotificationState:
      def __init__(self, stations: set[str], notify_all: bool = False):
          self.stations = stations
          self.notify_all = notify_all
          self.notified: set[str] = set()
          self.initialized: bool = False
  ```

**Why:** Self-contained subsystem with its own state. Currently entangled with global variables that tests must manually reset. A class makes the lifecycle explicit.

### 5. `display/` — All Rich Display Builders

Split across files by logical grouping. Each display builder becomes a function that takes `train` data and a `Config` object (where needed) as parameters, instead of reading globals.

**Key deduplication opportunity:**

`build_header()` and `build_compact_train_header()` share ~60% identical logic for:
- Finding next station (skip cancelled, find first non-departed)
- Calculating ETA with delay coloring
- Determining status style from `statusMsg`
- Rendering position bar between stations

Extract shared logic into a private helper (e.g., `_resolve_next_station_info()`) that returns a structured result, then have both functions use it.

### 6. `main.py` — CLI Entry Point

**What moves here:**
- `main()` (lines 1662-2043)

**Changes:**
Break the 382-line `main()` into focused functions:
- `parse_args() -> argparse.Namespace` — CLI argument definition and parsing
- `build_config(args) -> Config` — Convert parsed args to a `Config` object
- `setup_multi_train(config, console) -> str` — Multi-train connection detection and interactive prompts
- `run_single_train_loop(config, console)` — Single-train display loop
- `run_multi_train_loop(config, console)` — Multi-train display loop
- `main()` — Thin orchestrator: parse, configure, dispatch

---

## Migration Strategy

Refactor incrementally, keeping all 360 tests passing at each step:

### Phase 1: Extract pure functions (lowest risk)
1. Create `config.py` with constants and `Config` dataclass
2. Create `models.py` with all pure utility functions
3. Update `tracker.py` to import from these modules
4. Verify tests pass

### Phase 2: Extract stateful subsystems
4. Create `api.py` with `TrainCache` class
5. Create `notifications.py` with `NotificationState` class
6. Update `tracker.py` to use the new classes
7. Verify tests pass

### Phase 3: Split display functions
8. Create `display/` package with individual modules
9. Deduplicate header logic
10. Update `tracker.py` to import display builders
11. Verify tests pass

### Phase 4: Restructure main
12. Create `main.py` with decomposed `main()` function
13. Update `__init__.py` and entry point
14. Verify tests pass

### Phase 5: Update tests
15. Update test imports to reflect new module structure
16. Simplify `reset_globals` fixture (now just constructing new `Config`/`Cache`/`NotificationState` objects)
17. Final full test run

---

## Bug to Fix During Refactoring

In `calculate_layover()` (current lines 846-853):
```python
if layover_minutes < LAYOVER_RISKY:       # < 30 → risky
    result["layover_status"] = "risky"
elif layover_minutes < LAYOVER_TIGHT:      # < 45 → tight
    result["layover_status"] = "tight"
elif layover_minutes < LAYOVER_COMFORTABLE: # < 60 → tight (BUG: should be different?)
    result["layover_status"] = "tight"
else:                                       # >= 60 → comfortable
    result["layover_status"] = "comfortable"
```

The 45-60 minute range is labeled "tight" — same as 30-45 minutes. This seems unintentional. Consider whether 45-60 should be a distinct status or merged with "comfortable." Check the corresponding test (`TestLayoverBoundary::test_exactly_60_min_layover`) for intended behavior.

---

## What This Does NOT Change

- **Public API**: The `amtrak-status` CLI command and all its flags remain identical
- **Test assertions**: All existing test assertions remain valid (we're restructuring, not changing behavior)
- **Dependencies**: No new external packages
- **Python version support**: Still 3.10+
- **Entry point**: `pyproject.toml` entry point will update from `amtrak_status.tracker:main` to `amtrak_status.main:main`
