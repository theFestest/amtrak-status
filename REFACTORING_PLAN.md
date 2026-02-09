# Refactoring Plan: amtrak-status

## Status: Complete (Phase 6 cleanup applied)

All 5 original phases + Phase 6 cleanup executed successfully. 361 tests passing (1 xfailed).

---

## Original Problems (all addressed)

1. **Monolithic file** — `tracker.py` was 2,043 lines handling every concern. Now 778 lines (62% reduction), with logic distributed across 13 focused modules.

2. **Global state everywhere** — 18 scattered globals replaced with a single `_config = Config()` dataclass instance + 2 encapsulated state objects (`TrainCache`, `NotificationState`). Wrapper functions in tracker.py inject state into pure display/API functions.

3. **Duplicated display logic** — `build_header()` and `build_compact_train_header()` now share three private helpers: `_find_next_station_info()`, `_format_status()`, `_build_position_bar()`.

4. **382-line `main()` function** — Decomposed into 6 focused functions: `_build_arg_parser()`, `_apply_args_to_config()`, `_fetch_predeparture_schedule()`, `_setup_connection()`, `_run_single_train()`, `_run_multi_train()`, plus a slim `main()` orchestrator (~25 lines).

5. **Layover bug fixed** — `calculate_layover()` now uses `>=` thresholds: `>= 60` → comfortable, `>= 45` → tight, `< 45` → risky. Previously 45-60 minutes was incorrectly categorized as "tight" (same as 30-45).

---

## Final Module Structure

```
amtrak_status/
├── __init__.py              # Package exports (main, __version__)
├── config.py                # Constants + Config dataclass (27 lines)
├── models.py                # Pure time/station utilities (215 lines)
├── connection.py            # Connection logic + layover calculation (130 lines)
├── api.py                   # API fetching, caching, TrainCache class (208 lines)
├── notifications.py         # NotificationState class + send/check logic (141 lines)
├── tracker.py               # Orchestration, CLI, display loops (778 lines)
├── display/
│   ├── __init__.py          # Re-exports for convenience (23 lines)
│   ├── header.py            # Deduplicated header builders (241 lines)
│   ├── stations.py          # Station table display (166 lines)
│   ├── progress.py          # Progress bar (36 lines)
│   ├── compact.py           # Single-line compact display (82 lines)
│   ├── connection_display.py # Connection panel (115 lines)
│   ├── errors.py            # Error/not-found panels (30 lines)
│   └── predeparture.py      # Predeparture panels (42 lines)
```

Total: ~2,237 lines across 15 files (vs 2,043 in one file — modest growth from module boilerplate and wrapper functions).

---

## Deviations from Original Plan

1. **No separate `main.py`** — The restructured `main()` and its helpers remain in `tracker.py`. This avoids changing the entry point in `pyproject.toml` and `__init__.py`, preserving full backward compatibility. The main() decomposition still achieved its goal (6 focused functions instead of one 382-line function).

2. **`connection.py` split from `models.py`** — Connection logic (`find_overlapping_stations`, `get_station_times`, `get_station_status`, `calculate_layover`) was placed in its own module rather than in `models.py`, since it has distinct responsibilities (multi-train connection analysis vs general station utilities).

3. **`display/connection_display.py` instead of `display/connection.py`** — Renamed to avoid confusion with the top-level `connection.py` module.

4. **Wrapper functions in `tracker.py`** — Instead of updating all test mock targets to point at the new modules, we kept backward-compatible wrapper functions in `tracker.py` that pass module-level state. Tests that patch `tracker.function_name` continue to work. Mock targets for imports that moved completely (httpx, subprocess, sleep for retries) were updated to the new module paths.

5. **Config dataclass now used for runtime config** — The 6 scattered globals (`COMPACT_MODE`, `STATION_FROM`, `STATION_TO`, `FOCUS_CURRENT`, `CONNECTION_STATION`, `REFRESH_INTERVAL`) were consolidated into `_config = Config()`. Tests updated to use `tracker._config.compact_mode` etc.

---

## Phase Execution Log

### Phase 1: Extract pure functions ✓
- Created `config.py` (constants + Config dataclass)
- Created `models.py` (10 pure utility functions)
- Created `connection.py` (4 connection functions + layover bug fix)
- Updated test imports, fixed 2 layover tests
- 361 tests passing

### Phase 2: Extract stateful subsystems ✓
- Created `api.py` with `TrainCache` class (5 API functions)
- Created `notifications.py` with `NotificationState` class (3 notification functions)
- Updated mock patch targets: `httpx.Client` → `api.httpx.Client`, `subprocess` → `notifications.subprocess`, API `sleep` → `api.sleep`, `send_notification` → `notifications.send_notification`
- 361 tests passing

### Phase 3: Split display functions ✓
- Created `display/` package with 7 submodules
- Deduplicated header logic into 3 shared private helpers
- Display functions now take explicit parameters instead of reading globals
- Wrapper functions in tracker.py pass state to display functions
- 361 tests passing

### Phase 4: Restructure main() ✓
- Decomposed 382-line `main()` into 6 focused functions + slim orchestrator
- 361 tests passing

### Phase 5: Final cleanup & verification ✓
- Test imports already updated during phases 1-4
- `reset_globals` fixture simplified (8 assignments: 6 config + 2 state objects)
- `freeze_time` fixture patches `_now` across 3 modules (models, tracker, api)
- Final test run: 361 passed, 1 xfailed

### Phase 6: Codebase cleanup ✓
- Consolidated 6 module-level globals into single `_config = Config()` dataclass instance
- Removed `_apply_args_to_globals()` (with `global` keyword), replaced with `_apply_args_to_config()`
- Removed pure re-export wrapper `build_compact_train_header()` — now a direct module alias
- Made `_apply_main_title()` private (only used internally by `build_multi_train_display`)
- Fixed duplicate position bar rendering in `header.py` — `build_compact_train_header()` was calling `_build_position_bar()` then reimplementing it; now uses `compact=True` param
- Extracted `_build_partial_connection_panel()` helper, deduplicating ~40 lines across the train1-only and train2-only branches
- Extracted `_find_station_name()` helper to replace repeated station name lookup loops
- Cleaned up unused imports (API_BASE, RETRY_DELAY, layover constants, datetime)
- Updated conftest.py `reset_globals` fixture (3 assignments vs 8)
- Updated all test references from `tracker.COMPACT_MODE` → `tracker._config.compact_mode` etc.
- tracker.py reduced from 778 to 765 lines
- 361 tests passing

---

## What Did NOT Change

- **Public API**: The `amtrak-status` CLI command and all flags remain identical
- **Entry point**: `pyproject.toml` still points to `amtrak_status.tracker:main`
- **Dependencies**: No new external packages
- **Python version support**: Still 3.10+
- **Test count**: 361 passing (up from 360 — the layover bug fix converted 1 xfail to pass)
