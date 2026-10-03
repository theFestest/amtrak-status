# amtrak-status

[![PyPI](https://img.shields.io/pypi/v/amtrak-status.svg)](https://pypi.org/project/amtrak-status/)
[![Tests](https://github.com/theFestest/amtrak-status/actions/workflows/test.yml/badge.svg)](https://github.com/theFestest/amtrak-status/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/theFestest/amtrak-status?include_prereleases&label=changelog)](https://github.com/theFestest/amtrak-status/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/theFestest/amtrak-status/blob/main/LICENSE)

A clean terminal user interface (TUI) for tracking Amtrak train status in real-time. No more manual page refreshing!

## Features

- **Auto-refresh**: Automatically updates every 30 seconds (configurable)
- **Visual progress bar**: See journey completion at a glance
- **Station-by-station breakdown**: Scheduled, estimated, and actual times
- **Color-coded status**: Green for early/on-time, red for late, yellow for en-route
- **Clean TUI**: Built with Rich for a beautiful terminal experience

## Installation

### Using uv (recommended)

```bash
# Run directly without installing
uv run --with amtrak-status amtrak-status 42

# Or install as a project
uv sync
uv run amtrak-status 42
```

### Using pip

```bash
pip install amtrak-status
amtrak-status 42
```

## Usage

```bash
# Track the Pennsylvanian #42
amtrak-status 42

# Track the California Zephyr #5
amtrak-status 5

# Track with custom refresh interval (60 seconds)
amtrak-status 42 -r 60

# Display once and exit (no auto-refresh)
amtrak-status 42 --once

# Track a specific day's train (train 42 from the 26th)
amtrak-status 42-26
```

## Options

| Option | Description |
|--------|-------------|
| `train_number` | Amtrak train number (e.g., 42, 5, 91) |
| `-r, --refresh` | Refresh interval in seconds (default: 30) |
| `--once` | Display once and exit without auto-refresh |

## Common Train Numbers

| Number | Route |
|--------|-------|
| 42 | Pennsylvanian (Pittsburgh → NYC) |
| 43 | Pennsylvanian (NYC → Pittsburgh) |
| 5 | California Zephyr (Chicago → Emeryville) |
| 6 | California Zephyr (Emeryville → Chicago) |
| 91 | Silver Star (NYC → Miami) |
| 92 | Silver Star (Miami → NYC) |
| 79/80 | Carolinian |
| 19/20 | Crescent |

## How It Works

This tool uses the [Amtraker API](https://api-v3.amtraker.com), a community-built API that provides real-time Amtrak train tracking data. The API pulls from Amtrak's official Track Your Train system.

## Troubleshooting

**"Train Not Found"**
- The train may not have started its journey yet today
- Double-check the train number on [Amtrak's website](https://www.amtrak.com)
- The train may have already completed its journey

**No updates showing**
- The Amtraker API updates when Amtrak's system updates (typically every few minutes)
- Check your internet connection

## Development

This project uses [uv](https://docs.astral.sh/uv/). To set up a development environment with the test dependencies and run the tests:
```bash
cd amtrak-status
uv sync
uv run pytest
```

The tests are split in two:

- `tests/contract/` drive the real `amtrak-status` command against a fake API and a frozen clock, and compare everything it does (screen output with colours, HTTP requests, sleeps, notifications) with the golden files in `tests/contract/golden/`. They never import the package's internals, so they keep passing, unchanged, while the code is reorganised. If you change the output on purpose, regenerate the golden files and review the diff:
  ```bash
  UPDATE_GOLDEN=1 uv run pytest tests/contract
  git diff tests/contract/golden
  ```
- `tests/legacy/` are white-box tests of the current `tracker.py` functions. They will be migrated alongside the refactor described in [docs/refactor-design.md](docs/refactor-design.md).

Known bugs are listed in [docs/known-bugs.md](docs/known-bugs.md). Most have a strict `xfail` test in `tests/contract/test_known_bugs.py`, which will fail the run (by passing unexpectedly) once the bug is fixed.

## License

Apache-2.0

## Credits

- [Amtraker API](https://github.com/piemadd/amtrak) by piemadd
- [Rich](https://github.com/Textualize/rich) for the beautiful TUI components
