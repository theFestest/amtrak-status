#!/usr/bin/env python3
"""
amtrak-status — Amtrak Train Status Tracker TUI

A terminal user interface for tracking Amtrak train status with auto-refresh.
Uses the Amtraker API (https://api-v3.amtraker.com).

Usage:
    amtrak-status <train_number>
    amtrak-status 42                      # Track the Pennsylvanian #42
    amtrak-status 42 178                  # Track two trains with connection
    amtrak-status 42 178 --connection PHL # Specify connection station
    amtrak-status 42 --from PGH --to NYP  # Filter to your segment
    amtrak-status 42 --compact            # Single-line for status bars
    amtrak-status 42 --all                # Show all stops (no auto-focus)
    amtrak-status 42 --notify-at NYP      # Notify on arrival at station
    amtrak-status 42 --notify-all         # Notify on all arrivals
"""

import argparse
import sys
from time import sleep
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from .config import Config, MAX_RETRIES
from .models import (
    _now, parse_time, format_time, get_status_style,
    is_station_cancelled, find_station_index, find_current_station_index,
    filter_stations, calculate_progress, calculate_position_between_stations,
)
from .connection import (
    find_overlapping_stations, get_station_times, get_station_status,
    calculate_layover,
)
from .api import (
    TrainCache,
    fetch_train_data as _api_fetch_train_data,
    fetch_train_data_cached as _api_fetch_train_data_cached,
    fetch_station_schedule,
    get_train_schedule_from_station,
    build_predeparture_train_data,
)
from .notifications import (
    NotificationState,
    send_notification,
    initialize_notification_state as _notify_initialize,
    check_and_notify as _notify_check_and_notify,
)
from .display import (
    build_header as _display_build_header,
    build_compact_train_header as _display_build_compact_train_header,
    apply_main_title as _display_apply_main_title,
    build_stations_table as _display_build_stations_table,
    build_progress_bar,
    build_compact_display as _display_build_compact_display,
    build_connection_panel,
    build_error_panel,
    build_not_found_panel,
    build_predeparture_panel,
    build_predeparture_header,
)

# Re-export: no state injection needed, just a direct alias for display function
build_compact_train_header = _display_build_compact_train_header

# Runtime state: a single Config object replaces the old scattered globals
_config = Config()

# Encapsulated state objects
_cache = TrainCache()
_notify_state = NotificationState()


# Wrapper functions that inject module-level state into pure functions
def fetch_train_data(train_number: str) -> dict[str, Any] | None:
    """Fetch train data from the Amtraker API with retry logic."""
    return _api_fetch_train_data(train_number, _cache)


def fetch_train_data_cached(train_number: str) -> dict[str, Any] | None:
    """Fetch train data with per-train caching for multi-train mode."""
    return _api_fetch_train_data_cached(train_number, _cache)


def initialize_notification_state(train: dict) -> None:
    """Capture the initial state of stations so we don't notify for pre-existing arrivals."""
    _notify_initialize(train, _notify_state)


def check_and_notify(train: dict) -> list[str]:
    """Check if train has arrived at any stations we should notify about."""
    return _notify_check_and_notify(train, _notify_state)


def build_header(train: dict) -> Panel:
    """Build the header panel with train info."""
    return _display_build_header(
        train,
        last_fetch_time=_cache.last_fetch_time,
        last_error=_cache.last_error,
        refresh_interval=_config.refresh_interval,
    )


def build_stations_table(train: dict, focus: bool = True) -> Panel:
    """Build the stations table with optional filtering and focus."""
    return _display_build_stations_table(
        train,
        focus=focus,
        station_from=_config.station_from,
        station_to=_config.station_to,
        focus_current=_config.focus_current,
    )


def build_compact_display(train: dict) -> Text:
    """Build a single-line compact display for the train status."""
    return _display_build_compact_display(train, last_fetch_time=_cache.last_fetch_time)


def _apply_main_title(panel: Panel) -> None:
    """Add the main 'Amtrak Status' title and status subtitle to a panel."""
    _display_apply_main_title(
        panel,
        last_fetch_time=_cache.last_fetch_time,
        last_error=_cache.last_error,
        refresh_interval=_config.refresh_interval,
    )


def _find_station_name(train: dict, station_code: str) -> str:
    """Look up a station's display name from train data, falling back to the code."""
    for s in train.get("stations", []):
        if s.get("code", "").upper() == station_code.upper():
            return s.get("name", station_code)
    return station_code


def _build_partial_connection_panel(
    active_train: dict,
    active_num: str,
    missing_num: str,
    connection_station: str,
    active_is_arriving: bool,
) -> Panel:
    """Build a simplified connection panel when only one train has data.

    If active_is_arriving is True, the active train is arriving at the connection
    and the missing train is the departure. Otherwise, the active train is the
    departure and the missing train was the arrival.
    """
    content = Table.grid(padding=(0, 2))
    content.add_column(justify="left")

    if active_is_arriving:
        # Active train is arriving at connection; missing train departs later
        _, _, arr, _ = get_station_times(active_train, connection_station)
        sch_arr, _, _, _ = get_station_times(active_train, connection_station)
        arrival_time = arr or sch_arr
        arr_str = format_time(arrival_time) if arrival_time else "—"

        status = get_station_status(active_train, connection_station)
        if status == "Departed":
            status_text, status_style = "✓ Arrived", "green"
        elif status == "Station":
            status_text, status_style = "● At station", "cyan"
        else:
            status_text, status_style = "◯ En route", "yellow"

        route_name = active_train.get("routeName", f"Train #{active_num}")
        content.add_row(Text(f"{route_name} arrives: {arr_str}", style=status_style))
        content.add_row(Text(f"Status: {status_text}", style=status_style))
        content.add_row(Text(""))
        content.add_row(Text(f"Train #{missing_num} departure time will show", style="dim"))
        content.add_row(Text("once that train's data is available.", style="dim"))
    else:
        # Active train is the departure; missing train data unavailable
        content.add_row(Text(f"Train #{missing_num} data not available", style="yellow"))
        content.add_row(Text(""))

        _, sch_dep, _, dep = get_station_times(active_train, connection_station)
        depart_time = dep or sch_dep
        dep_str = format_time(depart_time) if depart_time else "—"

        route_name = active_train.get("routeName", f"Train #{active_num}")
        content.add_row(Text(f"{route_name} departs: {dep_str}"))

    station_name = _find_station_name(active_train, connection_station)

    return Panel(
        content,
        title=f"[bold yellow]🔗 Connection at {station_name} ({connection_station})[/]",
        border_style="yellow",
    )


def build_multi_train_display(train_numbers: list[str], connection_station: str, show_all: bool = False) -> Layout:
    """Build display for multiple trains with connection info."""
    layout = Layout()

    # Fetch data for all trains
    trains_data = []
    for num in train_numbers:
        data = fetch_train_data_cached(num)
        trains_data.append((num, data))

    train1_num, train1_data = trains_data[0]
    train2_num, train2_data = trains_data[1] if len(trains_data) > 1 else (None, None)

    # Check what we have
    train1_valid = train1_data and "error" not in train1_data
    train2_valid = train2_data and "error" not in train2_data

    # If neither train is valid, show error
    if not train1_valid and not train2_valid:
        error_content = Table.grid()
        error_content.add_row(Text(f"Train #{train1_num}: Not found or awaiting departure", style="yellow"))
        if train2_num:
            error_content.add_row(Text(f"Train #{train2_num}: Not found or awaiting departure", style="yellow"))
        error_content.add_row(Text(""))
        error_content.add_row(Text("Both trains need to be active for connection tracking.", style="dim"))
        error_content.add_row(Text("Try again once at least one train has departed.", style="dim"))

        layout.update(Panel(error_content, title="[bold yellow]Waiting for Train Data[/]", border_style="yellow"))
        return layout

    # Build layout - adapt based on what data we have
    if train1_valid and train2_valid:
        # Both trains active - full connection view
        layout.split(
            Layout(name="train1_header", size=6),
            Layout(name="connection", size=6),
            Layout(name="train2_header", size=6),
            Layout(name="stations", ratio=1),
        )

        train1_panel = _display_build_compact_train_header(train1_data)
        _apply_main_title(train1_panel)
        layout["train1_header"].update(train1_panel)
        layout["connection"].update(build_connection_panel(train1_data, train2_data, connection_station))
        layout["train2_header"].update(_display_build_compact_train_header(train2_data))

        # Show stations for the train that's currently active/relevant
        train1_at_connection = get_station_status(train1_data, connection_station) in ("Departed", "Station")

        if train1_at_connection:
            active_train = train2_data
            active_label = f"#{train2_num}"
        else:
            active_train = train1_data
            active_label = f"#{train1_num}"

        stations_panel = build_stations_table(active_train, focus=not show_all)
        stations_panel.title = f"[bold]Stations - Train {active_label}[/] [dim](green=actual, cyan=estimated)[/]"
        layout["stations"].update(stations_panel)

    elif train1_valid:
        # Only train 1 is active - show it with predeparture notice for train 2
        layout.split(
            Layout(name="train1_header", size=6),
            Layout(name="connection", size=5),
            Layout(name="train2_header", size=5),
            Layout(name="stations", ratio=1),
        )

        train1_panel = _display_build_compact_train_header(train1_data)
        _apply_main_title(train1_panel)
        layout["train1_header"].update(train1_panel)
        layout["train2_header"].update(build_predeparture_header(train2_num))
        layout["connection"].update(
            _build_partial_connection_panel(train1_data, train1_num, train2_num, connection_station, active_is_arriving=True)
        )

        stations_panel = build_stations_table(train1_data, focus=not show_all)
        stations_panel.title = f"[bold]Stations - Train #{train1_num}[/] [dim](green=actual, cyan=estimated)[/]"
        layout["stations"].update(stations_panel)

    else:
        # Only train 2 is active (unusual case - train 1 finished or not found)
        layout.split(
            Layout(name="train1_header", size=5),
            Layout(name="connection", size=5),
            Layout(name="train2_header", size=6),
            Layout(name="stations", ratio=1),
        )

        train1_panel = build_predeparture_header(train1_num)
        _apply_main_title(train1_panel)
        layout["train1_header"].update(train1_panel)
        layout["train2_header"].update(_display_build_compact_train_header(train2_data))
        layout["connection"].update(
            _build_partial_connection_panel(train2_data, train2_num, train1_num, connection_station, active_is_arriving=False)
        )

        stations_panel = build_stations_table(train2_data, focus=not show_all)
        stations_panel.title = f"[bold]Stations - Train #{train2_num}[/] [dim](green=actual, cyan=estimated)[/]"
        layout["stations"].update(stations_panel)
    
    return layout


def select_connection_station(console: Console, overlaps: list[str], train1: dict, train2: dict) -> str | None:
    """Interactive prompt to select connection station when multiple overlaps exist."""
    console.print("\n[bold yellow]Multiple possible connection stations found:[/]\n")
    
    for i, code in enumerate(overlaps, 1):
        # Get station name from either train
        name = code
        for station in train1.get("stations", []):
            if station.get("code", "").upper() == code:
                name = station.get("name", code)
                break
        console.print(f"  {i}. {name} ({code})")
    
    console.print()
    
    while True:
        choice = Prompt.ask(
            "Select connection station",
            choices=[str(i) for i in range(1, len(overlaps) + 1)] + [c.upper() for c in overlaps],
            default="1"
        )
        
        # Handle numeric choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(overlaps):
                return overlaps[idx]
        # Handle station code
        elif choice.upper() in overlaps:
            return choice.upper()
        
        console.print("[red]Invalid selection. Try again.[/]")


def build_display(train_number: str, show_all: bool = False) -> Layout | Text:
    """Build the full TUI display or compact display."""
    train_data = fetch_train_data(train_number)

    if _config.compact_mode:
        if train_data is None:
            return Text(f"🚂 Train #{train_number} not found", style="yellow")
        if "error" in train_data:
            return Text(f"🚂 Train #{train_number} error: {train_data['error']}", style="red")
        return build_compact_display(train_data)
    
    layout = Layout()
    
    if train_data is None:
        layout.update(build_not_found_panel(train_number))
        return layout
    
    if "error" in train_data:
        layout.update(build_error_panel(train_data["error"]))
        return layout
    
    layout.split(
        Layout(name="header", size=9),
        Layout(name="progress", size=3),
        Layout(name="stations"),
    )
    
    layout["header"].update(build_header(train_data))
    layout["progress"].update(build_progress_bar(train_data))
    layout["stations"].update(build_stations_table(train_data, focus=not show_all))
    
    return layout


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Track Amtrak train status in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s 42                      # Track the Pennsylvanian #42
    %(prog)s 42 178                  # Track two trains with auto-detected connection
    %(prog)s 42 178 --connection PHL # Track with specific connection station
    %(prog)s 42 --from PGH --to NYP  # Only show Pittsburgh to New York
    %(prog)s 42 --compact            # Single-line output for status bars
    %(prog)s 42 --all                # Show all stations (no auto-focus)
    %(prog)s 42 --notify-at NYP      # Notify when arriving at New York
    %(prog)s 42 --notify-at PGH,HBG  # Notify at multiple stations
    %(prog)s 42 --notify-all         # Notify at every station

Train IDs can optionally include the day: 42-26 (train 42 from the 26th)

Station codes are 3-letter codes like PGH (Pittsburgh), NYP (New York Penn),
CHI (Chicago), etc. Use --all to see all station codes on a route.
        """
    )
    parser.add_argument(
        "train_numbers",
        nargs="+",
        help="Amtrak train number(s) to track (e.g., 42 or 42 178 for connection)"
    )
    parser.add_argument(
        "-r", "--refresh",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Display once and exit (no auto-refresh)"
    )
    parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Compact single-line output (for status bars, tmux, etc.)"
    )
    parser.add_argument(
        "--connection",
        metavar="CODE",
        help="Station code for connection between trains (auto-detected if not specified)"
    )
    parser.add_argument(
        "--from",
        dest="from_station",
        metavar="CODE",
        help="Only show stations starting from this station code (e.g., PGH)"
    )
    parser.add_argument(
        "--to",
        dest="to_station",
        metavar="CODE",
        help="Only show stations up to this station code (e.g., NYP)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all stations without auto-focusing on current position"
    )
    parser.add_argument(
        "--no-focus",
        action="store_true",
        help="Disable auto-focus on current station (show all departed stops)"
    )
    parser.add_argument(
        "--notify-at",
        metavar="CODE",
        help="Send notification when train arrives at station(s). "
             "Use comma-separated codes for multiple (e.g., PGH,HBG,NYP)"
    )
    parser.add_argument(
        "--notify-all",
        action="store_true",
        help="Send notification when train arrives at any station"
    )
    return parser


def _apply_args_to_config(args) -> None:
    """Apply parsed CLI arguments to the module-level Config object."""
    _config.refresh_interval = args.refresh
    _config.compact_mode = args.compact
    _config.station_from = args.from_station
    _config.station_to = args.to_station
    _config.focus_current = not args.no_focus and not getattr(args, 'all', False)

    _notify_state.notify_all = args.notify_all
    if args.notify_at:
        _notify_state.stations.update(code.strip().upper() for code in args.notify_at.split(","))


def _fetch_predeparture_schedule(console, connection_station, missing_train):
    """Try to fetch and cache predeparture schedule data for a missing train."""
    station_data = fetch_station_schedule(connection_station)
    if station_data and "error" not in station_data:
        sched = get_train_schedule_from_station(station_data, missing_train)
        if sched:
            console.print(f"[green]✓ Found schedule for train {missing_train} at {connection_station}[/]")
            _cache.per_train[missing_train] = {
                "data": build_predeparture_train_data(missing_train, connection_station, sched),
                "fetch_time": _now(),
                "error": None
            }
            return True
        else:
            console.print(f"[yellow]No schedule found for train {missing_train}[/]")
    return False


def _setup_connection(console, train_numbers, connection_station):
    """Set up multi-train connection: detect or validate connection station.

    Returns the resolved connection station code.
    """
    console.print("[dim]Fetching train data...[/]")

    train1_data = fetch_train_data(train_numbers[0])
    train2_data = fetch_train_data(train_numbers[1])

    train1_valid = train1_data and "error" not in train1_data
    train2_valid = train2_data and "error" not in train2_data

    # Connection station already provided
    if connection_station:
        if not train1_valid and not train2_valid:
            console.print("[yellow]Neither train is active yet.[/]")
            console.print(f"[dim]Fetching schedule from {connection_station}...[/]")

            station_data = fetch_station_schedule(connection_station)
            if station_data and "error" not in station_data:
                sched1 = get_train_schedule_from_station(station_data, train_numbers[0])
                sched2 = get_train_schedule_from_station(station_data, train_numbers[1])

                if sched1 or sched2:
                    console.print(f"[green]✓ Found schedule data at {connection_station}[/]")
                    if sched1:
                        _cache.per_train[train_numbers[0]] = {
                            "data": build_predeparture_train_data(train_numbers[0], connection_station, sched1),
                            "fetch_time": _now(),
                            "error": None
                        }
                    if sched2:
                        _cache.per_train[train_numbers[1]] = {
                            "data": build_predeparture_train_data(train_numbers[1], connection_station, sched2),
                            "fetch_time": _now(),
                            "error": None
                        }
                else:
                    console.print(f"[yellow]No schedule found for trains at {connection_station}[/]")
            sleep(1)
        elif not train1_valid or not train2_valid:
            missing_train = train_numbers[0] if not train1_valid else train_numbers[1]
            console.print(f"[yellow]Train {missing_train} not active yet - checking station schedule...[/]")
            _fetch_predeparture_schedule(console, connection_station, missing_train)
            sleep(1)

        console.print(f"[green]✓ Connection station: {connection_station}[/]")
        sleep(1)
        return connection_station

    # Need to detect connection station
    if train1_valid and train2_valid:
        overlaps = find_overlapping_stations(train1_data, train2_data)

        if not overlaps:
            console.print("[red]No overlapping stations found between these trains.[/]")
            console.print("[dim]These trains don't share any stations. Use single-train tracking instead.[/]")
            sys.exit(1)
        elif len(overlaps) == 1:
            connection_station = overlaps[0]
            station_name = connection_station
            for s in train1_data.get("stations", []):
                if s.get("code", "").upper() == connection_station:
                    station_name = s.get("name", connection_station)
                    break
            console.print(f"[green]✓ Auto-detected connection at {station_name} ({connection_station})[/]")
            sleep(1)
        else:
            connection_station = select_connection_station(console, overlaps, train1_data, train2_data)
            if not connection_station:
                console.print("[red]No connection station selected.[/]")
                sys.exit(1)
            console.print(f"[green]✓ Connection set to {connection_station}[/]")
            sleep(1)

    elif train1_valid or train2_valid:
        missing_train = train_numbers[0] if not train1_valid else train_numbers[1]
        active_train = train_numbers[1] if not train1_valid else train_numbers[0]
        active_data = train2_data if not train1_valid else train1_data

        console.print(f"[yellow]Train {missing_train} is not active yet (may be predeparture).[/]")
        console.print(f"[dim]Train {active_train} is active.[/]")
        console.print()

        console.print("[bold]Please specify your connection station.[/]")
        console.print("[dim]Stations on the active train's route:[/]")

        stations = active_data.get("stations", [])
        for i, s in enumerate(stations[:15], 1):
            code = s.get("code", "???")
            name = s.get("name", code)
            console.print(f"  {code}: {name}")
        if len(stations) > 15:
            console.print(f"  ... and {len(stations) - 15} more")

        console.print()
        connection_station = Prompt.ask(
            "Enter connection station code",
            default=stations[0].get("code", "") if stations else ""
        ).upper()

        if not connection_station:
            console.print("[red]No connection station provided.[/]")
            sys.exit(1)

        console.print(f"[dim]Fetching schedule from {connection_station}...[/]")
        _fetch_predeparture_schedule(console, connection_station, missing_train)

        console.print(f"[green]✓ Connection set to {connection_station}[/]")
        sleep(1)

    else:
        console.print(f"[yellow]Neither train {train_numbers[0]} nor {train_numbers[1]} is active yet.[/]")
        console.print()
        connection_station = Prompt.ask(
            "Enter connection station code (e.g., PHL, NYP, WAS)"
        ).upper()

        if not connection_station:
            console.print("[red]No connection station provided.[/]")
            sys.exit(1)

        console.print(f"[dim]Fetching schedule from {connection_station}...[/]")
        station_data = fetch_station_schedule(connection_station)
        if station_data and "error" not in station_data:
            for train_num in train_numbers[:2]:
                sched = get_train_schedule_from_station(station_data, train_num)
                if sched:
                    console.print(f"[green]✓ Found schedule for train {train_num}[/]")
                    _cache.per_train[train_num] = {
                        "data": build_predeparture_train_data(train_num, connection_station, sched),
                        "fetch_time": _now(),
                        "error": None
                    }
                else:
                    console.print(f"[yellow]No schedule found for train {train_num}[/]")

        console.print(f"[green]✓ Connection set to {connection_station}[/]")
        sleep(1)

    return connection_station


def _run_single_train(console, train_number, args):
    """Run the single-train display loop."""
    show_all = getattr(args, 'all', False)

    if args.once:
        result = build_display(train_number, show_all=show_all)
        console.print(result)
        if _cache.last_successful_data:
            check_and_notify(_cache.last_successful_data)
        return

    if _config.compact_mode:
        result = build_display(train_number, show_all=show_all)
        console.print(result)
        if _cache.last_successful_data:
            check_and_notify(_cache.last_successful_data)
        try:
            while True:
                sleep(_config.refresh_interval)
                console.clear()
                console.print(build_display(train_number, show_all=show_all))
                if _cache.last_successful_data:
                    check_and_notify(_cache.last_successful_data)
        except KeyboardInterrupt:
            pass
        return

    try:
        with Live(
            build_display(train_number, show_all=show_all),
            console=console,
            refresh_per_second=1,
            screen=True
        ) as live:
            if _cache.last_successful_data:
                check_and_notify(_cache.last_successful_data)

            while True:
                sleep(_config.refresh_interval)
                live.update(build_display(train_number, show_all=show_all))
                if _cache.last_successful_data:
                    check_and_notify(_cache.last_successful_data)
    except KeyboardInterrupt:
        console.print("\n[dim]Tracking stopped.[/]")
        sys.exit(0)


def _run_multi_train(console, train_numbers, args):
    """Run the multi-train display loop."""
    show_all = getattr(args, 'all', False)

    def _check_notifications():
        """Check notifications for all tracked trains using cached data."""
        for num in train_numbers[:2]:
            if num in _cache.per_train and _cache.per_train[num].get("data"):
                check_and_notify(_cache.per_train[num]["data"])

    if args.once:
        result = build_multi_train_display(train_numbers[:2], _config.connection_station, show_all=show_all)
        console.print(result)
        _check_notifications()
        return

    if _config.compact_mode:
        console.print("[yellow]Compact mode with connections - showing basic info[/]")
        try:
            while True:
                console.clear()
                for num in train_numbers[:2]:
                    data = fetch_train_data_cached(num)
                    if data and "error" not in data:
                        console.print(build_compact_display(data))
                    else:
                        console.print(Text(f"🚂 Train #{num}: Error or not found", style="red"))
                _check_notifications()
                sleep(_config.refresh_interval)
        except KeyboardInterrupt:
            pass
        return

    try:
        with Live(
            build_multi_train_display(train_numbers[:2], _config.connection_station, show_all=show_all),
            console=console,
            refresh_per_second=1,
            screen=True
        ) as live:
            _check_notifications()

            while True:
                sleep(_config.refresh_interval)
                live.update(build_multi_train_display(train_numbers[:2], _config.connection_station, show_all=show_all))
                _check_notifications()
    except KeyboardInterrupt:
        console.print("\n[dim]Tracking stopped.[/]")
        sys.exit(0)


def main():
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    _apply_args_to_config(args)

    console = Console()
    train_numbers = args.train_numbers
    is_multi_train = len(train_numbers) >= 2

    if is_multi_train:
        _config.connection_station = args.connection.upper() if args.connection else None
        _config.connection_station = _setup_connection(console, train_numbers, _config.connection_station)

    # Show notification status if enabled
    if _notify_state.stations or _notify_state.notify_all:
        if _notify_state.notify_all:
            console.print("[dim]🔔 Notifications enabled for all stations[/]")
        else:
            console.print(f"[dim]🔔 Notifications enabled for: {', '.join(sorted(_notify_state.stations))}[/]")
        sleep(1)

    if is_multi_train:
        _run_multi_train(console, train_numbers, args)
    else:
        _run_single_train(console, train_numbers[0], args)


if __name__ == "__main__":
    main()
