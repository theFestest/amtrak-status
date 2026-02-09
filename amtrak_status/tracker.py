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
from datetime import datetime
from time import sleep
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from .config import (
    API_BASE, REFRESH_INTERVAL as _DEFAULT_REFRESH_INTERVAL,
    MAX_RETRIES, RETRY_DELAY,
    LAYOVER_COMFORTABLE, LAYOVER_TIGHT, LAYOVER_RISKY,
)
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

REFRESH_INTERVAL = _DEFAULT_REFRESH_INTERVAL

# Display options (set via command line args)
COMPACT_MODE = False
STATION_FROM: str | None = None
STATION_TO: str | None = None
FOCUS_CURRENT = True  # Auto-focus on current station

# Multi-train tracking
CONNECTION_STATION: str | None = None  # Station code where trains connect

# Encapsulated state objects (replace the old scattered globals)
_cache = TrainCache()
_notify_state = NotificationState()


# Wrapper functions that pass module-level state objects
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
    route_name = train.get("routeName", "Unknown Route")
    train_num = train.get("trainNum", "?")
    train_id = train.get("trainID", "")
    heading = train.get("heading", "")
    velocity = train.get("velocity", 0)
    train_state = train.get("trainState", "")
    status_msg = train.get("statusMsg", "")
    
    # Get current status
    stations = train.get("stations", [])
    completed, current_idx, total = calculate_progress(stations)
    
    # Find next station (first non-departed, non-cancelled station)
    next_station = "—"
    eta = "—"
    for station in stations:
        # Skip cancelled stops (no scheduled times)
        if is_station_cancelled(station):
            continue
        if station.get("status") in ("Enroute", "Station", ""):
            next_station = station.get("name", station.get("code", "?"))
            # arr field contains estimated arrival for future stations
            est_arr = parse_time(station.get("arr"))
            sch_arr = parse_time(station.get("schArr"))
            if est_arr:
                eta = format_time(est_arr)
                # Show difference from scheduled if we have both
                if sch_arr and est_arr != sch_arr:
                    diff_mins = (est_arr - sch_arr).total_seconds() / 60
                    if diff_mins > 0:
                        eta += f" [red](+{diff_mins:.0f}m)[/]"
                    elif diff_mins < 0:
                        eta += f" [green]({diff_mins:.0f}m)[/]"
            elif sch_arr:
                eta = f"{format_time(sch_arr)} [dim](sched)[/]"
            break
    
    # Get destination
    dest_name = train.get("destName", "")
    
    # Build status display
    if status_msg:
        display_status = status_msg
    elif train_state == "Predeparture":
        display_status = "Predeparture"
    else:
        display_status = "Active"
    
    # Status color
    if "early" in display_status.lower() or "on time" in display_status.lower():
        status_style = "green"
    elif "late" in display_status.lower() or "delay" in display_status.lower():
        status_style = "red"
    else:
        status_style = "white"
    
    header = Table.grid(padding=(0, 2))
    header.add_column(justify="left", style="bold white")
    header.add_column(justify="left")
    
    header.add_row(
        Text.from_markup(f"🚂 {route_name} [dim]#{train_num} ({train_id})[/]"),
        ""
    )
    header.add_row(
        Text.from_markup(f"Next: {next_station} [dim]@ {eta}[/]"),
        ""
    )
    speed_str = f"{velocity:.0f} mph" if velocity else "—"
    header.add_row(
        f"Heading: {heading or '—'} @ {speed_str}",
        Text(display_status, style=status_style)
    )
    
    # Add position progress bar between stations
    position = calculate_position_between_stations(train)
    if position and train_state != "Predeparture":
        last_code, next_code, progress_frac, mins_remaining = position
        
        # Build a mini progress bar: Position: LAST ▓▓▓▓▓▓░░░░ NEXT (XX min)
        bar_width = 20
        filled = int(progress_frac * bar_width)
        empty = bar_width - filled
        
        bar = f"[green]{'█' * filled}[/][dim]{'░' * empty}[/]"
        if mins_remaining > 0:
            time_str = f"({mins_remaining} min)" if mins_remaining != 1 else "(1 min)"
        else:
            time_str = "(arriving)"
        
        position_text = Text.from_markup(f"Position: {last_code} {bar} {next_code} [dim]{time_str}[/]")
        header.add_row(position_text, "")
    
    if dest_name:
        header.add_row(
            f"Destination: {dest_name}",
            ""
        )
    
    # Build subtitle with status indicator - show last SUCCESSFUL update time
    if _cache.last_fetch_time:
        update_str = f"Updated: {_cache.last_fetch_time.strftime('%H:%M:%S')}"
    else:
        update_str = "Updated: —"
    
    status_parts = [update_str]
    if _cache.last_error:
        status_parts.append(f"[yellow]⚠ {_cache.last_error}[/]")
    status_parts.append(f"Refresh: {REFRESH_INTERVAL}s")
    status_parts.append("Press Ctrl+C to quit")
    subtitle = " | ".join(status_parts)
    
    return Panel(
        header,
        title=f"[bold cyan]Amtrak Status[/]",
        subtitle=f"[dim]{subtitle}[/]",
        border_style="cyan"
    )


def build_progress_bar(train: dict) -> Panel:
    """Build a visual progress bar for the journey."""
    stations = train.get("stations", [])
    completed, current_idx, total = calculate_progress(stations)
    
    if total == 0:
        return Panel("No station data", title="Progress")
    
    # Get origin and destination
    origin = stations[0].get("name", stations[0].get("code", "?")) if stations else "?"
    dest = stations[-1].get("name", stations[-1].get("code", "?")) if stations else "?"
    
    progress = Progress(
        TextColumn("[bold blue]{task.fields[origin]}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="green"),
        TextColumn("[bold blue]{task.fields[dest]}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    
    task = progress.add_task(
        "journey",
        total=total,
        completed=completed,
        origin=origin[:15],
        dest=dest[:15]
    )
    
    return Panel(progress, title="[bold]Journey Progress[/]", border_style="blue")




def build_connection_panel(train1: dict, train2: dict, connection_station: str) -> Panel:
    """Build a panel showing connection status between two trains."""
    layover = calculate_layover(train1, train2, connection_station)
    
    train1_name = train1.get("routeName", "Train 1")
    train1_num = train1.get("trainNum", "?")
    train2_name = train2.get("routeName", "Train 2")
    train2_num = train2.get("trainNum", "?")
    
    # Build content
    content = Table.grid(padding=(0, 2))
    content.add_column(justify="right", style="dim")
    content.add_column(justify="left")
    content.add_column(justify="left")
    
    # Train 1 arrival
    arr_time = format_time(layover["train1_arrives"]) if layover["train1_arrives"] else "—"
    if layover["train1_status"] == "Departed":
        arr_style = "green"
        arr_label = "✓ Arrived"
    elif layover["train1_status"] == "Station":
        arr_style = "cyan bold"
        arr_label = "● At station"
    else:
        arr_style = "yellow"
        arr_label = "◯ Expected"
    
    content.add_row(
        f"{train1_name} #{train1_num}",
        Text(f"Arrives: {arr_time}", style=arr_style),
        Text(arr_label, style=arr_style)
    )
    
    # Layover indicator
    if layover["layover_minutes"] is not None:
        mins = layover["layover_minutes"]
        status = layover["layover_status"]
        
        if status == "missed":
            layover_style = "red bold"
            layover_icon = "✗"
            layover_text = f"MISSED by {abs(mins)} min"
        elif status == "risky":
            layover_style = "red"
            layover_icon = "⚠"
            layover_text = f"{mins} min layover (risky!)"
        elif status == "tight":
            layover_style = "yellow"
            layover_icon = "⚡"
            layover_text = f"{mins} min layover (tight)"
        else:
            layover_style = "green"
            layover_icon = "✓"
            hours = mins // 60
            remaining_mins = mins % 60
            if hours > 0:
                layover_text = f"{hours}h {remaining_mins}m layover"
            else:
                layover_text = f"{mins} min layover"
        
        content.add_row(
            "",
            Text(f"{layover_icon} {layover_text}", style=layover_style),
            ""
        )
    else:
        content.add_row("", Text("— Layover unknown", style="dim"), "")
    
    # Train 2 departure
    dep_time = format_time(layover["train2_departs"]) if layover["train2_departs"] else "—"
    if layover["train2_status"] == "Departed":
        dep_style = "red" if not layover["is_valid"] else "green"
        dep_label = "✗ Departed" if not layover["is_valid"] else "✓ Departed"
    elif layover["train2_status"] == "Station":
        dep_style = "cyan bold"
        dep_label = "● Boarding"
    else:
        dep_style = "dim"
        dep_label = "◯ Scheduled"
    
    content.add_row(
        f"{train2_name} #{train2_num}",
        Text(f"Departs: {dep_time}", style=dep_style),
        Text(dep_label, style=dep_style)
    )
    
    # Panel styling based on connection status
    if layover["layover_status"] == "missed":
        border_style = "red"
        title_style = "bold red"
    elif layover["layover_status"] == "risky":
        border_style = "red"
        title_style = "bold yellow"
    elif layover["layover_status"] == "tight":
        border_style = "yellow"
        title_style = "bold yellow"
    else:
        border_style = "green"
        title_style = "bold green"
    
    return Panel(
        content,
        title=f"[{title_style}]🔗 Connection at {layover['station_name']} ({layover['station_code']})[/]",
        border_style=border_style
    )


def build_compact_train_header(train: dict) -> Panel:
    """Build a more compact header for multi-train view."""
    route_name = train.get("routeName", "Unknown Route")
    train_num = train.get("trainNum", "?")
    train_id = train.get("trainID", "")
    velocity = train.get("velocity", 0)
    status_msg = train.get("statusMsg", "")
    train_state = train.get("trainState", "")
    is_predeparture_synthetic = train.get("_predeparture", False)
    
    stations = train.get("stations", [])
    
    # Handle predeparture synthetic data specially
    if is_predeparture_synthetic:
        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="left")
        
        header.add_row(
            Text(f"🚂 Train #{train_num}", style="bold"),
            Text("⏳ Predeparture", style="yellow")
        )
        
        # Show scheduled time at connection if available
        if stations:
            station = stations[0]
            sch_dep = parse_time(station.get("schDep"))
            sch_arr = parse_time(station.get("schArr"))
            station_code = station.get("code", "")
            
            if sch_dep:
                header.add_row(
                    Text.from_markup(f"Departs {station_code}: [cyan]{format_time(sch_dep)}[/] [dim](scheduled)[/]"),
                    ""
                )
            elif sch_arr:
                header.add_row(
                    Text.from_markup(f"Arrives {station_code}: [cyan]{format_time(sch_arr)}[/] [dim](scheduled)[/]"),
                    ""
                )
        
        header.add_row(
            Text("Live tracking begins at departure", style="dim"),
            ""
        )
        
        return Panel(header, border_style="yellow")
    
    # Normal active train header
    # Find next station (first non-departed, non-cancelled station)
    next_station = "—"
    eta = "—"
    for station in stations:
        if is_station_cancelled(station):
            continue
        if station.get("status") in ("Enroute", "Station", ""):
            next_station = station.get("name", station.get("code", "?"))
            est_arr = parse_time(station.get("arr"))
            sch_arr = parse_time(station.get("schArr"))
            if est_arr:
                eta = format_time(est_arr)
                if sch_arr and est_arr != sch_arr:
                    diff_mins = (est_arr - sch_arr).total_seconds() / 60
                    if diff_mins > 0:
                        eta += f" [red](+{diff_mins:.0f}m)[/]"
                    elif diff_mins < 0:
                        eta += f" [green]({diff_mins:.0f}m)[/]"
            elif sch_arr:
                eta = f"{format_time(sch_arr)} [dim](sched)[/]"
            break
    
    # Status display
    if status_msg:
        display_status = status_msg
    elif train_state == "Predeparture":
        display_status = "Predeparture"
    else:
        display_status = "Active"
    
    if "early" in display_status.lower() or "on time" in display_status.lower():
        status_style = "green"
    elif "late" in display_status.lower() or "delay" in display_status.lower():
        status_style = "red"
    elif train_state == "Predeparture":
        status_style = "yellow"
    else:
        status_style = "white"
    
    # Build compact header
    header = Table.grid(padding=(0, 2))
    header.add_column(justify="left")
    header.add_column(justify="left")
    
    speed_str = f"{velocity:.0f} mph" if velocity else "—"
    
    header.add_row(
        Text.from_markup(f"🚂 {route_name} [dim]#{train_num}[/]"),
        Text(display_status, style=status_style)
    )
    header.add_row(
        Text.from_markup(f"Next: {next_station} [dim]@ {eta}[/]"),
        Text(f"{speed_str}", style="dim")
    )
    
    # Position bar
    position = calculate_position_between_stations(train)
    if position and train_state != "Predeparture":
        last_code, next_code, progress_frac, mins_remaining = position
        bar_width = 15
        filled = int(progress_frac * bar_width)
        empty = bar_width - filled
        bar = f"[green]{'█' * filled}[/][dim]{'░' * empty}[/]"
        if mins_remaining > 0:
            time_str = f"({mins_remaining}m)"
        else:
            time_str = "(arriving)"
        header.add_row(
            Text.from_markup(f"{last_code} {bar} {next_code} [dim]{time_str}[/]"),
            ""
        )
    
    return Panel(header, border_style="cyan")


def build_predeparture_panel(train_number: str) -> Panel:
    """Build a panel for a train that hasn't departed yet."""
    content = Table.grid(padding=(0, 2))
    content.add_column(justify="left")
    
    content.add_row(Text(f"🚂 Train #{train_number}", style="bold"))
    content.add_row(Text(""))
    content.add_row(Text("⏳ Awaiting Departure", style="yellow"))
    content.add_row(Text(""))
    content.add_row(Text("Live tracking will begin once", style="dim"))
    content.add_row(Text("the train departs its origin.", style="dim"))
    
    return Panel(
        content,
        title=f"[bold yellow]Train #{train_number} - Predeparture[/]",
        border_style="yellow"
    )


def build_predeparture_header(train_number: str) -> Panel:
    """Build a compact predeparture header for multi-train view."""
    content = Table.grid(padding=(0, 2))
    content.add_column(justify="left")
    content.add_column(justify="left")
    
    content.add_row(
        Text(f"🚂 Train #{train_number}", style="bold"),
        Text("⏳ Awaiting Departure", style="yellow")
    )
    content.add_row(
        Text("Live tracking begins at departure", style="dim"),
        ""
    )
    
    return Panel(content, border_style="yellow")


def _apply_main_title(panel: Panel) -> None:
    """Add the main 'Amtrak Status' title and status subtitle to a panel."""
    panel.title = "[bold cyan]Amtrak Status[/]"
    status_parts = []
    if _cache.last_fetch_time:
        status_parts.append(f"Updated: {_cache.last_fetch_time.strftime('%H:%M:%S')}")
    else:
        status_parts.append("Updated: —")
    if _cache.last_error:
        status_parts.append(f"[yellow]⚠ {_cache.last_error}[/]")
    status_parts.append(f"Refresh: {REFRESH_INTERVAL}s")
    status_parts.append("Press Ctrl+C to quit")
    panel.subtitle = f"[dim]{' | '.join(status_parts)}[/]"


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
        
        train1_panel = build_compact_train_header(train1_data)
        _apply_main_title(train1_panel)
        layout["train1_header"].update(train1_panel)
        layout["connection"].update(build_connection_panel(train1_data, train2_data, connection_station))
        layout["train2_header"].update(build_compact_train_header(train2_data))
        
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
        
        train1_panel = build_compact_train_header(train1_data)
        _apply_main_title(train1_panel)
        layout["train1_header"].update(train1_panel)
        layout["train2_header"].update(build_predeparture_header(train2_num))
        
        # Show a simplified connection panel
        connection_content = Table.grid(padding=(0, 2))
        connection_content.add_column(justify="left")
        
        # Get train 1's arrival at connection
        _, _, arr1, _ = get_station_times(train1_data, connection_station)
        sch_arr1, _, _, _ = get_station_times(train1_data, connection_station)
        train1_arrives = arr1 or sch_arr1
        arr_str = format_time(train1_arrives) if train1_arrives else "—"
        
        train1_status = get_station_status(train1_data, connection_station)
        if train1_status == "Departed":
            status_text = "✓ Arrived"
            status_style = "green"
        elif train1_status == "Station":
            status_text = "● At station"
            status_style = "cyan"
        else:
            status_text = "◯ En route"
            status_style = "yellow"
        
        route_name = train1_data.get("routeName", f"Train #{train1_num}")
        connection_content.add_row(Text(f"{route_name} arrives: {arr_str}", style=status_style))
        connection_content.add_row(Text(f"Status: {status_text}", style=status_style))
        connection_content.add_row(Text(""))
        connection_content.add_row(Text(f"Train #{train2_num} departure time will show", style="dim"))
        connection_content.add_row(Text("once that train's data is available.", style="dim"))
        
        # Get station name
        station_name = connection_station
        for s in train1_data.get("stations", []):
            if s.get("code", "").upper() == connection_station.upper():
                station_name = s.get("name", connection_station)
                break
        
        layout["connection"].update(Panel(
            connection_content,
            title=f"[bold yellow]🔗 Connection at {station_name} ({connection_station})[/]",
            border_style="yellow"
        ))
        
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
        layout["train2_header"].update(build_compact_train_header(train2_data))
        
        # Simplified connection panel
        connection_content = Table.grid(padding=(0, 2))
        connection_content.add_column(justify="left")
        connection_content.add_row(Text(f"Train #{train1_num} data not available", style="yellow"))
        connection_content.add_row(Text(""))
        
        _, sch_dep2, _, dep2 = get_station_times(train2_data, connection_station)
        train2_departs = dep2 or sch_dep2
        dep_str = format_time(train2_departs) if train2_departs else "—"
        
        route_name = train2_data.get("routeName", f"Train #{train2_num}")
        connection_content.add_row(Text(f"{route_name} departs: {dep_str}"))
        
        station_name = connection_station
        for s in train2_data.get("stations", []):
            if s.get("code", "").upper() == connection_station.upper():
                station_name = s.get("name", connection_station)
                break
        
        layout["connection"].update(Panel(
            connection_content,
            title=f"[bold yellow]🔗 Connection at {station_name} ({connection_station})[/]",
            border_style="yellow"
        ))
        
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


def build_stations_table(train: dict, focus: bool = True) -> Panel:
    """Build the stations table with optional filtering and focus."""
    all_stations = train.get("stations", [])
    
    # Apply station filter if set
    stations, skipped_before, skipped_after = filter_stations(
        all_stations, STATION_FROM, STATION_TO
    )
    
    # Find current station index within filtered list for focusing
    current_idx = find_current_station_index(stations)
    
    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        expand=True,
    )
    
    table.add_column("", width=2, justify="center")
    table.add_column("Station", min_width=20)
    table.add_column("Sch Arr", width=10, justify="center")
    table.add_column("Sch Dep", width=10, justify="center")
    table.add_column("Act/Est Arr", width=12, justify="center")
    table.add_column("Act/Est Dep", width=12, justify="center")
    table.add_column("Status", width=14, justify="center")
    
    # Add "skipped before" indicator
    if skipped_before > 0:
        table.add_row(
            Text("⋮", style="dim"),
            Text(f"[{skipped_before} earlier stops omitted]", style="dim italic"),
            "", "", "", "", ""
        )
    
    # Determine which stations to show when focusing
    # Show: 2 departed stations + current + all future (or all if not focusing)
    if focus and FOCUS_CURRENT and len(stations) > 10:
        # Find how many to skip at the start (keep last 2 departed)
        departed_count = sum(1 for s in stations if s.get("status") == "Departed")
        skip_departed = max(0, departed_count - 2)
        
        if skip_departed > 0:
            table.add_row(
                Text("⋮", style="dim"),
                Text(f"[{skip_departed} departed stops hidden]", style="dim italic"),
                "", "", "", "", ""
            )
            stations = stations[skip_departed:]
    
    for station in stations:
        code = station.get("code", "???")
        name = station.get("name", code)
        status_text = station.get("status", "")
        platform = station.get("platform", "")
        
        # Check if this stop is cancelled
        cancelled = is_station_cancelled(station)
        
        if cancelled:
            # Show cancelled stops with strikethrough-like styling
            style = "dim strike"
            icon = "✗"
            table.add_row(
                Text(icon, style="red dim"),
                Text(f"{name} ({code})", style="dim"),
                "",
                "",
                "",
                "",
                Text("Cancelled", style="red dim")
            )
            continue
        
        style, icon = get_status_style(station)
        
        # Parse times
        sch_arr = parse_time(station.get("schArr"))
        sch_dep = parse_time(station.get("schDep"))
        arr = parse_time(station.get("arr"))
        dep = parse_time(station.get("dep"))
        
        # Format scheduled times
        sch_arr_str = format_time(sch_arr) if sch_arr else ""
        sch_dep_str = format_time(sch_dep) if sch_dep else ""
        
        # Format actual/estimated times based on status
        # For departed stations: arr/dep are ACTUAL times
        # For future stations: arr/dep are ESTIMATED times
        is_departed = status_text == "Departed"
        is_future = status_text in ("", "Enroute")
        is_current = status_text == "Station"
        
        if is_departed:
            # These are actual times - show in green
            arr_str = Text(format_time(arr), style="green") if arr else Text("")
            dep_str = Text(format_time(dep), style="green") if dep else Text("")
        elif is_current:
            # At station - arrival is actual, departure is estimated
            arr_str = Text(format_time(arr), style="green") if arr else Text("")
            dep_str = Text(format_time(dep), style="yellow") if dep else Text("")
        elif is_future:
            # Future - these are estimates
            arr_str = Text(format_time(arr), style="cyan") if arr else Text("")
            dep_str = Text(format_time(dep), style="cyan") if dep else Text("")
        else:
            arr_str = Text(format_time(arr)) if arr else Text("")
            dep_str = Text(format_time(dep)) if dep else Text("")
        
        # Build display status
        if platform and status_text in ("Enroute", "Station"):
            display_status = f"{status_text} (Plt {platform})"
        else:
            display_status = status_text or "Scheduled"
        
        # Color the status
        if status_text == "Departed":
            status_style = "green dim"
        elif status_text == "Station":
            status_style = "cyan bold"
        elif status_text == "Enroute":
            status_style = "yellow bold"
        else:
            status_style = "dim"
        
        table.add_row(
            Text(icon, style=style),
            Text(f"{name} ({code})", style=style),
            sch_arr_str,
            sch_dep_str,
            arr_str,
            dep_str,
            Text(display_status, style=status_style)
        )
    
    # Add "skipped after" indicator
    if skipped_after > 0:
        table.add_row(
            Text("⋮", style="dim"),
            Text(f"[{skipped_after} later stops omitted]", style="dim italic"),
            "", "", "", "", ""
        )
    
    # Build title with filter info
    title_parts = ["[bold]Stations[/]"]
    if STATION_FROM or STATION_TO:
        filter_desc = f"{STATION_FROM or 'start'} → {STATION_TO or 'end'}"
        title_parts.append(f"[dim]({filter_desc})[/]")
    title_parts.append("[dim](green=actual, cyan=estimated)[/]")
    
    return Panel(
        table,
        title=" ".join(title_parts),
        border_style="magenta"
    )


def build_compact_display(train: dict) -> Text:
    """Build a single-line compact display for the train status."""
    route_name = train.get("routeName", "Unknown")
    train_num = train.get("trainNum", "?")
    velocity = train.get("velocity", 0)
    status_msg = train.get("statusMsg", "")
    stations = train.get("stations", [])
    
    # Find next station
    next_station = "—"
    eta = "—"
    delay_str = ""
    
    for station in stations:
        if station.get("status") in ("Enroute", "Station", ""):
            next_station = station.get("code", "???")
            est_arr = parse_time(station.get("arr"))
            sch_arr = parse_time(station.get("schArr"))
            if est_arr:
                eta = format_time(est_arr)
                if sch_arr:
                    diff_mins = (est_arr - sch_arr).total_seconds() / 60
                    if diff_mins > 1:
                        delay_str = f" [red]+{diff_mins:.0f}m[/]"
                    elif diff_mins < -1:
                        delay_str = f" [green]{diff_mins:.0f}m[/]"
            elif sch_arr:
                eta = format_time(sch_arr)
            break
    
    # Calculate progress
    completed, _, total = calculate_progress(stations)
    progress_pct = (completed / total * 100) if total > 0 else 0
    
    # Get position between stations
    position = calculate_position_between_stations(train)
    
    # Build compact line
    speed_str = f"{velocity:.0f}mph" if velocity else "—"
    
    compact = Text()
    compact.append(f"🚂 {route_name} #{train_num}", style="bold")
    compact.append(f" | ")
    
    # Show position if available
    if position:
        last_code, next_code, progress_frac, mins_remaining = position
        pos_pct = int(progress_frac * 100)
        compact.append(f"{last_code}", style="green")
        compact.append(f"→{pos_pct}%→")
        compact.append(f"{next_code}", style="cyan")
        if mins_remaining > 0:
            compact.append(f" ({mins_remaining}m)")
        else:
            compact.append(" (arriving)")
    else:
        compact.append(f"Next: {next_station}", style="cyan")
    
    compact.append(f" @ {eta}")
    compact.append_text(Text.from_markup(delay_str))
    compact.append(f" | {speed_str}")
    compact.append(f" | {progress_pct:.0f}%")
    
    if status_msg:
        compact.append(f" | {status_msg}", style="yellow")
    
    if _cache.last_fetch_time:
        compact.append(f" | Updated {_cache.last_fetch_time.strftime('%H:%M:%S')}", style="dim")
    
    return compact


def build_error_panel(error: str) -> Panel:
    """Build an error display panel."""
    return Panel(
        Text(f"Error: {error}", style="bold red"),
        title="[bold red]Error[/]",
        border_style="red"
    )


def build_not_found_panel(train_number: str) -> Panel:
    """Build a not found display panel."""
    content = Text()
    content.append(f"Train #{train_number} not found.\n\n", style="bold yellow")
    content.append("This could mean:\n", style="white")
    content.append("• The train hasn't started its journey today\n", style="dim")
    content.append("• The train number is incorrect\n", style="dim")
    content.append("• The train has completed its journey\n", style="dim")
    content.append("\nTry checking the train number or wait for the train to depart.", style="white")
    
    return Panel(
        content,
        title="[bold yellow]Train Not Found[/]",
        border_style="yellow"
    )


def build_display(train_number: str, show_all: bool = False) -> Layout | Text:
    """Build the full TUI display or compact display."""
    train_data = fetch_train_data(train_number)
    
    if COMPACT_MODE:
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


def main():
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
    
    args = parser.parse_args()
    
    # Set global config from args
    global REFRESH_INTERVAL, COMPACT_MODE, STATION_FROM, STATION_TO, FOCUS_CURRENT
    global CONNECTION_STATION

    REFRESH_INTERVAL = args.refresh
    COMPACT_MODE = args.compact
    STATION_FROM = args.from_station
    STATION_TO = args.to_station
    FOCUS_CURRENT = not args.no_focus and not getattr(args, 'all', False)

    # Set up notifications
    _notify_state.notify_all = args.notify_all
    if args.notify_at:
        _notify_state.stations.update(code.strip().upper() for code in args.notify_at.split(","))
    
    console = Console()
    
    train_numbers = args.train_numbers
    is_multi_train = len(train_numbers) >= 2
    
    # Handle multi-train connection setup
    if is_multi_train:
        CONNECTION_STATION = args.connection.upper() if args.connection else None
        
        console.print("[dim]Fetching train data...[/]")
        
        # Fetch data for first two trains
        train1_data = fetch_train_data(train_numbers[0])
        train2_data = fetch_train_data(train_numbers[1])
        
        train1_valid = train1_data and "error" not in train1_data
        train2_valid = train2_data and "error" not in train2_data
        
        # If connection station provided, we can proceed even with one missing train
        if CONNECTION_STATION:
            if not train1_valid and not train2_valid:
                console.print(f"[yellow]Neither train is active yet.[/]")
                console.print(f"[dim]Fetching schedule from {CONNECTION_STATION}...[/]")
                
                # Try to get schedule info from station
                station_data = fetch_station_schedule(CONNECTION_STATION)
                if station_data and "error" not in station_data:
                    sched1 = get_train_schedule_from_station(station_data, train_numbers[0])
                    sched2 = get_train_schedule_from_station(station_data, train_numbers[1])
                    
                    if sched1 or sched2:
                        console.print(f"[green]✓ Found schedule data at {CONNECTION_STATION}[/]")
                        # Store synthetic predeparture data in cache
                        if sched1:
                            _cache.per_train[train_numbers[0]] = {
                                "data": build_predeparture_train_data(train_numbers[0], CONNECTION_STATION, sched1),
                                "fetch_time": _now(),
                                "error": None
                            }
                        if sched2:
                            _cache.per_train[train_numbers[1]] = {
                                "data": build_predeparture_train_data(train_numbers[1], CONNECTION_STATION, sched2),
                                "fetch_time": _now(),
                                "error": None
                            }
                    else:
                        console.print(f"[yellow]No schedule found for trains at {CONNECTION_STATION}[/]")
                sleep(1)
            elif not train1_valid or not train2_valid:
                missing_train = train_numbers[0] if not train1_valid else train_numbers[1]
                console.print(f"[yellow]Train {missing_train} not active yet - checking station schedule...[/]")
                
                station_data = fetch_station_schedule(CONNECTION_STATION)
                if station_data and "error" not in station_data:
                    sched = get_train_schedule_from_station(station_data, missing_train)
                    if sched:
                        console.print(f"[green]✓ Found schedule for train {missing_train} at {CONNECTION_STATION}[/]")
                        _cache.per_train[missing_train] = {
                            "data": build_predeparture_train_data(missing_train, CONNECTION_STATION, sched),
                            "fetch_time": _now(),
                            "error": None
                        }
                    else:
                        console.print(f"[yellow]No schedule found for train {missing_train}[/]")
                sleep(1)
            
            console.print(f"[green]✓ Connection station: {CONNECTION_STATION}[/]")
            sleep(1)
        
        # No connection station provided - need to detect it
        elif not CONNECTION_STATION:
            # If both trains are valid, auto-detect connection
            if train1_valid and train2_valid:
                overlaps = find_overlapping_stations(train1_data, train2_data)
                
                if not overlaps:
                    console.print("[red]No overlapping stations found between these trains.[/]")
                    console.print("[dim]These trains don't share any stations. Use single-train tracking instead.[/]")
                    sys.exit(1)
                elif len(overlaps) == 1:
                    CONNECTION_STATION = overlaps[0]
                    station_name = CONNECTION_STATION
                    for s in train1_data.get("stations", []):
                        if s.get("code", "").upper() == CONNECTION_STATION:
                            station_name = s.get("name", CONNECTION_STATION)
                            break
                    console.print(f"[green]✓ Auto-detected connection at {station_name} ({CONNECTION_STATION})[/]")
                    sleep(1)
                else:
                    CONNECTION_STATION = select_connection_station(console, overlaps, train1_data, train2_data)
                    if not CONNECTION_STATION:
                        console.print("[red]No connection station selected.[/]")
                        sys.exit(1)
                    console.print(f"[green]✓ Connection set to {CONNECTION_STATION}[/]")
                    sleep(1)
            
            # If only one train valid, we need user to provide connection station
            elif train1_valid or train2_valid:
                missing_train = train_numbers[0] if not train1_valid else train_numbers[1]
                active_train = train_numbers[1] if not train1_valid else train_numbers[0]
                active_data = train2_data if not train1_valid else train1_data
                
                console.print(f"[yellow]Train {missing_train} is not active yet (may be predeparture).[/]")
                console.print(f"[dim]Train {active_train} is active.[/]")
                console.print()
                
                # Show stations from the active train for user to pick connection
                console.print("[bold]Please specify your connection station.[/]")
                console.print("[dim]Stations on the active train's route:[/]")
                
                stations = active_data.get("stations", [])
                for i, s in enumerate(stations[:15], 1):  # Show first 15
                    code = s.get("code", "???")
                    name = s.get("name", code)
                    console.print(f"  {code}: {name}")
                if len(stations) > 15:
                    console.print(f"  ... and {len(stations) - 15} more")
                
                console.print()
                CONNECTION_STATION = Prompt.ask(
                    "Enter connection station code",
                    default=stations[0].get("code", "") if stations else ""
                ).upper()
                
                if not CONNECTION_STATION:
                    console.print("[red]No connection station provided.[/]")
                    sys.exit(1)
                
                # Try to get schedule for missing train from station
                console.print(f"[dim]Fetching schedule from {CONNECTION_STATION}...[/]")
                station_data = fetch_station_schedule(CONNECTION_STATION)
                if station_data and "error" not in station_data:
                    sched = get_train_schedule_from_station(station_data, missing_train)
                    if sched:
                        console.print(f"[green]✓ Found schedule for train {missing_train}[/]")
                        _cache.per_train[missing_train] = {
                            "data": build_predeparture_train_data(missing_train, CONNECTION_STATION, sched),
                            "fetch_time": _now(),
                            "error": None
                        }
                    else:
                        console.print(f"[yellow]No schedule found for train {missing_train} at {CONNECTION_STATION}[/]")
                
                console.print(f"[green]✓ Connection set to {CONNECTION_STATION}[/]")
                sleep(1)
            
            # Neither train valid
            else:
                console.print(f"[yellow]Neither train {train_numbers[0]} nor {train_numbers[1]} is active yet.[/]")
                console.print()
                CONNECTION_STATION = Prompt.ask(
                    "Enter connection station code (e.g., PHL, NYP, WAS)"
                ).upper()
                
                if not CONNECTION_STATION:
                    console.print("[red]No connection station provided.[/]")
                    sys.exit(1)
                
                # Try to get schedules for both trains from station
                console.print(f"[dim]Fetching schedule from {CONNECTION_STATION}...[/]")
                station_data = fetch_station_schedule(CONNECTION_STATION)
                if station_data and "error" not in station_data:
                    for train_num in train_numbers[:2]:
                        sched = get_train_schedule_from_station(station_data, train_num)
                        if sched:
                            console.print(f"[green]✓ Found schedule for train {train_num}[/]")
                            _cache.per_train[train_num] = {
                                "data": build_predeparture_train_data(train_num, CONNECTION_STATION, sched),
                                "fetch_time": _now(),
                                "error": None
                            }
                        else:
                            console.print(f"[yellow]No schedule found for train {train_num}[/]")
                
                console.print(f"[green]✓ Connection set to {CONNECTION_STATION}[/]")
                sleep(1)
    
    # Show notification status if enabled
    if _notify_state.stations or _notify_state.notify_all:
        if _notify_state.notify_all:
            console.print("[dim]🔔 Notifications enabled for all stations[/]")
        else:
            console.print(f"[dim]🔔 Notifications enabled for: {', '.join(sorted(_notify_state.stations))}[/]")
        sleep(1)
    
    # Single train mode
    if not is_multi_train:
        train_number = train_numbers[0]
        
        if args.once:
            result = build_display(train_number, show_all=getattr(args, 'all', False))
            console.print(result)
            if _cache.last_successful_data:
                check_and_notify(_cache.last_successful_data)
            return
        
        if COMPACT_MODE:
            result = build_display(train_number, show_all=getattr(args, 'all', False))
            console.print(result)
            if _cache.last_successful_data:
                check_and_notify(_cache.last_successful_data)
            try:
                while True:
                    sleep(REFRESH_INTERVAL)
                    console.clear()
                    console.print(build_display(train_number, show_all=getattr(args, 'all', False)))
                    if _cache.last_successful_data:
                        check_and_notify(_cache.last_successful_data)
            except KeyboardInterrupt:
                pass
            return
        
        try:
            with Live(
                build_display(train_number, show_all=getattr(args, 'all', False)),
                console=console,
                refresh_per_second=1,
                screen=True
            ) as live:
                if _cache.last_successful_data:
                    check_and_notify(_cache.last_successful_data)
                
                while True:
                    sleep(REFRESH_INTERVAL)
                    live.update(build_display(train_number, show_all=getattr(args, 'all', False)))
                    if _cache.last_successful_data:
                        check_and_notify(_cache.last_successful_data)
        except KeyboardInterrupt:
            console.print("\n[dim]Tracking stopped.[/]")
            sys.exit(0)
    
    # Multi-train mode
    else:
        def _check_multi_train_notifications():
            """Check notifications for all tracked trains using cached data."""
            for num in train_numbers[:2]:
                if num in _cache.per_train and _cache.per_train[num].get("data"):
                    check_and_notify(_cache.per_train[num]["data"])

        if args.once:
            result = build_multi_train_display(train_numbers[:2], CONNECTION_STATION, show_all=getattr(args, 'all', False))
            console.print(result)
            _check_multi_train_notifications()
            return

        if COMPACT_MODE:
            # Compact mode for multi-train - show both trains on separate lines
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
                    _check_multi_train_notifications()
                    sleep(REFRESH_INTERVAL)
            except KeyboardInterrupt:
                pass
            return

        try:
            with Live(
                build_multi_train_display(train_numbers[:2], CONNECTION_STATION, show_all=getattr(args, 'all', False)),
                console=console,
                refresh_per_second=1,
                screen=True
            ) as live:
                _check_multi_train_notifications()

                while True:
                    sleep(REFRESH_INTERVAL)
                    live.update(build_multi_train_display(train_numbers[:2], CONNECTION_STATION, show_all=getattr(args, 'all', False)))
                    _check_multi_train_notifications()
        except KeyboardInterrupt:
            console.print("\n[dim]Tracking stopped.[/]")
            sys.exit(0)


if __name__ == "__main__":
    main()
