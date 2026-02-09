"""Station table display."""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import (
    parse_time, format_time, get_status_style,
    is_station_cancelled, find_current_station_index, filter_stations,
)


def build_stations_table(
    train: dict,
    focus: bool = True,
    station_from: str | None = None,
    station_to: str | None = None,
    focus_current: bool = True,
) -> Panel:
    """Build the stations table with optional filtering and focus."""
    all_stations = train.get("stations", [])

    # Apply station filter if set
    stations, skipped_before, skipped_after = filter_stations(
        all_stations, station_from, station_to
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
    if focus and focus_current and len(stations) > 10:
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
            table.add_row(
                Text("✗", style="red dim"),
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
        is_departed = status_text == "Departed"
        is_future = status_text in ("", "Enroute")
        is_current = status_text == "Station"

        if is_departed:
            arr_str = Text(format_time(arr), style="green") if arr else Text("")
            dep_str = Text(format_time(dep), style="green") if dep else Text("")
        elif is_current:
            arr_str = Text(format_time(arr), style="green") if arr else Text("")
            dep_str = Text(format_time(dep), style="yellow") if dep else Text("")
        elif is_future:
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
    if station_from or station_to:
        filter_desc = f"{station_from or 'start'} → {station_to or 'end'}"
        title_parts.append(f"[dim]({filter_desc})[/]")
    title_parts.append("[dim](green=actual, cyan=estimated)[/]")

    return Panel(
        table,
        title=" ".join(title_parts),
        border_style="magenta"
    )
