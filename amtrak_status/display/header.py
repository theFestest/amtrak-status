"""Header panels for single-train and multi-train views."""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import (
    parse_time, format_time, is_station_cancelled,
    calculate_progress, calculate_position_between_stations,
)


def _find_next_station_info(stations: list[dict]) -> tuple[str, str]:
    """Find the next station name and ETA string. Shared by both header variants."""
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

    return next_station, eta


def _format_status(status_msg: str, train_state: str) -> tuple[str, str]:
    """Determine display status text and style. Shared by both header variants."""
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

    return display_status, status_style


def _build_position_bar(
    train: dict, train_state: str, bar_width: int = 20, compact: bool = False,
) -> Text | None:
    """Build the position progress bar between stations. Returns None if unavailable.

    When compact=True, omits the 'Position:' prefix and uses shorter time labels.
    """
    position = calculate_position_between_stations(train)
    if not position or train_state == "Predeparture":
        return None

    last_code, next_code, progress_frac, mins_remaining = position
    filled = int(progress_frac * bar_width)
    empty = bar_width - filled

    bar = f"[green]{'█' * filled}[/][dim]{'░' * empty}[/]"
    if mins_remaining > 0:
        if compact:
            time_str = f"({mins_remaining}m)"
        else:
            time_str = f"({mins_remaining} min)" if mins_remaining != 1 else "(1 min)"
    else:
        time_str = "(arriving)"

    prefix = "" if compact else "Position: "
    return Text.from_markup(f"{prefix}{last_code} {bar} {next_code} [dim]{time_str}[/]")


def build_header(train: dict, last_fetch_time=None, last_error=None, refresh_interval=30) -> Panel:
    """Build the header panel with train info."""
    route_name = train.get("routeName", "Unknown Route")
    train_num = train.get("trainNum", "?")
    train_id = train.get("trainID", "")
    heading = train.get("heading", "")
    velocity = train.get("velocity", 0)
    train_state = train.get("trainState", "")
    status_msg = train.get("statusMsg", "")

    stations = train.get("stations", [])
    next_station, eta = _find_next_station_info(stations)
    display_status, status_style = _format_status(status_msg, train_state)
    dest_name = train.get("destName", "")

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

    position_text = _build_position_bar(train, train_state, bar_width=20)
    if position_text:
        header.add_row(position_text, "")

    if dest_name:
        header.add_row(f"Destination: {dest_name}", "")

    # Build subtitle with status indicator
    if last_fetch_time:
        update_str = f"Updated: {last_fetch_time.strftime('%H:%M:%S')}"
    else:
        update_str = "Updated: —"

    status_parts = [update_str]
    if last_error:
        status_parts.append(f"[yellow]⚠ {last_error}[/]")
    status_parts.append(f"Refresh: {refresh_interval}s")
    status_parts.append("Press Ctrl+C to quit")
    subtitle = " | ".join(status_parts)

    return Panel(
        header,
        title="[bold cyan]Amtrak Status[/]",
        subtitle=f"[dim]{subtitle}[/]",
        border_style="cyan"
    )


def build_compact_train_header(train: dict) -> Panel:
    """Build a more compact header for multi-train view."""
    train_num = train.get("trainNum", "?")
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
    route_name = train.get("routeName", "Unknown Route")
    velocity = train.get("velocity", 0)
    status_msg = train.get("statusMsg", "")

    next_station, eta = _find_next_station_info(stations)
    display_status, status_style = _format_status(status_msg, train_state)

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

    # Position bar (narrower for compact view)
    position_text = _build_position_bar(train, train_state, bar_width=15, compact=True)
    if position_text:
        header.add_row(position_text, "")

    return Panel(header, border_style="cyan")


def apply_main_title(panel: Panel, last_fetch_time=None, last_error=None, refresh_interval=30) -> None:
    """Add the main 'Amtrak Status' title and status subtitle to a panel."""
    panel.title = "[bold cyan]Amtrak Status[/]"
    status_parts = []
    if last_fetch_time:
        status_parts.append(f"Updated: {last_fetch_time.strftime('%H:%M:%S')}")
    else:
        status_parts.append("Updated: —")
    if last_error:
        status_parts.append(f"[yellow]⚠ {last_error}[/]")
    status_parts.append(f"Refresh: {refresh_interval}s")
    status_parts.append("Press Ctrl+C to quit")
    panel.subtitle = f"[dim]{' | '.join(status_parts)}[/]"
