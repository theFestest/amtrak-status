"""Single-line compact display mode."""

from datetime import datetime

from rich.text import Text

from ..models import (
    parse_time, format_time, calculate_progress,
    calculate_position_between_stations,
)


def build_compact_display(train: dict, last_fetch_time: datetime | None = None) -> Text:
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
    compact.append(" | ")

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

    if last_fetch_time:
        compact.append(f" | Updated {last_fetch_time.strftime('%H:%M:%S')}", style="dim")

    return compact
