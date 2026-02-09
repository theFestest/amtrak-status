"""Connection panel between two trains."""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import format_time
from ..connection import calculate_layover


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
