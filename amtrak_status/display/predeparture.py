"""Predeparture display panels."""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
