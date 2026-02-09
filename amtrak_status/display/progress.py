"""Journey progress bar display."""

from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn

from ..models import calculate_progress


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
