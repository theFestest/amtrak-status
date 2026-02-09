"""Error and not-found display panels."""

from rich.panel import Panel
from rich.text import Text


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
