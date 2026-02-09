"""Display rendering components for amtrak-status."""

from .header import build_header, build_compact_train_header, apply_main_title
from .stations import build_stations_table
from .progress import build_progress_bar
from .compact import build_compact_display
from .connection_display import build_connection_panel
from .errors import build_error_panel, build_not_found_panel
from .predeparture import build_predeparture_panel, build_predeparture_header

__all__ = [
    "build_header",
    "build_compact_train_header",
    "apply_main_title",
    "build_stations_table",
    "build_progress_bar",
    "build_compact_display",
    "build_connection_panel",
    "build_error_panel",
    "build_not_found_panel",
    "build_predeparture_panel",
    "build_predeparture_header",
]
