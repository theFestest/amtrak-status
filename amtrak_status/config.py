"""Configuration constants and dataclass for amtrak-status."""

from dataclasses import dataclass, field

# API constants
API_BASE = "https://api-v3.amtraker.com/v3"
REFRESH_INTERVAL = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Connection/layover time thresholds (in minutes)
LAYOVER_COMFORTABLE = 60  # 1 hour or more - green
LAYOVER_TIGHT = 45        # 45 minutes - yellow
LAYOVER_RISKY = 30        # 30 minutes or less - red


@dataclass
class Config:
    """Runtime configuration built from CLI arguments."""
    compact_mode: bool = False
    station_from: str | None = None
    station_to: str | None = None
    focus_current: bool = True
    refresh_interval: int = REFRESH_INTERVAL
    notify_stations: set[str] = field(default_factory=set)
    notify_all: bool = False
    connection_station: str | None = None
