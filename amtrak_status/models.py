"""Pure utility functions for time parsing, station logic, and journey calculations."""

from datetime import datetime


def _now():
    """Current local time. Extracted for test patching."""
    return datetime.now()


def parse_time(time_val: str | None) -> datetime | None:
    """Parse an ISO 8601 time string from the API."""
    if not time_val or not isinstance(time_val, str):
        return None

    try:
        return datetime.fromisoformat(time_val.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def format_time(dt: datetime | None) -> str:
    """Format datetime for display."""
    if not dt:
        return "—"
    return dt.strftime("%I:%M %p").lstrip("0")


def get_status_style(station: dict) -> tuple[str, str]:
    """Get display style based on station status."""
    status = station.get("status", "")

    if status == "Departed":
        return "green", "✓"
    elif status == "Enroute":
        return "yellow bold", "→"
    elif status == "Station":
        return "cyan bold", "●"
    else:
        return "dim", "○"


def is_station_cancelled(station: dict) -> bool:
    """
    Detect if a station stop has been cancelled.

    Cancelled stops typically have:
    - No scheduled arrival AND no scheduled departure times, OR
    - Status that indicates cancellation, OR
    - Status is "Station" but no actual arrival/departure times (train never actually stopped)

    This is a heuristic based on observed API behavior.
    """
    status = station.get("status", "")

    # Check for explicit cancellation indicators in status
    status_lower = status.lower()
    if "cancel" in status_lower or "skip" in status_lower:
        return True

    # If there's no scheduled arrival AND no scheduled departure, likely cancelled
    sch_arr = station.get("schArr")
    sch_dep = station.get("schDep")

    if not sch_arr and not sch_dep:
        return True

    # If status is "Station" (meaning train should be there) but there are no
    # actual arrival/departure times, this stop was likely cancelled.
    # A real "Station" status would have at least an arrival time.
    if status == "Station":
        arr = station.get("arr")
        dep = station.get("dep")
        if not arr and not dep:
            return True

    return False


def find_station_index(stations: list[dict], code: str | None) -> int | None:
    """Find the index of a station by its code (case-insensitive)."""
    if not code:
        return None
    code_upper = code.upper()
    for i, station in enumerate(stations):
        if station.get("code", "").upper() == code_upper:
            return i
    return None


def find_current_station_index(stations: list[dict]) -> int:
    """Find the index of the current/next station (first non-departed, non-cancelled)."""
    for i, station in enumerate(stations):
        if is_station_cancelled(station):
            continue
        status = station.get("status", "")
        if status in ("Enroute", "Station", ""):
            return i
    return len(stations) - 1  # Default to last station if all departed


def filter_stations(stations: list[dict], from_code: str | None, to_code: str | None) -> tuple[list[dict], int, int]:
    """
    Filter stations to show only those between from_code and to_code.
    Returns (filtered_stations, skipped_before, skipped_after).
    """
    if not from_code and not to_code:
        return stations, 0, 0

    from_idx = find_station_index(stations, from_code)
    to_idx = find_station_index(stations, to_code)

    # Default to start/end if not found
    start_idx = from_idx if from_idx is not None else 0
    end_idx = to_idx if to_idx is not None else len(stations) - 1

    # Ensure start <= end
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    skipped_before = start_idx
    skipped_after = len(stations) - end_idx - 1

    return stations[start_idx:end_idx + 1], skipped_before, skipped_after


def calculate_progress(stations: list[dict]) -> tuple[int, int, int]:
    """Calculate journey progress. Returns (completed, current_idx, total).
    Skips cancelled stops in the count."""
    # Count non-cancelled stations
    active_stations = [s for s in stations if not is_station_cancelled(s)]
    total = len(active_stations)
    current_idx = 0
    completed = 0

    for i, station in enumerate(active_stations):
        status = station.get("status", "")
        if status == "Departed":
            completed += 1
            current_idx = i + 1
        elif status in ("Enroute", "Station"):
            current_idx = i
            break

    return completed, current_idx, total


def calculate_position_between_stations(train: dict) -> tuple[str, str, float, int] | None:
    """
    Calculate train's position between the last departed and next station using time.
    Returns (last_station_code, next_station_code, progress_fraction, minutes_remaining)
    or None if position can't be determined.

    Uses scheduled/actual departure from last station and estimated/scheduled arrival
    at next station to calculate progress as a fraction of time elapsed.
    Skips cancelled stops.
    """
    stations = train.get("stations", [])

    # Find last departed station and next station, skipping cancelled stops
    last_departed_idx = -1
    next_station_idx = -1

    for i, station in enumerate(stations):
        # Skip cancelled stops
        if is_station_cancelled(station):
            continue

        status = station.get("status", "")
        if status == "Departed":
            last_departed_idx = i
        elif status in ("Enroute", "Station", "") and next_station_idx == -1:
            next_station_idx = i
            break

    if last_departed_idx == -1 or next_station_idx == -1:
        return None

    last_station = stations[last_departed_idx]
    next_station = stations[next_station_idx]

    last_code = last_station.get("code", "").upper()
    next_code = next_station.get("code", "").upper()

    # Get departure time from last station (actual if available, else scheduled)
    dep_time = parse_time(last_station.get("dep")) or parse_time(last_station.get("schDep"))

    # Get arrival time at next station (estimated if available, else scheduled)
    arr_time = parse_time(next_station.get("arr")) or parse_time(next_station.get("schArr"))

    if not dep_time or not arr_time:
        return None

    # Get current time, matching timezone awareness of parsed times
    if dep_time.tzinfo is not None:
        from datetime import timezone
        now = datetime.now(timezone.utc).astimezone(dep_time.tzinfo)
    else:
        now = _now()

    # Calculate total segment duration and elapsed time
    total_duration = (arr_time - dep_time).total_seconds()
    elapsed = (now - dep_time).total_seconds()

    if total_duration <= 0:
        return (last_code, next_code, 1.0, 0)

    # Calculate progress (0.0 = just departed, 1.0 = arriving)
    progress = max(0.0, min(1.0, elapsed / total_duration))

    # Calculate minutes remaining
    remaining_seconds = max(0, (arr_time - now).total_seconds())
    minutes_remaining = int(remaining_seconds / 60)

    return (last_code, next_code, progress, minutes_remaining)
