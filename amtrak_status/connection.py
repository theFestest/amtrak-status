"""Multi-train connection logic: overlapping stations, layover calculation."""

from .config import LAYOVER_COMFORTABLE, LAYOVER_TIGHT
from .models import parse_time


def find_overlapping_stations(train1: dict, train2: dict) -> list[str]:
    """
    Find station codes that appear in both trains' routes.
    Returns list of station codes in the order they appear on train1's route.
    """
    stations1 = {s.get("code", "").upper() for s in train1.get("stations", [])}
    stations2 = {s.get("code", "").upper() for s in train2.get("stations", [])}

    overlap = stations1 & stations2

    # Return in order of train1's route
    return [
        s.get("code", "").upper()
        for s in train1.get("stations", [])
        if s.get("code", "").upper() in overlap
    ]


def get_station_times(train: dict, station_code: str) -> tuple:
    """
    Get scheduled and actual/estimated times for a station.
    Returns (sch_arr, sch_dep, actual_arr, actual_dep).
    """
    station_code = station_code.upper()
    for station in train.get("stations", []):
        if station.get("code", "").upper() == station_code:
            sch_arr = parse_time(station.get("schArr"))
            sch_dep = parse_time(station.get("schDep"))
            arr = parse_time(station.get("arr"))
            dep = parse_time(station.get("dep"))
            return sch_arr, sch_dep, arr, dep
    return None, None, None, None


def get_station_status(train: dict, station_code: str) -> str:
    """Get the status of a specific station on a train's route."""
    station_code = station_code.upper()
    for station in train.get("stations", []):
        if station.get("code", "").upper() == station_code:
            return station.get("status", "")
    return ""


def calculate_layover(train1: dict, train2: dict, connection_station: str) -> dict:
    """
    Calculate layover information between two trains at a connection station.

    Returns dict with:
        - station_code: The connection station code
        - station_name: The connection station name
        - train1_arrives: Arrival time of first train (actual/estimated or scheduled)
        - train2_departs: Departure time of second train (actual/estimated or scheduled)
        - layover_minutes: Minutes between arrival and departure
        - layover_status: 'comfortable', 'tight', 'risky', or 'missed'
        - train1_status: Status of train1 at connection station
        - train2_status: Status of train2 at connection station
        - is_valid: Whether connection is still possible
    """
    connection_station = connection_station.upper()

    # Get station name
    station_name = connection_station
    for station in train1.get("stations", []):
        if station.get("code", "").upper() == connection_station:
            station_name = station.get("name", connection_station)
            break

    # Get times for train1 arrival at connection
    sch_arr1, _, arr1, _ = get_station_times(train1, connection_station)
    train1_arrives = arr1 or sch_arr1

    # Get times for train2 departure from connection
    _, sch_dep2, _, dep2 = get_station_times(train2, connection_station)
    train2_departs = dep2 or sch_dep2

    # Get statuses
    train1_status = get_station_status(train1, connection_station)
    train2_status = get_station_status(train2, connection_station)

    result = {
        "station_code": connection_station,
        "station_name": station_name,
        "train1_arrives": train1_arrives,
        "train2_departs": train2_departs,
        "layover_minutes": None,
        "layover_status": "unknown",
        "train1_status": train1_status,
        "train2_status": train2_status,
        "is_valid": True,
    }

    if train1_arrives and train2_departs:
        # Handle timezone-aware comparison
        if train1_arrives.tzinfo != train2_departs.tzinfo:
            # Convert to same timezone for comparison
            if train2_departs.tzinfo is not None and train1_arrives.tzinfo is None:
                from datetime import timezone
                train1_arrives = train1_arrives.replace(tzinfo=timezone.utc)
            elif train1_arrives.tzinfo is not None and train2_departs.tzinfo is None:
                from datetime import timezone
                train2_departs = train2_departs.replace(tzinfo=timezone.utc)

        layover_seconds = (train2_departs - train1_arrives).total_seconds()
        layover_minutes = int(layover_seconds / 60)
        result["layover_minutes"] = layover_minutes

        if layover_minutes < 0:
            result["layover_status"] = "missed"
            result["is_valid"] = False
        elif layover_minutes >= LAYOVER_COMFORTABLE:
            result["layover_status"] = "comfortable"
        elif layover_minutes >= LAYOVER_TIGHT:
            result["layover_status"] = "tight"
        else:
            result["layover_status"] = "risky"

    # Check if train2 has already departed
    if train2_status == "Departed":
        # Check if train1 has arrived
        if train1_status not in ("Departed", "Station"):
            result["layover_status"] = "missed"
            result["is_valid"] = False

    return result
