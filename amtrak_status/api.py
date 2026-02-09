"""API communication, caching, and retry logic for the Amtraker API."""

from datetime import datetime
from time import sleep
from typing import Any

import httpx

from .config import API_BASE, MAX_RETRIES, RETRY_DELAY
from .models import _now


class TrainCache:
    """Encapsulates all caching state for API responses."""

    def __init__(self):
        self.last_successful_data: dict[str, Any] | None = None
        self.last_fetch_time: datetime | None = None
        self.last_error: str | None = None
        self.per_train: dict[str, dict[str, Any]] = {}


def fetch_train_data(train_number: str, cache: TrainCache) -> dict[str, Any] | None:
    """Fetch train data from the Amtraker API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{API_BASE}/trains/{train_number}")
                response.raise_for_status()
                data = response.json()

                # The API returns a dict with train number as key
                # Try multiple key formats since API can be inconsistent
                result = None
                for key in [train_number, str(train_number), train_number.lstrip("0")]:
                    if key in data and data[key]:
                        # May have multiple trains with same number (different days)
                        # Return the most recent one
                        result = data[key][0]
                        break

                # Also check if response has any data at all (single-key response)
                if result is None and len(data) == 1:
                    key = list(data.keys())[0]
                    if data[key]:
                        result = data[key][0]

                if result:
                    # Update single-train cache
                    cache.last_successful_data = result
                    cache.last_fetch_time = _now()
                    cache.last_error = None

                    # Also update per-train cache
                    if train_number not in cache.per_train:
                        cache.per_train[train_number] = {}
                    cache.per_train[train_number]["data"] = result
                    cache.per_train[train_number]["fetch_time"] = _now()
                    cache.per_train[train_number]["error"] = None

                    return result

                # Train not in response - could be API flakiness or genuinely not found
                # Use per-train cache if available
                if train_number in cache.per_train:
                    pt = cache.per_train[train_number]
                    if pt.get("data") and pt.get("fetch_time"):
                        age = (_now() - pt["fetch_time"]).total_seconds()
                        if age < 300:  # Use cache for up to 5 minutes
                            cache.last_error = "Train not in API response (using cached data)"
                            return pt["data"]

                return None

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY * (attempt + 1))
                continue

            # All retries failed - use per-train cache if available
            if train_number in cache.per_train:
                pt = cache.per_train[train_number]
                if pt.get("data") and pt.get("fetch_time"):
                    age = (_now() - pt["fetch_time"]).total_seconds()
                    if age < 300:
                        cache.last_error = f"{error_msg} (using cached data)"
                        return pt["data"]

            cache.last_error = error_msg
            return {"error": error_msg}

        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY * (attempt + 1))
                continue

            # All retries failed - use per-train cache if available
            if train_number in cache.per_train:
                pt = cache.per_train[train_number]
                if pt.get("data") and pt.get("fetch_time"):
                    age = (_now() - pt["fetch_time"]).total_seconds()
                    if age < 300:
                        cache.last_error = f"{str(e)} (using cached data)"
                        return pt["data"]

            cache.last_error = str(e)
            return {"error": str(e)}

    return None


def fetch_train_data_cached(train_number: str, cache: TrainCache) -> dict[str, Any] | None:
    """Fetch train data with per-train caching for multi-train mode."""
    if train_number not in cache.per_train:
        cache.per_train[train_number] = {"data": None, "fetch_time": None, "error": None}

    pt = cache.per_train[train_number]

    # Try to fetch new data
    result = fetch_train_data(train_number, cache)

    if result and "error" not in result:
        pt["data"] = result
        pt["fetch_time"] = _now()
        pt["error"] = None
    elif pt["data"] and pt["fetch_time"]:
        # Use cached data if fetch failed
        age = (_now() - pt["fetch_time"]).total_seconds()
        if age < 300:  # 5 minute cache
            pt["error"] = "Using cached data"
            return pt["data"]

    return result


def fetch_station_schedule(station_code: str) -> dict[str, Any] | None:
    """
    Fetch schedule data for a station from the Amtraker API.
    Returns dict with upcoming trains at this station.
    """
    station_code = station_code.upper()

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/stations/{station_code}")
            response.raise_for_status()
            data = response.json()
            return data
    except httpx.HTTPError as e:
        return {"error": str(e)}


def get_train_schedule_from_station(station_data: dict, train_number: str) -> dict | None:
    """
    Extract schedule info for a specific train from station data.
    Returns a dict with arrival/departure times if found.
    """
    if not station_data or "error" in station_data:
        return None

    train_number = str(train_number)

    # Station data is keyed by station code, with a list of trains
    for station_code, trains in station_data.items():
        if not isinstance(trains, list):
            continue
        for train in trains:
            train_num = str(train.get("trainNum", ""))
            if train_num == train_number or train_num.split("-")[0] == train_number:
                return {
                    "trainNum": train_num,
                    "schArr": train.get("schArr"),
                    "schDep": train.get("schDep"),
                    "arr": train.get("arr"),
                    "dep": train.get("dep"),
                    "status": train.get("status", ""),
                }

    return None


def build_predeparture_train_data(train_number: str, station_code: str, station_schedule: dict | None) -> dict:
    """
    Build a minimal train data dict for a predeparture train using station schedule info.
    This allows us to show connection times even when the train isn't active yet.
    """
    data = {
        "trainNum": train_number,
        "routeName": f"Train {train_number}",
        "trainState": "Predeparture",
        "stations": [],
        "_predeparture": True,  # Flag to indicate this is synthetic data
    }

    if station_schedule:
        # Add the connection station with schedule times
        data["stations"].append({
            "code": station_code,
            "name": station_code,  # We don't have the full name
            "schArr": station_schedule.get("schArr"),
            "schDep": station_schedule.get("schDep"),
            "arr": station_schedule.get("arr"),
            "dep": station_schedule.get("dep"),
            "status": "",
        })

    return data
