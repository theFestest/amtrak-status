"""Notification system: state tracking and cross-platform system alerts."""

import subprocess
import sys


class NotificationState:
    """Encapsulates all notification tracking state."""

    def __init__(self, stations: set[str] | None = None, notify_all: bool = False):
        self.stations: set[str] = stations or set()  # station codes to notify on
        self.notify_all: bool = notify_all
        self.notified: set[str] = set()  # stations already notified about
        self.initialized: bool = False


def initialize_notification_state(train: dict, state: NotificationState) -> None:
    """
    Capture the initial state of stations so we don't notify
    for arrivals/departures that happened before the script started.

    Marks any station with a non-empty, non-"Enroute" status as already seen.
    This handles edge cases like schedule errors where multiple stations
    might show as "Station" simultaneously.
    """
    if state.initialized:
        return

    stations = train.get("stations", [])
    found_first_future = False

    for station in stations:
        code = station.get("code", "").upper()
        status = station.get("status", "")

        # Mark stations as "seen" if they have any status indicating
        # the train has already been there (Departed, Station, or unknown/error states)
        # Only stations with "Enroute" or empty status are truly "future"
        if status == "Enroute" and not found_first_future:
            # This is the next station - don't mark it, but mark everything before
            found_first_future = True
        elif status in ("Departed", "Station") or (status and status not in ("Enroute", "")):
            # Already visited or has some other status (schedule error, etc.)
            state.notified.add(code)

    state.initialized = True


def send_notification(title: str, message: str) -> bool:
    """
    Send a system notification with fallback to terminal bell.
    Returns True if system notification was sent, False if fell back to bell.
    """
    try:
        if sys.platform == "darwin":
            # macOS
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=True,
                capture_output=True,
                timeout=5
            )
            return True
        elif sys.platform.startswith("linux"):
            # Linux with libnotify
            subprocess.run(
                ["notify-send", "-a", "Amtrak Tracker", title, message],
                check=True,
                capture_output=True,
                timeout=5
            )
            return True
        elif sys.platform == "win32":
            # Windows PowerShell toast
            ps_script = f'''
            Add-Type -AssemblyName System.Windows.Forms
            $balloon = New-Object System.Windows.Forms.NotifyIcon
            $balloon.Icon = [System.Drawing.SystemIcons]::Information
            $balloon.BalloonTipTitle = "{title}"
            $balloon.BalloonTipText = "{message}"
            $balloon.Visible = $true
            $balloon.ShowBalloonTip(5000)
            '''
            subprocess.run(
                ["powershell", "-Command", ps_script],
                check=True,
                capture_output=True,
                timeout=5
            )
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: terminal bell
    print("\a", end="", flush=True)
    return False


def check_and_notify(train: dict, state: NotificationState) -> list[str]:
    """
    Check if train has arrived at any stations we should notify about.
    Returns list of station codes that triggered notifications.

    Only notifies for NEW arrivals/departures since the script started.
    """
    if not state.stations and not state.notify_all:
        return []

    # Initialize state on first call - marks already-departed stations as "seen"
    initialize_notification_state(train, state)

    notified = []
    stations = train.get("stations", [])
    route_name = train.get("routeName", "Train")
    train_num = train.get("trainNum", "")

    for station in stations:
        code = station.get("code", "").upper()
        status = station.get("status", "")
        name = station.get("name", code)

        # Check if train is at or has departed this station
        if status in ("Station", "Departed"):
            # Should we notify for this station?
            should_notify = state.notify_all or code in state.stations

            # Have we already notified?
            if should_notify and code not in state.notified:
                state.notified.add(code)

                if status == "Station":
                    title = f"🚂 {route_name} #{train_num} Arriving"
                    message = f"Now arriving at {name} ({code})"
                else:
                    title = f"🚂 {route_name} #{train_num} Departed"
                    message = f"Departed from {name} ({code})"

                send_notification(title, message)
                notified.append(code)

    return notified
