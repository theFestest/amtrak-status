"""Layout-independent harness for driving ``amtrak-status`` end to end.

The contract tests exist to prove that refactoring does not change behaviour, so
they must keep working, unedited, while the code underneath is reorganised.  To
that end this harness never imports or patches anything inside ``amtrak_status``.
It only touches seams that survive any restructuring:

* the ``amtrak-status`` console-script entry point, resolved from the installed
  package metadata (so moving ``main`` just means updating ``pyproject.toml``);
* httpx's transport layer, so *any* ``httpx.Client`` created anywhere is served
  by :class:`FakeAmtrakerAPI`;
* the wall clock (via time-machine) and ``time.sleep`` wherever it was imported;
* ``sys.argv``, ``sys.stdin`` (interactive prompts) and ``sys.stdout``;
* ``subprocess.run`` wherever it was imported (desktop notifications).

Each run re-imports the package from scratch so module-level state cannot leak
between runs, then restores the previously imported modules so the white-box
tests in ``tests/legacy`` are unaffected.
"""

from __future__ import annotations

import contextlib
import copy
import difflib
import importlib
import importlib.metadata
import io
import json
import os
import pkgutil
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import httpx
import rich
import time_machine
from rich.markup import escape
from rich.style import Style
from rich.text import Text

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

API_HOST = "api-v3.amtraker.com"
TIMEZONE = "America/New_York"
# Most fixtures describe train 42 on 2026-02-08; 11:00 EST puts it between
# Altoona (departed 10:35) and Huntingdon (due 11:20).
DEFAULT_NOW = datetime(2026, 2, 8, 11, 0, 0, tzinfo=ZoneInfo(TIMEZONE))

_PACKAGE = "amtrak_status"
_REAL_SLEEP = time.sleep
_REAL_SUBPROCESS_RUN = subprocess.run


# =============================================================================
# Fake Amtraker API
# =============================================================================


@dataclass(frozen=True)
class HTTPStatus:
    """Respond with a non-2xx status code."""

    code: int
    body: str = ""


@dataclass(frozen=True)
class NetworkError:
    """Raise an httpx transport error instead of responding."""

    message: str = "Connection refused"


@dataclass(frozen=True)
class RawBody:
    """Respond 200 with a non-JSON body (e.g. an HTML error page)."""

    body: str
    content_type: str = "text/html"


class FakeAmtrakerAPI:
    """In-memory stand-in for https://api-v3.amtraker.com/v3.

    Responses are queued per path; the last queued response repeats forever,
    which models a feed that stops changing.  Unregistered train and station
    paths return ``[]`` exactly as the real server does for unknown IDs
    (see ``index.ts`` in github.com/piemadd/amtraker-v3).
    """

    def __init__(self) -> None:
        self._routes: dict[str, list[Any]] = {}
        self.requests: list[str] = []

    def train(self, train_id: str, *responses: Any) -> FakeAmtrakerAPI:
        self._routes[f"/v3/trains/{train_id}"] = list(responses)
        return self

    def station(self, code: str, *responses: Any) -> FakeAmtrakerAPI:
        self._routes[f"/v3/stations/{code}"] = list(responses)
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(path)
        if request.url.host != API_HOST:
            return httpx.Response(599, text=f"unexpected host {request.url.host}")
        queue = self._routes.get(path)
        if queue is None:
            response: Any = [] if path.startswith(("/v3/trains/", "/v3/stations/")) else HTTPStatus(404, "Not found")
        elif len(queue) > 1:
            response = queue.pop(0)
        else:
            response = queue[0]

        if isinstance(response, NetworkError):
            raise httpx.ConnectError(response.message, request=request)
        if isinstance(response, HTTPStatus):
            return httpx.Response(response.code, text=response.body, request=request)
        if isinstance(response, RawBody):
            return httpx.Response(
                200, text=response.body, headers={"content-type": response.content_type}, request=request
            )
        return httpx.Response(200, json=response, request=request)


# =============================================================================
# Fixture helpers
# =============================================================================


def load_fixture(name: str) -> Any:
    """Load a JSON fixture; returns a fresh deep copy each time."""
    return json.loads((FIXTURES_DIR / name).read_text())


def train_payload(fixture: str, key: str | None = None, **overrides: Any) -> dict:
    """Load a single-train fixture, optionally re-keying it or overriding fields."""
    data = load_fixture(fixture)
    (orig_key, trains), = data.items()
    trains = copy.deepcopy(trains)
    for t in trains:
        t.update(overrides)
    return {key or orig_key: trains}


# =============================================================================
# Running the CLI
# =============================================================================


@dataclass
class Notification:
    argv: list[str]


@dataclass
class RunResult:
    stdout: str
    exit_code: int | None
    requests: list[str]
    sleeps: list[float]
    notifications: list[Notification]
    exception: BaseException | None = None

    @property
    def plain(self) -> str:
        """stdout with ANSI styling removed."""
        return Text.from_ansi(self.stdout).plain


def _package_modules() -> dict[str, Any]:
    return {k: v for k, v in sys.modules.items() if k == _PACKAGE or k.startswith(_PACKAGE + ".")}


@contextlib.contextmanager
def _fresh_package_import():
    saved = _package_modules()
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in _package_modules():
            del sys.modules[name]
        sys.modules.update(saved)


def _load_entry_point() -> Callable[[], Any]:
    eps = importlib.metadata.entry_points(group="console_scripts")
    (ep,) = [e for e in eps if e.name == "amtrak-status"]
    main = ep.load()
    # Import every submodule now, so aliases such as ``from time import sleep`` in modules that
    # the app would only import lazily are patched too (never really sleep or notify).
    package = importlib.import_module(_PACKAGE)
    for info in pkgutil.walk_packages(package.__path__, prefix=_PACKAGE + "."):
        importlib.import_module(info.name)
    return main


def _replace_everywhere(stack: contextlib.ExitStack, module: Any, attr: str, original: Any, replacement: Any) -> None:
    """Patch ``module.attr`` and every alias of ``original`` inside the package."""
    stack.enter_context(_setattr(module, attr, replacement))
    for mod in list(_package_modules().values()):
        for name, value in list(vars(mod).items()):
            if value is original:
                stack.enter_context(_setattr(mod, name, replacement))


@contextlib.contextmanager
def _setattr(obj: Any, name: str, value: Any):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def _environ(**env: str | None):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if hasattr(time, "tzset"):
            time.tzset()


@dataclass
class App:
    """Runs the real CLI against a :class:`FakeAmtrakerAPI` with a frozen clock."""

    api: FakeAmtrakerAPI = field(default_factory=FakeAmtrakerAPI)
    now: datetime = DEFAULT_NOW
    width: int = 120
    height: int = 60
    color: bool = True
    platform: str = "linux"

    def run(
        self,
        *argv: str,
        stdin: str = "",
        polls: int | None = None,
        poll_interval: float | None = None,
        notify_ok: bool = True,
    ) -> RunResult:
        """Run ``amtrak-status *argv``.

        ``polls``: for refreshing modes, stop (as if Ctrl+C) when the refresh
        loop goes to sleep for the ``polls``-th time.  Refresh sleeps are those
        lasting ``poll_interval`` seconds (default: the ``-r`` value or 30).
        Every sleep advances the frozen clock by its duration.
        """
        interval = poll_interval if poll_interval is not None else _refresh_from_argv(argv)
        sleeps: list[float] = []
        notifications: list[Notification] = []
        out = io.StringIO()
        exit_code: int | None = 0
        exception: BaseException | None = None

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                _environ(
                    TZ=TIMEZONE,
                    COLUMNS=str(self.width),
                    LINES=str(self.height),
                    FORCE_COLOR="1" if self.color else None,
                    NO_COLOR=None,
                    TERM="xterm-256color",
                    COLORTERM=None,
                    TTY_COMPATIBLE=None,
                    TTY_INTERACTIVE=None,
                )
            )
            traveller = stack.enter_context(time_machine.travel(self.now, tick=False))
            # rich.prompt falls back to Rich's process-wide console; make it pick up this run's env.
            stack.enter_context(_setattr(rich, "_console", None))
            stack.enter_context(_fresh_package_import())
            main = _load_entry_point()

            refresh_sleeps = 0

            def fake_sleep(seconds: float) -> None:
                nonlocal refresh_sleeps
                sleeps.append(seconds)
                if seconds < 0:
                    _REAL_SLEEP(seconds)  # preserve the real ValueError
                traveller.shift(timedelta(seconds=seconds))
                if polls is not None and seconds == interval:
                    refresh_sleeps += 1
                    if refresh_sleeps >= polls:
                        raise KeyboardInterrupt

            def fake_run(cmd, *args, **kwargs):
                notifications.append(Notification(argv=list(cmd)))
                if not notify_ok:
                    raise FileNotFoundError(cmd[0])
                return subprocess.CompletedProcess(cmd, 0, b"", b"")

            _replace_everywhere(stack, time, "sleep", _REAL_SLEEP, fake_sleep)
            _replace_everywhere(stack, subprocess, "run", _REAL_SUBPROCESS_RUN, fake_run)
            stack.enter_context(_setattr(httpx.HTTPTransport, "handle_request", lambda _self, req: self.api.handle(req)))
            stack.enter_context(_setattr(sys, "platform", self.platform))
            stack.enter_context(_setattr(sys, "argv", ["amtrak-status", *argv]))
            stack.enter_context(_setattr(sys, "stdin", io.StringIO(stdin)))
            stack.enter_context(contextlib.redirect_stdout(out))

            try:
                returned = main()  # console scripts do sys.exit(main()), so honour an int return too
                if isinstance(returned, int):
                    exit_code = returned
            except SystemExit as e:
                exit_code = 0 if e.code is None else e.code if isinstance(e.code, int) else 1
            except KeyboardInterrupt:
                exit_code = 130
            except Exception as e:  # noqa: BLE001 - surfaced to tests via RunResult
                exception = e
                exit_code = None

        return RunResult(
            stdout=out.getvalue(),
            exit_code=exit_code,
            requests=list(self.api.requests),
            sleeps=sleeps,
            notifications=notifications,
            exception=exception,
        )


def _refresh_from_argv(argv: Iterable[str]) -> float:
    args = list(argv)
    for i, a in enumerate(args):
        if a in ("-r", "--refresh") and i + 1 < len(args):
            return float(args[i + 1])
        if a.startswith("--refresh="):
            return float(a.split("=", 1)[1])
    return 30.0


# =============================================================================
# Golden files
# =============================================================================

_STANDARD_COLOR_NAMES = {
    f"color({i})": name
    for i, name in enumerate(
        ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
        + [f"bright_{n}" for n in ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]]
    )
}
_COLOR_TOKEN_RE = re.compile(r"color\(\d+\)")


def _style_name(style: Style) -> str:
    name = _COLOR_TOKEN_RE.sub(lambda m: _STANDARD_COLOR_NAMES.get(m.group(0), m.group(0)), str(style))
    return "" if name == "none" else name


def _merge_runs(runs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fold unstyled whitespace into a run when the styles on both sides match."""
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(runs):
        name, txt = runs[i]
        if (
            merged
            and not name
            and txt.isspace()
            and i + 1 < len(runs)
            and runs[i + 1][0]
            and runs[i + 1][0] == merged[-1][0]
        ):
            merged[-1] = (merged[-1][0], merged[-1][1] + txt + runs[i + 1][1])
            i += 2
            continue
        if merged and merged[-1][0] == name:
            merged[-1] = (name, merged[-1][1] + txt)
        else:
            merged.append((name, txt))
        i += 1
    return merged


def _visible_on_whitespace(style: Style) -> bool:
    return bool(style.bgcolor or style.underline or style.strike or style.reverse)


def normalize(stdout: str) -> str:
    """Turn captured ANSI output into stable, reviewable text with style runs.

    Each run of identically-styled characters is written as ``[style]text[/]``
    (e.g. ``[green]✓[/] Pittsburgh``) so colour/emphasis regressions show up in
    golden diffs.  Whitespace carries no style unless the style is visible on
    whitespace (background, underline, strike, reverse).  Trailing whitespace
    and trailing blank lines are dropped, and runs of 3+ identical lines (the
    blank filler a full-screen Layout pads itself with) are collapsed into one
    line plus a ``(×N)`` count.
    """
    lines: list[str] = []
    for line in Text.from_ansi(stdout).split("\n", allow_blank=True):
        plain = line.plain
        styles: list[Style] = [Style.null()] * len(plain)
        for span in line.spans:
            st = span.style if isinstance(span.style, Style) else Style.parse(span.style)
            for i in range(span.start, min(span.end, len(plain))):
                styles[i] = styles[i] + st
        runs: list[tuple[str, str]] = []  # (style name, text)
        for ch, st in zip(plain, styles):
            name = "" if (ch.isspace() and not _visible_on_whitespace(st)) else _style_name(st)
            if runs and runs[-1][0] == name:
                runs[-1] = (name, runs[-1][1] + ch)
            else:
                runs.append((name, ch))
        rendered = "".join(f"[{name}]{escape(txt)}[/]" if name else escape(txt) for name, txt in _merge_runs(runs))
        lines.append(rendered.rstrip())
    while lines and not lines[-1]:
        lines.pop()

    collapsed: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        count = j - i + 1
        collapsed.append(f"{lines[i]}  (×{count})" if count >= 3 else lines[i])
        if count == 2:
            collapsed.append(lines[i])
        i = j + 1
    return "\n".join(collapsed) + "\n"


def golden_document(argv: Iterable[str], result: RunResult, stdin: str = "") -> str:
    """Everything observable about a run: side effects first, then the screen."""
    header = [
        f"# argv: amtrak-status {' '.join(argv)}",
        *([f"# stdin: {stdin!r}"] if stdin else []),
        f"# exit: {result.exit_code}"
        + (f" ({type(result.exception).__name__}: {result.exception})" if result.exception else ""),
        f"# requests: {', '.join(result.requests) or '-'}",
        f"# sleeps: {', '.join(f'{s:g}' for s in result.sleeps) or '-'}",
        *[f"# notification: {n.argv}" for n in result.notifications],
        "# " + "-" * 60,
    ]
    return "\n".join(header) + "\n" + normalize(result.stdout)


def assert_golden(name: str, document: str) -> None:
    """Compare ``document`` against ``golden/<name>.txt``.

    Set ``UPDATE_GOLDEN=1`` to (re)write golden files; review the diff before
    committing.  Refactoring PRs must not change any golden file.
    """
    actual = document
    path = GOLDEN_DIR / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    assert path.exists(), f"missing golden file {path.name}; run with UPDATE_GOLDEN=1 to create it"
    expected = path.read_text(encoding="utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{path.name}",
                tofile="actual",
            )
        )
        raise AssertionError(f"output differs from golden/{path.name} (UPDATE_GOLDEN=1 to accept):\n{diff}")
