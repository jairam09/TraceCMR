"""One TraceCMR window at a time.

A second launch (double-clicking the shortcut again, or opening a scan from Explorer/Finder) hands over to the
window that is already open and quits: that window comes to the front, and a scan given to the second launch opens
there. Two windows editing the same mouse would overwrite each other's autosave and offer to "restore" each other's
work.

The running window listens on a local socket named after the user; a new launch tries to connect first. A stale
socket left by a crash is cleared. TRACECMR_ALLOW_SECOND=1 allows a second window anyway (development).
"""

from __future__ import annotations

import getpass
import os
import re
from pathlib import Path
from typing import Callable

PREFIX = "TraceCMR-"


def server_name() -> str:
    try:
        user = getpass.getuser()
    except Exception:
        user = "user"
    return PREFIX + (re.sub(r"[^A-Za-z0-9_-]", "_", user) or "user")


def hand_over(argv: list[str] | None = None, timeout_ms: int = 1000) -> bool:
    """If a TraceCMR window is already open, ask it to come to the front (and open argv[0], if given) and return
    True: this launch should then quit. False when there's none (or a second window is allowed)."""
    if os.environ.get("TRACECMR_ALLOW_SECOND"):
        return False
    from qtpy.QtNetwork import QLocalSocket

    sock = QLocalSocket()
    sock.connectToServer(server_name())
    if not sock.waitForConnected(timeout_ms):
        return False
    msg = f"open\t{Path(argv[0]).expanduser().resolve()}\n" if argv else "raise\n"
    sock.write(msg.encode("utf-8"))
    sock.flush()
    sock.waitForBytesWritten(timeout_ms)
    sock.disconnectFromServer()
    return True


class Listener:
    """In the running window: answers later launches. on_raise() brings the window forward; on_open(path) opens a
    scan. Either may be called before the window exists (a launch during the splash): the callers handle that."""

    def __init__(self, on_raise: Callable[[], None], on_open: Callable[[str], None]):
        from qtpy.QtNetwork import QLocalServer

        self.on_raise, self.on_open = on_raise, on_open
        self.server = QLocalServer()
        name = server_name()
        if not self.server.listen(name):
            QLocalServer.removeServer(name)  # left behind by a crash (a socket file on macOS and Linux)
            self.server.listen(name)
        self.server.newConnection.connect(self._accept)
        self._conns: list = []

    @property
    def listening(self) -> bool:
        return self.server.isListening()

    def _accept(self) -> None:
        while self.server.hasPendingConnections():
            conn = self.server.nextPendingConnection()
            self._conns.append(conn)
            conn.readyRead.connect(lambda c=conn: self._read(c))
            conn.disconnected.connect(lambda c=conn: self._drop(c))
            if conn.bytesAvailable():
                self._read(conn)

    def _drop(self, conn) -> None:
        if conn.bytesAvailable():
            self._read(conn)
        if conn in self._conns:
            self._conns.remove(conn)
        conn.deleteLater()

    def _read(self, conn) -> None:
        data = bytes(conn.readAll()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            line = line.strip()
            if line.startswith("open\t"):
                self.on_raise()
                self.on_open(line.split("\t", 1)[1])
            elif line == "raise":
                self.on_raise()
