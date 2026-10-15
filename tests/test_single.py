"""One TraceCMR window at a time: a second launch hands over to the open window and quits."""

import os
import uuid

import pytest

pytest.importorskip("qtpy.QtNetwork")
if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    pytest.skip("needs a display (run under xvfb-run)", allow_module_level=True)

from tracecmr.app import single  # noqa: E402


@pytest.fixture
def name(monkeypatch):
    n = f"TraceCMR-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(single, "server_name", lambda: n)
    monkeypatch.delenv("TRACECMR_ALLOW_SECOND", raising=False)
    return n


def test_second_launch_hands_over_and_opens_the_scan(qtbot, name, tmp_path):
    assert not single.hand_over([])  # nothing open yet: this launch carries on
    raised, opened = [], []
    listener = single.Listener(lambda: raised.append(1), opened.append)
    assert listener.listening
    assert single.hand_over([])  # a window is open: hand over, quit
    qtbot.waitUntil(lambda: bool(raised), timeout=3000)
    scan = tmp_path / "Mouse1.h5"
    assert single.hand_over([str(scan)])
    qtbot.waitUntil(lambda: bool(opened), timeout=3000)
    assert opened == [str(scan.resolve())] and len(raised) == 2


def test_stale_socket_is_cleared_and_second_window_can_be_allowed(qtbot, name, monkeypatch):
    first = single.Listener(lambda: None, lambda p: None)
    first.server.close()  # like a crash: nobody answers any more
    second = single.Listener(lambda: None, lambda p: None)
    assert second.listening
    monkeypatch.setenv("TRACECMR_ALLOW_SECOND", "1")
    assert not single.hand_over([])
