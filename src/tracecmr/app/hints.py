"""Hover hints: a small dark card with a title, one line of explanation and the shortcut as keycaps, in place
of Qt's plain yellow-box tooltips.

    set_hint(button, "Smooth", "Smooth the selected contour on this slice.", ["S"])

install() puts one event filter on the application: when Qt would show a tooltip for a widget that has a hint,
the card appears under it instead (above it near the bottom of the screen). Moving straight on to the next
button shows its card at once, like toolbars in modern apps. Widgets without a hint keep their normal tooltip.
"""

from __future__ import annotations

import time

from qtpy.QtCore import QEvent, QObject, QPoint, QRectF, Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

PROP = "tracecmr_hint"
CARD = QColor("#1B1F25")
EDGE = QColor("#3A4048")
KEYCAP = ("QLabel { background: #262B32; border: 1px solid #3A4149; border-bottom: 2px solid #3A4149; "
          "border-radius: 4px; padding: 0px 5px; color: #D5D9DE; font-size: 11px; font-weight: 600; }")


def set_hint(widget: QWidget, title: str | None, body: str = "", keys: list[str] | str | None = None) -> None:
    """Give a widget a hover card. The plain tooltip is kept as text (screen readers, tests)."""
    if isinstance(keys, str):
        keys = [keys]
    widget.setProperty(PROP, {"title": title or "", "body": body or "", "keys": list(keys or [])})
    plain = " — ".join(x for x in (title, body) if x)
    if keys:
        plain += f" ({'+'.join(keys)})"
    widget.setToolTip(plain)


class HintCard(QWidget):
    """The card itself: rounded, dark, drawn by hand (no window shadow or effects, so it appears instantly)."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 10)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(5)
        self.title = QLabel()
        self.title.setStyleSheet("color: #F1F2F4; font-size: 13px; font-weight: 600; background: transparent;")
        head.addWidget(self.title)
        head.addStretch(1)
        head.addSpacing(10)
        self.keys_row = QHBoxLayout()
        self.keys_row.setSpacing(3)
        head.addLayout(self.keys_row)
        lay.addLayout(head)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setStyleSheet("color: #AEB5BE; font-size: 12px; background: transparent;")
        lay.addWidget(self.body)

    def fill(self, hint: dict) -> None:
        self.title.setText(hint["title"])
        self.title.setVisible(bool(hint["title"]))
        self.body.setText(hint["body"])
        self.body.setVisible(bool(hint["body"]))
        while self.keys_row.count():
            w = self.keys_row.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for k in hint["keys"]:
            cap = QLabel(k)
            cap.setStyleSheet(KEYCAP)
            self.keys_row.addWidget(cap)
        if hint["body"]:
            width = 300 if len(hint["body"]) > 60 else 240
            self.body.setFixedWidth(width - 24)
            self.setFixedWidth(width)
        else:  # just the name and its key: a small card that fits them
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self.adjustSize()
            self.setFixedWidth(self.sizeHint().width())
        self.adjustSize()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        p.fillPath(path, CARD)
        p.setPen(QPen(EDGE, 1))
        p.drawPath(path)
        p.end()


class HintFilter(QObject):
    """Shows the card instead of the tooltip for widgets with a hint."""

    QUICK_S = 0.6  # after a card closes, the next one within this time appears without the hover delay
    TIMEOUT_MS = 9000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.card = HintCard()
        self.owner: QWidget | None = None
        self._hidden_at = 0.0
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self.hide)

    def show_for(self, w: QWidget) -> None:
        hint = w.property(PROP)
        if not hint or not w.isVisible():
            return
        self.owner = w
        self.card.fill(hint)
        below = w.mapToGlobal(QPoint(w.width() // 2 - self.card.width() // 2, w.height() + 6))
        screen = w.screen().availableGeometry() if w.screen() is not None else None
        pos = below
        if screen is not None:
            if below.y() + self.card.height() > screen.bottom():  # no room below: above
                pos = w.mapToGlobal(QPoint(w.width() // 2 - self.card.width() // 2, -self.card.height() - 6))
            pos.setX(max(screen.left() + 6, min(screen.right() - self.card.width() - 6, pos.x())))
        self.card.move(pos)
        self.card.show()
        self.card.raise_()
        self._timeout.start(self.TIMEOUT_MS)

    def hide(self) -> None:
        if self.card.isVisible():
            self.card.hide()
            self._hidden_at = time.monotonic()
        self.owner = None

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        et = event.type()
        if not isinstance(obj, QWidget):
            return False
        if et == QEvent.Type.ToolTip and obj.property(PROP):
            self.show_for(obj)
            return True  # no plain tooltip as well
        if et == QEvent.Type.Enter and obj.property(PROP) and time.monotonic() - self._hidden_at < self.QUICK_S:
            self.show_for(obj)  # moving along the toolbar: no second wait
        elif obj is self.owner and et in (QEvent.Type.Leave, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel,
                                          QEvent.Type.Hide, QEvent.Type.KeyPress):
            self.hide()
        return False


_FILTER: HintFilter | None = None


def install() -> HintFilter | None:
    """Turn hover cards on for the whole app (once)."""
    global _FILTER
    app = QApplication.instance()
    if app is None:
        return None
    if _FILTER is None:
        _FILTER = HintFilter(app)
        app.installEventFilter(_FILTER)
    return _FILTER
