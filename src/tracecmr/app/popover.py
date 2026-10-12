"""A small floating card for quick settings (e.g. brightness and contrast): rounded and dark like the menus and
hover cards, a title you can drag it by, a × to close (or Esc). It floats over the image without blocking it."""

from __future__ import annotations

from qtpy.QtCore import QPoint, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

STYLE = """
QLabel { color: #D5D9DE; font-size: 13px; background: transparent; }
QLabel#title { color: #F1F2F4; font-size: 14px; font-weight: 600; }
QLabel#value { color: #9AA1A9; font-size: 12px; }
QLabel#note { color: #8F969F; font-size: 12px; }
QLabel#keycap { background: #262B32; border: 1px solid #3A4149; border-bottom: 2px solid #3A4149;
                border-radius: 4px; padding: 0px 5px; color: #D5D9DE; font-size: 11px; font-weight: 600; }
QPushButton { background: #2A3038; color: #E4E7EB; border: none; border-radius: 7px; padding: 6px 12px;
              font-size: 12px; font-weight: 600; }
QPushButton:hover { background: #343B44; }
QPushButton#close { background: transparent; color: #8F969F; font-size: 16px; padding: 0px 6px; }
QPushButton#close:hover { color: #F1F2F4; }
QSlider::groove:horizontal { height: 4px; background: #343A42; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #6B7684; border-radius: 2px; }
QSlider::handle:horizontal { background: #E4E7EB; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #FFFFFF; }
"""


def keycap(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("keycap")
    return lbl


class Popover(QWidget):
    def __init__(self, parent: QWidget, title: str):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(STYLE)
        self._drag: QPoint | None = None
        self._placed = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 12, 14)
        outer.setSpacing(10)
        head = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setObjectName("title")
        head.addWidget(self.title)
        head.addStretch(1)
        close = QPushButton("×")
        close.setObjectName("close")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.clicked.connect(self.hide)
        head.addWidget(close)
        outer.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body)

    def show_near(self, anchor: QWidget, offset: QPoint = QPoint(16, 16)) -> None:
        """Show at the top-left corner of `anchor` (the image area) the first time; afterwards where it was left."""
        self.adjustSize()
        if not self._placed and anchor is not None:
            self.move(anchor.mapToGlobal(offset))
            self._placed = True
        self.show()
        self.raise_()

    # drawing and dragging
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        p.fillPath(path, QColor("#1B1F25"))
        p.setPen(QPen(QColor("#3A4048"), 1))
        p.drawPath(path)
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 40:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            self.move(event.globalPosition().toPoint() - self._drag)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)
