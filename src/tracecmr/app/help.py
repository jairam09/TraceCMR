"""Help: the keyboard shortcuts sheet (press ?) and the short tour shown the first time the app opens.

Both are plain Qt drawn over the main window, in the app's dark theme, so they look like part of TraceCMR rather
than a separate help system.
"""

from __future__ import annotations

from typing import Callable

from qtpy.QtCore import QEasingCurve, QEvent, QPoint, QPropertyAnimation, QRect, QRectF, Qt, QVariantAnimation
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

ACCENT = "#D0646F"  # the logo's red
INK = "#F1F2F4"
MUTED = "#9AA1A9"
CARD = "#1B1F25"
EDGE = "#353B43"

# Every keyboard shortcut, grouped. The sheet is drawn from this list, so it stays the one place to update.
SHORTCUTS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Contours", [
        ("Q", "Switch Endo → Epi → RV"),
        ("S", "Smooth the selected contour"),
        ("P", "Show or hide the points"),
        ("Ctrl+Z", "Undo"),
        ("Ctrl+Y", "Redo"),
        ("]", "Grow the contour half a pixel"),
        ("[", "Shrink the contour half a pixel"),
        ("Delete", "Clear the contour on this slice"),
        ("Esc", "Cancel a click action"),
        ("Click line", "Add a point"),
        ("Right-click", "Delete a point"),
    ]),
    ("View", [
        ("H", "Hide or show the contours"),
        ("G", "Smooth or pixel image"),
        ("W", "Wall lines"),
        ("R", "Regions of interest and ruler"),
        ("Ctrl+E", "ED | ES side by side"),
        ("F", "Center on the heart"),
        ("+  /  −", "Zoom in / out"),
        ("Ctrl+0", "Center on the heart"),
        ("Space", "Pan mode"),
        ("Right-drag", "Pan"),
        ("Ctrl+drag", "Brightness and contrast"),
    ]),
    ("Move around", [
        ("↑  /  ↓", "Previous / next slice"),
        ("←  /  →", "Previous / next frame"),
        ("K", "Play or pause the cine"),
    ]),
    ("File and panels", [
        ("Ctrl+S", "Save"),
        ("Ctrl+Shift+S", "Save as"),
        ("Ctrl+1 / 2 / 3", "Results: Function, Wall, Regions"),
        ("Ctrl+Shift+F", "Focus on the scan"),
        ("Ctrl+Shift+R", "Show all panels"),
        ("?", "This list"),
    ]),
]

KEYCAP_STYLE = (f"QLabel {{ background: #262B32; border: 1px solid #3C434C; border-bottom: 2px solid #3C434C; "
                f"border-radius: 5px; padding: 1px 7px; color: {INK}; font-size: 12px; font-weight: 600; }}")


def _label(text: str, size: int, color: str, weight: int = 400, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: {weight}; background: transparent;")
    lbl.setWordWrap(wrap)
    return lbl


def _shadow(widget: QWidget) -> None:
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(36)
    fx.setOffset(0, 10)
    fx.setColor(QColor(0, 0, 0, 150))
    widget.setGraphicsEffect(fx)


BUTTON_STYLE = f"""
QPushButton {{ border: none; border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 600; }}
QPushButton#primary {{ background: {INK}; color: #12151A; }}
QPushButton#primary:hover {{ background: #FFFFFF; }}
QPushButton#secondary {{ background: #2A3038; color: {INK}; }}
QPushButton#secondary:hover {{ background: #343B44; }}
QPushButton#quiet {{ background: transparent; color: {MUTED}; padding: 7px 4px; font-weight: 500; }}
QPushButton#quiet:hover {{ color: {INK}; }}
"""


def _button(text: str, kind: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(kind)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keys go to the tour / sheet, not the buttons
    return b


class ShortcutSheet(QDialog):
    """Every shortcut on one card, centred over the window. Esc, ? or a click outside closes it."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("tracecmr_shortcuts")
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 36)  # room for the shadow
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(f"QFrame#card {{ background: {CARD}; border: 1px solid {EDGE}; border-radius: 14px; }}"
                           + BUTTON_STYLE)
        _shadow(card)
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(26, 22, 26, 22)
        lay.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(_label("Keyboard shortcuts", 17, INK, 600))
        head.addStretch(1)
        close = _button("Close", "quiet")
        close.clicked.connect(self.close)
        head.addWidget(close)
        lay.addLayout(head)
        cols = QHBoxLayout()
        cols.setSpacing(36)
        columns = [SHORTCUTS[:2], SHORTCUTS[2:]]
        for groups in columns:
            grid = QGridLayout()  # one grid per column, so the descriptions line up across its groups
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(6)
            r = 0
            for title, rows in groups:
                cap = _label(title.upper(), 11, MUTED, 600)
                cap.setStyleSheet(cap.styleSheet() + f" letter-spacing: 1px; padding-top: {10 if r else 4}px;")
                grid.addWidget(cap, r, 0, 1, 2)
                r += 1
                for keys, what in rows:
                    key = QLabel(keys)
                    key.setStyleSheet(KEYCAP_STYLE)
                    grid.addWidget(key, r, 0, Qt.AlignmentFlag.AlignLeft)
                    grid.addWidget(_label(what, 13, "#C9CED4"), r, 1)
                    r += 1
            grid.setColumnStretch(1, 1)
            grid.setRowStretch(r, 1)
            cols.addLayout(grid)
        lay.addLayout(cols)
        lay.addWidget(_label("Hover over any button to see what it does.", 12, MUTED))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Question, Qt.Key.Key_F1):
            self.close()
            return
        super().keyPressEvent(event)

    def popup(self) -> None:
        """Show centred over the parent window."""
        self.adjustSize()
        win = self.parentWidget().window() if self.parentWidget() is not None else None
        if win is not None:
            centre = win.mapToGlobal(win.rect().center())
            self.move(centre - QPoint(self.width() // 2, self.height() // 2))
        self.show()
        self.raise_()
        self.activateWindow()


class TourStep:
    """One stop on the tour: what to point at (a function returning widgets, so docks found later still work),
    a title and one or two sentences. No target = a card in the middle of the window."""

    def __init__(self, title: str, body: str, target: Callable[[], list] | None = None):
        self.title, self.body, self.target = title, body, target


class Tour(QWidget):
    """A spotlight tour: the window dims, a rounded window opens around one part of the interface, and a card
    beside it explains it. Next / Back / Skip, or the arrow keys, Enter and Esc. Drawn as a child of the main
    window, so it moves and resizes with it."""

    HOLE_PAD = 6
    MARGIN = 16

    def __init__(self, window: QWidget, steps: list[TourStep], on_finish: Callable[[bool], None] | None = None,
                 logo=None):
        super().__init__(window)
        self.setObjectName("tracecmr_tour")
        self.window_ = window
        self.steps = steps
        self.on_finish = on_finish
        self.index = -1
        self._hole = QRectF()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setGeometry(window.rect())
        window.installEventFilter(self)

        self.card = QFrame(self)
        self.card.setObjectName("card")
        self.card.setStyleSheet(f"QFrame#card {{ background: {CARD}; border: 1px solid {EDGE}; border-radius: 14px; }}"
                                + BUTTON_STYLE)
        _shadow(self.card)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(8)
        self.logo = QLabel()
        self.logo.setStyleSheet("background: transparent;")
        if logo is not None:
            self.logo.setPixmap(logo)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.logo.setContentsMargins(0, 4, 0, 8)
        lay.addWidget(self.logo)
        self.lbl_step = _label("", 11, ACCENT, 600)
        self.lbl_step.setStyleSheet(self.lbl_step.styleSheet() + " letter-spacing: 1px;")
        lay.addWidget(self.lbl_step)
        self.lbl_title = _label("", 17, INK, 600, wrap=True)
        lay.addWidget(self.lbl_title)
        self.lbl_body = _label("", 13, "#C3C8CF", wrap=True)
        self.lbl_body.setMinimumWidth(300)
        self.lbl_body.setMaximumWidth(340)
        lay.addWidget(self.lbl_body)
        lay.addSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_skip = _button("Skip tour", "quiet")
        self.btn_back = _button("Back", "secondary")
        self.btn_next = _button("Next", "primary")
        row.addWidget(self.btn_skip)
        row.addStretch(1)
        row.addWidget(self.btn_back)
        row.addWidget(self.btn_next)
        lay.addLayout(row)
        self.btn_skip.clicked.connect(lambda: self.finish(completed=False))
        self.btn_back.clicked.connect(self.back)
        self.btn_next.clicked.connect(self.next)

        self._hole_anim = QVariantAnimation(self)
        self._hole_anim.setDuration(320)
        self._hole_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hole_anim.valueChanged.connect(self._set_hole)
        self._card_anim = QPropertyAnimation(self.card, b"pos", self)
        self._card_anim.setDuration(320)
        self._card_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ---------- running ----------

    def start(self) -> None:
        self.show()
        self.raise_()
        self.setFocus()
        self.go(0, animate=False)

    def next(self) -> None:
        if self.index >= len(self.steps) - 1:
            self.finish(completed=True)
        else:
            self.go(self.index + 1)

    def back(self) -> None:
        if self.index > 0:
            self.go(self.index - 1)

    def finish(self, completed: bool) -> None:
        self.window_.removeEventFilter(self)
        self.hide()
        if self.on_finish is not None:
            self.on_finish(completed)
        self.deleteLater()

    def go(self, i: int, animate: bool = True) -> None:
        self.index = i
        step = self.steps[i]
        n = len(self.steps)
        first, last = i == 0, i == n - 1
        self.logo.setVisible(first and self.logo.pixmap() is not None and not self.logo.pixmap().isNull())
        self.lbl_step.setText("" if first else f"{i} OF {n - 1}")
        self.lbl_step.setVisible(not first)
        self.lbl_title.setText(step.title)
        self.lbl_body.setText(step.body)
        self.btn_back.setVisible(not first)
        self.btn_next.setText("Start the tour" if first else ("Done" if last else "Next"))
        self.btn_skip.setText("Not now" if first else "Skip tour")
        self.btn_skip.setVisible(not last)
        self.card.adjustSize()
        hole = self._target_rect(step)
        card_pos = self._card_position(hole)
        if animate and self.isVisible():
            self._hole_anim.stop()
            self._hole_anim.setStartValue(self._hole if not self._hole.isNull() else self._collapsed(hole))
            self._hole_anim.setEndValue(hole if not hole.isNull() else self._collapsed(self._hole))
            self._hole_anim.start()
            self._card_anim.stop()
            self._card_anim.setStartValue(self.card.pos())
            self._card_anim.setEndValue(card_pos)
            self._card_anim.start()
        else:
            self._set_hole(hole)
            self.card.move(card_pos)
        self.update()

    # ---------- geometry ----------

    def _target_rect(self, step: TourStep) -> QRectF:
        """The area to light up: the union of the step's visible widgets, padded. Empty = no spotlight."""
        if step.target is None:
            return QRectF()
        try:
            widgets = [w for w in step.target() if w is not None and w.isVisible()]
        except RuntimeError:  # a widget was deleted
            widgets = []
        rect = QRect()
        for w in widgets:
            parent = w.parentWidget()
            while parent is not None:  # a button scrolled out of sight in the toolbar: scroll it into view
                if isinstance(parent, QScrollArea):
                    parent.ensureWidgetVisible(w, 24, 0)
                    break
                parent = parent.parentWidget()
            top_left = w.mapTo(self.window_, QPoint(0, 0))
            rect = rect.united(QRect(top_left, w.size()))
        if rect.isNull():
            return QRectF()
        p = self.HOLE_PAD
        return QRectF(rect.adjusted(-p, -p, p, p)).intersected(QRectF(self.rect()))

    def _collapsed(self, around: QRectF) -> QRectF:
        """A zero-size hole at the centre of `around` (or the window), so the spotlight grows from there."""
        c = around.center() if not around.isNull() else QRectF(self.rect()).center()
        return QRectF(c.x(), c.y(), 0, 0)

    def _card_position(self, hole: QRectF) -> QPoint:
        """Beside the spotlight: below it if there is room, else above, right, left; centred when there is none."""
        w, h = self.card.width(), self.card.height()
        W, H, m = self.width(), self.height(), self.MARGIN
        if hole.isNull():
            return QPoint((W - w) // 2, (H - h) // 2)
        gap = 14

        def clamp_x(x):
            return int(max(m, min(W - w - m, x)))

        def clamp_y(y):
            return int(max(m, min(H - h - m, y)))

        if hole.bottom() + gap + h + m <= H:
            return QPoint(clamp_x(hole.center().x() - w / 2), int(hole.bottom() + gap))
        if hole.top() - gap - h >= m:
            return QPoint(clamp_x(hole.center().x() - w / 2), int(hole.top() - gap - h))
        if hole.right() + gap + w + m <= W:
            return QPoint(int(hole.right() + gap), clamp_y(hole.center().y() - h / 2))
        if hole.left() - gap - w >= m:
            return QPoint(int(hole.left() - gap - w), clamp_y(hole.center().y() - h / 2))
        return QPoint((W - w) // 2, (H - h) // 2)

    def _set_hole(self, rect) -> None:
        self._hole = QRectF(rect)
        self.update()

    # ---------- drawing and input ----------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        shade = QPainterPath()
        shade.addRect(QRectF(self.rect()))
        if self._hole.width() > 1 and self._hole.height() > 1:
            hole = QPainterPath()
            hole.addRoundedRect(self._hole, 10, 10)
            shade = shade.subtracted(hole)
            p.fillPath(shade, QColor(6, 8, 11, 178))
            pen = QPen(QColor(ACCENT))
            pen.setWidthF(2.0)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(self._hole, 10, 10)
        else:
            p.fillPath(shade, QColor(6, 8, 11, 178))
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        event.accept()  # the app underneath waits until the tour is over

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.next()
        elif key == Qt.Key.Key_Left:
            self.back()
        elif key == Qt.Key.Key_Escape:
            self.finish(completed=False)
        else:
            event.accept()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.window_ and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.window_.rect())
            if 0 <= self.index < len(self.steps):
                self.go(self.index, animate=False)
        return False
