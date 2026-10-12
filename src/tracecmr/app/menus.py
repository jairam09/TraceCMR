"""Toolbar dropdown menus in TraceCMR's own style: rounded dark card, an icon per entry, a check column for
on/off entries, and keyboard shortcuts drawn as small keycaps on the right (e.g. [Ctrl] [S]).

Qt's plain menus can only print a shortcut as text after a tab, so each entry here is a small widget row
(QWidgetAction). Clicking a row, or Enter on the highlighted one, triggers its action like a normal menu item.
"""

from __future__ import annotations

from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QMenu, QSizePolicy, QWidget, QWidgetAction

try:
    from qtpy.QtGui import QActionGroup
except ImportError:  # Qt5
    from qtpy.QtWidgets import QActionGroup

TEXT = "#E4E7EB"
DIM = "#6B7178"
HOVER = "#2C323A"

MENU_STYLE = """
QMenu { background: #1B1F25; border: 1px solid #353B43; border-radius: 10px; padding: 6px; }
QMenu::separator { height: 1px; background: #2E343C; margin: 5px 8px; }
QMenu::item { color: #E4E7EB; padding: 7px 14px 7px 12px; border-radius: 6px; font-size: 13px; }
QMenu::item:selected { background: #2C323A; }
QMenu::item:disabled { color: #6B7178; }
QMenu::indicator { width: 14px; height: 14px; left: 6px; }
QLabel#keycap { background: #262B32; border: 1px solid #3A4149; border-bottom: 2px solid #3A4149;
                border-radius: 4px; padding: 0px 5px; color: #C9CED4; font-size: 11px; font-weight: 600; }
QLabel#keycap:disabled { color: #6B7178; }
QLabel#rowtext { color: #E4E7EB; font-size: 13px; background: transparent; }
QLabel#rowtext:disabled { color: #6B7178; }
QWidget#row { background: transparent; border-radius: 6px; }
QWidget#row[active="true"] { background: #2C323A; }
"""


def style_menu(menu: QMenu) -> QMenu:
    """Rounded, borderless dark card (also used for the plain Save and Wall menus)."""
    menu.setWindowFlags(menu.windowFlags() | Qt.WindowType.FramelessWindowHint
                        | Qt.WindowType.NoDropShadowWindowHint)
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    menu.setStyleSheet(MENU_STYLE)
    return menu


def _icon_pixmap(name: str | None, color: str = "#C9CED4"):
    if not name:
        return None
    try:
        import qtawesome as qta

        return qta.icon(name, color=color, color_disabled=DIM).pixmap(16, 16)
    except Exception:
        return None


class _Row(QWidget):
    """One menu entry: [✓] [icon] text ........ [key] [key]"""

    def __init__(self, menu: "ToolMenu", action: QWidgetAction, text: str, icon: str | None,
                 keys: list[str], checkable: bool):
        super().__init__()
        self.menu, self.action, self.checkable = menu, action, checkable
        self.setObjectName("row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMouseTracking(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 10, 6)
        lay.setSpacing(9)
        self.check = QLabel()
        self.check.setFixedWidth(14)
        self.check.setStyleSheet("background: transparent;")
        self._check_pm = _icon_pixmap("mdi6.check", "#E4E7EB")
        lay.addWidget(self.check)
        self.icon = QLabel()
        self.icon.setFixedWidth(16)
        self.icon.setStyleSheet("background: transparent;")
        self._icon_name = icon
        lay.addWidget(self.icon)
        self.label = QLabel(text)
        self.label.setObjectName("rowtext")
        lay.addWidget(self.label)
        lay.addStretch(1)
        lay.addSpacing(18)
        self.caps = []
        for k in keys:
            cap = QLabel(k)
            cap.setObjectName("keycap")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap.setMinimumWidth(20)
            cap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lay.addWidget(cap)
            self.caps.append(cap)
        lay.setSpacing(6)
        self.refresh()

    def refresh(self) -> None:
        on = self.action.isEnabled()
        self.setEnabled(on)
        self.check.setPixmap(self._check_pm if (self.checkable and self.action.isChecked() and self._check_pm)
                             else _blank())
        pm = _icon_pixmap(self._icon_name, "#C9CED4" if on else DIM)
        self.icon.setPixmap(pm if pm is not None else _blank())

    def set_active(self, on: bool) -> None:
        if self.property("active") != on:
            self.setProperty("active", on)
            self.style().unpolish(self)
            self.style().polish(self)

    # mouse: highlight on hover, trigger on release (like a normal menu item)
    def enterEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled():
            self.menu.setActiveAction(self.action)
            self.menu._highlight(self.action)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.set_active(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled() and self.rect().contains(event.position().toPoint() if hasattr(event, "position")
                                                     else event.pos()):
            self.menu.close()
            self.action.trigger()
        event.accept()


_BLANK = None


def _blank():
    global _BLANK
    if _BLANK is None:
        from qtpy.QtGui import QPixmap

        _BLANK = QPixmap(16, 16)
        _BLANK.fill(Qt.GlobalColor.transparent)
    return _BLANK


class ToolMenu(QMenu):
    """A styled dropdown. add() returns the QAction, so callers can keep, check or disable it as usual."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        style_menu(self)
        self._rows: dict[QWidgetAction, _Row] = {}
        self.hovered.connect(self._highlight)  # keyboard up/down
        self.aboutToShow.connect(self._refresh)

    def add(self, text: str, slot: Callable | None = None, keys: str | list[str] | None = None,
            icon: str | None = None, checkable: bool = False, checked: bool = False, tooltip: str | None = None,
            group: QActionGroup | None = None) -> QWidgetAction:
        action = QWidgetAction(self)
        action.setText(text)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if group is not None:
            action.setActionGroup(group)
        if isinstance(keys, str):
            keys = [keys]
        row = _Row(self, action, text, icon, list(keys or []), checkable)
        if tooltip:
            from .hints import set_hint

            set_hint(row, None, tooltip)
            action.setToolTip(tooltip)
        action.setDefaultWidget(row)
        self._rows[action] = row
        action.changed.connect(row.refresh)
        if slot is not None:
            if checkable and group is None:  # on/off entry: follows its action however it changes
                action.toggled.connect(lambda on, f=slot: f(on))
            elif checkable:  # one of a group: only the one picked
                action.triggered.connect(lambda on=False, f=slot: f(on))
            else:
                action.triggered.connect(lambda _=False, f=slot: f())
        self.addAction(action)
        return action

    def _highlight(self, action) -> None:
        for a, row in self._rows.items():
            row.set_active(a is action and row.isEnabled())

    def _refresh(self) -> None:
        for row in self._rows.values():
            row.refresh()
            row.set_active(False)
