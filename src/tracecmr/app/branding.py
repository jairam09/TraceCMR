"""TraceCMR's name, logo and start screen in place of napari's.

TraceCMR runs inside napari, which by default names the app "napari", puts the napari logo on the window and
taskbar, and shows a napari welcome screen (logo, "drag file(s) here", napari tips) until something is open.
`prepare()` runs before the viewer is created and claims the app name, Windows taskbar ID and window icon;
`start_screen()` turns napari's welcome screen into a TraceCMR one. Everything here is cosmetic: any failure
(e.g. a napari version that moved things) leaves napari's own screen and the app still works.
"""

from __future__ import annotations

import sys
from pathlib import Path

RES = Path(__file__).with_name("resources")
ICON_SVG = RES / "tracecmr-icon.svg"
ICON_PNG = RES / "tracecmr.png"
LOGO = {"dark": RES / "tracecmr-logo-dark.svg", "light": RES / "tracecmr-logo.svg"}  # wordmark for each theme
APP_ID = "TraceCMR.TraceCMR"
SUPPORT_EMAIL = "tracecmr.support@gmail.com"  # Help → Report a problem, Help → About, README
MUTED = {"dark": "#98A2AD", "light": "#5B6472"}  # secondary text on the start screen (readable on black)
FONT_DIR = RES / "fonts"
FONT = "Inter"  # the interface font, bundled (SIL Open Font License, resources/fonts/OFL.txt)


def _windows_app_id() -> None:
    """Tell Windows this is its own app (not python.exe), so the taskbar shows TraceCMR's icon and groups its
    windows under it. Must run before any window is created."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


_APP = None


def create_app():
    """Create the Qt application as TraceCMR, before napari is even imported (napari takes several seconds to
    load), so the splash screen can show at once. Returns the QApplication."""
    _windows_app_id()
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    from .. import __version__

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
        app.setApplicationName("TraceCMR")
        app.setApplicationVersion(__version__)
        app.setOrganizationName("TraceCMR")
        app.setOrganizationDomain("tracecmr")
    if ICON_PNG.exists():
        app.setWindowIcon(QIcon(str(ICON_PNG)))
    use_app_font()
    no_ui_animations()
    global _APP
    _APP = app  # Qt destroys the application if nothing holds on to it
    return app


def prepare() -> None:
    """Before napari.Viewer(): make sure the Qt app runs under TraceCMR's name and icon, not napari's.

    If create_app() already made the app, napari's own setup still runs (theme files, clean shutdown) but its
    name and logo are not applied. Otherwise napari creates the app here, with TraceCMR's details. napari puts its
    logo back on the window whenever the theme or logo setting changes; it reads the icon from
    `_QtMainWindow._window_icon` when that is set, which is the hook used here.
    """
    _windows_app_id()
    try:
        from napari._qt.qt_event_loop import get_qapp
        from napari._qt.qt_main_window import _QtMainWindow
        from qtpy.QtWidgets import QApplication

        from .. import __version__

        if ICON_SVG.exists():
            _QtMainWindow._window_icon = str(ICON_SVG)
        if QApplication.instance() is None:
            get_qapp(app_name="TraceCMR", app_version=__version__, icon=str(ICON_SVG), org_name="TraceCMR",
                     org_domain="tracecmr", app_id=APP_ID)
        else:
            app = get_qapp()  # napari's first-run setup on our app
            if not getattr(app, "_tracecmr_tooltips", False):
                from napari._qt.qt_event_filters import QtToolTipEventFilter

                app._tracecmr_tooltip_filter = QtToolTipEventFilter()  # wraps long tooltips, as napari does
                app.installEventFilter(app._tracecmr_tooltip_filter)
                app._tracecmr_tooltips = True
    except Exception:
        pass
    use_app_font()
    no_ui_animations()


class Splash:
    """A small window with the logo shown the moment TraceCMR is opened, while the viewer and the panel load
    (about 10 to 20 seconds on a laptop), so a double-click never looks like nothing happened. Under the logo:
    a thin red line that fills as loading progresses, and one word of status."""

    W, H = 520, 300
    BAR_W, BAR_Y = 180, 214  # progress line: width, and its height on the splash
    BG, TRACK, FILL = "#0E1318", "#232B34", "#D0646F"

    def __init__(self):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QSplashScreen

        from .. import __version__

        self.version = __version__
        self.fraction, self.text = 0.0, "Starting"
        self.widget = QSplashScreen(self._draw(), Qt.WindowType.WindowStaysOnTopHint)
        self.widget.setWindowTitle("TraceCMR")
        self.widget.show()
        self._refresh()

    def _draw(self):
        from qtpy.QtCore import QRectF, Qt
        from qtpy.QtGui import QColor, QPainter, QPixmap

        pm = QPixmap(self.W, self.H)
        pm.fill(QColor(self.BG))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(int((self.W - 300) / 2), 84, _logo_pixmap("dark", 300))
        x0 = (self.W - self.BAR_W) / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.TRACK))
        p.drawRoundedRect(QRectF(x0, self.BAR_Y, self.BAR_W, 2), 1, 1)
        if self.fraction > 0:
            p.setBrush(QColor(self.FILL))
            p.drawRoundedRect(QRectF(x0, self.BAR_Y, self.BAR_W * min(1.0, self.fraction), 2), 1, 1)
        f = p.font()
        f.setFamily(FONT)
        f.setHintingPreference(f.HintingPreference.PreferNoHinting)
        f.setPixelSize(13)
        p.setFont(f)
        p.setPen(QColor(MUTED["dark"]))
        p.drawText(QRectF(0, self.BAR_Y + 14, self.W, 20), Qt.AlignmentFlag.AlignCenter, f"{self.text}…")
        f.setPixelSize(11)
        p.setFont(f)
        p.setPen(QColor("#5E6873"))
        p.drawText(QRectF(0, 14, self.W - 18, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"v{self.version}")
        p.end()
        return pm

    def _refresh(self) -> None:
        from qtpy.QtWidgets import QApplication

        self.widget.setPixmap(self._draw())
        QApplication.processEvents()

    def progress(self, fraction: float, text: str | None = None) -> None:
        """Fill the line to `fraction` (0 to 1); optionally change the status word."""
        self.fraction = max(self.fraction, float(fraction))
        if text:
            self.text = text
        self._refresh()

    def message(self, text: str) -> None:
        self.progress(self.fraction, text)

    def track_imports(self, start: float, end: float, expected: int = 2000):
        """While loading, move the line a little with every Python module imported (about 2000 at startup), so it
        glides instead of jumping between a few checkpoints. Use as `with splash.track_imports(0.05, 0.85): ...`."""
        return _ImportTicker(self, start, end, expected)

    def close(self, window=None) -> None:
        try:
            self.progress(1.0)
            if window is not None:
                self.widget.finish(window)
            else:
                self.widget.close()
        except Exception:
            pass


class _ImportTicker:
    """Import hook that never finds anything itself; it only counts imports and repaints the splash now and then
    (Python is busy loading, so this is the only moment the splash can be redrawn)."""

    def __init__(self, splash: Splash, start: float, end: float, expected: int):
        self.splash, self.start, self.end, self.expected = splash, start, end, max(1, expected)
        self.n, self.last = 0, 0.0

    def find_spec(self, name, path=None, target=None):
        import math
        import time

        self.n += 1
        now = time.monotonic()
        if now - self.last > 0.03:  # at most ~30 repaints a second
            self.last = now
            eased = 1 - math.exp(-self.n / (0.45 * self.expected))  # slows near the end, never overshoots
            try:
                self.splash.progress(self.start + (self.end - self.start) * eased)
            except Exception:
                pass
        return None

    def __enter__(self):
        import sys

        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        import sys

        if self in sys.meta_path:
            sys.meta_path.remove(self)
        return False


def load_fonts() -> list[str]:
    """Register the bundled fonts with Qt (they are not installed on the computer). Returns the families loaded."""
    families: list[str] = []
    try:
        from qtpy.QtGui import QFontDatabase

        for f in sorted(FONT_DIR.glob("*.ttf")):
            i = QFontDatabase.addApplicationFont(str(f))
            if i >= 0:
                families += QFontDatabase.applicationFontFamilies(i)
    except Exception:
        pass
    return sorted(set(families))


def no_ui_animations() -> None:
    """Menus, dropdowns and tooltips appear instantly. Windows fades or slides them in by default, which makes
    every toolbar menu feel a beat slow."""
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        for effect in (Qt.UIEffect.UI_AnimateMenu, Qt.UIEffect.UI_FadeMenu, Qt.UIEffect.UI_AnimateCombo,
                       Qt.UIEffect.UI_AnimateTooltip, Qt.UIEffect.UI_FadeTooltip, Qt.UIEffect.UI_AnimateToolBox):
            QApplication.setEffectEnabled(effect, False)
    except Exception:
        pass


def use_app_font() -> bool:
    """Set the whole interface (napari's menus and layer list too) in Inter, so it looks the same on every
    computer instead of falling back to Segoe UI / San Francisco. Sizes stay napari's. False if unavailable."""
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None or FONT not in load_fonts():
            return False
        f = app.font()
        f.setFamily(FONT)
        # draw Inter as designed (smooth, like macOS) instead of snapping strokes to the pixel grid, which
        # thins and breaks up light text on dark backgrounds on Windows
        f.setHintingPreference(f.HintingPreference.PreferNoHinting)
        app.setFont(f)
        no_ui_animations()
        return True
    except Exception:
        return False


def set_icon(viewer) -> None:
    """The TraceCMR logo on the window's title bar and taskbar button."""
    try:
        from qtpy.QtGui import QIcon
        from qtpy.QtWidgets import QApplication

        if ICON_PNG.exists():
            icon = QIcon(str(ICON_PNG))
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)
            viewer.window._qt_window.setWindowIcon(icon)
    except Exception:
        pass


TRACE_LEN = 91.78  # length of the logo's red trace, in the logo's own units (for the draw-in animation)


def _logo_pixmap(theme: str, width: int, trace: float = 1.0):
    """The wordmark at `width` logical pixels, rendered at the screen's pixel density so it stays sharp.
    `trace` (0 to 1) is how much of the red trace is drawn, for the start screen's draw-in animation."""
    from qtpy.QtCore import QByteArray, QRectF, QSize, Qt
    from qtpy.QtGui import QGuiApplication, QPainter, QPixmap
    from qtpy.QtSvg import QSvgRenderer

    path = LOGO["light" if theme == "light" else "dark"]
    if trace >= 1.0:
        r = QSvgRenderer(str(path))
    else:
        svg = path.read_text(encoding="utf-8")
        if trace <= 0.005:
            dash = 'stroke-opacity="0"'
        else:
            dash = f'stroke-dasharray="{TRACE_LEN:.2f} {TRACE_LEN:.2f}" stroke-dashoffset="{TRACE_LEN * (1 - trace):.2f}"'
        svg = svg.replace('pathLength="100"', dash, 1)
        r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    vb = r.viewBoxF()
    h = width * vb.height() / vb.width()
    screen = QGuiApplication.primaryScreen()
    dpr = max(1.0, screen.devicePixelRatio()) if screen is not None else 1.0
    pm = QPixmap(QSize(int(round(width * dpr)), int(round(h * dpr))))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    r.render(p, QRectF(0, 0, width * dpr, h * dpr))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def animate_logo(viewer, delay_ms: int = 450) -> bool:
    """Play the start screen's one-time trace draw-in (set up by start_screen(animate=True)), after a short pause
    so the window is settled and the eye has landed on the logo before the line starts."""
    try:
        from qtpy.QtCore import QTimer

        play = viewer.window._qt_viewer._welcome_widget._tracecmr_play
        QTimer.singleShot(delay_ms, play)
        return True
    except Exception:
        return False


def _theme_kind(viewer) -> str:
    try:
        from napari.utils.theme import get_theme

        name = viewer.theme
        if name == "system":
            from napari.utils.theme import get_system_theme

            name = get_system_theme()
        return "light" if get_theme(name).id == "light" or name == "light" else "dark"
    except Exception:
        return "dark"


def start_screen(viewer, open_path=None, animate: bool = False) -> bool:
    """Replace napari's welcome screen (shown while nothing is open) with TraceCMR's: the logo, one quiet line
    on how to start, and the version. A mouse folder or .h5 file dropped on it opens in TraceCMR through
    `open_path`. With `animate`, the logo's red trace starts undrawn and animate_logo() draws it in once (call it
    when the window is on screen). Returns False if napari's screen could not be found (then it is left as is)."""
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

        from .. import __version__

        qv = viewer.window._qt_viewer
        w = qv._welcome_widget
        napari_parts = {w._image.parentWidget(), w._label.parentWidget()}
    except Exception:
        return False
    try:
        for part in napari_parts:  # napari's logo, "drag file(s) here", shortcuts and "Did you know?" tips
            if part is not None and part is not w:
                part.hide()

        screen = QWidget(w)
        screen.setObjectName("tracecmr_start")
        screen.setStyleSheet("background: transparent;")
        col = QVBoxLayout(screen)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(0)
        logo = QLabel()
        logo.setObjectName("tracecmr_logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("Open a scan to begin")
        hint.setObjectName("tracecmr_hint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        version = QLabel(f"v{__version__}")
        version.setObjectName("tracecmr_version")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addStretch(3)
        col.addWidget(logo)
        col.addSpacing(22)
        col.addWidget(hint)
        col.addStretch(4)  # sits a little above the middle, where the eye expects it
        col.addWidget(version)
        w.layout().addWidget(screen, 0, 0)

        state = {"trace": 0.0 if animate else 1.0}

        def draw_logo():
            logo.setPixmap(_logo_pixmap(_theme_kind(viewer), 280, state["trace"]))

        def restyle(_event=None):
            kind = _theme_kind(viewer)
            draw_logo()
            muted = MUTED[kind]
            hint.setStyleSheet(f"background: transparent; color: {muted};")
            version.setStyleSheet(f"background: transparent; color: {muted};")
            for lbl, px in ((hint, 14), (version, 11)):
                f = lbl.font()
                f.setFamily(FONT)
                f.setPixelSize(px)
                f.setHintingPreference(f.HintingPreference.PreferNoHinting)
                lbl.setFont(f)
            logo.setStyleSheet("background: transparent;")

        restyle()
        viewer.events.theme.connect(restyle)
        w._tracecmr = screen  # keeps a handle for tests and later tweaks
        w._tracecmr_trace = state

        def play(duration_ms: int = 1600):
            """Draw the red trace in once, like a contour being traced: starts gently so the first stroke is
            seen, speeds up through the middle, settles at the end."""
            import math

            from qtpy.QtCore import QElapsedTimer, QTimer

            clock, timer = QElapsedTimer(), QTimer(w)
            clock.start()

            def step():
                t = min(1.0, clock.elapsed() / duration_ms)
                state["trace"] = 0.5 - 0.5 * math.cos(math.pi * t)  # gentle start and finish
                draw_logo()
                if t >= 1.0:
                    timer.stop()

            timer.timeout.connect(step)
            timer.start(16)
            w._tracecmr_anim = timer

        w._tracecmr_play = play

        if open_path is not None:
            try:
                w.urls_dropped.disconnect()
            except (TypeError, RuntimeError):
                pass

            def dropped(event):
                w._set_drag_highlight(False)
                urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
                paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
                p = next((p for p in paths if p.is_dir() or p.suffix.lower() == ".h5"), None)
                if p is not None:
                    event.acceptProposedAction()
                    open_path(p)

            w.urls_dropped.connect(dropped)
    except Exception:
        return False
    return True
