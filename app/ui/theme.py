"""Centralized application theme: design tokens + Qt stylesheet.

The previous "liquid glass" surface has been retired in favour of a precise,
editorial, pro-instrument aesthetic ("STUDIO"). Two parallel palettes — light
and dark — share a single QSS template so the structure never drifts between
modes. Light is now a warm paper-tone (Capture-One-like) and dark is a deep
charcoal with a signature jade accent.

Public surface (kept stable for backward compatibility with imports):

* Token constants: ``BG_APP`` / ``SURFACE`` / ``ACCENT`` … (light variant)
* Parallel dark constants: ``DARK_BG_APP`` / ``DARK_SURFACE`` / ``DARK_ACCENT`` …
* Aggregate dicts: ``LIGHT_TOKENS`` / ``DARK_TOKENS``
* Stylesheets: ``APP_STYLESHEET`` (= light) + ``DARK_APP_STYLESHEET``
* Palette builders: ``build_palette(mode)``
* Apply entry points: ``apply_app_theme(app, mode, resolved=None)``

Mode is one of "light" / "dark" / "system". ``apply_app_theme`` resolves
"system" once via ``QStyleHints.colorScheme()`` (PySide6 6.5+) and honours
``resolved`` if the caller already detected it.
"""
from __future__ import annotations

from string import Template
from typing import Optional

# ----------------------------------------------------------------- LIGHT ---
#: Warm paper-tone canvas (deliberately *not* #FFFFFF — pure white reads as a
#: photo editor's blank canvas and is hostile to OLED displays).
BG_APP = "#F4F1EB"
SURFACE = "#FBF8F2"
SURFACE_MUTED = "#EFEAE0"
SURFACE_HOVER = "#E8E1D3"
SURFACE_PRESSED = "#DED5C2"
BORDER = "#D8CFBC"
BORDER_STRONG = "#B7AC93"
DIVIDER = "#E2D9C7"

TEXT = "#1B1B1A"
TEXT_STRONG = "#0E0E0D"
TEXT_SECONDARY = "#3D3A33"
#: Quiet-but-readable text: hints, captions, secondary metadata. Every variant
#: clears WCAG AA (4.5:1) against the surfaces it is drawn on.
TEXT_MUTED = "#5C574C"
TEXT_FAINT = "#8E8878"

#: Accent: signature brand tone, paired across both palettes for depth.
ACCENT = "#0A6B50"
ACCENT_HOVER = "#085943"
ACCENT_PRESSED = "#064636"
ACCENT_SOFT = "#DCEFE5"
ACCENT_SOFT_BORDER = "#9DCFB8"
ACCENT_SOFT_TEXT = "#085943"
ACCENT_DISABLED = "#B8D7C8"
ACCENT_LIGHT = "#5DC9A8"

SUCCESS = "#1B6E3D"
WARNING = "#7E4F00"
DANGER = "#9A2929"
NEUTRAL = "#8E8878"
DANGER_SOFT = "#F5E1DD"
DANGER_SOFT_BORDER = "#D89A92"

#: Progress-bar fill: medium-saturation tints. The text-safe semantic colours
#: above are tuned for 4.5:1 contrast and read almost black when used as a
#: fat bar fill — the bar gets its own lighter, quieter family.
PROGRESS_TRACK = "#EAE3D5"
PROGRESS_NORMAL = "#3BAF8A"
PROGRESS_SUCCESS = "#43A56E"
PROGRESS_WARNING = "#C8973F"
PROGRESS_ERROR = "#D3716F"
PROGRESS_NEUTRAL = "#AEA694"

# ------------------------------------------------------------------ DARK ---
#: Deep charcoal canvas. Deliberately not #000000 — pure black eats anti-
#: aliasing of 1px strokes and makes the accent too aggressive.
DARK_BG_APP = "#0E0F12"
DARK_SURFACE = "#15171B"
DARK_SURFACE_MUTED = "#111319"
DARK_SURFACE_HOVER = "#1C1F25"
DARK_SURFACE_PRESSED = "#23262D"
DARK_BORDER = "#262932"
DARK_BORDER_STRONG = "#363A42"
DARK_DIVIDER = "#1B1E24"

DARK_TEXT = "#E5E3DC"
DARK_TEXT_STRONG = "#F4F2EA"
DARK_TEXT_SECONDARY = "#C4C1B6"
DARK_TEXT_MUTED = "#9F9B8E"
DARK_TEXT_FAINT = "#5C5A53"

DARK_ACCENT = "#5EEAD4"
DARK_ACCENT_HOVER = "#2DD4BF"
DARK_ACCENT_PRESSED = "#14B8A6"
DARK_ACCENT_SOFT = "#102A2A"
DARK_ACCENT_SOFT_BORDER = "#2F6F66"
DARK_ACCENT_SOFT_TEXT = "#A7F3D0"
DARK_ACCENT_DISABLED = "#2C5750"
DARK_ACCENT_LIGHT = "#99F6E4"

DARK_SUCCESS = "#4ADE80"
DARK_WARNING = "#FBBF24"
DARK_DANGER = "#F87171"
DARK_NEUTRAL = "#5A5A5A"
DARK_DANGER_SOFT = "#2A1318"
DARK_DANGER_SOFT_BORDER = "#7F2D2D"

#: See the light-mode PROGRESS_* block: mid-tone fills, calm track.
DARK_PROGRESS_TRACK = "#23272F"
DARK_PROGRESS_NORMAL = "#3EC4A7"
DARK_PROGRESS_SUCCESS = "#4FC47E"
DARK_PROGRESS_WARNING = "#D8AC4F"
DARK_PROGRESS_ERROR = "#E07E7E"
DARK_PROGRESS_NEUTRAL = "#5C636E"

# --- Fonts ----------------------------------------------------------------
#: Body + display. Outfit is geometric, has a wide weight range (300–900),
# and reads distinctly at both small and very-large sizes — a deliberate move
# away from Segoe UI Variable / Inter / Space Grotesk.
UI_FONT = '"Outfit", "Segoe UI Variable Text", "Microsoft YaHei UI"'
UI_FONT_FAMILY = "Outfit"
UI_FONT_POINT_SIZE = 10
#: Technical readouts (file sizes, paths, model registry names, log prefixes).
#: JetBrains Mono has a wider character width than Cascadia Mono, which keeps
#: log lines readable without turning every digit into a serif.
MONO_FONT = '"JetBrains Mono", "Cascadia Mono", "Consolas", "Microsoft YaHei UI"'

#: Log level colours are semantic status cues, not decoration. Each variant is
#: tuned for the surface it is read on and clears WCAG AA (4.5:1) against it.
LOG_LEVEL_COLORS = {
    "Info": "#5C574C",
    "Warning": "#7E4F00",
    "Error": "#9A2929",
    "Success": "#1B6E3D",
}

DARK_LOG_LEVEL_COLORS = {
    "Info": "#9F9B8E",
    "Warning": "#FBBF24",
    "Error": "#F87171",
    "Success": "#4ADE80",
}


def log_level_colors(mode: str) -> dict[str, str]:
    """Log level colours for ``mode`` ("light" / "dark")."""
    return DARK_LOG_LEVEL_COLORS if mode == "dark" else LOG_LEVEL_COLORS

# --- Token aggregates ------------------------------------------------------
LIGHT_TOKENS = {
    "BG_APP": BG_APP,
    "SURFACE": SURFACE,
    "SURFACE_MUTED": SURFACE_MUTED,
    "SURFACE_HOVER": SURFACE_HOVER,
    "SURFACE_PRESSED": SURFACE_PRESSED,
    "BORDER": BORDER,
    "BORDER_STRONG": BORDER_STRONG,
    "DIVIDER": DIVIDER,
    "TEXT": TEXT,
    "TEXT_STRONG": TEXT_STRONG,
    "TEXT_SECONDARY": TEXT_SECONDARY,
    "TEXT_MUTED": TEXT_MUTED,
    "TEXT_FAINT": TEXT_FAINT,
    "ACCENT": ACCENT,
    "ACCENT_HOVER": ACCENT_HOVER,
    "ACCENT_PRESSED": ACCENT_PRESSED,
    "ACCENT_SOFT": ACCENT_SOFT,
    "ACCENT_SOFT_BORDER": ACCENT_SOFT_BORDER,
    "ACCENT_SOFT_TEXT": ACCENT_SOFT_TEXT,
    "ACCENT_DISABLED": ACCENT_DISABLED,
    "ACCENT_LIGHT": ACCENT_LIGHT,
    "SUCCESS": SUCCESS,
    "WARNING": WARNING,
    "DANGER": DANGER,
    "NEUTRAL": NEUTRAL,
    "DANGER_SOFT": DANGER_SOFT,
    "DANGER_SOFT_BORDER": DANGER_SOFT_BORDER,
    "PROGRESS_TRACK": PROGRESS_TRACK,
    "PROGRESS_NORMAL": PROGRESS_NORMAL,
    "PROGRESS_SUCCESS": PROGRESS_SUCCESS,
    "PROGRESS_WARNING": PROGRESS_WARNING,
    "PROGRESS_ERROR": PROGRESS_ERROR,
    "PROGRESS_NEUTRAL": PROGRESS_NEUTRAL,
    # Hairline grid + film-grain texture (procedural, see BackdropWidget).
    "GRID_RGB": "139, 132, 108",
    "GRAIN_RGB": "60, 50, 30",
    "GRAIN_ALPHA": "0.045",
    "UI_FONT": UI_FONT,
    "MONO_FONT": MONO_FONT,
}

DARK_TOKENS = {
    "BG_APP": DARK_BG_APP,
    "SURFACE": DARK_SURFACE,
    "SURFACE_MUTED": DARK_SURFACE_MUTED,
    "SURFACE_HOVER": DARK_SURFACE_HOVER,
    "SURFACE_PRESSED": DARK_SURFACE_PRESSED,
    "BORDER": DARK_BORDER,
    "BORDER_STRONG": DARK_BORDER_STRONG,
    "DIVIDER": DARK_DIVIDER,
    "TEXT": DARK_TEXT,
    "TEXT_STRONG": DARK_TEXT_STRONG,
    "TEXT_SECONDARY": DARK_TEXT_SECONDARY,
    "TEXT_MUTED": DARK_TEXT_MUTED,
    "TEXT_FAINT": DARK_TEXT_FAINT,
    "ACCENT": DARK_ACCENT,
    "ACCENT_HOVER": DARK_ACCENT_HOVER,
    "ACCENT_PRESSED": DARK_ACCENT_PRESSED,
    "ACCENT_SOFT": DARK_ACCENT_SOFT,
    "ACCENT_SOFT_BORDER": DARK_ACCENT_SOFT_BORDER,
    "ACCENT_SOFT_TEXT": DARK_ACCENT_SOFT_TEXT,
    "ACCENT_DISABLED": DARK_ACCENT_DISABLED,
    "ACCENT_LIGHT": DARK_ACCENT_LIGHT,
    "SUCCESS": DARK_SUCCESS,
    "WARNING": DARK_WARNING,
    "DANGER": DARK_DANGER,
    "NEUTRAL": DARK_NEUTRAL,
    "DANGER_SOFT": DARK_DANGER_SOFT,
    "DANGER_SOFT_BORDER": DARK_DANGER_SOFT_BORDER,
    "PROGRESS_TRACK": DARK_PROGRESS_TRACK,
    "PROGRESS_NORMAL": DARK_PROGRESS_NORMAL,
    "PROGRESS_SUCCESS": DARK_PROGRESS_SUCCESS,
    "PROGRESS_WARNING": DARK_PROGRESS_WARNING,
    "PROGRESS_ERROR": DARK_PROGRESS_ERROR,
    "PROGRESS_NEUTRAL": DARK_PROGRESS_NEUTRAL,
    "GRID_RGB": "70, 80, 90",
    "GRAIN_RGB": "200, 200, 200",
    "GRAIN_ALPHA": "0.04",
    "UI_FONT": UI_FONT,
    "MONO_FONT": MONO_FONT,
}


def tokens_for(mode: str) -> dict:
    """Return the token dict for ``mode`` ("light" or "dark")."""
    return LIGHT_TOKENS if mode == "light" else DARK_TOKENS


# --- The single QSS template; both palettes share this structure ---------
_STYLESHEET_TEMPLATE = Template(
    r"""
/* ============================ base ============================ */
QMainWindow,
QWidget#appRoot,
QWidget#contentArea {
    background-color: $BG_APP;
}

QWidget#pageBody {
    background-color: transparent;
}

QScrollArea {
    background: transparent;
    border: none;
}

QDialog,
QMessageBox {
    background-color: $SURFACE;
}

QMessageBox QLabel {
    color: $TEXT;
    font-size: 13px;
}

QMessageBox QPushButton {
    min-width: 78px;
}

QToolTip {
    background-color: $TEXT_STRONG;
    color: $SURFACE;
    border: 1px solid $BORDER_STRONG;
    padding: 4px 8px;
    font-family: $UI_FONT;
}

/* ============================ sidebar ========================= */
QFrame#sidebar {
    background-color: $SURFACE_MUTED;
    border: none;
    border-right: 1px solid $BORDER;
}

QLabel#brandMark {
    background-color: $ACCENT;
    color: $SURFACE;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 800;
    font-family: $UI_FONT;
    letter-spacing: 0.5px;
}

QLabel#brandTitle {
    color: $TEXT_STRONG;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

QLabel#brandSubtitle {
    color: $TEXT_MUTED;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

QLabel#navSection {
    color: $TEXT_MUTED;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    padding: 0 12px;
    text-transform: uppercase;
}

QPushButton#navButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 9px 12px;
    color: $TEXT_SECONDARY;
    font-size: 13px;
    font-weight: 500;
    text-align: left;
}

QPushButton#navButton:hover {
    background-color: $SURFACE_HOVER;
    color: $TEXT_STRONG;
}

QPushButton#navButton:checked {
    background-color: $ACCENT_SOFT;
    border-color: $ACCENT_SOFT_BORDER;
    color: $ACCENT_SOFT_TEXT;
    font-weight: 600;
}

QPushButton#navButton:focus {
    border-color: $ACCENT;
}

QFrame#sidebarFooter {
    background-color: transparent;
    border: 1px solid $BORDER;
    border-radius: 6px;
}

QLabel#sidebarFooterText {
    color: $TEXT_MUTED;
    font-size: 11px;
}

/* ============================ header ========================== */
QFrame#headerCard {
    background-color: transparent;
    border: none;
}

QLabel#headerEyebrow {
    color: $ACCENT;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}

QLabel#headerTitle {
    color: $TEXT_STRONG;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.6px;
}

QLabel#headerSubtitle {
    color: $TEXT_MUTED;
    font-size: 13px;
    font-weight: 400;
}

QLabel#gpuInfoLabel {
    background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 8px 12px;
    color: $TEXT_SECONDARY;
    font-family: $MONO_FONT;
    font-size: 11px;
    font-weight: 500;
}

/* ============================ cards =========================== */
QFrame#card {
    background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 8px;
}

QLabel#cardEyebrow {
    color: $ACCENT;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}

QLabel#cardTitle {
    color: $TEXT_STRONG;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

QLabel#cardHint {
    color: $TEXT_MUTED;
    font-size: 12px;
}

QFrame#cardDivider {
    background-color: $DIVIDER;
    border: none;
}

/* ============================ inputs ========================== */
QLineEdit,
QComboBox,
QSpinBox,
QPlainTextEdit,
QListWidget {
    background-color: $SURFACE_MUTED;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 6px 10px;
    color: $TEXT;
    selection-background-color: $ACCENT;
    selection-color: $SURFACE;
}

QLineEdit:hover,
QComboBox:hover,
QSpinBox:hover,
QPlainTextEdit:hover,
QListWidget:hover {
    border-color: $BORDER_STRONG;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QPlainTextEdit:focus,
QListWidget:focus {
    border-color: $ACCENT;
    background-color: $SURFACE;
}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QPlainTextEdit:disabled,
QListWidget:disabled {
    background-color: $SURFACE_MUTED;
    color: $TEXT_FAINT;
    border-color: $DIVIDER;
}

QComboBox QAbstractItemView {
    background-color: $SURFACE;
    border: 1px solid $BORDER_STRONG;
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: $ACCENT_SOFT;
    selection-color: $ACCENT_SOFT_TEXT;
}

QComboBox {
    padding-right: 28px;
}

/* Hide the native drop-down button: ChevronComboBox draws its own arrow in
   paintEvent. Without this rule the two arrows stack on top of each other. */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 0;
    height: 0;
    border: none;
    background: transparent;
}

QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}

/* Hide the native step buttons entirely. PlusMinusSpinBox draws the
   "+" and "−" glyphs and handles its own clicks in the right-hand
   strip, so the QSS subcontrols would only cause confusion (wrong
   half stepping the value, double click handling, etc.). */
QSpinBox {
    padding-right: 24px;
}

QSpinBox::up-button,
QSpinBox::down-button {
    subcontrol-origin: border;
    width: 0;
    height: 0;
    border: none;
    background: transparent;
}

QSpinBox::up-arrow,
QSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}

QComboBox QAbstractItemView::item {
    min-height: 30px;
    padding: 4px 10px;
    border-radius: 6px;
    color: $TEXT_SECONDARY;
}

QComboBox QAbstractItemView::item:hover {
    background-color: $SURFACE_HOVER;
    color: $TEXT_STRONG;
}

/* ============================ buttons ========================= */
QPushButton {
    background-color: $SURFACE;
    border: 1px solid $BORDER_STRONG;
    border-radius: 6px;
    padding: 7px 14px;
    color: $TEXT_SECONDARY;
    font-weight: 500;
    font-family: $UI_FONT;
}

QPushButton:hover {
    background-color: $SURFACE_HOVER;
    border-color: $TEXT_FAINT;
    color: $TEXT_STRONG;
}

QPushButton:pressed {
    background-color: $SURFACE_PRESSED;
}

QPushButton:disabled {
    background-color: $SURFACE_MUTED;
    color: $TEXT_FAINT;
    border-color: $DIVIDER;
}

QPushButton:focus {
    border-color: $ACCENT;
}

QPushButton#primaryButton {
    background-color: $ACCENT;
    border: 1px solid $ACCENT;
    border-radius: 6px;
    padding: 9px 26px;
    color: $SURFACE;
    font-weight: 700;
    font-family: $UI_FONT;
    letter-spacing: 0.2px;
}

QPushButton#primaryButton:hover {
    background-color: $ACCENT_HOVER;
    border-color: $ACCENT_HOVER;
}

QPushButton#primaryButton:pressed {
    background-color: $ACCENT_PRESSED;
    border-color: $ACCENT_PRESSED;
}

QPushButton#primaryButton:disabled {
    background-color: $ACCENT_DISABLED;
    border-color: $ACCENT_DISABLED;
    color: $SURFACE;
}

QPushButton#softButton {
    background-color: $SURFACE_MUTED;
    border: 1px solid $BORDER;
    color: $TEXT_SECONDARY;
}

QPushButton#softButton:hover {
    background-color: $SURFACE_HOVER;
    color: $TEXT_STRONG;
    border-color: $BORDER_STRONG;
}

QPushButton#ghostButton {
    background-color: transparent;
    border: 1px solid transparent;
    color: $TEXT_MUTED;
    padding: 5px 10px;
}

QPushButton#ghostButton:hover {
    background-color: $SURFACE_HOVER;
    border-color: transparent;
    color: $TEXT_STRONG;
}

QPushButton#cancelButton:hover {
    background-color: $DANGER_SOFT;
    border-color: $DANGER_SOFT_BORDER;
    color: $DANGER;
}

/* Segmented control (the comparison mode switch): two buttons that read as one
   group, where the checked one is the active view. */
QPushButton#segmentButton {
    background-color: transparent;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 5px 14px;
    color: $TEXT_SECONDARY;
    font-size: 12px;
    font-weight: 500;
}

QPushButton#segmentButton:hover {
    background-color: $SURFACE_HOVER;
    border-color: $BORDER_STRONG;
    color: $TEXT_STRONG;
}

QPushButton#segmentButton:checked {
    background-color: $ACCENT_SOFT;
    border-color: $ACCENT_SOFT_BORDER;
    color: $ACCENT_SOFT_TEXT;
    font-weight: 600;
}

QPushButton#segmentButton:checked:hover {
    background-color: $ACCENT_SOFT;
    border-color: $ACCENT;
}

/* Match the primary action height so the task row reads as one control group. */
QPushButton#cancelButton {
    padding: 9px 20px;
}

/* ============================ progress ======================== */
QProgressBar {
    background-color: $PROGRESS_TRACK;
    border: 1px solid $BORDER;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
}

QProgressBar::chunk {
    background-color: $PROGRESS_NORMAL;
    border-radius: 3px;
}

/* ============================ sliders ========================= */
QSlider {
    background: transparent;
    min-height: 24px;
}

QSlider::groove:horizontal {
    height: 4px;
    background: $DIVIDER;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: $ACCENT;
    border-radius: 2px;
}

QSlider::add-page:horizontal {
    background: $DIVIDER;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: $SURFACE;
    border: 2px solid $ACCENT;
    width: 14px;
    height: 14px;
    margin: -7px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: $ACCENT_SOFT;
}

QSlider::handle:horizontal:pressed {
    background: $ACCENT;
    border-color: $ACCENT_PRESSED;
}

QSlider:disabled::handle:horizontal {
    border-color: $NEUTRAL;
}

/* ============================ checkbox ======================== */
QCheckBox {
    background: transparent;
    color: $TEXT_SECONDARY;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid $BORDER_STRONG;
    border-radius: 4px;
    background-color: $SURFACE_MUTED;
}

QCheckBox::indicator:hover {
    border-color: $ACCENT;
}

QCheckBox::indicator:checked {
    background-color: $ACCENT;
    border-color: $ACCENT;
}

QCheckBox::indicator:disabled {
    background-color: $SURFACE_MUTED;
    border-color: $DIVIDER;
}

/* ============================ lists =========================== */
QListWidget {
    padding: 6px;
    outline: none;
}

QListWidget::item {
    padding: 7px 10px;
    border-radius: 6px;
    color: $TEXT_SECONDARY;
}

QListWidget::item:hover {
    background-color: $SURFACE_HOVER;
    color: $TEXT_STRONG;
}

QListWidget::item:selected {
    background-color: $ACCENT_SOFT;
    color: $ACCENT_SOFT_TEXT;
}

/* Settings-page nav rail — denser than the global list style because the
   rail is meant to scan quickly rather than display rich rows. */
QListWidget#settingsNav {
    background: transparent;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 6px;
}
QListWidget#settingsNav::item {
    padding: 9px 12px;
    font-weight: 600;
}

/* ============================ labels ========================== */
QLabel {
    background: transparent;
    color: $TEXT;
}

QLabel#fieldLabel {
    color: $TEXT_SECONDARY;
    font-weight: 500;
    font-size: 12px;
}

QLabel#valueStrong {
    color: $TEXT_STRONG;
    font-weight: 700;
}

QLabel#valueBadge {
    background-color: $ACCENT_SOFT;
    border: 1px solid $ACCENT_SOFT_BORDER;
    border-radius: 4px;
    padding: 2px 8px;
    color: $ACCENT_SOFT_TEXT;
    font-family: $MONO_FONT;
    font-size: 11px;
    font-weight: 600;
}

QLabel#mutedLabel {
    color: $TEXT_MUTED;
    font-size: 12px;
}

QLabel#statusLabel {
    color: $TEXT_SECONDARY;
    font-size: 12px;
    font-weight: 500;
}

QLabel#videoMetaLabel {
    font-size: 12px;
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid $BORDER;
    background: $SURFACE_MUTED;
    font-family: $MONO_FONT;
}
QLabel#videoMetaLabel[state="placeholder"] {
    color: $TEXT_MUTED;
}
QLabel#videoMetaLabel[state="ok"] {
    color: $TEXT;
    border-color: $ACCENT_SOFT_BORDER;
    background: $ACCENT_SOFT;
}
QLabel#videoMetaLabel[state="danger"] {
    color: $DANGER;
    border-color: $DANGER_SOFT_BORDER;
    background: $DANGER_SOFT;
}

QLabel#logTitle {
    color: $TEXT_STRONG;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

/* ============================ log view ======================== */
QPlainTextEdit#logView {
    background-color: $SURFACE_MUTED;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 10px 12px;
    color: $TEXT_SECONDARY;
    font-family: $MONO_FONT;
    font-size: 12px;
}

/* ============================ scrollbars ====================== */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 4px 0;
}

QScrollBar::handle:vertical {
    background: $BORDER_STRONG;
    min-height: 34px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: $TEXT_FAINT;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0 4px;
}

QScrollBar::handle:horizontal {
    background: $BORDER_STRONG;
    min-width: 34px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: $TEXT_FAINT;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    height: 0;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
}
"""
)


def build_stylesheet(mode: str = "light") -> str:
    """Generate the QSS for the given mode ('light' / 'dark')."""
    return _STYLESHEET_TEMPLATE.substitute(tokens_for(mode))


APP_STYLESHEET = build_stylesheet("light")
DARK_APP_STYLESHEET = build_stylesheet("dark")

# --- Internal: last applied mode (used by widgets that paint themselves). -
_CURRENT_MODE: str = "light"


def current_mode() -> str:
    """Return the last-resolved theme mode ("light" or "dark")."""
    return _CURRENT_MODE


def set_current_mode(mode: str) -> None:
    """Update the cached mode without touching QApplication; useful for tests."""
    global _CURRENT_MODE
    if mode not in ("light", "dark"):
        raise ValueError(f"unknown theme mode: {mode}")
    _CURRENT_MODE = mode


def _detect_system_mode() -> str:
    """Best-effort detection of the OS dark-mode preference.

    Falls back to "light" when PySide6 / Qt is unavailable (e.g. from a unit
    test that imports this module before creating a QApplication).
    """
    try:
        from PySide6.QtCore import QCoreApplication, Qt  # noqa: WPS433
    except Exception:
        return "light"

    app = QCoreApplication.instance()
    if app is None:
        return "light"
    try:
        scheme = app.styleHints().colorScheme()
    except Exception:
        return "light"
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    if scheme == Qt.ColorScheme.Light:
        return "light"
    return "light"


def resolve_mode(mode: str, resolved: Optional[str] = None) -> str:
    """Collapse "system" to a concrete "light" / "dark"."""
    if mode == "system":
        return resolved if resolved in ("light", "dark") else _detect_system_mode()
    return mode


def build_palette(mode: str = "light"):
    """Return a QPalette matching the design tokens for ``mode``."""
    from PySide6.QtGui import QColor, QPalette

    tokens = tokens_for(mode)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(tokens["BG_APP"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(tokens["TEXT"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(tokens["SURFACE"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(tokens["SURFACE_MUTED"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(tokens["TEXT"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(tokens["SURFACE"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(tokens["TEXT_SECONDARY"]))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(tokens["SURFACE"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens["ACCENT"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens["SURFACE"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(tokens["TEXT_STRONG"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(tokens["SURFACE"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(tokens["TEXT_FAINT"]))
    palette.setColor(QPalette.ColorRole.Link, QColor(tokens["ACCENT"]))
    palette.setColor(QPalette.ColorRole.Mid, QColor(tokens["BORDER_STRONG"]))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(tokens["DIVIDER"]))
    palette.setColor(QPalette.ColorRole.Light, QColor(tokens["SURFACE"]))
    palette.setColor(QPalette.ColorRole.Dark, QColor(tokens["BORDER_STRONG"]))

    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(tokens["TEXT_FAINT"]))
    palette.setColor(disabled, QPalette.ColorRole.Base, QColor(tokens["SURFACE_MUTED"]))
    palette.setColor(disabled, QPalette.ColorRole.Button, QColor(tokens["SURFACE_MUTED"]))
    palette.setColor(disabled, QPalette.ColorRole.Highlight, QColor(tokens["BORDER_STRONG"]))
    return palette


def _load_embedded_fonts() -> list[str]:
    """Load bundled fonts (Outfit + JetBrains Mono) if available.

    The repo does not ship the font binaries; on dev machines where the user
    has them installed system-wide this is a no-op. On packaged builds without
    Qt's default font fallback the family still resolves through the cascade
    in ``UI_FONT`` / ``MONO_FONT``.
    """
    try:
        from PySide6.QtGui import QFontDatabase
    except Exception:
        return []
    # No ttf files ship with the package — system fallback covers this. Hook
    # left in place so a future build can ``addApplicationFont`` from disk.
    return []


def apply_app_theme(app, mode: str = "light", resolved: Optional[str] = None) -> str:
    """Apply Fusion, palette, UI font and stylesheet to ``app``.

    ``mode`` may be "system" (auto-detected), "light" or "dark". The resolved
    mode is cached via :func:`current_mode` so widgets that paint themselves
    (GlassFrame / GlassButton / GlassProgressBar / BackdropWidget) can pick
    the matching token set. Returns the resolved mode.

    Safe to call again at runtime: every live widget is re-polished, so a
    theme switch takes effect on the spot instead of at the next restart.
    """
    from PySide6.QtGui import QFont

    concrete = resolve_mode(mode, resolved)
    set_current_mode(concrete)
    _load_embedded_fonts()
    app.setStyle("Fusion")
    app.setPalette(build_palette(concrete))
    app.setFont(QFont(UI_FONT_FAMILY, UI_FONT_POINT_SIZE))
    app.setStyleSheet(build_stylesheet(concrete))
    repolish(app)
    return concrete


def repolish(app) -> None:
    """Force existing widgets to re-read the stylesheet immediately.

    ``setStyleSheet`` alone re-polishes widgets lazily: ones that are not
    currently visible, or that cache colours at paint time (the glass
    widgets), can keep the previous theme until something else makes them
    repaint. Walking the live widget list and re-polishing is what makes a
    theme switch instant.
    """
    for widget in app.allWidgets():
        if widget is None:
            continue
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
    for window in app.topLevelWidgets():
        if window is not None:
            window.update()