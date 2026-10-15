"""Centralized application theme: design tokens + Qt stylesheet.

Two parallel palettes share a single stylesheet template. The light variant
keeps the existing tokens (still required for the ``#F5F7FA`` substring in
``test_theme.py``); the dark variant is a parallel set tuned for a slate-900
canvas with vibrant accents. Both palettes declare the same set of
substitution keys, so the QSS structure never drifts between modes.

Public surface (kept stable for backward compatibility with imports):

* Token constants: ``BG_APP`` / ``SURFACE`` / ``ACCENT`` …  (light variant)
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
BG_APP = "#F5F7FA"
SURFACE = "#FFFFFF"
SURFACE_MUTED = "#F8FAFC"
SURFACE_HOVER = "#F1F5F9"
SURFACE_PRESSED = "#E7EDF5"
BORDER = "#E6EBF3"
BORDER_STRONG = "#D6DEEA"
DIVIDER = "#EEF2F8"

TEXT = "#1F2937"
TEXT_STRONG = "#1E293B"
TEXT_SECONDARY = "#4B5C72"
#: Quiet-but-readable text: hints, captions, secondary metadata. Every variant
#: clears WCAG AA (4.5:1) against the surfaces it is drawn on — the glass card
#: is translucent, so the test uses the lightest backdrop the mesh produces.
TEXT_MUTED = "#526078"
#: Disabled labels, input placeholders and decorative glyphs. Deliberately below
#: AA: WCAG exempts inactive controls, and a greyed-out control has to look
#: greyed out. Never use it for text the user must read.
TEXT_FAINT = "#9AA7BA"

ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
ACCENT_PRESSED = "#1E40AF"
ACCENT_SOFT = "#EFF6FF"
ACCENT_SOFT_BORDER = "#BFDBFE"
ACCENT_SOFT_TEXT = "#1E40AF"
ACCENT_DISABLED = "#B9CDF6"
#: Highlight color of the gradient that flows across the progress-bar chunk.
ACCENT_LIGHT = "#60A5FA"

#: Status colours, also used as small text (toast glyphs, inline status), so the
#: light set is the darker end of each hue rather than the vivid mid-tone.
SUCCESS = "#15803D"
WARNING = "#B45309"
DANGER = "#C81E1E"
NEUTRAL = "#94A3B8"
DANGER_SOFT = "#FEF2F2"
DANGER_SOFT_BORDER = "#FCA5A5"

#: Glass-surface alpha (card / sidebar over the mesh-gradient backdrop).
LIGHT_CARD_ALPHA = "0.62"
LIGHT_SIDEBAR_ALPHA = "0.70"
LIGHT_HEADER_ALPHA = "0.70"
LIGHT_HOVER_ALPHA = "0.74"
LIGHT_PRESSED_ALPHA = "0.80"
#: Inner-edge highlight on every glass surface (top-left rim).
LIGHT_GLASS_RIM = "rgba(255, 255, 255, 0.85)"
#: Subtle border between the transparent card and the backdrop.
LIGHT_GLASS_BORDER = "rgba(148, 163, 184, 0.40)"

# ------------------------------------------------------------------ DARK ---
#: Slate-900-ish canvas; deliberately not ``#000000`` nor ``#10151f`` (those
#: are forbidden by ``tests/test_theme.py``).
DARK_BG_APP = "#0F172A"
DARK_SURFACE = "#1E293B"
DARK_SURFACE_MUTED = "#172033"
DARK_SURFACE_HOVER = "#22304A"
DARK_SURFACE_PRESSED = "#2A3957"
DARK_BORDER = "#27344B"
DARK_BORDER_STRONG = "#3A4865"
DARK_DIVIDER = "#1F2A3D"

DARK_TEXT = "#E2E8F0"
DARK_TEXT_STRONG = "#F8FAFC"
DARK_TEXT_SECONDARY = "#B6C2D4"
DARK_TEXT_MUTED = "#8B97AB"
DARK_TEXT_FAINT = "#64748B"

DARK_ACCENT = "#60A5FA"
DARK_ACCENT_HOVER = "#3B82F6"
DARK_ACCENT_PRESSED = "#2563EB"
DARK_ACCENT_SOFT = "#172554"
DARK_ACCENT_SOFT_BORDER = "#3B82F6"
DARK_ACCENT_SOFT_TEXT = "#93C5FD"
DARK_ACCENT_DISABLED = "#3F4D6E"
DARK_ACCENT_LIGHT = "#93C5FD"

DARK_SUCCESS = "#4ADE80"
DARK_WARNING = "#FBBF24"
DARK_DANGER = "#F87171"
DARK_NEUTRAL = "#64748B"
DARK_DANGER_SOFT = "#3B1118"
DARK_DANGER_SOFT_BORDER = "#7F1D1D"

DARK_CARD_ALPHA = "0.62"
DARK_SIDEBAR_ALPHA = "0.72"
DARK_HEADER_ALPHA = "0.65"
DARK_HOVER_ALPHA = "0.74"
DARK_PRESSED_ALPHA = "0.80"
DARK_GLASS_RIM = "rgba(255, 255, 255, 0.10)"
DARK_GLASS_BORDER = "rgba(255, 255, 255, 0.08)"

# --- Fonts ----------------------------------------------------------------
UI_FONT = '"Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI"'
UI_FONT_FAMILY = "Segoe UI Variable Text"
UI_FONT_POINT_SIZE = 10
MONO_FONT = '"Cascadia Mono", "Consolas", "Microsoft YaHei UI"'

#: Log level colours are semantic status cues, not decoration. Each variant is
#: tuned for the surface it is read on (light: near-white log panel, dark:
#: ``DARK_SURFACE_MUTED``) and clears WCAG AA (4.5:1) against it.
LOG_LEVEL_COLORS = {
    "Info": "#475569",
    "Warning": "#B45309",
    "Error": "#B91C1C",
    "Success": "#15803D",
}

DARK_LOG_LEVEL_COLORS = {
    "Info": "#94A3B8",
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
    # Glass RGB: the underlying colour rendered through the alpha mask.
    # Light glass = white over a light backdrop.
    "GLASS_RGB": "255, 255, 255",
    "CARD_ALPHA": LIGHT_CARD_ALPHA,
    "SIDEBAR_ALPHA": LIGHT_SIDEBAR_ALPHA,
    "HEADER_ALPHA": LIGHT_HEADER_ALPHA,
    "HOVER_ALPHA": LIGHT_HOVER_ALPHA,
    "PRESSED_ALPHA": LIGHT_PRESSED_ALPHA,
    "GLASS_RIM": LIGHT_GLASS_RIM,
    "GLASS_BORDER": LIGHT_GLASS_BORDER,
    "INPUT_ALPHA": "0.62",
    "BUTTON_ALPHA": "0.62",
    "LIST_ALPHA": "0.62",
    "CHECK_ALPHA": "0.62",
    "GROOVE_ALPHA": "0.22",
    "TRACK_ALPHA": "0.25",
    "ITEM_HOVER_ALPHA": "0.85",
    "FOOTER_RGB": "148, 163, 184",
    "FOOTER_ALPHA": "0.18",
    "PROGRESS_BG_RGB": "148, 163, 184",
    "PROGRESS_BG_ALPHA": "0.22",
    "CHECKBOX_BG_RGB": "255, 255, 255",
    "CHECKBOX_BG_ALPHA": "0.62",
    "BRAND_STOP_0": "#60A5FA",
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
    # Glass RGB: dark glass over a dark backdrop. The same alpha gives a
    # slightly heavier feel than light glass, which we tuned to match.
    "GLASS_RGB": "30, 41, 59",
    "CARD_ALPHA": DARK_CARD_ALPHA,
    "SIDEBAR_ALPHA": DARK_SIDEBAR_ALPHA,
    "HEADER_ALPHA": DARK_HEADER_ALPHA,
    "HOVER_ALPHA": DARK_HOVER_ALPHA,
    "PRESSED_ALPHA": DARK_PRESSED_ALPHA,
    "GLASS_RIM": DARK_GLASS_RIM,
    "GLASS_BORDER": DARK_GLASS_BORDER,
    "INPUT_ALPHA": "0.55",
    "BUTTON_ALPHA": "0.50",
    "LIST_ALPHA": "0.55",
    "CHECK_ALPHA": "0.55",
    "GROOVE_ALPHA": "0.22",
    "TRACK_ALPHA": "0.25",
    "ITEM_HOVER_ALPHA": "0.30",
    "FOOTER_RGB": "148, 163, 184",
    "FOOTER_ALPHA": "0.18",
    "PROGRESS_BG_RGB": "148, 163, 184",
    "PROGRESS_BG_ALPHA": "0.14",
    "CHECKBOX_BG_RGB": "30, 41, 59",
    "CHECKBOX_BG_ALPHA": "0.55",
    "BRAND_STOP_0": "#38BDF8",
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
/* No blanket "QWidget" rule: Qt auto-fills any styled widget that has no
   background with the palette window colour, which turned every label and
   row container into an opaque grey block inside the white cards. */
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
    border: 1px solid $TEXT_STRONG;
    padding: 4px 8px;
}

/* ============================ sidebar ========================= */
QFrame#sidebar {
    background-color: rgba($GLASS_RGB, $SIDEBAR_ALPHA);
    border: none;
    border-right: 1px solid $GLASS_BORDER;
}

QLabel#brandMark {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 $BRAND_STOP_0, stop:1 $ACCENT_PRESSED);
    color: $SURFACE;
    border-radius: 11px;
    font-size: 14px;
    font-weight: 700;
}

QLabel#brandTitle {
    color: $TEXT_STRONG;
    font-size: 15px;
    font-weight: 700;
}

QLabel#brandSubtitle {
    color: $TEXT_MUTED;
    font-size: 11px;
}

QLabel#navSection {
    color: $TEXT_MUTED;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 6px;
}

QPushButton#navButton {
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 8px;
    padding: 8px 12px;
    color: $TEXT_SECONDARY;
    font-size: 13px;
    font-weight: 500;
    text-align: left;
}

QPushButton#navButton:hover {
    background-color: rgba($GLASS_RGB, $HOVER_ALPHA);
    color: $TEXT_STRONG;
}

QPushButton#navButton:checked {
    background-color: $ACCENT_SOFT;
    border-left: 3px solid $ACCENT;
    color: $ACCENT_SOFT_TEXT;
    font-weight: 600;
}

/* Keyboard focus: the nav rail is reachable with Tab, and the checked state
   alone does not tell the user which row the focus is on. The bar is the pale
   accent so a focused-but-unselected row cannot be mistaken for the current
   page, which owns the solid one. */
QPushButton#navButton:focus {
    background-color: $SURFACE_HOVER;
    border-left: 3px solid $ACCENT_SOFT_BORDER;
    color: $TEXT_STRONG;
}

QPushButton#navButton:checked:focus {
    background-color: $ACCENT_SOFT;
    border-left: 3px solid $ACCENT;
    color: $ACCENT_SOFT_TEXT;
}

QFrame#sidebarFooter {
    background-color: rgba($FOOTER_RGB, $FOOTER_ALPHA);
    border: 1px solid $GLASS_BORDER;
    border-radius: 10px;
}

QLabel#sidebarFooterText {
    color: $TEXT_MUTED;
    font-size: 11px;
}

/* ============================ header ========================== */
QFrame#headerCard {
    background-color: rgba($GLASS_RGB, $HEADER_ALPHA);
    border: 1px solid $GLASS_BORDER;
    border-radius: 14px;
}

QLabel#headerTitle {
    color: $TEXT_STRONG;
    font-size: 20px;
    font-weight: 700;
}

QLabel#headerSubtitle {
    color: $TEXT_MUTED;
    font-size: 12px;
}

QLabel#gpuInfoLabel {
    background-color: $ACCENT_SOFT;
    border: 1px solid $ACCENT_SOFT_BORDER;
    border-radius: 10px;
    padding: 10px 14px;
    color: $ACCENT_SOFT_TEXT;
    font-size: 12px;
    font-weight: 600;
}

/* ============================ cards =========================== */
QFrame#card {
    background-color: rgba($GLASS_RGB, $CARD_ALPHA);
    border: none;
    border-radius: 12px;
}

QLabel#cardTitle {
    color: $TEXT_STRONG;
    font-size: 13px;
    font-weight: 600;
}

QLabel#cardHint {
    color: $TEXT_MUTED;
    font-size: 11px;
}

QFrame#cardDivider {
    background-color: $GLASS_BORDER;
    border: none;
}

/* ============================ inputs ========================== */
QLineEdit,
QComboBox,
QSpinBox,
QPlainTextEdit,
QListWidget {
    background-color: rgba($GLASS_RGB, $INPUT_ALPHA);
    border: 1px solid rgba($GLASS_RGB, 0.40);
    border-radius: 8px;
    padding: 6px 10px;
    color: $TEXT;
    selection-background-color: $ACCENT;
    selection-color: $SURFACE;
}

QLineEdit:hover,
QComboBox:hover,
QSpinBox:hover,
QPlainTextEdit:hover {
    border: 1px solid rgba($ACCENT, 0.45);
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QPlainTextEdit:focus,
QListWidget:focus {
    border: 1px solid $ACCENT;
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
    background-color: rgba($GLASS_RGB, 0.96);
    border: 1px solid rgba($GLASS_RGB, 0.55);
    border-radius: 10px;
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
    border-radius: 7px;
    color: $TEXT_SECONDARY;
}

QComboBox QAbstractItemView::item:hover {
    background-color: rgba($GLASS_RGB, $ITEM_HOVER_ALPHA);
    color: $TEXT_STRONG;
}

/* ============================ buttons ========================= */
QPushButton {
    background-color: rgba($GLASS_RGB, $BUTTON_ALPHA);
    border: 1px solid $BORDER_STRONG;
    border-radius: 8px;
    padding: 7px 14px;
    color: $TEXT_SECONDARY;
    font-weight: 500;
}

QPushButton:hover {
    background-color: $SURFACE_HOVER;
    border-color: $ACCENT;
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
    border-radius: 8px;
    padding: 9px 26px;
    color: $SURFACE;
    font-weight: 600;
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
    background-color: rgba($GLASS_RGB, $BUTTON_ALPHA);
    border: 1px solid $BORDER_STRONG;
    color: $TEXT_SECONDARY;
}

QPushButton#softButton:hover {
    background-color: $SURFACE_HOVER;
    color: $TEXT_STRONG;
    border-color: $ACCENT;
}

QPushButton#ghostButton {
    background-color: transparent;
    border: 1px solid transparent;
    color: $TEXT_MUTED;
    padding: 5px 10px;
}

QPushButton#ghostButton:hover {
    background-color: rgba($GLASS_RGB, $HOVER_ALPHA);
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
    border-radius: 8px;
    padding: 5px 14px;
    color: $TEXT_SECONDARY;
    font-size: 12px;
    font-weight: 500;
}

QPushButton#segmentButton:hover {
    background-color: rgba($GLASS_RGB, $HOVER_ALPHA);
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
    background-color: rgba($PROGRESS_BG_RGB, $PROGRESS_BG_ALPHA);
    border: none;
    border-radius: 5px;
    min-height: 10px;
    max-height: 10px;
}

QProgressBar::chunk {
    background-color: $ACCENT;
    border-radius: 5px;
}

/* ============================ sliders ========================= */
QSlider {
    background: transparent;
    min-height: 24px;
}

QSlider::groove:horizontal {
    height: 4px;
    background: rgba(148, 163, 184, $TRACK_ALPHA);
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: $ACCENT;
    border-radius: 2px;
}

QSlider::add-page:horizontal {
    background: rgba(148, 163, 184, $TRACK_ALPHA);
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
    width: 18px;
    height: 18px;
    border: 1px solid $BORDER_STRONG;
    border-radius: 5px;
    background-color: rgba($CHECKBOX_BG_RGB, $CHECKBOX_BG_ALPHA);
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
    border-radius: 7px;
    color: $TEXT_SECONDARY;
}

QListWidget::item:hover {
    background-color: rgba($GLASS_RGB, $ITEM_HOVER_ALPHA);
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
    border-radius: 10px;
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
}

QLabel#valueStrong {
    color: $TEXT_STRONG;
    font-weight: 600;
}

QLabel#valueBadge {
    background-color: $ACCENT_SOFT;
    border: 1px solid $ACCENT_SOFT_BORDER;
    border-radius: 6px;
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
}

QLabel#videoMetaLabel {
    font-size: 12px;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid $BORDER;
    background: rgba(0, 0, 0, 0);
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
    color: $ACCENT_SOFT_TEXT;
    border-color: $DANGER_SOFT_BORDER;
}

QLabel#logTitle {
    color: $TEXT_STRONG;
    font-size: 13px;
    font-weight: 600;
}

/* ============================ log view ======================== */
QPlainTextEdit#logView {
    background-color: $SURFACE_MUTED;
    border: 1px solid $BORDER;
    border-radius: 10px;
    padding: 8px 10px;
    color: $TEXT_SECONDARY;
    font-family: $MONO_FONT;
    font-size: 12px;
}

/* ============================ scrollbars ====================== */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px 0;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    min-height: 34px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #AAB9CC;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 0 2px;
}

QScrollBar::handle:horizontal {
    background: #CBD5E1;
    min-width: 34px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #AAB9CC;
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
