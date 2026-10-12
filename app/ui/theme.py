"""Centralized application theme: design tokens + Qt stylesheet.

The visual system is a modern light surface stack: a soft blue-grey canvas
(``#F5F7FA``), white elevated cards, and a single blue action colour. Green,
amber and red are reserved for semantic status feedback only.

Colours are declared once as module constants and injected into the Qt
stylesheet template, so Python-side painting (log colours, progress colours)
and QSS never drift apart.
"""
from __future__ import annotations

from string import Template

# --- Canvas & surfaces ----------------------------------------------------
BG_APP = "#F5F7FA"
SURFACE = "#FFFFFF"
SURFACE_MUTED = "#F8FAFC"
SURFACE_HOVER = "#F1F5F9"
SURFACE_PRESSED = "#E7EDF5"
BORDER = "#E6EBF3"
BORDER_STRONG = "#D6DEEA"
DIVIDER = "#EEF2F8"

# --- Typography colours ---------------------------------------------------
TEXT = "#1F2937"
TEXT_STRONG = "#1E293B"
TEXT_SECONDARY = "#4B5C72"
TEXT_MUTED = "#7C8AA0"
TEXT_FAINT = "#9AA7BA"

# --- Accent ---------------------------------------------------------------
ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
ACCENT_PRESSED = "#1E40AF"
ACCENT_SOFT = "#EFF6FF"
ACCENT_SOFT_BORDER = "#BFDBFE"
ACCENT_SOFT_TEXT = "#1E40AF"
ACCENT_DISABLED = "#B9CDF6"

# --- Semantic -------------------------------------------------------------
SUCCESS = "#16A34A"
WARNING = "#D97706"
DANGER = "#DC2626"
NEUTRAL = "#94A3B8"
DANGER_SOFT = "#FEF2F2"
DANGER_SOFT_BORDER = "#FCA5A5"

# --- Fonts ----------------------------------------------------------------
UI_FONT = '"Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI"'
UI_FONT_FAMILY = "Segoe UI Variable Text"
UI_FONT_POINT_SIZE = 10
MONO_FONT = '"Cascadia Mono", "Consolas", "Microsoft YaHei UI"'

#: Log level colours are semantic status cues, not decoration.
LOG_LEVEL_COLORS = {
    "Info": "#475569",
    "Warning": "#B45309",
    "Error": "#B91C1C",
    "Success": "#15803D",
}

_STYLESHEET_TEMPLATE = Template(
    """
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
    background-color: $BG_APP;
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
    background-color: $SURFACE;
    border: none;
    border-right: 1px solid $BORDER;
}

QLabel#brandMark {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #3B82F6, stop:1 $ACCENT_PRESSED);
    color: $SURFACE;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
}

QLabel#brandTitle {
    color: $TEXT_STRONG;
    font-size: 15px;
    font-weight: 700;
}

QLabel#brandSubtitle {
    color: $TEXT_FAINT;
    font-size: 11px;
}

QLabel#navSection {
    color: $TEXT_FAINT;
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
    background-color: $SURFACE_HOVER;
    color: $TEXT_STRONG;
}

QPushButton#navButton:checked {
    background-color: $ACCENT_SOFT;
    border-left: 3px solid $ACCENT;
    color: $ACCENT_SOFT_TEXT;
    font-weight: 600;
}

QFrame#sidebarFooter {
    background-color: $SURFACE_MUTED;
    border: 1px solid $DIVIDER;
    border-radius: 10px;
}

QLabel#sidebarFooterText {
    color: $TEXT_MUTED;
    font-size: 11px;
}

/* ============================ header ========================== */
QFrame#headerCard {
    background-color: $SURFACE;
    border: 1px solid $BORDER;
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
    background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 12px;
}

QLabel#cardTitle {
    color: $TEXT_STRONG;
    font-size: 13px;
    font-weight: 600;
}

QLabel#cardHint {
    color: $TEXT_FAINT;
    font-size: 11px;
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
    background-color: $SURFACE;
    border: 1px solid $BORDER_STRONG;
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
    border-color: #C3D0E1;
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
    background-color: $SURFACE;
    border: 1px solid $BORDER_STRONG;
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: $ACCENT_SOFT;
    selection-color: $ACCENT_SOFT_TEXT;
}

/* Room for the hand-drawn chevron indicator from widgets.ChevronComboBox. */
QComboBox {
    padding-right: 28px;
}

QComboBox QAbstractItemView::item {
    min-height: 26px;
    padding: 2px 8px;
    border-radius: 6px;
}

/* ============================ buttons ========================= */
QPushButton {
    background-color: $SURFACE;
    border: 1px solid $BORDER_STRONG;
    border-radius: 8px;
    padding: 7px 14px;
    color: $TEXT_SECONDARY;
    font-weight: 500;
}

QPushButton:hover {
    background-color: $SURFACE_MUTED;
    border-color: #C3D0E1;
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
    background-color: $SURFACE_HOVER;
    border: 1px solid transparent;
    color: $TEXT_SECONDARY;
}

QPushButton#softButton:hover {
    background-color: $SURFACE_PRESSED;
    border-color: transparent;
    color: $TEXT_STRONG;
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
    background-color: $SURFACE_MUTED;
    border: 1px solid $BORDER;
    border-radius: 8px;
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
    background-color: $SURFACE_PRESSED;
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
    background: $SURFACE_PRESSED;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: $ACCENT;
    border-radius: 2px;
}

QSlider::add-page:horizontal {
    background: $SURFACE_PRESSED;
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
    background-color: $SURFACE;
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
    background-color: $SURFACE_MUTED;
    color: $TEXT_STRONG;
}

QListWidget::item:selected {
    background-color: $ACCENT_SOFT;
    color: $ACCENT_SOFT_TEXT;
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

APP_STYLESHEET = _STYLESHEET_TEMPLATE.substitute(
    BG_APP=BG_APP,
    SURFACE=SURFACE,
    SURFACE_MUTED=SURFACE_MUTED,
    SURFACE_HOVER=SURFACE_HOVER,
    SURFACE_PRESSED=SURFACE_PRESSED,
    BORDER=BORDER,
    BORDER_STRONG=BORDER_STRONG,
    DIVIDER=DIVIDER,
    TEXT=TEXT,
    TEXT_STRONG=TEXT_STRONG,
    TEXT_SECONDARY=TEXT_SECONDARY,
    TEXT_MUTED=TEXT_MUTED,
    TEXT_FAINT=TEXT_FAINT,
    ACCENT=ACCENT,
    ACCENT_HOVER=ACCENT_HOVER,
    ACCENT_PRESSED=ACCENT_PRESSED,
    ACCENT_SOFT=ACCENT_SOFT,
    ACCENT_SOFT_BORDER=ACCENT_SOFT_BORDER,
    ACCENT_SOFT_TEXT=ACCENT_SOFT_TEXT,
    ACCENT_DISABLED=ACCENT_DISABLED,
    DANGER=DANGER,
    DANGER_SOFT=DANGER_SOFT,
    DANGER_SOFT_BORDER=DANGER_SOFT_BORDER,
    NEUTRAL=NEUTRAL,
    UI_FONT=UI_FONT,
    MONO_FONT=MONO_FONT,
)


def build_palette():
    """Return a QPalette matching the design tokens.

    The stylesheet covers the shell itself; the palette keeps native surfaces
    (message boxes, menus, tooltips) on the same light system.
    """
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_APP))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_MUTED))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_SECONDARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(TEXT_STRONG))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Mid, QColor(BORDER_STRONG))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(DIVIDER))
    palette.setColor(QPalette.ColorRole.Light, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Dark, QColor(BORDER_STRONG))

    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(TEXT_FAINT))
    palette.setColor(disabled, QPalette.ColorRole.Base, QColor(SURFACE_MUTED))
    palette.setColor(disabled, QPalette.ColorRole.Button, QColor(SURFACE_MUTED))
    palette.setColor(disabled, QPalette.ColorRole.Highlight, QColor(BORDER_STRONG))
    return palette


def apply_app_theme(app) -> None:
    """Apply the Fusion style, palette, UI font and stylesheet to an app."""
    from PySide6.QtGui import QFont

    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setFont(QFont(UI_FONT_FAMILY, UI_FONT_POINT_SIZE))
    app.setStyleSheet(APP_STYLESHEET)
