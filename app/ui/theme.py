"""Centralized application theme.

The visual system intentionally avoids the earlier near-black shell and uses a
clean light slate palette with a single blue action color. Green, amber, and
red are reserved for semantic log/status cues.
"""
from __future__ import annotations


LOG_LEVEL_COLORS = {
    "Info": "#475569",
    "Warning": "#B45309",
    "Error": "#B91C1C",
    "Success": "#15803D",
}


APP_STYLESHEET = """
QMainWindow,
QWidget {
    background-color: #F5F7FA;
    color: #1F2937;
    font-family: "Segoe UI", "Microsoft YaHei UI";
    font-size: 13px;
}

QLabel#gpuInfoLabel {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 10px 12px;
    color: #1E40AF;
    font-weight: 600;
}

QLabel#mutedLabel {
    color: #64748B;
    font-size: 12px;
}

QTabWidget::pane {
    border: 1px solid #D8E0EA;
    border-radius: 6px;
    top: -1px;
    background-color: #FFFFFF;
}

QTabBar::tab {
    background: #E9EEF5;
    color: #475569;
    padding: 9px 18px;
    border: 1px solid #D8E0EA;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background: #FFFFFF;
    color: #2563EB;
    border-bottom: 1px solid #FFFFFF;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: #DDE6F1;
}

QGroupBox {
    border: 1px solid #D8E0EA;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    background-color: #FFFFFF;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #334155;
}

QLineEdit,
QComboBox,
QSpinBox,
QPlainTextEdit,
QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 5px 7px;
    color: #1F2937;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QPlainTextEdit:focus,
QListWidget:focus {
    border-color: #2563EB;
}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QPlainTextEdit:disabled,
QListWidget:disabled {
    background-color: #EEF2F7;
    color: #94A3B8;
}

QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 6px 12px;
    color: #1F2937;
}

QPushButton:hover {
    background-color: #F1F5F9;
    border-color: #94A3B8;
}

QPushButton:pressed {
    background-color: #E2E8F0;
}

QPushButton:disabled {
    background-color: #F1F5F9;
    color: #94A3B8;
    border-color: #E2E8F0;
}

QPushButton#primaryButton {
    background-color: #2563EB;
    border-color: #2563EB;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
}

QPushButton#primaryButton:pressed {
    background-color: #1E40AF;
}

QProgressBar {
    background-color: #EEF2F7;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    min-height: 16px;
    text-align: center;
    color: #1F2937;
}

QProgressBar::chunk {
    background-color: #2563EB;
    border-radius: 4px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #94A3B8;
    border-radius: 3px;
    background-color: #FFFFFF;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #2563EB;
}
"""
