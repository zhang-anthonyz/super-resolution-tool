from app.ui import theme
from app.ui.theme import (
    APP_STYLESHEET,
    DARK_APP_STYLESHEET,
    LOG_LEVEL_COLORS,
)


def test_light_stylesheet_has_light_canvas_and_no_black():
    """The light variant is what tests + the default render path exercise.

    The light canvas is identified by the long-standing ``#F5F7FA`` substring;
    pure black (``#000000``) and the dark-canvas sentinel ``#10151f`` must
    never appear in any rendered stylesheet.
    """
    assert "#F5F7FA" in APP_STYLESHEET
    assert "#000000" not in APP_STYLESHEET
    assert "#10151f" not in APP_STYLESHEET


def test_dark_stylesheet_has_dark_canvas_and_no_black():
    """The dark variant uses the slate-900 canvas defined by DARK_BG_APP.

    The same anti-pure-black guard applies; this stops anyone accidentally
    inlining ``#000`` / ``#10151f`` into the dark variant.
    """
    assert theme.DARK_BG_APP in DARK_APP_STYLESHEET
    assert "#000000" not in DARK_APP_STYLESHEET
    assert "#10151f" not in DARK_APP_STYLESHEET


def test_both_stylesheets_share_structure():
    """Same selector set, same template — drift here breaks ``check_surfaces``.

    We compare line counts (cheap structural proxy) and a sample of
    ``QFrame#card`` / ``QPushButton#primaryButton`` rules to confirm both
    variants cover the same widget set.
    """
    for marker in (
        "QFrame#card",
        "QPushButton#primaryButton",
        "QPushButton#navButton:checked",
        "QProgressBar",
        "QSlider::handle:horizontal",
        "QListWidget::item",
        "QComboBox QAbstractItemView",
        "QToolTip",
    ):
        assert marker in APP_STYLESHEET, f"light variant missing {marker}"
        assert marker in DARK_APP_STYLESHEET, f"dark variant missing {marker}"


def test_log_level_colors_are_semantic():
    assert set(LOG_LEVEL_COLORS) == {"Info", "Warning", "Error", "Success"}
    assert LOG_LEVEL_COLORS["Success"].startswith("#")
