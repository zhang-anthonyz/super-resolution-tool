from app.ui.theme import APP_STYLESHEET, LOG_LEVEL_COLORS


def test_theme_is_light_not_pure_black():
    assert "#F5F7FA" in APP_STYLESHEET
    assert "#10151f" not in APP_STYLESHEET
    assert "#000000" not in APP_STYLESHEET


def test_log_level_colors_are_semantic():
    assert set(LOG_LEVEL_COLORS) == {"Info", "Warning", "Error", "Success"}
    assert LOG_LEVEL_COLORS["Success"].startswith("#")
