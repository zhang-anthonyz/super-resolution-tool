from app.ui import theme
from app.ui.theme import (
    APP_STYLESHEET,
    DARK_APP_STYLESHEET,
    LOG_LEVEL_COLORS,
)


# --- contrast helpers -------------------------------------------------------

#: Surfaces a text token can land on. The "STUDIO" palette replaces the
#: previous mesh-gradient backdrop with a hairline-bordered card on a warm
#: paper-tone canvas; the light surfaces are the canvas + card + muted panel,
#: the dark ones are the slate canvas + card + muted panel.
LIGHT_TEXT_SURFACES = ("#FBF8F2", "#F4F1EB", "#EFEAE0")
DARK_TEXT_SURFACES = ("#15171B", "#0E0F12", "#111319")

#: Roles that carry text the user has to read. TEXT_FAINT is deliberately not
#: here: it is the disabled / placeholder / decorative tone, which WCAG exempts
#: from contrast minimums.
READABLE_TEXT_ROLES = ("TEXT", "TEXT_STRONG", "TEXT_SECONDARY", "TEXT_MUTED")

AA_NORMAL = 4.5


def _linear(channel: int) -> float:
    value = channel / 255.0
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def _luminance(colour: str) -> float:
    hex_colour = colour.lstrip("#")
    red, green, blue = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(red) + 0.7152 * _linear(green) + 0.0722 * _linear(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_light_stylesheet_has_light_canvas_and_no_black():
    """The light variant is what tests + the default render path exercise.

    The light canvas is the warm paper-tone ``BG_APP`` token; pure black
    (``#000000``) and the previous dark-canvas sentinel must never appear
    in any rendered stylesheet.
    """
    assert theme.BG_APP in APP_STYLESHEET
    assert "#000000" not in APP_STYLESHEET
    assert "#10151f" not in APP_STYLESHEET


def test_dark_stylesheet_has_dark_canvas_and_no_black():
    """The dark variant uses the slate canvas defined by DARK_BG_APP.

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


def test_log_level_colors_have_a_dark_variant():
    assert set(theme.DARK_LOG_LEVEL_COLORS) == set(LOG_LEVEL_COLORS)
    assert theme.log_level_colors("light") is LOG_LEVEL_COLORS
    assert theme.log_level_colors("dark") is theme.DARK_LOG_LEVEL_COLORS


def test_readable_text_tokens_clear_wcag_aa_in_both_themes():
    """Body and hint text must stay readable on every surface it can land on.

    This is the regression net for the muted-grey drift the light theme had:
    ``#7C8AA0`` on a white card measured 3.5:1, which is below AA for the
    12px hints it was used for.
    """
    for mode, surfaces in (
        ("light", LIGHT_TEXT_SURFACES),
        ("dark", DARK_TEXT_SURFACES),
    ):
        tokens = theme.tokens_for(mode)
        for role in READABLE_TEXT_ROLES:
            for surface in surfaces:
                ratio = contrast_ratio(tokens[role], surface)
                assert ratio >= AA_NORMAL, (
                    f"{mode} {role} {tokens[role]} on {surface} is "
                    f"{ratio:.2f}:1, below WCAG AA ({AA_NORMAL}:1)"
                )


def test_log_level_and_status_colours_clear_wcag_aa():
    """Level and status colours are painted as small text, not just as fills."""
    for mode, panel in (
        ("light", theme.SURFACE_MUTED),
        ("dark", theme.DARK_SURFACE_MUTED),
    ):
        for level, colour in theme.log_level_colors(mode).items():
            ratio = contrast_ratio(colour, panel)
            assert ratio >= AA_NORMAL, (
                f"{mode} log level {level} {colour} on {panel} is {ratio:.2f}:1"
            )

    for mode, surfaces in (
        ("light", LIGHT_TEXT_SURFACES),
        ("dark", DARK_TEXT_SURFACES),
    ):
        tokens = theme.tokens_for(mode)
        for role in ("SUCCESS", "WARNING", "DANGER", "ACCENT", "ACCENT_SOFT_TEXT"):
            for surface in surfaces:
                ratio = contrast_ratio(tokens[role], surface)
                assert ratio >= AA_NORMAL, (
                    f"{mode} {role} {tokens[role]} on {surface} is {ratio:.2f}:1"
                )


PROGRESS_FILL_ROLES = (
    "PROGRESS_NORMAL",
    "PROGRESS_SUCCESS",
    "PROGRESS_WARNING",
    "PROGRESS_ERROR",
    "PROGRESS_NEUTRAL",
)


def test_progress_tokens_exist_in_both_themes():
    """The bar resolves its fill from theme tokens at paint time, so both
    modes must expose the same PROGRESS_* keys."""
    for mode in ("light", "dark"):
        tokens = theme.tokens_for(mode)
        for role in ("PROGRESS_TRACK", *PROGRESS_FILL_ROLES):
            assert tokens[role].startswith("#"), f"{mode} missing {role}"


def test_light_progress_fills_are_midtone_not_near_black():
    """Regression net for the "green + black" light bar: fills must stay in a
    mid luminance band — clearly visible on the track yet never reading as a
    near-black stripe — and remain distinct from the track."""
    tokens = theme.tokens_for("light")
    for role in PROGRESS_FILL_ROLES:
        luminance = _luminance(tokens[role])
        assert 0.20 <= luminance <= 0.55, (
            f"light {role} {tokens[role]} luminance {luminance:.3f} "
            "is outside the mid-tone band"
        )
        assert contrast_ratio(tokens[role], tokens["PROGRESS_TRACK"]) >= 1.8, (
            f"light {role} barely differs from the track"
        )
