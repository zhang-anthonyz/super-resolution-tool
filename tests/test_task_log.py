"""Step-logging helpers: console records, UI sink routing, formatting."""
import logging

import pytest

from app.core.task_log import (
    TaskLog,
    configure_logging,
    format_bytes,
    format_duration,
)


def test_format_bytes_scales_units():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024**2) == "5.0 MB"
    assert format_bytes(3 * 1024**3) == "3.0 GB"


def test_format_duration_switches_to_minutes():
    assert format_duration(0.4234) == "0.42s"
    assert format_duration(59.99).endswith("s")
    assert format_duration(65.0) == "1m 05.0s"


def test_step_logger_writes_to_console_and_sink(caplog):
    lines = []
    log = TaskLog("app.test.task_log", sink=lambda level, text: lines.append((level, text)))

    with caplog.at_level(logging.INFO, logger="app.test.task_log"):
        log.info("解码完成 %sx%s", 64, 32)
        log.warning("跳过已存在的输出")

    assert [record.getMessage() for record in caplog.records] == [
        "解码完成 64x32",
        "跳过已存在的输出",
    ]
    assert lines == [
        ("Info", "解码完成 64x32"),
        ("Warning", "跳过已存在的输出"),
    ]


def test_debug_lines_stay_out_of_the_ui(caplog):
    """Per-tile debug output would drown the run log, so it is console-only."""
    lines = []
    log = TaskLog("app.test.task_log", sink=lambda level, text: lines.append((level, text)))

    with caplog.at_level(logging.DEBUG, logger="app.test.task_log"):
        log.debug("tile 7/48 · 62ms")

    assert caplog.records[0].levelno == logging.DEBUG
    assert lines == []


def test_success_level_maps_to_the_success_colour():
    lines = []
    log = TaskLog("app.test.task_log", sink=lambda level, text: lines.append((level, text)))

    log.success("批量完成")

    assert lines == [("Success", "批量完成")]


def test_bind_adds_a_sink_without_touching_the_source():
    lines = []
    base = TaskLog("app.test.task_log")

    bound = base.bind(lambda level, text: lines.append(text))
    base.info("只有控制台")
    bound.info("控制台 + 界面")

    assert base.sink is None
    assert bound.sink is not None
    assert lines == ["控制台 + 界面"]


@pytest.fixture(autouse=True)
def _restore_app_logger():
    """configure_logging mutates a process-global logger; keep tests isolated."""
    namespace = logging.getLogger("app")
    handlers = list(namespace.handlers)
    level, propagate = namespace.level, namespace.propagate
    yield
    for handler in list(namespace.handlers):
        if handler not in handlers:
            namespace.removeHandler(handler)
    namespace.setLevel(level)
    namespace.propagate = propagate


def test_configure_logging_is_idempotent():
    namespace = logging.getLogger("app")

    configure_logging("info", log_dir="", force=True)
    first = list(namespace.handlers)
    configure_logging("debug", log_dir="")
    second = list(namespace.handlers)

    assert first == second, "repeated configuration must not stack handlers"
    assert namespace.level == logging.DEBUG
    assert namespace.propagate is False


def test_configure_logging_honours_the_environment(monkeypatch):
    monkeypatch.setenv("SRT_LOG_LEVEL", "debug")

    configure_logging()

    assert logging.getLogger("app").level == logging.DEBUG


def test_configure_logging_survives_an_unwritable_log_directory():
    """A packaged install may sit in a read-only location; logging still works."""
    configure_logging("info", log_dir="\0invalid\0path", force=True)

    assert logging.getLogger("app").handlers
