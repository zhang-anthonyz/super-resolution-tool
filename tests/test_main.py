"""Entry-point helpers.

The packaged build runs windowed: launched by double-click, ``sys.stdout`` and
``sys.stderr`` are ``None``, so any print in the startup/self-check path has to
survive that (and the outcome has to be readable from the log file instead).
"""
import sys

from app.main import _is_verbose, _safe_print


def test_safe_print_survives_missing_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    _safe_print("should not raise")
    _safe_print("nor this", file=sys.stderr)


def test_safe_print_uses_stdout_by_default(capsys):
    _safe_print("自检通过")

    assert capsys.readouterr().out.strip() == "自检通过"


def test_safe_print_honours_the_requested_stream(capsys):
    _safe_print("自检失败：模型缺失", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.err.strip() == "自检失败：模型缺失"
    assert captured.out == ""


def test_verbose_flags():
    assert _is_verbose(["-v"])
    assert _is_verbose(["--verbose"])
    assert _is_verbose(["app/main.py", "--debug"])
    assert not _is_verbose([])
    assert not _is_verbose(["app/main.py"])
