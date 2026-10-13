"""Tests for the BasicSR/ESRGAN key remapping used by scripts/export_onnx.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_onnx import (  # noqa: E402
    detect_rrdb_blocks,
    is_basicsr_state,
    remap_state_dict,
)


def _basicsr_state(num_block: int = 2) -> dict:
    state: dict = {
        "model.0.weight": 1,
        "model.0.bias": 2,
    }
    for index in range(num_block):
        for rdb in (1, 2, 3):
            for conv in (1, 2):
                state[f"model.1.sub.{index}.RDB{rdb}.conv{conv}.0.weight"] = 3
                state[f"model.1.sub.{index}.RDB{rdb}.conv{conv}.0.bias"] = 4
    state[f"model.1.sub.{num_block}.weight"] = 5
    state[f"model.1.sub.{num_block}.bias"] = 6
    state["model.3.weight"] = 7
    state["model.6.weight"] = 8
    state["model.8.weight"] = 9
    state["model.10.weight"] = 10
    return state


def test_is_basicsr_state_detects_legacy_naming():
    assert is_basicsr_state(_basicsr_state()) is True
    assert is_basicsr_state({"conv_first.weight": 1}) is False


def test_detect_rrdb_blocks_counts_rdb_layers_only():
    assert detect_rrdb_blocks(_basicsr_state(num_block=23)) == 23
    assert detect_rrdb_blocks({"model.1.sub.5.weight": 1}) == 0


def test_remap_state_dict_translates_every_module():
    num_block = 2
    remapped = remap_state_dict(_basicsr_state(num_block), num_block)

    assert remapped["conv_first.weight"] == 1
    assert remapped["body.0.rdb1.conv1.weight"] == 3
    assert remapped["body.1.rdb3.conv2.bias"] == 4
    # sub.N（尾部卷积）是主干卷积，不是第 N 个 RRDB 块
    assert remapped[f"conv_body.weight"] == 5
    assert remapped["conv_up1.weight"] == 7
    assert remapped["conv_up2.weight"] == 8
    assert remapped["conv_hr.weight"] == 9
    assert remapped["conv_last.weight"] == 10
    assert not any(key.startswith("model.") for key in remapped)


def test_remap_state_dict_keeps_unknown_keys():
    remapped = remap_state_dict({"something.else.weight": 1}, 23)

    assert remapped == {"something.else.weight": 1}


def test_remap_state_dict_is_a_noop_for_already_native_keys():
    native = {"conv_first.weight": 1, "body.0.rdb1.conv1.weight": 2}

    assert remap_state_dict(native, 23) == native


# ---- Real-CUGAN 2x ----------------------------------------------------------

_CUGAN_SOURCE = (
    ROOT / "models" / "onnx" / "_source" / "realcugan-2x" / "up2x-latest-no-denoise.pth"
)


def test_cugan2x_wrapper_matches_the_shipped_weight_keys():
    """The ONNX wrapper must consume the upstream checkpoint with strict=True.

    Guards the two things that actually differ from Real-ESRGAN: the submodule
    path ``unet1`` / ``unet2`` (not ``unet.unet1``) and the ``seblock`` naming
    inside ``UNetConv`` (not ``se``).
    """
    torch = pytest.importorskip("torch")
    if not _CUGAN_SOURCE.is_file():
        pytest.skip("Real-CUGAN source weights are not present in this checkout")

    from export_onnx import _CUGAN2xONNX

    state = torch.load(_CUGAN_SOURCE, map_location="cpu", weights_only=True)
    model = _CUGAN2xONNX()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected


def test_cugan2x_wrapper_doubles_spatial_size():
    torch = pytest.importorskip("torch")
    if not _CUGAN_SOURCE.is_file():
        pytest.skip("Real-CUGAN source weights are not present in this checkout")

    from export_onnx import _CUGAN2xONNX

    model = _CUGAN2xONNX()
    model.load_state_dict(
        torch.load(_CUGAN_SOURCE, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 96))
    assert out.shape == (1, 3, 128, 192)
    # fp 输出（不是 upstream CLI 那条取整成 byte 的分支），引擎的后处理契约靠它
    assert out.dtype == torch.float32
