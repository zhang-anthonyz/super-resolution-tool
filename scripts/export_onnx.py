"""将 Real-ESRGAN / Real-CUGAN 官方 .pth 权重导出为 ONNX 模型。

用法：

  # Real-ESRGAN 2x/4x RRDB（X2plus、x4plus、x4plus_anime_6B、4x-UltraSharp 等）
  python scripts/export_onnx.py --input models/onnx/RealESRGAN_x2plus.pth \\
      --output models/onnx/RealESRGAN_x2plus.onnx --scale 2 --half

  # SRVGG 轻量通用（realesr-general / animevideov3 系列）
  python scripts/export_onnx.py --input realesr-general-x4v3.pth \\
      --output models/onnx/realesr-general-x4v3.onnx --arch srvgg --scale 4

  # Real-CUGAN 2x（5 档）
  python scripts/export_onnx.py \\
      --input models/onnx/_source/realcugan-2x/up2x-latest-no-denoise.pth \\
      --output models/onnx/realcugan-2x-no-denoise.onnx \\
      --arch cugan2x --variant no-denoise --half

Real-ESRGAN 输入/输出均为 RGB、[0,1] 归一化；
Real-CUGAN 2x 的输入是 RGB、[0,1] 归一化，输出同样是 RGB、[0,1] 浮点（与 OnnxEngine
的 preprocess_image / postprocess_tensor 契约一致）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# 上游 Real-CUGAN 网络定义（vendored 到 scripts/_cugan/，保持与上游逐字一致）。
# 需要 scripts/ 在 sys.path 上，直接执行本文件时它天然是 sys.path[0]。
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _cugan.upcunet_v3 import UNet1 as _CUGAN_UNet1  # noqa: E402
from _cugan.upcunet_v3 import UNet2 as _CUGAN_UNet2  # noqa: E402

#: BasicSR 时代的 RRDB 权重把模块挂在 ``model.N`` 下，且 RRDB 内部卷积外层还有
#: 一层 Sequential（``conv1.0.weight``）。社区模型（4x-UltraSharp / Remacri /
#: NMKD 等）几乎都是这个命名，这里做一次性重命名。
_BASICSR_FIXED = {
    "model.0.": "conv_first.",
    "model.3.": "conv_up1.",
    "model.6.": "conv_up2.",
    "model.8.": "conv_hr.",
    "model.10.": "conv_last.",
}

_BASICSR_BODY = re.compile(r"^model\.1\.sub\.(\d+)\.(.*)$")
_BASICSR_LEAF = re.compile(r"^RDB(\d)\.conv(\d)\.0\.(weight|bias)$")


def is_basicsr_state(state) -> bool:
    """Return whether the checkpoint uses the BasicSR/ESRGAN key naming."""
    return any(
        isinstance(key, str) and key.startswith("model.") and ".sub." in key
        for key in state
    )


def detect_rrdb_blocks(state) -> int:
    """Number of RRDB blocks in a BasicSR-style checkpoint.

    Counted from the keys that actually carry ``RDBx`` layers, so checkpoints
    with or without a trailing trunk convolution in the same Sequential both
    report the right block count.
    """
    indices = {
        int(match.group(1))
        for key in state
        if isinstance(key, str)
        and (match := _BASICSR_BODY.match(key))
        and _BASICSR_LEAF.match(match.group(2))
    }
    if not indices:
        return 0
    return max(indices) + 1


def remap_state_dict(state: dict, num_block: int) -> dict:
    """Translate BasicSR/ESRGAN key names into this file's RRDBNet layout."""
    remapped: dict = {}
    for key, value in state.items():
        if not isinstance(key, str):
            remapped[key] = value
            continue

        body_match = _BASICSR_BODY.match(key)
        if body_match:
            index = int(body_match.group(1))
            rest = body_match.group(2)
            if index >= num_block:
                # 主干卷积（trunk conv）与 RRDB 块同处一个 Sequential
                remapped[f"conv_body.{rest}"] = value
                continue
            leaf = _BASICSR_LEAF.match(rest)
            if leaf:
                rdb, conv, param = leaf.groups()
                remapped[f"body.{index}.rdb{rdb}.conv{conv}.{param}"] = value
                continue
            remapped[f"body.{index}.{rest}"] = value
            continue

        for source, target in _BASICSR_FIXED.items():
            if key.startswith(source):
                remapped[target + key[len(source):]] = value
                break
        else:
            remapped[key] = value
    return remapped


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """紧凑 VGG 风格 SR 网络（realesr-general / animevideov3 系列）。

    上采样在最后一层完成，HR 特征空间不做卷积；输出为像素重组 + 最近邻残差。
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == "prelu":
                self.body.append(nn.PReLU(num_parameters=num_feat))
            elif act_type == "leakyrelu":
                self.body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base
        return out


class _DMLSafeUNet1(_CUGAN_UNet1):
    """``UNet1`` 的 DirectML 安全版：负 padding 裁剪改成等价切片。

    ``F.pad(x, (-4, -4, -4, -4))`` 会lower 成 Slice，而 DirectML execution
    provider 把它的 ends 算错——探针实测 max|diff| 3.85（CPU EP 为 0），而且
    **静默出错**：ONNX 输出是一张看起来正常、实则错误的图（Set5 上 PSNR 掉到
    约 7 dB，正常约 25 dB）。切片完全等价且在所有 EP 上正确，所以在这里重建
    forward，而不改动 vendored 的上游文件。
    """

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[:, :, 4:-4, 4:-4]  # upstream: F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        return self.conv_bottom(x3)


class _DMLSafeUNet2(_CUGAN_UNet2):
    """``UNet2`` 的 DirectML 安全版（两处负 padding → 切片）。见上。"""

    def forward(self, x, alpha=1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[:, :, 16:-16, 16:-16]  # upstream: F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = x2[:, :, 4:-4, 4:-4]  # upstream: F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        x4 = x4 * alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        return self.conv_bottom(x5)


class _CUGAN2xONNX(nn.Module):
    """Real-CUGAN 2x 整图路径的轻量包装，导出 ONNX 时使用。

    上游的 ``UpCunet2x.forward`` 留有分块 / cache / pro 等只对推理 CLI
    有意义的控制参数（且 ``pro=False`` 那一支还把 fp 输出截成 ``byte``，不可
    ONNX 化）。这里从内部取出 ``unet1`` / ``unet2`` 复用，按整图路径的尺寸
    对齐方式拼一个 **fp 浮点输出** 的 forward。

    两个 UNet 换成 :class:`_DMLSafeUNet1` / :class:`_DMLSafeUNet2`：数值与上游
    逐元素一致，只是把 DirectML 会算错的负 padding 换成等价切片。权重键名仍是
    ``unet1.*`` / ``unet2.*``，可直接 strict 加载上游 checkpoint。
    """

    def __init__(self):
        super().__init__()
        self.unet1 = _DMLSafeUNet1(in_channels=3, out_channels=3, deconv=True)
        self.unet2 = _DMLSafeUNet2(in_channels=3, out_channels=3, deconv=False)

    def forward(self, x):
        n, c, h0, w0 = x.shape
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), mode="reflect")
        x = self.unet1(x)
        x0 = self.unet2(x, alpha=1.0)
        x = x[:, :, 20:-20, 20:-20]  # upstream: F.pad(x, (-20, -20, -20, -20))
        out = x0 + x
        if w0 != pw or h0 != ph:
            out = out[:, :, : h0 * 2, : w0 * 2]
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_block", type=int, default=23)
    parser.add_argument("--half", action="store_true", help="导出 fp16 模型")
    parser.add_argument("--arch", choices=["rrdbnet", "srvgg", "cugan2x"], default="rrdbnet")
    parser.add_argument(
        "--variant",
        default="",
        help="Real-CUGAN 2x 内部用于区分权重文件的标签，只在 arch=cugan2x 时记录到 print",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = (
        ckpt.get("params_ema")
        or ckpt.get("params")
        or ckpt.get("state_dict")
        or ckpt
    )

    if args.arch == "cugan2x":
        # 5 档权重文件结构完全一致（unet1.* / unet2.*），不重映射。
        model = _CUGAN2xONNX()
        print(
            f"Real-CUGAN 2x 通用版（upstream UpCunet2x，unet1+unet2），导出为 ONNX（{args.variant or '未指定 variant'}）"
        )
    elif args.arch == "srvgg":
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
            upscale=args.scale, act_type="prelu",
        )
    else:
        num_block = args.num_block
        if is_basicsr_state(state):
            detected = detect_rrdb_blocks(state)
            if detected:
                num_block = detected
            state = remap_state_dict(state, num_block)
            print(f"检测到 BasicSR/ESRGAN 旧命名权重，已重映射（RRDB blocks={num_block}）")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=args.scale,
            num_feat=64, num_block=num_block, num_grow_ch=32,
        )
    model.load_state_dict(state, strict=True)
    model.eval()

    # 假输入：Real-CUGAN 2x 用 64x64（unet 下采样两次到 16x16）；RRDB x2 不允许奇数
    # 用 64；其余用 64。
    dummy = torch.randn(1, 3, 64, 64)
    if args.half:
        model = model.half()
        dummy = dummy.half()
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "N", 2: "H", 3: "W"},
            "output": {0: "N", 2: "H", 3: "W"},
        },
        dynamo=False,
    )
    print(f"导出完成: {args.output}")


if __name__ == "__main__":
    main()
