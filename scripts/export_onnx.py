"""将 Real-ESRGAN 官方 .pth 权重导出为 ONNX 模型。

用法：
    python scripts/export_onnx.py --input models/onnx/RealESRGAN_x4plus.pth \
        --output models/onnx/RealESRGAN_x4plus.onnx --scale 4

Real-ESRGAN 输入/输出均为 RGB、[0,1] 归一化，与 OnnxEngine 预处理契约一致。
"""
from __future__ import annotations

import argparse
import re

import torch
from torch import nn
from torch.nn import functional as F

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_block", type=int, default=23)
    parser.add_argument("--half", action="store_true", help="导出 fp16 模型")
    parser.add_argument("--arch", choices=["rrdbnet", "srvgg"], default="rrdbnet")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt

    num_block = args.num_block
    if args.arch == "rrdbnet" and is_basicsr_state(state):
        detected = detect_rrdb_blocks(state)
        if detected:
            num_block = detected
        state = remap_state_dict(state, num_block)
        print(f"检测到 BasicSR/ESRGAN 旧命名权重，已重映射（RRDB blocks={num_block}）")

    if args.arch == "srvgg":
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
            upscale=args.scale, act_type="prelu",
        )
    else:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=args.scale,
            num_feat=64, num_block=num_block, num_grow_ch=32,
        )
    model.load_state_dict(state, strict=True)
    model.eval()

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
