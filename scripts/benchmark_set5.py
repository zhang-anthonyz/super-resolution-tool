"""Set5 基线评测：PSNR / SSIM / 边缘能量，按固定退化协议跑超分模型。

设计原则与 `docs/model-export-notes.md` 记录的口径一致：

* ``clean{N}x``：HR 双三次下采样 N 倍得 LR，模型放大 N 倍后与 HR 比 PSNR/SSIM
  （经典 SR 口径，考察保真度）。
* ``real{N}x``：在上面再叠加 GaussianBlur(0.8) + σ6 高斯噪声（真实退化口径）。
* ``noisy{N}x``：LR 加 σ25 高斯噪声后送模型（考察"降噪 + 放大"的端到端表现；
  降噪档位之间的差别主要看这一项）。
* ``bicubic``：不做超分，直接把 LR 用双三次放大 N 倍——作为"不做超分"的参照线，
  任何模型的 PSNR/SSIM 都应当明显高于它。

只依赖 numpy + Pillow（不引入 scipy / scikit-image）：

* PSNR：全通道 MSE 对 255 峰值。
* SSIM：灰度加权 0.299/0.587/0.114 + 11×11 高斯窗（σ=1.5，截断 3.5σ → 半径 5）、
  K1=0.01 / K2=0.03、data_range=255，取窗口有效区平均（与 scikit-image 默认口径一致，
  但不依赖它，便于任何人直接复跑）。
* edge：灰度 3×3 Laplacian 的绝对值均值（越大 = 细节/噪声越多，需与 PSNR 一起看）。

用法：

    python scripts/benchmark_set5.py --gt-dir D:\\model_output\\models\\candidates\\gt
    python scripts/benchmark_set5.py --gt-dir <dir> --scale 2 --models RealESRGAN_x2plus
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 每个测试的退化参数：下采样倍率 / 高斯模糊半径 / 噪声 σ
_INPUT_NOISE_SIGMA = 25.0
_REAL_BLUR_RADIUS = 0.8
_REAL_NOISE_SIGMA = 6.0


def _to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float64)


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _filter2d(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """可分离高斯滤波 + 有效区裁剪（与 scikit-image 的边界处理一致）。"""
    radius = len(kernel) // 2
    padded = np.pad(plane, radius, mode="reflect")
    rows = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    cols = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, rows)
    return cols


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """灰度 SSIM（高斯窗 11×11, σ=1.5, K1=0.01, K2=0.03）。"""
    x = _luma(reference)
    y = _luma(candidate)
    kernel = _gaussian_window()

    mu_x = _filter2d(x, kernel)
    mu_y = _filter2d(y, kernel)
    sigma_x2 = _filter2d(x * x, kernel) - mu_x * mu_x
    sigma_y2 = _filter2d(y * y, kernel) - mu_y * mu_y
    sigma_xy = _filter2d(x * y, kernel) - mu_x * mu_y

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / denominator))


_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def edge_energy(image: np.ndarray) -> float:
    """灰度 Laplacian 绝对值均值（细节/噪声指标）。"""
    plane = _luma(image)
    padded = np.pad(plane, 1, mode="reflect")
    response = (
        _LAPLACIAN[0, 1] * padded[:-2, 1:-1]
        + _LAPLACIAN[1, 0] * padded[1:-1, :-2]
        + _LAPLACIAN[1, 1] * padded[1:-1, 1:-1]
        + _LAPLACIAN[1, 2] * padded[1:-1, 2:]
        + _LAPLACIAN[2, 1] * padded[2:, 1:-1]
    )
    return float(np.mean(np.abs(response)))


def degrade(hr: Image.Image, scale: int, test: str, rng: np.random.Generator) -> Image.Image:
    """按协议生成 LR 输入。"""
    width, height = hr.size
    lr_size = (max(1, width // scale), max(1, height // scale))
    lr = hr.resize(lr_size, Image.Resampling.BICUBIC)

    sigma = 0.0
    if test.startswith("real"):
        lr = lr.filter(ImageFilter.GaussianBlur(_REAL_BLUR_RADIUS))
        sigma = _REAL_NOISE_SIGMA
    elif test.startswith("noisy"):
        sigma = _INPUT_NOISE_SIGMA

    if sigma > 0.0:
        array = np.asarray(lr, dtype=np.float64)
        array = array + rng.normal(0.0, sigma, array.shape)
        lr = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    return lr


def upscale_bicubic(lr: Image.Image, size: tuple[int, int]) -> Image.Image:
    return lr.resize(size, Image.Resampling.BICUBIC)


def build_engine(model_name: str):
    """加载一个内置模型（与 GUI 走同一条路径：注册表 + OnnxEngine）。"""
    from app.core.backend_selector import BackendSelector
    from app.core.types import GpuInfo, GpuType, GpuVendor
    from app.models.model_manager import builtin_models, load_model

    registry = {model.name: model for model in builtin_models()}
    if model_name not in registry:
        raise SystemExit(f"未知模型：{model_name}（可选：{', '.join(sorted(registry))}）")
    meta = registry[model_name]
    gpu = GpuInfo(
        vendor=GpuVendor.AMD,
        name="AMD Radeon RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=24 * 1024**3,
        vendor_id=0x1002,
    )
    backend = BackendSelector(gpu).default_backend()
    engine = load_model(meta)
    return engine, meta, backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True, help="Set5 HR 目录（baby/bird/butterfly/head/woman.png）")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--models", nargs="*", default=[], help="留空 = 全部 2x 内置模型")
    parser.add_argument("--tests", nargs="*", default=["clean", "real", "noisy", "bicubic"])
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--out", default="", help="报告输出路径（JSON + 文本表）")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    images = sorted(gt_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"{gt_dir} 下没有 HR 图片")

    model_names = args.models or None
    if model_names is None:
        from app.models.model_manager import builtin_models

        model_names = [m.name for m in builtin_models() if m.scale == args.scale]
    print(f"Set5 HR: {len(images)} 张 · 倍率 {args.scale}x · 模型 {len(model_names)} 个\n")

    rows: list[dict] = []
    for model_name in model_names:
        started = time.perf_counter()
        engine, meta, backend = build_engine(model_name)
        load_seconds = time.perf_counter() - started
        for test in args.tests:
            rng = np.random.default_rng(20261013)
            per_image: dict[str, list[float]] = {"psnr": [], "ssim": [], "edge": []}
            seconds = 0.0
            for path in images:
                with Image.open(path) as handle:
                    hr = handle.convert("RGB")
                if test == "bicubic":
                    lr = degrade(hr, args.scale, "clean", rng)
                    output = upscale_bicubic(lr, hr.size)
                else:
                    lr = degrade(hr, args.scale, test, rng)
                    started = time.perf_counter()
                    result = engine.upscale_image(np.asarray(lr), args.scale)
                    seconds += time.perf_counter() - started
                    output = Image.fromarray(result)
                    if output.size != hr.size:
                        output = output.resize(hr.size, Image.Resampling.BICUBIC)

                reference = _to_float(hr)
                candidate = _to_float(output)
                per_image["psnr"].append(psnr(reference, candidate))
                per_image["ssim"].append(ssim(reference, candidate))
                per_image["edge"].append(edge_energy(candidate))

            row = {
                "model": model_name,
                "test": f"{test}{args.scale}x",
                "backend": backend.value,
                "precision": getattr(engine, "precision_label", ""),
                "load_seconds": round(load_seconds, 3),
                "psnr": round(float(np.mean(per_image["psnr"])), 2),
                "ssim": round(float(np.mean(per_image["ssim"])), 4),
                "edge": round(float(np.mean(per_image["edge"])), 2),
                "seconds_per_image": round(seconds / len(images), 3),
                "per_image_psnr": [round(v, 2) for v in per_image["psnr"]],
            }
            rows.append(row)
            print(
                f"{row['model']:<28} {row['test']:<9} {row['backend']:<9} {row['precision']:<5} "
                f"PSNR={row['psnr']:6.2f}  SSIM={row['ssim']:.4f}  edge={row['edge']:5.2f}  "
                f"{row['seconds_per_image']:.2f}s/张"
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
