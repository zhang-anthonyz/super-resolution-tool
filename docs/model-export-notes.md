# 模型导出与评测笔记

## 1. 导出

### Real-ESRGAN / 社区 RRDB 权重（`scripts/export_onnx.py`）

```powershell
python scripts/export_onnx.py --input <权重>.pth --output models/onnx/<名称>.onnx --scale 4 [--half]
```

- 支持 `--arch rrdbnet`（默认，RRDB 23 块）与 `--arch srvgg`（SRVGG 轻量）。
- `--scale` 必须与权重的训练倍率一致（`RealESRGAN_x2plus.pth` → `--scale 2`、`x4plus` → `--scale 4`）。
  **注意**：2x RRDB 内部会做一次 `pixel_unshuffle(2)`，因此推理时输入 H/W 都必须为偶数；
  本仓库的 512 px 分块推理保证这一点，奇数尺寸会越界。
- **自动识别 BasicSR/ESRGAN 旧命名**：社区权重（4x-UltraSharp、Remacri、NMKD…）的键名是
  `model.0` / `model.1.sub.N.RDBx.convY.0` / `model.3,6,8,10`，其中：
  - `model.1.sub.0 .. sub.N-1` 是 RRDB 块，`sub.N` 是主干卷积（trunk conv，映射为 `conv_body`）；
  - RRDB 内部卷积外层还有一层 Sequential，需把 `convX.0.weight` 还原成 `convX.weight`。
  脚本会据 `RDBx` 键自动推断块数并重映射，无需手改权重。
- 验证方式：导出时用 `strict=True` 加载，**0 缺失 / 0 多余**才算架构匹配。
- 数值一致性：导出后**至少 4 个被 2（4）整除的尺寸**比较 ONNX（CPU EP）与 torch 实现的输出，
  `max|diff|` 应 ≤ 1e-4（实测 fp32 全部 ≤ 2e-6，视为 0；fp16 因舍入会略大但仍在 1/255 量级）。
  2x 模型避开奇数尺寸（与 `pixel_unshuffle` 限制一致），4x 模型避开非 4 整数倍尺寸。

### SwinIR 权重（官方 release，Apache-2.0）

官方权重在 `JingyunLiang/SwinIR` 的 release `v0.0` 资产里（不在 git 仓库内）。
导出步骤（本仓库实测）：

1. 取官方 `models/network_swinir.py`（依赖 timm 的 `DropPath/to_2tuple/trunc_normal_`，
   推理期可用桩模块替代：DropPath 在 eval 下是恒等，`trunc_normal_` 只影响随后会被
   权重覆盖的初始化）。
2. 按任务构造 `SwinIR(...)`：

   | 任务 | 权重 | upscale | img_size | upsampler |
   |---|---|---|---|---|
   | 经典超分 x4 | `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth` | 4 | 48 | `pixelshuffle` |
   | 真实世界超分 x4 | `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | 4 | 64 | `nearest+conv` |
   | 彩色降噪 σ=25 | `005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth` | 1 | 128 | `''` |

   其余参数一致：`window_size=8, img_range=1.0, depths=[6]*6, embed_dim=180,
   num_heads=[6]*6, mlp_ratio=2, resi_connection="1conv"`。
3. `torch.onnx.export(..., opset_version=17, dynamic_axes={H,W})`。
   **必须 fp32**：SwinIR 的相对位置偏置掩码恒为 float32，`.half()` 会在 matmul 处报
   `expected m1 and m2 to have the same dtype`。
4. **假输入尺寸必须 ≠ 权重的 `img_size`**（本仓库用 `img_size + 32`）。
   踩过的坑：SwinIR 的 `WindowAttention.forward` 里有一句
   `if self.input_resolution == x_size:`——当导出用的假输入尺寸恰好等于该权重的
   `img_size`（如 realSR 的 64，用了 64×64 假输入）时，trace 会走这条"预计算缓存"
   分支，把窗口网格常量**写死进计算图**。后果是导出的 ONNX 只在 64 的整数倍尺寸下
   能跑，遇到 70×92 这类尺寸直接崩：
   `ReshapeHelper input_shape_size == size was false ... Input shape:{108,6,64,64},
   requested shape:{1,64,6,64,64}`（108 = 9×12 个窗口，写死的 64 = 8×8 窗口）。
   换用 96×96 假输入即可走动态分支，任意尺寸可用。
5. 一致性校验（务必做）：**至少 4 个尺寸**（含假输入尺寸、非窗口整数倍、非 64 整数倍、
   素数型尺寸如 97×131）比较 ONNX 与 torch 输出，`max|diff|` 应为 0（实测全部 0.0000）。
   只测一个尺寸会被上面那条缓存分支蒙混过关。
6. 交付前再用 onnxruntime（目标 EP，如 DirectML）做一次尺寸扫描，确认输出形状 = 输入 × 倍率。

## 2. 评测协议（Set5）

- 测试集：Set5 原始 HR 图（512×512 / 288×288 / 256×256 / 280×280 / 228×344）。
- **clean4x**：LR = 双三次下采样 4 倍 → 模型输出与 HR 比 PSNR/SSIM（经典 SR 口径）。
- **real4x**：在 clean4x 的 LR 上再叠加 高斯模糊 radius=0.8 + σ6 噪声 + JPEG q45（真实退化口径）。
- **denoise**：HR + 高斯噪声 σ=25，与干净 HR 比 PSNR/SSIM，并统计平均 Laplacian 能量
  （`edge`，越低≈噪声越少、越高≈细节越多，需与 PSNR 一起看）。
- 硬件：AMD RX 7900 XT + onnxruntime-directml 1.24.4，tile=512，单 tile 无拼接干扰。

### 基线数值（2026-10 实测，供回归对比）

| 模型 | clean4x PSNR/SSIM | real4x PSNR/SSIM | 秒/张 |
|---|---|---|---|
| realesr-general-x4v3 | 24.45 / 0.7849 | 24.27 / 0.7132 | 0.01 |
| RealESRGAN_x4plus_fp16 | 24.32 / 0.7644 | 23.32 / 0.6690 | 0.07 |
| 4x-UltraSharp fp16 | 24.28 / 0.7391 | 22.84 / 0.6238 | 0.07 |
| SwinIR_x4_realSR_GAN | 24.84 / 0.7647 | 23.80 / 0.6780 | 0.14 |
| **SwinIR_x4_classical** | **30.68 / 0.8866** | 22.82 / 0.6128 | 0.14 |

（上表 real4x 口径：双三次 1/4 → GaussianBlur 0.8 → σ6 噪声，不含 JPEG；
更早一版表格的 real4x 另加了 JPEG q45，故不能跨表直接比较。
`SwinIR_x4_classical` 在干净/双三次退化输入上比现有默认模型高 **+6.2dB**，
这是"干净图保真度最高"的由来；真实退化场景仍以 realesr-general-x4v3 最稳。）

降噪对照（σ=25，denoise-only 口径）：

| 方案 | PSNR | SSIM | edge |
|---|---|---|---|
| 不降噪 | 20.82 | 0.4977 | 58.93 |
| 旧版高斯混合 s=0.5 / 1.0 | 24.43 / 25.57 | — | 34.37 / 9.28 |
| 新版导向滤波 s=0.5 / 1.0 | 28.09 / 26.28 | 0.7592 / 0.7725 | 26.59 / 17.42 |
| SwinIR colorDN25 s=1.0 | 31.60 | 0.8903 | 9.74 |

端到端（LR 加 σ25 噪声后 4x 超分，SwinIR 经典 SR）：

| 方案 | PSNR | SSIM |
|---|---|---|
| 不降噪 | 16.64 | 0.3328 |
| 内置导向滤波 s=1.0 | 22.80 | 0.5770 |
| SwinIR 预降噪 s=1.0 | 25.38 | 0.7451 |

## 3. 已知陷阱

- SCUNet 的公开 ONNX 在 **DirectML 上会进程级崩溃**（exit 0xC0000005，CPU EP 正常），
  故未采用；如需盲降噪优先考虑 DRUNet/DPIR 一类纯 CNN 结构。
- 带外部数据（`.onnx` + `.onnx.data`）的模型需要 spec 一并打包 `*.data`，
  当前内置模型已全部避免该形式（单文件）。
- SwinIR 在任意尺寸（含非窗口整数倍）下与 torch 输出完全一致，tile 拼接安全；
  实测 tile=200 与 tile=512 的 PSNR 完全相同。
