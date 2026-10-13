# 内置模型来源与授权

本目录下的 `.onnx` 为应用内置模型。除特别说明外，均为公开预训练权重的格式转换
（`.pth` → ONNX），未做任何再训练。

| 模型文件 | 类型 | 来源权重 | 授权 | 备注 |
|---|---|---|---|---|
| `realesr-general-x4v3.onnx` | 超分 4x | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesr-general-x4v3.pth` | BSD-3-Clause | SRVGG 轻量通用模型，默认真实场景模型 |
| `realesr-general-wdn-x4v3.onnx` | 降噪 x4 | 同上，`realesr-general-wdn-x4v3.pth` | BSD-3-Clause | 与主模型输出按强度插值（同倍率） |
| `RealESRGAN_x2plus.onnx` | 超分 2x | 同上，`RealESRGAN_x2plus.pth`（23 块 RRDB，fp16 导出） | BSD-3-Clause | 轻度放大；**模型内部 `pixel_unshuffle(2)` 要求输入 H/W 均为偶数**，由分块推理保证 |
| `RealESRGAN_x4plus_anime_6B.onnx` | 超分 4x | 同上，`RealESRGAN_x4plus_anime_6B.pth` | BSD-3-Clause | 二次元场景默认模型 |
| `RealESRGAN_x4plus_fp16.onnx` | 超分 4x | 同上，`RealESRGAN_x4plus.pth`（fp16） | BSD-3-Clause | RRDB 重模型，高锐度 |
| `SwinIR_x4_classical.onnx` | 超分 4x | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) release `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth` | Apache-2.0 | 双三次退化训练；干净图保真度最高（PSNR 向） |
| `SwinIR_x4_realSR_GAN.onnx` | 超分 4x | 同上 release `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | Apache-2.0 | BSRGAN 退化训练；真实世界退化更自然 |
| `SwinIR_colorDN25.onnx` | 降噪 1x | 同上 release `005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth` | Apache-2.0 | 输入尺度预降噪（σ=25） |
| `4x-UltraSharp_fp16.onnx` | 超分 4x | 社区权重 `4x-UltraSharp.pth`（Kim2091，原作者仓库已下线，权重以镜像形式流传） | **未声明 / 无明确授权** | ⚠️ 锐度最强，但**不可用于商业分发**；随应用内置由用户明确接受风险后加入 |
| `realcugan-2x-no-denoise.onnx` | 超分 2x | [bilibili/ailab](https://github.com/bilibili/ailab) `Real-CUGAN` release `updated_weights.zip` → `up2x-latest-no-denoise.pth` | MIT | 双 UNet，2D 动漫 2x，无降噪 |
| `realcugan-2x-denoise1x.onnx` | 超分 2x | 同上 → `up2x-latest-denoise1x.pth` | MIT | 双 UNet，2D 动漫 2x，轻度降噪（σ≈12.5） |
| `realcugan-2x-denoise2x.onnx` | 超分 2x | 同上 → `up2x-latest-denoise2x.pth` | MIT | 双 UNet，2D 动漫 2x，中度降噪（σ≈25） |
| `realcugan-2x-denoise3x.onnx` | 超分 2x | 同上 → `up2x-latest-denoise3x.pth` | MIT | 双 UNet，2D 动漫 2x，强降噪（σ≈50） |
| `realcugan-2x-conservative.onnx` | 超分 2x | 同上 → `up2x-latest-conservative.pth` | MIT | 双 UNet，2D 动漫 2x，保守档（降噪最轻） |

Real-CUGAN 部分说明：

- 上游是 **UNet**（`unet1` + `unet2`），不是 RRDB/SRVGG，因此本仓库 `scripts/export_onnx.py`
  新增了 `--arch cugan2x` 入口；网络定义直接复用上游 `Real-CUGAN/upcunet_v3.py`
  （存放在 `scripts/_cugan/upcunet_v3.py`，保持原样以便逐行对照）。
- 上游 `UpCunet2x.forward` 带 `tile_mode / cache_mode / alpha / pro` 等 CLI 专用参数，
  且整图分支 `pro=False` 会把输出取整成 `byte`——不可 ONNX 化。导出器改为直接取
  `unet1` / `unet2` 拼一条只做整图推理、输出保持浮点的 forward。
- **数值一致性**：fp32 导出在 6 个尺寸（64×64 / 200×100 / 96×192 / 200×200 / 64×128 / 100×70）
  上 torch vs ONNX `max|diff| ≤ 7e-6`；fp16 因 SEBlock 里 `mean(...).half()` 的量化
  会带来 ≤1 阶的差异（实测 ≤0.96/255），属 fp16 推理本身的误差，肉眼不可见。
- 上游 `LICENSE` 为 **MIT**（Copyright (c) 2022 bilibili），可随应用内置分发。

## 重新导出方法

- Real-ESRGAN 系列（RRDB / SRVGG 架构）：
  ```powershell
  python scripts/export_onnx.py --input <权重>.pth --output models/onnx/<名称>.onnx --scale 4
  ```
  脚本会自动识别 BasicSR/ESRGAN 旧命名（`model.0` / `model.1.sub.N.RDBx.convY.0`）并重映射，
  因此社区 RRDB 权重（4x-UltraSharp、4x-Remacri、NMKD 系列等）可直接导出。
  加 `--half` 导出 fp16。
- SwinIR 系列：使用官方 `models/network_swinir.py` 构造对应任务模型后 `torch.onnx.export`
  （动态轴、fp32——SwinIR 相对位置偏置在 fp16 下存在 dtype 冲突），详见
  `docs/model-export-notes.md`。
- 所有导出完成后，用 Set5 双三次退化 + PSNR/SSIM 做一次回归对比（`docs/model-export-notes.md`
  记录了协议与基线数值）。
