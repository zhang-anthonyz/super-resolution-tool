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
