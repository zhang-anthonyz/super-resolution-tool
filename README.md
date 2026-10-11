# 图片/视频超分工具（Super Resolution Tool）

本地离线运行的图片 + 视频超分辨率工具，兼容 NVIDIA / AMD / Intel 独立显卡。

## 特性

- **环境自检**：启动前检测独立显卡（三厂），无独显则拦截提示
- **跨厂商后端**：NVIDIA→CUDA；AMD→DirectML(推荐)/Vulkan(可选)；Intel→DirectML/OpenVINO
- **三场景预设**：真实 / 2D 动漫 / 3D·CG，选中自动映射默认模型（可手改）
- **7 个内置超分模型**：轻量通用 / 重质量 RRDB / 锐利社区模型 / SwinIR 经典与真实世界超分
- **独立降噪模型**：可选 SwinIR 彩色降噪（1x 输入尺度预降噪）或 Real-ESRGAN wdn（同倍率插值）；
  不选降噪模型时回退到**边缘保持的导向滤波**（替代原先的高斯模糊，同样降噪量下保留更多细节）
- **图片超分**：单张 + 批量（jpg/png/webp），自定义倍率，自定义导入模型（ONNX，支持社区 RRDB 权重自动重映射）
- **视频超分**：单条视频逐帧超分 + 三厂硬件编码（仅 8bit SDR，10bit/HDR 不支持）
- **开箱即用**：打包内置 Python 环境，无需用户安装

## 内置模型

| 模型 | 倍率 | 定位 |
|---|---|---|
| realesr-general-x4v3 | 4x | 轻量通用（默认，最快） |
| RealESRGAN_x2plus | 2x | 轻度放大 |
| RealESRGAN_x4plus_anime_6B | 4x | 2D 动漫 |
| RealESRGAN_x4plus_fp16 | 4x | RRDB 重模型，高锐度 |
| SwinIR_x4_classical | 4x | 经典超分（干净图保真度最高，较慢） |
| SwinIR_x4_realSR_GAN | 4x | 真实世界退化，纹理自然（较慢） |
| 4x-UltraSharp_fp16 | 4x | 社区锐利模型（授权未声明，见 `models/LICENSES.md`） |

降噪模型（在"降噪模型"下拉框单独选择）：`SwinIR_colorDN25`（1x 预降噪）、
`realesr-general-wdn-x4v3`（4x 同倍率插值）。

模型来源、授权与导出/评测方法见 `models/LICENSES.md` 与 `docs/model-export-notes.md`。

## 技术栈

Python 3.14 · PySide6 · ONNX Runtime（DirectML/CUDA）· ncnn · ffmpeg · NumPy/Pillow

## 目录结构

```
super-resolution-tool/
├── app/
│   ├── core/                 # 核心层：环境检测 / 后端选择 / 推理引擎 / 后处理
│   │   ├── types.py          # 共享类型契约（GPU/后端/场景/模型）
│   │   ├── environment_checker.py
│   │   ├── backend_selector.py
│   │   ├── sr_engine.py      # OnnxEngine / VulkanEngine + 降噪路由
│   │   └── postprocess.py    # 导向滤波降噪 + USM 锐化
│   ├── models/               # 模型管理（注册/下载/导入/导出重映射）
│   ├── pipelines/            # 图片管线 / 视频管线
│   └── ui/                   # PySide6 界面（theme / widgets / model_selector）
├── runtime/                  # 双 onnxruntime 运行时（构建期填充）
├── vendor/                   # ffmpeg / ncnn 二进制
├── models/                   # ONNX 模型 + LICENSES.md
├── scripts/                  # export_onnx.py / ui_preview.py
├── docs/                     # 技术方案、任务书、模型导出与评测笔记
└── tests/
```

## 开发命令

```powershell
python -m pytest -q                      # 单元测试（GPU 集成测试默认跳过）
python scripts/ui_preview.py             # 离屏渲染各页面 + 布局/配色自检（截图到 D:\model_output\images）
python scripts/export_onnx.py --help     # 权重 → ONNX（支持 BasicSR/ESRGAN 旧命名自动重映射）
pyinstaller SuperResolutionTool.spec     # 打包绿色免安装目录
```

## 文档

- 快速开始：[QUICKSTART.md](QUICKSTART.md)
- 技术方案：`docs/技术方案.md`
- 模型导出与评测协议：`docs/model-export-notes.md`
- 模型授权清单：`models/LICENSES.md`

## 当前状态

- [x] 需求分析与技术方案（v1.2）
- [x] W1 环境检测
- [x] W2 SR引擎·ONNX
- [x] W3 SR引擎·Vulkan
- [x] W4 图片管线
- [x] W5 视频管线
- [x] W6 UI+打包
- [x] W7 打磨（视觉重构、模型扩充、降噪改造）
