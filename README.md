# 图片/视频超分工具（Super Resolution Tool）

本地离线运行的图片 + 视频超分辨率工具，兼容 NVIDIA / AMD / Intel 独立显卡。

## 特性

- **环境自检**：启动前检测独立显卡（三厂），无独显则拦截提示
- **跨厂商后端**：NVIDIA→CUDA；AMD→DirectML(推荐)/Vulkan(可选)；Intel→DirectML/OpenVINO
- **三场景预设**：真实 / 2D 动漫 / 3D·CG，选中自动映射默认模型（可手改）
- **图片超分**：单张 + 批量（jpg/png/webp），自定义倍率，自定义导入模型（ONNX）
- **视频超分**：单条视频逐帧超分 + 三厂硬件编码（仅 8bit SDR，10bit/HDR 不支持）
- **开箱即用**：打包内置 Python 环境，无需用户安装

## 技术栈

Python 3.11 · PySide6 · ONNX Runtime（CUDA/DirectML）· ncnn · Anime4KCPP · ffmpeg

## 目录结构

```
super-resolution-tool/
├── app/
│   ├── core/                 # 核心层：环境检测 / 后端选择 / 推理引擎
│   │   ├── types.py          # 共享类型契约（GPU/后端/场景/模型）
│   │   ├── environment_checker.py
│   │   ├── backend_selector.py
│   │   └── sr_engine.py
│   ├── models/               # 模型管理（注册/下载/导入/Anime4K）
│   ├── pipelines/            # 图片管线 / 视频管线
│   └── ui/                   # PySide6 界面
├── runtime/                  # 双 onnxruntime 运行时（构建期填充）
├── vendor/                   # ffmpeg / ncnn / Anime4KCPP 二进制
├── models/                   # ONNX 模型缓存
├── docs/                     # 技术方案与委派任务书
└── tests/
```

## 文档

- 技术方案：`docs/技术方案.md`
- 各 Wave 委派任务书：`docs/wave-*.md`

## 当前状态

- [x] 需求分析与技术方案（v1.2）
- [ ] W1 环境检测
- [ ] W2 SR引擎·ONNX
- [ ] W3 SR引擎·Vulkan
- [ ] W4 图片管线
- [ ] W5 视频管线
- [ ] W6 UI+打包
- [ ] W7 打磨
