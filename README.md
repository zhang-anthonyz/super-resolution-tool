# 图片 / 视频超分工具（Super Resolution Tool）

> 完全本地离线的图片与视频超分辨率工作台。桌面 GUI + 分块推理 + 三厂硬件编码，
> 兼容 NVIDIA / AMD / Intel 独立显卡，**不上传任何素材、不依赖云端服务**。

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011%20x64-0078D6?logo=windows&logoColor=white)
![UI](https://img.shields.io/badge/UI-PySide6%206.12-41CD52?logo=qt&logoColor=white)
![Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20%28DirectML%20%2F%20CUDA%29-005CED)
![Tests](https://img.shields.io/badge/Tests-221%20passed-success)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 目录

- [简介](#简介)
- [软件运行截图](#软件运行截图)
- [特性](#特性)
- [界面与工作流](#界面与工作流)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [内置模型](#内置模型)
- [技术架构](#技术架构)
- [性能](#性能)
- [目录结构](#目录结构)
- [开发与测试](#开发与测试)
- [打包分发](#打包分发)
- [配置与环境变量](#配置与环境变量)
- [日志与排查](#日志与排查)
- [已知限制](#已知限制)
- [路线图](#路线图)
- [常见问题](#常见问题)
- [许可与第三方资源](#许可与第三方资源)
- [致谢](#致谢)

---

## 简介

这是一个**跨厂商、纯离线**的超分工具：

- **图片**：单张或批量（递归目录），支持 jpg / png / webp，输出保持原格式，可选回缩原尺寸或按百分比缩放；
- **视频**：逐帧超分 + 显卡硬件编码（NVENC / AMF / QSV），音频原样复制；
- **对比预览**：单张任务完成后自动进入对照页，放大镜 / 滑块两种方式核对细节；
- **7 个内置超分模型 + 2 个降噪模型**，支持导入社区 ONNX 权重，也提供 `.pth → ONNX` 导出脚本。

设计上刻意做了几个"不妥协"的选择：

| 取向 | 具体表现 |
|---|---|
| 离线优先 | 无网络请求、无遥测；模型与二进制随包分发，运行期只读本地文件 |
| 用 GPU 说话 | 启动即枚举显卡，**没有独显直接拒绝启动**（这是设计行为，不是故障） |
| 过程可解释 | 解码 / 设置 / 每 N 块推理与剩余时间 / 后处理 / 编码逐步落日志，终端与界面同源 |
| 不牺牲保真换速度 | 提速手段都以"新旧输出逐像素差 ≤ 1 级"为前提验证后才采用（见[性能](#性能)） |

**运行前提**：Windows 10 / 11 64 位 + NVIDIA / AMD / Intel 独立显卡（建议显存 ≥ 4 GB，SwinIR 系列建议 ≥ 6 GB）。

---

## 软件运行截图

图片超分页：输入列表、模型与后端、输出与画质（截图取自打包版的实际运行）

<img width="2560" height="1369" alt="图片超分页" src="https://github.com/user-attachments/assets/0d66d35d-b09d-4823-aae8-c06c11e7f36d" />

一次实际任务：进度、分步运行日志与结果（Vulkan 后端，输出为 4x 的 50% 百分比缩放）

<img width="2560" height="1380" alt="运行中的分步日志" src="https://github.com/user-attachments/assets/774794f7-b274-4640-af58-d16f20f11683" />

对比预览 · 滑块对比：拖动分割线，左侧原图、右侧超分输出

<img width="2560" height="1385" alt="滑块对比" src="https://github.com/user-attachments/assets/3486783f-be78-46af-b490-bf628b050372" />

对比预览 · 放大镜对比：光标处的细节在两个放大格中同步呈现

<img width="2560" height="1391" alt="放大镜对比" src="https://github.com/user-attachments/assets/02d287b3-d985-4c6c-a86c-edeeff726be2" />

## 特性

**推理与后端**

- 双 ONNX Runtime 动态加载：`onnxruntime-gpu`（CUDA / TensorRT EP）与 `onnxruntime-directml`（DirectML EP）
  按后端起用对应运行时目录，互不冲突
- 后端选择器按厂商给默认值（NVIDIA→CUDA、AMD→DirectML、Intel→DirectML/OpenVINO，兜底 CPU），
  下拉框中未就绪的后端标注「（未就绪）」
- Vulkan 第二后端（ncnn Real-ESRGAN CLI 子进程），AMD 无 DirectML 时的退路
- 运行时能力探测：实际生效的 execution provider 与推理精度（fp16 / fp32）会回显到日志与界面
- 超分会话按「后端 + 模型」缓存复用，省去每次任务 0.3–3 s 的加载时间

**图片超分**

- 分块推理：默认 512 px 分块、10 px 重叠、重叠带线性羽化拼接（接缝台阶 ≤ 12 级），
  显存不足自动降级到 256 px
- 三档输出压缩（JPEG 质量 / PNG 压缩级别 / WEBP 质量或无损），默认档位以"10 倍保存速度换 14% 体积"为取舍
- 输出尺寸三选一：保持超分尺寸 / 回缩原图尺寸 / 按超分结果的百分比（5–400%）等比缩放
- 批量任务逐个报告进度与耗时（解码 / 推理 / 后处理 / 编码分段计时，含 MP/s），同名文件可跳过或覆盖

**视频超分**

- 解码 → 推理 → 编码全程走管道（rawvideo rgb24），不落盘中间帧
- 三厂硬件编码器路由 + 参数：NVENC `-cq 21 -preset p4`、AMF `-rc cqp -qp_i 18 -qp_p 20`、
  QSV `-global_quality 22`；编码器失败自动回退 `libx264 / libx265`
- 音频流原样复制（remux），并保留像素宽高比（`setsar`）与旋转元数据（`rotate`）
- 仅接受 8bit SDR，检测到 10bit / HDR（`p010`、`smpte2084`、`bt2020` 等）直接给出明确失败原因

**降噪与锐化**

- 降噪三路自动路由：`1x` 降噪模型在**输入尺度**预降噪 → 同倍率降噪模型与超分输出按强度插值 →
  无可用模型时回退**内置边缘保持降噪**（导向滤波）
- 边缘锐化（USM）独立于降噪，默认 0.25，可同时"平滑 + 提锐"
- 强度均为 0.00–1.00 连续可调，0 即关闭

**对比预览**

- 独立页面占满画布，单张任务完成后自动跳转（批量不跳转，避免指代不清）
- **放大镜对比**：两个放大格固定在画布中央（稳定注视点），光标移到自己身下时沿最短一侧**轻微让位**、
  离开即归位；蓝框标出取样位置，倍率 1x–10x
- **滑块对比**：竖直分割线左右对照，滚轮**以光标为中心**缩放、拖动平移（夹紧在图片范围内）、双击复位
- 适应窗口 / 1:1 像素 / 可最大化的独立大窗口（几何与模式、倍率、分割线位置都会被记住）
- 预览按 20 MP 内存预算自动降采样并标注比例，避免两张几十兆像素图占满内存

**工程化**

- 分步运行日志：终端 + 轮转文件（4 MB × 3）+ 界面日志面板同源，`-v` 展开到每块 tile
- 绿色免安装打包（PyInstaller one-dir，约 900 MB，内置 Python 运行时与全部依赖），
  自带无界面自检 `--headless`
- 221 项自动化测试（含 DirectML 真机集成测试）与离屏 UI 渲染审计脚本

---

## 界面与工作流

左侧导航分四个页面，顶部标题栏右侧常驻显示`厂商 / 显卡型号 / 显存 / 可用后端`：

```
工作台
├── 图片超分   输入 → 模型与后端 → 输出与画质 → 任务 → 运行日志
├── 视频超分   输入 → 模型与后端 → 输出与编码 → 任务 → 运行日志
└── 对比预览   放大镜 / 滑块对照（单张任务完成后自动跳转）
系统
└── 设置       推理设备信息、自定义模型导入
```

典型流程：

1. 图片页添加文件（支持多选 / 文件夹递归），选场景预设与模型，设定输出目录与尺寸策略；
2. 点「开始超分」，进度条与运行日志实时刷新，可随时取消；
3. 单张任务完成后自动进入**对比预览**页核对细节，满意即取输出文件；
4. 视频页同理，只是输出固定为 `<原文件名>_x<倍率>.mp4`。

---

## 快速开始

### 方式一：直接使用打包版（零安装）

```
SuperResolutionTool\SuperResolutionTool.exe
```

目录内已含 Python 运行时、PySide6、ONNX Runtime、FFmpeg / FFprobe、ncnn 与 9 个 ONNX 模型，
**无需安装 Python 或任何依赖**。请整体保留该目录（模型、DLL、FFmpeg 都在同级），建议拷到固定位置再运行。

无界面自检（加载默认模型 → DirectML → 内置 FFmpeg → 实跑一次 4x 超分）：

```powershell
.\SuperResolutionTool\SuperResolutionTool.exe --headless
# 成功退出码 0 并打印 HEADLESS_SMOKE_OK；失败打印 HEADLESS_SMOKE_FAILED: <原因>
```

### 方式二：从源码运行（开发用）

```powershell
git clone https://github.com/zhang-anthonyz/super-resolution-tool.git
cd super-resolution-tool

python -m pip install -r requirements.txt

# 按显卡二选一（两者模块名相同，不要同时安装）：
python -m pip install onnxruntime-directml   # AMD / Intel / 通用（DirectML）
python -m pip install onnxruntime-gpu        # NVIDIA（CUDA EP）

python app\main.py              # 启动界面
python app\main.py --headless   # 无界面自检
python app\main.py -v           # 展开到每块 tile 的调试日志
```

源码运行还需要两样仓库未入库的资源（体积原因，见 `.gitignore`）：

| 资源 | 放置位置 | 获取方式 |
|---|---|---|
| FFmpeg / FFprobe | `vendor/ffmpeg.exe`、`vendor/ffprobe.exe` | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) full build（含三厂硬件编码器） |
| ONNX 模型 | `models/onnx/*.onnx` | 按下文[内置模型](#内置模型)一节的来源与 `scripts/export_onnx.py` 自行导出 |
| ncnn（可选，Vulkan 后端） | `vendor/realesrgan-ncnn-vulkan.exe` + `vendor/ncnn-models/` | Real-ESRGAN 官方 ncnn 发布包 |

---

## 使用指南

### 图片超分

| 设置项 | 说明 |
|---|---|
| 输入 | jpg / jpeg / png / webp；支持多选与文件夹递归收集，列表内可多选移除 |
| 场景预设 | `真实` / `2D 动漫` / `3D·CG` —— 选中即套用该场景默认模型，之后可手动改 |
| 超分模型 | 内置模型 + 已导入的自定义模型；也可直接指定任意 ONNX 路径 |
| 倍率 | 2x / 3x / 4x / 自定义 2–8；固定倍率模型会自动纠正并在日志提示 |
| 输出尺寸 | 保持超分尺寸 / 回缩到原图尺寸 / 按百分比（相对超分结果，5–400%）等比缩放 |
| 输出压缩 | 高质量 / 高压缩 / 无损·最高质量 |
| 覆盖已存在文件 | 不勾选时同名输出跳过并在日志标注 |

输出命名：

| 场景 | 文件名 |
|---|---|
| 保持超分尺寸 | `<原名>_x<倍率>.<原扩展名>` |
| 回缩原图尺寸 | `<原名>_x<倍率>_restored.<原扩展名>` |
| 百分比缩放 | `<原名>_x<倍率>_p<百分比>.<原扩展名>`（100% 不加后缀） |

### 视频超分

1. 选视频与输出目录 → 选编码器（H.264 / HEVC）与倍率 → 开始；
2. 逐帧推理后由内置 FFmpeg 硬件编码，音频原样复制；
3. 耗时 ≈ `总帧数 × 每帧推理时间`，长视频建议先剪一小段试跑。

### 降噪与锐化

| 降噪模型选项 | 行为 |
|---|---|
| 无（内置边缘保持降噪） | 导向滤波：窗口半径 1–4 px，正则化项 `eps = (0.02→0.16) × 255` 再平方，随强度增长，边缘保持、平坦区强平滑 |
| `realesr-general-wdn-x4v3` | 4x 同倍率：降噪结果与超分输出按强度插值，强降噪 |
| `SwinIR_colorDN25` | 1x：在**输入尺度**先降噪再送超分（σ=25 训练），端到端质量最高 |

- 降噪强度 `0.00–1.00`（0 = 关闭）；边缘锐化默认 `0.25`，过高会产生振铃，建议 0.1–0.4。

### 导入自定义模型

**设置 → 自定义模型 → 导入 ONNX 模型到模型库**：填写名称、倍率与 ONNX 路径，
注册表写入 `<程序目录>\models\custom_models.json`（每机状态，已在 `.gitignore` 中忽略）。
社区 RRDB 权重（4x-UltraSharp、Remacri、NMKD 等）可用 `scripts/export_onnx.py` 自动重映射键名后导出。

---

## 内置模型

### 超分模型（7 个）

体积为 ONNX 单文件大小（十进制 MB）。

| 模型 | 倍率 | 体积 | 定位 | 授权 |
|---|---|---|---|---|
| `realesr-general-x4v3` | 4x | 2.4 MB | **默认**（真实 / 3D·CG）：SRVGG 轻量通用，速度最快 | BSD-3-Clause |
| `RealESRGAN_x2plus` | 2x | 33.7 MB | 轻度放大，无降噪 | BSD-3-Clause |
| `RealESRGAN_x4plus_anime_6B` | 4x | 9.0 MB | **2D 动漫默认**：边缘自然 | BSD-3-Clause |
| `RealESRGAN_x4plus_fp16` | 4x | 33.7 MB | RRDB 重模型，高锐度细节 | BSD-3-Clause |
| `SwinIR_x4_classical` | 4x | 58.2 MB | 干净图保真度最高（Set5 clean4x **30.68 dB**，比默认高 +6.2 dB），较慢 | Apache-2.0 |
| `SwinIR_x4_realSR_GAN` | 4x | 57.5 MB | BSRGAN 真实退化训练，纹理自然，较慢 | Apache-2.0 |
| `4x-UltraSharp_fp16` | 4x | 33.7 MB | 社区锐利模型，细节最锐 | ⚠️ 授权未声明 |

### 降噪模型（2 个，独立于超分模型选择）

| 模型 | 倍率 | 体积 | 行为 | 授权 |
|---|---|---|---|---|
| `realesr-general-wdn-x4v3` | 4x | 2.4 MB | 同倍率，与超分输出插值 | BSD-3-Clause |
| `SwinIR_colorDN25` | 1x | 56.6 MB | 输入尺度预降噪（σ=25） | Apache-2.0 |

来源、授权与再导出步骤见 [`models/LICENSES.md`](models/LICENSES.md)；
导出踩过的坑（SwinIR 假输入尺寸写死窗口网格、fp16 的 dtype 冲突、SCUNet 在 DirectML 崩溃等）见
[`docs/model-export-notes.md`](docs/model-export-notes.md)，其中也记录了 Set5 基线与降噪对照数值。

---

## 技术架构

### 分层

| 层 | 模块 | 职责 |
|---|---|---|
| 契约层 | `app/core/types.py` | GPU / 后端 / 场景 / 模型 / 结果与进度的共享数据结构，各层只依赖它 |
| 核心层 | `core/environment_checker.py`、`core/backend_selector.py`、`core/runtime_capabilities.py` | 枚举显卡（DXGI，失败退 WMI）、挑选独显、探测 execution provider、按后端起用对应运行时 |
| 推理层 | `core/sr_engine.py`、`core/tiling.py`、`core/postprocess.py` | `OnnxEngine` / `VulkanEngine`、分块与拼接、导向滤波降噪与 USM 锐化 |
| 管线层 | `pipelines/image_pipeline.py`、`pipelines/video_pipeline.py`、`pipelines/ffmpeg_utils.py` | 单张 / 批量 / 逐帧任务编排、编码与封装、FFmpeg 调用与探测 |
| 模型层 | `models/model_manager.py` | 内置注册表、自定义导入与注册、场景默认映射 |
| 界面层 | `ui/main_window.py`、`ui/compare_view.py`、`ui/theme.py`、`ui/workers.py` | 四页导航、对比画布、令牌化主题与自绘控件、Qt 工作线程 |

### 数据流

```
图片：文件 → Pillow 解码 → [1x 降噪模型预降噪] → 分块推理（512px/重叠10px，逐块羽化拼接写 uint8）
      → [同倍率降噪插值] → 导向滤波 / USM → 尺寸策略 → Pillow 编码 → 结果与分步日志

视频：FFprobe 探测（8bit SDR / 帧率 / SAR / 旋转）→ FFmpeg 解码管道（rawvideo rgb24）
      → 逐帧同上推理 → FFmpeg 硬件编码管道 → remux 音频 → 输出 mp4
```

### 关键设计

**双 ONNX Runtime 共存。** `onnxruntime-gpu` 与 `onnxruntime-directml` 的包名同为 `onnxruntime`，
无法装进同一个环境。构建期把两者分别解包到 `runtime/cuda/` 与 `runtime/dml/`，
运行时按后端把对应目录临时加入 `sys.path`、清掉模块缓存后导入（目录不存在时回退到当前环境已安装的那一个，
即源码运行的常见情形）。

**分块推理与羽化拼接。** 每个 512 px 分块独立推理后**直接写入 uint8 输出图的对应位置**，
只有与左 / 上相邻块共享的重叠带做线性羽化（`tile_size - 10` 步进）。这条路线同时解决了两个问题：
峰值内存从"全图 float32 累加器（曾经外溢到磁盘 memmap，67 MP 输出约 1.9 GB 临时文件）"降到**一张输出图**；
接缝最大台阶从 33 级降到 **12 级**。显存不足时自动降级到 256 px 并记录日志。

**降噪路由。** 由引擎回答"这次降噪请求由谁服务"：`1x` 模型走输入尺度预降噪（结果按强度与原图混合），
同倍率模型与超分输出插值，两者都不适用则回退内置导向滤波并给出日志提示——
因此任意模型组合都不会出现"降噪无效但界面显示已开启"的情况。

**导向滤波用"平移相加"求窗口和。** 9 次平移相加取代 cumsum 实现，配合全程 float32 与分带直写 uint8，
16.7 MP 输出从 5.1 s 降到 1.7 s；同时修正了"离均差平方 ≠ 窗口方差"的实现错误。

**编码策略。** 刻意不使用 Pillow 的 PNG `optimize=True`：它触发 libpng 逐行滤波器搜索，
4096² 输出实测 14.9 s vs 1.3 s，只换来约 14% 体积；WEBP 固定 `method=4`。

**视频管道化。** 解码与编码各由一个 FFmpeg 子进程承担，中间帧以 rawvideo rgb24 走管道，
推理在两者之间逐帧进行；编码器按厂商 + 编码格式选择，失败时（例如驱动不支持该 profile）
自动用 `libx264 / libx265` 重跑，并把尝试过程写进日志。

---

## 性能

实测环境：**AMD RX 7900 XT 24 GB + onnxruntime-directml 1.24.4**，模型 `realesr-general-x4v3`。

| 场景 | 优化前 | 优化后 |
|---|---|---|
| 1024² → 4096²（降噪 0.5 + 锐化 0.3） | 17.9 s | **4.5 s** |
| 2048² → 8192²（67 MP，降噪 0.5） | 72.1 s | **15.5 s** |
| PNG 编码 4096² | 14.9 s | 1.4 s |
| 内置导向滤波 16.7 MP | 5.1 s | 1.7 s |
| 峰值内存 67 MP 输出 | ~1.9 GB 临时文件 | 一张输出图 |

提速前后输出经 PSNR 对比校核（55.3 / 53.9 dB，逐像素平均差 0.05–0.06 级），即**质量不变、耗时降为 1/4 左右**。

反直觉但实测确认、因此**不要**再尝试的优化：

- **分块批处理在 DirectML 上更慢**：batch 2/4/8 为 23.8 / 21.5 / 19.7 ms/块，batch=1 仅 16.8 ms；更大 tile（768/1024）同样更慢；
- **float32 全图累加器**比直接写 uint8 慢 2.3 倍，且超过 512 MB 会外溢磁盘。

会话加载时间（影响任务启动）：`realesr-general-x4v3` 0.25 s、`4x-UltraSharp_fp16` 0.76 s、SwinIR **3.0 s**——
因此超分会话按「后端 + 模型」缓存复用。

---

## 目录结构

```
super-resolution-tool/
├── app/
│   ├── main.py                 入口：GUI / --headless 自检 / -v 日志级别
│   ├── core/                   环境检测、后端选择、推理引擎、分块、后处理、日志、设置
│   ├── models/                 模型注册表、自定义导入、场景默认映射
│   ├── pipelines/              图片管线、视频管线、FFmpeg 调用与探测
│   └── ui/                     PySide6 界面：主窗口、对比画布、主题、控件、工作线程
├── models/
│   ├── onnx/                   9 个内置 ONNX 模型（*.onnx 未入库）
│   └── LICENSES.md             模型来源与授权清单
├── vendor/                     ffmpeg / ffprobe / ncnn（未入库）
├── scripts/
│   ├── export_onnx.py          .pth → ONNX 导出（自动重映射社区 RRDB 键名）
│   └── ui_preview.py           离屏渲染全部页面 + 布局/配色/主题程序化审计
├── tests/                      221 项测试（pytest）
├── docs/                       技术方案、模型导出笔记
├── SuperResolutionTool.spec    PyInstaller 打包配置
├── config.example.yaml         外部工具路径等可选配置示例
├── QUICKSTART.md               零基础逐步操作说明书（安装、上手、模型与降噪速查、常见问题）
└── requirements.txt
```

---

## 开发与测试

```powershell
python -m pytest -q                    # 全量单元测试
python -m pytest -m slow               # 标记为 slow 的集成测试（真机编码）
python scripts/ui_preview.py --out-dir D:\model_output\images --tag after
python app/main.py --headless          # 端到端自检（模型 + DirectML + FFmpeg）
```

测试分布（共 **221** 项）：

| 领域 | 文件（用例数） |
|---|---|
| 对比预览 | `test_compare_view.py` (37)：几何映射、放大格定位与让位、光标锚定缩放、平移夹紧、设置往返 |
| 图片管线 | `test_image_pipeline.py` (29)：命名、压缩档、尺寸策略、批量与取消、写盘回读 |
| 推理引擎 | `test_sr_engine.py` (16)、`test_sr_engine_integration.py` (1, DirectML 真机) |
| 分块与后处理 | `test_tiling.py` (11)、`test_postprocess.py` (7)、`test_denoise_routing.py` (9) |
| 环境与后端 | `test_environment_checker.py` (13)、`test_backend_selector.py` (11)、`test_vulkan_engine.py` (6) |
| 视频管线 | `test_video_pipeline.py` (6)、`test_video_pipeline_integration.py` (1, slow)、`test_ffmpeg_utils.py` (13) |
| 界面与日志 | `test_ui_logic.py` (15)、`test_ui_progress.py` (13)、`test_model_selector.py` (5)、`test_task_log.py` (9)、`test_main.py` (4)、`test_theme.py` (2) |
| 模型与导出 | `test_model_manager.py` (8)、`test_export_onnx.py` (5) |

`scripts/ui_preview.py` 是无头环境下的界面回归网：把每个页面（含对比页的放大镜 / 滑块两种模式，
用真实图片对驱动）渲染成 PNG，并程序化检查控件未塌陷、下拉箭头与微调按钮确实画出了像素、
色板未回退到 Fusion 默认灰（`#EFEFEF`）、导航选中态配色正确、标签未遮盖卡片背景、页面无横向裁切。

---

## 打包分发

```powershell
python -m PyInstaller SuperResolutionTool.spec --noconfirm
python dist\SuperResolutionTool\SuperResolutionTool.exe --headless   # 产物自检
```

- **one-dir 绿色版**（`contents_directory="."`），产物约 **900 MB**：内置 Python 运行时、PySide6、
  ONNX Runtime、FFmpeg / FFprobe（含三厂硬件编码器）、ncnn 与 9 个 ONNX 模型；
- spec 细节：排除 `torch / onnx / scipy / pandas / matplotlib` 等开发期依赖；
  采集 `onnxruntime` 与 `shiboken6` 动态库；过滤 `icu*.dll`（某些本地 ICU 构建只导出带版本的 C++ 符号，
  会导致 `QtWidgets` 报 WinError 127）；
- 打包版无控制台，因此自检结论除打印外同时写入日志文件，退出码 0 / 1 可脚本判定；
- 验证产物完整性：`--headless` 能加载默认模型 + DirectML + 内置 FFmpeg 并实跑一次 4x 超分即为通过。

---

## 配置与环境变量

| 配置 | 位置 | 说明 |
|---|---|---|
| 界面设置 | `%LOCALAPPDATA%\SuperResolutionTool\settings.json` | 输出目录、覆盖策略、尺寸与压缩档、后端、降噪 / 锐化强度与模型、对比预览偏好、大窗口几何 |
| 外部工具路径 | `<程序目录>\config.yaml`（复制 `config.example.yaml`） | 目前生效的是 `paths.ffmpeg / ffprobe / ncnn`；示例中的其它段落为预留项 |
| 日志目录 | `%LOCALAPPDATA%\SuperResolutionTool\logs\` | 4 MB × 3 轮转 |
| 自定义模型注册表 | `<程序目录>\models\custom_models.json` | 每机状态，已 `.gitignore` |
| 临时文件 | 系统临时目录（`srt-video-*` / `sr_smoke_*`） | 任务结束即清理 |

环境变量：

| 变量 | 作用 |
|---|---|
| `SRT_LOG_LEVEL=debug` | 展开到每块 tile 的日志（等价于 `-v`） |
| `SRT_SETTINGS_DIR=<路径>` | 重定向设置与日志目录（便于测试与便携使用） |

---

## 日志与排查

一次图片任务会打印这样一条链路（数值为示意）：

```
解码 input.png · 1024x768 · PNG · 1.2 MB · 0.021s
设置 倍率 4x · 输出尺寸 保持超分尺寸 · 降噪 降噪模型预降噪（输入尺度）强度 0.50 · 锐化 0.25 · 运行时 DirectML · fp16
输入尺度预降噪完成 · 0.412s · 输出 1024x768
推理开始 · 共 9 块 tile · 输出 4096x3072
tile 4/9 · 17ms/块 · 剩余 0.08s              # 逐块进度（默认每若干块打印一条，-v 展开到每块）
推理完成 · 0.370s · 4096x3072 · 34.0 MP/s
后处理完成 · 边缘保持降噪 0.50 · 1.732s
编码 PNG（compress_level=6）→ 12.4 MB · 1.402s · 8.8 MB/s
完成 output_x4.png · 总耗时 4.5s（解码 0.02s / 推理 0.37s / 后处理 1.73s / 编码 1.40s）
```

- 终端、轮转文件与界面「运行日志」面板**同源**，界面不会只显示一个粗糙的百分比；
- 无头 / 打包环境（无 stdout）也不会丢信息，结论写进日志文件；
- 面板支持复制 / 清空，反馈问题可直接粘贴整段日志。

---

## 已知限制

| 限制 | 说明 |
|---|---|
| 仅 Windows | 显卡枚举走 DXGI / WMI，编码器路由为三厂 Windows 驱动 |
| 必须有独立显卡 | 无独显直接拒绝启动（设计行为）；核显机型不受支持 |
| 视频仅 8bit SDR | 10bit / HDR 输入会被明确拒绝，不会静默降级 |
| Vulkan 后端模型子集 | ncnn 侧只有 `x4plus` / `anime` 系列，其它模型自动回退 DirectML |
| CUDA / TensorRT 路径未实测 | 代码已按 EP 实现，但本机为 AMD，尚缺 NVIDIA 真机验证 |
| `4x-UltraSharp_fp16` 授权未声明 | 仅供本地个人使用，**不可用于商业分发**；介意授权可自行从模型列表移除 |
| 模型与二进制不入库 | 仓库不含 `.onnx` / FFmpeg / ncnn（体积原因），源码运行需按上文自行准备 |

---

## 路线图

核心功能已全部落地：环境检测与后端选择、ONNX 推理引擎（DirectML / CUDA / OpenVINO / CPU）、
Vulkan 引擎（ncnn）、图片管线（单张 / 批量 / 尺寸与压缩策略）、视频管线（逐帧 + 硬件编码 + 音频 remux）、
界面与打包（四页导航 / 对比预览 / 绿色免安装产物）、视觉与模型打磨（7 超分 + 2 降噪）。

待办：

- **NVIDIA 真机验证**：CUDA / TensorRT 路径已按 execution provider 实现，但开发机为 AMD，尚缺实测；
- **CI**：GitHub Actions 跑单元测试（GPU 用例自动 skip）与离屏 UI 审计脚本；
- **更多模型**：DRUNet / DPIR 一类纯 CNN 盲降噪（SCUNet 的公开 ONNX 在 DirectML 上会进程级崩溃，已排除）、
  Anime4KCPP 集成（`config.example.yaml` 已预留路径）；
- **配置接线**：让 `config.yaml` 的 `inference` / `video` 段落真正生效（目前只有 `paths` 段落被读取）。

---

## 常见问题

**启动弹窗「未检测到独立显卡」？**
这是设计行为：工具只服务独显（DXGI `DedicatedVideoMemory` 枚举，WMI 兜底），核显机型会直接退出。

**后端下拉框里有的项标「（未就绪）」？**
该后端对应的 execution provider 在当前运行时里不存在（例如没装 `onnxruntime-gpu` 时的 CUDA）。

**同样的图为什么 SwinIR 慢很多？**
SwinIR 单会话加载约 3 s、单块推理更重；换来的收益是干净图保真度（Set5 clean4x 高 6.2 dB）。
真实退化照片仍以 `realesr-general-x4v3` 最稳。

**PNG 输出为什么不用最高压缩？**
`optimize=True` 实测慢 11 倍只小 14%，默认档位已用压缩级别换取 10 倍保存速度；
要更小体积可在「输出压缩」里选「高压缩」。

**批量任务为什么没有对比预览？**
逐张输出会不断刷新对照，指代不清；需要对照时单张跑一次即可。

**能处理 10bit / HDR 视频吗？**
不能，目前只支持 8bit SDR，且会明确告知而不是给出一份偏色的输出。

---

## 许可与第三方资源

- **本仓库代码**：[MIT](LICENSE)。该许可只覆盖仓库中的源代码，不覆盖下列模型权重与第三方二进制。
- **内置模型**：均为公开预训练权重的格式转换，各自授权见 [`models/LICENSES.md`](models/LICENSES.md)，
  其中 `4x-UltraSharp_fp16` **无明确授权、不可商业分发**。模型权重不入库（`.gitignore`），
  由使用者按来源自行准备。
- **第三方二进制**：FFmpeg / FFprobe（gyan.dev full build，含 x264 / x265 等 GPL 组件）、
  ncnn 与 `realesrgan-ncnn-vulkan`（Real-ESRGAN 官方发布）均不入库，由使用者自行获取并遵守其授权。
- **运行期依赖**：PySide6（LGPL）、ONNX Runtime（MIT）、NumPy（BSD-3）、Pillow（MIT-CMU）。

---

## 致谢

本项目的模型与推理路线建立在以下开源工作之上：

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）：RRDB / SRVGG 架构与全部 Real-ESRGAN 权重
- [SwinIR](https://github.com/JingyunLiang/SwinIR)（Apache-2.0）：SwinIR-M 经典超分 / 真实世界超分 / 彩色降噪权重
- [ncnn](https://github.com/Tencent/ncnn) 与 `realesrgan-ncnn-vulkan`：Vulkan 后端
- [FFmpeg](https://ffmpeg.org/)：视频解码 / 硬件编码 / 音频封装
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)：DirectML / CUDA 推理
- [PySide6 / Qt](https://doc.qt.io/qtforpython/)：桌面界面
- [Upscayl](https://github.com/upscayl/upscayl)：对比预览的放大镜与滑块视图参考了它的交互设计

---

更多逐步操作说明见 [`QUICKSTART.md`](QUICKSTART.md)，架构与设计取舍见 [`docs/技术方案.md`](docs/技术方案.md)。
