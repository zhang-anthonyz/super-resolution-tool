# 快速开始（Quick Start）

> 图片 / 视频超分工具 —— 完全本地离线的超分辨率工作台，兼容 NVIDIA / AMD / Intel 独立显卡。
> 本机已验证配置：AMD RX 7900 XT 24GB，DirectML 后端，产物 `dist/SuperResolutionTool/` 约 900 MB。

---

## 1. 运行前提

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11（64 位） |
| 显卡 | **必须有独立显卡**：NVIDIA / AMD / Intel 独显 |
| 显存 | 建议 ≥ 4 GB（4x 模型 + 512 分块；SwinIR 系列权重较大，建议 ≥ 6 GB） |
| 网络 | **不需要**。全部处理在本地完成，不上传任何素材 |

> 启动时程序会枚举显卡（DXGI，失败则 WMI 兜底）并挑选显存最大的独显。
> 若无独显，弹窗「未检测到独立显卡」并直接退出 —— 这是设计行为，不是故障。

后端映射规则：

| 显卡厂商 | 候选后端（按优先级） | 备注 |
|---|---|---|
| NVIDIA | CUDA → DirectML → CPU | CUDA 需 onnxruntime-gpu |
| AMD | DirectML → Vulkan → CPU | DirectML 为推荐后端 |
| Intel | DirectML → OpenVINO → CPU | 独显需 Arc / Iris Xe Max 等 |
| 未知 | CPU | 仅在无独显时才可能落到这里 |

后端下拉框中标注「（未就绪）」的项表示当前运行时缺少对应 provider，选择后无法启动任务。

---

## 2. 方式一：直接使用打包版（推荐，零安装）

```
SuperResolutionTool\SuperResolutionTool.exe
```

- 该目录已内置 Python 运行时、PySide6、ONNX Runtime、FFmpeg、ncnn 与 14 个 ONNX 模型，
  **不需要安装 Python 或任何依赖**。
- ⚠️ 目录必须整体保留，不要单独把 exe 拷出来 —— 模型、DLL、FFmpeg 都在同级目录里。
- 建议把整个 `SuperResolutionTool` 文件夹拷到固定位置（如 `D:\Tools\`）再运行。

验证产物完整性（无界面自检：加载默认模型 → DirectML → 内置 FFmpeg → 实跑一次 4x 超分）：

```powershell
.\dist\SuperResolutionTool\SuperResolutionTool.exe --headless
# 成功时退出码为 0；失败会打印 HEADLESS_SMOKE_FAILED: <原因>
```

---

## 3. 方式二：从源码运行（开发用）

```powershell
cd D:\code_workspace\super-resolution-tool

# 建议使用 Python 3.14（打包产物即 python314）
python -m pip install -r requirements.txt

# 按显卡二选一装 ONNX Runtime：
python -m pip install onnxruntime-directml   # AMD / Intel / 通用（DirectML）
python -m pip install onnxruntime-gpu        # NVIDIA（CUDA EP），两者不要同时装

python app\main.py            # 启动界面
python app\main.py --headless # 无界面自检
```

可选配置：把 `config.example.yaml` 复制为 `config.yaml`，可覆盖 FFmpeg / ffprobe / ncnn 路径、默认后端、
视频编码参数、模型缓存目录。配置缺失时全部走内置默认值，**不配也能跑**。

---

## 4. 界面结构

```
左侧导航
├── 工作台
│   ├── 图片超分   单张或批量，jpg / png / webp
│   ├── 视频超分   单条视频逐帧超分 + 硬件编码
│   └── 对比预览   原图 / 超分结果对照（单张任务完成后自动跳转）
└── 系统
    └── 设置       推理设备信息、导入自定义 ONNX 模型
```

左侧底部常驻提示「本地离线推理 · 不上传任何图片 / 视频」；顶部标题栏右侧显示
`厂商 / 显卡型号 / 显存 / 可用后端` 摘要。

---

## 5. 五分钟上手：图片超分

1. 进入 **图片超分** 页 → 点「添加文件」（可多选）或「添加文件夹」（递归收集）。
   列表内支持多选，「移除选中」可剔除误加项。
2. **模型与后端** 卡片：
   - **场景预设**：`真实` / `2D 动漫` / `3D·CG` —— 切换即自动套用该场景的默认模型与倍率（之后可手改）。
   - **超分模型**：默认下拉列出全部内置模型 + 已导入的自定义模型；选「自定义模型」时在
     **模型文件** 行填/浏览 ONNX 路径。
   - **推理后端**：AMD 选 DirectML、NVIDIA 选 CUDA。
   - **降噪模型** 与 **降噪强度 / 边缘锐化** 见第 7 节。
3. **输出与画质** 卡片：
   - **输出目录**：默认 `<程序目录>\output`，可改（自动记忆上次设置）。
   - **倍率**：`2x / 3x / 4x / 自定义(2–8)`。固定倍率模型（如 `RealESRGAN_x2plus` 固定 2x）
     会自动纠正并在日志里提示「已忽略所选 Nx」。
   - **输出尺寸**：`保持超分尺寸` / `回缩到原图尺寸` / `按百分比等比例缩放（相对超分结果）`。
     选百分比后右侧出现百分数输入框（5–400%）：按**超分结果**的尺寸等比例缩放，长宽同比不变形。
     例如 4x 模型选 50% 相当于最终 2x；4x 模型选 25% 与「回缩到原图尺寸」效果一致；
     100% 等同于保持超分尺寸（不做缩放、文件名也不加后缀）。
   - **输出压缩**：`高质量` / `高压缩` / `无损·最高质量`（作用于 JPEG 质量、PNG 压缩级别、
     WEBP 质量或无损）。
   - **覆盖已存在文件**：不勾选时同名输出会被跳过（日志标「已跳过」）。
4. 点 **开始超分**，进度条与「运行日志」实时刷新；随时可 **取消**。

输出命名规则：

| 场景 | 文件名 |
|---|---|
| 保持超分尺寸 | `<原名>_x<倍率>.<原扩展名>` |
| 回缩到原图尺寸 | `<原名>_x<倍率>_restored.<原扩展名>` |
| 按百分比缩放 | `<原名>_x<倍率>_p<百分比>.<原扩展名>`（如 `_p50`；100% 不加后缀） |

---

## 6. 五分钟上手：视频超分

1. 进入 **视频超分** 页 → 选 **视频文件**（单条）与 **输出目录**。
2. 模型与后端设置同图片页，两者各自独立记忆。
3. **输出与编码**：**编码器** `H.264` / `HEVC`，**倍率** `2x / 3x / 4x / 自定义`。
4. 点 **开始超分** → 逐帧推理后由内置 FFmpeg 硬件编码，音频原样复制（remux）。

输出固定为 `<原文件名>_x<倍率>.mp4`。

⚠️ 视频限制：**仅支持 8bit SDR**。检测到 10bit / HDR（如 `p010`、`smpte2084`、`bt2020`）
会直接返回失败并提示「仅支持 8bit SDR，检测到 10bit/HDR 视频」。
视频任务耗时与帧数成正比（`总帧数 × 每帧推理时间`），长视频建议先剪一小段试跑。

---

## 7. 模型与降噪速查

内置 12 个超分模型（`models/onnx/`）：

| 模型 | 倍率 | 定位 |
|---|---|---|
| realesr-general-x4v3 | 4x | 轻量通用，默认最快（真实 / 3D·CG 默认） |
| RealESRGAN_x2plus | 2x | 真实照片轻度放大（RRDB，官方 fp16 导出），无降噪 |
| RealESRGAN_x4plus_anime_6B | 4x | 2D 动漫（2D 动漫场景默认） |
| RealESRGAN_x4plus_fp16 | 4x | RRDB 重模型，高锐度 |
| SwinIR_x4_classical | 4x | 干净图保真度最高，较慢 |
| SwinIR_x4_realSR_GAN | 4x | 真实退化，纹理自然，较慢 |
| 4x-UltraSharp_fp16 | 4x | 社区锐利模型，细节最锐 |
| realcugan-2x-no-denoise | 2x | 2D 动漫 2x：Real-CUGAN 双 UNet，保留细节 |
| realcugan-2x-denoise1x | 2x | 2D 动漫 2x：轻度降噪（σ≈12.5） |
| realcugan-2x-denoise2x | 2x | 2D 动漫 2x：中度降噪（σ≈25） |
| realcugan-2x-denoise3x | 2x | 2D 动漫 2x：强降噪（σ≈50） |
| realcugan-2x-conservative | 2x | 2D 动漫 2x：保守档（降噪最轻） |

> 2x 分两条线：**真实照片**选 `RealESRGAN_x2plus`；**2D 动漫 / 截图**选 Real-CUGAN 五档
> （按画面的噪声程度挑降噪档，噪点多选 denoise2x/3x）。这 5 档都是 2.6 MB 的双 UNet，比 RRDB 轻。
> 注意：**2x 模型目前只能在 ONNX 后端（DirectML / CUDA / CPU）使用**，Vulkan(ncnn) 侧没有 2x 通用权重。

降噪模型（**降噪模型** 下拉框，独立于超分模型）：

| 选项 | 行为 |
|---|---|
| 无（内置边缘保持降噪） | 用导向滤波做边缘保持降噪（默认回退方案） |
| realesr-general-wdn-x4v3 | 4x 同倍率：与超分输出按强度插值 |
| SwinIR_colorDN25 | 1x：在**输入尺度**先预降噪，再送超分 |

- **降噪强度** 滑块 `0.00–1.00`，`0.00` 即「关闭」。
- **边缘锐化** 滑块默认 `0.25`（USM 锐化），过高会产生振铃，建议 0.1–0.4。
- **Vulkan** 后端仅内置 `x4plus` / `anime` 系列模型（ncnn 侧无 x2plus 与 general），
  其他模型会自动回退到 DirectML。

---

## 8. 导入自定义模型（设置页）

**设置 → 自定义模型 → 导入 ONNX 模型到模型库**：

- 仅接受 `.onnx`；导入后复制到 `<程序目录>\models\onnx\`，重名自动追加 ` (1)`。
- 导入成功后该模型出现在图片 / 视频页的 **超分模型** 下拉（标为「自定义 · 场景 · 说明」）。
- 自定义模型默认按「倍率可变（2/3/4）」处理；文件名含 `x4plus` 或 `realesrgan_x4` 前缀的
  会被识别为固定 4x。
- 推理时按 `input` / `output` 节点名、RGB、0–1 归一化读取；不符合该约定的模型需先转换
  （可用 `scripts/export_onnx.py`，支持 BasicSR / ESRGAN 旧命名自动重映射）。

---

## 9. 配置与状态文件位置

| 内容 | 路径 |
|---|---|
| 单次运行偏好（输出目录、倍率、后端、降噪/锐化强度、编码器…） | `%LOCALAPPDATA%\SuperResolutionTool\settings.json` |
| 运行日志（自动轮转，保留 3 份） | `%LOCALAPPDATA%\SuperResolutionTool\logs\super-resolution-tool.log` |
| 可选配置文件 | `<程序目录>\config.yaml`（由 `config.example.yaml` 复制） |
| 内置模型 | `<程序目录>\models\onnx\` |
| 默认输出目录 | `<程序目录>\output\` |

---

## 10. 运行日志怎么看

「运行日志」卡片与终端（从源码运行时）打印同一套分步信息，每张图片包含：

| 步骤 | 内容 |
|---|---|
| 解码 | 文件名 / 像素尺寸 / 源格式 / 文件大小 / 解码耗时 |
| 设置 | 倍率、降噪路由（降噪模型预降噪 / 同倍率插值 / 内置边缘保持）、锐化、实际运行时与精度（fp16 / fp32） |
| 推理 | 分块数量与 tile 尺寸、每块耗时、剩余时间；结束后给出总耗时与 MP/s |
| 后处理 | 导向滤波降噪 / 锐化强度与耗时 |
| 编码 | 格式、压缩参数、输出体积、耗时与 MB/s |
| 汇总 | 单张总耗时与各阶段占比；批量结束再给成功 / 失败数与平均耗时 |

想看到**每一块 tile** 的明细（排查用时），从源码运行时加 `-v`，或设置环境变量
`SRT_LOG_LEVEL=debug`：

```powershell
python app\main.py -v                                   # 终端打印每个 tile 的耗时
$env:SRT_LOG_LEVEL="debug"; python app\main.py           # 同上，环境变量方式
```

日志卡片右上角「复制 / 清空」可把整段日志拷进剪贴板，反馈问题时直接粘贴即可。

---

## 11. 对比预览（单张任务完成后自动出现）

单张图片任务完成后，界面会**自动切到「对比预览」页**——对比画面占满整个画布（不是挤在设置旁边的小卡片），
左侧导航随时可切回「图片超分」继续跑下一张。

| 方式 | 操作 |
|---|---|
| **放大镜对比**（默认） | 鼠标在图上移动，**画布中央**的两个放大格同步显示：左「原图」、右「超分输出」，看的是同一块区域；蓝框标出取样位置。滚轮 / 滑块调放大镜倍率（1x–10x） |
| **滑块对比** | 一条竖直分割线：左侧原图、右侧输出。**拖分割线**移动分界、**拖其它位置**平移画面、**滚轮以光标为中心缩放**、**双击复位到整图** |

工具栏：`对比方式` 分段按钮、`放大` 滑块（放大镜倍率 / 视图缩放，按模式各记一份）、
`适应窗口`、`1:1 像素`、`大窗口对比`（独立窗口，可最大化，尺寸被记住）、`返回图片超分`。

设计参考了 Upscayl 的对照视图：两个放大格**固定停在画布中央**——一个稳定的注视点，鼠标在图上怎么走都不用找它们；
当光标移到自己身下（怕挡住正在看的像素）时，放大格才**朝最近的空档轻微让位**（上限约半个格，且只走最短的一侧，
光标穿过格子中线时不会来回跳），光标离开即归位。缩放锚定光标，放大时看的细节不会跑掉；平移被限制在图片范围内，
画面不会被拖丢。

状态栏显示 `原图 1024×768 → 输出 4096×3072 · 1:1 预览`——超大图会按内存预算自动缩小预览，
并在状态里标明缩放比例（为了不让两张几十兆像素的图占满内存）。

原图一侧按**与输出相同的尺寸**呈现（等比缩放），所以两侧像素一一对应，看的是「超分比直接缩放多了什么细节」，
这也是各种超分对比图的通行做法。批量任务不显示预览（避免指代不清）；需要对比时单张跑一次即可。

---

## 12. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 启动弹「未检测到独立显卡」 | 只有核显 / 驱动异常（DXGI 枚举为空且 WMI 兜底失败）。装好独显驱动后重试 |
| 后端项显示「（未就绪）」 | 当前 onnxruntime 不含该 provider。AMD/Intel 装 `onnxruntime-directml`，NVIDIA 装 `onnxruntime-gpu` |
| 选 Vulkan 后仍走 DirectML | 正常回退：ncnn 只带 x4plus / anime 系列模型，或 `realesr-animevideov3` 之外的模型无 Vulkan 支持 |
| 视频报「仅支持 8bit SDR」 | 10bit / HDR 源暂不支持；先用 FFmpeg 转 8bit SDR 再处理 |
| 图片任务日志显示「已跳过」 | 输出同名文件已存在且未勾选「覆盖已存在文件」 |
| 首次推理卡顿 | 模型加载 + provider 初始化一次性开销，属正常；后续同模型任务会明显更快 |
| 换了目录后启动失败 | 打包产物必须整目录保留（模型、DLL、`vendor\ffmpeg.exe` 与 exe 同目录） |
| 显存不足 / 报推理错误 | 换更轻的模型（`realesr-general-x4v3`）、降低倍率，或关闭独立降噪模型走内置降噪 |

---

## 13. 开发者命令

```powershell
python -m pytest -q                     # 单元测试（GPU 集成测试默认跳过）
python -m pytest -m slow                # 显式运行集成测试（耗时，需独显）
python scripts/ui_preview.py            # 离屏渲染各页面并做布局/配色自检（截图到 D:\model_output\images）
python scripts/export_onnx.py --help    # .pth → ONNX（含旧命名自动重映射与保真校验）
pyinstaller SuperResolutionTool.spec    # 打包绿色免安装目录到 dist\
```

改 UI 后建议顺序：`ui_preview.py` 看布局截图 → `pytest -q` 保证逻辑不回归。

更多细节：`README.md`（特性总览）、`docs/技术方案.md`（架构与决策）、
`docs/model-export-notes.md`（模型导出与评测协议）、`models/LICENSES.md`（模型授权清单）。
