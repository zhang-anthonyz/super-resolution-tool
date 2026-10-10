# 打包任务书：PyInstaller 绿色免安装版

## 目标

用 PyInstaller 打成开箱即用的绿色免安装目录（one-dir），用户双击 exe 即可运行，无需安装 Python。

## 前置事实

- 依赖已装：PySide6 6.12、onnxruntime-directml 1.24.4、numpy 2.5、Pillow、PyInstaller 6.22
- `vendor/ffmpeg.exe` + `vendor/ffprobe.exe` 已就位（181MB gyan.dev full build，含三厂编码器）
- `models/onnx/RealESRGAN_x4plus.onnx` 已就位（63.9MB）
- 本机 AMD，DirectML 后端可测

## 关键：路径统一（打包前必须改）

打包后 `__file__` 指向 PyInstaller 临时目录，现有 `_PROJECT_ROOT = Path(__file__).resolve().parents[2]` 会失效。需统一：

1. 新增 `app/core/paths.py`：
   ```python
   def app_root() -> Path:
       if getattr(sys, "frozen", False):  # PyInstaller 打包后
           return Path(sys.executable).resolve().parent
       return Path(__file__).resolve().parents[2]
   ```
2. 改造以下模块，统一用 `app_root()` 定位 `vendor/`、`models/`：
   - `app/core/environment_checker.py` 的 `_resolve_vendor_path`
   - `app/models/model_manager.py` 的 `_resolve_model_path` / `_DEFAULT_MODEL_PATH`
   - `app/pipelines/ffmpeg_utils.py` 的 ffmpeg/ffprobe 定位：**优先 `app_root()/vendor/ffmpeg.exe`，不存在再回退 `shutil.which("ffmpeg")`**（当前只用 PATH，打包后 PATH 里没有 ffmpeg，必须改）

## 要实现的内容

1. **路径统一**（见上）
2. **PyInstaller spec**（`SuperResolutionTool.spec`）：
   - one-dir 模式，入口 `app/main.py`，name `SuperResolutionTool`
   - `datas`：`vendor/ffmpeg.exe`、`vendor/ffprobe.exe`、`models/onnx/RealESRGAN_x4plus.onnx`（复制到 dist 对应目录）
   - 收集 onnxruntime-directml 的 DLL（onnxruntime 有官方 hook，DirectML 版若缺 DLL 需手动补充）
   - PySide6 走官方 hook
   - 排除 torch/onnxscript/pytest 等开发依赖（减小体积）
3. **打包**：`pyinstaller SuperResolutionTool.spec`
4. **验证产物**：
   - `dist/SuperResolutionTool/SuperResolutionTool.exe` 能启动（先 offscreen smoke 测入口逻辑）
   - 产物目录下 `vendor/` 和 `models/onnx/` 存在
   - 用产物实际超分一张图（`QT_QPA_PLATFORM=offscreen` 或直接命令行跑管线，验证 DirectML + ffmpeg + 模型从产物目录加载）

## 验收标准

1. `python -m pytest -q` 全部通过（路径改造后原有 77 测试仍绿）
2. `pyinstaller SuperResolutionTool.spec` 成功，`dist/SuperResolutionTool/` 生成
3. 产物目录含 `vendor/ffmpeg.exe`、`models/onnx/RealESRGAN_x4plus.onnx`
4. 产物可执行文件能启动（offscreen），且能加载模型/ffmpeg 完成一次超分

## 注意

- **V1 仅打包 onnxruntime-directml**（三厂通吃 DirectML）；CUDA/TensorRT 需 onnxruntime-gpu 且本机无 NVIDIA 无法验证，留作后续在 NVIDIA 机器上补充
- torch/onnxscript 是导出模型的开发工具，**不打包**进运行时
- 体积预期：ffmpeg 181MB + 模型 64MB + PySide6/onnxruntime 数百 MB，总体数百 MB 属正常
- 完成后 `git commit`，message 格式 `feat(build): PyInstaller 打包支持`
