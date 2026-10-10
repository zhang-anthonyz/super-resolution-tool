# Wave 6 委派任务书：PySide6 图形界面

## 目标

实现 PySide6 桌面界面，集成图片/视频超分管线，异步执行、进度/取消/日志齐备，启动环境自检。

## 前置事实

- 管线已就绪：`image_pipeline`、`video_pipeline`、`model_manager`、`environment_checker`、`OnnxEngine`
- PySide6 已安装（如未装先 `pip install PySide6`）
- 契约类型：`types.py`（`Scene`/`Backend`/`VideoCodec`/`GpuInfo`/`SuperResResult`/`ProgressInfo`）
- 本机 AMD RX 7900 XT，DirectML 后端；真实模型 `models/onnx/RealESRGAN_x4plus.onnx`

## 边界

- 新增 `app/ui/` 包（`main_window.py`、`workers.py` 等）
- 新增 `app/main.py` 入口（启动自检 + 启动主窗口）
- 不得改动 `types.py` 字段、`sr_engine.py`、`image_pipeline.py`、`video_pipeline.py`、`model_manager.py` 接口
- 配色：深色为主、单一品牌色点缀（蓝/青）、语义色仅用于状态提示（成功绿/警告橙/错误红）

## 要实现的内容

### 1. 入口 `app/main.py`

- 启动自检：`check_environment()`；无独显时弹 `QMessageBox.critical` 提示「未检测到独立显卡」并退出
- 自检通过后显示主窗口

### 2. 主窗口 `app/ui/main_window.py`

- 顶部 Tab：`图片超分` / `视频超分` / `设置`
- **启动时显示检测到的显卡信息**（厂商/型号/显存/可用后端）
- **场景选择器**：真实 / 2D 动漫 / 3D·CG（下拉或单选），选后自动填默认模型（可改）
- **后端选择器**：按厂商显示可用后端（AMD 显示 DirectML/Vulkan），默认厂商推荐
- **视频 Tab 常驻提示**：小字「仅支持 8bit SDR 视频，10bit/HDR 暂不支持」

### 3. 图片 Tab

- 添加文件（多选）/ 添加文件夹（递归过滤 jpg/png/webp）
- 倍率选择（2/3/4，或自定义）
- 输出目录选择、覆盖开关
- 开始 / 取消按钮
- 进度条 + 日志面板（分级 Info/Warning/Error，带时间戳，可复制）

### 4. 视频 Tab

- 选择单条视频文件
- 编码器选择：H.264（默认）/ HEVC
- 开始 / 取消按钮、进度条（帧级）、日志

### 5. 异步模型 `app/ui/workers.py`

- `ImageUpscaleWorker(QThread)`：跑 `upscale_batch`，信号 `progress`/`finished`/`error`/`cancelled`
- `VideoUpscaleWorker(QThread)`：跑 `upscale_video`，同上
- 取消通过线程安全标志（`threading.Event` 或 QThread 的 requestInterruption）连接 `cancel_flag`

### 6. 错误分级

- 环境问题 → 启动拦截弹窗
- 参数/文件问题 → 内联提示（QMessageBox.warning 或状态栏）
- 运行时错误 → 日志面板分级记录，不崩溃

## 验收标准

1. `python -m pytest -q` 全部通过（新增 UI 逻辑测试，不含 GUI 交互）
2. UI 逻辑单测覆盖：场景→默认模型映射、后端选择器内容、8bit 提示逻辑（可用非 GUI 测试）
3. **smoke 测试**：`python -c "from app.main import ..."` 或 headless 创建主窗口（`QT_QPA_PLATFORM=offscreen`）能实例化不崩溃
4. 本机实测：`python app/main.py` 能启动窗口、自检显示 AMD RX 7900 XT、图片 Tab 能选文件并触发超分

## 注意

- GUI 测试用 `QT_QPA_PLATFORM=offscreen` 环境变量避免弹窗阻塞
- 所有耗时操作（超分）必须在 QThread，UI 线程不阻塞
- 信号连接用 `pyqtSignal`，跨线程更新 UI 走信号槽，不直接操作控件
- 完成后 `git commit`，message 格式 `feat(w6): 实现 PySide6 图形界面`
