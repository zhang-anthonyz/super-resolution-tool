# Wave B 委派任务书：Vulkan 后端 + UI 视觉调整

## 目标

1. 实现 Vulkan 后端（realesrgan-ncnn-vulkan），让 AMD 用户可选 Vulkan
2. 修复进度条（初始静止 + 任务百分比）
3. 重新设计界面主题（摆脱纯黑）

## 前置事实

- `vendor/realesrgan-ncnn-vulkan.exe` + `vendor/vcomp140.dll` + `vendor/ncnn-models/` 已就位
- ncnn 模型（`vendor/ncnn-models/`）：`realesrgan-x4plus`、`realesrgan-x4plus-anime`、`realesr-animevideov3-x2/x3/x4`
- ncnn exe 用法：`realesrgan-ncnn-vulkan.exe -i in.png -o out.png -n <model> -s <scale>`（`-n` 用模型名不含扩展名）
- `app/core/sr_engine.py` 已有 `VulkanEngine` 空壳（`load_model`/`upscale_image` 抛 NotImplementedError）
- `app/ui/workers.py` 的 `_load_engine` 目前对 VULKAN 抛「暂未就绪」

## 要实现的内容

### 1. VulkanEngine（`app/core/sr_engine.py`）

- `load_model(model)`：记录模型名与 ncnn 可执行路径（`app_root()/vendor/realesrgan-ncnn-vulkan.exe`），校验存在
- `upscale_image(image, scale, progress_cb)`：文件级实现
  1. 写临时输入 png（Pillow）
  2. subprocess 调 `realesrgan-ncnn-vulkan.exe -i in.png -o out.png -n <ncnn_model> -s <scale>`
  3. 读回输出 png → RGB ndarray
  4. 完成后清理临时文件；检查 `cancelled` 抛 `SuperResCancelled`
- ncnn 模型名映射：`RealESRGAN_x4plus`→`realesrgan-x4plus`、`RealESRGAN_x4plus_anime_6B`→`realesrgan-x4plus-anime`、`animevideov3`→`realesr-animevideov3-x{scale}`（无对应映射时用模型文件名 stem 作为 `-n` 值）

### 2. 后端可用性（`app/core/environment_checker.py`）

- `available_backends` 检测 `vendor/realesrgan-ncnn-vulkan.exe` 存在 → AMD 返回含 `VULKAN`（已有 `_ncnn_path` 逻辑，确认指向正确文件名）

### 3. workers 支持 Vulkan（`app/ui/workers.py`）

- `_load_engine`：VULKAN 分支实例化 `VulkanEngine(backend)` 并 `load_model`，不再抛「暂未就绪」

### 4. 进度条修复（`app/ui/main_window.py`）

- 图片/视频两个 `QProgressBar` 初始改为 `setRange(0, 100)` + `setValue(0)`（**静止不滚动**）
- 任务开始时重置为 0；任务中根据 `ProgressInfo.current/total` 更新 `setRange(0, total)` + `setValue(current)`，进度条显示**具体百分比**
- 任务结束恢复静止状态

### 5. 主题重新设计（`app/ui/main_window.py` 或新增 `app/ui/theme.py`）

- 摆脱纯黑，采用**现代、有层次感的配色**（例如浅灰蓝深色主题，或干净浅色主题），语义色（成功绿/警告橙/错误红）仅用于状态提示
- 用统一 QSS 或 Qt palette 实现，覆盖 QMainWindow/QGroupBox/QPushButton/QProgressBar/QLabel 等
- 控件圆角、留白、字体层级清晰，观感专业（参考 Linear/Notion 类现代桌面风格）

## 验收标准

1. `python -m pytest -q` 全部通过（新增 VulkanEngine、后端可用性、进度条逻辑测试）
2. 本机实测：`VulkanEngine` 能调 ncnn 超分一张图（AMD Vulkan），输出正确
3. `available_backends(AMD)` 现在返回含 `VULKAN`
4. offscreen 实例化主窗口：进度条 `value()==0` 且 `maximum()==100`（静止）
5. 主题肉眼可见非纯黑、观感现代

## 注意

- Vulkan 后端是文件级粗粒度，进度按「文件级」回报（当前只有单张，进度 0→1）
- ncnn exe 首次运行可能打印 Vulkan 设备信息到 stderr，忽略即可
- 完成后 `git commit`，message `feat: Vulkan 后端 + 进度条修复 + 主题重设计`
