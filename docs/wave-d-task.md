# Wave D 委派任务书：UI 排版优化 + 详细日志 + 进度条细化

## 目标

1. 优化 UI 排版（控件拥挤问题）
2. 日志详细输出（每个操作都有日志）
3. 进度条细化（不要 0% 直接跳到 100%）

## 前置事实

- `app/pipelines/image_pipeline.py` 的 `upscale_single` 调 `engine.upscale_image` **未传 progress_cb**，导致单张图 tile 级进度丢失
- `upscale_batch` 只按「张数」报进度（`_report_progress` current=张序号），单张图时 0→100
- `OnnxEngine.upscale_image` 已支持 tile 级 progress_cb（按 tile 数回调）
- `app/ui/main_window.py` 有 `LogPanel.append(level, message)`，但日志点较少（仅开始/完成/错误）
- `app/ui/model_selector.py` 的 QFormLayout 5 行控件（模型/文件/后端/降噪/信息）较拥挤

## 要实现的内容

### 1. 进度条细化（核心）

- `upscale_single` 新增 `progress_cb` 参数，透传给 `engine.upscale_image(image, scale, denoise_strength=..., progress_cb=tile_cb)`
- `upscale_batch` 综合张级 + tile 级进度：
  - 传入一个「tile 进度回调」，把当前张的 tile 进度映射为整体进度：
    `整体 current = (index - 1) + (tile_current / tile_total)`，`整体 total = total_images`
  - 即批量 10 张时，第 3 张处理到一半 → 进度 `(2 + 0.5) / 10 = 25%`
  - `ProgressInfo` 的 `message` 带「第 index/total 张 · tile x/y」
- 视频管线帧级进度已有，确认 `_report_progress` 的 `current/total` 已反映帧进度

### 2. 详细日志

在 `main_window.py` 的图片/视频任务流程中，**每个关键操作追加日志**：

- 图片超分：
  - `Info` 检测到显卡（厂商/型号/显存）
  - `Info` 加载模型：模型名 / 后端 / 倍率 / 降噪强度（如有）
  - `Info` 开始处理 N 张图片
  - 每张：`Info` 处理 `文件名`（进度百分比）→ 完成后 `Info` 保存到 `路径`（耗时 Xs）
  - `Success` 完成统计：成功 X / 失败 Y / 总耗时
  - `Error` 单张失败（文件路径 + 原因）
- 视频超分：
  - `Info` 视频信息（分辨率/帧率/帧数/编码器）
  - `Info` 逐帧进度（节流，如每 10 帧或每秒一条）
  - `Success` 完成（输出路径 + 耗时）
- 进度回调的 message 同步追加到日志（tile/帧进度节流输出，避免刷屏）

### 3. UI 排版优化

- `ModelSelector`：用 `QGroupBox` 分组（如「模型」「降噪」），增加行间距（spacing 10~12），控件宽度合理（模型下拉不无限拉长）
- 图片/视频 Tab：整体布局增加留白，分组间距统一；「输入」「输出与倍率」「任务」分组之间 spacing 合理
- 视觉上不再「挤在一起」，分组清晰、层次分明

## 验收标准

1. `python -m pytest -q` 全部通过（新增 upscale_single/批量 tile 进度、日志相关测试）
2. 实测：单张图超分时进度条有**中间值**（tile 级，非 0→100 跳变）
3. 批量 5 张图，进度条能反映「第几张 + 该张的 tile 进度」的连续变化
4. 日志面板能看到：加载模型、每张图处理+保存+耗时、完成统计
5. offscreen 实例化主窗口，视觉布局分组清晰

## 注意

- 日志节流：tile/帧级进度日志不要每条都输出（刷屏），按合理间隔（如进度变化 ≥10% 或每秒）
- 进度条 value 用整数（QProgressBar），浮点 current 四舍五入
- 完成后 `git commit`，message `feat: UI 排版优化 + 详细日志 + 进度条细化`
