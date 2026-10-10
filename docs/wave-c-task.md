# Wave C 委派任务书：模型选择入 Tab + 降噪强度调节

## 目标

1. 把模型选择（含倍率自动带出）从「设置」Tab 移到「图片超分」「视频超分」两个 Tab，各自独立选择
2. 降噪模型支持**降噪强度系数**连续调节（DNI 插值：general-x4v3 弱降噪 ↔ general-wdn-x4v3 强降噪）

## 前置事实

- 当前模型选择（`model_combo`/`model_path_edit`/`backend_combo`）在设置 Tab（`_build_settings_tab`）
- 图片/视频 Tab 已有倍率选择器（`image_scale_selector`/`video_scale_selector`），`_refresh_model_for_scene` 已实现「选模型→带出倍率」
- `realesr-general-x4v3`（弱降噪）与 `realesr-general-wdn-x4v3`（强降噪）是同一 SRVGG 架构、不同权重，官方用 DNI 权重插值实现降噪强度
- `app/core/types.py` 的 `ModelMeta` 是 frozen dataclass

## 要实现的内容

### 1. 降噪强度（核心）

**方案：输出插值**（V1 务实版，连续调节）

- `ModelMeta` 新增字段 `denoise_pair_path: Optional[Path] = None`（降噪对的第二个模型路径）
- 注册表：`realesr-general-x4v3` 标记 `denoise_pair_path = realesr-general-wdn-x4v3.onnx`；**从内置下拉里移除单独的 `realesr-general-wdn-x4v3`**（合并进 general 的降噪强度）
- `OnnxEngine`：
  - `load_model` 若 `denoise_pair_path` 存在，额外加载第二个 session（同 backend/providers）
  - `upscale_image` 新增 `denoise_strength: float = 0.5` 参数：`0.0`=纯弱降噪、`1.0`=纯强降噪；输出 = `(1-a) * out_weak + a * out_strong`
  - 无 denoise_pair 的模型忽略该参数
- `image_pipeline.upscale_single`/`upscale_batch` 与 `video_pipeline.upscale_video` 透传 `denoise_strength`

### 2. UI 重构（模型选择入 Tab）

- 新增可复用控件 `ModelSelector`（`app/ui/model_selector.py`）：含模型下拉、模型路径（浏览）、推理后端、降噪强度滑块（仅降噪模型显示）、倍率联动逻辑
- 图片 Tab、视频 Tab 各放一个 `ModelSelector` 实例（独立选择模型）
- 选中模型 → 自动带出对应倍率（复用 `model_scales`）；选中降噪模型 → 显示降噪强度滑块（`QSlider` 0~100 或 `QDoubleSpinBox`），否则隐藏
- 设置 Tab 保留：显卡信息、自定义模型导入（`copy_model_to_library` 相关）
- 两个 Tab 的任务启动分别使用各自的 ModelSelector 所选模型 + 倍率 + 降噪强度

### 3. 倍率联动确认

- 图片/视频 Tab 选中模型后，倍率选择器自动更新为该模型支持的倍率（`model_scales`）
- x2plus → 2x、general → 4x、anime → 4x、x4plus → 4x

## 验收标准

1. `python -m pytest -q` 全部通过（新增降噪插值、ModelSelector 逻辑、倍率联动测试）
2. 引擎实测：`denoise_strength=0.0` 输出 == general-x4v3 单独推理；`1.0` == wdn 单独推理；`0.5` 为两者平均
3. offscreen 实例化：图片/视频 Tab 各有模型下拉；选中 general 显示降噪滑块，选中 x2plus/anime 隐藏滑块
4. 倍率：选 x2plus → 倍率锁 2x；选 general → 4x

## 注意

- 输出插值是 DNI 权重插值的近似（SRVGG 含 PReLU 非线性），降噪强度调节目的（多/少降噪）可满足；若后续需精确权重插值再升级
- 降噪强度滑块默认值 0.5（平衡）
- 两个 Tab 的模型选择互不影响（独立状态）
- 完成后 `git commit`，message `feat: 模型选择入 Tab + 降噪强度调节`
