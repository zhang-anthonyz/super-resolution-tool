# Wave A 委派任务书：引擎 fp16 + 模型注册表 + 降噪 + UI 逻辑

## 目标

实现 fp16 推理支持、扩充模型注册表（4 个 fp16 模型）、降噪集成、性能优化，以及 UI 纯逻辑层改造（倍率按模型、场景改名、模型导入）。

## 前置事实

- 5 个 ONNX 模型已在 `models/onnx/`：
  | 文件 | dtype | 场景 | 特性 |
  |------|-------|------|------|
  | `realesr-general-x4v3.onnx` | fp16 | 真实/3D·CG | 通用 + **适度降噪**，SRVGG 轻量（快） |
  | `realesr-general-wdn-x4v3.onnx` | fp16 | 真实/3D·CG | 通用 + **强降噪** |
  | `RealESRGAN_x4plus_anime_6B.onnx` | fp16 | 2D 动漫 | 动漫/插画，边缘自然 |
  | `RealESRGAN_x4plus_fp16.onnx` | fp16 | 真实 | 高锐度、细节强（无降噪） |
  | `RealESRGAN_x4plus.onnx` | fp32 | 真实 | 同上 fp32 版（可删或保留作参考） |
- 全部固定 4x 倍率，无 scale-aware 输入
- DirectML 实测 fp16 比 fp32 快 **2.1x**
- 现有 `app/core/sr_engine.py` 的 `OnnxEngine`、`app/models/model_manager.py`、`app/ui/ui_logic.py`

## 要实现的内容

### 1. OnnxEngine 支持 fp16（`app/core/sr_engine.py` + `tiling.py`）

- `load_model` 后检测输入 dtype：`session.get_inputs()[0].type`，`tensor(float16)` → 引擎记 `self._fp16 = True`
- `upscale_image`：输入预处理用对应 dtype（fp16 模型 → `np.float16` 输入），输出后处理前转回 `np.float32` 再走现有 `postprocess_tensor`
- 具体：`preprocess_image` 增加可选 `dtype` 参数；`_upscale_tiled` 里 `feeds` 用 `self._fp16` 对应的 dtype

### 2. 模型注册表（`app/models/model_manager.py`）

`builtin_models()` 返回 4 个 fp16 模型，每个 `ModelMeta`：
- `name`（模型名）、`scale=4`、`scene`、`backend=DIRECTML`、`model_path`、`description`（含场景 + 降噪说明）
- 场景映射 `scene_default`：
  - `Scene.REALISTIC` → `realesr-general-x4v3`（通用降噪）
  - `Scene.ANIME_2D` → `RealESRGAN_x4plus_anime_6B`
  - `Scene.CG_3D` → `realesr-general-x4v3`
- 保留 `x4plus_fp16` 作为「高锐度无降噪」备选模型（在列表里，非默认）

### 3. 降噪集成

- 降噪通过模型选择实现：`general-x4v3`（适度降噪）vs `general-wdn-x4v3`（强降噪）
- 模型 `description` 里说明降噪特性；UI 下拉里可见

### 4. UI 纯逻辑（`app/ui/ui_logic.py`）

- **倍率按模型**：新增 `model_scales(model) -> list[int]`，固定 scale 的模型返回 `[scale]`，scale-aware 返回 `[2,3,4]`；`fixed_scale_for_model` 已存在可复用
- **场景改名**：`SCENE_OPTIONS` 语义不变，但 UI 展示时改为「超分模型」概念——新增 `model_options() -> list[tuple[str, str]]`（模型名 → 场景/降噪提示），供下拉使用
- **模型导入**：新增 `supported_model_formats()` 返回支持格式说明（`.onnx` + 元数据 `.json`），以及 `copy_model_to_library(src, models_dir) -> Path`（复制 onnx 到 `app_root()/models/onnx/`，重名处理）
- 删除/替换旧 `DEFAULT_MODEL_RELATIVE_PATH` 单模型假设，改为从注册表取

### 5. 性能

- 默认加载 fp16 模型（注册表里全是 fp16）
- 保留 fp32 模型路径兼容（引擎自动检测 dtype）

## 验收标准

1. `python -m pytest -q` 全部通过（新增 fp16 引擎、注册表、倍率按模型、模型导入测试）
2. 4 个 fp16 模型均能经 `OnnxEngine` 加载并超分（DirectML），输出尺寸正确、dtype=uint8
3. 实测：128×128 → 512×512 用 `realesr-general-x4v3`（SRVGG 轻量）耗时明显低于 x4plus
4. `scene_default` 三种场景都返回有效模型；`model_scales` 对固定模型返回 `[4]`

## 注意

- fp16 输入输出 dtype 转换要在引擎层透明处理，不影响上层管线（image/video pipeline）
- `general` 系列是 SRVGG（轻量、快），`x4plus` 是 RRDB（重、慢但细节强）——描述里体现差异
- 完成后 `git commit`，message `feat: fp16 引擎 + 模型注册表扩充 + 降噪集成`
