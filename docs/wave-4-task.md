# Wave 4 委派任务书：图片管线 + 模型管理

## 目标

实现图片超分管线（单张 + 批量）与模型管理，让工具能端到端超分图片。

## 前置事实

- DirectML 引擎（W2）已就绪：`OnnxEngine`（`app/core/sr_engine.py`）
- 环境检测（W1）已就绪：`detect_gpus` / `check_environment` / `available_backends`
- 真实 ONNX 模型已就位：`models/onnx/RealESRGAN_x4plus.onnx`（4x 通用，输入输出 [0,1] RGB）
  - 若尚未生成，先运行 `python scripts/export_onnx.py --input models/onnx/RealESRGAN_x4plus.pth --output models/onnx/RealESRGAN_x4plus.onnx --scale 4`
- 契约类型在 `app/core/types.py`（`ModelMeta`/`Scene`/`Backend`/`SuperResResult`/`ProgressInfo`）

## 边界

- 新增 `app/models/model_manager.py`（模型管理）
- 新增 `app/pipelines/image_pipeline.py`（图片管线）
- 不得改动 `types.py` 现有字段、`sr_engine.py` 的 `OnnxEngine` 接口、`environment_checker.py` 接口
- 可新增 `app/models/builtin_models.py`（内置模型注册表）

## 要实现的内容

### 1. 模型管理（`app/models/model_manager.py`）

- `builtin_models()` → `list[ModelMeta]`：内置模型注册表，V1 先注册：
  - `RealESRGAN_x4plus`：scale=4、scene=REALISTIC、backend=DIRECTML、path=`models/onnx/RealESRGAN_x4plus.onnx`
  - 注册表留好扩展点，后续补充 anime/general/SRMD 模型
- `scene_default(scene: Scene) -> ModelMeta`：场景 → 默认模型映射
- `load_model(meta) -> OnnxEngine`：根据 meta.backend 实例化引擎并 load_model
- `import_custom_model(onnx_path, meta_json_path) -> ModelMeta`：导入自定义 ONNX + 元数据 JSON（元数据字段与 ModelMeta 对齐，缺失时智能探测 scale 输入）
- `download_model(url, dest, sha256=None)`：模型下载（requests/urllib，断点续传可选，SHA256 校验）

### 2. 图片管线（`app/pipelines/image_pipeline.py`）

- `SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}`
- `upscale_single(image_path, out_path, engine, scale) -> SuperResResult`：
  - Pillow 打开 → RGB ndarray → engine.upscale_image → 保存（jpg/png/webp 按扩展名）
  - 异常捕获为 `SuperResResult(ok=False, error=...)`
- `upscale_batch(paths, out_dir, engine, scale, progress_cb, cancel_flag) -> list[SuperResResult]`：
  - 输入支持：文件列表 + 文件夹（递归，按 SUPPORTED_EXTENSIONS 过滤）
  - 输出命名：`{stem}_x{scale}{ext}`；已存在时按配置覆盖或跳过
  - 进度回调按「当前张数/总数」汇报；取消标志检查；失败单张不中断整体，收集错误列表

## 验收标准

1. `python -m pytest -q` 全部通过（新增 model_manager / image_pipeline 测试）
2. 单测覆盖：
   - `builtin_models` / `scene_default` 映射正确
   - 扩展名过滤、批量输出命名、文件夹递归
   - 失败单张不中断、错误收集
3. 端到端实测：用 PIL 生成一张 128×128 测试图，经 `upscale_single`（DirectML + RealESRGAN_x4plus）超分到 512×512，输出文件存在且尺寸正确

## 注意

- 图片加载统一转 RGB（PIL 的 `convert("RGB")`，处理 RGBA/灰度/palette 图）
- webp 保存用 Pillow 原生支持；注意 JPEG 无 alpha 通道
- 批量用「GPU 串行 + 进度回报」，不要多线程并发推理（单卡）
- `SuperResCancelled` 要向上传递为取消状态，不当作普通错误
- 完成后 `git commit`，message 格式 `feat(w4): 实现图片管线与模型管理`
