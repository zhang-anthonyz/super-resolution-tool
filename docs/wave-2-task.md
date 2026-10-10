# Wave 2 委派任务书：SR 引擎·ONNX 后端

## 目标

实现 ONNX Runtime 推理引擎（`OnnxEngine`）与后端选择器（`BackendSelector`），完成图片超分的核心推理能力。

## 边界

- 实现 `app/core/sr_engine.py` 的 `OnnxEngine` 类
- 实现 `app/core/backend_selector.py` 的 `BackendSelector` 三个方法
- 可新增辅助模块（如 `app/core/tiling.py` 放分块逻辑），但**不得改动 `app/core/types.py` 现有字段**
- 契约签名不得变更

## 环境事实（本机）

- Python 3.14.3、numpy 2.5.3、onnxruntime-directml 1.24.4 已安装
- 可用 provider：`['DmlExecutionProvider', 'CPUExecutionProvider']`
- 本机 AMD RX 7900 XT：**DirectML 后端可实测；CUDA 后端本机不可测**，代码需支持但用单测兜底

## 要实现的内容

### 1. BackendSelector（`app/core/backend_selector.py`）

- `default_backend()`：NVIDIA→`CUDA`、AMD→`DIRECTML`、Intel→`DIRECTML`、UNKNOWN→`CPU`
- `select(preferred)`：`preferred` 为空用默认；不在该厂商可用列表时回退默认并 log 原因
- `load_onnxruntime(backend)`：返回 onnxruntime 模块
  - 开发阶段：直接 `import onnxruntime`（本机已装 directml 版），并校验目标 provider 存在
  - 双运行时目录加载（`runtime/cuda`、`runtime/dml`）留清晰 TODO 注释，W6 打包阶段实现

### 2. OnnxEngine（`app/core/sr_engine.py`）

实现 `load_model` 与 `upscale_image`：

- `load_model(model: ModelMeta)`
  - 用 `onnxruntime.InferenceSession` 加载 `model.model_path`
  - EP 选择：DIRECTML→`['DmlExecutionProvider','CPUExecutionProvider']`、CUDA→`['CUDAExecutionProvider','CPUExecutionProvider']`、CPU→`['CPUExecutionProvider']`
  - 输入/输出名：若 model 未声明则取 `session.get_inputs()[0].name` / `get_outputs()[0].name`
  - 精度：DirectML 默认 fp32；CUDA 默认 fp16（可切），失败回退 fp32

- `upscale_image(image, scale, progress_cb)`
  - 预处理：uint8→float32、`[0,255]→[0,1]`、HWC→CHW、加 batch 维（NCHW）
  - tile 分块：默认 `tile_size=512`、`tile_pad=10` 重叠区；显存不足时降 256（V1 可先固定策略）
  - 逐 tile 推理 + 拼接（重叠区取平均）
  - scale-aware：若 `model.scale_input` 存在（如 SRMD）额外传 scale 张量；`denoise_input` 同理
  - 后处理：`[0,1]→[0,255]`、CHW→HWC、uint8
  - 每 tile 后检查 `self.cancelled`，取消抛 `SuperResCancelled`
  - 进度回调按 tile 数汇报

## 验收标准

1. `python -m pytest -q` 全部通过（新增 OnnxEngine/BackendSelector 测试）
2. 单测覆盖：
   - `BackendSelector.default_backend` / `select` 的厂商路由
   - tile 分块与拼接（numpy 构造，不依赖真实模型）
   - 预处理/后处理数值正确性
3. 集成测试：用 `onnx` 库构造简单 ONNX 模型（如 2x 最近邻上采样），验证 OnnxEngine 在 **DirectML EP** 下能加载并输出正确尺寸/数值（若缺 onnx 库先 `pip install onnx`）
4. 本机实测：构造 64×64 随机图，经 OnnxEngine（DirectML）超分到 128×128 输出

## 注意

- onnxruntime-directml 的 fp16 支持有限，**DirectML 路径默认 fp32，不要默认 fp16**
- numpy 2.x 移除了 `np.float_` 等别名，避免使用弃用 API
- tile 拼接边界：图尺寸不整除 tile_size 时，最后一块 clamp 到图边界，勿越界
- 保持 `SuperResEngine` 抽象接口不变
- 完成后 `git commit`，message 格式 `feat(w2): 实现 ONNX 推理引擎`
