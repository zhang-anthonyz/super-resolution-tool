# Wave 1 委派任务书：环境检测模块

## 目标

实现 `app/core/environment_checker.py` 中所有标记 `NotImplementedError` 的函数，完成启动环境检测能力（显卡枚举、独显判定、后端可用性、视频位深检测）。

## 边界

- 主要修改 `app/core/environment_checker.py`
- 可新增辅助模块（如 `app/core/_dxgi.py` 放 DXGI ctypes 封装、`app/core/_nvidia.py` 放 pynvml 封装），但**不得改动 `app/core/types.py` 的现有字段**
- 契约签名（函数名、参数、返回类型）**不得变更**，其他模块将依赖这些签名
- 纯 Python + ctypes + 子进程，不引入额外重型第三方依赖（pynvml 除外，已在 requirements.txt）

## 要实现的函数

1. `detect_gpus() -> list[GpuInfo]`
2. `check_environment() -> GpuInfo`
3. `available_backends(gpu: GpuInfo) -> list[Backend]`
4. `detect_video_bit_depth(video_path: str) -> tuple[int, bool]`

（`find_discrete_gpu` 已实现，无需改动。）

## 技术要点

### 1. DXGI 枚举（首选，精确显存）

用 ctypes 调 `dxgi.dll`：

- `CreateDXGIFactory` → `EnumAdapters` 遍历
- `IDXGIAdapter::GetDesc` 读 `DXGI_ADAPTER_DESC`：
  - `Description`（WCHAR[128]，型号名）
  - `VendorId`（0x10DE=NVIDIA、0x1002=AMD、0x8086=Intel）
  - `DedicatedVideoMemory`（UINT64，精确显存）
- 注意 64 位结构体对齐与 ctypes padding；Description 是宽字符串需 `.decode('utf-16')`
- WMI 的 `Win32_VideoController.AdapterRAM` 是 32 位 DWORD，超 4GB 会被截断，**仅作兜底**，不可作为主依据

### 2. 独显 vs 核显判定

- NVIDIA/AMD：`DedicatedVideoMemory > 0` 即独显（核显共享内存，dedicated 为 0 或极小）
- Intel：核显（UHD/Iris）`DedicatedVideoMemory` 通常为 0；独显（Arc/Iris Xe MAX）> 0
- 综合判定：`DedicatedVideoMemory > 0` 判定独显；对 Intel 额外参考型号名（含 Arc/Iris Xe MAX）避免误判

### 3. NVIDIA 复核（pynvml）

- `nvmlInit` → `nvmlDeviceGetCount` → `nvmlDeviceGetHandleByIndex` → `nvmlDeviceGetName` / `nvmlDeviceGetMemoryInfo` / `nvmlSystemGetDriverVersion`
- **无 NVIDIA 卡时 `nvmlInit` 会抛异常**，必须 try/except 兜底，不影响 AMD/Intel 检测

### 4. 后端可用性

按厂商静态返回可用后端列表（W2 接入 onnxruntime 后再补动态 provider 检测）：

- NVIDIA: `[CUDA, DIRECTML, CPU]`
- AMD: `[DIRECTML, VULKAN, CPU]`
- Intel: `[DIRECTML, CPU]`
- Vulkan 后端额外检测 `vendor/realesrgan-ncnn-vulkan.exe` 是否存在（不存在则从列表剔除）

### 5. 视频位深检测

用 ffprobe 子进程：

```
ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt,color_primaries,color_transfer -of json <video>
```

- bit_depth 从 `pix_fmt` 解析（`yuv420p`=8、`yuv420p10le`=10、`yuv420p12le`=12 等）
- is_hdr：`color_transfer` 为 `smpte2084`/`arib-std-b67`，或 `color_primaries` 为 `bt2020`
- ffprobe 路径优先从 config 读取，默认 `vendor/ffprobe.exe`；找不到时返回 `(8, False)` 并警告，不崩溃

## 验收标准

1. 代码可 import，语法正确，无运行时异常
2. 新增 `tests/test_environment_checker.py`，至少覆盖：
   - `find_discrete_gpu` 从混合列表选最大显存独显
   - `GpuInfo.dedicated_vram_gb` 换算正确
   - 厂商 ID 映射（0x10DE/0x1002/0x8086）
   - `detect_video_bit_depth` 的 pix_fmt 位深解析（可用 mock/样本，不依赖真实 ffprobe）
3. 在本机运行 `detect_gpus()` 能枚举出至少一张显卡（本机为 AMD RX 7900 XT，独显，24GB）
4. 用 `python -c "from app.core.environment_checker import detect_gpus, check_environment; print(detect_gpus())"` 实测输出

## 注意

- 所有外部调用（nvmlInit、ffprobe、DXGI）都要 try/except，检测失败不崩溃，返回合理默认值
- 本机无 NVIDIA 卡，nvidia 相关代码路径必须能在无 NVIDIA 卡时安全跳过
- 完成后提交 git commit，message 格式 `feat(w1): 实现环境检测模块`
