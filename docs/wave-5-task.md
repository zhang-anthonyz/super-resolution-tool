# Wave 5 委派任务书：视频超分管线

## 目标

实现单条视频超分：ffmpeg 解码 → 逐帧超分 → 硬件编码 → 音频 remux，支持三厂编码器路由与 8bit 拦截。

## 前置事实

- 引擎（W2）`OnnxEngine`、环境检测（W1）`detect_video_bit_depth`、模型（W4）已就绪
- 本机 ffmpeg/ffprobe 已在 PATH（含三厂硬编：h264_amf/hevc_amf、h264_nvenc、h264_qsv 等）
- 本机 AMD：视频编码默认走 `h264_amf`/`hevc_amf`
- 契约类型见 `app/core/types.py`（`VideoCodec`/`SuperResResult`/`ProgressInfo`/`GpuVendor`）

## 边界

- 新增 `app/pipelines/video_pipeline.py`
- 可新增 `app/pipelines/ffmpeg_utils.py`（ffmpeg 子进程/编码器路由辅助）
- 不得改动 `types.py` 现有字段、`sr_engine.py` 接口、`image_pipeline.py` 接口

## 要实现的内容

### 1. ffmpeg 辅助（`app/pipelines/ffmpeg_utils.py`）

- `probe_video(path) -> dict`：ffprobe 取 width/height/fps/pix_fmt/color_transfer/color_primaries/总帧数/是否有音频流
- `encoder_for_vendor(vendor, codec) -> str`：编码器路由
  - H264：NVIDIA→`h264_nvenc`、AMD→`h264_amf`、Intel→`h264_qsv`、兜底→`libx264`
  - HEVC：NVIDIA→`hevc_nvenc`、AMD→`hevc_amf`、Intel→`hevc_qsv`、兜底→`libx265`
- `run_ffmpeg(args) -> subprocess.Popen`：安全启动 ffmpeg 子进程（不经过 shell）

### 2. 视频管线（`app/pipelines/video_pipeline.py`）

`upscale_video(input_path, out_path, engine, scale, codec=VideoCodec.H264, progress_cb=None, cancel_flag=None) -> SuperResResult`

流程：
1. **8bit 拦截**：`detect_video_bit_depth` 检测，`bit_depth > 8` 或 `is_hdr` → 返回 `ok=False, error="仅支持 8bit SDR，检测到 10bit/HDR 视频"`，不开始处理
2. **probe**：取 width/height/fps/帧数；输出尺寸 = `width*scale × height*scale`
3. **解码 pipe**：`ffmpeg -i input -f rawvideo -pix_fmt rgb24 -`（stdout），Python 按 `W*H*3` 字节逐帧读，reshape `(H,W,3)` uint8
4. **逐帧超分**：`engine.upscale_image(frame, scale)`；每帧检查取消；进度按「当前帧/总帧数」汇报
5. **编码 pipe**：`ffmpeg -f rawvideo -pix_fmt rgb24 -s {W*scale}x{H*scale} -r {fps} -i - -c:v {encoder} -pix_fmt yuv420p {quality} out_video.mp4`（stdin 写回超分帧）
6. **音频 remux**：若有音频流，`ffmpeg -i input -vn -c:a copy audio.ext` 提取，再 `ffmpeg -i out_video.mp4 -i audio.ext -c:v copy -c:a copy -movflags +faststart out_path`
   - 无音频流则直接改名
7. **编码器回退**：硬编启动失败（如无该厂商硬编）自动回退 `libx264`/`libx265` 并 log
8. **取消**：`SuperResCancelled` 向上传递；中断时清理临时文件

## 验收标准

1. `python -m pytest -q` 全部通过
2. 单测覆盖：`encoder_for_vendor` 三厂路由 + 兜底、`probe_video` 解析（mock ffprobe 输出）、8bit 拦截逻辑、帧尺寸计算
3. 集成测试：用 ffmpeg 生成一段 2 秒 320×240 测试视频（`testsrc` + 正弦音频），经 `upscale_video`（DirectML + RealESRGAN_x4plus）超分到 1280×960，验证输出存在、分辨率正确、时长正确（音画同步）
   - 集成测试标注 `@pytest.mark.slow`，可 `-m "not slow"` 跳过（避免常规测试过慢）

## 注意

- 视频逐帧超分很慢（单帧 ~0.1-0.2s），测试视频务必短（≤2 秒）
- rawvideo pipe 用 `subprocess.Popen` 的 `stdout=PIPE`/`stdin=PIPE`，注意避免管道死锁（读写要配对、及时 flush）
- rgb24 无音频，音频单独提取；H.264 输出 `-pix_fmt yuv420p` 保证兼容
- 帧数未知时（VFR）用「已处理帧数」进度，不强制总帧
- 完成后 `git commit`，message 格式 `feat(w5): 实现视频超分管线`
