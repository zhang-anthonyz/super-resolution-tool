"""下载并准备打包所需的 FFmpeg / FFprobe 二进制。

源码构建前运行一次即可：

    python scripts/prepare_vendor.py                      # 默认：BtbN（GitHub，zip）
    python scripts/prepare_vendor.py --source gyan        # gyan.dev essentials（zip）
    python scripts/prepare_vendor.py --source gyan-full   # gyan.dev 完整构建（.7z，需 7-Zip）

产物落在 ``vendor/ffmpeg.exe`` 与 ``vendor/ffprobe.exe``，即
``SuperResolutionTool.spec`` 与运行时查找的位置。三个源都是 GPL 静态构建、
都带 nvenc / amf / qsv；默认用 BtbN 的 GitHub release，因为从 GitHub 拉取
通常比 gyan.dev 快一个数量级（实测 7 MB/s vs 0.1 MB/s）。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "vendor"
REQUIRED = ("ffmpeg.exe", "ffprobe.exe")

#: 源名 -> (下载地址, 压缩格式)
SOURCES = {
    "btbn": (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-win64-gpl.zip",
        "zip",
    ),
    "gyan": (
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "zip",
    ),
    "gyan-full": (
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z",
        "7z",
    ),
}
DEFAULT_SOURCE = "btbn"

#: 应用实际调用的三厂硬件编码器，下载后逐个确认存在。
EXPECTED_ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv")


def _human(size: float) -> str:
    return f"{size / 1048576:.1f} MB"


def _download(url: str, target: Path) -> None:
    """流式下载并打印进度。环境变量 HTTPS_PROXY 会被 urllib 自动采用。"""
    request = urllib.request.Request(
        url, headers={"User-Agent": "super-resolution-tool/prepare-vendor"}
    )
    last_report = 0.0
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last_report >= 0.5:
                last_report = now
                if total:
                    percent = done * 100 / total
                    print(
                        f"\r  下载 {_human(done)} / {_human(total)} ({percent:.0f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  下载 {_human(done)}", end="", flush=True)
    print()


def _extract_from_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        picked = {}
        for info in bundle.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if path.name.lower() in REQUIRED and path.parent.name.lower() == "bin":
                picked[path.name.lower()] = info
        missing = [name for name in REQUIRED if name not in picked]
        if missing:
            raise SystemExit(f"压缩包内未找到 {', '.join(missing)}（{archive.name}）")
        for name, info in picked.items():
            with bundle.open(info) as source, open(dest / name, "wb") as target:
                shutil.copyfileobj(source, target)


def _find_7z() -> str | None:
    for candidate in ("7z", "7za", "7zr"):
        found = shutil.which(candidate)
        if found:
            return found
    for path in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(path).is_file():
            return path
    return None


def _extract_from_7z(archive: Path, dest: Path) -> None:
    tool = _find_7z()
    if tool is None:
        raise SystemExit(
            "gyan-full 使用 .7z 压缩包，需要本机安装 7-Zip（https://www.7-zip.org/）。\n"
            "不想装 7-Zip 就用默认源（BtbN zip）或 --source gyan（essentials zip）。"
        )
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [tool, "x", str(archive), f"-o{tmp}", "-y"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"7-Zip 解包失败：{result.stderr.strip() or result.stdout.strip()}")
        resolved: dict[str, Path] = {}
        for path in Path(tmp).rglob("*.exe"):
            key = path.name.lower()
            if key in REQUIRED and key not in resolved and path.parent.name.lower() == "bin":
                resolved[key] = path
        missing = [name for name in REQUIRED if name not in resolved]
        if missing:
            raise SystemExit(f"压缩包内未找到 {', '.join(missing)}（{archive.name}）")
        for name, path in resolved.items():
            shutil.copy2(path, dest / name)


def _run_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise SystemExit(f"{executable.name} 无法执行：{result.stderr.strip()}")
    return (result.stdout.splitlines() or ["(无版本输出)"])[0].strip()


def _missing_encoders(ffmpeg: Path) -> list[str]:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout
    return [name for name in EXPECTED_ENCODERS if name not in output]


def _report(dest: Path) -> int:
    ffmpeg = dest / "ffmpeg.exe"
    ffprobe = dest / "ffprobe.exe"
    print(f"FFmpeg  {_human(ffmpeg.stat().st_size)}  {_run_version(ffmpeg)}")
    print(f"FFprobe {_human(ffprobe.stat().st_size)}  {_run_version(ffprobe)}")
    missing = _missing_encoders(ffmpeg)
    if missing:
        print(f"警告：缺少硬件编码器 {', '.join(missing)}（视频硬编会回退 CPU）")
        return 1
    print("硬件编码器齐全：" + ", ".join(EXPECTED_ENCODERS))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载打包所需的 FFmpeg / FFprobe 到 vendor/")
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"输出目录（默认 {DEFAULT_DEST}）",
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default=DEFAULT_SOURCE,
        help=f"下载源（默认 {DEFAULT_SOURCE}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 vendor 已就绪也重新下载",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    ready = all((dest / name).is_file() for name in REQUIRED)
    if ready and not args.force:
        print(f"vendor 已就绪：{dest}（重新下载请加 --force）")
        return _report(dest)

    dest.mkdir(parents=True, exist_ok=True)
    url, archive_format = SOURCES[args.source]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / Path(url).name
        print(f"从 {args.source} 下载：{url}")
        try:
            _download(url, archive)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(
                f"下载失败：{exc}\n"
                "若需要代理，先设置 HTTPS_PROXY=http://127.0.0.1:7890 再重试；\n"
                "也可以换源：--source gyan（gyan.dev essentials）或 --source gyan-full。"
            ) from exc
        print(f"下载完成：{_human(archive.stat().st_size)}")
        if archive_format == "7z":
            _extract_from_7z(archive, dest)
        else:
            _extract_from_zip(archive, dest)

    for name in REQUIRED:
        if not (dest / name).is_file():
            raise SystemExit(f"准备失败：{dest / name} 不存在")
    print(f"已写入 {dest}")
    return _report(dest)


if __name__ == "__main__":
    sys.exit(main())
