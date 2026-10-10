# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH).resolve()

onnx_models = [
    (str(path), "models/onnx")
    for path in (project_root / "models" / "onnx").glob("*.onnx")
]

datas = [
    (str(project_root / "vendor" / "ffmpeg.exe"), "vendor"),
    (str(project_root / "vendor" / "ffprobe.exe"), "vendor"),
    (str(project_root / "vendor" / "realesrgan-ncnn-vulkan.exe"), "vendor"),
    (str(project_root / "vendor" / "vcomp140.dll"), "vendor"),
    (str(project_root / "vendor" / "ncnn-models"), "vendor/ncnn-models"),
] + onnx_models

binaries = collect_dynamic_libs("onnxruntime")
hiddenimports = collect_submodules("onnxruntime.capi")

a = Analysis(
    [str(project_root / "app" / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "onnxscript",
        "pytest",
        "onnx",
        "sympy",
        "networkx",
        "scipy",
        "pandas",
        "matplotlib",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SuperResolutionTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    contents_directory=".",
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SuperResolutionTool",
)
