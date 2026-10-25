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
    (str(project_root / "assets" / "app.ico"), "assets"),
] + onnx_models

runtime_datas = [
    (str(runtime_dir), f"runtime/{runtime_dir.name}")
    for runtime_dir in (
        project_root / "runtime" / "cuda",
        project_root / "runtime" / "dml",
        project_root / "runtime" / "openvino",
    )
    if runtime_dir.is_dir()
]
datas += runtime_datas

shiboken_binaries = collect_dynamic_libs("shiboken6")
shiboken_root = (
    Path(shiboken_binaries[0][0]).parent if shiboken_binaries else None
)
binaries = collect_dynamic_libs("onnxruntime")
if shiboken_root is not None:
    binaries += [
        (str(path), ".")
        for path in shiboken_root.glob("*.dll")
    ]
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

# PyInstaller can pick up ICU DLLs from unrelated tools on PATH. Qt6Core in
# PySide6 expects the system ICU C ABI, while some local ICU builds only expose
# versioned C++ exports, which makes QtWidgets fail with WinError 127.
_drop_icu = {"icuuc.dll", "icuin.dll", "icudt78.dll"}
a.binaries = [
    entry for entry in a.binaries
    if Path(entry[0]).name.lower() not in _drop_icu
]

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
    icon=str(project_root / "assets" / "app.ico"),
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
