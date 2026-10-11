"""Model registration, custom import, and download helpers."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from ..core.paths import app_root
from ..core.sr_engine import OnnxEngine
from ..core.types import Backend, ModelMeta, Scene

_SR_MODEL_NAMES = (
    "realesr-general-x4v3",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "RealESRGAN_x4plus_fp16",
    "SwinIR_x4_classical",
    "SwinIR_x4_realSR_GAN",
    "4x-UltraSharp_fp16",
)

#: 纯降噪模型：可作为降噪模型挂到任意超分模型上（1x 走输入尺度预降噪，
#: 同倍率则与超分输出按强度插值）。
_DENOISE_MODEL_NAMES = (
    "realesr-general-wdn-x4v3",
    "SwinIR_colorDN25",
)

_MODEL_SCALES = {
    "realesr-general-x4v3": 4,
    "RealESRGAN_x2plus": 2,
    "RealESRGAN_x4plus_anime_6B": 4,
    "RealESRGAN_x4plus_fp16": 4,
    "SwinIR_x4_classical": 4,
    "SwinIR_x4_realSR_GAN": 4,
    "4x-UltraSharp_fp16": 4,
    "realesr-general-wdn-x4v3": 4,
    "SwinIR_colorDN25": 1,
}

_MODEL_PATHS = {
    "realesr-general-x4v3": Path("models/onnx/realesr-general-x4v3.onnx"),
    "realesr-general-wdn-x4v3": Path("models/onnx/realesr-general-wdn-x4v3.onnx"),
    "RealESRGAN_x2plus": Path("models/onnx/RealESRGAN_x2plus.onnx"),
    "RealESRGAN_x4plus_anime_6B": Path(
        "models/onnx/RealESRGAN_x4plus_anime_6B.onnx"
    ),
    "RealESRGAN_x4plus_fp16": Path("models/onnx/RealESRGAN_x4plus_fp16.onnx"),
    "SwinIR_x4_classical": Path("models/onnx/SwinIR_x4_classical.onnx"),
    "SwinIR_x4_realSR_GAN": Path("models/onnx/SwinIR_x4_realSR_GAN.onnx"),
    "4x-UltraSharp_fp16": Path("models/onnx/4x-UltraSharp_fp16.onnx"),
    "SwinIR_colorDN25": Path("models/onnx/SwinIR_colorDN25.onnx"),
}

_MODEL_DESCRIPTIONS = {
    "realesr-general-x4v3": ("真实/3D·CG，SRVGG 轻量通用模型，适度降噪（快）"),
    "RealESRGAN_x2plus": ("真实，RRDB 2x 模型，适合轻度放大（2 倍），无降噪"),
    "RealESRGAN_x4plus_anime_6B": ("2D 动漫，RRDB 模型，边缘自然，无降噪"),
    "RealESRGAN_x4plus_fp16": ("真实，RRDB 重模型，高锐度细节，无降噪"),
    "SwinIR_x4_classical": (
        "真实，SwinIR-M 经典超分（双三次退化），干净图保真度最高，无降噪（较慢）"
    ),
    "SwinIR_x4_realSR_GAN": (
        "真实，SwinIR-M 真实世界超分（BSRGAN 退化），纹理自然，无降噪（较慢）"
    ),
    "4x-UltraSharp_fp16": (
        "真实/3D·CG，社区 RRDB 锐利模型（fp16），细节锐度最强，无降噪"
    ),
    "realesr-general-wdn-x4v3": (
        "降噪模型（x4 同倍率），与超分输出按强度插值，强降噪"
    ),
    "SwinIR_colorDN25": (
        "降噪模型（1x 预降噪），SwinIR 彩色降噪 σ=25，在输入尺度先降噪"
    ),
}

_SCENE_BY_NAME = {
    "realesr-general-x4v3": Scene.REALISTIC,
    "RealESRGAN_x2plus": Scene.REALISTIC,
    "RealESRGAN_x4plus_anime_6B": Scene.ANIME_2D,
    "RealESRGAN_x4plus_fp16": Scene.REALISTIC,
    "SwinIR_x4_classical": Scene.REALISTIC,
    "SwinIR_x4_realSR_GAN": Scene.REALISTIC,
    "4x-UltraSharp_fp16": Scene.REALISTIC,
    "realesr-general-wdn-x4v3": Scene.REALISTIC,
    "SwinIR_colorDN25": Scene.REALISTIC,
}

_MODEL_KINDS = {
    "realesr-general-wdn-x4v3": "denoise",
    "SwinIR_colorDN25": "denoise",
}

_SCENE_DEFAULT_NAMES = {
    Scene.REALISTIC: "realesr-general-x4v3",
    Scene.ANIME_2D: "RealESRGAN_x4plus_anime_6B",
    Scene.CG_3D: "realesr-general-x4v3",
}

_CUSTOM_REGISTRY_NAME = "custom_models.json"


def _build_model(name: str) -> ModelMeta:
    """Build one registry entry from the static tables above."""
    return ModelMeta(
        name=name,
        scale=_MODEL_SCALES[name],
        scene=_SCENE_BY_NAME[name],
        backend=Backend.DIRECTML,
        model_path=_MODEL_PATHS[name],
        input_name="input",
        output_name="output",
        scale_input=None,
        denoise_input=None,
        channel_order="RGB",
        normalize=(0.0, 1.0),
        tile_size=512,
        description=_MODEL_DESCRIPTIONS[name],
        kind=_MODEL_KINDS.get(name, "sr"),
    )


def builtin_models() -> list[ModelMeta]:
    """Return the built-in super-resolution model registry."""
    return [_build_model(name) for name in _SR_MODEL_NAMES]


def denoise_models() -> list[ModelMeta]:
    """Return standalone denoise models, attachable to any SR model."""
    return [_build_model(name) for name in _DENOISE_MODEL_NAMES]


def scene_default(scene: Scene) -> ModelMeta:
    """Map a scene preset to its default model."""
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene enum value")

    name = _SCENE_DEFAULT_NAMES.get(scene)
    if name is None:
        raise ValueError(f"No default model registered for scene: {scene.value}")

    for model in builtin_models():
        if model.name == name:
            return model
    raise RuntimeError(f"Default model {name!r} is missing from the registry")


def custom_registry_path(models_dir: Optional[str | Path] = None) -> Path:
    """Return the JSON registry that stores user-imported models."""
    if models_dir is None:
        return app_root() / "models" / _CUSTOM_REGISTRY_NAME
    return Path(models_dir) / _CUSTOM_REGISTRY_NAME


def _portable_model_path(path: Path) -> Path:
    path = Path(path)
    try:
        return path.resolve().relative_to(app_root().resolve())
    except (OSError, ValueError):
        return path


def _model_meta_to_payload(meta: ModelMeta) -> dict[str, Any]:
    return {
        "name": meta.name,
        "scale": meta.scale,
        "scene": meta.scene.value,
        "backend": meta.backend.value,
        "model_path": str(_portable_model_path(Path(meta.model_path))),
        "denoise_pair_path": (
            str(_portable_model_path(Path(meta.denoise_pair_path)))
            if meta.denoise_pair_path is not None
            else None
        ),
        "denoise_pair_scale": meta.denoise_pair_scale,
        "kind": meta.kind,
        "input_name": meta.input_name,
        "output_name": meta.output_name,
        "scale_input": meta.scale_input,
        "denoise_input": meta.denoise_input,
        "channel_order": meta.channel_order,
        "normalize": list(meta.normalize),
        "tile_size": meta.tile_size,
        "description": meta.description,
    }


def _model_meta_from_payload(payload: dict[str, Any]) -> ModelMeta:
    return ModelMeta(
        name=str(payload.get("name") or ""),
        scale=_parse_scale(payload.get("scale")),
        scene=_parse_scene(payload.get("scene")),
        backend=_parse_backend(payload.get("backend")),
        model_path=Path(str(payload.get("model_path") or "")),
        denoise_pair_path=(
            Path(str(payload["denoise_pair_path"]))
            if payload.get("denoise_pair_path")
            else None
        ),
        denoise_pair_scale=_parse_scale(payload.get("denoise_pair_scale")),
        kind=str(payload.get("kind") or "sr"),
        input_name=str(payload.get("input_name") or "input"),
        output_name=str(payload.get("output_name") or "output"),
        scale_input=payload.get("scale_input"),
        denoise_input=payload.get("denoise_input"),
        channel_order=str(payload.get("channel_order") or "RGB"),
        normalize=_parse_normalize(payload.get("normalize")),
        tile_size=int(payload.get("tile_size") or 512),
        description=str(payload.get("description") or ""),
    )


def registered_custom_models(
    registry_path: Optional[str | Path] = None,
) -> list[ModelMeta]:
    """Return user-imported models from the local JSON registry."""
    path = Path(registry_path) if registry_path is not None else custom_registry_path()
    if not path.is_file():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    items = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []

    models: list[ModelMeta] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        try:
            models.append(_model_meta_from_payload(item))
        except Exception:
            continue
    return models


def register_custom_model(
    meta: ModelMeta,
    registry_path: Optional[str | Path] = None,
) -> Path:
    """Persist a custom model descriptor, replacing the same model name."""
    path = Path(registry_path) if registry_path is not None else custom_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [
        model
        for model in registered_custom_models(path)
        if model.name != meta.name
    ]
    existing.append(meta)
    payload = {"models": [_model_meta_to_payload(model) for model in existing]}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _unique_library_target(directory: Path, source: Path) -> Path:
    target = directory / source.name
    if not target.exists():
        return target
    for index in range(1, 10000):
        candidate = directory / f"{source.stem} ({index}){source.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate a unique model file name")


def _resolve_model_path(model: ModelMeta) -> Path:
    """Resolve relative paths against the project root where possible."""
    path = Path(model.model_path)
    if path.is_absolute():
        return path

    root_candidate = app_root() / path
    if root_candidate.exists():
        return root_candidate
    return path


def load_model(meta: ModelMeta) -> OnnxEngine:
    """Instantiate an ONNX engine for ``meta.backend`` and load its model."""
    engine = OnnxEngine(meta.backend)
    load_meta = replace(meta, model_path=_resolve_model_path(meta))
    engine.load_model(load_meta)
    return engine


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model metadata JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Model metadata must be a JSON object: {path}")
    return payload


def _parse_scene(value: Any) -> Scene:
    if isinstance(value, Scene):
        return value
    if value is None:
        return Scene.REALISTIC
    return Scene(str(value).strip().lower())


def _parse_backend(value: Any) -> Backend:
    if isinstance(value, Backend):
        return value
    if value is None:
        return Backend.DIRECTML
    return Backend(str(value).strip().lower())


def _parse_scale(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _parse_normalize(value: Any) -> tuple[float, float]:
    if value is None:
        return (0.0, 1.0)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("normalize must be a [mean, std] pair")
    return (float(value[0]), float(value[1]))


def _onnx_input_shape(meta: Any) -> Optional[list[Optional[int]]]:
    try:
        tensor_type = meta.type.tensor_type
    except AttributeError:
        return None

    if not tensor_type.HasField("shape"):
        return None

    shape: list[Optional[int]] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(None)
        else:
            shape.append(None)
    return shape


def _looks_like_scale_name(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in {"scale", "scales", "scale_factor", "scale_ratio"}:
        return True
    return "scale" in lowered


def _detect_scale_input(onnx_path: Path) -> Optional[str]:
    """Return a likely dynamic scale input name from an ONNX graph.

    This is intentionally best-effort: named scale inputs take precedence, and
    otherwise a non-image scalar/vector input is used. Model loaders can still
    override this with an explicit ``scale_input`` metadata field.
    """
    try:
        import onnx
    except ImportError:
        return None

    try:
        graph_model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception:
        return None

    graph_inputs = list(graph_model.graph.input)
    for graph_input in graph_inputs:
        if _looks_like_scale_name(graph_input.name):
            return graph_input.name

    for graph_input in graph_inputs:
        if graph_input.name.lower() == "input":
            continue
        shape = _onnx_input_shape(graph_input)
        if shape is not None and len(shape) <= 1:
            return graph_input.name
    return None


def import_custom_model(
    onnx_path: str | Path,
    meta_json_path: str | Path,
) -> ModelMeta:
    """Import a custom ONNX model and its optional metadata file."""
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model file does not exist: {onnx_path}")

    data = _load_json(Path(meta_json_path))
    model_path_value = data.get("model_path")
    model_path = Path(model_path_value) if model_path_value else onnx_path

    scale_input = data.get("scale_input")
    if scale_input is None:
        scale_input = _detect_scale_input(onnx_path)

    return ModelMeta(
        name=str(data.get("name") or onnx_path.stem),
        scale=_parse_scale(data.get("scale")),
        scene=_parse_scene(data.get("scene")),
        backend=_parse_backend(data.get("backend")),
        model_path=model_path,
        input_name=str(data.get("input_name") or "input"),
        output_name=str(data.get("output_name") or "output"),
        scale_input=scale_input,
        denoise_input=data.get("denoise_input"),
        denoise_pair_scale=_parse_scale(data.get("denoise_pair_scale")),
        kind=str(data.get("kind") or "sr"),
        channel_order=str(data.get("channel_order") or "RGB"),
        normalize=_parse_normalize(data.get("normalize")),
        tile_size=int(data.get("tile_size") or 512),
        description=str(data.get("description") or ""),
    )


def import_custom_model_to_library(
    onnx_path: str | Path,
    meta_json_path: Optional[str | Path] = None,
    models_dir: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
) -> ModelMeta:
    """Copy an ONNX model into the library, register it, and return metadata."""
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model file does not exist: {source}")
    if source.suffix.lower() != ".onnx":
        raise ValueError("Only .onnx model files can be imported")

    library_dir = (
        Path(models_dir) if models_dir is not None else app_root() / "models" / "onnx"
    )
    library_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve().parent == library_dir.resolve():
        target = source
    else:
        target = _unique_library_target(library_dir, source)
        target.write_bytes(source.read_bytes())

    metadata_path: Optional[Path] = None
    if meta_json_path is not None:
        metadata_path = Path(meta_json_path)
    else:
        sibling = source.with_suffix(".json")
        if sibling.is_file():
            metadata_path = sibling

    if metadata_path is not None and metadata_path.is_file():
        meta = import_custom_model(target, metadata_path)
        meta = replace(meta, model_path=target)
    else:
        scale_input = _detect_scale_input(target)
        meta = ModelMeta(
            name=target.stem,
            scale=None,
            scene=Scene.REALISTIC,
            backend=Backend.DIRECTML,
            model_path=target,
            input_name="input",
            output_name="output",
            scale_input=scale_input,
            denoise_input=None,
            channel_order="RGB",
            normalize=(0.0, 1.0),
            tile_size=512,
            description="User-imported ONNX super-resolution model",
        )

    if not meta.name:
        meta = replace(meta, name=target.stem)
    register_custom_model(meta, registry_path)
    return meta


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    url: str,
    dest: str | Path,
    sha256: Optional[str] = None,
) -> Path:
    """Download a model file, optionally verifying its SHA-256 digest."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected = sha256.lower() if sha256 else None
    if expected and dest.is_file() and _file_sha256(dest) == expected:
        return dest

    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)

        if expected and digest.hexdigest() != expected:
            raise ValueError(
                f"SHA-256 mismatch for {dest.name}: "
                f"expected {expected}, got {digest.hexdigest()}"
            )

        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return dest
