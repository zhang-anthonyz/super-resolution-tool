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

_DEFAULT_MODEL_PATH = Path("models/onnx/RealESRGAN_x4plus.onnx")
_SCENE_DEFAULT_NAMES = {
    Scene.REALISTIC: "RealESRGAN_x4plus",
}


def builtin_models() -> list[ModelMeta]:
    """Return the V1 built-in model registry.

    The registry intentionally stays as a function returning fresh dataclass
    instances. New anime/general/SRMD entries should be added here later.
    """
    return [
        ModelMeta(
            name="RealESRGAN_x4plus",
            scale=4,
            scene=Scene.REALISTIC,
            backend=Backend.DIRECTML,
            model_path=_DEFAULT_MODEL_PATH,
            input_name="input",
            output_name="output",
            scale_input=None,
            denoise_input=None,
            channel_order="RGB",
            normalize=(0.0, 1.0),
            tile_size=512,
            description="Real-ESRGAN general 4x super-resolution model",
        )
    ]


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
        channel_order=str(data.get("channel_order") or "RGB"),
        normalize=_parse_normalize(data.get("normalize")),
        tile_size=int(data.get("tile_size") or 512),
        description=str(data.get("description") or ""),
    )


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
