"""Reusable model, backend, and denoise-strength selector for task tabs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.paths import app_root
from ..core.types import Backend, ModelMeta, Scene
from .ui_logic import (
    BackendChoice,
    build_model_meta,
    copy_model_to_library,
    fixed_scale_for_model,
    model_for_name,
    model_options,
    model_scales,
    recommended_backend_index,
    resolve_project_path,
    supported_model_formats,
)


def is_denoise_model(model: Optional[ModelMeta]) -> bool:
    """Return whether a model exposes an output-interpolation denoise pair."""
    return model is not None and model.denoise_pair_path is not None


class ModelSelector(QWidget):
    """Independent model selection controls used by image and video tabs."""

    model_changed = Signal(object)
    denoise_strength_changed = Signal(float)

    def __init__(
        self,
        backend_options: Sequence[BackendChoice],
        parent: Optional[QWidget] = None,
        scale_selector: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._scale_selector = scale_selector
        # 垂直方向固定，避免被 Tab 中 stretch 控件（文件列表/日志）压缩到不可交互
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(12)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(28)
        for name, display in model_options():
            self.model_combo.addItem(f"{name} · {display}", name)
        self.model_combo.addItem("自定义模型", None)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_browse_button = QPushButton("浏览")
        self.model_browse_button.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_path_edit, 1)
        model_layout.addWidget(self.model_browse_button)

        self.backend_combo = QComboBox()
        self.backend_combo.setMinimumHeight(28)
        self._populate_backend_combo(backend_options)

        self.denoise_row = QWidget()
        denoise_layout = QHBoxLayout(self.denoise_row)
        denoise_layout.setContentsMargins(0, 0, 0, 0)
        denoise_layout.setSpacing(8)
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setMinimumHeight(22)
        self.denoise_slider.setValue(50)
        self.denoise_slider.setTickInterval(10)
        self.denoise_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.denoise_value_label = QLabel("0.50")
        self.denoise_value_label.setMinimumWidth(42)
        self.denoise_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        denoise_layout.addWidget(self.denoise_slider, 1)
        denoise_layout.addWidget(self.denoise_value_label)
        self.denoise_row.setVisible(False)

        self.model_info_label = QLabel()
        self.model_info_label.setObjectName("mutedLabel")
        self.model_info_label.setWordWrap(True)

        form.addRow("超分模型", self.model_combo)
        form.addRow("模型文件", model_row)
        form.addRow("推理后端", self.backend_combo)
        form.addRow("降噪强度", self.denoise_row)
        form.addRow("", self.model_info_label)
        layout.addLayout(form)

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.backend_combo.currentIndexChanged.connect(self._update_model_info)
        self.denoise_slider.valueChanged.connect(self._on_denoise_changed)

        self._on_model_changed()

    def set_scale_selector(self, scale_selector: Optional[QWidget]) -> None:
        self._scale_selector = scale_selector
        model = self.selected_registry_model()
        if model is not None:
            self._sync_scale_selector(model)

    def _populate_backend_combo(self, backend_options: Sequence[BackendChoice]) -> None:
        self.backend_combo.clear()
        for choice in backend_options:
            self.backend_combo.addItem(choice.display_label, choice.backend)
            if not choice.available:
                index = self.backend_combo.count() - 1
                item = self.backend_combo.model().item(index)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        self.backend_combo.setCurrentIndex(recommended_backend_index(backend_options))

    def _on_model_changed(self) -> None:
        model = self.selected_registry_model()
        if model is not None:
            self.model_path_edit.setText(str(resolve_project_path(model.model_path)))
            self._sync_scale_selector(model)

        has_denoise = is_denoise_model(model)
        self.denoise_row.setVisible(has_denoise)
        self._update_model_info()
        self.model_changed.emit(model)

    def _sync_scale_selector(self, model: ModelMeta) -> None:
        if self._scale_selector is None:
            return
        set_scales = getattr(self._scale_selector, "set_scales", None)
        if callable(set_scales):
            set_scales(model_scales(model))

    def _on_denoise_changed(self, value: int) -> None:
        strength = value / 100.0
        self.denoise_value_label.setText(f"{strength:.2f}")
        self.denoise_strength_changed.emit(strength)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型",
            str(app_root() / "models" / "onnx"),
            supported_model_formats(),
        )
        if not path:
            return

        library_dir = app_root() / "models" / "onnx"
        selected_path = Path(path)
        if selected_path.resolve().parent != library_dir.resolve():
            selected_path = copy_model_to_library(selected_path)
        self.model_path_edit.setText(str(selected_path))

        custom_index = self.model_combo.findData(None)
        if custom_index >= 0:
            self.model_combo.setCurrentIndex(custom_index)
        self._update_model_info()

    def selected_model_name(self) -> Optional[str]:
        value = self.model_combo.currentData()
        return str(value) if value else None

    def selected_registry_model(self) -> Optional[ModelMeta]:
        name = self.selected_model_name()
        if name is None:
            return None
        return model_for_name(name)

    def current_backend(self) -> Backend:
        backend = self.backend_combo.currentData()
        if isinstance(backend, Backend):
            return backend
        try:
            return Backend(str(backend).lower())
        except (TypeError, ValueError):
            return Backend.CPU

    def current_scene(self) -> Scene:
        model = self.selected_registry_model()
        return model.scene if model is not None else Scene.REALISTIC

    def current_model_path(self) -> Path:
        text = self.model_path_edit.text().strip()
        return resolve_project_path(text) if text else app_root()

    def current_model(self) -> ModelMeta:
        registry_model = self.selected_registry_model()
        if registry_model is not None:
            path = resolve_project_path(registry_model.model_path)
            return build_model_meta(
                path,
                registry_model.scene,
                self.current_backend(),
                registry_model.scale,
                denoise_pair_path=registry_model.denoise_pair_path,
            )

        path = self.current_model_path()
        return build_model_meta(
            path,
            self.current_scene(),
            self.current_backend(),
            fixed_scale_for_model(path),
        )

    def denoise_strength(self) -> float:
        return self.denoise_slider.value() / 100.0

    def _update_model_info(self) -> None:
        model = self.selected_registry_model()
        description = model.description if model is not None else "自定义 ONNX 模型"
        runtime_model = self.current_model()
        scale_text = f"{runtime_model.scale}x" if runtime_model.scale else "自动"
        self.model_info_label.setText(
            f"当前模型：{runtime_model.name} / 固定倍率：{scale_text} / {description}"
        )
