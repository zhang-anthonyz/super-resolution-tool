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
from ..models.model_manager import import_custom_model_to_library
from .ui_logic import (
    SCENE_OPTIONS,
    BackendChoice,
    all_denoise_options,
    all_model_options,
    build_model_meta,
    copy_model_to_library,
    default_model_name,
    fixed_scale_for_model,
    model_for_name,
    model_scales,
    recommended_backend_index,
    resolve_project_path,
    scene_label,
    supported_model_formats,
)
from .widgets import ChevronComboBox


def is_denoise_model(model: Optional[ModelMeta]) -> bool:
    """Return whether this registry entry is a standalone denoise model."""
    return model is not None and model.kind == "denoise"


def _field_label(text: str) -> QLabel:
    """Form label styled by the shared fieldLabel theme rule."""
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


def _model_supports_vulkan(name: str) -> bool:
    """ncnn 仅内置 x4plus / anime 系列模型（无 x2plus / general）。"""
    lowered = name.lower()
    return "x4plus" in lowered or "anime" in lowered


class ModelSelector(QWidget):
    """Independent model selection controls used by image and video tabs."""

    model_changed = Signal(object)
    denoise_strength_changed = Signal(float)
    sharpen_strength_changed = Signal(float)

    def __init__(
        self,
        backend_options: Sequence[BackendChoice],
        parent: Optional[QWidget] = None,
        scale_selector: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._scale_selector = scale_selector
        self._syncing_scene = False
        self._suppress_custom_path_clear = False
        # 垂直方向固定，避免被 Tab 中 stretch 控件（文件列表/日志）压缩到不可交互
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(12)

        self.scene_combo = ChevronComboBox()
        self.scene_combo.setMinimumHeight(28)
        for scene, label in SCENE_OPTIONS:
            self.scene_combo.addItem(label, scene)

        self.model_combo = ChevronComboBox()
        self.model_combo.setMinimumHeight(28)
        # Long registry names would otherwise force the whole page wider than
        # the viewport; elide the field and let the popup show full text.
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.model_combo.setMinimumContentsLength(16)
        self.model_combo.view().setMinimumWidth(560)
        self._populate_model_combo()

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_browse_button = QPushButton("浏览")
        self.model_browse_button.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_path_edit, 1)
        model_layout.addWidget(self.model_browse_button)

        self.backend_combo = ChevronComboBox()
        self.backend_combo.setMinimumHeight(28)
        self._populate_backend_combo(backend_options)

        self.denoise_combo = ChevronComboBox()
        self.denoise_combo.setMinimumHeight(28)
        # 说明文案很长，不能让下拉框把整个页面撑宽（字段内省略、弹层显示全文）
        self.denoise_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.denoise_combo.setMinimumContentsLength(18)
        self.denoise_combo.view().setMinimumWidth(560)
        self.denoise_combo.setToolTip(
            "选择降噪模型：1x 模型在输入尺度预降噪，与主模型同倍率的模型则与输出插值；"
            "选“无”时使用内置的边缘保持降噪"
        )
        self.denoise_combo.addItem("无（内置边缘保持降噪）", None)
        for name, display in all_denoise_options():
            self.denoise_combo.addItem(f"{name} · {display}", name)

        self.denoise_row = QWidget()
        denoise_layout = QHBoxLayout(self.denoise_row)
        denoise_layout.setContentsMargins(0, 0, 0, 0)
        denoise_layout.setSpacing(8)
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setMinimumHeight(24)
        # 默认不降噪：内置滤波同样以"细节优先"为原则，但默认保持原始细节。
        self.denoise_slider.setValue(0)
        self.denoise_slider.setTickInterval(10)
        self.denoise_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.denoise_value_label = QLabel("0.00")
        self.denoise_value_label.setObjectName("valueBadge")
        self.denoise_value_label.setMinimumWidth(48)
        self.denoise_value_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        denoise_layout.addWidget(self.denoise_slider, 1)
        denoise_layout.addWidget(self.denoise_value_label)
        self.denoise_row.setVisible(False)

        self.sharpen_row = QWidget()
        sharpen_layout = QHBoxLayout(self.sharpen_row)
        sharpen_layout.setContentsMargins(0, 0, 0, 0)
        sharpen_layout.setSpacing(8)
        self.sharpen_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpen_slider.setRange(0, 100)
        self.sharpen_slider.setMinimumHeight(24)
        self.sharpen_slider.setValue(25)
        self.sharpen_slider.setTickInterval(10)
        self.sharpen_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sharpen_value_label = QLabel("0.25")
        self.sharpen_value_label.setObjectName("valueBadge")
        self.sharpen_value_label.setMinimumWidth(48)
        self.sharpen_value_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        sharpen_layout.addWidget(self.sharpen_slider, 1)
        sharpen_layout.addWidget(self.sharpen_value_label)
        self.sharpen_row.setVisible(True)

        self.model_info_label = QLabel()
        self.model_info_label.setObjectName("mutedLabel")
        self.model_info_label.setWordWrap(True)

        form.addRow(_field_label("场景预设"), self.scene_combo)
        form.addRow(_field_label("超分模型"), self.model_combo)
        form.addRow(_field_label("模型文件"), model_row)
        form.addRow(_field_label("推理后端"), self.backend_combo)
        form.addRow(_field_label("降噪模型"), self.denoise_combo)
        form.addRow(_field_label("降噪强度"), self.denoise_row)
        form.addRow(_field_label("边缘锐化"), self.sharpen_row)
        form.addRow(_field_label(""), self.model_info_label)
        layout.addLayout(form)

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.denoise_slider.valueChanged.connect(self._on_denoise_changed)
        self.sharpen_slider.valueChanged.connect(self._on_sharpen_changed)

        self._on_backend_changed()
        self._on_model_changed()

    def refresh_models(self) -> None:
        """Reload built-in and custom model options after imports."""
        current_name = self.selected_model_name()
        current_path = self.model_path_edit.text().strip()
        self._populate_model_combo()
        if current_name:
            index = self.model_combo.findData(current_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        if current_path:
            self.model_path_edit.setText(current_path)

    def _populate_model_combo(self) -> None:
        current = self.model_combo.currentData() if hasattr(self, "model_combo") else None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for name, display in all_model_options():
            self.model_combo.addItem(f"{name} · {display}", name)
        self.model_combo.addItem("自定义模型", None)
        if current is not None:
            index = self.model_combo.findData(current)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

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
            self._sync_scene_for_model(model)
        elif not self._suppress_custom_path_clear:
            self.model_path_edit.clear()

        self.denoise_row.setVisible(True)
        self.sharpen_row.setVisible(True)
        self._update_model_info()
        self.model_changed.emit(model)

    def _sync_scene_for_model(self, model: ModelMeta) -> None:
        index = self.scene_combo.findData(model.scene)
        if index < 0 or index == self.scene_combo.currentIndex():
            return
        self._syncing_scene = True
        try:
            self.scene_combo.setCurrentIndex(index)
        finally:
            self._syncing_scene = False

    def _on_scene_changed(self) -> None:
        if self._syncing_scene:
            return
        scene = self._current_scene_value()
        if scene is None:
            return
        default_name = default_model_name(scene)
        index = self.model_combo.findData(default_name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def _current_scene_value(self) -> Optional[Scene]:
        value = self.scene_combo.currentData()
        if isinstance(value, Scene):
            return value
        try:
            return Scene(str(value))
        except ValueError:
            return None

    def _sync_scale_selector(self, model: ModelMeta) -> None:
        if self._scale_selector is None:
            return
        set_scales = getattr(self._scale_selector, "set_scales", None)
        if callable(set_scales):
            set_scales(model_scales(model))

    def _on_backend_changed(self) -> None:
        """Vulkan 后端禁用 ncnn 无对应模型（x2plus / general），并自动切到可用项。"""
        backend = self.current_backend()
        first_enabled = -1
        for i in range(self.model_combo.count()):
            name = self.model_combo.itemData(i)
            if name is None:
                continue
            item = self.model_combo.model().item(i)
            supported = backend != Backend.VULKAN or _model_supports_vulkan(str(name))
            if supported:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                if first_enabled < 0:
                    first_enabled = i
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        current = self.model_combo.currentIndex()
        current_item = self.model_combo.model().item(current)
        if current_item is not None and not (current_item.flags() & Qt.ItemFlag.ItemIsEnabled):
            if first_enabled >= 0:
                self.model_combo.setCurrentIndex(first_enabled)
        self._update_model_info()

    def _on_denoise_changed(self, value: int) -> None:
        strength = value / 100.0
        self.denoise_value_label.setText(f"{strength:.2f}")
        self.denoise_strength_changed.emit(strength)

    def _on_sharpen_changed(self, value: int) -> None:
        strength = value / 100.0
        self.sharpen_value_label.setText(f"{strength:.2f}")
        self.sharpen_strength_changed.emit(strength)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型",
            str(app_root() / "models" / "onnx"),
            supported_model_formats(),
        )
        if not path:
            return

        selected_path = Path(path)
        library_dir = app_root() / "models" / "onnx"
        if selected_path.resolve().parent != library_dir.resolve():
            selected_path = copy_model_to_library(selected_path)

        custom_index = self.model_combo.findData(None)
        if custom_index >= 0:
            self._suppress_custom_path_clear = True
            try:
                self.model_combo.setCurrentIndex(custom_index)
            finally:
                self._suppress_custom_path_clear = False
        self.model_path_edit.setText(str(selected_path))
        self._update_model_info()

    def import_and_register_model(self) -> Optional[ModelMeta]:
        """Import an ONNX model into the persistent registry and select it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 ONNX 模型",
            str(app_root() / "models" / "onnx"),
            "ONNX 模型 (*.onnx)",
        )
        if not path:
            return None
        return self.import_and_register_model_from_path(path)

    def import_and_register_model_from_path(self, path: str | Path) -> ModelMeta:
        """Import a known ONNX path into the persistent registry and select it."""
        meta = import_custom_model_to_library(path)
        self.refresh_models()
        index = self.model_combo.findData(meta.name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        return meta

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

    def set_backend(self, backend: Backend) -> None:
        index = self.backend_combo.findData(backend)
        if index < 0:
            return
        item = self.backend_combo.model().item(index)
        if item is not None and not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        self.backend_combo.setCurrentIndex(index)

    def denoise_model_name(self) -> Optional[str]:
        """Return the selected denoise model name (None = built-in filter only)."""
        value = self.denoise_combo.currentData()
        return str(value) if value else None

    def set_denoise_model(self, name: Optional[str]) -> None:
        index = self.denoise_combo.findData(name) if name else 0
        if index >= 0:
            self.denoise_combo.setCurrentIndex(index)

    def current_denoise_model(self) -> Optional[ModelMeta]:
        name = self.denoise_model_name()
        if name is None:
            return None
        return model_for_name(name)

    def set_denoise_strength(self, strength: float) -> None:
        self.denoise_slider.setValue(int(round(max(0.0, min(1.0, strength)) * 100)))

    def set_sharpen_strength(self, strength: float) -> None:
        self.sharpen_slider.setValue(int(round(max(0.0, min(1.0, strength)) * 100)))

    def current_scene(self) -> Scene:
        scene = self._current_scene_value()
        if scene is not None:
            return scene
        model = self.selected_registry_model()
        return model.scene if model is not None else Scene.REALISTIC

    def current_model_path(self) -> Path:
        text = self.model_path_edit.text().strip()
        return resolve_project_path(text) if text else app_root()

    def current_model(self) -> ModelMeta:
        denoise = self.current_denoise_model()
        denoise_path = (
            resolve_project_path(denoise.model_path) if denoise is not None else None
        )
        denoise_scale = denoise.scale if denoise is not None else None
        registry_model = self.selected_registry_model()
        if registry_model is not None:
            path = resolve_project_path(registry_model.model_path)
            return build_model_meta(
                path,
                registry_model.scene,
                self.current_backend(),
                registry_model.scale,
                denoise_pair_path=denoise_path or registry_model.denoise_pair_path,
                denoise_pair_scale=denoise_scale,
            )

        path = self.current_model_path()
        return build_model_meta(
            path,
            self.current_scene(),
            self.current_backend(),
            fixed_scale_for_model(path),
            denoise_pair_path=denoise_path,
            denoise_pair_scale=denoise_scale,
        )

    def denoise_strength(self) -> float:
        return self.denoise_slider.value() / 100.0

    def sharpen_strength(self) -> float:
        return self.sharpen_slider.value() / 100.0

    def _update_model_info(self) -> None:
        model = self.selected_registry_model()
        description = model.description if model is not None else "自定义 ONNX 模型"
        runtime_model = self.current_model()
        if not self.model_path_edit.text().strip():
            self.model_info_label.setText("当前模型：未选择模型文件")
            return
        scale_text = f"{runtime_model.scale}x" if runtime_model.scale else "自动"
        self.model_info_label.setText(
            f"当前模型：{runtime_model.name} / 固定倍率：{scale_text} / {description}"
        )
