"""Main PySide6 window for the super-resolution tool."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.backend_selector import BackendSelector
from ..core.environment_checker import available_backends
from ..core.types import (
    Backend,
    GpuInfo,
    ModelMeta,
    ProgressInfo,
    Scene,
    SuperResResult,
    VideoCodec,
)
from . import ui_logic
from .ui_logic import BackendChoice
from .workers import ImageUpscaleWorker, VideoUpscaleWorker


VIDEO_CODEC_OPTIONS: tuple[tuple[VideoCodec, str], ...] = (
    (VideoCodec.H264, "H.264"),
    (VideoCodec.HEVC, "HEVC"),
)


class ScaleSelector(QWidget):
    """Reusable 2x/3x/4x/custom scale control."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.combo = QComboBox()
        self.combo.addItem("2x", 2)
        self.combo.addItem("3x", 3)
        self.combo.addItem("4x", 4)
        self.combo.addItem("自定义", None)
        self.combo.setCurrentIndex(2)

        self.custom_spin = QSpinBox()
        self.custom_spin.setRange(2, 8)
        self.custom_spin.setValue(4)
        self.custom_spin.setEnabled(False)

        layout.addWidget(self.combo, 1)
        layout.addWidget(self.custom_spin)
        self.combo.currentIndexChanged.connect(self._sync_custom_state)

    def _sync_custom_state(self) -> None:
        self.custom_spin.setEnabled(self.combo.currentData() is None)

    def set_scales(self, scales: Sequence[int]) -> None:
        """Replace the selector options with model-specific scales."""
        self.combo.clear()
        for scale in scales:
            self.combo.addItem(f"{scale}x", int(scale))
        if len(scales) > 1:
            self.combo.addItem("自定义", None)
        preferred_index = next(
            (index for index in range(self.combo.count()) if self.combo.itemData(index) == 4),
            0,
        )
        self.combo.setCurrentIndex(preferred_index)
        self._sync_custom_state()

    def scale(self) -> int:
        selected = self.combo.currentData()
        return self.custom_spin.value() if selected is None else int(selected)


class LogPanel(QWidget):
    """Timestamped, color-coded, copyable log panel."""

    LEVEL_COLORS = {
        "Info": QColor("#9ca3af"),
        "Warning": QColor("#fbbf24"),
        "Error": QColor("#f87171"),
        "Success": QColor("#4ade80"),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(5000)
        self.text_edit.setPlaceholderText("运行日志会显示在这里")

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        copy_button = QPushButton("复制日志")
        clear_button = QPushButton("清空")
        copy_button.clicked.connect(self.copy_to_clipboard)
        clear_button.clicked.connect(self.clear)
        button_row.addWidget(copy_button)
        button_row.addWidget(clear_button)

        layout.addWidget(self.text_edit)
        layout.addLayout(button_row)

    def append(self, level: str, message: str) -> None:
        text = ui_logic.format_log_line(level, message) + "\n"
        cursor = QTextCursor(self.text_edit.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, self._format_for_level(level))
        self.text_edit.ensureCursorVisible()

    def clear(self) -> None:
        self.text_edit.clear()

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.text_edit.toPlainText())

    @classmethod
    def _format_for_level(cls, level: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(cls.LEVEL_COLORS.get(level, QColor("#e5e7eb")))
        return fmt


class MainWindow(QMainWindow):
    """Top-level application window with image, video, and settings tabs."""

    def __init__(self, gpu: GpuInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.gpu = gpu
        self.backend_selector = BackendSelector(gpu)
        self.available = available_backends(gpu)
        self.backend_options: list[BackendChoice] = ui_logic.backend_choices(
            gpu,
            self.available,
        )
        self._image_worker: Optional[ImageUpscaleWorker] = None
        self._video_worker: Optional[VideoUpscaleWorker] = None

        self.setWindowTitle("Super Resolution Tool")
        self.resize(1120, 820)
        self.setMinimumSize(860, 640)
        self._build_ui()
        self._connect_signals()
        self._refresh_model_for_scene()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        self.gpu_label = QLabel(ui_logic.gpu_summary(self.gpu, self.available))
        self.gpu_label.setObjectName("gpuInfoLabel")
        self.gpu_label.setWordWrap(True)
        root_layout.addWidget(self.gpu_label)

        self.tabs = QTabWidget()
        self.settings_tab = self._build_settings_tab()
        self.image_tab = self._build_image_tab()
        self.video_tab = self._build_video_tab()
        self.tabs.addTab(self.image_tab, "图片超分")
        self.tabs.addTab(self.video_tab, "视频超分")
        self.tabs.addTab(self.settings_tab, "设置")
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        gpu_group = QGroupBox("检测到的显卡")
        gpu_layout = QVBoxLayout(gpu_group)
        self.gpu_detail_label = QLabel(ui_logic.gpu_summary(self.gpu, self.available))
        self.gpu_detail_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_detail_label)
        layout.addWidget(gpu_group)

        config_group = QGroupBox("超分配置")
        form = QFormLayout(config_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.model_combo = QComboBox()
        for name, display in ui_logic.model_options():
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
        self._populate_backend_combo()

        self.model_info_label = QLabel()
        self.model_info_label.setObjectName("mutedLabel")

        form.addRow("超分模型", self.model_combo)
        form.addRow("模型文件", model_row)
        form.addRow("推理后端", self.backend_combo)
        form.addRow("", self.model_info_label)
        layout.addWidget(config_group)
        layout.addStretch(1)
        return page

    def _build_image_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        input_group = QGroupBox("输入图片")
        input_layout = QVBoxLayout(input_group)
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        input_buttons = QHBoxLayout()
        self.image_add_files_button = QPushButton("添加文件")
        self.image_add_folder_button = QPushButton("添加文件夹")
        self.image_remove_button = QPushButton("移除选中")
        self.image_add_files_button.clicked.connect(self._add_image_files)
        self.image_add_folder_button.clicked.connect(self._add_image_folder)
        self.image_remove_button.clicked.connect(self._remove_selected_images)
        input_buttons.addWidget(self.image_add_files_button)
        input_buttons.addWidget(self.image_add_folder_button)
        input_buttons.addWidget(self.image_remove_button)
        input_buttons.addStretch(1)
        input_layout.addWidget(self.image_list, 1)
        input_layout.addLayout(input_buttons)
        layout.addWidget(input_group, 2)

        output_group = QGroupBox("输出与倍率")
        output_form = QFormLayout(output_group)
        output_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.image_output_dir_edit = QLineEdit(str(ui_logic.app_root() / "output"))
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.image_output_dir_edit, 1)
        self.image_output_button = QPushButton("浏览")
        self.image_output_button.clicked.connect(self._browse_image_output)
        output_layout.addWidget(self.image_output_button)
        self.image_scale_selector = ScaleSelector()
        self.image_overwrite_check = QCheckBox("覆盖已存在文件")
        output_form.addRow("输出目录", output_row)
        output_form.addRow("倍率", self.image_scale_selector)
        output_form.addRow("", self.image_overwrite_check)
        layout.addWidget(output_group)

        run_group = QGroupBox("任务")
        run_layout = QVBoxLayout(run_group)
        self.image_progress = QProgressBar()
        self.image_progress.setRange(0, 0)
        self.image_status = QLabel("就绪")
        self.image_status.setObjectName("mutedLabel")
        controls = QHBoxLayout()
        self.image_start_button = QPushButton("开始")
        self.image_start_button.setObjectName("primaryButton")
        self.image_cancel_button = QPushButton("取消")
        self.image_cancel_button.setEnabled(False)
        controls.addWidget(self.image_start_button)
        controls.addWidget(self.image_cancel_button)
        controls.addStretch(1)
        run_layout.addWidget(self.image_progress)
        run_layout.addWidget(self.image_status)
        run_layout.addLayout(controls)
        layout.addWidget(run_group)

        self.image_log = LogPanel()
        layout.addWidget(self.image_log, 2)
        return page

    def _build_video_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        input_group = QGroupBox("输入视频")
        input_form = QFormLayout(input_group)
        input_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.video_input_edit = QLineEdit()
        video_input_row = QWidget()
        video_input_layout = QHBoxLayout(video_input_row)
        video_input_layout.setContentsMargins(0, 0, 0, 0)
        video_input_layout.addWidget(self.video_input_edit, 1)
        self.video_input_button = QPushButton("浏览")
        self.video_input_button.clicked.connect(self._browse_video_input)
        video_input_layout.addWidget(self.video_input_button)

        self.video_output_dir_edit = QLineEdit(str(ui_logic.app_root() / "output"))
        video_output_row = QWidget()
        video_output_layout = QHBoxLayout(video_output_row)
        video_output_layout.setContentsMargins(0, 0, 0, 0)
        video_output_layout.addWidget(self.video_output_dir_edit, 1)
        self.video_output_button = QPushButton("浏览")
        self.video_output_button.clicked.connect(self._browse_video_output)
        video_output_layout.addWidget(self.video_output_button)

        input_form.addRow("视频文件", video_input_row)
        input_form.addRow("输出目录", video_output_row)

        self.video_8bit_label = QLabel(ui_logic.video_8bit_notice())
        self.video_8bit_label.setObjectName("mutedLabel")
        self.video_8bit_label.setWordWrap(True)
        input_form.addRow("", self.video_8bit_label)
        layout.addWidget(input_group)

        output_group = QGroupBox("输出与倍率")
        output_form = QFormLayout(output_group)
        output_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.video_codec_combo = QComboBox()
        for codec, label in VIDEO_CODEC_OPTIONS:
            self.video_codec_combo.addItem(label, codec)
        self.video_scale_selector = ScaleSelector()
        self.video_overwrite_check = QCheckBox("覆盖已存在文件")
        output_form.addRow("编码器", self.video_codec_combo)
        output_form.addRow("倍率", self.video_scale_selector)
        output_form.addRow("", self.video_overwrite_check)
        layout.addWidget(output_group)

        run_group = QGroupBox("任务")
        run_layout = QVBoxLayout(run_group)
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 0)
        self.video_status = QLabel("就绪")
        self.video_status.setObjectName("mutedLabel")
        controls = QHBoxLayout()
        self.video_start_button = QPushButton("开始")
        self.video_start_button.setObjectName("primaryButton")
        self.video_cancel_button = QPushButton("取消")
        self.video_cancel_button.setEnabled(False)
        controls.addWidget(self.video_start_button)
        controls.addWidget(self.video_cancel_button)
        controls.addStretch(1)
        run_layout.addWidget(self.video_progress)
        run_layout.addWidget(self.video_status)
        run_layout.addLayout(controls)
        layout.addWidget(run_group)

        self.video_log = LogPanel()
        layout.addWidget(self.video_log, 2)
        return page

    def _connect_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_model_for_scene)
        self.image_start_button.clicked.connect(self.start_image_upscale)
        self.image_cancel_button.clicked.connect(self.cancel_image_upscale)
        self.video_start_button.clicked.connect(self.start_video_upscale)
        self.video_cancel_button.clicked.connect(self.cancel_video_upscale)

    def _populate_backend_combo(self) -> None:
        self.backend_combo.clear()
        for choice in self.backend_options:
            self.backend_combo.addItem(choice.display_label, choice.backend)
            if not choice.available:
                index = self.backend_combo.count() - 1
                item = self.backend_combo.model().item(index)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        recommended = ui_logic.recommended_backend_index(self.backend_options)
        self.backend_combo.setCurrentIndex(recommended)

    def _refresh_model_for_scene(self, *_args: object) -> None:
        model = self._selected_registry_model()
        if model is not None:
            path = ui_logic.resolve_project_path(model.model_path)
            self.model_path_edit.setText(str(path))
            scales = ui_logic.model_scales(model)
            self.image_scale_selector.set_scales(scales)
            self.video_scale_selector.set_scales(scales)
        self._update_model_info()

    def _update_model_info(self) -> None:
        model = self.current_model()
        registry_model = self._selected_registry_model()
        description = (
            registry_model.description
            if registry_model is not None
            else model.description
        )
        scale_text = f"{model.scale}x" if model.scale else "自动"
        self.model_info_label.setText(
            f"当前模型：{model.name} / 固定倍率：{scale_text} / {description}"
        )

    def _selected_model_name(self) -> Optional[str]:
        value = self.model_combo.currentData()
        return str(value) if value else None

    def _selected_registry_model(self) -> Optional[ModelMeta]:
        name = self._selected_model_name()
        if name is None:
            return None
        return ui_logic.model_for_name(name)

    def current_model(self) -> ModelMeta:
        registry_model = self._selected_registry_model()
        if registry_model is not None:
            path = ui_logic.resolve_project_path(registry_model.model_path)
            return ui_logic.build_model_meta(
                path,
                registry_model.scene,
                self.current_backend(),
                registry_model.scale,
            )

        path = self._current_model_path()
        return ui_logic.build_model_meta(
            path,
            self.current_scene(),
            self.current_backend(),
            ui_logic.fixed_scale_for_model(path),
        )

    def current_scene(self) -> Scene:
        registry_model = self._selected_registry_model()
        return registry_model.scene if registry_model is not None else Scene.REALISTIC

    def current_backend(self) -> Backend:
        backend = self.backend_combo.currentData()
        if isinstance(backend, Backend):
            return backend
        try:
            return Backend(str(backend).lower())
        except (TypeError, ValueError):
            return self.backend_selector.default_backend()

    def current_video_codec(self) -> VideoCodec:
        codec = self.video_codec_combo.currentData()
        if isinstance(codec, VideoCodec):
            return codec
        try:
            return VideoCodec(str(codec).lower())
        except (TypeError, ValueError):
            return VideoCodec.H264

    def _current_model_path(self) -> Path:
        text = self.model_path_edit.text().strip()
        return ui_logic.resolve_project_path(text) if text else ui_logic.app_root()

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型",
            str(ui_logic.app_root() / "models" / "onnx"),
            ui_logic.supported_model_formats(),
        )
        if path:
            library_dir = ui_logic.app_root() / "models" / "onnx"
            selected_path = Path(path)
            if selected_path.resolve().parent != library_dir.resolve():
                selected_path = ui_logic.copy_model_to_library(selected_path)
            self.model_path_edit.setText(str(selected_path))
            custom_index = self.model_combo.findData(None)
            if custom_index >= 0:
                self.model_combo.setCurrentIndex(custom_index)
            self._update_model_info()

    def _add_image_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.webp);;所有文件 (*)",
        )
        if paths:
            self._add_image_paths(paths)

    def _add_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self._add_image_paths([folder])

    def _add_image_paths(self, paths: Sequence[str]) -> None:
        files = ui_logic.collect_image_files(paths)
        existing = self._image_path_set()
        for path in files:
            if path in existing:
                continue
            item = QListWidgetItem(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.image_list.addItem(item)
            existing.add(path)
        if not files and paths:
            self.image_log.append("Warning", "未找到 jpg/png/webp 图片")
        self.image_status.setText(f"已添加 {self.image_list.count()} 张图片")

    def _remove_selected_images(self) -> None:
        for item in self.image_list.selectedItems():
            self.image_list.takeItem(self.image_list.row(item))
        self.image_status.setText(f"已添加 {self.image_list.count()} 张图片")

    def _image_path_set(self) -> set[Path]:
        return {
            Path(self.image_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.image_list.count())
        }

    def _image_paths(self) -> list[Path]:
        return [
            Path(self.image_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.image_list.count())
        ]

    def _browse_image_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片输出目录")
        if folder:
            self.image_output_dir_edit.setText(folder)

    def _browse_video_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;所有文件 (*)",
        )
        if path:
            self.video_input_edit.setText(path)

    def _browse_video_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择视频输出目录")
        if folder:
            self.video_output_dir_edit.setText(folder)

    def start_image_upscale(self) -> None:
        if self._image_worker is not None and self._image_worker.isRunning():
            return
        paths = self._image_paths()
        if not paths:
            QMessageBox.warning(self, "缺少输入", "请先添加至少一张图片")
            return
        out_dir_text = self.image_output_dir_edit.text().strip()
        if not out_dir_text:
            QMessageBox.warning(self, "缺少输出目录", "请选择图片输出目录")
            return

        model_path = self._current_model_path()
        if not model_path.is_file():
            QMessageBox.warning(self, "模型不存在", f"未找到模型文件：\n{model_path}")
            return

        requested_scale = self.image_scale_selector.scale()
        scale, scale_changed = ui_logic.effective_scale(model_path, requested_scale)
        model = self.current_model()
        backend = self.current_backend()
        out_dir = ui_logic.resolve_project_path(out_dir_text)

        self.image_log.append(
            "Info",
            f"开始处理 {len(paths)} 张图片：后端 {ui_logic.backend_label(backend, self.gpu)}",
        )
        if scale_changed:
            self.image_log.append(
                "Warning",
                f"当前模型固定 {scale}x，已忽略所选 {requested_scale}x",
            )

        worker = ImageUpscaleWorker(
            paths=paths,
            out_dir=out_dir,
            model=model,
            scale=scale,
            backend=backend,
            overwrite=self.image_overwrite_check.isChecked(),
            parent=self,
        )
        worker.progress.connect(self._on_image_progress)
        worker.finished.connect(self._on_image_finished)
        worker.error.connect(self._on_image_error)
        worker.cancelled.connect(self._on_image_cancelled)
        self._image_worker = worker
        self._set_image_running(True)
        worker.start()

    def cancel_image_upscale(self) -> None:
        if self._image_worker is not None:
            self._image_worker.cancel()
            self.image_cancel_button.setEnabled(False)
            self.image_status.setText("正在取消...")

    def start_video_upscale(self) -> None:
        if self._video_worker is not None and self._video_worker.isRunning():
            return
        input_text = self.video_input_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "缺少输入", "请选择一条视频文件")
            return
        input_path = ui_logic.resolve_project_path(input_text)
        if not input_path.is_file():
            QMessageBox.warning(self, "文件不存在", f"未找到视频文件：\n{input_path}")
            return
        out_dir_text = self.video_output_dir_edit.text().strip()
        if not out_dir_text:
            QMessageBox.warning(self, "缺少输出目录", "请选择视频输出目录")
            return
        out_dir = ui_logic.resolve_project_path(out_dir_text)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = self._current_model_path()
        if not model_path.is_file():
            QMessageBox.warning(self, "模型不存在", f"未找到模型文件：\n{model_path}")
            return

        requested_scale = self.video_scale_selector.scale()
        scale, scale_changed = ui_logic.effective_scale(model_path, requested_scale)
        codec = self.current_video_codec()
        backend = self.current_backend()
        out_path = out_dir / f"{input_path.stem}_x{scale}.mp4"

        if out_path.exists() and not self.video_overwrite_check.isChecked():
            QMessageBox.warning(
                self,
                "输出已存在",
                f"输出文件已存在，请勾选覆盖或更换目录：\n{out_path}",
            )
            return

        model = self.current_model()
        self.video_log.append(
            "Info",
            f"开始处理视频：{input_path.name} / 后端 "
            f"{ui_logic.backend_label(backend, self.gpu)} / {codec.value}",
        )
        if scale_changed:
            self.video_log.append(
                "Warning",
                f"当前模型固定 {scale}x，已忽略所选 {requested_scale}x",
            )

        worker = VideoUpscaleWorker(
            input_path=input_path,
            out_path=out_path,
            model=model,
            scale=scale,
            backend=backend,
            codec=codec,
            parent=self,
        )
        worker.progress.connect(self._on_video_progress)
        worker.finished.connect(self._on_video_finished)
        worker.error.connect(self._on_video_error)
        worker.cancelled.connect(self._on_video_cancelled)
        self._video_worker = worker
        self._set_video_running(True)
        worker.start()

    def cancel_video_upscale(self) -> None:
        if self._video_worker is not None:
            self._video_worker.cancel()
            self.video_cancel_button.setEnabled(False)
            self.video_status.setText("正在取消...")

    def _on_image_progress(self, info: ProgressInfo) -> None:
        if info.total > 0:
            self.image_progress.setRange(0, info.total)
            self.image_progress.setValue(info.current)
        else:
            self.image_progress.setRange(0, 0)
        self.image_status.setText(info.message)

    def _on_video_progress(self, info: ProgressInfo) -> None:
        if info.total > 0:
            self.video_progress.setRange(0, info.total)
            self.video_progress.setValue(info.current)
        else:
            self.video_progress.setRange(0, 0)
        self.video_status.setText(info.message)

    def _on_image_finished(self, results: object) -> None:
        self._set_image_running(False)
        results = list(results or [])
        ok_count = sum(1 for result in results if getattr(result, "ok", False))
        if results:
            self.image_progress.setRange(0, len(results))
            self.image_progress.setValue(len(results))
        self.image_status.setText(f"完成：成功 {ok_count}/{len(results)}")
        self.image_log.append("Success", f"处理完成，成功 {ok_count}/{len(results)}")
        for result in results:
            if not getattr(result, "ok", False):
                output = getattr(result, "output_path", None)
                message = getattr(result, "error", None) or "未知错误"
                self.image_log.append("Error", f"{output or '图片'}：{message}")
        self._cleanup_image_worker()

    def _on_image_error(self, message: str) -> None:
        self._set_image_running(False)
        self.image_status.setText("处理失败")
        self.image_log.append("Error", message)
        self._cleanup_image_worker()

    def _on_image_cancelled(self) -> None:
        self._set_image_running(False)
        self.image_status.setText("已取消")
        self.image_log.append("Warning", "任务已取消")
        self._cleanup_image_worker()

    def _on_video_finished(self, result: SuperResResult) -> None:
        self._set_video_running(False)
        self.video_progress.setRange(0, 1)
        self.video_progress.setValue(1)
        self.video_status.setText("完成")
        self.video_log.append(
            "Success",
            f"视频处理完成：{getattr(result, 'output_path', '')}",
        )
        self._cleanup_video_worker()

    def _on_video_error(self, message: str) -> None:
        self._set_video_running(False)
        self.video_status.setText("处理失败")
        self.video_log.append("Error", message)
        self._cleanup_video_worker()

    def _on_video_cancelled(self) -> None:
        self._set_video_running(False)
        self.video_status.setText("已取消")
        self.video_log.append("Warning", "任务已取消")
        self._cleanup_video_worker()

    def _set_image_running(self, running: bool) -> None:
        self.image_start_button.setEnabled(not running)
        self.image_cancel_button.setEnabled(running)

    def _set_video_running(self, running: bool) -> None:
        self.video_start_button.setEnabled(not running)
        self.video_cancel_button.setEnabled(running)

    def _cleanup_image_worker(self) -> None:
        worker = self._image_worker
        self._image_worker = None
        if worker is not None:
            worker.deleteLater()

    def _cleanup_video_worker(self) -> None:
        worker = self._video_worker
        self._video_worker = None
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event) -> None:
        for worker in (self._image_worker, self._video_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                if not worker.wait(3000):
                    event.ignore()
                    return
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #10151f;
                color: #e5e7eb;
                font-family: "Segoe UI", "Microsoft YaHei UI";
                font-size: 13px;
            }
            QLabel#gpuInfoLabel {
                background-color: #132232;
                border: 1px solid #155e75;
                border-radius: 6px;
                padding: 10px 12px;
                color: #a5f3fc;
                font-weight: 600;
            }
            QLabel#mutedLabel {
                color: #9ca3af;
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #263242;
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                background: #151c28;
                color: #cbd5e1;
                padding: 9px 18px;
                border: 1px solid #263242;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #0f766e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #263242;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #7dd3fc;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
                background-color: #151c28;
                border: 1px solid #2b3949;
                border-radius: 5px;
                padding: 5px 7px;
                color: #e5e7eb;
                selection-background-color: #0f766e;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
            QPlainTextEdit:focus, QListWidget:focus {
                border-color: #22d3ee;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 6px 12px;
                color: #e5e7eb;
            }
            QPushButton:hover {
                background-color: #27364a;
                border-color: #475569;
            }
            QPushButton:pressed {
                background-color: #16212e;
            }
            QPushButton:disabled {
                background-color: #151c28;
                color: #64748b;
                border-color: #263242;
            }
            QPushButton#primaryButton {
                background-color: #0f766e;
                border-color: #14b8a6;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover {
                background-color: #0d9488;
            }
            QProgressBar {
                background-color: #151c28;
                border: 1px solid #2b3949;
                border-radius: 5px;
                text-align: center;
                color: #e5e7eb;
            }
            QProgressBar::chunk {
                background-color: #14b8a6;
                border-radius: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #475569;
                border-radius: 3px;
                background-color: #151c28;
            }
            QCheckBox::indicator:checked {
                background-color: #14b8a6;
                border-color: #14b8a6;
            }
            """
        )
