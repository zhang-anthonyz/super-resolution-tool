"""Main PySide6 window for the super-resolution tool."""
from __future__ import annotations

import time
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
    GpuInfo,
    ProgressInfo,
    SuperResResult,
    VideoCodec,
)
from ..pipelines.ffmpeg_utils import probe_video
from . import ui_logic
from . import theme
from .ui_logic import BackendChoice
from .model_selector import ModelSelector
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

    LEVEL_COLORS = {key: QColor(value) for key, value in theme.LOG_LEVEL_COLORS.items()}

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
        self._image_progress_key: Optional[tuple[int, str]] = None
        self._image_progress_last_time = 0.0
        self._image_progress_last_percent = -1.0
        self._video_progress_last_time = 0.0
        self._video_progress_last_frame = -1

        self.setWindowTitle("Super Resolution Tool")
        self.resize(1120, 820)
        self.setMinimumSize(860, 640)
        self._build_ui()
        self._connect_signals()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

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

        import_group = QGroupBox("自定义模型导入")
        import_layout = QVBoxLayout(import_group)
        self.import_model_button = QPushButton("导入 ONNX 模型到模型库")
        self.import_model_button.clicked.connect(self._import_custom_model)
        self.import_model_status = QLabel("可导入自定义 ONNX 模型，随后在各任务 Tab 中选择“自定义模型”")
        self.import_model_status.setObjectName("mutedLabel")
        self.import_model_status.setWordWrap(True)
        import_layout.addWidget(self.import_model_button)
        import_layout.addWidget(self.import_model_status)
        layout.addWidget(import_group)
        layout.addStretch(1)
        return page

    def _build_image_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

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

        self.image_scale_selector = ScaleSelector()
        model_group = QGroupBox("超分模型")
        model_layout = QVBoxLayout(model_group)
        self.image_model_selector = ModelSelector(
            self.backend_options,
            parent=page,
            scale_selector=self.image_scale_selector,
        )
        model_layout.addWidget(self.image_model_selector)
        layout.addWidget(model_group)

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
        self.image_overwrite_check = QCheckBox("覆盖已存在文件")
        output_form.addRow("输出目录", output_row)
        output_form.addRow("倍率", self.image_scale_selector)
        output_form.addRow("", self.image_overwrite_check)
        layout.addWidget(output_group)

        run_group = QGroupBox("任务")
        run_layout = QVBoxLayout(run_group)
        self.image_progress = QProgressBar()
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
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
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

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

        self.video_scale_selector = ScaleSelector()
        model_group = QGroupBox("超分模型")
        model_layout = QVBoxLayout(model_group)
        self.video_model_selector = ModelSelector(
            self.backend_options,
            parent=page,
            scale_selector=self.video_scale_selector,
        )
        model_layout.addWidget(self.video_model_selector)
        layout.addWidget(model_group)

        output_group = QGroupBox("输出与倍率")
        output_form = QFormLayout(output_group)
        output_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.video_codec_combo = QComboBox()
        for codec, label in VIDEO_CODEC_OPTIONS:
            self.video_codec_combo.addItem(label, codec)
        self.video_overwrite_check = QCheckBox("覆盖已存在文件")
        output_form.addRow("编码器", self.video_codec_combo)
        output_form.addRow("倍率", self.video_scale_selector)
        output_form.addRow("", self.video_overwrite_check)
        layout.addWidget(output_group)

        run_group = QGroupBox("任务")
        run_layout = QVBoxLayout(run_group)
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
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
        self.image_start_button.clicked.connect(self.start_image_upscale)
        self.image_cancel_button.clicked.connect(self.cancel_image_upscale)
        self.video_start_button.clicked.connect(self.start_video_upscale)
        self.video_cancel_button.clicked.connect(self.cancel_video_upscale)

    def current_video_codec(self) -> VideoCodec:
        codec = self.video_codec_combo.currentData()
        if isinstance(codec, VideoCodec):
            return codec
        try:
            return VideoCodec(str(codec).lower())
        except (TypeError, ValueError):
            return VideoCodec.H264

    def _import_custom_model(self) -> None:
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
            self.import_model_status.setText(f"已导入：{selected_path}")

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

    def _reset_image_progress(self) -> None:
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)

    def _reset_video_progress(self) -> None:
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)

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

        model_path = self.image_model_selector.current_model_path()
        if not model_path.is_file():
            QMessageBox.warning(self, "模型不存在", f"未找到模型文件：\n{model_path}")
            return

        requested_scale = self.image_scale_selector.scale()
        scale, scale_changed = ui_logic.effective_scale(model_path, requested_scale)
        model = self.image_model_selector.current_model()
        backend = self.image_model_selector.current_backend()
        denoise_strength = self.image_model_selector.denoise_strength()
        out_dir = ui_logic.resolve_project_path(out_dir_text)
        self._image_progress_key = None
        self._image_progress_last_time = 0.0
        self._image_progress_last_percent = -1.0
        self._reset_image_progress()

        self.image_log.append(
            "Info",
            f"检测到显卡：{ui_logic.gpu_summary(self.gpu, self.available)}",
        )
        denoise_text = (
            f"{denoise_strength:.2f}"
            if getattr(model, "denoise_pair_path", None)
            else "无"
        )
        self.image_log.append(
            "Info",
            f"加载模型：{model.name} / {ui_logic.backend_label(backend, self.gpu)} / "
            f"{scale}x / 降噪强度 {denoise_text}",
        )
        self.image_log.append(
            "Info",
            f"开始处理 {len(paths)} 张图片：输出目录 {out_dir}",
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
            denoise_strength=denoise_strength,
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

        model_path = self.video_model_selector.current_model_path()
        if not model_path.is_file():
            QMessageBox.warning(self, "模型不存在", f"未找到模型文件：\n{model_path}")
            return

        requested_scale = self.video_scale_selector.scale()
        scale, scale_changed = ui_logic.effective_scale(model_path, requested_scale)
        codec = self.current_video_codec()
        backend = self.video_model_selector.current_backend()
        denoise_strength = self.video_model_selector.denoise_strength()
        out_path = out_dir / f"{input_path.stem}_x{scale}.mp4"

        if out_path.exists() and not self.video_overwrite_check.isChecked():
            QMessageBox.warning(
                self,
                "输出已存在",
                f"输出文件已存在，请勾选覆盖或更换目录：\n{out_path}",
            )
            return

        model = self.video_model_selector.current_model()
        self._video_progress_last_time = 0.0
        self._video_progress_last_frame = -1
        self._reset_video_progress()
        self.video_log.append(
            "Info",
            f"检测到显卡：{ui_logic.gpu_summary(self.gpu, self.available)}",
        )
        denoise_text = (
            f"{denoise_strength:.2f}"
            if getattr(model, "denoise_pair_path", None)
            else "无"
        )
        self.video_log.append(
            "Info",
            f"加载模型：{model.name} / {ui_logic.backend_label(backend, self.gpu)} / "
            f"{scale}x / 降噪强度 {denoise_text}",
        )
        try:
            video_info = probe_video(input_path)
            video_detail = (
                f"{video_info['width']}x{video_info['height']} / "
                f"{video_info['fps']:.2f} fps / "
                f"{video_info['total_frames']} 帧 / {codec.value}"
            )
        except Exception:
            video_detail = "无法读取视频信息"
        self.video_log.append(
            "Info",
            f"视频信息：{input_path.name} / {video_detail}",
        )
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
            denoise_strength=denoise_strength,
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
            self.image_progress.setValue(ui_logic.progress_bar_value(info.current))
        else:
            self.image_progress.setRange(0, 100)
            self.image_progress.setValue(0)
        self.image_status.setText(info.message)
        self._log_image_progress(info)

    def _on_video_progress(self, info: ProgressInfo) -> None:
        if info.total > 0:
            self.video_progress.setRange(0, info.total)
            self.video_progress.setValue(ui_logic.progress_bar_value(info.current))
        else:
            self.video_progress.setRange(0, 100)
            self.video_progress.setValue(0)
        self.video_status.setText(info.message)
        self._log_video_progress(info)

    def _log_image_progress(self, info: ProgressInfo) -> None:
        source_name = info.source_name
        image_key = None
        if info.image_index is not None and source_name is not None:
            image_key = (int(info.image_index), source_name)

        if image_key is not None and image_key != self._image_progress_key:
            self._image_progress_key = image_key
            self._image_progress_last_percent = -1.0
            self.image_log.append(
                "Info",
                f"处理 {source_name}（{info.percent:.1f}%）",
            )

        now = time.monotonic()
        should_log_tile = (
            info.phase == "tile"
            and info.message
            and (
                info.percent >= self._image_progress_last_percent + 10.0
                or now - self._image_progress_last_time >= 1.0
                or info.percent >= 99.999
            )
        )
        if should_log_tile:
            self._image_progress_last_time = now
            self._image_progress_last_percent = info.percent
            self.image_log.append("Info", info.message)

    def _log_video_progress(self, info: ProgressInfo) -> None:
        now = time.monotonic()
        current = int(info.current)
        should_log = (
            info.message
            and (
                current % 10 == 0
                or current == info.total
                or now - self._video_progress_last_time >= 1.0
            )
        )
        if not should_log:
            return
        self._video_progress_last_time = now
        self._video_progress_last_frame = current
        self.video_log.append("Info", info.message)

    def _on_image_finished(self, results: object) -> None:
        self._set_image_running(False)
        results = list(results or [])
        ok_count = sum(1 for result in results if getattr(result, "ok", False))
        fail_count = len(results) - ok_count
        total_elapsed = sum(
            float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
            for result in results
        )
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(100 if results else 0)
        self.image_status.setText(
            f"完成：成功 {ok_count} / 失败 {fail_count}"
        )
        for result in results:
            elapsed = float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
            if getattr(result, "ok", False):
                output = getattr(result, "output_path", None)
                self.image_log.append(
                    "Info",
                    f"保存到 {output}（耗时 {elapsed:.1f}s）",
                )
            else:
                source = getattr(result, "input_path", None)
                output = getattr(result, "output_path", None)
                message = getattr(result, "error", None) or "未知错误"
                self.image_log.append(
                    "Error",
                    f"{source or output or '图片'}：{message}",
                )
        self.image_log.append(
            "Success",
            f"完成统计：成功 {ok_count} / 失败 {fail_count} / "
            f"总耗时 {total_elapsed:.1f}s",
        )
        self._cleanup_image_worker()

    def _on_image_error(self, message: str) -> None:
        self._set_image_running(False)
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        self.image_status.setText("处理失败")
        self.image_log.append("Error", message)
        self._cleanup_image_worker()

    def _on_image_cancelled(self) -> None:
        self._set_image_running(False)
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        self.image_status.setText("已取消")
        self.image_log.append("Warning", "任务已取消")
        self._cleanup_image_worker()

    def _on_video_finished(self, result: SuperResResult) -> None:
        self._set_video_running(False)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(100)
        self.video_status.setText("完成")
        elapsed = float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
        self.video_log.append(
            "Success",
            f"视频处理完成：{getattr(result, 'output_path', '')} "
            f"（耗时 {elapsed:.1f}s）",
        )
        self._cleanup_video_worker()

    def _on_video_error(self, message: str) -> None:
        self._set_video_running(False)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_status.setText("处理失败")
        self.video_log.append("Error", message)
        self._cleanup_video_worker()

    def _on_video_cancelled(self) -> None:
        self._set_video_running(False)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
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
        self.setStyleSheet(theme.APP_STYLESHEET)
