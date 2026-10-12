"""Main PySide6 window for the super-resolution tool."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
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
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.backend_selector import BackendSelector
from ..core.app_settings import load_settings, save_settings
from ..core.environment_checker import available_backends
from ..core.types import (
    Backend,
    GpuInfo,
    ProgressInfo,
    SuperResResult,
    VideoCodec,
)
from ..pipelines.ffmpeg_utils import probe_video
from ..pipelines.image_pipeline import (
    PERCENT_DEFAULT as _PERCENT_DEFAULT,
    PERCENT_MAX as _PERCENT_MAX,
    PERCENT_MIN as _PERCENT_MIN,
    RESIZE_MODE_PERCENT,
    resize_summary,
)
from . import ui_logic
from . import theme
from .ui_logic import BackendChoice
from .model_selector import ModelSelector
from .widgets import ChevronComboBox
from . import compare_view
from .workers import ImageUpscaleWorker, VideoUpscaleWorker


_PROGRESS_CHUNK_QSS = "QProgressBar::chunk { background-color: %s; border-radius: 5px; }"
_PROGRESS_COLOR_SUCCESS = theme.SUCCESS
_PROGRESS_COLOR_ERROR = theme.DANGER
_PROGRESS_COLOR_PARTIAL = theme.WARNING
_PROGRESS_COLOR_CANCELLED = theme.NEUTRAL

#: Height of the inline comparison area. The card sits at the end of
#: a scroll page next to a tall log panel, so it needs room of its own:
#: at less than this the block strip squeezes the image area flat.
_COMPARE_VIEW_MIN_HEIGHT = 560


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

        self.combo = ChevronComboBox()
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


class LogPanel(QFrame):
    """Timestamped, color-coded, copyable log card."""

    LEVEL_COLORS = {key: QColor(value) for key, value in theme.LOG_LEVEL_COLORS.items()}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("运行日志")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch(1)
        copy_button = QPushButton("复制")
        copy_button.setObjectName("ghostButton")
        clear_button = QPushButton("清空")
        clear_button.setObjectName("ghostButton")
        copy_button.clicked.connect(self.copy_to_clipboard)
        clear_button.clicked.connect(self.clear)
        header.addWidget(copy_button)
        header.addWidget(clear_button)
        layout.addLayout(header)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("logView")
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(5000)
        self.text_edit.setPlaceholderText("运行日志会显示在这里")
        layout.addWidget(self.text_edit, 1)

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
        fmt.setForeground(cls.LEVEL_COLORS.get(level, QColor(theme.TEXT_SECONDARY)))
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
        self._preview_worker: Optional[QThread] = None
        self._preview_token = 0
        self._compare_dialog: Optional[compare_view.ComparisonDialog] = None
        self._image_progress_key: Optional[tuple[int, str]] = None
        self._video_progress_last_time = 0.0
        self._video_progress_last_frame = -1
        self._settings = load_settings()

        self.setWindowTitle("Super Resolution Tool · 本地超分工作台")
        self.resize(1240, 840)
        self.setMinimumSize(1000, 680)
        self._build_ui()
        self._connect_signals()
        self._apply_style()
        self._apply_saved_settings()

    #: Page title/subtitle shown in the top header, indexed like the nav rail.
    PAGE_META: tuple[tuple[str, str], ...] = (
        ("图片超分", "单张或批量图片超分 · 支持 jpg / png / webp"),
        ("视频超分", "逐帧超分与硬件编码 · 仅支持 8bit SDR"),
        ("设置", "推理设备信息与自定义模型管理"),
    )

    #: Grouped navigation entries, in page order.
    NAV_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("工作台", ("图片超分", "视频超分")),
        ("系统", ("设置",)),
    )

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_content(), 1)
        self.setCentralWidget(root)

    # ------------------------------------------------------------- sidebar --
    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(236)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(16)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        mark = QLabel("SR")
        mark.setObjectName("brandMark")
        mark.setFixedSize(38, 38)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(1)
        brand_title = QLabel("Super Resolution")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("本地超分工作台")
        brand_subtitle.setObjectName("brandSubtitle")
        brand_text.addWidget(brand_title)
        brand_text.addWidget(brand_subtitle)
        brand.addWidget(mark)
        brand.addLayout(brand_text)
        brand.addStretch(1)
        layout.addLayout(brand)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self._nav_buttons: list[QPushButton] = []
        page_index = 0
        for section, titles in self.NAV_SECTIONS:
            section_label = QLabel(section)
            section_label.setObjectName("navSection")
            layout.addWidget(section_label)
            for title in titles:
                button = QPushButton(title)
                button.setObjectName("navButton")
                button.setCheckable(True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.setMinimumHeight(38)
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                self.nav_group.addButton(button, page_index)
                self._nav_buttons.append(button)
                layout.addWidget(button)
                page_index += 1
        self.nav_group.idClicked.connect(self._set_page)

        layout.addStretch(1)
        layout.addWidget(self._build_sidebar_footer())
        return sidebar

    def _build_sidebar_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("sidebarFooter")
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        for text in ("本地离线推理", "不上传任何图片 / 视频"):
            line = QLabel(text)
            line.setObjectName("sidebarFooterText")
            layout.addWidget(line)
        return footer

    # ------------------------------------------------------------- content --
    def _build_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("contentArea")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        layout.addWidget(self._build_header())

        self.stack = QStackedWidget()
        self.image_tab = self._build_image_tab()
        self.video_tab = self._build_video_tab()
        self.settings_tab = self._build_settings_tab()
        self.stack.addWidget(self.image_tab)
        self.stack.addWidget(self.video_tab)
        self.stack.addWidget(self.settings_tab)
        layout.addWidget(self.stack, 1)
        self._set_page(0)
        return content

    def _build_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("headerCard")
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(26)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(15, 23, 42, 24))
        card.setGraphicsEffect(shadow)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(20)

        text_column = QVBoxLayout()
        text_column.setSpacing(3)
        self.header_title = QLabel(self.PAGE_META[0][0])
        self.header_title.setObjectName("headerTitle")
        self.header_subtitle = QLabel(self.PAGE_META[0][1])
        self.header_subtitle.setObjectName("headerSubtitle")
        text_column.addWidget(self.header_title)
        text_column.addWidget(self.header_subtitle)
        layout.addLayout(text_column, 1)

        self.gpu_label = QLabel(ui_logic.gpu_summary(self.gpu, self.available))
        self.gpu_label.setObjectName("gpuInfoLabel")
        self.gpu_label.setWordWrap(True)
        self.gpu_label.setMaximumWidth(400)
        layout.addWidget(self.gpu_label, 0, Qt.AlignmentFlag.AlignVCenter)
        return card

    def _set_page(self, index: int) -> None:
        """Switch the visible page and keep header + nav rail in sync."""
        self.stack.setCurrentIndex(index)
        title, subtitle = self.PAGE_META[index]
        self.header_title.setText(title)
        self.header_subtitle.setText(subtitle)
        self._nav_buttons[index].setChecked(True)

    # --------------------------------------------------------- card helpers --
    def _make_scroll_page(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)

        page = QWidget()
        page.setObjectName("pageBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 2, 8, 2)
        layout.setSpacing(14)
        scroll.setWidget(page)
        return scroll, layout

    def _make_card(self, title: str, hint: str = "") -> tuple[QFrame, QVBoxLayout]:
        """Create a white card with a title row, divider and body layout."""
        card = QFrame()
        card.setObjectName("card")

        outer = QVBoxLayout(card)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(13)

        header = QHBoxLayout()
        header.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        header.addWidget(title_label)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("cardHint")
            header.addWidget(hint_label)
        header.addStretch(1)
        outer.addLayout(header)

        divider = QFrame()
        divider.setObjectName("cardDivider")
        divider.setFixedHeight(1)
        outer.addWidget(divider)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        outer.addLayout(body)
        return card, body

    def _make_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(16)
        form.setContentsMargins(0, 0, 0, 0)
        return form

    def _form_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _value_label(self, text: str, strong: bool = False) -> QLabel:
        label = QLabel(text)
        if strong:
            label.setObjectName("valueStrong")
        label.setWordWrap(True)
        return label

    def _build_settings_tab(self) -> QWidget:
        scroll, layout = self._make_scroll_page()

        device_card, device_body = self._make_card("推理设备", "启动时自动检测")
        device_form = self._make_form()
        device_form.addRow(
            self._form_label("显卡厂商"),
            self._value_label(ui_logic.VENDOR_LABELS.get(self.gpu.vendor, "未知厂商")),
        )
        device_form.addRow(
            self._form_label("显卡型号"),
            self._value_label(self.gpu.name, strong=True),
        )
        device_form.addRow(
            self._form_label("独立显存"),
            self._value_label(f"{self.gpu.dedicated_vram_gb:.1f} GB"),
        )
        device_form.addRow(
            self._form_label("可用后端"),
            self._value_label(self._available_backend_text(), strong=True),
        )
        device_body.addLayout(device_form)
        layout.addWidget(device_card)

        import_card, import_body = self._make_card(
            "自定义模型",
            "导入 ONNX 模型后可在此后各任务中选用",
        )
        self.import_model_button = QPushButton("导入 ONNX 模型到模型库")
        self.import_model_button.setObjectName("softButton")
        self.import_model_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_model_button.clicked.connect(self._import_custom_model)
        self.import_model_status = QLabel("尚未导入自定义模型")
        self.import_model_status.setObjectName("mutedLabel")
        self.import_model_status.setWordWrap(True)
        import_body.addWidget(self.import_model_button, 0, Qt.AlignmentFlag.AlignLeft)
        import_body.addWidget(self.import_model_status)
        layout.addWidget(import_card)

        about_card, about_body = self._make_card("关于")
        for text in (
            "推理后端：ONNX Runtime（CUDA / DirectML / OpenVINO）与 ncnn Vulkan",
            "视频编码：内置 ffmpeg，支持 H.264 / HEVC 硬件编码",
            "隐私：全部处理在本地完成，不联网、不上传素材",
        ):
            about_body.addWidget(self._value_label(text))
        layout.addWidget(about_card)
        layout.addStretch(1)
        return scroll

    def _available_backend_text(self) -> str:
        labels = [
            choice.display_label
            for choice in ui_logic.backend_choices(self.gpu, self.available)
            if choice.available
        ]
        return "、".join(labels) or "无可用后端"

    def _build_image_tab(self) -> QWidget:
        scroll, layout = self._make_scroll_page()

        input_card, input_body = self._make_card("输入图片", "支持 jpg / png / webp，可批量加入")
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.setMinimumHeight(150)
        input_buttons = QHBoxLayout()
        input_buttons.setSpacing(8)
        self.image_add_files_button = QPushButton("添加文件")
        self.image_add_files_button.setObjectName("softButton")
        self.image_add_folder_button = QPushButton("添加文件夹")
        self.image_add_folder_button.setObjectName("softButton")
        self.image_remove_button = QPushButton("移除选中")
        self.image_remove_button.setObjectName("ghostButton")
        self.image_add_files_button.clicked.connect(self._add_image_files)
        self.image_add_folder_button.clicked.connect(self._add_image_folder)
        self.image_remove_button.clicked.connect(self._remove_selected_images)
        input_buttons.addWidget(self.image_add_files_button)
        input_buttons.addWidget(self.image_add_folder_button)
        input_buttons.addWidget(self.image_remove_button)
        input_buttons.addStretch(1)
        for button in (
            self.image_add_files_button,
            self.image_add_folder_button,
            self.image_remove_button,
        ):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        input_body.addWidget(self.image_list, 1)
        input_body.addLayout(input_buttons)
        layout.addWidget(input_card)

        model_card, model_body = self._make_card(
            "模型与后端",
            "场景预设自动匹配模型；降噪模型可选独立降噪网络，留空则用内置边缘保持降噪",
        )
        self.image_scale_selector = ScaleSelector()
        self.image_model_selector = ModelSelector(
            self.backend_options,
            parent=model_card,
            scale_selector=self.image_scale_selector,
        )
        model_body.addWidget(self.image_model_selector)
        layout.addWidget(model_card)

        output_card, output_body = self._make_card("输出与画质")
        output_form = self._make_form()
        self.image_output_dir_edit = QLineEdit(str(ui_logic.app_root() / "output"))
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(8)
        output_layout.addWidget(self.image_output_dir_edit, 1)
        self.image_output_button = QPushButton("浏览")
        self.image_output_button.setObjectName("softButton")
        self.image_output_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_output_button.clicked.connect(self._browse_image_output)
        output_layout.addWidget(self.image_output_button)
        self.image_overwrite_check = QCheckBox("覆盖已存在文件")
        self.image_resize_mode_combo = ChevronComboBox()
        self.image_resize_mode_combo.addItem("保持超分尺寸", "keep")
        self.image_resize_mode_combo.addItem("回缩到原图尺寸", "restore")
        self.image_resize_mode_combo.addItem(
            "按百分比等比例缩放（相对超分结果）", "percent"
        )
        self.image_resize_percent_spin = QSpinBox()
        self.image_resize_percent_spin.setRange(
            _PERCENT_MIN, _PERCENT_MAX
        )
        self.image_resize_percent_spin.setSingleStep(5)
        self.image_resize_percent_spin.setValue(_PERCENT_DEFAULT)
        self.image_resize_percent_spin.setSuffix(" %")
        self.image_resize_percent_spin.setToolTip(
            f"按超分结果的百分比等比例缩放输出（{_PERCENT_MIN}–{_PERCENT_MAX}%，"
            "100% 等于保持超分尺寸）"
        )
        self.image_resize_percent_spin.setVisible(False)
        resize_row = QWidget()
        resize_layout = QHBoxLayout(resize_row)
        resize_layout.setContentsMargins(0, 0, 0, 0)
        resize_layout.setSpacing(8)
        resize_layout.addWidget(self.image_resize_mode_combo, 1)
        resize_layout.addWidget(self.image_resize_percent_spin)
        self.image_resize_mode_combo.currentIndexChanged.connect(
            self._sync_resize_percent_state
        )
        self.image_resize_percent_spin.valueChanged.connect(
            self._sync_resize_percent_state
        )
        self.image_compression_combo = ChevronComboBox()
        self.image_compression_combo.addItem("高质量", "high")
        self.image_compression_combo.addItem("高压缩", "compressed")
        self.image_compression_combo.addItem("无损 / 最高质量", "lossless")
        output_form.addRow(self._form_label("输出目录"), output_row)
        output_form.addRow(self._form_label("倍率"), self.image_scale_selector)
        output_form.addRow(self._form_label("输出尺寸"), resize_row)
        output_form.addRow(self._form_label("输出压缩"), self.image_compression_combo)
        output_form.addRow(self._form_label(""), self.image_overwrite_check)
        output_body.addLayout(output_form)
        layout.addWidget(output_card)

        run_card, run_body = self._make_card("任务", "运行中可随时取消")
        self.image_progress = QProgressBar()
        self.image_progress.setRange(0, 1000)
        self.image_progress.setValue(0)
        self.image_progress.setTextVisible(False)
        self.image_status = QLabel("就绪")
        self.image_status.setObjectName("statusLabel")
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.image_start_button = QPushButton("开始超分")
        self.image_start_button.setObjectName("primaryButton")
        self.image_start_button.setMinimumWidth(118)
        self.image_start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_cancel_button = QPushButton("取消")
        self.image_cancel_button.setObjectName("cancelButton")
        self.image_cancel_button.setEnabled(False)
        controls.addWidget(self.image_status, 1)
        controls.addWidget(self.image_cancel_button)
        controls.addWidget(self.image_start_button)
        run_body.addWidget(self.image_progress)
        run_body.addLayout(controls)
        layout.addWidget(run_card)

        self.image_log = LogPanel()
        self.image_log.setMinimumHeight(158)
        layout.addWidget(self.image_log)

        self.image_scroll = scroll
        self.image_compare_card, compare_body = self._make_card(
            "对比预览",
            "单张图片任务完成后自动载入：鼠标移动查看同一像素块，或拖动垂直滑块对比原图与超分结果",
        )
        self.image_compare_panel = compare_view.ComparisonPanel()
        # The card lives at the end of a scroll page, so it needs real height of
        # its own; the block strip and the wipe view both get room from it.
        self.image_compare_panel.view.setMinimumHeight(_COMPARE_VIEW_MIN_HEIGHT)
        self.image_compare_panel.add_action(
            self._compare_button("适应窗口", self.image_compare_view_fit)
        )
        self.image_compare_panel.add_action(
            self._compare_button("1:1 像素", self.image_compare_view_pixels)
        )
        self.image_compare_panel.add_action(
            self._compare_button("大窗口对比", self.open_image_compare_window)
        )
        self.image_compare_panel.add_action(
            self._compare_button("关闭预览", self.close_image_compare)
        )
        compare_body.addWidget(self.image_compare_panel, 1)
        compare_body.addWidget(
            self._value_label(
                "像素块对比：鼠标在图上移动，下方同步显示原图与输出同一区域；滚轮放大取样倍率\n"
                "垂直滑块对比：拖动分割线切换原图/输出，拖其它位置平移，滚轮缩放\n"
                "觉得太小就点「大窗口对比」：独立窗口可任意拉伸或最大化，尺寸会被记住",
            )
        )
        layout.addWidget(self.image_compare_card, 1)
        self.image_compare_card.setVisible(False)

        # The panel owns the widgets; keep these names for the rest of the window.
        self.image_compare_view = self.image_compare_panel.view
        self.image_compare_mode_combo = self.image_compare_panel.mode_combo
        self.image_compare_status = self.image_compare_panel.status
        return scroll

    def _build_video_tab(self) -> QWidget:
        scroll, layout = self._make_scroll_page()

        input_card, input_body = self._make_card("输入视频", ui_logic.video_8bit_notice())
        input_form = self._make_form()
        self.video_input_edit = QLineEdit()
        video_input_row = QWidget()
        video_input_layout = QHBoxLayout(video_input_row)
        video_input_layout.setContentsMargins(0, 0, 0, 0)
        video_input_layout.setSpacing(8)
        video_input_layout.addWidget(self.video_input_edit, 1)
        self.video_input_button = QPushButton("浏览")
        self.video_input_button.setObjectName("softButton")
        self.video_input_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.video_input_button.clicked.connect(self._browse_video_input)
        video_input_layout.addWidget(self.video_input_button)

        self.video_output_dir_edit = QLineEdit(str(ui_logic.app_root() / "output"))
        video_output_row = QWidget()
        video_output_layout = QHBoxLayout(video_output_row)
        video_output_layout.setContentsMargins(0, 0, 0, 0)
        video_output_layout.setSpacing(8)
        video_output_layout.addWidget(self.video_output_dir_edit, 1)
        self.video_output_button = QPushButton("浏览")
        self.video_output_button.setObjectName("softButton")
        self.video_output_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.video_output_button.clicked.connect(self._browse_video_output)
        video_output_layout.addWidget(self.video_output_button)

        input_form.addRow(self._form_label("视频文件"), video_input_row)
        input_form.addRow(self._form_label("输出目录"), video_output_row)
        input_body.addLayout(input_form)
        layout.addWidget(input_card)

        model_card, model_body = self._make_card(
            "模型与后端",
            "逐帧超分，模型与图片任务共用；降噪模型可独立选择",
        )
        self.video_scale_selector = ScaleSelector()
        self.video_model_selector = ModelSelector(
            self.backend_options,
            parent=model_card,
            scale_selector=self.video_scale_selector,
        )
        model_body.addWidget(self.video_model_selector)
        layout.addWidget(model_card)

        output_card, output_body = self._make_card("输出与编码")
        output_form = self._make_form()
        self.video_codec_combo = ChevronComboBox()
        for codec, label in VIDEO_CODEC_OPTIONS:
            self.video_codec_combo.addItem(label, codec)
        self.video_overwrite_check = QCheckBox("覆盖已存在文件")
        output_form.addRow(self._form_label("编码器"), self.video_codec_combo)
        output_form.addRow(self._form_label("倍率"), self.video_scale_selector)
        output_form.addRow(self._form_label(""), self.video_overwrite_check)
        output_body.addLayout(output_form)
        layout.addWidget(output_card)

        run_card, run_body = self._make_card("任务", "运行中可随时取消")
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 1000)
        self.video_progress.setValue(0)
        self.video_progress.setTextVisible(False)
        self.video_status = QLabel("就绪")
        self.video_status.setObjectName("statusLabel")
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.video_start_button = QPushButton("开始超分")
        self.video_start_button.setObjectName("primaryButton")
        self.video_start_button.setMinimumWidth(118)
        self.video_start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.video_cancel_button = QPushButton("取消")
        self.video_cancel_button.setObjectName("cancelButton")
        self.video_cancel_button.setEnabled(False)
        controls.addWidget(self.video_status, 1)
        controls.addWidget(self.video_cancel_button)
        controls.addWidget(self.video_start_button)
        run_body.addWidget(self.video_progress)
        run_body.addLayout(controls)
        layout.addWidget(run_card)

        self.video_log = LogPanel()
        self.video_log.setMinimumHeight(158)
        layout.addWidget(self.video_log)
        return scroll

    def _connect_signals(self) -> None:
        self.image_start_button.clicked.connect(self.start_image_upscale)
        self.image_cancel_button.clicked.connect(self.cancel_image_upscale)
        self.video_start_button.clicked.connect(self.start_video_upscale)
        self.video_cancel_button.clicked.connect(self.cancel_video_upscale)

    def _apply_saved_settings(self) -> None:
        image_output = self._settings.get("image_output_dir")
        if isinstance(image_output, str) and image_output:
            self.image_output_dir_edit.setText(image_output)
        video_output = self._settings.get("video_output_dir")
        if isinstance(video_output, str) and video_output:
            self.video_output_dir_edit.setText(video_output)
        self.image_overwrite_check.setChecked(
            bool(self._settings.get("image_overwrite", False))
        )
        self.video_overwrite_check.setChecked(
            bool(self._settings.get("video_overwrite", False))
        )
        image_resize_mode = str(self._settings.get("image_resize_mode") or "keep")
        resize_index = self.image_resize_mode_combo.findData(image_resize_mode)
        if resize_index >= 0:
            self.image_resize_mode_combo.setCurrentIndex(resize_index)
        try:
            percent = int(
                self._settings.get("image_resize_percent", _PERCENT_DEFAULT)
            )
        except (TypeError, ValueError):
            percent = _PERCENT_DEFAULT
        self.image_resize_percent_spin.setValue(
            max(_PERCENT_MIN, min(_PERCENT_MAX, percent))
        )
        self._sync_resize_percent_state()
        compression_profile = str(
            self._settings.get("image_compression_profile") or "high"
        )
        compression_index = self.image_compression_combo.findData(
            compression_profile
        )
        if compression_index >= 0:
            self.image_compression_combo.setCurrentIndex(compression_index)
        codec_value = str(self._settings.get("video_codec") or "")
        if codec_value:
            try:
                index = self.video_codec_combo.findData(VideoCodec(codec_value.lower()))
            except ValueError:
                index = -1
            if index >= 0:
                self.video_codec_combo.setCurrentIndex(index)

        try:
            image_backend = Backend(str(self._settings.get("image_backend") or ""))
            self.image_model_selector.set_backend(image_backend)
        except ValueError:
            pass
        try:
            video_backend = Backend(str(self._settings.get("video_backend") or ""))
            self.video_model_selector.set_backend(video_backend)
        except ValueError:
            pass
        try:
            image_denoise = float(self._settings.get("image_denoise_strength", 0.0))
            self.image_model_selector.set_denoise_strength(image_denoise)
            image_sharpen = float(self._settings.get("image_sharpen_strength", 0.25))
            self.image_model_selector.set_sharpen_strength(image_sharpen)
            video_denoise = float(self._settings.get("video_denoise_strength", 0.0))
            self.video_model_selector.set_denoise_strength(video_denoise)
            video_sharpen = float(self._settings.get("video_sharpen_strength", 0.25))
            self.video_model_selector.set_sharpen_strength(video_sharpen)
        except (TypeError, ValueError):
            pass

        image_denoise_model = self._settings.get("image_denoise_model")
        if isinstance(image_denoise_model, str) and image_denoise_model:
            self.image_model_selector.set_denoise_model(image_denoise_model)
        video_denoise_model = self._settings.get("video_denoise_model")
        if isinstance(video_denoise_model, str) and video_denoise_model:
            self.video_model_selector.set_denoise_model(video_denoise_model)

    def _window_settings(self) -> dict[str, object]:
        return {
            "image_output_dir": self.image_output_dir_edit.text().strip(),
            "video_output_dir": self.video_output_dir_edit.text().strip(),
            "image_overwrite": self.image_overwrite_check.isChecked(),
            "image_resize_mode": self.image_resize_mode_combo.currentData(),
            "image_resize_percent": self.image_resize_percent_spin.value(),
            "image_compression_profile": self.image_compression_combo.currentData(),
            "video_overwrite": self.video_overwrite_check.isChecked(),
            "video_codec": self.current_video_codec().value,
            "image_backend": self.image_model_selector.current_backend().value,
            "video_backend": self.video_model_selector.current_backend().value,
            "image_denoise_strength": self.image_model_selector.denoise_strength(),
            "image_sharpen_strength": self.image_model_selector.sharpen_strength(),
            "video_denoise_strength": self.video_model_selector.denoise_strength(),
            "video_sharpen_strength": self.video_model_selector.sharpen_strength(),
            "image_denoise_model": self.image_model_selector.denoise_model_name(),
            "video_denoise_model": self.video_model_selector.denoise_model_name(),
        }

    def _save_window_settings(self) -> None:
        save_settings(self._window_settings())

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
            "ONNX 模型 (*.onnx)",
        )
        if path:
            try:
                meta = self.image_model_selector.import_and_register_model_from_path(path)
                self.video_model_selector.refresh_models()
                self.import_model_status.setText(
                    f"已导入并注册：{meta.name} / {meta.model_path}"
                )
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", str(exc))

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
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
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

    def _sync_resize_percent_state(self) -> None:
        """The percent field only applies to the percentage resize mode."""
        percent_mode = (
            self.image_resize_mode_combo.currentData() == RESIZE_MODE_PERCENT
        )
        self.image_resize_percent_spin.setVisible(percent_mode)
        if percent_mode:
            mode = self.resize_mode()
            self.image_resize_percent_spin.setToolTip(
                f"输出尺寸 = 超分结果 × {self.image_resize_percent_spin.value()}%"
                f"（{mode}）"
            )

    def resize_mode(self) -> str:
        """Requested output-size handling (keep / restore / percent)."""
        return str(self.image_resize_mode_combo.currentData() or "keep")

    def resize_percent(self) -> Optional[float]:
        """Percentage of the super-resolved result, or None when unused."""
        if self.resize_mode() != RESIZE_MODE_PERCENT:
            return None
        return float(self.image_resize_percent_spin.value())

    def resize_options(self) -> tuple[bool, Optional[float]]:
        """``(resize_to_original, resize_percent)`` for the pipeline."""
        return (
            self.resize_mode() == "restore",
            self.resize_percent(),
        )

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

    def _set_progress_color(self, bar: QProgressBar, color: str) -> None:
        bar.setStyleSheet(_PROGRESS_CHUNK_QSS % color)

    def _reset_progress_color(self, bar: QProgressBar) -> None:
        bar.setStyleSheet("")

    def _reset_image_progress(self) -> None:
        self.image_progress.setRange(0, 1000)
        self.image_progress.setValue(0)
        self._reset_progress_color(self.image_progress)

    def _reset_video_progress(self) -> None:
        self.video_progress.setRange(0, 1000)
        self.video_progress.setValue(0)
        self._reset_progress_color(self.video_progress)

    def start_image_upscale(self) -> None:
        if self._task_busy():
            QMessageBox.information(self, "任务进行中", "请等待当前任务结束后再开始新任务")
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
        sharpen_strength = self.image_model_selector.sharpen_strength()
        resize_to_original, resize_percent = self.resize_options()
        compression_profile = str(
            self.image_compression_combo.currentData() or "high"
        )
        out_dir = ui_logic.resolve_project_path(out_dir_text)
        self._image_progress_key = None
        self._reset_image_progress()

        self.image_log.append(
            "Info",
            f"检测到显卡：{ui_logic.gpu_summary(self.gpu, self.available)}",
        )
        denoise_text = ui_logic.denoise_summary(model, denoise_strength)
        self.image_log.append(
            "Info",
            f"加载模型：{model.name} / {ui_logic.backend_label(backend, self.gpu)} / "
            f"{scale}x / 输出尺寸 {resize_summary(resize_to_original, resize_percent)} / "
            f"降噪强度 {denoise_text} / "
            f"边缘锐化 {sharpen_strength:.2f}",
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
            sharpen_strength=sharpen_strength,
            resize_to_original=resize_to_original,
            compression_profile=compression_profile,
            resize_percent=resize_percent,
            parent=self,
        )
        worker.progress.connect(self._on_image_progress)
        worker.log.connect(self._on_image_log)
        worker.finished.connect(self._on_image_finished)
        worker.error.connect(self._on_image_error)
        worker.cancelled.connect(self._on_image_cancelled)
        self._image_worker = worker
        worker.runtime.connect(self._on_image_runtime)
        self._set_image_running(True)
        worker.start()

    def cancel_image_upscale(self) -> None:
        if self._image_worker is not None:
            self._image_worker.cancel()
            self.image_cancel_button.setEnabled(False)
            self.image_status.setText("正在取消...")

    def start_video_upscale(self) -> None:
        if self._task_busy():
            QMessageBox.information(self, "任务进行中", "请等待当前任务结束后再开始新任务")
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
        sharpen_strength = self.video_model_selector.sharpen_strength()
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
        denoise_text = ui_logic.denoise_summary(model, denoise_strength)
        self.video_log.append(
            "Info",
            f"加载模型：{model.name} / {ui_logic.backend_label(backend, self.gpu)} / "
            f"{scale}x / 降噪强度 {denoise_text} / "
            f"边缘锐化 {sharpen_strength:.2f}",
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
            sharpen_strength=sharpen_strength,
            parent=self,
        )
        worker.progress.connect(self._on_video_progress)
        worker.finished.connect(self._on_video_finished)
        worker.error.connect(self._on_video_error)
        worker.cancelled.connect(self._on_video_cancelled)
        self._video_worker = worker
        worker.runtime.connect(self._on_video_runtime)
        self._set_video_running(True)
        worker.start()

    def cancel_video_upscale(self) -> None:
        if self._video_worker is not None:
            self._video_worker.cancel()
            self.video_cancel_button.setEnabled(False)
            self.video_status.setText("正在取消...")

    def _on_image_progress(self, info: ProgressInfo) -> None:
        self.image_progress.setRange(0, 1000)
        self.image_progress.setValue(ui_logic.running_progress_value(info.percent))
        self.image_status.setText(info.message)
        self._log_image_progress(info)

    def _on_image_runtime(self, label: str) -> None:
        self.image_log.append("Info", f"实际推理后端：{label}")

    def _compare_button(self, text: str, slot) -> QPushButton:
        return compare_view.ghost_button(text, slot)

    def _on_image_log(self, level: str, message: str) -> None:
        """Step log lines produced by the pipeline (decode / tiles / encode)."""
        self.image_log.append(level, message)

    # ---- comparison preview --------------------------------------------------

    def _hide_compare_preview(self) -> None:
        self._preview_token += 1
        self.image_compare_card.setVisible(False)
        self.image_compare_view.set_pair(None)
        if self._compare_dialog is not None:
            self._compare_dialog.close()
        self._cleanup_preview_worker()

    def close_image_compare(self) -> None:
        self._hide_compare_preview()

    def _on_compare_mode_changed(self, mode: object = None) -> None:
        """Keep both comparison surfaces (inline card and big window) in step."""
        mode = str(
            mode
            or self.image_compare_mode_combo.currentData()
            or compare_view.MODE_BLOCK
        )
        self.image_compare_view.set_mode(mode)
        if self._compare_dialog is not None:
            self._compare_dialog.panel.set_mode(mode)

    def _on_compare_window_mode_changed(self, mode: str) -> None:
        self.image_compare_panel.set_mode(mode)

    def image_compare_view_fit(self) -> None:
        self.image_compare_view.fit_to_view()
        if self._compare_dialog is not None:
            self._compare_dialog.view().fit_to_view()

    def image_compare_view_pixels(self) -> None:
        self.image_compare_view.one_to_one()
        if self._compare_dialog is not None:
            self._compare_dialog.view().one_to_one()

    def open_image_compare_window(self) -> None:
        """Open the same comparison in a big, resizable window."""
        if not self.image_compare_view.has_pair():
            return
        dialog = self._compare_dialog
        if dialog is None:
            dialog = compare_view.ComparisonDialog(self)
            dialog.panel.mode_changed.connect(self._on_compare_window_mode_changed)
            dialog.finished.connect(self._on_compare_window_closed)
            self._compare_dialog = dialog
            self._apply_saved_compare_window(dialog)

        source = self.image_compare_view
        target = dialog.view()
        target.set_pair(source.pair())
        dialog.panel.set_mode(source.mode())
        target.set_split_ratio(source.split_ratio())
        target.set_zoom(source.zoom())
        dialog.panel.set_summary(self.image_compare_status.text())
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _apply_saved_compare_window(self, dialog) -> None:
        """Restore the size the big window had last time."""
        saved = self._settings.get("image_compare_window")
        maximized = False
        if isinstance(saved, dict):
            width = int(saved.get("width") or 0)
            height = int(saved.get("height") or 0)
            if width > 400 and height > 300:
                dialog.resize(width, height)
            maximized = bool(saved.get("maximized", False))
        if maximized:
            dialog.showMaximized()

    def _store_compare_window_state(self) -> None:
        dialog = self._compare_dialog
        if dialog is None:
            return
        maximized = dialog.isMaximized()
        size = dialog.normalGeometry().size() if maximized else dialog.size()
        self._settings["image_compare_window"] = {
            "width": int(size.width()),
            "height": int(size.height()),
            "maximized": maximized,
        }
        save_settings(self._window_settings())

    def _on_compare_window_closed(self, _result: int) -> None:
        """Remember the window's size and adopt its mode/zoom for the card."""
        self._store_compare_window_state()
        self._sync_compare_from_window()

    def _sync_compare_from_window(self) -> None:
        """Adopt the big window's mode/zoom when it closes."""
        dialog = self._compare_dialog
        if dialog is None:
            return
        source = dialog.view()
        self.image_compare_panel.set_mode(source.mode())
        self.image_compare_view.set_split_ratio(source.split_ratio())
        self.image_compare_view.set_zoom(source.zoom())

    def _cleanup_preview_worker(self) -> None:
        worker = self._preview_worker
        self._preview_worker = None
        if worker is not None:
            worker.deleteLater()

    def _maybe_show_compare_preview(self, results: object) -> None:
        """Show an input/output comparison once a single-image task finished."""
        results = list(results or [])
        usable = [
            result
            for result in results
            if getattr(result, "ok", False)
            and getattr(result, "input_path", None)
            and getattr(result, "output_path", None)
        ]
        if len(results) != 1 or len(usable) != 1:
            self._hide_compare_preview()
            return

        result = usable[0]
        self._preview_token += 1
        token = self._preview_token
        self._cleanup_preview_worker()
        self.image_compare_card.setVisible(True)
        self.image_compare_view.set_pair(None)
        self.image_compare_status.setText("正在载入对比图…")
        worker = compare_view.PreviewLoadWorker(
            token, result.input_path, result.output_path, parent=self
        )
        worker.loaded.connect(self._on_preview_loaded)
        worker.failed.connect(self._on_preview_failed)
        self._preview_worker = worker
        worker.start()

    def _on_preview_loaded(self, token: object, pair: object) -> None:
        self._cleanup_preview_worker()
        if token != self._preview_token:
            return
        self.image_compare_view.set_pair(pair)
        summary = (
            f"原图 {pair.input_true_text} → 输出 {pair.output_true_text} · "
            f"{pair.scaled_note}"
        )
        self.image_compare_status.setText(summary)
        if self._compare_dialog is not None:
            self._compare_dialog.view().set_pair(pair)
            self._compare_dialog.panel.set_summary(summary)
        # Scroll just enough to show the card's header (its view is tall).
        self.image_scroll.ensureWidgetVisible(self.image_compare_status, 0, 40)
        self.image_log.append("Info", compare_view.format_pair_summary(pair))

    def _on_preview_failed(self, token: object, message: str) -> None:
        self._cleanup_preview_worker()
        if token != self._preview_token:
            return
        self.image_compare_status.setText("对比图载入失败")
        self.image_log.append("Warning", f"对比预览不可用：{message}")

    def _on_video_progress(self, info: ProgressInfo) -> None:
        self.video_progress.setRange(0, 1000)
        self.video_progress.setValue(ui_logic.running_progress_value(info.percent))
        self.video_status.setText(info.message)
        self._log_video_progress(info)

    def _on_video_runtime(self, label: str) -> None:
        self.video_log.append("Info", f"实际推理后端：{label}")

    def _log_image_progress(self, info: ProgressInfo) -> None:
        """Log the start of each image in a batch.

        Per-step lines (decode, every N-th tile, encode) arrive through
        ``_on_image_log``; echoing the raw progress here as well would double
        every tile in the panel.
        """
        source_name = info.source_name
        if info.image_index is None or source_name is None:
            return
        image_key = (int(info.image_index), source_name)
        if image_key == self._image_progress_key:
            return
        self._image_progress_key = image_key
        self.image_log.append(
            "Info",
            f"处理 {source_name}（第 {info.image_index}/{info.image_total} 张）",
        )

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
        self.image_progress.setRange(0, 1000)
        self.image_progress.setValue(1000 if results else 0)
        if results and fail_count == 0:
            self._set_progress_color(self.image_progress, _PROGRESS_COLOR_SUCCESS)
        elif ok_count == 0:
            self._set_progress_color(self.image_progress, _PROGRESS_COLOR_ERROR)
        else:
            self._set_progress_color(self.image_progress, _PROGRESS_COLOR_PARTIAL)
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
        self._maybe_show_compare_preview(results)
        self._cleanup_image_worker()

    def _on_image_error(self, message: str) -> None:
        self._set_image_running(False)
        self.image_progress.setRange(0, 1000)
        self.image_progress.setValue(0)
        self._set_progress_color(self.image_progress, _PROGRESS_COLOR_ERROR)
        self.image_status.setText("处理失败")
        self.image_log.append("Error", message)
        self._cleanup_image_worker()

    def _on_image_cancelled(self) -> None:
        self._set_image_running(False)
        self.image_progress.setRange(0, 1000)
        self.image_progress.setValue(0)
        self._set_progress_color(self.image_progress, _PROGRESS_COLOR_CANCELLED)
        self.image_status.setText("已取消")
        self.image_log.append("Warning", "任务已取消")
        self._cleanup_image_worker()

    def _on_video_finished(self, result: SuperResResult) -> None:
        self._set_video_running(False)
        self.video_progress.setRange(0, 1000)
        self.video_progress.setValue(1000)
        self._set_progress_color(self.video_progress, _PROGRESS_COLOR_SUCCESS)
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
        self.video_progress.setRange(0, 1000)
        self.video_progress.setValue(0)
        self._set_progress_color(self.video_progress, _PROGRESS_COLOR_ERROR)
        self.video_status.setText("处理失败")
        self.video_log.append("Error", message)
        self._cleanup_video_worker()

    def _on_video_cancelled(self) -> None:
        self._set_video_running(False)
        self.video_progress.setRange(0, 1000)
        self.video_progress.setValue(0)
        self._set_progress_color(self.video_progress, _PROGRESS_COLOR_CANCELLED)
        self.video_status.setText("已取消")
        self.video_log.append("Warning", "任务已取消")
        self._cleanup_video_worker()

    def _set_image_running(self, running: bool) -> None:
        if running:
            self.image_start_button.setEnabled(False)
            self.video_start_button.setEnabled(False)
            self.image_cancel_button.setEnabled(True)
            self.video_cancel_button.setEnabled(False)
            return
        self._refresh_task_controls()

    def _set_video_running(self, running: bool) -> None:
        if running:
            self.image_start_button.setEnabled(False)
            self.video_start_button.setEnabled(False)
            self.image_cancel_button.setEnabled(False)
            self.video_cancel_button.setEnabled(True)
            return
        self._refresh_task_controls()

    def _task_busy(self) -> bool:
        return any(
            worker is not None and worker.isRunning()
            for worker in (self._image_worker, self._video_worker)
        )

    def _refresh_task_controls(self) -> None:
        image_running = self._image_worker is not None and self._image_worker.isRunning()
        video_running = self._video_worker is not None and self._video_worker.isRunning()
        busy = image_running or video_running
        self.image_start_button.setEnabled(not busy)
        self.video_start_button.setEnabled(not busy)
        self.image_cancel_button.setEnabled(image_running)
        self.video_cancel_button.setEnabled(video_running)

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
        self._save_window_settings()
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setFont(QFont(theme.UI_FONT_FAMILY, theme.UI_FONT_POINT_SIZE))
        self.setStyleSheet(theme.APP_STYLESHEET)
