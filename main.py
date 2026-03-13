import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa
import pyaudio
import threading
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QFileDialog, QGroupBox, QGridLayout, QStatusBar,
                             QProgressBar, QSplitter, QFrame, QStackedWidget,
                             QGraphicsDropShadowEffect, QSizePolicy, QComboBox, QDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QDragEnterEvent, QDropEvent
from scipy.ndimage import zoom, uniform_filter1d
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 核心参数
CHUNK = 2048          
FORMAT = pyaudio.paFloat32
CHANNELS = 1
SR = 16000            
N_FFT = 2048
HOP_LENGTH = 512
STFT_DISPLAY_FRAMES = 30  

# 默认参数
VOCAL_FREQ_MIN = 250
VOCAL_FREQ_MAX = 2800
VOCAL_GAIN = 3.0
BG_GAIN = 0.0

# 全局状态
audio_data = None
vocal_data = None
play_position = 0
is_playing = False
pause_flag = False

current_audio_chunk = np.zeros(CHUNK)
current_vocal_chunk = np.zeros(CHUNK)
current_stft_data = np.zeros((N_FFT//2+1, STFT_DISPLAY_FRAMES))
current_vocal_stft_data = np.zeros((N_FFT//2+1, STFT_DISPLAY_FRAMES))

# 波形数据缓存
WAVEFORM_DISPLAY_SAMPLES = 2000  # 波形显示点数
current_waveform_data = np.zeros(WAVEFORM_DISPLAY_SAMPLES)
current_vocal_waveform_data = np.zeros(WAVEFORM_DISPLAY_SAMPLES)

# 预计算频率bins
freq_bins = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
vocal_freq_mask = (freq_bins >= VOCAL_FREQ_MIN) & (freq_bins <= VOCAL_FREQ_MAX)

# 颜色主题
COLORS = {
    'primary': '#2D9CDB',
    'primary_dark': '#1A7AB8',
    'secondary': '#27AE60',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'bg_light': '#F8F9FA',
    'bg_dark': '#1E1E1E',
    'text_dark': '#2C3E50',
    'text_light': '#FFFFFF',
    'border': '#E0E0E0',
    'shadow': 'rgba(0, 0, 0, 0.1)'
}


class SignalEmitter(QObject):
    """用于线程间通信的信号发射器"""
    update_signal = pyqtSignal()
    status_signal = pyqtSignal(str)


class ProcessingDialog(QDialog):
    """处理进度弹窗"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 180)
        # 设置无边框和透明背景
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建圆角容器
        self.container = QFrame()
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 16px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self.container)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 6)
        self.container.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)
        
        # 标题
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_dark']};")
        layout.addWidget(self.title_label)
        
        self.label = QLabel("正在处理，请稍候...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_dark']};")
        layout.addWidget(self.label)
        
        # 提示标签
        self.hint_label = QLabel("⚠️ 处理过程中界面可能暂时无响应，请耐心等待")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet(f"font-size: 11px; color: {COLORS['warning']};")
        layout.addWidget(self.hint_label)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 10px;
                text-align: center;
                height: 20px;
                background-color: #E0E0E0;
                font-size: 12px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 10px;
            }}
        """)
        layout.addWidget(self.progress)
        
        main_layout.addWidget(self.container)
    
    def update_progress(self, value, text=None):
        self.progress.setValue(value)
        if text:
            self.label.setText(text)


class ProcessingThread(QThread):
    """后台处理线程"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)
    
    def __init__(self, audio, processor_func):
        super().__init__()
        self.audio = audio
        self.processor_func = processor_func
    
    def run(self):
        try:
            result = self.processor_func(self.audio, self.progress_signal)
            self.finished_signal.emit(result)
        except Exception as e:
            self.finished_signal.emit(e)


class DropArea(QFrame):
    """拖放上传区域"""
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self.setCursor(Qt.PointingHandCursor)
        self.has_file = False
        
        self.setup_ui()
        self.update_style()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)
        
        # 上传图标
        self.icon_label = QLabel("📁")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 48px;")
        layout.addWidget(self.icon_label)
        
        # 提示文字
        self.text_label = QLabel("拖放音频文件至此\n或点击上传")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet(f"font-size: 14px; color: {COLORS['text_dark']}; line-height: 1.5;")
        layout.addWidget(self.text_label)
        
        # 文件信息（初始隐藏）
        self.file_info = QWidget()
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        self.file_name = QLabel()
        self.file_name.setAlignment(Qt.AlignCenter)
        self.file_name.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['primary']};")
        info_layout.addWidget(self.file_name)
        
        self.file_meta = QLabel()
        self.file_meta.setAlignment(Qt.AlignCenter)
        self.file_meta.setStyleSheet(f"font-size: 12px; color: #7F8C8D;")
        info_layout.addWidget(self.file_meta)
        
        self.file_info.setLayout(info_layout)
        self.file_info.hide()
        layout.addWidget(self.file_info)
        
        self.setLayout(layout)
    
    def update_style(self, hover=False):
        border_color = COLORS['primary'] if hover else COLORS['border']
        bg_color = "#E3F2FD" if hover else "white"
        
        self.setStyleSheet(f"""
            DropArea {{
                border: 2px dashed {border_color};
                border-radius: 16px;
                background-color: {bg_color};
            }}
        """)
    
    def set_file(self, file_path):
        self.has_file = True
        self.file_path = file_path
        
        # 显示文件信息
        import os
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        self.file_name.setText(file_name)
        self.file_meta.setText(f"{file_size:.1f} MB")
        
        self.icon_label.hide()
        self.text_label.hide()
        self.file_info.show()
        
        self.update_style(False)
    
    def clear(self):
        self.has_file = False
        self.file_path = None
        
        self.icon_label.show()
        self.text_label.show()
        self.file_info.hide()
        
        self.update_style(False)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.update_style(True)
    
    def dragLeaveEvent(self, event):
        self.update_style(False)
    
    def dropEvent(self, event: QDropEvent):
        self.update_style(False)
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                self.file_dropped.emit(file_path)
    
    def mousePressEvent(self, event):
        if not self.has_file:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择音频文件", "", 
                "音频文件 (*.wav *.mp3 *.flac *.m4a *.ogg);;所有文件 (*.*)"
            )
            if file_path:
                self.file_dropped.emit(file_path)


class ModernButton(QPushButton):
    """现代化按钮"""
    def __init__(self, text, btn_type='primary', parent=None):
        super().__init__(text, parent)
        self.btn_type = btn_type
        self.setMinimumHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self.update_style()
    
    def update_style(self):
        if self.btn_type == 'primary':
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 12px 24px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['primary_dark']};
                }}
                QPushButton:pressed {{
                    background-color: #145A8C;
                }}
                QPushButton:disabled {{
                    background-color: #B0BEC5;
                }}
            """)
        elif self.btn_type == 'secondary':
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {COLORS['secondary']};
                    border: 2px solid {COLORS['secondary']};
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 12px 24px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['secondary']};
                    color: white;
                }}
                QPushButton:disabled {{
                    border-color: #B0BEC5;
                    color: #B0BEC5;
                }}
            """)
        elif self.btn_type == 'danger':
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {COLORS['danger']};
                    border: 2px solid {COLORS['danger']};
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 12px 24px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['danger']};
                    color: white;
                }}
            """)


class AudioProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人声分离器")
        self.setMinimumSize(1200, 800)
        
        # 设置全局样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_light']};
            }}
            QLabel {{
                color: {COLORS['text_dark']};
            }}
        """)
        
        # 信号发射器
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.update_signal.connect(self.update_plots)
        
        # 初始化UI
        self.setup_ui()
        
        # 播放线程
        self.play_thread = None
        
        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)
        
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(50)
    
    def setup_ui(self):
        """设置UI界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)
        
        # ===== 左侧操作区（60%）=====
        left_panel = QWidget()
        left_panel.setMaximumWidth(700)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(20)
        left_panel.setLayout(left_layout)
        
        # 标题
        title_layout = QHBoxLayout()
        title_label = QLabel("🎵 人声分离器")
        title_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLORS['text_dark']};")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        left_layout.addLayout(title_layout)
        
        # Step 1: 文件上传区
        upload_group = QGroupBox("Step 1: 上传音频")
        upload_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: none;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 0 10px 0;
                color: {COLORS['text_dark']};
            }}
        """)
        upload_layout = QVBoxLayout()
        
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.load_audio)
        upload_layout.addWidget(self.drop_area)
        
        upload_group.setLayout(upload_layout)
        left_layout.addWidget(upload_group)
        
        # Step 2: 核心功能按钮区
        action_group = QGroupBox("Step 2: 提取音频")
        action_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: none;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 0 10px 0;
                color: {COLORS['text_dark']};
            }}
        """)
        action_layout = QVBoxLayout()
        action_layout.setSpacing(12)
        
        self.extract_vocal_btn = ModernButton("🎤 提取人声", "primary")
        self.extract_vocal_btn.clicked.connect(self.extract_vocal)
        self.extract_vocal_btn.setEnabled(False)
        action_layout.addWidget(self.extract_vocal_btn)
        
        btn_row = QHBoxLayout()
        
        self.reset_btn = ModernButton("↺ 重置", "danger")
        self.reset_btn.clicked.connect(self.reset_all)
        btn_row.addWidget(self.reset_btn)
        
        action_layout.addLayout(btn_row)
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        # Step 3: 音频控制区
        control_group = QGroupBox("Step 3: 播放控制")
        control_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: none;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 0 10px 0;
                color: {COLORS['text_dark']};
            }}
        """)
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        
        # 播放模式切换
        mode_layout = QHBoxLayout()
        mode_label = QLabel("播放模式:")
        mode_label.setStyleSheet("font-size: 12px; color: #7F8C8D;")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["原音频", "提取后人声"])
        self.mode_combo.setEnabled(False)
        self.mode_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
                background-color: white;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        control_layout.addLayout(mode_layout)
        
        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(False)
        self.progress_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: #E0E0E0;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['primary']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['primary']};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS['primary_dark']};
            }}
        """)
        # 连接进度条信号
        self.progress_slider.sliderPressed.connect(self.on_progress_pressed)
        self.progress_slider.sliderReleased.connect(self.on_progress_released)
        self.progress_slider.valueChanged.connect(self.on_progress_changed)
        control_layout.addWidget(self.progress_slider)
        
        # 是否正在拖动进度条
        self.is_seeking = False
        
        # 时间显示
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 12px; color: #7F8C8D;")
        control_layout.addWidget(self.time_label)
        
        # 播放控制按钮
        play_layout = QHBoxLayout()
        play_layout.setAlignment(Qt.AlignCenter)
        play_layout.setSpacing(15)
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(60, 60)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 30px;
                font-size: 24px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: #B0BEC5;
            }}
        """)
        self.play_btn.clicked.connect(self.play_audio)
        play_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setFixedSize(48, 48)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                border: none;
                border-radius: 24px;
                font-size: 18px;
            }}
            QPushButton:hover {{
                background-color: #C0392B;
            }}
        """)
        self.stop_btn.clicked.connect(self.stop_audio)
        play_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(play_layout)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # Step 4: 导出区
        export_group = QGroupBox("Step 4: 导出音频")
        export_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: none;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 0 10px 0;
                color: {COLORS['text_dark']};
            }}
        """)
        export_layout = QVBoxLayout()
        
        self.export_btn = ModernButton("💾 导出音频", "primary")
        self.export_btn.clicked.connect(self.export_audio)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel, 60)
        
        # ===== 右侧可视化区（40%）=====
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # 频谱图卡片
        viz_card = QFrame()
        viz_card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-radius: 16px;
            }}
        """)
        viz_layout = QVBoxLayout()
        viz_layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题
        viz_title = QLabel("📊 STFT频谱图")
        viz_title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        viz_layout.addWidget(viz_title)
        
        # Matplotlib图表 - 2行2列：左边STFT，右边波形
        self.fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.patch.set_facecolor(COLORS['bg_dark'])
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 原始STFT图（左上）
        self.ax_stft = axes[0, 0]
        self.ax_stft.set_facecolor(COLORS['bg_dark'])
        self.im_stft = self.ax_stft.imshow(
            current_stft_data,
            aspect='auto',
            origin='lower',
            extent=[0, STFT_DISPLAY_FRAMES*CHUNK/SR, 0, SR/2],
            cmap='plasma',
            vmin=-80, vmax=0
        )
        self.ax_stft.set_title('原始音频频谱', color='white', fontsize=10)
        self.ax_stft.set_xlabel('时间 (s)', color='white')
        self.ax_stft.set_ylabel('频率 (Hz)', color='white')
        self.ax_stft.tick_params(colors='white')
        
        # 原始音频波形（右上）
        self.ax_waveform = axes[0, 1]
        self.ax_waveform.set_facecolor(COLORS['bg_dark'])
        self.line_waveform, = self.ax_waveform.plot([], [], color='#00ff88', linewidth=0.5)
        self.ax_waveform.set_xlim(0, STFT_DISPLAY_FRAMES*CHUNK/SR)
        self.ax_waveform.set_ylim(-1, 1)
        self.ax_waveform.set_title('原始音频波形', color='white', fontsize=10)
        self.ax_waveform.set_xlabel('时间 (s)', color='white')
        self.ax_waveform.set_ylabel('振幅', color='white')
        self.ax_waveform.tick_params(colors='white')
        
        # 处理后STFT图（左下）
        self.ax_vocal_stft = axes[1, 0]
        self.ax_vocal_stft.set_facecolor(COLORS['bg_dark'])
        self.im_vocal_stft = self.ax_vocal_stft.imshow(
            current_vocal_stft_data,
            aspect='auto',
            origin='lower',
            extent=[0, STFT_DISPLAY_FRAMES*CHUNK/SR, 0, SR/2],
            cmap='plasma',
            vmin=-80, vmax=0
        )
        self.ax_vocal_stft.set_title('处理后音频频谱', color='white', fontsize=10)
        self.ax_vocal_stft.set_xlabel('时间 (s)', color='white')
        self.ax_vocal_stft.set_ylabel('频率 (Hz)', color='white')
        self.ax_vocal_stft.tick_params(colors='white')
        
        # 处理后音频波形（右下）
        self.ax_vocal_waveform = axes[1, 1]
        self.ax_vocal_waveform.set_facecolor(COLORS['bg_dark'])
        self.line_vocal_waveform, = self.ax_vocal_waveform.plot([], [], color='#ff6b6b', linewidth=0.5)
        self.ax_vocal_waveform.set_xlim(0, STFT_DISPLAY_FRAMES*CHUNK/SR)
        self.ax_vocal_waveform.set_ylim(-1, 1)
        self.ax_vocal_waveform.set_title('处理后音频波形', color='white', fontsize=10)
        self.ax_vocal_waveform.set_xlabel('时间 (s)', color='white')
        self.ax_vocal_waveform.set_ylabel('振幅', color='white')
        self.ax_vocal_waveform.tick_params(colors='white')
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: transparent;")
        viz_layout.addWidget(self.canvas)
        
        viz_card.setLayout(viz_layout)
        right_layout.addWidget(viz_card)
        
        main_layout.addWidget(right_panel, 40)
    
    def load_audio(self, file_path):
        """加载音频文件"""
        self.drop_area.set_file(file_path)
        
        # 显示加载弹窗
        self.loading_dialog = ProcessingDialog("正在加载音频", self)
        self.loading_dialog.show()
        
        # 启动后台线程
        self.load_thread = WorkerThread(self._load_audio_worker, file_path)
        self.load_thread.finished_signal.connect(self._load_audio_finished)
        self.load_thread.start()
    
    def _load_audio_worker(self, file_path):
        """后台加载音频"""
        try:
            y, _ = librosa.load(file_path, sr=SR, mono=True)
            y = y / np.max(np.abs(y))
            return y
        except Exception as e:
            return e
    
    def _load_audio_finished(self, result):
        """加载完成回调"""
        self.loading_dialog.close()
        
        if isinstance(result, Exception):
            self.show_toast(f"加载失败: {str(result)}", "error")
            return
        
        global audio_data, play_position
        audio_data = result
        play_position = 0
        
        # 启用提取按钮
        self.extract_vocal_btn.setEnabled(True)
        self.show_toast("音频加载成功！", "success")
        self.update_time_display()
    
    def extract_vocal(self):
        """提取人声"""
        self._extract_audio()
    
    def _extract_audio(self):
        """提取音频"""
        global vocal_data
        
        # 禁用按钮
        self.extract_vocal_btn.setEnabled(False)
        
        # 显示处理弹窗
        self.processing_dialog = ProcessingDialog("正在提取音频", self)
        self.processing_dialog.show()
        
        # 启动后台线程
        self.process_thread = ProcessingThread(audio_data, self._process_audio_worker)
        self.process_thread.progress_signal.connect(self._update_processing_progress)
        self.process_thread.finished_signal.connect(self._process_audio_finished)
        self.process_thread.start()
    
    def _process_audio_worker(self, audio, progress_signal):
        """后台处理音频"""
        result = self.high_purity_vocal_separation_offline(audio, progress_signal)
        return result
    
    def _update_processing_progress(self, value, text):
        """更新处理进度"""
        self.processing_dialog.update_progress(value, text)
    
    def _process_audio_finished(self, result):
        """处理完成回调"""
        self.processing_dialog.close()
        
        if isinstance(result, Exception):
            self.show_toast(f"处理失败: {str(result)}", "error")
            self.extract_vocal_btn.setEnabled(True)
            return
        
        global vocal_data
        vocal_data = result
        
        # 启用播放和导出按钮
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.progress_slider.setEnabled(True)
        
        self.show_toast("人声提取完成！", "success")
        self.update_time_display()
    
    def high_purity_vocal_separation_offline(self, audio, progress_signal=None):
        """非实时高质量人声分离 - 恢复原始有效版本"""
        global audio_stft_cache, vocal_stft_cache
        
        def emit_progress(value, text):
            if progress_signal:
                progress_signal.emit(value, text)
        
        emit_progress(5, "正在初始化参数...")
        
        # 使用固定参数
        current_freq_min = 250
        current_freq_max = 2800
        current_vocal_gain = 3.0
        current_bg_gain = 0.0
        
        n_fft_large = 4096
        hop_large = 1024
        
        emit_progress(10, "正在计算频谱...")
        
        freq_bins_large = librosa.fft_frequencies(sr=SR, n_fft=n_fft_large)
        vocal_mask = (freq_bins_large >= current_freq_min) & (freq_bins_large <= current_freq_max)
        
        stft = librosa.stft(audio, n_fft=n_fft_large, hop_length=hop_large, window='hann')
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        emit_progress(20, "正在计算谱特征...")
        
        epsilon = 1e-10
        geometric_mean = np.exp(np.mean(np.log(mag + epsilon), axis=0))
        arithmetic_mean = np.mean(mag, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + epsilon)
        
        flatness_mask = 1.0 - spectral_flatness
        flatness_mask = np.clip(flatness_mask * 2, 0.3, 1.5)
        flatness_mask = np.tile(flatness_mask, (mag.shape[0], 1))
        
        local_std = np.array([np.std(mag[i:i+10, :], axis=0) for i in range(0, mag.shape[0]-10, 10)])
        if local_std.shape[0] > 0:
            contrast_mask = zoom(local_std, (mag.shape[0]/local_std.shape[0], 1), order=1)
            contrast_mask = contrast_mask / (np.max(contrast_mask) + epsilon)
            contrast_mask = np.clip(contrast_mask * 1.5 + 0.5, 0.5, 1.5)
        else:
            contrast_mask = np.ones_like(mag)
        
        if np.any(~vocal_mask):
            noise_profile = np.median(mag[~vocal_mask, :], axis=0, keepdims=True)
            mag_subtracted = mag.copy()
            mag_subtracted[vocal_mask, :] = np.maximum(
                mag[vocal_mask, :] - noise_profile * 0.8, 0
            )
        else:
            mag_subtracted = mag
        
        effective_vocal_gain = max(current_vocal_gain, 0.01)
        effective_bg_gain = max(current_bg_gain, 0.01)
        
        mask = np.zeros_like(mag)
        mask[vocal_mask, :] = effective_vocal_gain
        mask[~vocal_mask, :] = effective_bg_gain
        
        mask = mask * flatness_mask * contrast_mask
        
        vocal_mag = mag_subtracted * mask
        if current_bg_gain <= 0.01:
            vocal_mag[~vocal_mask, :] = 0
        
        frame_max = np.max(vocal_mag, axis=0, keepdims=True)
        threshold_mask = (vocal_mag > frame_max * 0.1).astype(float)
        vocal_mag = vocal_mag * threshold_mask
        
        vocal_stft = vocal_mag * np.exp(1j * phase)
        vocal_audio = librosa.istft(
            vocal_stft, 
            hop_length=hop_large, 
            window='hann',
            length=len(audio)
        )
        
        emit_progress(30, "正在进行HPSS分离...")
        
        # 第二步：HPSS分离
        D = librosa.stft(vocal_audio, n_fft=2048, hop_length=512)
        
        D_harmonic, D_percussive = librosa.decompose.hpss(
            D, 
            kernel_size=31,
            power=1.0,
            mask=False
        )
        
        harmonic_mag = np.abs(D_harmonic)
        percussive_mag = np.abs(D_percussive)
        suppression_ratio = percussive_mag / (harmonic_mag + percussive_mag + 1e-10)
        suppression_mask = 1.0 - np.clip(suppression_ratio * 0.5, 0, 0.5)
        D_harmonic_enhanced = D_harmonic * suppression_mask
        
        vocal_audio_hpss = librosa.istft(D_harmonic_enhanced, hop_length=512, length=len(audio))
        
        emit_progress(50, "正在进行基频检测...")
        
        # 第三步：基频检测 - 使用 torchfcpe
        hop_length_yin = 512
        
        try:
            from torchfcpe import spawn_bundled_infer_model
            import torch
            
            # 加载模型
            infer_model = spawn_bundled_infer_model(device='cpu')
            
            # 归一化音频到 [-1, 1]
            audio_normalized = vocal_audio_hpss / (np.max(np.abs(vocal_audio_hpss)) + 1e-8)
            
            # 转换为tensor
            audio_tensor = torch.from_numpy(audio_normalized).float().unsqueeze(0)
            
            # 检测基频
            with torch.no_grad():
                f0 = infer_model.infer(
                    audio_tensor,
                    sr=SR,
                    decoder_mode='local_argmax',
                    threshold=0.006
                )
            
            # 转换为numpy
            if isinstance(f0, torch.Tensor):
                f0 = f0.squeeze().cpu().numpy()
            
            # 插值到音频长度
            audio_times = np.arange(len(vocal_audio_hpss)) / SR
            f0_times = np.linspace(0, len(vocal_audio_hpss) / SR, len(f0))
            f0_interp = np.interp(audio_times, f0_times, f0)
            
            # 生成pitch_mask：有基频的区域保持，无基频的区域衰减
            pitch_mask = np.where(f0_interp > 50, 1.0, 0.3)
            pitch_mask = uniform_filter1d(pitch_mask, size=int(SR * 0.1))
            pitch_mask = np.clip(pitch_mask, 0.3, 1.0)
            
        except Exception as e:
            print(f"torchfcpe failed: {e}, using pyin")
            # 回退到pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                vocal_audio_hpss, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=SR,
                hop_length=hop_length_yin
            )
            
            if f0 is not None and len(f0) > 0:
                f0_filled = np.nan_to_num(f0, nan=0)
                times = librosa.frames_to_time(np.arange(len(f0)), sr=SR, hop_length=hop_length_yin)
                audio_times = np.arange(len(vocal_audio_hpss)) / SR
                f0_interp = np.interp(audio_times, times, f0_filled)
                
                pitch_mask = np.where(f0_interp > 0, 1.0, 0.7)
                pitch_mask = uniform_filter1d(pitch_mask, size=int(SR * 0.1))
                pitch_mask = np.clip(pitch_mask, 0.5, 1.0)
            else:
                pitch_mask = np.ones(len(vocal_audio_hpss))
        
        vocal_audio_pitch = vocal_audio_hpss * pitch_mask
        
        emit_progress(70, "正在进行瞬态检测...")
        
        # 第四步：瞬态检测
        onset_env = librosa.onset.onset_strength(y=vocal_audio_pitch, sr=SR, hop_length=hop_length_yin)
        
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=SR, hop_length=hop_length_yin)
        onset_interp = np.interp(audio_times, onset_times, onset_env)
        
        if np.max(onset_interp) > 0:
            onset_interp = onset_interp / np.max(onset_interp)
        
        transient_threshold = np.mean(onset_interp) + 1.5 * np.std(onset_interp)
        transient_mask = np.where(onset_interp > transient_threshold, 0.7, 1.0)
        transient_mask = uniform_filter1d(transient_mask, size=int(SR * 0.05))
        
        vocal_audio_transient = vocal_audio_pitch * transient_mask
        
        # 第五步：混合处理
        vocal_audio_mixed = 0.4 * vocal_audio_transient + 0.6 * vocal_audio_hpss
        
        emit_progress(85, "正在进行EQ优化...")
        
        # 第六步：EQ处理
        vocal_stft_eq = librosa.stft(vocal_audio_mixed, n_fft=4096, hop_length=1024)
        vocal_mag_eq = np.abs(vocal_stft_eq)
        vocal_phase_eq = np.angle(vocal_stft_eq)
        
        freq_bins_eq = librosa.fft_frequencies(sr=SR, n_fft=4096)
        eq_gains = np.ones_like(freq_bins_eq)
        
        eq_gains[freq_bins_eq < 200] = 0.707
        eq_gains[(freq_bins_eq >= 2000) & (freq_bins_eq <= 8000)] *= 1.413
        eq_gains[freq_bins_eq >= 10000] *= 1.259
        
        eq_gains_2d = eq_gains.reshape(-1, 1)
        vocal_mag_eq_boosted = vocal_mag_eq * eq_gains_2d
        
        vocal_stft_eq_boosted = vocal_mag_eq_boosted * np.exp(1j * vocal_phase_eq)
        vocal_audio_eq = librosa.istft(vocal_stft_eq_boosted, hop_length=1024, length=len(audio))
        
        # 高通滤波
        sos = signal.butter(4, 100, 'hp', fs=SR, output='sos')
        vocal_audio_eq = signal.sosfiltfilt(sos, vocal_audio_eq)
        
        emit_progress(88, "正在进行高频增强...")
        
        # 第七步：高频增强（2k-6kHz频段提升）
        vocal_stft_high = librosa.stft(vocal_audio_eq, n_fft=4096, hop_length=1024)
        vocal_mag_high = np.abs(vocal_stft_high)
        vocal_phase_high = np.angle(vocal_stft_high)
        
        freq_bins_high = librosa.fft_frequencies(sr=SR, n_fft=4096)
        
        # 创建增益掩码：只增强2k-6kHz，其他不动
        high_freq_mask = np.ones_like(freq_bins_high)
        # 2k-6kHz提升+4dB（约1.585倍）
        high_freq_mask[(freq_bins_high >= 2000) & (freq_bins_high <= 6000)] *= 1.585
        # 低于2k、高于6k保持1.0（不动）
        
        high_freq_mask_2d = high_freq_mask.reshape(-1, 1)
        vocal_mag_enhanced = vocal_mag_high * high_freq_mask_2d
        
        vocal_stft_enhanced = vocal_mag_enhanced * np.exp(1j * vocal_phase_high)
        vocal_audio_enhanced = librosa.istft(vocal_stft_enhanced, hop_length=1024, length=len(audio))
        
        emit_progress(90, "正在进行频带掩码...")
        
        # 第八步：频带掩码（只保留300Hz-7kHz）
        vocal_stft_mask = librosa.stft(vocal_audio_enhanced, n_fft=4096, hop_length=1024)
        vocal_mag_mask = np.abs(vocal_stft_mask)
        vocal_phase_mask = np.angle(vocal_stft_mask)
        
        freq_bins_mask = librosa.fft_frequencies(sr=SR, n_fft=4096)
        
        # 创建频带掩码：300Hz-7kHz保留，外面衰减
        band_mask = np.ones_like(freq_bins_mask) * 0.1  # 默认衰减到10%
        band_mask[(freq_bins_mask >= 300) & (freq_bins_mask <= 7000)] = 1.0  # 保留频段
        # 过渡带平滑
        transition_low = (freq_bins_mask >= 150) & (freq_bins_mask < 300)
        transition_high = (freq_bins_mask > 7000) & (freq_bins_mask <= 8500)
        band_mask[transition_low] = 0.5  # 过渡带50%
        band_mask[transition_high] = 0.3  # 过渡带30%
        
        band_mask_2d = band_mask.reshape(-1, 1)
        vocal_mag_masked = vocal_mag_mask * band_mask_2d
        
        vocal_stft_masked = vocal_mag_masked * np.exp(1j * vocal_phase_mask)
        vocal_audio_masked = librosa.istft(vocal_stft_masked, hop_length=1024, length=len(audio))
        
        # 归一化并提升音量
        max_val = np.max(np.abs(vocal_audio_masked))
        if max_val > 1e-10:
            vocal_audio_final = vocal_audio_masked / max_val * 1.5
        vocal_audio_final = np.clip(vocal_audio_final, -1.0, 1.0)
        
        emit_progress(95, "正在预计算频谱数据...")
        
        # 预计算所有频谱数据
        n_display_bins = N_FFT // 2 + 1
        n_chunks = len(audio) // CHUNK
        
        audio_stft_cache = np.zeros((n_display_bins, n_chunks))
        vocal_stft_cache = np.zeros((n_display_bins, n_chunks))
        
        n_fft_fast = 512
        hop_fast = 128
        
        # 每处理10%的块更新一次进度
        progress_interval = max(1, n_chunks // 10)
        
        for i in range(n_chunks):
            start = i * CHUNK
            end = start + CHUNK
            
            audio_chunk = audio[start:end]
            if len(audio_chunk) == CHUNK:
                stft = librosa.stft(audio_chunk, n_fft=n_fft_fast, hop_length=hop_fast)
                stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                stft_frame = np.mean(stft_db, axis=1)
                if len(stft_frame) != n_display_bins:
                    stft_frame = np.interp(
                        np.linspace(0, len(stft_frame)-1, n_display_bins),
                        np.arange(len(stft_frame)),
                        stft_frame
                    )
                audio_stft_cache[:, i] = stft_frame
            
            vocal_chunk = vocal_audio_final[start:end]
            if len(vocal_chunk) == CHUNK:
                stft = librosa.stft(vocal_chunk, n_fft=n_fft_fast, hop_length=hop_fast)
                stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                stft_frame = np.mean(stft_db, axis=1)
                if len(stft_frame) != n_display_bins:
                    stft_frame = np.interp(
                        np.linspace(0, len(stft_frame)-1, n_display_bins),
                        np.arange(len(stft_frame)),
                        stft_frame
                    )
                vocal_stft_cache[:, i] = stft_frame
            
            # 每处理一定数量块后更新进度
            if i % progress_interval == 0:
                progress_value = 95 + int((i / n_chunks) * 5)
                emit_progress(progress_value, f"正在预计算频谱数据... ({i+1}/{n_chunks})")
        
        emit_progress(100, "处理完成!")
        
        return vocal_audio_final
    
    def play_audio(self):
        """播放音频"""
        global is_playing, pause_flag
        
        if is_playing and pause_flag:
            pause_flag = False
            self.play_btn.setText("⏸")
            return
        
        if not is_playing:
            is_playing = True
            pause_flag = False
            self.play_btn.setText("⏸")
            self.play_thread = threading.Thread(target=self._play_audio, daemon=True)
            self.play_thread.start()
    
    def _play_audio(self):
        """内部播放函数 - 根据播放模式切换音频源"""
        global play_position, current_audio_chunk, current_vocal_chunk, is_playing, pause_flag
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SR, output=True, frames_per_buffer=CHUNK)
        
        while is_playing:
            if pause_flag:
                time.sleep(0.01)
                continue
            
            # 每次循环都获取当前播放模式（支持实时切换）
            mode = self.mode_combo.currentIndex()  # 0:原音频, 1:提取后人声
            
            if play_position + CHUNK < len(audio_data):
                current_audio_chunk = audio_data[play_position:play_position+CHUNK]
                
                # 根据播放模式选择音频源
                if mode == 0:  # 原音频
                    current_vocal_chunk = audio_data[play_position:play_position+CHUNK]
                elif mode == 1:  # 提取后人声
                    if vocal_data is not None:
                        current_vocal_chunk = vocal_data[play_position:play_position+CHUNK]
                    else:
                        current_vocal_chunk = audio_data[play_position:play_position+CHUNK]
                
                play_position += CHUNK
            else:
                current_audio_chunk = np.zeros(CHUNK)
                current_vocal_chunk = np.zeros(CHUNK)
                is_playing = False
                play_position = 0
                self.play_btn.setText("▶")
            
            stream.write(current_vocal_chunk.astype(np.float32).tobytes())
            self.calc_stft()
        
        stream.close()
        p.terminate()
    
    def stop_audio(self):
        """停止播放"""
        global is_playing, pause_flag, play_position
        is_playing = False
        pause_flag = False
        play_position = 0
        self.play_btn.setText("▶")
        self.update_time_display()
    
    def export_audio(self):
        """导出音频"""
        if vocal_data is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出音频", "extracted_vocal.wav",
            "WAV文件 (*.wav);;MP3文件 (*.mp3)"
        )
        
        if file_path:
            try:
                import soundfile as sf
                sf.write(file_path, vocal_data, SR)
                self.show_toast("导出成功！", "success")
            except Exception as e:
                self.show_toast(f"导出失败: {str(e)}", "error")
    
    def reset_all(self):
        """重置所有"""
        global audio_data, vocal_data, play_position, is_playing, pause_flag
        global current_stft_data, current_vocal_stft_data, current_waveform_data, current_vocal_waveform_data
        
        is_playing = False
        pause_flag = False
        play_position = 0
        audio_data = None
        vocal_data = None
        
        # 重置显示数据
        current_stft_data = np.zeros((N_FFT//2+1, STFT_DISPLAY_FRAMES))
        current_vocal_stft_data = np.zeros((N_FFT//2+1, STFT_DISPLAY_FRAMES))
        current_waveform_data = np.zeros(WAVEFORM_DISPLAY_SAMPLES)
        current_vocal_waveform_data = np.zeros(WAVEFORM_DISPLAY_SAMPLES)
        
        self.drop_area.clear()
        self.extract_vocal_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.progress_slider.setEnabled(False)
        self.play_btn.setText("▶")
        
        self.update_time_display()
        self.show_toast("已重置", "info")
    
    def calc_stft(self):
        """计算STFT和波形"""
        global current_stft_data, current_vocal_stft_data, current_waveform_data, current_vocal_waveform_data
        
        # 计算STFT
        stft = librosa.stft(current_audio_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        stft_frame = np.mean(stft_db, axis=1) if stft_db.shape[1] > 0 else stft_db[:, 0]
        current_stft_data = np.roll(current_stft_data, -1, axis=1)
        current_stft_data[:, -1] = stft_frame
        
        vocal_stft = librosa.stft(current_vocal_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        vocal_stft_db = librosa.amplitude_to_db(np.abs(vocal_stft), ref=np.max)
        vocal_stft_frame = np.mean(vocal_stft_db, axis=1) if vocal_stft_db.shape[1] > 0 else vocal_stft_db[:, 0]
        current_vocal_stft_data = np.roll(current_vocal_stft_data, -1, axis=1)
        current_vocal_stft_data[:, -1] = vocal_stft_frame
        
        # 更新波形数据 - 降采样到显示点数
        if len(current_audio_chunk) > 0:
            # 对音频块进行降采样
            step = len(current_audio_chunk) // WAVEFORM_DISPLAY_SAMPLES + 1
            waveform_resampled = current_audio_chunk[::step][:WAVEFORM_DISPLAY_SAMPLES]
            current_waveform_data = np.roll(current_waveform_data, -len(waveform_resampled))
            current_waveform_data[-len(waveform_resampled):] = waveform_resampled
        
        if len(current_vocal_chunk) > 0:
            step = len(current_vocal_chunk) // WAVEFORM_DISPLAY_SAMPLES + 1
            vocal_waveform_resampled = current_vocal_chunk[::step][:WAVEFORM_DISPLAY_SAMPLES]
            current_vocal_waveform_data = np.roll(current_vocal_waveform_data, -len(vocal_waveform_resampled))
            current_vocal_waveform_data[-len(vocal_waveform_resampled):] = vocal_waveform_resampled
    
    def update_plots(self):
        """更新图表"""
        try:
            # 更新STFT图
            self.im_stft.set_data(current_stft_data)
            self.im_vocal_stft.set_data(current_vocal_stft_data)
            
            # 更新波形图
            time_axis = np.linspace(0, STFT_DISPLAY_FRAMES*CHUNK/SR, WAVEFORM_DISPLAY_SAMPLES)
            self.line_waveform.set_data(time_axis, current_waveform_data)
            self.line_vocal_waveform.set_data(time_axis, current_vocal_waveform_data)
            
            # 自动调整波形Y轴范围
            if np.max(np.abs(current_waveform_data)) > 0:
                self.ax_waveform.set_ylim(-1.2, 1.2)
            if np.max(np.abs(current_vocal_waveform_data)) > 0:
                self.ax_vocal_waveform.set_ylim(-1.2, 1.2)
            
            self.canvas.draw()
        except:
            pass
    
    def update_progress(self):
        """更新进度"""
        if audio_data is not None and not self.is_seeking:
            progress = int((play_position / len(audio_data)) * 1000)
            self.progress_slider.setValue(progress)
            self.update_time_display()
    
    def on_progress_pressed(self):
        """开始拖动进度条"""
        self.is_seeking = True
    
    def on_progress_released(self):
        """释放进度条 - 跳转到指定位置"""
        global play_position
        if audio_data is not None:
            # 计算新的播放位置
            progress_percent = self.progress_slider.value() / 1000.0
            new_position = int(len(audio_data) * progress_percent)
            # 对齐到CHUNK边界
            play_position = (new_position // CHUNK) * CHUNK
            # 更新当前音频块
            if play_position + CHUNK < len(audio_data):
                global current_audio_chunk, current_vocal_chunk
                current_audio_chunk = audio_data[play_position:play_position+CHUNK]
                current_vocal_chunk = vocal_data[play_position:play_position+CHUNK]
            self.update_time_display()
        self.is_seeking = False
    
    def on_progress_changed(self, value):
        """进度条值改变 - 更新时间显示"""
        if audio_data is not None and self.is_seeking:
            progress_percent = value / 1000.0
            current_time = (len(audio_data) * progress_percent) / SR
            total_time = len(audio_data) / SR
            self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(total_time)}")
    
    def update_time_display(self):
        """更新时间显示"""
        if audio_data is not None:
            current_time = play_position / SR
            total_time = len(audio_data) / SR
            self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(total_time)}")
    
    def format_time(self, seconds):
        """格式化时间"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def show_toast(self, message, msg_type="info"):
        """显示Toast提示"""
        # 简化的Toast实现
        from PyQt5.QtWidgets import QMessageBox
        
        if msg_type == "success":
            QMessageBox.information(self, "提示", message)
        elif msg_type == "error":
            QMessageBox.critical(self, "错误", message)
        else:
            QMessageBox.information(self, "提示", message)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        global is_playing
        is_playing = False
        event.accept()


class WorkerThread(QThread):
    """通用工作线程"""
    finished_signal = pyqtSignal(object)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished_signal.emit(result)
        except Exception as e:
            self.finished_signal.emit(e)


def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 设置调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS['bg_light']))
    palette.setColor(QPalette.WindowText, QColor(COLORS['text_dark']))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(236, 240, 241))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS['text_dark']))
    palette.setColor(QPalette.Text, QColor(COLORS['text_dark']))
    palette.setColor(QPalette.Button, QColor(COLORS['primary']))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(COLORS['danger']))
    palette.setColor(QPalette.Highlight, QColor(COLORS['primary']))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = AudioProcessor()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()