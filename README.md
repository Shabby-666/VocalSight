# 人声分离器

基于 STFT 频谱分析和深度学习的人声分离工具，支持实时播放和频谱可视化。

## 功能特性

- 🎵 **人声提取**：使用频谱掩码、HPSS分离、基频检测等技术提取纯净人声
- 📊 **实时可视化**：显示原始音频和处理后音频的STFT频谱图和波形图
- 🎧 **实时播放**：支持原音频和提取后人声的实时播放对比
- 💾 **音频导出**：支持导出提取后的人声为WAV格式
- 🎯 **深度学习**：集成 torchfcpe 进行基频检测（自动回退到pyin）

## 技术栈

- **GUI**: PyQt5
- **音频处理**: librosa, soundfile, pyaudio
- **深度学习**: torch, torchfcpe
- **可视化**: matplotlib
- **信号处理**: scipy

## 安装

### 环境要求

- Python 3.8+
- Windows/Linux/MacOS

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行程序

```bash
python main_ui.py
```

### 使用步骤

1. **上传音频**：拖拽或点击上传音频文件（支持 MP3, WAV, FLAC, M4A, OGG）
2. **提取人声**：点击"提取人声"按钮，等待处理完成
3. **播放对比**：选择播放模式（原音频/提取后人声），点击播放按钮
4. **导出音频**：点击"导出音频"按钮保存提取后的人声

## 处理流程

1. **频谱分析**：计算STFT频谱，使用4096点FFT
2. **频谱掩码**：保留250-2800Hz人声频段
3. **谱减法降噪**：使用非人声区域作为噪声参考
4. **谱平坦度分析**：区分谐波和噪声
5. **HPSS分离**：分离谐波和打击乐成分
6. **基频检测**：使用torchfcpe检测人声基频
7. **瞬态检测**：检测并处理瞬态噪声
8. **EQ优化**：增强2-2.8kHz频段
9. **高频叠加**：从原音频提取4-7kHz高频叠加到人声

## 构建EXE

```bash
pyinstaller --name="人声分离器" --windowed --onefile main_ui.py
```

构建完成后，EXE文件位于 `dist/人声分离器.exe`

## 项目结构

```
.
├── main_ui.py          # 主程序
├── requirements.txt    # 依赖列表
├── README.md          # 项目说明
├── 最后的备份.py      # 备份文件
└── dist/              # 构建输出目录
    └── 人声分离器.exe  # 可执行文件
```

## 许可证

MIT License

## 致谢

- [librosa](https://librosa.org/) - 音频处理库
- [torchfcpe](https://github.com/CNChTu/FCPE) - 基频检测模型
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI框架
