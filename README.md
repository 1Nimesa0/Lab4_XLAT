# BÀI THỰC HÀNH LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)
### HỌC PHẦN: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI (CSE457)
**Trường Đại học Thủy Lợi • Khoa Công nghệ Thông tin • Bộ môn Trí tuệ Nhân tạo**

---

## 📌 Tổng quan Dự án
Dự án Lab 4 nghiên cứu và hiện thực hóa toàn diện 3 thế hệ công nghệ tổng hợp tiếng nói (Text-to-Speech):
1. **Concatenative TTS & Unit Selection:** Ghép nối đơn vị từ miền hẹp (*Case study: Thông báo xếp hàng*), tối ưu hóa chuỗi ứng viên bằng thuật toán Quy hoạch động Viterbi, và xử lý triệt tiêu tiếng click biên nối bằng Cross-fade.
2. **HMM-based Parametric Speech Synthesis (HTS):** Mô hình hóa tham số âm học thống kê với 3-state HMM, thuật toán sinh tham số liên tục MLPG (Maximum Likelihood Parameter Generation) với ràng buộc Delta, và tổng hợp sóng âm qua bộ lọc cộng hưởng Formants.
3. **Pretrained Neural TTS:** Mô hình học sâu End-to-End hai giai đoạn (*Acoustic Model + Neural Vocoder*) chạy inference offline, đo độ trễ suy luận (Latency) và so sánh chất lượng tự nhiên.

---

## 📂 Cấu trúc Thư mục và Tệp tin

```
d:\Lab4\
├── lab4_data/
│   ├── units/               # 48 file âm thanh WAV mono 16kHz (16 từ vựng x 3 candidates)
│   ├── neural_cache/        # Bộ nhớ đệm âm thanh offline cho Neural TTS
│   └── metadata.csv         # Bảng thông số: nhãn, thời lượng, RMS, candidate, file path
├── audio_output/            # Thư mục lưu các file âm thanh kết quả xuất xưởng
│   ├── raw_concat.wav       # Ghép thô candidate 01 (xfade = 0ms)
│   ├── best_concat.wav      # Ghép tối ưu Viterbi + Cross-fade 15ms
│   ├── hmm_tts.wav          # Tổng hợp tham số HMM chuẩn
│   ├── hmm_tts_dur0.8.wav   # HMM can thiệp thời lượng (0.8x nhanh)
│   ├── hmm_tts_f0_1.1.wav   # HMM can thiệp cao độ (1.1x cao)
│   ├── neural_tts.wav       # Tổng hợp Neural TTS câu 1
│   ├── neural_tts_sent2.wav # Tổng hợp Neural TTS câu 2
│   ├── neural_tts_sent3.wav # Tổng hợp Neural TTS câu 3
│   ├── concat_xfade_0ms.wav # Thí nghiệm crossfade 0ms
│   ├── concat_xfade_5ms.wav # Thí nghiệm crossfade 5ms
│   ├── concat_xfade_15ms.wav# Thí nghiệm crossfade 15ms
│   ├── concat_xfade_30ms.wav# Thí nghiệm crossfade 30ms
│   ├── speed_0.8x.wav       # Thí nghiệm kéo dài thời gian (WSOLA)
│   ├── speed_1.2x.wav       # Thí nghiệm nén thời gian (WSOLA)
│   ├── pitch_shift_minus2.wav # Thí nghiệm hạ cao độ (-2 semitones)
│   └── pitch_shift_plus2.wav  # Thí nghiệm tăng cao độ (+2 semitones)
├── figures/                 # Thư mục chứa 6 đồ thị học thuật chất lượng cao (300 DPI)
│   ├── 01_inventory_stats.png                   # Histogram thời lượng và năng lượng RMS
│   ├── 02_spectrogram_boundaries_comparison.png # Waveform & Spectrogram trước/sau Cross-fade
│   ├── 03_unit_selection_trellis.png            # Lưới ứng viên Trellis và đường đi Viterbi
│   ├── 04_hmm_parameter_trajectories.png        # Quỹ đạo F0 contour và Formants F1/F2/F3
│   ├── 05_tts_paradigms_comparison.png          # So sánh Waveform & Spectrogram 3 phương pháp
│   └── 06_prosody_modifications.png             # Đo kiểm Np (F0 = Fs/Np) và Spectrogram pitch-shift
├── speech_dsp_utils.py      # Thư viện xử lý tín hiệu: RMS, OLA, Cross-fade, ACF F0, Pitch/Time mod
├── unit_selection.py        # Text Normalization miền hẹp, Target/Join Cost, Thuật toán Viterbi Search
├── hmm_tts_backend.py       # Backend tổng hợp HMM, MLPG Trajectory Generator, Formant Resonators
├── neural_tts_backend.py    # Backend Neural TTS, trích xuất Mel-spectrogram, đo Latency
├── generate_inventory.py    # Script tự động tạo bộ Unit Inventory 48 files chuẩn
├── Lab04.py                 # File thực thi chính chạy toàn bộ quy trình từ A đến G
├── Lab04.ipynb              # Jupyter Notebook có đầy đủ kết quả thực thi, đồ thị và audio player
├── ratings.csv              # Bảng đánh giá kiểm thử nghe chủ quan (Blind Listening Test MOS)
├── REPORT_LAB04.md          # Báo cáo học thuật chi tiết 5-8 trang, giải đáp đầy đủ 10 câu hỏi
└── README.md                # Tài liệu hướng dẫn sử dụng và tổng kết dự án
```

---

## 🚀 Hướng dẫn Chạy thực nghiệm

### 1. Môi trường yêu cầu
Đảm bảo Python 3.10+ và các thư viện bắt buộc đã được cài đặt:
```bash
pip install numpy scipy matplotlib soundfile librosa pandas
```

### 2. Chạy toàn bộ Pipeline bằng Python Script
```bash
python Lab04.py
```
Toàn bộ file âm thanh sẽ được tự động xuất vào `audio_output/`, các biểu đồ lưu tại `figures/`, và bảng thống kê kết quả sẽ được in ra console.

### 3. Trải nghiệm trực quan với Jupyter Notebook
Mở file `Lab04.ipynb` trên JupyterLab hoặc VS Code để quan sát các ô code, biểu đồ phổ và nghe trực tiếp các file âm thanh thông qua widget `IPython.display.Audio`.

---

## 📊 Tóm tắt Kết quả So sánh 3 Phương pháp

| Phương pháp | Độ rõ (Intelligibility) | Tự nhiên (Naturalness) | Độ liền mạch (Continuity) | MOS Tổng hợp | Độ trễ (Latency) | Kích thước / Bộ nhớ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Concatenation (Unit Selection)** | 4.71 / 5.0 | 3.38 / 5.0 | 3.71 / 5.0 | **3.93 / 5.0** | **< 15 ms** | 10–50MB (Word units) |
| **HMM-based Parametric (HTS)** | **4.96 / 5.0** | 2.75 / 5.0 | 4.67 / 5.0 | **4.12 / 5.0** | < 25 ms | **1–5MB** (Gaussian params) |
| **Neural TTS (End-to-End)** | 4.92 / 5.0 | **4.62 / 5.0** | **4.75 / 5.0** | **4.77 / 5.0** | 100–300 ms | 50–200MB (Model weights) |

* Chi tiết lý thuyết, công thức toán học và giải đáp 10 câu hỏi của giảng viên xem tại:* [REPORT_LAB04.md](file:///d:/Lab4/REPORT_LAB04.md).
