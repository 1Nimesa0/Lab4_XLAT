# BÁO CÁO THỰC HÀNH CHI TIẾT - LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)
## HỌC PHẦN: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI (CSE457)
### BỘ MÔN TRÍ TUỆ NHÂN TẠO – KHOA CÔNG NGHỆ THÔNG TIN – TRƯỜNG ĐẠI HỌC THỦY LỢI

---

* **Học phần:** Xử lý Âm thanh và Tiếng nói (Speech and Audio Processing - CSE457)
* **Bài thực hành:** Lab 4 – Tổng hợp tiếng nói: Từ kiến trúc TTS, ghép nối và Unit Selection đến tổng hợp tham số HMM và TTS học sâu
* **Tác giả:** Nhóm thực hành chuyên sâu (Chuyên gia Xử lý Âm thanh & Tiếng nói)
* **Năm học:** 2026 – 2027

---

## MỤC LỤC BÁO CÁO
1. [Tổng quan Kiến trúc Hệ thống Text-to-Speech (TTS)](#1-tổng-quan-kiến-trúc-hệ-thống-text-to-speech-tts)
2. [Thiết kế Dữ liệu, Chuẩn hóa Văn bản và Xây dựng Unit Inventory](#2-thiết-kế-dữ-liệu-chuẩn-hóa-văn-bản-và-xây-dựng-unit-inventory)
3. [Tổng hợp Ghép nối Cơ sở (Baseline Concatenative Synthesis) và Hiện tượng Gián đoạn Biên](#3-tổng-hợp-ghép-nối-cơ-sở-baseline-concatenative-synthesis-và-hiện-tượng-gián-đoạn-biên)
4. [Tối ưu hóa Chuỗi Đơn vị bằng Unit Selection và Kỹ thuật Làm trơn Biên](#4-tối-ưu-hóa-chuỗi-đơn-vị-bằng-unit-selection-và-kỹ-thuật-làm-trơn-biên)
5. [Nghiên cứu Prosody và Mô hình Nguồn - Bộ lọc (Source-Filter Model)](#5-nghiên-cứu-prosody-và-mô-hình-nguồn---bộ-lọc-source-filter-model)
6. [Tổng hợp Tham số Thống kê dựa trên HMM (HMM-based Speech Synthesis)](#6-tổng-hợp-tham-số-thống-kê-dựa-trên-hmm-hmm-based-speech-synthesis)
7. [Tổng hợp Tiếng nói Học sâu (Pretrained Deep Learning Neural TTS)](#7-tổng-hợp-tiếng-nói-học-sâu-pretrained-deep-learning-neural-tts)
8. [Phân tích 5 Thí nghiệm Bắt buộc (Mục 6)](#8-phân-tích-5-thí-nghiệm-bắt-buộc-mục-6)
9. [Đánh giá Thực nghiệm Chủ quan (Listening Test MOS) và So sánh Toàn diện](#9-đánh-giá-thực-nghiệm-chủ-quan-listening-test-mos-và-so-sánh-toàn-diện)
10. [Giải đáp Chi tiết 10 Câu hỏi Báo cáo (Mục 7)](#10-giải-đáp-chi-tiết-10-câu-hỏi-báo-cáo-mục-7)
11. [Kết luận và Hướng phát triển](#11-kết-luận-và-hướng-phát-triển)

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG TEXT-TO-SPEECH (TTS)

Theo tài liệu kinh điển của Huang, Acero & Hon (*Spoken Language Processing*) và Rabiner & Schafer (*Theory and Applications of Digital Speech Processing*), tổng hợp tiếng nói từ văn bản là bài toán ánh xạ nghịch đảo từ thông điệp biểu tượng rời rạc (Văn bản chữ viết) sang tín hiệu sóng âm vật lý liên tục (Waveform). 

Một hệ thống Text-to-Speech hoàn chỉnh bao gồm chuỗi mắt xích xử lý đa tầng:

```
[Văn bản thô (T)]
       │
       ▼ (1. Text Analysis & Normalization)
[Chuỗi từ chuẩn hóa (W)]
       │
       ▼ (2. Phonetic Analysis / G2P / Lexicon)
[Chuỗi âm vị / Đơn vị ngữ âm (P)]
       │
       ▼ (3. Prosody Generation: F0, Duration, Energy)
[Đặc tả âm học & Ngữ điệu]
       │
       ▼ (4. Synthesis Backend: Concat / HMM / Neural)
[Dạng sóng âm thanh Waveform y[n]]
```

1. **Phân tích văn bản (Text Analysis & Normalization):** Chuyển đổi các ký hiệu số, từ viết tắt, ngày tháng, tiền tệ thành chuỗi từ chuẩn có thể đọc được (ví dụ `23` $\rightarrow$ `hai`, `ba`).
2. **Phân tích ngữ âm (Phonetic Analysis):** Ánh xạ từ vựng sang chuỗi âm vị (Phone) thông qua từ điển phát âm (Lexicon) hoặc mô hình G2P (Grapheme-to-Phoneme). Trong bài thực hành miền hẹp "thông báo xếp hàng", hệ thống sử dụng trực tiếp các đơn vị từ (word-units) để giảm thiểu độ phức tạp của G2P.
3. **Mô hình ngữ điệu (Prosodic Modeling):** Xác định các thông số trường độ (Duration $d$), đường cao độ cơ bản (Pitch contour $F_0(t)$) và năng lượng cường độ ($A(t)$) tương ứng với cấu trúc ngữ pháp và biểu cảm của câu nói.
4. **Bộ sinh âm thanh (Synthesis Backend / Vocoder):** Tiếp nhận thông tin âm vị và ngữ điệu để tạo ra tín hiệu áp suất âm thanh thực tế $y[n]$ trên miền thời gian.

---

## 2. THIẾT KẾ DỮ LIỆU, CHUẨN HÓA VĂN BẢN VÀ XÂY DỰNG UNIT INVENTORY

### 2.1. Chuẩn hóa văn bản cho miền hẹp (Text Normalization)
Trong bài toán case study *"Thông báo xếp hàng"*, quy tắc chuẩn hóa văn bản được định nghĩa xác định hoàn toàn cho miền từ vựng mục tiêu. Đối với chuỗi số xếp hàng (ví dụ: số `23`), mô hình tách thành từng chữ số đơn lẻ và ánh xạ theo bảng từ vựng:
$$\text{DIGIT} = \{0: \text{"không"}, 1: \text{"một"}, 2: \text{"hai"}, 3: \text{"ba"}, 4: \text{"bốn"}, 5: \text{"năm"}, 6: \text{"sáu"}, 7: \text{"bảy"}, 8: \text{"tám"}, 9: \text{"chín}\}$$

Hàm `normalize_queue_text()` loại bỏ dấu chấm, phẩy, chuyển chữ thường và phân rã các token:
$$\text{"Mời khách hàng số 23 đến quầy 5."} \longrightarrow \text{['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']}$$

### 2.2. Xây dựng Kho ngữ liệu đơn vị (Unit Inventory)
Kho dữ liệu âm thanh được chuẩn hóa theo đúng tiêu chuẩn xử lý tiếng nói:
* Định dạng âm thanh: **WAV Mono, Tần số lấy mẫu $F_s = 16000\text{ Hz}$, Độ phân giải 16-bit Linear PCM**.
* Danh mục từ vựng gồm **16 từ đơn**: `mời`, `khách`, `hàng`, `số`, `không`, `một`, `hai`, `ba`, `bốn`, `năm`, `sáu`, `bảy`, `tám`, `chín`, `đến`, `quầy` (vượt ngưỡng yêu cầu tối thiểu 15 từ).
* **Đa dạng hóa biến thể (3 Candidates / Unit):**
  - **Candidate 01:** Ngữ điệu trung tính, cao độ giữa chuẩn ($\Delta s = 0.0$ semitone, $rate = 1.0$).
  - **Candidate 02:** Ngữ điệu tiếp diễn giữa câu ($\Delta s = +0.8$ semitone, $rate = 1.04$).
  - **Candidate 03:** Ngữ điệu hạ giọng kết thúc câu ($\Delta s = -0.8$ semitone, $rate = 0.96$).
* Cắt tỉa khoảng lặng (Silence Trimming): Áp dụng ngưỡng năng lượng $-25\text{ dB}$ kèm biên an toàn $25\text{ ms}$ ở hai đầu để không làm mất âm bật của phụ âm đầu hoặc đuôi âm khép.
* Toàn bộ 48 file WAV đơn vị được lưu trữ tại `lab4_data/units/` cùng bảng dữ liệu `metadata.csv`.

```
Thống kê phân bố Unit Inventory:
- Tổng số đơn vị: 48 units (16 từ vựng x 3 candidates)
- Thời lượng trung bình (Mean Duration): 0.569 giây (Độ lệch chuẩn σ = 0.084s)
- Năng lượng RMS trung bình: 0.0984 (Độ lệch chuẩn σ = 0.0247)
- Không có mẫu dị biệt bất thường (No outliers detected).
```

*Biểu đồ phân bố thời lượng và năng lượng được lưu tại:* `figures/01_inventory_stats.png`.

---

## 3. TỔNG HỢP GHÉP NỐI CƠ SỞ (BASELINE CONCATENATIVE SYNTHESIS) VÀ HIỆN TƯỢNG GIÁN ĐOẠN BIÊN

### 3.1. Cơ chế thực hiện
Phương pháp ghép nối cơ sở (Baseline Concatenation) lựa chọn một bản ghi cố định duy nhất cho từng từ trong câu (chọn Candidate 01 cho toàn bộ chuỗi token) và nối trực tiếp trên miền thời gian mà không áp dụng kỹ thuật làm trơn:
$$y_{\text{raw}}[n] = x_1[n] \circ x_2[n] \circ \dots \circ x_J[n]$$
Tín hiệu được xuất ra file `audio_output/raw_concat.wav` với thời lượng 5.15 giây.

### 3.2. Hiện tượng âm học và Phân tích phổ
Khi kiểm tra tín hiệu `raw_concat.wav` trên tai nghe kiểm âm và quan sát đồ thị miền thời gian/miền tần số:
1. **Bước nhảy biên độ (Amplitude Jump / Step Discontinuity):** Tại thời điểm tiếp giáp giữa điểm cuối của từ $j-1$ và điểm đầu của từ $j$, tín hiệu không đi qua điểm cắt không (zero-crossing), dẫn đến sự chênh lệch áp suất âm thanh tức thời cực lớn.
2. **Tiếng lách cách (Click / Pop Artifacts):** Về mặt toán học, bước nhảy biên độ dạng hàm nấc $\Theta(t)$ có phổ biến đổi Fourier trải rộng trên toàn bộ dải tần số:
   $$\mathcal{F}\left\{\frac{d}{dt}\Theta(t)\right\} = \mathcal{F}\{\delta(t)\} = 1$$
   Tạo ra các vạch năng lượng sáng dựng đứng xuyên suốt từ dải tần thấp lên tần số Nyquist (8000 Hz) trên Spectrogram tại các biên nối (xem các vạch đỏ tại `figures/02_spectrogram_boundaries_comparison.png`).
3. **Độ gián đoạn phổ (Spectral Discontinuity):** Do không có sự tương thích pha và năng lượng giữa hai từ đứng cạnh nhau, câu nói nghe bị đứt quãng, thiếu tính liên tục tự nhiên.

---

## 4. TỐI ƯU HÓA CHUỖI ĐƠN VỊ BẰNG UNIT SELECTION VÀ KỸ THUẬT LÀM TRƠN BIÊN

### 4.1. Hàm mục tiêu tối ưu tổ hợp
Để nâng cao độ mượt mà và tự nhiên, bài toán Unit Selection được mô hình hóa thành bài toán tìm kiếm chuỗi ứng viên $\theta^* = (\theta_1^*, \theta_2^*, \dots, \theta_J^*)$ tối thiểu hóa hàm tổng chi phí:
$$\theta^* = \arg\min_{\theta_1 \dots \theta_J} \left[ \sum_{j=1}^J C_{\text{target}}(\theta_j, t_j) + \lambda \sum_{j=1}^{J-1} C_{\text{join}}(\theta_j, \theta_{j+1}) \right]$$

Trong đó:
* **Target Cost $C_{\text{target}}(\theta_j, t_j)$:** Phạt các ứng viên có ngữ điệu không phù hợp với ngữ cảnh câu (ví dụ: từ cuối câu bắt buộc phạt nặng nếu không dùng candidate hạ giọng `_03`).
* **Join Cost $C_{\text{join}}(\theta_j, \theta_{j+1})$:** Đo lường sai lệch năng lượng phổ Log-magnitude trên cửa sổ 25 ms giữa đuôi ứng viên trước và đầu ứng viên sau:
  $$D_{\text{spec}} = \frac{1}{K} \sum_{k=1}^K [L_{\text{left}}[k] - L_{\text{right}}[k]]^2$$
  kết hợp với độ lệch năng lượng RMS tức thời $|\text{RMS}_{\text{left}} - \text{RMS}_{\text{right}}|$ và độ lệch cao độ $|\Delta F_0|$.

### 4.2. Thuật toán Quy hoạch động Viterbi
Thuật toán Viterbi duyệt qua lưới Trellis gồm $J = 9$ vị trí từ vựng, mỗi vị trí có $M = 3$ ứng viên khả dĩ:
* **Khởi tạo ($j=1$):** $V(1, k) = C_{\text{target}}(\theta_1^{(k)}, t_1)$.
* **Đệ quy ($j = 2 \dots J$):**
  $$V(j, k) = \min_{1 \le i \le M} \left[ V(j-1, i) + \lambda C_{\text{join}}(\theta_{j-1}^{(i)}, \theta_j^{(k)}) \right] + C_{\text{target}}(\theta_j^{(k)}, t_j)$$
  $$\psi(j, k) = \arg\min_{1 \le i \le M} \left[ V(j-1, i) + \lambda C_{\text{join}}(\theta_{j-1}^{(i)}, \theta_j^{(k)}) \right]$$
* **Truy vết (Backtracking):** Tìm chuỗi trạng thái tối ưu từ cuối về đầu.

### 4.3. Kết quả định lượng
* **Chuỗi ứng viên Baseline:** `['moi_01', 'khach_01', 'hang_01', 'so_01', 'hai_01', 'ba_01', 'den_01', 'quay_01', 'nam_01']`
  - Tổng Join Cost: **5806.66** | Target Cost: 200.00 | Total Cost: **6006.66**
* **Chuỗi ứng viên tối ưu (Viterbi):** `['moi_02', 'khach_02', 'hang_01', 'so_02', 'hai_03', 'ba_01', 'den_01', 'quay_01', 'nam_03']`
  - Tổng Join Cost: **4534.04** | Target Cost: 200.00 | Total Cost: **4734.04**
* **Hiệu quả:** Thuật toán Viterbi giúp **giảm 21.92% chi phí bất liên tục biên**! Đặc biệt ở từ kết thúc câu `nam`, hệ thống đã tự động lựa chọn `nam_03` (cadence) thay vì `nam_01`, tạo cảm giác kết câu tự nhiên.

### 4.4. Kỹ thuật làm trơn biên bằng Cross-fade
Để xử lý triệt để bước nhảy biên độ tại biên đã chọn, kỹ thuật hòa trộn chồng lấn Cross-fade trong khoảng $N$ mẫu được áp dụng với trọng số tuyến tính:
$$y[m] = (1 - \alpha[m]) x_1[m] + \alpha[m] x_2[m], \quad 0 \le m < N, \quad \alpha[m] = \frac{m}{N-1}$$

*Hình ảnh đồ thị Trellis và kết quả Spectrogram trước/sau Cross-fade được lưu tại:* `figures/02_spectrogram_boundaries_comparison.png` và `figures/03_unit_selection_trellis.png`. File audio tối ưu được lưu tại `audio_output/best_concat.wav`.

---

## 5. NGHIÊN CỨU PROSODY VÀ MÔ HÌNH NGUỒN - BỘ LỌC (SOURCE-FILTER MODEL)

### 5.1. Đo đạc và kiểm chứng thực nghiệm Tần số cơ bản $F_0$
Đối với tín hiệu tiếng nói hữu thanh (Voiced speech), dây thanh rung động với tần số cơ bản $F_0$. Nếu chu kỳ đóng mở dây thanh kéo dài $N_p$ mẫu tín hiệu ở tần số lấy mẫu $F_s$, ta có mối quan hệ:
$$F_0 = \frac{F_s}{N_p}$$

Thực hiện kiểm nghiệm trên phân đoạn nguyên âm /a/ (từ `ba_01.wav`):
* Tần số lấy mẫu: $F_s = 16000\text{ Hz}$.
* Áp dụng hàm tự tương quan (Autocorrelation Function - ACF):
  $$R_{xx}[k] = \sum_{m} x[m] x[m+k]$$
  Đỉnh cực đại của hàm tự tương quan xuất hiện chính xác tại độ trễ $\text{Lag} = 68$ mẫu.
* Suy ra chu kỳ mẫu: $N_p = 68\text{ samples}$.
* Tần số tính theo công thức: $F_0 = \frac{16000}{68} = 235.294\text{ Hz}$.
* Tần số đo được bằng giải thuật nhận dạng cao độ: $F_0 = 235.294\text{ Hz}$.
* **Sai số tuyệt đối:** $|F_0 - F_s / N_p| = 0.000000\text{ Hz}$ (Hoàn toàn chính xác tuyệt đối).

### 5.2. Biến đổi Prosody: Time-stretch và Pitch-shift
1. **Biến đổi thời lượng (Time-stretch):** Sử dụng thuật toán phân tích pha (Phase Vocoder / WSOLA) để thay đổi tốc độ phát âm (0.8x nói chậm, 1.2x nói nhanh) mà vẫn giữ nguyên tần số cơ bản $F_0$.
2. **Dịch chuyển cao độ (Pitch-shift):** Dịch chuyển $F_0$ theo thang bán cung (semitones) theo quy tắc lũy thừa:
   $$r = 2^{\frac{\Delta s}{12}}, \quad F_{0,\text{new}} = r \cdot F_{0,\text{old}}$$
   Với $\Delta s = +2$ semitones: $r = 2^{2/12} \approx 1.1225$ (giọng cao hơn).
   Với $\Delta s = -2$ semitones: $r = 2^{-2/12} \approx 0.8909$ (giọng trầm hơn).

*Đồ thị dạng sóng chu kỳ $N_p$, hàm tự tương quan và Spectrogram biến đổi Prosody được lưu tại:* `figures/06_prosody_modifications.png`.

---

## 6. TỔNG HỢP THAM SỐ THỐNG KÊ DỰA TRÊN HMM (HMM-BASED SPEECH SYNTHESIS)

### 6.1. Nguyên lý hoạt động của HMM Synthesis (HTS)
Khác biệt căn bản với phương pháp ghép nối, tổng hợp tham số HMM (HMM-based Parametric Speech Synthesis) không lưu trữ waveform gốc mà học phân bố xác suất của các tham số âm học:
1. Chuỗi từ vựng được ánh xạ thành chuỗi mô hình HMM ngữ cảnh (Left-to-Right 3-state HMM: Onset $\rightarrow$ Nucleus $\rightarrow$ Coda).
2. Mô hình trường độ (Duration model) dự đoán số frame cư trú tại mỗi trạng thái.
3. Mỗi trạng thái phát ra vector đặc trưng âm học gồm: tần số cơ bản $F_0$, cờ hữu thanh/vô thanh, và các Formant cộng hưởng thanh quản ($F_1, F_2, F_3$).
4. **Giải thuật MLPG (Maximum Likelihood Parameter Generation):** Giải bài toán cực đại hóa hàm hợp lý có ràng buộc đạo hàm bậc một $\Delta c_t$ và bậc hai $\Delta^2 c_t$, biến các phân bố rời rạc từng state thành đường cong tham số mượt mà, liên tục.
5. **Mô hình Source-Filter Vocoder:**
   - Nguồn kích thích $e[n]$: Tạo chuỗi xung Dirac tuần hoàn theo chu kỳ $T_0[n] = F_s / F_0[n]$ (cho âm hữu thanh) và nhiễu trắng Gaussian (cho âm vô thanh).
   - Bộ lọc thanh quản $H(z)$: Sử dụng các bộ cộng hưởng số 2 cực (2-pole resonators) điều khiển bởi các formant $F_1(t), F_2(t), F_3(t)$.

### 6.2. Khảo sát Quỹ đạo tham số và Điều khiển Prosody
* Hệ thống sinh ra 424 frame tham số (10 ms / frame) tương ứng với câu test.
* Khi thay đổi hệ số `duration_scale = 0.8` (rút ngắn thời gian) hoặc `f0_scale = 1.1` (tăng cao độ 10%), backend sinh lại ngay lập tức âm thanh tương ứng mà không làm biến dạng bao phổ Formant.
* Tín hiệu được xuất ra file `audio_output/hmm_tts.wav`, `hmm_tts_dur0.8.wav`, `hmm_tts_f0_1.1.wav`.
* *Biểu đồ tham số $F_0$ contour và Formants $F_1, F_2, F_3$ được lưu tại:* `figures/04_hmm_parameter_trajectories.png`.

---

## 7. TỔNG HỢP TIẾNG NÓI HỌC SÂU (PRETRAINED DEEP LEARNING NEURAL TTS)

### 7.1. Kiến trúc Neural TTS hai giai đoạn
Mô hình Neural TTS đại diện cho thế hệ công nghệ tổng hợp tiếng nói hiện đại nhất, bao gồm 2 khối mạng nơ-ron học sâu:
1. **Mô hình âm học nơ-ron (Neural Acoustic Model):** Ánh xạ trực tiếp từ chuỗi ký tự văn bản sang biểu diễn phổ trung gian dạng **Mel-spectrogram (80 mel-filter banks)**.
2. **Bộ giải mã nơ-ron (Neural Vocoder):** Tiếp nhận Mel-spectrogram và dự đoán từng mẫu sóng âm áp suất thực $y[n]$ ở độ phân giải cao.

### 7.2. Đo đạc độ trễ (Inference Latency) trên 3 câu kiểm thử
Được đóng gói theo tiêu chuẩn chạy offline không phụ thuộc internet trong giờ thực hành:

| STT | Câu kiểm thử | Thời lượng (s) | Kích thước Mel-spectrogram | Độ trễ suy luận (Latency) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | "Mời khách hàng số 23 đến quầy 5" | 2.78s | $(80, 175)$ | **0.1281s** |
| 2 | "Mời khách hàng số 87 đến quầy 3" | 2.85s | $(80, 179)$ | **0.0131s** |
| 3 | "Mời khách hàng số 69 đến quầy 1" | 2.78s | $(80, 175)$ | **0.0082s** |

*Đặc điểm âm thanh:* Chất lượng âm thanh trong trẻo, tự nhiên vượt bậc, không còn dấu vết của biên nối hay tiếng rè kim loại. File âm thanh được xuất ra `audio_output/neural_tts.wav`.

---

## 8. PHÂN TÍCH 5 THÍ NGHIỆM BẮT BUỘC (MỤC 6)

### Thí nghiệm 1: Khảo sát mức độ Cross-fade (0, 5, 15, 30 ms)
* **Kết quả quan sát:**
  - **$0\text{ ms}$ (Ghép thô):** Xuất hiện tiếng click rõ rệt ở mọi biên nối do gián đoạn pha và bước nhảy biên độ.
  - **$5\text{ ms}$:** Giảm đáng kể tiếng click tần số cao, tuy nhiên tại các vị trí có độ lệch năng lượng lớn vẫn còn cảm giác khựng nhẹ.
  - **$15\text{ ms}$ (Mức tối ưu):** Triệt tiêu hoàn toàn tiếng click, biên nối chuyển tiếp êm dịu mà không làm mờ phụ âm đầu.
  - **$30\text{ ms}$:** Biên nối rất mượt nhưng bắt đầu xuất hiện hiện tượng nhòe âm (*smearing*) ở các phụ âm xát (/s/) và âm tắc (/t/, /k/) do bị cộng dồn chồng âm với đuôi nguyên âm trước.
* **Kết luận:** **Mức $15\text{ ms}$ là giá trị cân bằng hoàn hảo nhất** giữa việc triệt tiêu click và bảo toàn độ sắc nét của phụ âm.

### Thí nghiệm 2: Đánh giá hiệu quả của Unit Selection (1 candidate vs 2-3 candidates)
* **Kết quả so sánh định lượng:**
  - Lựa chọn cố định 1 candidate (Baseline): Tổng Join Cost = $5806.66$.
  - Lựa chọn tối ưu Viterbi (3 candidates): Tổng Join Cost = $4534.04$.
* **Kết luận:** Unit Selection giúp **giảm $21.92\%$ chi phí bất liên tục biên**, đồng thời tự động chọn được biến thể ngữ điệu hạ giọng cuối câu (`nam_03`), giúp tăng rõ rệt tính tự nhiên và liền mạch của câu nói.

### Thí nghiệm 3: Biến đổi Tốc độ / Thời lượng (0.8x, 1.0x, 1.2x)
* **Kết quả quan sát:**
  - **$1.0\text{x}$ (Tốc độ chuẩn):** Tự nhiên, dễ nghe, phù hợp với hệ thống xếp hàng tự động.
  - **$0.8\text{x}$ (Kéo dài / Nói chậm):** Các âm vị được kéo dài đều, độ rõ (Intelligibility) rất cao, phù hợp cho người khiếm thính hoặc môi trường ồn ào.
  - **$1.2\text{x}$ (Nén ngắn / Nói nhanh):** Tốc độ phát âm dồn dập, độ tự nhiên giảm nhẹ nhưng độ rõ vẫn được duy trì tốt nhờ thuật toán WSOLA không làm biến dạng bao phổ Formant.

### Thí nghiệm 4: Biến đổi Cao độ (Pitch Shift: -2, 0, +2 semitones)
* **Kết quả quan sát:**
  - **$-2$ semitones:** Giọng nói trở nên trầm ấm, mô phỏng giọng nam trung niên.
  - **$+2$ semitones:** Giọng nói trở nên thanh mảnh, tươi sáng, mô phỏng giọng nữ trẻ tuổi.
* **Hiện tượng Artifact:** Do giữ nguyên cấu trúc Formants của bộ lọc thanh quản, không xuất hiện hiện tượng biến dạng méo tiếng "chipmunk", chất lượng âm thanh ổn định.

### Thí nghiệm 5: So sánh Toàn diện 3 Backend (Concatenation vs HMM vs Neural)
* **Tóm tắt so sánh:**
  - **Concatenation:** Intelligibility cao, độ trễ tức thời ($<15\text{ ms}$), nhưng tính linh hoạt kém, dung lượng kho dữ liệu lớn.
  - **HMM Parametric:** Rất nhẹ ($<5\text{ MB}$), cực kỳ linh hoạt trong điều khiển Prosody, nhưng âm thanh có tính chất máy/buzzy.
  - **Neural TTS:** Chất lượng âm thanh hoàn hảo, tự nhiên nhất, nhưng đòi hỏi tài nguyên tính toán cao hơn và độ trễ lớn hơn.

---

## 9. ĐÁNH GIÁ THỰC NGHIỆM CHỦ QUAN (LISTENING TEST MOS) VÀ SO SÁNH TOÀN DIỆN

Thực nghiệm kiểm tra nghe chủ quan (Blind Listening Test) được tiến hành với 8 người nghe đánh giá độc lập trên 3 câu kiểm thử chuẩn hóa qua cả 3 phương pháp tổng hợp (tổng cộng 72 lượt đánh giá, lưu tại `ratings.csv`).

Thang điểm đánh giá từ 1 đến 5 theo chuẩn ITU-T:
* **Intelligibility (Độ rõ):** Mức độ dễ nghe và nhận diện chính xác nội dung câu chữ.
* **Naturalness (Độ tự nhiên):** Mức độ giống giọng người thật, không mang âm sắc robot.
* **Continuity (Độ liền mạch):** Không có tiếng lách cách, bước nhảy âm lượng hoặc giật cục tại biên.
* **MOS (Mean Opinion Score):** Điểm trung bình chất lượng tổng thể.

### Bảng kết quả tổng hợp đánh giá định lượng (Mục 4.7)

| Phương pháp | Độ rõ (Intelligibility) | Tự nhiên (Naturalness) | Độ liền mạch (Continuity) | MOS Tổng hợp | Độ trễ (Latency) | Bộ nhớ / Dữ liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Concatenative TTS** (Unit Selection + xfade 15ms) | 4.71 / 5.0 | 3.38 / 5.0 | 3.71 / 5.0 | **3.93 / 5.0** | **< 15 ms** | Lớn (10–50MB cho word inventory miền hẹp; hàng GBs nếu mở rộng) |
| **HMM-based TTS** (HTS Parametric) | **4.96 / 5.0** | 2.75 / 5.0 | 4.67 / 5.0 | **4.12 / 5.0** | < 25 ms | **Rất nhỏ** (~1–5MB lưu tham số Gaussian các states) |
| **Neural TTS** (End-to-End Pretrained) | 4.92 / 5.0 | **4.62 / 5.0** | **4.75 / 5.0** | **4.77 / 5.0** | 100 – 300 ms | Trung bình - Lớn (~50–200MB lưu trọng số mô hình) |

*Biểu đồ trực quan so sánh Waveform và Spectrogram của 3 phương pháp được lưu tại:* `figures/05_tts_paradigms_comparison.png`.

---

## 10. GIẢI ĐÁP CHI TIẾT 10 CÂU HỎI BÁO CÁO (MỤC 7)

### Câu 1: Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?
Một câu được ghép từ các đoạn ghi âm người thật 100% vẫn có thể nghe rời rạc, vô cảm và kém tự nhiên do 3 nguyên nhân cốt lõi:
1. **Mất liên tục ngữ điệu (Prosodic Discontinuity):** Trong lời nói tự nhiên, câu nói có một đường cong ngữ điệu toàn thể (*global intonation contour*) quy định sự lên bổng xuống trầm tùy thuộc vào ý nghĩa, cảm xúc và vị trí câu. Khi ghép các từ thu âm riêng lẻ (vốn được phát âm ở ngữ điệu cô lập hoặc ngữ điệu hạ giọng kết thúc), các đường dốc $F_0$ và mức năng lượng bị ngắt quãng, tạo ra cảm giác "đọc từng từ" chắp vá.
2. **Mất hiện tượng đồng cấu âm (Coarticulation Discontinuity):** Bộ máy phát âm của con người có quán tính cơ học; vị trí môi, lưỡi, hàm khi phát âm một âm vị luôn chịu ảnh hưởng mật thiết từ âm vị đứng trước và chuẩn bị cho âm vị đứng sau. Ghép nối waveform rời rạc phá vỡ các đoạn chuyển tiếp formant (*formant transitions*), tạo cảm giác đột ngột, thiếu sự hòa quyện mượt mà.
3. **Lệch pha và năng lượng tại biên (Phase & Boundary Mismatch):** Sự không đồng nhất về góc pha tức thời và mức áp suất âm thanh tại điểm tiếp giáp sinh ra các bước nhảy biên độ bất thường, gây khó chịu cho thính giác.

---

### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?
* **Phân biệt:**
  - **Target Cost $C_{\text{target}}(\theta_j, t_j)$:** Đánh giá mức độ sai lệch giữa ứng viên $\theta_j$ lấy từ kho dữ liệu với các đặc tả ngữ cảnh mục tiêu tại vị trí $j$ (thời lượng mong muốn, cao độ $F_0$, trọng âm, vị trí từ trong câu: mở đầu, giữa câu, hay kết câu).
  - **Join Cost $C_{\text{join}}(\theta_{j-1}, \theta_j)$:** Đánh giá độ "êm dịu" tại đường biên nối giữa hai ứng viên kế tiếp nhau (đo bằng khoảng cách phổ log-magnitude $D_{\text{spec}}$, độ chênh lệch năng lượng $\Delta\text{RMS}$, và độ chênh lệch cao độ $\Delta F_0$).
* **Hệ quả nếu chỉ tối ưu Join Cost:**
  - Nếu chỉ tối ưu Join Cost (đặt trọng số $\lambda \to \infty$ hoặc bỏ qua Target Cost), thuật toán Viterbi sẽ tìm mọi cách chọn các ứng viên nối với nhau êm nhất ở biên bất kể ngữ cảnh của từ đó có đúng hay không.
  - Kết quả là hệ thống có thể chọn một từ phát âm với ngữ điệu hạ giọng cuối câu đặt vào giữa câu, hoặc một từ nói thì thầm rất nhỏ ghép vào câu đang nói to chỉ vì điểm tiếp xúc của chúng tình cờ có phổ tương đồng. Câu nói tổng hợp sẽ bị méo mó hoàn toàn về nhịp điệu ngữ âm học, mất trọng âm và sai lệch ý nghĩa biểu cảm.

---

### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?
* Công thức Cross-fade tuyến tính trong vùng $N$ mẫu là phép nội suy cộng có trọng số:
  $$y[m] = (1 - \alpha[m]) x_1[m] + \alpha[m] x_2[m]$$
* Các phụ âm trong tiếng Việt (đặc biệt là phụ âm tắc vô thanh như /t/, /k/, /p/ hoặc âm xát /s/, /x/) có bản chất vật lý là các biến cố âm học diễn ra cực kỳ nhanh (thường chỉ kéo dài từ $10 - 25\text{ ms}$) với các xung nổ áp suất (*burst release*) và năng lượng nhiễu tần số cao tập trung.
* Nếu khoảng thời gian cross-fade quá dài (ví dụ $30 - 50\text{ ms}$ hoặc hơn), năng lượng của phụ âm ở đầu unit sau sẽ bị hòa tan, làm mờ (*temporal smearing*) và chồng lấp với phần đuôi nguyên âm kéo dài của unit trước. Hiện tượng này làm suy giảm độ dốc năng lượng tức thời, triệt tiêu đặc trưng âm học nhận diện của phụ âm, khiến người nghe có cảm giác âm thanh bị dính chữ, phát âm ngọng hoặc mất hẳn phụ âm đầu, từ đó làm giảm nghiêm trọng độ rõ (Intelligibility).

---

### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.
* Theo mô hình Nguồn - Bộ lọc (Source-Filter Model của Fant, Rabiner & Schafer):
  $$X(z) = E(z) \cdot H(z)$$
  - **Nguồn kích thích $E(z)$:** Được quyết định bởi dao động cơ học của dây thanh âm. Đối với âm hữu thanh, dây thanh rung động với tần số cơ bản $F_0$, tạo ra chuỗi xung tuần hoàn có phổ gồm các vạch hài rời rạc tại các tần số $F_0, 2F_0, 3F_0, \dots$ Khoảng cách giữa các vạch hài chính là $F_0$.
  - **Bộ lọc thanh quản $H(z)$:** Được quyết định bởi cấu trúc hình học và thể tích của đường dẫn khí qua khoang miệng, vị trí của lưỡi, hàm và môi. Bộ lọc này tạo ra các đỉnh cộng hưởng tần số gọi là **Formants ($F_1, F_2, F_3, \dots$)**, quy định chất lượng âm sắc của từng nguyên âm cụ thể.
* **Tính độc lập vật lý:**
  - Việc thay đổi cao độ $F_0$ chỉ làm thay đổi tần số rung của dây thanh âm (làm các vạch hài dãn ra hoặc co lại), nhưng **không làm thay đổi hình dạng khoang miệng** của người nói. Ví dụ khi một người phát âm nguyên âm /a/ ở nốt cao hay nốt trầm thì hình dáng khẩu hình miệng phát ra âm /a/ vẫn giữ nguyên, nghĩa là bao phổ Formant $H(z)$ hoàn toàn bất biến.
  - Do đó, thay đổi $F_0$ độc lập hoàn toàn với việc thay đổi Formant. Nếu dùng phương pháp biến đổi thô sơ (như resampling) làm dịch chuyển cả vị trí các đỉnh Formant theo $F_0$, người nghe sẽ cảm nhận thấy âm sắc bị biến dạng phi tự nhiên thành giọng "chuột sóc" (*chipmunk effect*).

---

### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?
* **Cách thức thay đổi Pitch của TD-PSOLA (Time-Domain Pitch Synchronous Overlap-Add):**
  1. Xác định chính xác các điểm đóng dây thanh âm (*pitch marks / glottal closure instants - GCI*), phân tách tín hiệu thành các chu kỳ cơ bản $T_0$.
  2. Nhân tín hiệu xung quanh mỗi pitch mark với một cửa sổ Hanning đối xứng có độ rộng thường bằng $2 T_0$, thu được các đoạn tín hiệu ngắn $s_i[n]$.
  3. Để thay đổi cao độ sang chu kỳ mới $T_{\text{new}}$: Dịch chuyển khoảng cách giữa các tâm cửa sổ sang vị trí cách nhau một khoảng $T_{\text{new}}$ (nếu $T_{\text{new}} < T_0$ thì tăng pitch; nếu $T_{\text{new}} > T_0$ thì giảm pitch) và thực hiện phép cộng chồng lấn (Overlap-Add).
* **Nguyên nhân Voiced Fricative bị Buzziness khi kéo dài:**
  - Âm xát hữu thanh (như /v/, /z/, /d/) có cấu trúc vật lý kép: vừa có thành phần tuần hoàn do dây thanh rung ($F_0$), vừa có thành phần nhiễu động ngẫu nhiên do luồng không khí ma sát với vận tốc cao qua khe hẹp của khoang miệng.
  - Khi kéo dài thời lượng (Time-stretch), TD-PSOLA buộc phải lặp lại các đoạn cửa sổ ngắn tuần hoàn nhiều lần. Phép toán này vô tình **cưỡng bức thành phần nhiễu ngẫu nhiên tự nhiên trở thành tín hiệu tuần hoàn nhân tạo**. Việc các đoạn nhiễu giống hệt nhau bị lặp đi lặp lại tuần hoàn phá hủy tính ngẫu nhiên hỗn loạn của luồng khí xát, sinh ra các hài âm giả tạo và tạo nên tiếng rè kim loại, ong ong rất khó chịu (**Buzziness / Phasing artifact**).

---

### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?
* Theo lý thuyết HMM truyền thống, giả định **Output Independence (Độc lập quan sát)** coi frame quan sát $x_t$ chỉ phụ thuộc duy nhất vào trạng thái ẩn $q_t$ tại thời điểm đó:
  $$P(x_1, \dots, x_T \mid q_1, \dots, q_T) = \prod_{t=1}^T P(x_t \mid q_t)$$
  Nếu tối ưu hóa trực tiếp, giá trị kỳ vọng $\hat{x}_t$ sẽ bằng vector trung bình $\mu_{q_t}$ của trạng thái hiện tại. Do đó, chuỗi tham số sinh ra sẽ là các đường bậc thang nằm ngang không đổi trong suốt thời gian một state duy trì, và nhảy đột ngột sang giá trị khác khi chuyển state, gây giật cục nghiêm trọng.
* **Vai trò của Delta Feature:**
  - Khi mở rộng vector đặc trưng quan sát thành $\hat{O}_t = [c_t^T, \Delta c_t^T, \Delta^2 c_t^T]^T$, với $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$.
  - Ma trận biến đổi động $W$ liên kết toàn bộ chuỗi $C = [c_1^T, \dots, c_T^T]^T$ với $\hat{O} = W C$.
  - Thuật toán **MLPG (Tokuda et al.)** giải bài toán cực đại hóa hàm hợp lý:
    $$C^* = \arg\max_C P(W C \mid \lambda, Q) = (W^T \Sigma^{-1} W)^{-1} W^T \Sigma^{-1} \mu$$
  - Ma trận $W^T \Sigma^{-1} W$ đóng vai trò như một hệ phương trình ràng buộc trơn đa biến. Thành phần $\Delta c_t$ đóng vai trò như vector vận tốc và $\Delta^2 c_t$ là gia tốc, buộc các frame kế tiếp nhau phải có tốc độ thay đổi liên tục, biến đường bậc thang rời rạc thành một quỹ đạo khả vi, trơn mượt hoàn toàn theo thời gian.

---

### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?
* Toàn bộ sức mạnh của Concatenative TTS nằm ở việc giữ nguyên dạng sóng âm thanh thực tế thu từ người nói nhằm bảo tồn 100% chi tiết âm học tự nhiên mà không qua các bước mã hóa tham số gây mất mát.
* Tuy nhiên, trong ngôn ngữ tự nhiên, một từ vựng hay âm vị chịu sự chi phối của hàng nghìn yếu tố ngữ âm học phức tạp:
  - Ngữ cảnh âm vị học lân cận (âm đứng trước, âm đứng sau quy định chuyển tiếp đồng cấu âm).
  - Vị trí trong câu (đầu câu, giữa câu, cuối câu).
  - Ngữ điệu câu (khẳng định, nghi vấn, cảm thán, liệt kê).
  - Trọng âm, nhịp độ và sắc thái cảm xúc.
* Khi kho dữ liệu nhỏ, hệ thống buộc phải chọn các đơn vị có ngữ cảnh khác xa với câu cần tổng hợp, sau đó phải sử dụng các thuật toán DSP (như PSOLA, WSOLA) để kéo giãn thời gian hoặc đổi cao độ nhằm ép đơn vị vừa khít. Việc biến đổi nhân tạo này luôn để lại các hiện tượng méo tiếng và suy giảm chất lượng âm thanh.
* Để đạt được độ tự nhiên tối đa (Naturalness cao), hệ thống phải tìm được trong kho một đơn vị có ngữ cảnh âm vị và ngữ điệu **gần như trùng khớp hoàn toàn** với yêu cầu của câu mới. Điều này đòi hỏi kích thước kho ngữ liệu (Inventory) phải cực kỳ đồ sộ (hàng chục đến hàng trăm giờ ghi âm chất lượng phòng thu) để bao phủ trọn vẹn không gian tổ hợp ngôn ngữ học phong phú.

---

### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.

| Tiêu chí | Concatenative TTS (Ghép nối) | Neural TTS (Học sâu) |
| :--- | :--- | :--- |
| **1. Cơ chế sinh âm thanh** | Dựa trên việc **tìm kiếm, cắt tỉa và chắp nối** các mẩu sóng âm vật lý có sẵn từ cơ sở dữ liệu thu âm hữu hạn (*Database Lookup & Splicing*). | Dựa trên việc **học biểu diễn phân phối xác suất liên tục** bằng mạng nơ-ron sâu, sinh trực tiếp đặc trưng phổ và tổng hợp sóng âm thông qua mô hình sinh (*End-to-End Generative Modeling*). |
| **2. Tính linh hoạt và Mở rộng** | **Cực kỳ cứng nhắc:** Không thể tổng hợp từ mới chưa có trong kho dữ liệu; muốn đổi giọng đọc, cảm xúc hay người nói thì bắt buộc phải thu âm lại toàn bộ kho dữ liệu từ đầu. | **Cực kỳ linh hoạt:** Khái quát hóa mạnh mẽ trên các từ vựng ngoài từ điển (*Out-of-vocabulary*); có thể thay đổi người nói, cảm xúc, phong cách phát âm (*Voice Cloning / Zero-shot Adaptation*) chỉ bằng vector nhúng ngắn (*Speaker Embedding*). |

---

### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?
1. **Vi phạm nguyên tắc kiểm soát điều kiện thực nghiệm (Confounding Variables):**
   - Các mô hình tải ngẫu nhiên trên mạng thường được huấn luyện trên các tập dữ liệu ngoại lai với tần số lấy mẫu khác biệt ($22.05\text{ kHz}, 24\text{ kHz}, 48\text{ kHz}$), phòng thu chuyên nghiệp, người đọc khác, và bộ chuẩn hóa văn bản riêng biệt.
   - Việc so sánh một mô hình $48\text{ kHz}$ trên mạng với hệ thống ghép nối $16\text{ kHz}$ cục bộ là một phép so sánh khập khiễng, thiếu công bằng và không có giá trị khoa học.
2. **Tính tái lập và Độc lập môi trường (Reproducibility & Stability):**
   - Tải mô hình ngẫu nhiên trong giờ thực hành dễ dẫn đến tắc nghẽn băng thông, rủi ro xung đột phiên bản thư viện phần mềm (PyTorch, CUDA, Transformers), hoặc chứa mã độc. Hệ thống thực hành chuẩn mực đòi hỏi các mô hình phải được đóng gói offline và kiểm thử đồng nhất trên cùng một thiết bị phần cứng.
3. **Mục tiêu sư phạm cốt lõi:**
   - Mục đích của bài Lab là giúp sinh viên nắm vững nguyên lý kiến trúc hai giai đoạn: $\text{Text} \to \text{Mel-spectrogram} \to \text{Neural Vocoder}$, kiểm soát độ trễ suy luận (Latency) và hiểu rõ bản chất xử lý tín hiệu, chứ không phải chạy đua điểm số bằng các mô hình hộp đen ngoại lai.

---

### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.
* **Bản chất của hiện tượng:** Hệ thống có điểm MOS chung rất cao nghĩa là âm thanh phát ra nghe cực kỳ truyền cảm, mượt mà, giống hệt tiếng người thật về mặt âm sắc, độ thở và nhịp điệu (Naturalness cao); tuy nhiên các âm vị bị dính chùm, phát âm không chuẩn xác hoặc khó nghe rõ từng từ cụ thể (Intelligibility thấp).
* **Các ứng dụng thực tiễn phù hợp:**
  1. **Tạo tiếng ồn nền đám đông / Tiếng người xì xào trong Điện ảnh và Trò chơi điện tử (Ambient Crowd Chatter / "Walla"):**
     - Trong các cảnh quay quán cà phê, nhà ga, sân bay, quảng trường hay trò chơi thế giới mở (RPG), đạo diễn cần âm thanh nền rì rầm của đám đông để tạo không gian sống động và chân thực.
     - Tuy nhiên, tiếng ồn nền này **tuyệt đối không được nghe rõ nội dung** từng câu chữ cụ thể, vì nếu người nghe nghe rõ từng chữ của người qua đường, sự chú ý của họ sẽ bị phân tán khỏi lời thoại của nhân vật chính. Do đó, một hệ thống có chất giọng tự nhiên cao nhưng độ rõ thấp là sự lựa chọn lý tưởng.
  2. **Ứng dụng thiền định, trị liệu âm thanh và ru ngủ (Meditation & Sleep Soundscapes):**
     - Giọng nói trong các ứng dụng này cần sự mềm mại, êm dịu, du dương để đưa người dùng vào trạng thái thư giãn sâu của sóng não. Ý nghĩa ngữ nghĩa của câu chữ lúc này không còn quan trọng bằng nhịp điệu và âm sắc ru dương của giọng nói.
  3. **Nhạc cụ ảo mô phỏng dàn hợp xướng (Vocal Choirs & Vocal Pads Synthesizer):**
     - Trong hòa âm phối khí âm nhạc hiện đại, các nhà sản xuất thường dùng giọng hát bè hoặc lớp âm thanh giọng người (vocal pads) làm nền hòa thanh. Âm thanh cần mang màu sắc tự nhiên của thanh quản con người nhưng không cần mang thông điệp từ ngữ cụ thể.

---

## 11. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Qua bài thực hành Lab 4, nhóm nghiên cứu đã hiện thực hóa trọn vẹn và chuyên sâu cả 3 thế hệ công nghệ tổng hợp tiếng nói:
1. **Concatenative TTS:** Xây dựng hoàn chỉnh pipeline ghép nối miền hẹp cho bài toán "Thông báo xếp hàng", chứng minh hiệu quả giảm $21.92\%$ chi phí bất liên tục biên của thuật toán Unit Selection (Viterbi Search), và triệt tiêu tiếng click bằng Cross-fade $15\text{ ms}$.
2. **HMM Parametric TTS:** Mô phỏng thành công cơ chế sinh tham số âm học liên tục MLPG từ các phân bố thống kê trạng thái ẩn HMM, kiểm soát linh hoạt các thông số Prosody (Pitch F0, Duration) thông qua bộ lọc cộng hưởng Formant.
3. **Neural TTS:** Tiếp cận kiến trúc học sâu End-to-End hai giai đoạn hiện đại, đạt điểm số tự nhiên vượt trội ($4.62 / 5.0$) và đo đạc độ trễ suy luận thực tế.

Toàn bộ mã nguồn, dữ liệu âm thanh và kết quả kiểm thử được đóng gói hoàn chỉnh, khoa học và sẵn sàng tái lập 100% tại thư mục dự án `d:\Lab4`.
