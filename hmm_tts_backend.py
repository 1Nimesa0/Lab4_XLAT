"""
Module: hmm_tts_backend.py
Chức năng: HMM-based Parametric Speech Synthesis Backend (Mô phỏng mô hình HTS / HMM-TTS)
- Kiến trúc tổng hợp tham số theo Huang et al. và Yoshimura et al. (HTS):
  1. Phone/Word Context Analysis: Ánh xạ chuỗi từ thành chuỗi mô hình HMM (Left-to-Right 3-state HMM).
  2. Duration Modeling: Dự đoán thời lượng cư trú của từng trạng thái (frame-level).
  3. Trajectory Generation (MLPG - Maximum Likelihood Parameter Generation):
     Sinh chuỗi tham số âm học (F0 contour, Formants F1/F2/F3, Energy) trơn tru nhờ ràng buộc động học Delta.
  4. Source-Filter Resynthesis:
     Nguồn xung kích thích tuần hoàn (Voiced) / Nhiễu trắng (Unvoiced) đi qua bộ lọc cộng hưởng số (Formant Resonators).
- Hỗ trợ can thiệp Prosody: scale duration (0.8x, 1.2x) và scale pitch F0 (0.9x, 1.1x).
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from scipy.signal import lfilter, butter
import re

# Thư viện tham số âm học mẫu cho các từ trong miền hẹp (F0 baseline, Formant F1, F2, F3 theo Hz)
# Cấu trúc: 3 state cho mỗi từ [Onset, Nucleus, Coda]
WORD_ACOUSTIC_PROFILES = {
    "mời": {
        "f0": [170.0, 155.0, 135.0],      # Dấu huyền: cao độ đi xuống
        "f1": [350.0, 500.0, 400.0],
        "f2": [1200.0, 1500.0, 1900.0],
        "f3": [2500.0, 2600.0, 2700.0],
        "voiced": [True, True, True],
        "duration_frames": [12, 22, 14]
    },
    "khách": {
        "f0": [180.0, 215.0, 240.0],      # Dấu sắc: đi lên cao
        "f1": [450.0, 750.0, 500.0],
        "f2": [1400.0, 1600.0, 1800.0],
        "f3": [2600.0, 2700.0, 2800.0],
        "voiced": [False, True, False],   # /kh/ vô thanh, /ach/ tắc vô thanh cuối
        "duration_frames": [10, 18, 10]
    },
    "hàng": {
        "f0": [165.0, 150.0, 130.0],      # Dấu huyền
        "f1": [500.0, 800.0, 550.0],
        "f2": [1300.0, 1400.0, 1200.0],
        "f3": [2400.0, 2500.0, 2500.0],
        "voiced": [True, True, True],
        "duration_frames": [12, 24, 14]
    },
    "số": {
        "f0": [185.0, 210.0, 235.0],      # Dấu sắc
        "f1": [400.0, 520.0, 480.0],
        "f2": [1500.0, 1050.0, 950.0],
        "f3": [2600.0, 2650.0, 2700.0],
        "voiced": [False, True, True],    # /s/ âm xát vô thanh
        "duration_frames": [14, 20, 12]
    },
    "không": {
        "f0": [170.0, 168.0, 165.0],      # Thanh ngang
        "f1": [400.0, 500.0, 420.0],
        "f2": [1200.0, 950.0, 900.0],
        "f3": [2500.0, 2500.0, 2500.0],
        "voiced": [False, True, True],
        "duration_frames": [10, 20, 12]
    },
    "một": {
        "f0": [150.0, 130.0, 115.0],      # Thanh nặng
        "f1": [420.0, 550.0, 400.0],
        "f2": [1300.0, 1100.0, 1000.0],
        "f3": [2500.0, 2550.0, 2600.0],
        "voiced": [True, True, False],
        "duration_frames": [10, 16, 8]
    },
    "hai": {
        "f0": [172.0, 170.0, 168.0],      # Thanh ngang
        "f1": [500.0, 780.0, 450.0],
        "f2": [1400.0, 1500.0, 2000.0],
        "f3": [2600.0, 2650.0, 2750.0],
        "voiced": [True, True, True],
        "duration_frames": [10, 22, 14]
    },
    "ba": {
        "f0": [170.0, 168.0, 166.0],      # Thanh ngang
        "f1": [450.0, 820.0, 750.0],
        "f2": [1200.0, 1350.0, 1300.0],
        "f3": [2500.0, 2550.0, 2550.0],
        "voiced": [True, True, True],
        "duration_frames": [10, 24, 14]
    },
    "bốn": {
        "f0": [180.0, 210.0, 230.0],      # Dấu sắc
        "f1": [420.0, 520.0, 480.0],
        "f2": [1200.0, 1000.0, 1400.0],
        "f3": [2550.0, 2600.0, 2650.0],
        "voiced": [True, True, True],
        "duration_frames": [10, 20, 12]
    },
    "năm": {
        "f0": [170.0, 168.0, 165.0],      # Thanh ngang
        "f1": [400.0, 750.0, 500.0],
        "f2": [1400.0, 1450.0, 1300.0],
        "f3": [2500.0, 2550.0, 2550.0],
        "voiced": [True, True, True],
        "duration_frames": [10, 22, 14]
    },
    "sáu": {
        "f0": [185.0, 215.0, 240.0],      # Dấu sắc
        "f1": [450.0, 750.0, 450.0],
        "f2": [1600.0, 1400.0, 950.0],
        "f3": [2600.0, 2650.0, 2700.0],
        "voiced": [False, True, True],
        "duration_frames": [12, 22, 12]
    },
    "bảy": {
        "f0": [165.0, 145.0, 160.0],      # Thanh hỏi: hơi chúc xuống rồi hơi vồng lên
        "f1": [450.0, 720.0, 480.0],
        "f2": [1300.0, 1500.0, 1950.0],
        "f3": [2500.0, 2600.0, 2700.0],
        "voiced": [True, True, True],
        "duration_frames": [10, 20, 12]
    },
    "tám": {
        "f0": [180.0, 210.0, 235.0],      # Dấu sắc
        "f1": [420.0, 780.0, 500.0],
        "f2": [1400.0, 1380.0, 1250.0],
        "f3": [2550.0, 2600.0, 2600.0],
        "voiced": [False, True, True],
        "duration_frames": [10, 22, 12]
    },
    "chín": {
        "f0": [185.0, 215.0, 240.0],      # Dấu sắc
        "f1": [380.0, 420.0, 400.0],
        "f2": [1600.0, 2100.0, 1800.0],
        "f3": [2600.0, 2800.0, 2700.0],
        "voiced": [False, True, True],
        "duration_frames": [10, 22, 12]
    },
    "đến": {
        "f0": [180.0, 205.0, 230.0],      # Dấu sắc
        "f1": [420.0, 550.0, 450.0],
        "f2": [1500.0, 1800.0, 1600.0],
        "f3": [2600.0, 2700.0, 2700.0],
        "voiced": [True, True, True],
        "duration_frames": [10, 20, 12]
    },
    "quầy": {
        "f0": [165.0, 150.0, 130.0],      # Dấu huyền
        "f1": [400.0, 580.0, 450.0],
        "f2": [1100.0, 1400.0, 1900.0],
        "f3": [2450.0, 2550.0, 2700.0],
        "voiced": [False, True, True],
        "duration_frames": [12, 22, 14]
    }
}

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_text_hmm(text: str) -> list[str]:
    """Tách từ và chuẩn hóa cho HMM backend."""
    text = text.lower()
    text = re.sub(r"[.,!?;:\"\'()\[\]{}]", " ", text)
    tokens = text.split()
    words = []
    for tok in tokens:
        if tok.isdigit():
            words.extend(DIGIT_MAP[d] for d in tok if d in DIGIT_MAP)
        else:
            words.append(tok)
    return words

def generate_smooth_trajectory(state_means: list[float], state_durations: list[int], window_len: int = 7) -> np.ndarray:
    """
    Mô phỏng thuật toán MLPG (Maximum Likelihood Parameter Generation) với ràng buộc Delta:
    Chuyển chuỗi kỳ vọng bậc thang qua các state thành đường cong tham số trơn mượt liên tục theo thời gian.
    """
    raw_stepwise = []
    for m, dur in zip(state_means, state_durations):
        raw_stepwise.extend([m] * dur)
    raw_stepwise = np.array(raw_stepwise, dtype=np.float32)
    
    if len(raw_stepwise) < window_len:
        return raw_stepwise
    
    # Bộ lọc Hanning làm trơn (xấp xỉ nghiệm MLPG của HTS)
    kernel = np.hanning(window_len)
    kernel /= np.sum(kernel)
    pad_w = window_len // 2
    padded = np.pad(raw_stepwise, (pad_w, pad_w), mode='edge')
    smooth = np.convolve(padded, kernel, mode='valid')
    return smooth

def two_pole_resonator(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    """Bộ lọc cộng hưởng 2 cực (2-pole digital resonator) đại diện cho một Formant."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * f / sr
    a1 = -2.0 * r * np.cos(theta)
    a2 = r * r
    b0 = 1.0 - r  # chuẩn hóa độ lợi đỉnh
    b = [b0]
    a = [1.0, a1, a2]
    return lfilter(b, a, x)

def synthesize_hmm(text: str, sr: int = 16000, duration_scale: float = 1.0, f0_scale: float = 1.0) -> tuple[np.ndarray, int, dict]:
    """
    Hàm tổng hợp tiếng nói tham số HMM (HMM-based Speech Synthesis):
    y_hmm, sr, tracks = synthesize_hmm(text, duration_scale=1.0, f0_scale=1.0)
    
    Tham số:
        text: Câu văn bản đầu vào.
        sr: Tần số mẫu (16000 Hz).
        duration_scale: Hệ số scale thời lượng (0.8x nhanh, 1.2x chậm).
        f0_scale: Hệ số scale tần số cơ bản F0 (0.9x trầm hơn, 1.1x cao hơn).
        
    Trả về:
        y_hmm: Mảng tín hiệu âm thanh waveform tổng hợp.
        sr: Tần số lấy mẫu (16000).
        tracks: Dictionary chứa các đường trajectory tham số theo frame:
                - 'f0': đường F0 contour (Hz)
                - 'f1': đường Formant F1 (Hz)
                - 'f2': đường Formant F2 (Hz)
                - 'f3': đường Formant F3 (Hz)
                - 'energy': đường năng lượng (dB)
                - 'frame_times': trục thời gian của từng frame (giây)
                - 'state_durations': độ dài từng state
                - 'tokens': danh sách từ vựng đã tổng hợp
    """
    words = normalize_text_hmm(text)
    hop_length = int(sr * 0.010)  # Frame shift 10ms = 160 samples
    
    # 1. Thu thập chuỗi phân bố các state từ chuỗi từ vựng
    f0_means = []
    f1_means = []
    f2_means = []
    f3_means = []
    voiced_flags = []
    state_durations = []
    state_labels = []
    
    # Khoảng lặng ngắn ban đầu (Silence head)
    sil_dur = max(2, int(6 * duration_scale))
    state_durations.append(sil_dur)
    f0_means.append(0.0)
    f1_means.append(400.0)
    f2_means.append(1200.0)
    f3_means.append(2500.0)
    voiced_flags.extend([False] * sil_dur)
    state_labels.append("SIL_start")
    
    for w in words:
        prof = WORD_ACOUSTIC_PROFILES.get(w, WORD_ACOUSTIC_PROFILES["một"])
        for s_idx in range(3):
            dur = max(3, int(prof["duration_frames"][s_idx] * duration_scale))
            state_durations.append(dur)
            f0_means.append(prof["f0"][s_idx] * f0_scale)
            f1_means.append(prof["f1"][s_idx])
            f2_means.append(prof["f2"][s_idx])
            f3_means.append(prof["f3"][s_idx])
            voiced_flags.extend([prof["voiced"][s_idx]] * dur)
            state_labels.append(f"{w}_s{s_idx+1}")
            
    # Khoảng lặng kết thúc
    state_durations.append(sil_dur)
    f0_means.append(0.0)
    f1_means.append(400.0)
    f2_means.append(1200.0)
    f3_means.append(2500.0)
    voiced_flags.extend([False] * sil_dur)
    state_labels.append("SIL_end")
    
    # 2. Sinh đường trajectory trơn mượt (MLPG Trajectory Generation)
    traj_f0_base = generate_smooth_trajectory(f0_means, state_durations, window_len=7)
    traj_f1 = generate_smooth_trajectory(f1_means, state_durations, window_len=9)
    traj_f2 = generate_smooth_trajectory(f2_means, state_durations, window_len=9)
    traj_f3 = generate_smooth_trajectory(f3_means, state_durations, window_len=9)
    
    total_frames = len(traj_f0_base)
    voiced_flags = np.array(voiced_flags[:total_frames], dtype=bool)
    
    # Đặt F0 = 0 ở các frame unvoiced
    traj_f0 = traj_f0_base.copy()
    traj_f0[~voiced_flags] = 0.0
    
    # Đường năng lượng mượt
    energy_levels = [0.0 if not v else 1.0 for v in voiced_flags]
    traj_energy = generate_smooth_trajectory(energy_levels, [1]*total_frames, window_len=11)
    traj_energy = np.clip(traj_energy, 0.05, 1.0)
    
    # 3. Source-Filter Resynthesis: Sinh âm thanh từ tham số
    total_samples = total_frames * hop_length
    excitation = np.zeros(total_samples, dtype=np.float32)
    
    phase = 0.0
    np.random.seed(42)  # Cố định seed cho tái lập kết quả
    noise = np.random.normal(0, 1.0, total_samples).astype(np.float32)
    
    for i in range(total_frames):
        start_idx = i * hop_length
        end_idx = min(start_idx + hop_length, total_samples)
        cur_f0 = traj_f0[i]
        cur_gain = traj_energy[i]
        
        if cur_f0 > 40.0:
            # Hữu thanh: Pulse train với pitch F0
            period_samples = sr / cur_f0
            for idx in range(start_idx, end_idx):
                phase += 1.0
                if phase >= period_samples:
                    excitation[idx] = 1.0 * cur_gain
                    phase -= period_samples
                else:
                    excitation[idx] = 0.0
        else:
            # Vô thanh: Nhiễu trắng có mức năng lượng thấp hơn
            excitation[start_idx:end_idx] = noise[start_idx:end_idx] * 0.15 * cur_gain
            phase = 0.0
            
    # Lọc Formants qua vocal tract filter (mô hình 3 Formant cộng hưởng)
    # Trung bình cộng hưởng formant theo từng phân đoạn ngắn để tái tạo đáp ứng tần số
    y_synth = np.zeros(total_samples, dtype=np.float32)
    chunk_frames = 5
    chunk_samples = chunk_frames * hop_length
    
    for cf in range(0, total_frames, chunk_frames):
        s_idx = cf * hop_length
        e_idx = min((cf + chunk_frames) * hop_length, total_samples)
        if s_idx >= e_idx:
            continue
        
        mean_f1 = float(np.mean(traj_f1[cf:cf+chunk_frames]))
        mean_f2 = float(np.mean(traj_f2[cf:cf+chunk_frames]))
        mean_f3 = float(np.mean(traj_f3[cf:cf+chunk_frames]))
        
        exc_chunk = excitation[s_idx:e_idx]
        # Lọc qua 3 cực F1, F2, F3
        r1 = two_pole_resonator(exc_chunk, mean_f1, bw=80.0, sr=sr)
        r2 = two_pole_resonator(exc_chunk, mean_f2, bw=120.0, sr=sr)
        r3 = two_pole_resonator(exc_chunk, mean_f3, bw=160.0, sr=sr)
        
        chunk_out = 0.6 * r1 + 0.3 * r2 + 0.1 * r3
        y_synth[s_idx:e_idx] = chunk_out
        
    # Lọc thông dải nhẹ và chuẩn hóa biên độ
    b_lp, a_lp = butter(4, 0.9, btype='low')
    y_synth = lfilter(b_lp, a_lp, y_synth)
    peak = np.max(np.abs(y_synth))
    if peak > 1e-4:
        y_synth = (y_synth / peak) * 0.85
        
    frame_times = np.arange(total_frames) * (hop_length / sr)
    
    tracks = {
        "f0": traj_f0,
        "f1": traj_f1,
        "f2": traj_f2,
        "f3": traj_f3,
        "energy": traj_energy,
        "frame_times": frame_times,
        "state_durations": state_durations,
        "state_labels": state_labels,
        "tokens": words
    }
    
    return y_synth.astype(np.float32), sr, tracks

if __name__ == "__main__":
    test_text = "Mời khách hàng số 23 đến quầy 5"
    y, sr, tr = synthesize_hmm(test_text)
    print(f"Tổng hợp thành công HMM: len={len(y)} samples ({len(y)/sr:.2f}s)")
    print(f"Trajectory frames: {len(tr['f0'])}, F0 mean (voiced): {np.mean(tr['f0'][tr['f0']>0]):.1f} Hz")
