"""
Module: speech_dsp_utils.py
Chức năng: Bộ công cụ xử lý tín hiệu âm thanh và ngữ âm học (DSP for Speech) cho Lab 4:
- Đọc/ghi và chuẩn hóa mức năng lượng (RMS Matching).
- Ghép nối biên với Cross-fade / Overlap-Add (OLA).
- Đo lường chi phí nối (Join Cost): Sai khác phổ Log-Magnitude, bước nhảy RMS và chênh lệch F0.
- Ước lượng Pitch F0 (Autocorrelation / ACF) và kiểm chứng chu kỳ mẫu Np: F0 ≈ Fs / Np.
- Biến đổi Prosody: Time-stretch (WSOLA/Phase Vocoder) và Pitch-shift qua semitones: r = 2^(Δs/12).
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import correlate

def load_unit(path: str, sr: int = 16000) -> np.ndarray:
    """Đọc file WAV mono float32 tại tần số mẫu sr."""
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != sr:
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return x

def calc_rms(x: np.ndarray) -> float:
    """Tính Root-Mean-Square năng lượng của tín hiệu."""
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    """Cân bằng năng lượng RMS của đơn vị về mức mục tiêu (tránh clipping)."""
    current_rms = calc_rms(x)
    if current_rms < 1e-6:
        return x
    scale = target / current_rms
    y = x * scale
    return np.clip(y, -0.99, 0.99)

def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Thực hiện ghép nối hai đoạn tín hiệu a và b với vùng chồng lấn n mẫu (Cross-fade).
    y[m] = (1 - alpha[m]) * a_tail[m] + alpha[m] * b_head[m], 0 <= m < n.
    """
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    
    alpha = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])

def concat_units(paths: list[str], sr: int = 16000, xfade_ms: float = 15.0, target_rms: float = 0.08) -> tuple[np.ndarray, list[int]]:
    """
    Ghép chuỗi unit từ danh sách file paths với thời gian crossfade xfade_ms.
    Trả về:
        waveform: tín hiệu đã ghép
        boundary_indices: danh sách chỉ số mẫu tại các biên nối (dùng để vẽ và phân tích)
    """
    if not paths:
        return np.array([], dtype=np.float32), []
    
    xs = [match_rms(load_unit(p, sr), target=target_rms) for p in paths]
    y = xs[0]
    n = int(sr * xfade_ms / 1000.0)
    boundaries = []
    
    current_pos = len(xs[0])
    for x in xs[1:]:
        boundaries.append(current_pos - n // 2)
        y = crossfade(y, x, n)
        current_pos = len(y)
        
    return y, boundaries

def compute_f0_segment(x_seg: np.ndarray, sr: int = 16000, fmin: float = 60.0, fmax: float = 400.0) -> tuple[float, int]:
    """
    Ước lượng F0 trên đoạn tín hiệu ngắn (20-40ms) sử dụng hàm tự tương quan (Autocorrelation).
    Trả về (F0_hz, Np_samples).
    """
    if len(x_seg) < int(sr / fmin):
        return 0.0, 0
    
    # Loại bỏ thành phần DC
    x = x_seg - np.mean(x_seg)
    # Tự tương quan
    r = correlate(x, x, mode="full")
    r = r[len(r)//2:]  # Lấy nửa lag không âm
    
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag >= len(r):
        max_lag = len(r) - 1
    if min_lag >= max_lag:
        return 0.0, 0
        
    peak_lag = min_lag + np.argmax(r[min_lag:max_lag])
    # Kiểm tra mức độ tuần hoàn (harmonicity)
    if r[0] > 1e-8 and (r[peak_lag] / r[0]) > 0.3:
        f0 = sr / peak_lag
        return float(f0), int(peak_lag)
    return 0.0, int(peak_lag)

def join_cost(a: np.ndarray, b: np.ndarray, sr: int = 16000, ms: float = 25.0, nfft: int = 512,
              w_spec: float = 1.0, w_rms: float = 0.5, w_f0: float = 0.05) -> float:
    """
    Tính chi phí nối (Join Cost / Concatenation Cost) giữa điểm kết thúc của unit a
    và điểm bắt đầu của unit b theo đúng công thức chuẩn:
    C_join = w_spec * D_spec + w_rms * |RMS_a - RMS_b| + w_f0 * |F0_a - F0_b|
    """
    n = int(sr * ms / 1000.0)
    n = min(n, len(a), len(b))
    if n < 32:
        return 0.0
    
    tail_a = a[-n:]
    head_b = b[:n]
    
    w = np.hanning(n)
    A = np.fft.rfft(tail_a * w, n=nfft)
    B = np.fft.rfft(head_b * w, n=nfft)
    
    La = 20.0 * np.log10(np.maximum(np.abs(A), 1e-8))
    Lb = 20.0 * np.log10(np.maximum(np.abs(B), 1e-8))
    d_spec = float(np.mean((La - Lb) ** 2))
    
    # Chênh lệch RMS
    rms_a = calc_rms(tail_a)
    rms_b = calc_rms(head_b)
    d_rms = abs(rms_a - rms_b) * 100.0  # scale cho tương đương thang đo
    
    # Chênh lệch F0
    f0_a, _ = compute_f0_segment(tail_a, sr=sr)
    f0_b, _ = compute_f0_segment(head_b, sr=sr)
    d_f0 = abs(f0_a - f0_b) if (f0_a > 0 and f0_b > 0) else 0.0
    
    total_join = w_spec * d_spec + w_rms * d_rms + w_f0 * d_f0
    return float(total_join)

def modify_duration(x: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Thay đổi tốc độ/thời lượng (Time-stretch) giữ nguyên cao độ cơ bản."""
    if abs(rate - 1.0) < 1e-3:
        return x
    return librosa.effects.time_stretch(x, rate=rate)

def modify_pitch(x: np.ndarray, sr: int = 16000, n_semitones: float = 0.0) -> np.ndarray:
    """
    Dịch cao độ (Pitch Shift) theo số bán cung (semitones):
    r = 2^(n_semitones / 12), F0_new = r * F0_old.
    """
    if abs(n_semitones) < 1e-3:
        return x
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=n_semitones)
