"""
Tự động sinh bộ dữ liệu âm thanh đơn vị (Unit Inventory) cho Lab 4:
- Miền hẹp: Thông báo xếp hàng ("Mời khách hàng số ... đến quầy ...")
- 16 từ vựng: mời, khách, hàng, số, không, một, hai, ba, bốn, năm, sáu, bảy, tám, chín, đến, quầy
- Mỗi từ gồm 3 candidate (_01: chuẩn/giọng giữa, _02: ngữ điệu tiếp diễn/cao hơn nhẹ, _03: ngữ điệu kết thúc/trầm hơn nhẹ)
- Định dạng chuẩn: WAV mono, Fs = 16000 Hz, 16-bit PCM.
- Cắt silence biên gọn gàng, tính các thông số RMS, duration và lưu metadata.csv.
"""

import os
import io
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS
import pandas as pd

VOCABULARY = {
    "moi": "mời",
    "khach": "khách",
    "hang": "hàng",
    "so": "số",
    "khong": "không",
    "mot": "một",
    "hai": "hai",
    "ba": "ba",
    "bon": "bốn",
    "nam": "năm",
    "sau": "sáu",
    "bay": "bảy",
    "tam": "tám",
    "chin": "chín",
    "den": "đến",
    "quay": "quầy"
}

# Cấu hình biến thiên nhẹ để tạo candidate đa dạng tự nhiên cho bài toán Unit Selection
CANDIDATE_CONFIGS = [
    {"suffix": "01", "pitch_shift": 0.0, "rate": 1.0, "gain": 1.0, "desc": "Neutral / Mid tone"},
    {"suffix": "02", "pitch_shift": 0.8, "rate": 1.04, "gain": 1.05, "desc": "Continuation / Slightly higher pitch"},
    {"suffix": "03", "pitch_shift": -0.8, "rate": 0.96, "gain": 0.95, "desc": "Cadence / Slightly lower pitch"}
]

def trim_silence_with_margin(y, sr=16000, top_db=25, margin_ms=25):
    """Cắt khoảng lặng đầu cuối nhưng giữ lại biên an toàn tránh mất phụ âm đầu/cuối."""
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    if len(non_silent_intervals) == 0:
        return y
    
    start_sample = non_silent_intervals[0][0]
    end_sample = non_silent_intervals[-1][1]
    
    margin_samples = int(sr * margin_ms / 1000)
    start_sample = max(0, start_sample - margin_samples)
    end_sample = min(len(y), end_sample + margin_samples)
    
    return y[start_sample:end_sample]

def calc_rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))

def build_inventory(output_dir="lab4_data/units", meta_path="lab4_data/metadata.csv", sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    
    print("=" * 65)
    print("   BẮT ĐẦU TỰ ĐỘNG KHỞI TẠO UNIT INVENTORY (CSE457 - LAB 4)     ")
    print("=" * 65)
    
    for word_key, word_text in VOCABULARY.items():
        print(f"\n[Xử lý từ: '{word_text}' ({word_key})]")
        
        # 1. Sinh âm thanh gốc bằng gTTS
        try:
            tts = gTTS(text=word_text, lang="vi")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            y_raw, _ = librosa.load(fp, sr=sr, mono=True)
        except Exception as e:
            print(f"  [!] Lỗi khi gọi gTTS cho từ '{word_text}': {e}")
            continue
        
        # Cắt silence gọn gàng
        y_clean = trim_silence_with_margin(y_raw, sr=sr, top_db=25, margin_ms=25)
        
        # 2. Tạo 3 candidate
        for cfg in CANDIDATE_CONFIGS:
            c_suffix = cfg["suffix"]
            c_pitch = cfg["pitch_shift"]
            c_rate = cfg["rate"]
            c_gain = cfg["gain"]
            
            y_cand = y_clean.copy()
            if abs(c_rate - 1.0) > 1e-3:
                y_cand = librosa.effects.time_stretch(y_cand, rate=c_rate)
            if abs(c_pitch) > 1e-3:
                y_cand = librosa.effects.pitch_shift(y_cand, sr=sr, n_steps=c_pitch)
                
            y_cand = y_cand * c_gain
            # Chống clipping
            peak = np.max(np.abs(y_cand))
            if peak > 0.98:
                y_cand = y_cand / peak * 0.98
                
            out_filename = f"{word_key}_{c_suffix}.wav"
            out_filepath = os.path.join(output_dir, out_filename)
            
            # Ghi file 16-bit PCM WAV
            sf.write(out_filepath, y_cand.astype(np.float32), sr, subtype="PCM_16")
            
            dur = len(y_cand) / sr
            unit_rms = calc_rms(y_cand)
            
            records.append({
                "unit_id": f"{word_key}_{c_suffix}",
                "label": word_text,
                "key": word_key,
                "candidate": c_suffix,
                "speaker": "Speaker_A",
                "duration_sec": round(dur, 4),
                "num_samples": len(y_cand),
                "rms": round(unit_rms, 5),
                "file_path": out_filepath.replace("\\", "/"),
                "description": cfg["desc"]
            })
            print(f"  ✓ Đã tạo: {out_filename:<16} | Dur: {dur:.3f}s | RMS: {unit_rms:.4f} | {cfg['desc']}")
            time.sleep(0.05)
            
    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 65)
    print(f"Tổng cộng đã tạo thành công {len(df)} unit files!")
    print(f"Metadata lưu tại: {os.path.abspath(meta_path)}")
    print("=" * 65)
    return df

if __name__ == "__main__":
    build_inventory()
