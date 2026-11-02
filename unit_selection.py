"""
Module: unit_selection.py
Chức năng: Bộ chuẩn hóa văn bản miền hẹp và Thuật toán Unit Selection (Dynamic Programming / Viterbi Search)
- Chuẩn hóa văn bản thông báo xếp hàng (chuyển số, bỏ ký tự đặc biệt).
- Xây dựng ma trận lưới Trellis (Candidate Grid).
- Tính toán Target Cost và Join Cost giữa các ứng viên liền kề.
- Thuật toán Viterbi giải bài toán tối ưu tổ hợp:
      θ* = arg min { Σ_j C_target(θ_j, t_j) + λ Σ_j C_join(θ_{j-1}, θ_j) }
- So sánh chi phí giữa Baseline cố định (chọn candidate 01) và Unit Selection tối ưu.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import re
import numpy as np
import pandas as pd
from speech_dsp_utils import load_unit, join_cost, calc_rms

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

WORD_TO_KEY = {
    "mời": "moi", "khách": "khach", "hàng": "hang", "số": "so",
    "không": "khong", "một": "mot", "hai": "hai", "ba": "ba",
    "bốn": "bon", "năm": "nam", "sáu": "sau", "bảy": "bay",
    "tám": "tam", "chín": "chin", "đến": "den", "quầy": "quay"
}

def normalize_queue_text(text: str) -> list[str]:
    """
    Chuẩn hóa văn bản cho miền hẹp 'thông báo xếp hàng'.
    Ví dụ: 'Mời khách hàng số 23 đến quầy 5.'
    -> ['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']
    """
    # Lowercase và xóa các dấu câu không cần thiết
    text = text.lower()
    text = re.sub(r"[.,!?;:\"\'()\[\]{}]", " ", text)
    tokens = text.split()
    
    out_words = []
    for tok in tokens:
        if tok.isdigit():
            # Đọc từng chữ số theo quy tắc miền hẹp thông báo số xếp hàng
            out_words.extend(DIGIT_MAP[d] for d in tok if d in DIGIT_MAP)
        else:
            out_words.append(tok)
            
    return out_words

class UnitSelectionSynthesizer:
    def __init__(self, metadata_path: str = "lab4_data/metadata.csv", sr: int = 16000):
        self.sr = sr
        self.metadata_path = metadata_path
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Không tìm thấy metadata tại {metadata_path}!")
        self.df_meta = pd.read_csv(metadata_path)
        
        # Cache âm thanh để tăng tốc độ tính toán join cost
        self.audio_cache = {}
        for _, row in self.df_meta.iterrows():
            uid = row["unit_id"]
            p = row["file_path"]
            if os.path.exists(p):
                self.audio_cache[uid] = load_unit(p, sr=self.sr)
            else:
                print(f"[Cảnh báo] File {p} không tồn tại!")

    def get_candidates_for_word(self, word: str) -> list[dict]:
        """Lấy danh sách các ứng viên âm thanh phù hợp với từ vựng."""
        key = WORD_TO_KEY.get(word, word)
        cand_df = self.df_meta[self.df_meta["key"] == key]
        if cand_df.empty:
            raise ValueError(f"Từ '{word}' (key: '{key}') không có trong Unit Inventory!")
        return cand_df.to_dict("records")

    def compute_target_cost(self, cand_meta: dict, pos_idx: int, total_len: int) -> float:
        """
        Tính Target Cost: Đo lường mức độ phù hợp của candidate với vị trí ngữ điệu.
        - Đầu câu (pos_idx == 0): ưu tiên candidate trung tính / tiếp diễn.
        - Giữa câu: ưu tiên candidate trung tính (01) hoặc tiếp diễn (02).
        - Cuối câu (pos_idx == total_len - 1): ưu tiên candidate trầm hạ giọng / cadence (03).
        """
        cand_type = str(cand_meta["candidate"])
        cost = 0.0
        if pos_idx == total_len - 1:
            # Cuối câu hạ giọng
            if cand_type == "03":
                cost = 0.0
            elif cand_type == "01":
                cost = 15.0
            else:
                cost = 35.0
        elif pos_idx == 0:
            # Mở đầu câu
            if cand_type == "01":
                cost = 0.0
            elif cand_type == "02":
                cost = 10.0
            else:
                cost = 25.0
        else:
            # Giữa câu
            if cand_type in ["01", "02"]:
                cost = 0.0
            else:
                cost = 20.0
        return cost

    def select_units_viterbi(self, words: list[str], lambda_join: float = 1.0) -> tuple[list[dict], dict]:
        """
        Thuật toán Quy hoạch động Viterbi tìm chuỗi candidate tối ưu.
        """
        N = len(words)
        trellis_candidates = [self.get_candidates_for_word(w) for w in words]
        
        # Viterbi table: dp[j][k] là chi phí tích lũy nhỏ nhất tại token thứ j, candidate k
        # backpointer[j][k] lưu chỉ số candidate ở bước j-1
        dp = []
        backpointer = []
        
        # Bước 1: Khởi tạo bước đầu tiên (j = 0)
        first_cands = trellis_candidates[0]
        init_costs = []
        for cand in first_cands:
            t_cost = self.compute_target_cost(cand, 0, N)
            init_costs.append(t_cost)
        dp.append(init_costs)
        backpointer.append([-1] * len(first_cands))
        
        # Bước 2: Đệ quy theo thời gian (j = 1 -> N-1)
        for j in range(1, N):
            curr_cands = trellis_candidates[j]
            prev_cands = trellis_candidates[j-1]
            
            curr_dp = []
            curr_bp = []
            
            for k, c_curr in enumerate(curr_cands):
                t_cost = self.compute_target_cost(c_curr, j, N)
                audio_curr = self.audio_cache[c_curr["unit_id"]]
                
                best_cost = float("inf")
                best_prev_idx = -1
                
                for i, c_prev in enumerate(prev_cands):
                    audio_prev = self.audio_cache[c_prev["unit_id"]]
                    j_cost = join_cost(audio_prev, audio_curr, sr=self.sr)
                    
                    total_step_cost = dp[j-1][i] + lambda_join * j_cost + t_cost
                    if total_step_cost < best_cost:
                        best_cost = total_step_cost
                        best_prev_idx = i
                        
                curr_dp.append(best_cost)
                curr_bp.append(best_prev_idx)
                
            dp.append(curr_dp)
            backpointer.append(curr_bp)
            
        # Bước 3: Truy vết tìm đường đi tối ưu
        best_last_idx = int(np.argmin(dp[-1]))
        total_min_cost = dp[-1][best_last_idx]
        
        best_path_indices = [best_last_idx]
        for j in range(N - 1, 0, -1):
            prev_idx = backpointer[j][best_path_indices[-1]]
            best_path_indices.append(prev_idx)
        best_path_indices.reverse()
        
        selected_units = [trellis_candidates[j][idx] for j, idx in enumerate(best_path_indices)]
        
        # Thống kê chi phí chi tiết
        total_target_cost = sum(self.compute_target_cost(selected_units[j], j, N) for j in range(N))
        total_join_cost = sum(
            join_cost(self.audio_cache[selected_units[j-1]["unit_id"]],
                      self.audio_cache[selected_units[j]["unit_id"]], sr=self.sr)
            for j in range(1, N)
        )
        
        cost_stats = {
            "total_cost": total_min_cost,
            "total_target_cost": total_target_cost,
            "total_join_cost": total_join_cost,
            "selected_ids": [u["unit_id"] for u in selected_units]
        }
        return selected_units, cost_stats

    def select_baseline_units(self, words: list[str]) -> tuple[list[dict], dict]:
        """Chọn cố định candidate đầu tiên ('01') cho mỗi token (Baseline Concatenation)."""
        N = len(words)
        selected_units = []
        for w in words:
            cands = self.get_candidates_for_word(w)
            selected_units.append(cands[0])  # Chọn candidate 01
            
        total_target_cost = sum(self.compute_target_cost(selected_units[j], j, N) for j in range(N))
        total_join_cost = sum(
            join_cost(self.audio_cache[selected_units[j-1]["unit_id"]],
                      self.audio_cache[selected_units[j]["unit_id"]], sr=self.sr)
            for j in range(1, N)
        )
        total_cost = total_target_cost + total_join_cost
        
        cost_stats = {
            "total_cost": total_cost,
            "total_target_cost": total_target_cost,
            "total_join_cost": total_join_cost,
            "selected_ids": [u["unit_id"] for u in selected_units]
        }
        return selected_units, cost_stats
