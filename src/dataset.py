# -*- coding: utf-8 -*-
"""
特徴量抽出・前処理モジュール。
局開始時の点数・局情報から16次元基本特徴量を生成し、17次元の特徴量ベクトルを構築します。
"""

import os
import glob
import json
import numpy as np
from typing import Tuple, List, Dict, Optional

def get_additional_features(round_val: float, p: int) -> Tuple[float, float, float]:
    """
    プレイヤー p (0:東, 1:南, 2:西, 3:北) に対する追加特徴量を計算する。
    1. 親番フラグ (is_oya: 1.0 or 0.0)
    2. 残り親回数率 (remaining_oya_rate)
    3. 局進行度 (round_progress)
    """
    r_int = int(np.ceil(round_val))
    oya_p = (r_int - 1) % 4
    is_oya = 1.0 if p == oya_p else 0.0

    r_max = 12 if r_int > 8 else 8
    oya_count = 0
    for k in range(r_int, r_max + 1):
        if (k - 1) % 4 == p:
            oya_count += 1
    remaining_oya_rate = min(1.0, float(oya_count) / 2.0)
    round_progress = round_val / 12.0

    return is_oya, remaining_oya_rate, round_progress

def build_16d_features(
    round_val: float,
    honba: float,
    kyotaku: float,
    scores: List[float],
    p: int
) -> List[float]:
    """
    1局面およびプレイヤー座席 p に対する16次元の正規化済み特徴量を計算する。
    """
    sorted_players = sorted([(scores[i], -i) for i in range(4)], reverse=True)
    gap1 = sorted_players[0][0] - scores[p]
    gap2 = sorted_players[1][0] - scores[p]
    gap3 = sorted_players[2][0] - scores[p]
    gap4 = sorted_players[3][0] - scores[p]

    is_oya, remaining_oya_rate, round_progress = get_additional_features(round_val, p)
    min_score_rate = round(min(scores) / 100000.0, 5)

    feat = [
        round_val,
        round(honba / 10.0, 5),
        round(kyotaku / 5.0, 5),
        round(scores[0] / 100000.0, 5),
        round(scores[1] / 100000.0, 5),
        round(scores[2] / 100000.0, 5),
        round(scores[3] / 100000.0, 5),
        round(gap1 / 100000.0, 5),
        round(gap2 / 100000.0, 5),
        round(gap3 / 100000.0, 5),
        round(gap4 / 100000.0, 5),
        float(p),
        is_oya,
        remaining_oya_rate,
        round_progress,
        min_score_rate
    ]
    return feat

def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    data_dir 内の rank_dist_dataset_*.npz ファイル群をロードし、
    (16次元特徴量 X, 最終順位 y (0-3), chunk_names, source_files) を返す。
    chunk_names には各サンプルが属するチャンクファイル名（例: rank_dist_dataset_00000001）を記録。
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {data_dir}")

    all_features = []
    all_ranks = []
    all_chunk_names = []
    all_source_files = []

    for f_path in npz_files:
        chunk_name = os.path.splitext(os.path.basename(f_path))[0]
        d = np.load(f_path, allow_pickle=True)
        ex_X = d["X"]
        ex_y = d["y"]
        ex_indices = d["record_file_indices"]
        source_files = [str(s) for s in d["source_files"]]

        for idx in range(len(ex_X)):
            round_val = ex_X[idx, 0]
            honba = ex_X[idx, 1]
            kyotaku = ex_X[idx, 2]
            s0 = float(round(ex_X[idx, 3] / 100.0) * 100.0)
            s1 = float(round(ex_X[idx, 4] / 100.0) * 100.0)
            s2 = float(round(ex_X[idx, 5] / 100.0) * 100.0)
            s3 = float(round(ex_X[idx, 6] / 100.0) * 100.0)

            final_ranks = ex_y[idx] - 1  # 1-4位 -> 0-3インデックス
            scores = [s0, s1, s2, s3]

            s_file = source_files[int(ex_indices[idx])] if int(ex_indices[idx]) < len(source_files) else source_files[0]

            for p in range(4):
                feat = build_16d_features(round_val, honba, kyotaku, scores, p)
                all_features.append(feat)
                all_ranks.append(int(final_ranks[p]))
                all_chunk_names.append(chunk_name)
                all_source_files.append(s_file)

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_ranks, dtype=np.int64),
        all_chunk_names,
        all_source_files
    )
